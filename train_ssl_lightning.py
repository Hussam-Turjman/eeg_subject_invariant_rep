import inspect
import os
import uuid
from functools import partial

import lightning.pytorch as pl
import torch
import torch.distributed as dist
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from sklearn.metrics import classification_report, accuracy_score

from subject_invariant_rep.eeg_datasets.dataset_interface import EEGSubset
from subject_invariant_rep.eeg_datasets.fixed_dataset import FixedDataset
from subject_invariant_rep.min2net.utils import DataLoader
from subject_invariant_rep.models.downstream import DownstreamNet
from subject_invariant_rep.models.moco import MoCo
from subject_invariant_rep.models.ssl_impl import SSLModel, SubjectIdentifier, MoCoLoss
from subject_invariant_rep.models.ssl_loader import create_ssl_loaders
from subject_invariant_rep.models.utils import set_freeze, evaluate_ssl
from subject_invariant_rep.utils.path import Path
from subject_invariant_rep.augmentations import create_augmentations
from training_utils import TrainerInterface, create_args, save_results


class LitSSLDownstream(pl.LightningModule):
    def __init__(self, num_classes, latent_dim, encoder):
        super(LitSSLDownstream, self).__init__()
        self.save_hyperparameters("num_classes", "latent_dim")
        self.downstream = DownstreamNet(embed_dim=latent_dim, num_classes=num_classes)
        self.encoder = encoder
        set_freeze(self.encoder, freeze=True)
        self.loss = torch.nn.CrossEntropyLoss()
        # Important: This property activates manual optimization.
        self.automatic_optimization = False

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        X, y, _ = batch
        if type(X) is list:
            X = X[0]
        y = y.float()
        with torch.no_grad():
            latent = self.encoder(X)
        logits = self.downstream(latent)
        loss = self.loss(logits, y)
        opt.zero_grad()
        self.manual_backward(loss)
        opt.step()
        self.log_dict({"downstream_train_loss": loss}, prog_bar=True, sync_dist=True)

    def validation_step(self, batch, batch_idx):
        X, y, _ = batch
        if type(X) is list:
            X = X[0]
        y = y.float()
        with torch.no_grad():
            latent = self.encoder(X)

        logits = self.downstream(latent)
        loss = self.loss(logits, y)

        self.log_dict({"downstream_val_loss": loss}, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.downstream.parameters(), lr=1e-3)
        return optimizer


class LitSSLEmbedder(pl.LightningModule):
    def __init__(self, num_subjects, num_channels,
                 sampling_rate, window_len, arg_device,
                 batch_size, optimizer_config, loss_weights=None):
        super(LitSSLEmbedder, self).__init__()
        if loss_weights is None:
            # identifier loss, moco loss weights
            loss_weights = [0.5, 1.0]

        self.save_hyperparameters("num_subjects", "num_channels",
                                  "sampling_rate", "window_len",
                                  "arg_device", "batch_size",
                                  "optimizer_config", "loss_weights",
                                  )

        self.loss_weights = loss_weights
        self.optimizer_config = optimizer_config
        self.moco = MoCo(base_encoder=SSLModel,
                         current_device=arg_device,
                         eeg_channels_count=num_channels,
                         K=batch_size * 2048,
                         window_len=window_len,
                         freq=sampling_rate,
                         )
        self.subject_identifier = SubjectIdentifier(num_subjects=num_subjects,
                                                    embedding_size=self.moco.encoder_q.embedding_size)
        self.moco_criterion = MoCoLoss()
        self.identifier_criterion = torch.nn.CrossEntropyLoss()
        # Important: This property activates manual optimization.
        self.automatic_optimization = False

    def training_step(self, batch, batch_idx):
        moco_opt, identifier_opt = self.optimizers()
        X, _, subject_ids = batch
        X_original, x_t1, x_t2 = X

        subject_ids = subject_ids.float()

        logits, labels = self.moco(x_t1, x_t2)
        with torch.no_grad():
            fixed_h = self.moco.encoder_q.embed(x_t1)
        subject_identity = self.subject_identifier(fixed_h)

        h = self.moco.encoder_q.encoder(x_t1)
        with torch.no_grad():
            fixed_identity = self.subject_identifier.identity(h)

        adversarial_loss = self.identifier_criterion(subject_identity, subject_ids)
        loss = self.moco_criterion(logits, labels, fixed_identity, subject_ids)

        moco_opt.zero_grad()
        self.manual_backward(loss)
        moco_opt.step()

        identifier_opt.zero_grad()
        self.manual_backward(adversarial_loss)
        identifier_opt.step()
        identifier_accuracy = accuracy_score(y_true=subject_ids.cpu().numpy().argmax(axis=1),
                                             y_pred=fixed_identity.cpu().numpy().argmax(axis=1))
        self.log_dict(
            {"adversarial_loss": adversarial_loss, "moco_loss": loss,
             "identifier_accuracy": identifier_accuracy},
            prog_bar=True, sync_dist=True)

    def validation_step(self, batch, batch_idx):
        X, _, subjects_ids = batch
        X_original, x_t1, x_t2 = X
        subjects_ids = subjects_ids.float()

        logits, labels = self.moco(x_t1, x_t2)
        h = self.moco.encoder_q.encoder(X_original)
        subject_identity = self.subject_identifier(h)
        adversarial_loss = self.identifier_criterion(subject_identity, subjects_ids)

        loss = self.moco_criterion(logits, labels, subject_identity, subjects_ids)
        identifier_accuracy = accuracy_score(y_true=subjects_ids.cpu().numpy().argmax(axis=1),
                                             y_pred=subject_identity.cpu().numpy().argmax(axis=1))

        self.log_dict(
            {"adversarial_val_loss": adversarial_loss,
             "moco_val_loss": loss,
             "identifier_val_accuracy": identifier_accuracy
             },
            prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        # lr = self.optimizer_config["lr"]
        moco_opt = torch.optim.Adam(self.moco.parameters(),
                                    **self.optimizer_config
                                    )

        identifier_opt = torch.optim.Adam(self.subject_identifier.parameters(),
                                          **self.optimizer_config
                                          )
        return moco_opt, identifier_opt


def folds_loop(fold, subject, loader, model_output_dir,
               input_shape, num_class, latent_dim, margin, loss_weights,
               batch_size, epochs, patience, es_patience, lr, min_lr, factor,
               log_path, n_subjects, drop_last, cpu_count,
               transform, num_channels, window_len, sampling_rate, device, optimizer_config, class_combo, train_type
               ):
    channels = input_shape[2]
    samples = input_shape[1]
    if fold == 1 or cpu_count is not None:
        dist.init_process_group("nccl",
                                rank=0,
                                init_method=f"file://{Path.cwd()}/{os.path.basename(__file__)}_{inspect.stack()[0][3]}{str(uuid.uuid4())}.method",
                                world_size=1)
    model_name = 'S{:03d}_fold{:02d}'.format(subject, fold)

    # load dataset
    train_loader, val_loader, test_loader = create_ssl_loaders(loader=loader,
                                                               fold=fold,
                                                               transform=transform,
                                                               batch_size=batch_size,
                                                               drop_last=drop_last,
                                                               num_class=num_class,
                                                               n_subjects=n_subjects,
                                                               EEGSubsetClass=EEGSubset,
                                                               FixedDatasetClass=FixedDataset,
                                                               )

    embedder = LitSSLEmbedder(
        window_len=window_len,
        loss_weights=loss_weights,
        batch_size=batch_size,
        num_subjects=n_subjects,
        num_channels=num_channels,
        sampling_rate=sampling_rate,
        arg_device=device,
        optimizer_config=optimizer_config

    )
    save_dir = Path("logs", "ssl").path
    run_name = f"{class_combo}_{train_type}"
    for m in ["E", "D"]:
        p = Path(save_dir, run_name, f"{model_name}_{m}")
        if p.exists():
            p.remove()

    logger = TensorBoardLogger(save_dir, name=run_name, version=f"{model_name}_E")
    trainer = pl.Trainer(max_epochs=epochs,
                         accelerator="gpu",
                         logger=logger,
                         callbacks=[EarlyStopping(monitor="moco_val_loss", mode="min", patience=es_patience)])
    trainer.fit(embedder, train_loader, val_loader)

    embedder.eval()
    downstream_model = LitSSLDownstream(num_classes=num_class,
                                        latent_dim=embedder.moco.encoder_q.embedding_size,
                                        encoder=embedder.moco.encoder_q.encoder)
    logger = TensorBoardLogger(save_dir, name=run_name, version=f"{model_name}_D")
    downstream_trainer = pl.Trainer(max_epochs=epochs,
                                    accelerator="gpu",
                                    logger=logger,
                                    callbacks=[
                                        EarlyStopping(monitor="downstream_val_loss", mode="min", patience=es_patience)])
    downstream_trainer.fit(downstream_model, train_loader, val_loader)

    y_true_train, y_pred_train, embeddings_train, ids_true_train, ids_pred_train = evaluate_ssl(loader=train_loader,
                                                                                      encoder=embedder.moco.encoder_q.encoder,
                                                                                      downstream=downstream_model.downstream,
                                                                                      subject_identifier=embedder.subject_identifier)

    y_true_val, y_pred_val, embeddings_val, ids_true_val, ids_pred_val = evaluate_ssl(loader=val_loader,
                                                                                      encoder=embedder.moco.encoder_q.encoder,
                                                                                      downstream=downstream_model.downstream,
                                                                                      subject_identifier=embedder.subject_identifier)
    report = classification_report(y_true=y_true_val,
                                   y_pred=y_pred_val,
                                   zero_division=0,
                                   )
    print("-------------Validation------------------", flush=True)
    print(report, flush=True)
    # print("Subject identification report")
    # print(classification_report(y_true=ids_true_val,
    #                             y_pred=ids_pred_val,
    #                             zero_division=0,
    #                             ))

    y_true, y_pred, embeddings, ids_true, ids_pred = evaluate_ssl(loader=test_loader,
                                                                  encoder=embedder.moco.encoder_q.encoder,
                                                                  downstream=downstream_model.downstream,
                                                                  subject_identifier=embedder.subject_identifier)
    report = classification_report(y_true=y_true,
                                   y_pred=y_pred,
                                   zero_division=0,
                                   )
    print("-------------Test------------------", flush=True)
    print(report, flush=True)
    accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
    # print("Subject identification report")
    # print(classification_report(y_true=ids_true,
    #                             y_pred=ids_pred,
    #                             zero_division=0,
    #                             ))
    torch.cuda.empty_cache()

    return dict(
        y_true=y_true,
        y_pred=y_pred,
        embeddings=embeddings,
        y_true_val=y_true_val,
        y_pred_val=y_pred_val,
        embeddings_val=embeddings_val,
        y_true_train=y_true_train,
        y_pred_train=y_pred_train,
        embeddings_train=embeddings_train,
        accuracy=accuracy,
        fold=fold,
        subject=subject,
    )


def k_fold_cross_validation(model_config, dataset_name, time_domain_config, raw_config,
                            subject, class_combo, input_dir,
                            model_output_dir, train_type, device, transform):
    # create object of DataLoader
    training_config = model_config[train_type]
    n_folds = time_domain_config["k_folds"]

    tasks = time_domain_config["tasks"]
    class_combo_config = None
    for task in tasks:
        if task["name"] == class_combo:
            class_combo_config = task
            break
    if class_combo_config is None:
        raise Exception(
            f"Class combination {class_combo} does not have any associated config. Generate the dataset first")

    input_shape = class_combo_config["input_shape"]
    num_class = class_combo_config["num_class"]
    epochs = training_config["epochs"]
    batch_size = training_config["batch_size"]
    lr = training_config["lr"]
    min_lr = training_config["min_lr"]
    factor = training_config["factor"]
    patience = training_config["patience"]
    es_patience = training_config["es_patience"]
    log_path = model_output_dir.path
    data_format = model_config["data_format"]
    data_type = training_config["data_type"]
    margin = model_config["margin"]
    loss_weights = model_config["loss_weights"]
    num_channels = raw_config["n_chs"]
    optimizer_config = model_config["optimizer"]
    window_len = raw_config["window_len"]
    sampling_rate = raw_config["sampling_rate"]
    n_subjects = raw_config["n_subjects"]
    patience_tolerance = training_config["patience_tolerance"]
    drop_last = training_config["drop_last"]
    latent_dim = num_channels if num_class == 2 else 64
    cpu_count = None
    loader = DataLoader(dataset=dataset_name,
                        train_type=train_type,
                        subject=subject,
                        data_format=data_format,
                        data_type=data_type,
                        dataset_path=input_dir.path,
                        class_combo=class_combo,
                        )
    callback = partial(folds_loop,
                       subject=subject, loader=loader, model_output_dir=model_output_dir,
                       input_shape=input_shape, num_class=num_class, latent_dim=latent_dim, margin=margin,
                       loss_weights=loss_weights,
                       batch_size=batch_size, epochs=epochs,
                       patience=patience, es_patience=es_patience,
                       lr=lr, min_lr=min_lr, factor=factor,
                       log_path=log_path, n_subjects=n_subjects, drop_last=drop_last,
                       cpu_count=cpu_count,
                       transform=transform, num_channels=num_channels,
                       window_len=window_len, sampling_rate=sampling_rate,
                       device=device, optimizer_config=optimizer_config,
                       class_combo=class_combo, train_type=train_type
                       )

    save_results(callback=callback, n_folds=n_folds,
                 model_output_dir=model_output_dir,
                 cpu_count=cpu_count,
                 subject=subject, use_torch_mp=cpu_count is not None)


class Trainer(TrainerInterface):

    def on_training_started(self, subject: int, exp: str):
        k_fold_cross_validation(subject=subject,
                                class_combo=self.class_combo,
                                model_config=self.model_config,
                                time_domain_config=self.time_domain_config,
                                raw_config=self.raw_config,
                                input_dir=self.input_dir,
                                model_output_dir=self.model_output_dir,
                                train_type=self.train_type,
                                dataset_name=self.dataset_name,
                                transform=self.augmentations,
                                device=self.device
                                )

    def on_exp(self, model_output_dir: Path, model_config: dict):
        pick_smp_freq = self.raw_config["sampling_rate"]
        window_len = self.raw_config["window_len"]
        self.model_config = model_config
        self.model_output_dir = model_output_dir
        self.augmentations = create_augmentations(model_config["transformations"],
                                                  sfreq=pick_smp_freq,
                                                  window_len=window_len,
                                                  sensor_positions=self.sensor_positions)


def run():
    # torch_mp.set_start_method('spawn')
    args = create_args()

    trainer = Trainer(subject=args.subject,
                      exp=args.exp,
                      class_combo=args.class_combo,
                      train_type=args.train_type,
                      dataset_name=args.dataset_name,
                      continue_training=args.continue_training,
                      model_base_name="ssl",
                      GPU=args.GPU,
                      )
    trainer.start()


if __name__ == '__main__':
    run()
