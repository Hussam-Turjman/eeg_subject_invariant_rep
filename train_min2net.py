import argparse
import json
import os
from functools import partial

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score

from tensorflow.keras.optimizers import Adam

from subject_invariant_rep.models.loss import mean_squared_error, triplet_loss
from subject_invariant_rep.models.min2net import MIN2NetClassifier
from subject_invariant_rep.min2net.utils import DataLoader
from subject_invariant_rep.utils.path import Path
from training_utils import TrainerInterface, create_args, get_results, save_results


def sort_labels(X, y):
    X = X[y.argsort()]
    y = y[y.argsort()]
    return X, y


def folds_loop(fold, subject, loader, model_output_dir,
               input_shape, num_class, latent_dim, margin, loss_weights,
               batch_size, epochs, patience, es_patience, lr, min_lr, factor,
               log_path
               ):
    channels = input_shape[2]
    samples = input_shape[1]

    model_name = 'S{:03d}_fold{:02d}'.format(subject, fold)
    model = MIN2NetClassifier(
        # input_shape=input_shape,
        class_balancing=True,
        #           f1_average='binary' if num_class == 2 else 'macro',
        #          num_class=num_class,
        #         loss=[mean_squared_error, triplet_loss(margin=margin), 'sparse_categorical_crossentropy'],
        loss_weights=loss_weights,
        epochs=epochs,
        batch_size=batch_size,
        #        optimizer=Adam(beta_1=0.9, beta_2=0.999, epsilon=1e-08),
        #       lr=lr,
        min_lr=min_lr,
        factor=factor,
        patience=patience,
        es_patience=es_patience,
        latent_dim=latent_dim,
        #     log_path=log_path,
        #                model_name=model_name
        subject=subject,
        fold=fold,
        model_output_dir=model_output_dir,
        num_classes=num_class,
        channels=channels,
        samples=samples
    )

    # load dataset
    X_train, y_train = loader.load_train_set(fold=fold)
    X_val, y_val = loader.load_val_set(fold=fold)
    X_test, y_test = loader.load_test_set(fold=fold)

    # X_train, y_train = sort_labels(X_train, y_train)
    # X_val, y_val = sort_labels(X_val, y_val)
    # X_test, y_test = sort_labels(X_test, y_test)

    # X_train /= 1000
    # X_val /= 1000
    # X_test /= 1000
    # print(y_train)
    # print(y_val)
    # print(y_test)
    # print(np.min(X_train), np.max(X_train))
    # exit()
    # train and test using MIN2Net
    model.fit(X_train, y_train, X_val, y_val)
    y_pred, embeddings = model.predict(X_test)
    y_pred_val, embeddings_val = model.predict(X_val)
    accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
    tf.keras.backend.clear_session()
    if model.filepath.exists():
        model.filepath.remove()
    return dict(
        y_true=np.array(y_test),
        y_pred=np.array(y_pred),
        embeddings=np.array(embeddings),
        y_true_val=np.array(y_val),
        y_pred_val=np.array(y_pred_val),
        embeddings_val=np.array(embeddings_val),
        accuracy=accuracy,
        fold=fold,
        subject=subject,
    )


def k_fold_cross_validation(model_config, dataset_name, time_domain_config, raw_config,
                            subject, class_combo, input_dir,
                            model_output_dir, train_type):
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
    n_chs = raw_config["n_chs"]

    latent_dim = n_chs if num_class == 2 else 64

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
                       input_shape=input_shape, num_class=num_class, latent_dim=latent_dim,
                       margin=margin, loss_weights=loss_weights,
                       batch_size=batch_size, epochs=epochs, patience=patience,
                       es_patience=es_patience, lr=lr, min_lr=min_lr, factor=factor,
                       log_path=log_path

                       )

    save_results(callback=callback, n_folds=n_folds,
                 model_output_dir=model_output_dir, subject=subject)


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
                                dataset_name=self.dataset_name
                                )

    def on_exp(self, model_output_dir: Path, model_config: dict):
        self.model_config = model_config
        self.model_output_dir = model_output_dir


def run():
    args = create_args()

    trainer = Trainer(subject=args.subject,
                      exp=args.exp,
                      class_combo=args.class_combo,
                      train_type=args.train_type,
                      dataset_name=args.dataset_name,
                      continue_training=args.continue_training,
                      model_base_name="MIN2Net",
                      GPU=args.GPU,
                      )
    trainer.start()


if __name__ == '__main__':
    run()
