from functools import partial

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from tensorflow.keras import backend as K

from subject_invariant_rep.min2net.utils import DataLoader
from subject_invariant_rep.models.eegnet import EEGNetClassifier
from subject_invariant_rep.utils.path import Path
from training_utils import create_args, TrainerInterface, save_results

# while the default tensorflow ordering is 'channels_last' we set it here
# to be explicit in case if the user has changed the default ordering
K.set_image_data_format('channels_last')


def folds_loop(fold, subject, loader, model_output_dir,
               freq, num_classes, channels, samples, batch_size, epochs, patience):
    X_train, y_train, subject_ids_train = loader.load_train_set(fold=fold, load_subject_ids=True)
    X_val, y_val, subject_ids_val = loader.load_val_set(fold=fold, load_subject_ids=True)
    X_test, y_test, subject_ids_test = loader.load_test_set(fold=fold, load_subject_ids=True)

    clf = EEGNetClassifier(
        num_classes=num_classes,
        channels=channels,
        samples=samples,
        model_output_dir=model_output_dir,
        subject=subject,
        batch_size=batch_size,
        epochs=epochs,
        fold=fold, patience=patience
    )

    clf.fit(X_train=X_train, Y_train=y_train, X_validate=X_val, Y_validate=y_val)

    y_pred_val,embedding_val = clf.predict(X_val)
    report = classification_report(y_true=y_val,
                                   y_pred=y_pred_val,
                                   )
    print("-------------Validation------------------", flush=True)
    print(report, flush=True)

    y_pred,embedding = clf.predict(X_test)

    report = classification_report(y_true=y_test,
                                   y_pred=y_pred,
                                   )
    print(report, flush=True)
    accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
    tf.keras.backend.clear_session()

    clf.filepath.remove()
    return dict(
        y_true=np.array(y_test),
        y_pred=np.array(y_pred),
        y_true_val=np.array(y_val),
        y_pred_val=np.array(y_pred_val),
        accuracy=accuracy,
        fold=fold,
        subject=subject,
        embeddings=embedding,
        embeddings_val=embedding_val,
    )


def k_fold_cross_validation(
        subject,
        class_combo,
        dataset_name: str,
        model_config,
        time_domain_config,
        raw_config,
        input_dir,
        model_output_dir,
        train_type
):
    training_config = model_config[train_type]
    n_folds = time_domain_config["k_folds"]
    # assert 1 <= fold <= n_folds
    tasks = time_domain_config["tasks"]
    class_combo_config = None
    for task in tasks:
        if task["name"] == class_combo:
            class_combo_config = task
            break
    if class_combo_config is None:
        raise Exception(
            f"Class combination {class_combo} does not have any associated config. Generate the dataset first")

    data_format = model_config["data_format"]
    data_type = training_config["data_type"]
    print(training_config)
    epochs = training_config["epochs"]
    batch_size = training_config["batch_size"]
    es_patience = training_config["es_patience"]
    num_classes = class_combo_config["num_class"]
    input_shape = class_combo_config["input_shape"]
    channels = input_shape[2]
    samples = input_shape[1]
    freq = raw_config["sampling_rate"]
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
                       freq=freq,
                       num_classes=num_classes,
                       channels=channels,
                       samples=samples, batch_size=batch_size,
                       epochs=epochs, patience=es_patience
                       )

    save_results(callback=callback, n_folds=n_folds, model_output_dir=model_output_dir, subject=subject)


class Trainer(TrainerInterface):

    def on_training_started(self, subject: int, exp: str):
        k_fold_cross_validation(subject=subject,
                                dataset_name=self.dataset_name,
                                class_combo=self.class_combo,
                                model_config=self.model_config,
                                raw_config=self.raw_config,
                                time_domain_config=self.time_domain_config,
                                input_dir=self.input_dir,
                                model_output_dir=self.model_output_dir,
                                train_type=self.train_type
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
                      model_base_name="EEGNet",
                      GPU=args.GPU,
                      )
    trainer.start()


if __name__ == "__main__":
    run()
