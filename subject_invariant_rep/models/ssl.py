import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input, Flatten, ELU, MaxPooling1D
from tensorflow.keras.models import Model

from .loss import triplet_loss


def create_res_block(inputs, in_channels, out_channels):
    block = BatchNormalization(axis=2)(inputs)
    block = ELU()(block)
    residual = Conv1D(in_channels, out_channels, padding='same')
    block = Conv1D(in_channels, out_channels, padding='same')(block)
    block = BatchNormalization()(block)
    block = ELU()(block)
    block = Conv1D(out_channels, out_channels, padding='same')(block)
    if in_channels != out_channels:
        block = residual(block)
    return block


def create_encoder_g(channels, samples):
    in_channels = channels
    out_channels = channels // 2
    pool_size = 2 if channels == 8 else 4
    encoder_input = Input(shape=(channels, samples))
    en_conv = Conv1D(in_channels, out_channels, activation=None)(encoder_input)
    en_conv = create_res_block(en_conv, out_channels, out_channels)
    en_conv = MaxPooling1D(pool_size=pool_size)(en_conv)
    en_conv = create_res_block(en_conv, out_channels, in_channels)
    en_conv = MaxPooling1D(pool_size=pool_size)(en_conv)
    en_conv = create_res_block(en_conv, in_channels, in_channels * 2)
    en_conv = ELU()(en_conv)
    en_conv = Flatten()(en_conv)
    model = Model(inputs=encoder_input, outputs=en_conv, name='encoderG')
    model.summary()
    return model, encoder_input


def create_model_f(latent_dim):
    out_channels = latent_dim // 2

    input1 = Input(shape=(latent_dim,), name='model_f_input')
    model_f = Dense(out_channels, name='model_f_dense', activation='relu')(input1)
    model_f = Dense(out_channels, name='model_f_dense2', activation='relu')(model_f)
    model_f = Dense(out_channels, name='model_f_dense3', activation='relu')(model_f)
    model_f = Dense(out_channels // 2, name='model_f_dense4', activation='relu')(model_f)
    model = Model(inputs=input1, outputs=model_f, name='modelF')
    model.summary()
    return model


def create_subject_identifier(num_subjects, latent_dim):
    input1 = Input(shape=(latent_dim,), name='subject_identifier_input')
    dens_size = latent_dim // 2
    subject_identifier = Dense(dens_size, name='subject_identifier_dense', activation='relu')(input1)
    subject_identifier = Dense(dens_size, name='subject_identifier_dense2', activation='relu')(subject_identifier)
    subject_identifier = Dense(dens_size, name='subject_identifier_dense3', activation='relu')(subject_identifier)
    subject_identifier = Dense(num_subjects, name='subject_identifier_dense4', activation='softmax')(subject_identifier)
    model = Model(inputs=input1, outputs=subject_identifier, name='subject_identifier')
    model.summary()
    return model


def create_ssl_model(channels, samples, num_class):
    encoder_g, encoder_input = create_encoder_g(channels=channels, samples=samples)
    latent_dim = encoder_g.output_shape[-1]
    #model_f = create_model_f(latent_dim=latent_dim)
    latent = encoder_g(encoder_input)
    #train_xr = model_f(latent)
    z = Dense(num_class, activation='softmax', #kernel_constraint=max_norm(0.5),
              name='classifier')(latent)
    model = Model(inputs=encoder_input, outputs=[z],
                  name='SSL')
    model.summary()
    return model, latent_dim


class SSLClassifier(object):
    def __init__(self, num_classes, channels, samples, model_output_dir,
                 subject, batch_size, epochs, fold, latent_dim, es_patience=20, patience=5,
                 class_balancing=False, margin=1.0, verbose=2, factor=0.5, mode="min",
                 loss_weights=None, seed=42,
                 min_lr=0.0001, shuffle=True, monitor="val_loss"):
        if loss_weights is None:
            loss_weights = [
                0.5,
                1.0
            ]
        self.filepath = model_output_dir.copy().join('S{:03d}_fold{:03d}_checkpoint.h5'.format(subject, fold))

        self.shuffle = shuffle
        self.batch_size = batch_size
        self.epochs = epochs
        self.channels = channels
        self.samples = samples
        self.kernels = 1
        self.loss_weights = loss_weights
        if self.filepath.exists():
            self.filepath.remove()

        self.loss = ['sparse_categorical_crossentropy']

        np.random.seed(seed)
        tf.random.set_seed(seed)
        self.model, latent_dim = create_ssl_model(channels=channels, samples=samples, num_class=num_classes)
        # compile the model and set the optimizers
        self.model.compile(loss=self.loss, optimizer='adam',
                           metrics=['accuracy'], loss_weights=self.loss_weights, run_eagerly=False)
        # count number of parameters in the model
        self.model_params_count = self.model.count_params()

        # set a valid path for your system to record model checkpoints
        self.checkpointer = ModelCheckpoint(filepath=self.filepath.path, verbose=1,
                                            save_best_only=True)
        # self.class_weights = {i: 1 for i in range(num_classes)}
        self.es = EarlyStopping(monitor=monitor, mode=mode,
                                verbose=verbose, patience=es_patience)

        self.reduce_lr = ReduceLROnPlateau(monitor=monitor, patience=patience,
                                           factor=factor, mode=mode, verbose=verbose,
                                           min_lr=min_lr)

    def fit(self, X_train, y_train, X_val, y_val):
        self.model.fit(x=X_train, y=y_train,
                       batch_size=self.batch_size, shuffle=self.shuffle,
                       epochs=self.epochs,
                       validation_data=(X_val,  y_val
                                        ),
                       callbacks=[self.checkpointer, self.reduce_lr, self.es],
                       # class_weight=self.class_weights
                       )
        if self.filepath.exists():
            self.model.load_weights(self.filepath.path)
        else:
            print("No checkpoint found: {}".format(self.filepath.path))
        return self

    def predict(self, X_test):
        y_pred_clf = self.model.predict(X_test)
        y_pred = np.argmax(y_pred_clf, axis=1)
        encoder = self.model.get_layer("encoderG")
        embeddings = encoder(X_test)

        return y_pred, embeddings


__all__ = ["SSLClassifier"]
