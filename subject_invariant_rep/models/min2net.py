import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, CSVLogger
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D, AveragePooling2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import Dense, Reshape
from tensorflow.keras.layers import Input, Flatten
from tensorflow.keras.models import Model

from .loss import mean_squared_error, triplet_loss


def create_encoder(input_shape, latent_dim, filter_1, filter_2, pool_size_1, pool_size_2):
    encoder_input = Input(input_shape)
    en_conv = Conv2D(filter_1, (1, 64), activation='elu', padding="same",
                     kernel_constraint=max_norm(2., axis=(0, 1, 2)))(encoder_input)
    en_conv = BatchNormalization(axis=3, epsilon=1e-05, momentum=0.1)(en_conv)
    en_conv = AveragePooling2D(pool_size=pool_size_1)(en_conv)
    en_conv = Conv2D(filter_2, (1, 32), activation='elu', padding="same",
                     kernel_constraint=max_norm(2., axis=(0, 1, 2)))(en_conv)
    en_conv = BatchNormalization(axis=3, epsilon=1e-05, momentum=0.1)(en_conv)
    en_conv = AveragePooling2D(pool_size=pool_size_2)(en_conv)
    en_conv = Flatten()(en_conv)
    encoder_output = Dense(latent_dim, kernel_constraint=max_norm(0.5))(en_conv)
    encoder = Model(inputs=encoder_input, outputs=encoder_output, name='encoder')
    encoder.summary()
    return encoder, encoder_input


def create_decoder(latent_dim, filter_1, filter_2, pool_size_1, pool_size_2, flatten_size):
    decoder_input = Input(shape=(latent_dim,), name='decoder_input')
    de_conv = Dense(1 * flatten_size * filter_2, activation='elu',
                    kernel_constraint=max_norm(0.5))(decoder_input)
    de_conv = Reshape((1, flatten_size, filter_2))(de_conv)
    de_conv = Conv2DTranspose(filters=filter_2, kernel_size=(1, 64),
                              activation='elu', padding='same', strides=pool_size_2,
                              kernel_constraint=max_norm(2., axis=(0, 1, 2)))(de_conv)
    decoder_output = Conv2DTranspose(filters=filter_1, kernel_size=(1, 32),
                                     activation='elu', padding='same', strides=pool_size_1,
                                     kernel_constraint=max_norm(2., axis=(0, 1, 2)))(de_conv)
    decoder = Model(inputs=decoder_input, outputs=decoder_output, name='decoder')
    decoder.summary()
    return decoder


def MIN2Net(channels, samples, num_class,
            kernels=1, filter_2=10,
            subsampling_size=100, pool_size_2=(1, 4),
            latent_dim=256):
    'encoder'
    filter_1 = channels
    pool_size_1 = (1, samples // subsampling_size)
    flatten_size = samples // pool_size_1[1] // pool_size_2[1]
    input_shape = (kernels, samples, channels)
    # kernels,samples,channels
    # channels,samples,kernels

    'Build the computation graph for training'
    encoder, encoder_input = create_encoder(input_shape=input_shape, latent_dim=latent_dim,
                                            filter_1=filter_1, filter_2=filter_2,
                                            pool_size_1=pool_size_1, pool_size_2=pool_size_2
                                            )
    latent = encoder(encoder_input)
    decoder = create_decoder(latent_dim=latent_dim, filter_1=filter_1, filter_2=filter_2,
                             pool_size_1=pool_size_1, pool_size_2=pool_size_2,
                             flatten_size=flatten_size)
    train_xr = decoder(latent)
    z = Dense(num_class, activation='softmax', kernel_constraint=max_norm(0.5),
              name='classifier')(latent)

    return Model(inputs=encoder_input, outputs=[train_xr, latent, z],
                 name='MIN2Net')


class MIN2NetClassifier(object):
    def __init__(self, num_classes, channels, samples, model_output_dir,
                 subject, batch_size, epochs, fold, latent_dim, es_patience=20, patience=5,
                 class_balancing=False, margin=1.0, verbose=2, factor=0.5, mode="min",
                 loss_weights=None, seed=42,
                 min_lr=0.0001, shuffle=True, monitor="val_loss"):
        if loss_weights is None:
            loss_weights = [
                0.5,
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

        self.loss = [mean_squared_error, triplet_loss(margin=margin), 'sparse_categorical_crossentropy']

        np.random.seed(seed)
        tf.random.set_seed(seed)
        self.model = MIN2Net(channels=channels, samples=samples, num_class=num_classes,
                             kernels=1, filter_2=10,
                             subsampling_size=100, pool_size_2=(1, 4),
                             latent_dim=latent_dim)
        # compile the model and set the optimizers
        self.model.compile(loss=self.loss, optimizer='adam',
                           metrics=['accuracy'], loss_weights=self.loss_weights, run_eagerly=False)
        # count number of parameters in the model
        self.model_params_count = self.model.count_params()

        # set a valid path for your system to record model checkpoints
        self.checkpointer = ModelCheckpoint(filepath=self.filepath.path, verbose=1,
                                            save_best_only=True)
        self.csv_logger = CSVLogger(
            model_output_dir.copy().join('S{:03d}_fold{:03d}_log.csv'.format(subject, fold)).path)
        # self.class_weights = {i: 1 for i in range(num_classes)}
        self.es = EarlyStopping(monitor=monitor, mode=mode,
                                verbose=verbose, patience=es_patience)

        self.reduce_lr = ReduceLROnPlateau(monitor=monitor, patience=patience,
                                           factor=factor, mode=mode, verbose=verbose,
                                           min_lr=min_lr)

    def fit(self, X_train, y_train, X_val, y_val):
        self.model.fit(x=X_train, y=[X_train, y_train, y_train],
                       batch_size=self.batch_size, shuffle=self.shuffle,
                       epochs=self.epochs,
                       validation_data=(X_val, [X_val, y_val, y_val]
                                        ),
                       callbacks=[self.checkpointer, self.csv_logger, self.reduce_lr, self.es],
                       # class_weight=self.class_weights
                       )
        if self.filepath.exists():
            self.model.load_weights(self.filepath.path)
        else:
            print("No checkpoint found: {}".format(self.filepath.path))
        return self

    def predict(self, X_test):
        y_pred_decoder, y_pred_trip, y_pred_clf = self.model.predict(X_test)
        y_pred = np.argmax(y_pred_clf, axis=1)
        encoder = self.model.get_layer("encoder")
        embeddings = encoder(X_test)

        return y_pred, embeddings


__all__ = ["MIN2NetClassifier"]
