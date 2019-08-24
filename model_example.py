import tensorflow as tf
import keras
from keras import backend as K

from keras.layers import Dense, Dropout, Activation, GlobalAveragePooling2D, Input
from keras.losses import categorical_crossentropy
from keras.models import Model
from keras.optimizers import Nadam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.metrics import categorical_accuracy
from keras.applications import *
from keras.preprocessing.image import ImageDataGenerator

from .cutmix_generator import CutMixGenerator

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


def get_steps(num_samples, batch_size):
    if (num_samples % batch_size) > 0:
        return (num_samples // batch_size) + 1
    else:
        return num_samples // batch_size

def get_callback(model_path):
    callback_list = [
        ModelCheckpoint(filepath=model_path, monitor='val_f1_m',
                        verbose=1, save_best_only=True, mode='max'),
        ReduceLROnPlateau(monitor='val_f1_m',
                          factor=0.2,
                          patience=4,
                          min_lr=1e-7,
                          cooldown=1,
                          verbose=1, mode='max'),
        EarlyStopping(monitor='val_f1_m', patience=10, mode='max')
    ]
    return callback_list


class Lam_Loss(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Lam_Loss, self).__init__(**kwargs)

    def call(self, inputs):
        loss = 0
        targets, shuffled_targets, lam = inputs
        enable_loss = tf.cond(tf.count_nonzero(shuffled_targets) > 0, lambda: True, lambda: False)

        loss = tf.cond(enable_loss,
                       lambda: lam * categorical_crossentropy(targets, shuffled_targets) * (
                                   1 - lam) * categorical_crossentropy(targets, shuffled_targets), lambda: 0.)
        self.add_loss(loss)

        return loss


def create_model(BASE_MODEL, training=True):
    if training:
        image = Input((299, 299, 3))
        shuffled_targets = Input((196,))
        lam = Input((1,))
    else:
        image = Input((299, 299, 3))
    return create_base_model([image, shuffled_targets, lam], BASE_MODEL, training)


def create_base_model(inputs, BASE_MODEL, training=True):
    if training and (len(inputs) > 1):
        image, shuffled_targets, lam = inputs
        loss = Lam_Loss()
    else:
        image = inputs
        shuffled_targets = None
        lam = None

    base_model = BASE_MODEL(weights=None, input_shape=(299, 299, 3), include_top=False)

    x = base_model(image)
    x = GlobalAveragePooling2D()(x)
    x = Dense(2048, kernel_initializer='he_normal')(x)
    x = Dropout(0.3)(x)
    x = Activation('relu')(x)
    x = Dense(196, activation='softmax')(x)

    if (training):
        lam_loss = loss([x, shuffled_targets, lam])

    model = Model(inputs=inputs, outputs=x)
    if (training):
        nadam = Nadam(lr=lr)

        model.compile(optimizer=nadam, loss='categorical_crossentropy', metrics=[categorical_accuracy,
                                                        f1_m, precision_m, recall_m])

    return model

if __name__ == '__main__':
    lr = 1e-4

    model = create_model(InceptionResNetV2, training=True)

    # example
    train_datagen = ImageDataGenerator(
        rotation_range=60,
        brightness_range=[0.5, 1.5],
        #     shear_range = 0.25,
        width_shift_range=0.30,
        height_shift_range=0.30,
        horizontal_flip=True,
        vertical_flip=False,
        zoom_range=0.25,
        fill_mode='nearest',
        rescale=1. / 255)

    val_datagen = ImageDataGenerator(rescale=1. / 255)

    # change your's
    train_generator = CutMixGenerator(TRAIN_DF, train_datagen, alpha=1, num_classes=196, batch_size=BATCH_SIZE,
                                      img_size=IMG_SIZE, cutmix_in_train=True)
    val_generator = CutMixGenerator(VAL_DF, val_datagen, alpha=1, num_classes=196,
                                    batch_size=BATCH_SIZE, img_size=IMG_SIZE, cutmix_in_train=False)


    ############# fit your model ###############
    #...
