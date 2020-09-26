# -*- coding: utf-8 -*-
# Author: zhao chen
# Date: 2020/9/19

from keras import Model
from keras.layers import Input, Conv2D, Flatten, Dense, MaxPool2D
from keras.preprocessing import image
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import numpy as np
import keras.backend.tensorflow_backend as KTF


def VGG16(num_classes):
    image_input = Input(shape=(224, 224, 3))

    # block1
    x = Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same", name="block1_conv1")(image_input)
    x = Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same", name="block1_conv2")(x)
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name="block1_pool")(x)

    # block2
    x = Conv2D(filters=128, kernel_size=(3, 3), activation="relu", padding="same", name="block2_conv1")(x)
    x = Conv2D(filters=128, kernel_size=(3, 3), activation="relu", padding="same", name="block2_conv2")(x)
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name="block2_pool")(x)

    # block3
    x = Conv2D(filters=256, kernel_size=(3, 3), activation="relu", padding="same", name="block3_conv1")(x)
    x = Conv2D(filters=256, kernel_size=(3, 3), activation="relu", padding="same", name="block3_conv2")(x)
    x = Conv2D(filters=256, kernel_size=(3, 3), activation="relu", padding="same", name="block3_conv3")(x)
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name="block3_pool")(x)

    # block4
    x = Conv2D(filters=512, kernel_size=(3, 3), activation="relu", padding="same", name="block4_conv1")(x)
    x = Conv2D(filters=512, kernel_size=(3, 3), activation="relu", padding="same", name="block4_conv2")(x)
    x = Conv2D(filters=512, kernel_size=(3, 3), activation="relu", padding="same", name="block4_conv3")(x)
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name="block4_pool")(x)

    # block5
    x = Conv2D(filters=512, kernel_size=(3, 3), activation="relu", padding="same", name="block5_conv1")(x)
    x = Conv2D(filters=512, kernel_size=(3, 3), activation="relu", padding="same", name="block5_conv2")(x)
    x = Conv2D(filters=512, kernel_size=(3, 3), activation="relu", padding="same", name="block5_conv3")(x)
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name="block5_pool")(x)

    #
    x = Conv2D(filters=256, kernel_size=(7, 7), activation="relu", padding="valid", name="block6_conv4")(x)
    x = Flatten(name="flatten")(x)
    x = Dense(256, activation="relu", name="fc1")(x)
    x = Dense(256, activation="relu", name="fc2")(x)
    x = Dense(num_classes, activation="softmax", name="output")(x)

    model = Model(image_input, x, name="vgg16")

    return model


def run(args):
    data_path = args.data_path
    model_save_path = args.model_save_path
    batch_size = args.batch_size
    pre_batch_size = args.pre_batch_size
    USE_GPU = args.use_gpu

    if USE_GPU:
        # import os
        # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        import tensorflow as tf
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        KTF.set_session(sess)

    model = VGG16(2)
    model.summary()

    model.compile(optimizer=Adam(), loss="categorical_crossentropy", metrics=["crossentropy", "acc"])

    train_gen = image.ImageDataGenerator(rescale=1.0 / 225, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
    test_gen = image.ImageDataGenerator(rescale=1.0 / 225)

    train_data = train_gen.flow_from_directory(data_path + "/train", target_size=(224, 224), batch_size=batch_size,
                                               class_mode="categorical")

    valid_data = test_gen.flow_from_directory(data_path + "/valid", target_size=(224, 224), batch_size=pre_batch_size,
                                              class_mode="categorical")

    test_data = test_gen.flow_from_directory(data_path + "/test", target_size=(224, 224), batch_size=pre_batch_size,
                                             class_mode="categorical")

    plateau = ReduceLROnPlateau(monitor="val_loss", verbose=0, mode='min', factor=0.1, patience=2)

    earlyStopping = EarlyStopping(min_delta=0.01,
                                  patience=3)
    checkpoint = ModelCheckpoint("{0}VGG16-model.h5".format(model_save_path),
                                 monitor='val_loss', verbose=1, mode='min', save_best_only=True)

    model.fit_generator(train_data, steps_per_epoch=np.ceil(2000 / 4), epochs=2, validation_data=valid_data,
                        validation_steps=int(1000 / 128), callbacks=[plateau, earlyStopping, checkpoint])

    preds = model.evaluate_generator(test_data)

    print(preds)


if __name__ == '__main__':
    import argparse
    import time

    start = time.time()

    parser = argparse.ArgumentParser(description="behavior sequence model script")
    parser.add_argument("--data_path", type=str, default="/home/mi/data/inputs/kerasDataSmall/")
    parser.add_argument("--model_save_path", type=str, default="/home/mi/data/outputs/h5/")
    parser.add_argument("--batch_size", type=int, default=36)
    parser.add_argument("--pre_batch_size", type=int, default=256)
    parser.add_argument("--use_gpu", type=bool, default=False)
    args = parser.parse_args()
    run(args)

    end = time.time()
    print("train model used time: {} m".format(round((end - start) / 60), 2))
