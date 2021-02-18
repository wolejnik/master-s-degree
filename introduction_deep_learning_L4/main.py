import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import SGD
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.layers import Activation, BatchNormalization
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt


def load_images():
    images = []
    labels = []
    for i in range(40):
        base_path = os.path.sep.join([".//data", f"s{i+1}"])
        for j in range(10):
            image_path = os.path.sep.join([base_path, f"{j+1}.pgm"])
            image = plt.imread(image_path)
            images.append(image)
            labels.append(i)

    X = np.array(images)
    y = np.array(labels)
    y = tf.keras.utils.to_categorical(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=42
    )

    return X_train, X_test, y_train, y_test



def create_model(img_shape):
    inputShape = (img_shape[0], img_shape[1], 1)

    filters = (32, 64, 128)
    inputs = Input(shape=inputShape)

    # for (i, f) in enumerate(filters):

    # if i == 0:
    x = inputs
        # # CONV => RELU => BN => POOL
        # x = Conv2D(f, (3, 3), padding="same", activation='relu')(x)
        # x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(32, (3, 3), padding="same", activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(64, (3, 3), padding="same", activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(128, (3, 3), padding="same", activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.25)(x)
    x = Flatten()(x)
    x = Dense(40, activation='softmax')(x)

    model = Model(inputs, x)

    opt = SGD(lr=1e-3, momentum=1e-3 / 200)

    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])

    return model

def test_model_noise():
    X_train, X_test, y_train, y_test = load_images()

    height, width = X_test[0].shape
    noises = np.arange(0, 100, 5)
    # noises = np.arange(-100, 100, 5)

    acc = []

    model = create_model(X_train[0].shape)
    model.fit(X_train, y_train, epochs=100, batch_size=64, verbose=True)
    # res = model.evaluate(X_test, y_test)

    for noise in noises:
        X_test_2 = X_test + noise * np.random.randn(height, width)
        # X_test_2 = X_test + noise

        # model = load_model("model.h5")
        res = model.evaluate(X_test_2, y_test)
        print(f"Noise: {noise}, Res: {res}")
        acc.append(res[1])

    plt.plot(noises, acc)
    plt.xlabel("Szum")
    plt.ylabel("Dokładność")
    plt.show()

def test_model_brighness():
    X_train, X_test, y_train, y_test = load_images()

    height, width = X_test[0].shape
    brighnesses = np.arange(-100, 100, 5)

    acc = []
    model = create_model(X_train[0].shape)
    model.fit(X_train, y_train, epochs=100, batch_size=64, verbose=True)
    # res = model.evaluate(X_test, y_test)

    for brighness in brighnesses:
        X_test_2 = X_test + brighness

        # model = load_model("model.h5")
        res = model.evaluate(X_test_2, y_test)
        print(f"Noise: {brighness}, Res: {res}")
        acc.append(res[1])

    plt.plot(brighnesses, acc)
    plt.xlabel("Jasność")
    plt.ylabel("Dokładność")
    plt.show()


def main():
    """ Main program """
    # X_train, X_test, y_train, y_test = load_images();
    test_model_brighness()
    # model = create_model(X_train[0].shape)
    # model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=True)
    # score = model.evaluate(X_test, y_test, verbose=0)
    # print('Test loss:', score[0])
    # print('Test accuracy:', score[1])

    return 0

if __name__ == "__main__":
    main()