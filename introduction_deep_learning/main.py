import numpy as np
import pandas as pd
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import SGD, Adam, Adadelta, Adagrad, RMSprop
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt


def load_set():
    (trainX, trainY), (testX, testY) = mnist.load_data()
    trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
    testX = testX.reshape((testX.shape[0], 28, 28, 1))
    trainY = to_categorical(trainY)
    testY = to_categorical(testY)

    return trainX, trainY, testX, testY


def preprocessing(train, test):
    train_norm = train.astype('float32')
    test_norm = test.astype('float32')
    train_norm /= 255.0
    test_norm /= 255.0

    return train_norm, test_norm


def create_model(act='sigmoid', optimizer='SGD', l=0.001, m=0.9):
    model = Sequential()
    model.add(Flatten())
    model.add(Dense(10, activation=act, input_shape=(28 * 28, 1)))
    if optimizer == 'SGD':
        opt = SGD(lr=l, momentum=m)
    elif optimizer == 'Adam':
        opt = Adam(lr=l)
    elif optimizer == 'Adadelta':
        opt = Adadelta(lr=l)
    elif optimizer == 'Adagrad':
        opt = Adagrad(lr=l)
    else:
        opt = RMSprop(lr=l, momentum=m)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def load_preprocess_image(fname, modification=1):
    img = load_img(fname, color_mode="grayscale", target_size=(28, 28))
    if modification == 1:
        img = img + 50 * np.random.randn(28, 28)
    elif modification == 2:
        img = np.rot90(img)
    elif modification == 3:
        img = np.roll(img, -5, axis=0)
    elif modification == 4:
        img = np.roll(img, -5, axis=1)
    img = img_to_array(img)
    img = img.reshape(1, 28, 28, 1)
    img = img.astype('float32')
    img /= 255.0
    return img


if __name__ == '__main__':
    # test data
    trainX, trainY, testX, testY = load_set()
    trainX, testX = preprocessing(trainX, testX)

    # #creating model
    model = create_model()
    model.fit(trainX, trainY, epochs=10, batch_size=256, verbose=1)
    model.save("model_mlp.h5")

    # # testing model
    res = model.evaluate(testX, (testY))
    print(res)

