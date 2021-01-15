from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import SGD, RMSprop, Adam, Adadelta, Adagrad
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

from itertools import product
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

import warnings

warnings.filterwarnings("ignore")


# wczytanie treningowego zbioru danych
def load_dataset():
    # wczytanie zbioru benchmarkowego
    (trainX, trainY), (testX, testY) = mnist.load_data()

    # dostosowanie zbioru do zadania klasyfikacji
    trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
    testX = testX.reshape((testX.shape[0], 28, 28, 1))

    trainY = to_categorical(trainY)
    testY = to_categorical(testY)
    return trainX, trainY, testX, testY


# normalizacja obrazow testowych i treningowych
def prepare_dataset(train, test):
    # konwersja do typu float
    train_normalized = train.astype('float32')
    test_normalized = test.astype('float32')
    # normalizacja do zakresu 0-1
    train_normalized = train_normalized / 255.0
    test_normalized = test_normalized / 255.0

    return train_normalized, test_normalized


# utworzenie modelu - zbudowanie prostego MLP
def prepare_model(lr, lw):
    # wspolne
    model = Sequential()
    #
    if (lw == 1):
        ## dla wariantu 1w
        model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
        model.add(MaxPooling2D((2, 2)))
    elif (lw == 2):
        ## dla wariantu 2w
        model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform'))
        model.add(MaxPooling2D((2, 2)))

    elif (lw == 3):
        ## dla wariantu 3w
        model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_uniform'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(8, (3, 3), activation='relu', kernel_initializer='he_uniform'))
        model.add(MaxPooling2D((2, 2)))

    ## wspólne
    model.add(Flatten())
    model.add(Dense(10, activation='softmax'))
    # najskuteczniejsze
    opt = SGD(lr=lr, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


# program
# zaladowanie zbioru danych
trainX, trainY, testX, testY = load_dataset()
# przygotowanie danych
trainX, testX = prepare_dataset(trainX, testX)

# czesc 1
# liczba epok
epochs = [10]
# krok uczenia
learning_rates = [0.1, 0.01, 0.001]
# learning_rates = [0.1]
# liczba warstw
LW = [1, 2, 3]

list_epoch = []
list_lr = []
list_layers = []
list_cost = []
list_acc = []

t1 = time.time()
for elem in product(epochs, LW, learning_rates):
    epo = elem[0]
    lw = elem[1]
    lr = elem[2]
    print('Epoki: ', epo, 'learning rate: ', lr, 'Warstwy: ', lw)

    # utworzenie modelu
    model = prepare_model(lr=lr, lw=lw)
    model.fit(trainX, trainY, epochs=epo, batch_size=256, verbose=0)
    eval = model.evaluate(testX, (testY))

    list_epoch.append(epo)
    list_lr.append(lr)
    list_layers.append(lw)
    list_cost.append(eval[0])
    list_acc.append(eval[1])

t2 = time.time() - t1
print('Czas trwania: ', t2, ' sekund')
dane = {'Epoki': list_epoch, 'Warstwy': list_layers,
        'Krok uczenia': list_lr, 'Koszt': list_cost, 'Dokladnosc': list_acc}
df = pd.DataFrame(dane)
df.to_excel("output" + str(epochs[0]) + ".xlsx")