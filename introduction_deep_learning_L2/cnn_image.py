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


# zaladowanie obrazu
def load_image(filename, deformation):
    # wczytanie obrazu z pliku
    img = load_img(filename, color_mode="grayscale", target_size=(28, 28))

    # deformacja
    if deformation == "szum maly":
        img = img + 25 * np.random.randn(28, 28)
    elif deformation == "szum sredni":
        img = img + 65 * np.random.randn(28, 28)
    elif deformation == "szum duzy":
        img = img + 80 * np.random.randn(28, 28)
    elif deformation == "obroty":
        img = np.rot90(img)
    elif deformation == "przesuniecie x":
        img = np.roll(img, 5, axis=1)  # x
    elif deformation == "przesuniecie y":
        img = np.roll(img, 5, axis=0)  # y

    # konwersja do macierzy
    img = img_to_array(img)

    img = img.reshape(1, 28, 28, 1)
    # konwersja do typu float
    img = img.astype('float32')
    # normalizacja
    img = img / 255.0
    return img


# rozpoznawanie przykladu
def test_picture(model, deformation):
    # zaladowanie obrazu
    img = load_image('sample_image.png', deformation)
    # predykcja
    digit = model.predict_classes(img)

    plt.imshow(img.reshape(28, 28), cmap='gray')
    plt.title('wynik klasyfikacji: ' + str(digit[0]) + ', deformacja: ' + deformation)
    plt.savefig(deformation + ".png")
    plt.show()


# program
# zaladowanie zbioru danych
trainX, trainY, testX, testY = load_dataset()
# przygotowanie danych
trainX, testX = prepare_dataset(trainX, testX)

# czesc 2 obraz test
# utworzenie modelu z wybranymi parametrami
lr = 0.01
epo = 100
lw = 2
# utworzenie modelu


model = prepare_model(lr=lr, lw=lw)
model.fit(trainX, trainY, epochs=epo, batch_size=256, verbose=0)
model.save('final_model_cnn' + '.h5')

model = load_model('final_model_cnn' + '.h5')
eval = model.evaluate(testX, (testY))
print('Loss: ', eval[0], ' Accuracy: ', eval[1])

deformations = ['brak', 'szum maly', 'szum sredni', 'szum duzy', 'obroty', 'przesuniecie x', 'przesuniecie y']

# testowanie dla obrazu
for deformation in deformations:
    test_picture(model, deformation)
