from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import concatenate


def load_attributes(inputPath):
    cols = ["bedrooms", "bathrooms", "area", "zipcode", "price"]
    df = pd.read_csv(inputPath, sep=" ", header=None, names=cols)

    zipcodes = df["zipcode"].value_counts().keys().tolist()
    counts = df["zipcode"].value_counts().tolist()

    for (zipcode, count) in zip(zipcodes, counts):

        if count < 20:
            idxs = df[df["zipcode"] == zipcode].index
            df.drop(idxs, inplace=True)

    return df


def process_attributes(df, train, test):
    continuous = ["bedrooms", "bathrooms", "area"]

    cs = MinMaxScaler()
    trainContinuous = cs.fit_transform(train[continuous])
    testContinuous = cs.transform(test[continuous])

    zipBinarizer = LabelBinarizer().fit(df["zipcode"])
    trainCategorical = zipBinarizer.transform(train["zipcode"])
    testCategorical = zipBinarizer.transform(test["zipcode"])

    trainX = np.hstack([trainCategorical, trainContinuous])
    testX = np.hstack([testCategorical,
                       testContinuous])

    return (trainX, testX)


def create_mlp(dim):
    model = Sequential()
    model.add(Dense(8, input_dim=dim, activation="relu"))
    model.add(Dense(4, activation="relu"))

    return model


def scale(im, nR, nC):
    nR0 = len(im)  # source number of rows
    nC0 = len(im[0])  # source number of columns 
    return [[im[int(nR0 * r / nR)][int(nC0 * c / nC)]
             for c in range(nC)] for r in range(nR)]


def load_images(df, inputPath):
    images = []

    for i in df.index.values:

        basePath = os.path.sep.join([inputPath, "{}_*".format(i + 1)])
        housePaths = sorted(list(glob.glob(basePath)))

        inputImages = []
        outputImage = np.zeros((64, 64, 3), dtype="uint8")

        for housePath in housePaths:
            image = plt.imread(housePath)
            # 4 images
            image = scale(image, 32, 32)
            # only frontal
            # image = scale(image, 64, 64)
            inputImages.append(image)

        # print(inputImages)
        # fig = plt.figure(figsize=(8, 8))
        # columns = 2
        # rows = 2
        # for i in range(1, columns * rows + 1):
        #     fig.add_subplot(rows, columns, i)
        #     plt.imshow(inputImages[i-1])
        # plt.show()

        # 4 images
        outputImage[0:32, 0:32] = inputImages[0]
        outputImage[0:32, 32:64] = inputImages[1]
        outputImage[32:64, 32:64] = inputImages[2]
        outputImage[32:64, 0:32] = inputImages[3]

        # only frontal
        # outputImage[0:64, 0:64] = inputImages[2]

        images.append(outputImage)
        # images.append(outputImage[32:64, 32:64])
    # plt.imshow(images[0])
    # plt.show()
    # plt.imshow(images[1])
    # plt.show()
    # plt.imshow(images[2])
    # plt.show()
    # plt.imshow(images[3])
    # plt.show()

    return np.array(images)


def create_cnn(width, height, depth, filters=(16, 32, 64)):
    inputShape = (height, width, depth)
    chanDim = -1

    inputs = Input(shape=inputShape)

    for (i, f) in enumerate(filters):

        if i == 0:
            x = inputs
        # CONV => RELU => BN => POOL
        x = Conv2D(f, (3, 3), padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=chanDim)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Flatten()(x)
    x = Dense(16)(x)
    x = Activation("relu")(x)
    x = BatchNormalization(axis=chanDim)(x)
    x = Dropout(0.5)(x)

    x = Dense(4)(x)
    x = Activation("relu")(x)

    model = Model(inputs, x)

    return model


def prepare_dataset():
    df = load_attributes('houses/HousesInfo.txt')
    images = load_images(df, 'houses')
    images = images / 255.0

    trainAttrX, testAttrX, trainImagesX, testImagesX = train_test_split(df, images, test_size=0.5)
    maxPrice = trainAttrX["price"].max()
    trainY = np.array(trainAttrX["price"] / maxPrice)
    testY = np.array(testAttrX["price"] / maxPrice)

    (trainAttrX, testAttrX) = process_attributes(df, trainAttrX, testAttrX)

    trainAttrX = np.array(trainAttrX)
    testAttrX = np.array(testAttrX)
    return trainAttrX, trainImagesX, trainY, testAttrX, testImagesX, testY


def create_models(trainAttrX):
    mlp = create_mlp(trainAttrX.shape[1])
    cnn = create_cnn(64, 64, 3)
    combinedInput = concatenate([mlp.output, cnn.output])
    x = Dense(4, activation="relu")(combinedInput)
    x = Dense(1, activation="linear")(x)
    model = Model(inputs=[mlp.input, cnn.input], outputs=x)
    return model


def simulation():
    trainAttrX, trainImagesX, trainY, testAttrX, testImagesX, testY = prepare_dataset()
    m = create_models(trainAttrX)

    lr = [0.0001, 0.001, 0.01]
    epochs = [10, 100, 1000]

    f = open('results_4img.csv', 'w')
    f.write('Epoki,Kroki uczenia,Błąd\n')

    for l in lr:
        for e in epochs:
            opt = Adam(lr=l, decay=0.001 / 200)
            m.compile(loss="mean_absolute_percentage_error", optimizer=opt)
            m.fit(x=[trainAttrX, trainImagesX], y=trainY, validation_data=([testAttrX, testImagesX], testY), epochs=e,
                  batch_size=8)
            print()
            print(f"Epoki: {e}, krok uczenia: {l}")
            print("Średni błąd: {:.2f}%".format(m.evaluate((testAttrX, testImagesX), (testY))))

            f.write(str(e) + ',')
            f.write(str(l) + ',')
            f.write("{:.2f}%".format(m.evaluate((testAttrX, testImagesX), (testY))) + '\n')
    f.close()


if __name__ == "__main__":
    simulation()
