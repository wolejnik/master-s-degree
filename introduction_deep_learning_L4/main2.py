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

    # X_train = X_train / 255
    # X_test = X_test / 255

    return X_train, X_test, y_train, y_test


class MyModel(tf.keras.Model):

  def __init__(self):
    super(MyModel, self).__init__()
    self.dense1 = tf.keras.layers.Conv2D(32, kernel_size=(3, 3))
    self.dense2 = tf.keras.layers.Conv2D(32, kernel_size=(3, 3))
    self.dense3 = tf.keras.layers.Flatten()
    self.dense4 = tf.keras.layers.Dense(4, activation=tf.nn.relu)
    self.dense5 = tf.keras.layers.Dense(5, activation=tf.nn.softmax)
    self.dropout = tf.keras.layers.Dropout(0.5)

  def call(self, inputs, training=False):
    x = self.dense1(inputs)
    if training:
      x = self.dropout(x, training=training)
    return self.dense2(x)

model = MyModel()


def main():
    """ Main program """
    X_train, X_test, y_train, y_test = load_images();
    # X_train = tf.data.Dataset.from_tensors(X_train)
    # # X_test = tf.data.Dataset.from_tensors(X_test)
    # y_train = tf.data.Dataset.from_tensors(y_train)
    # # y_test = tf.data.Dataset.from_tensors(y_test)

    # print(type(X_train))
    #
    # batch_size = 128
    # num_classes = 10
    # epochs = 12
    #
    # model = Sequential()
    # model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    # model.add(MaxPooling2D((2, 2)))
    # model.add(Conv2D(64, (3, 3), activation='relu'))
    # model.add(MaxPooling2D((2, 2)))
    # model.add(Conv2D(64, (3, 3), activation='relu'))
    # model.add(Flatten())
    # model.add(Dense(64, activation='relu'))
    # model.add(Dense(10, activation='softmax'))
    # # Few simple configurations
    # model.compile(loss='categorical_crossentropy',
    #               optimizer=SGD(momentum=0.5, decay=0.0004), metrics=['accuracy'])
    # # Run the algorithm!
    # model.fit(X_train, y_train, epochs=25, batch_size=10)
    # res = model.evaluate(X_test, y_test)
    # # Save the weights to use for later
    # # model.save_weights("cifar10.hdf5")
    # # Finally print the accuracy of our model!
    # print("Accuracy: &2.f%%" % (model.evaluate(X_test, y_test)[1] * 100))

    # model = Sequential()
    # model.add(Conv2D(32, kernel_size=(3, 3),
    #                  activation='relu',
    #                  input_shape=(img_rows, img_cols, 1)))
    # model.add(Conv2D(64, (3, 3), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))
    # model.add(Flatten())
    # model.add(Dense(128, activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(num_classes, activation='softmax'))
    #
    # model.compile(loss=tf.keras.losses.categorical_crossentropy,
    #               optimizer=tf.keras.optimizers.Adadelta(),
    #               metrics=['accuracy'])
    #
    # model.fit(X_train, y_train,
    #           batch_size=batch_size,
    #           epochs=epochs,
    #           verbose=1,
    #           validation_data=(X_test, y_test))
    # score = model.evaluate(X_test, y_test, verbose=0)
    # print('Test loss:', score[0])
    # print('Test accuracy:', score[1])

    model = create_model(X_train[0].shape)
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=True)
    score = model.evaluate(X_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    return 0

if __name__ == "__main__":
    main()