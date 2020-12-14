import matplotlib
import matplotlib.pyplot as plt
import cv2
import skimage
from skimage import data
from skimage.filters import try_all_threshold
import os
from skimage.color import rgb2gray,rgba2rgb
from skimage import io
from skimage.filters import threshold_otsu, threshold_minimum, threshold_triangle, threshold_local, rank, threshold_yen
from skimage.morphology import disk
import numpy as np
import matplotlib.image as im
from scipy.spatial import distance


def fourier_desc(img, size):
    """

    :param img:
    :param size:
    :return:
    """
    img_fft = np.fft.fft2(img)
    spectrum = np.log(1 + np.abs(img_fft))

    out = []
    for i in range(0, size):
        tmp = []
        for j in range(0, size):
            tmp.append(spectrum[i][j])
        out.append(tmp)
    return list(np.concatenate(out).flat)


def fit_fft(train, size):
    train_learned = []
    for i, item in enumerate(train):
        train_learned.append(fourier_desc(item, size))
    return train_learned


def classify_fft(img, train, name, size):
    tmp = fourier_desc(img, size)
    #print(tmp)
    res = enc_idx(find_nearest(train, tmp))
    out = []
    # print("Deskryptor - Fourier: ", res)
    out.append(1) if res == name else out.append(0)
    return out


def enc_idx(i):
    n = ["sign_1", "sign_2", "sign_3", "sign_4", "sign_5", "sign_6", "sign_7", "sign_8", "sign_9", "sign_10"]
    return n[i]


def find_nearest(array, value):
    dist=[]
    array = np.asarray(array)
    value = np.asarray(value)
    for i in range(len(array)):
        dist.append(np.linalg.norm(array[i] - value))
    dist = np.asarray(dist)
    idx = dist.argmin()
    return idx


    # distances = distance.cdist([value], array, "cosine")[0]
    # myvals = np.dot(array, value)
    # max_index = np.argmax(distances)
    # max_similarity = distances[max_index]
    # print("Most similar vector to target is index %s with %s" % (max_index, max_similarity))
    # return max_index


def score(scores):
    # print(scores)
    res = 0
    for i in range(len(scores)):
        res += scores[i][0]
    res = res / len(scores) * 100
    return res


if __name__ == "__main__":
    print('Dwuwymiarowy deskryptor Fouriera (2D Fourier)')
    sizes=[5, 10, 15, 20 ,50]
    for s in sizes:
        train = []
        for i in range(1, 11):
            p = "./TRAIN/sign_" + str(i) + "_1.png"
            img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            train.append(img)

        clf = fit_fft(train, s)

        folders = ["sign_1", "sign_2", "sign_3", "sign_4", "sign_5", "sign_6", "sign_7", "sign_8", "sign_9", "sign_10"]
        results = []
        for name in folders:
            for i in range(4, 14):
                filename = "./TEST_DATA_FOURIER/" + name + "/sign_" + str(i) + ".png"
                img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
                results.append(classify_fft(img, clf, name, s))

        scores = score(results)
        print("Rozmiar => ",s,"x",s,"| Poprawność => ", scores, "%")