import matplotlib
import matplotlib.pyplot as plt
import cv2
import skimage
from skimage import data
from skimage.color import rgb2gray, rgba2rgb
from skimage import io
import numpy as np
import matplotlib.image as im
from scipy.interpolate import UnivariateSpline
from collections import defaultdict
import math


def unl_fourier_desc(img, size):
    #cartesian coordinates of image to polar coordinates
    thresh = cv2.Canny(img, 30, 200)
    value = np.sqrt(((thresh.shape[0] / 2.0) ** 2.0) + ((thresh.shape[1] / 2.0) ** 2.0))
    polar_image = cv2.linearPolar(thresh, (thresh.shape[0] / 2, thresh.shape[1] / 2), value, cv2.WARP_FILL_OUTLIERS)
    polar_image = polar_image.astype(np.uint8)
    cv2.imshow("Image", thresh)
    cv2.imshow("Polar Image", polar_image)
    cv2.waitKey(0)

    # fourier
    img_fft = np.fft.fft2(polar_image)
    spectrum = np.log(1 + np.abs(img_fft))
    # plt.imshow(spectrum, "gray")
    # plt.title("Spectrum")
    # plt.show()
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
        train_learned.append(unl_fourier_desc(item, size))
    return train_learned

def classify_signature(img, train, name, size):
    tmp = unl_fourier_desc(img, size)
    res = enc_idx(find_nearest(train, tmp))
    out = []
    #print("Deskryptor - UNL-Fourier: ", res)
    out.append(1) if res == name else out.append(0)
    return out

def enc_idx(i):
    n = ["sign_1", "sign_2", "sign_3", "sign_4", "sign_5", "sign_6", "sign_7", "sign_8", "sign_9", "sign_10"]
    return n[i]

def find_nearest(array, value):
    dist = []
    array = np.asarray(array)
    value = np.asarray(value)
    for i in range(len(array)):
        dist.append(np.linalg.norm(array[i] - value))
    dist = np.asarray(dist)
    idx = dist.argmin()
    return idx


def score(scores):
    res = 0
    for i in range(len(scores)):
        res += scores[i][0]
    res = res / len(scores) * 100
    return res


if __name__ == "__main__":

    print('Deskryptor UNL Fourier')
    print("-------------------------------------------------------------")

    sizes = [5, 10, 15, 20, 30, 50]
    for s in sizes:
        train = []
        for i in range(1, 11):
            p = "./TRAIN/sign_" + str(i) + "_1.png"
            img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            train.append(img)

        clf = fit_fft(train, s)

        folders = ["sign_1", "sign_2", "sign_3", "sign_4", "sign_5", "sign_6", "sign_7", "sign_8", "sign_9", "sign_10", "sign_11"]
        results = []
        for name in folders:
            for i in range(4, 14):
                filename = "./TEST_DATA_FOURIER/" + name + "/sign_" + str(i) + ".png"
                img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
                results.append(classify_signature(img, clf, name, s))

        scores = score(results)
        print("UNL Fourier => ", s, "x", s, ": ", scores, "%")

