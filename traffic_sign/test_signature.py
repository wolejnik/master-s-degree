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


def signature_desc(img):
    ret, thresh = cv2.threshold(img, 0, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    M = cv2.moments(contours[-1])
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        cX, cY = 0, 0
    centroid = [cX, cY]
    # cv2.circle(img, (cX, cY), 5, (255, 255, 255), -1)
    # cv2.putText(img, "centroid", (cX - 25, cY - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    # cv2.imshow("Image", img)
    # cv2.waitKey(0)
    dist = []
    for c in contours[-1]:
        dist.append(np.linalg.norm(c - centroid))
    return dist


def fit_signature(train):
    train_learned = []
    for item in train:
        train_learned.append(signature_desc(item))
    default_size = max([len(i) for i in train_learned])
    train_learned2 = interpolate_to_fixed_size(train_learned, default_size)
    return train_learned2


def interpolate_to_fixed_size(given_list, default_size):
    for i in range(len(given_list)):
        # old_indices = np.arange(0, len(given_list[i]))
        new_indices = np.linspace(0, len(given_list[i]) - 1, default_size)
        # spl = UnivariateSpline(old_indices, given_list[i])
        x = np.linspace(0, 1, 200)
        y = x ** 2
        spl = UnivariateSpline(x, y)
        given_list[i] = spl(new_indices)
    return given_list

def classify_signature(img, train, name):
    tmp = signature_desc(img)
    default_size = max([len(i) for i in train])
    img2 = interpolate_to_fixed_size([tmp], default_size)
    res = find_nearest(train, img2)
    out = []
    out.append(1) if res == name else out.append(0)
    return out

def find_nearest(array, value):
    dist=[]
    array = np.asarray(array)
    value = np.asarray(value)
    for i in range(len(array)):
        dist.append(np.linalg.norm(array[i] - value))
    dist = np.asarray(dist)
    idx = dist.argmin()
    n = ["sign_1", "sign_2", "sign_3", "sign_4", "sign_5", "sign_6", "sign_7", "sign_8", "sign_9", "sign_10"]
    return n[idx]


def score(scores):
    res = 0
    for i in range(len(scores)):
        res += scores[i][0]
    res = res / len(scores) * 100
    return res

if __name__ == "__main__":
    train = []
    for i in range(1, 11):
        p = "./TRAIN/sign_" + str(i) + "_1.png"
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        train.append(img)

    clf = fit_signature(train)

    folders = ["sign_1", "sign_2", "sign_3", "sign_4", "sign_5", "sign_6", "sign_7", "sign_8", "sign_9", "sign_10"]
    results = []
    for name in folders:
        for i in range(4, 14):
            filename = "./TEST_DATA_FOURIER/" + name + "/sign_" + str(i) + ".png"
            img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
            results.append(classify_signature(img, clf, name))

    scores = score(results)
    print("Poprawność Sygnatura: ", scores, "%")




