import matplotlib
import matplotlib.pyplot as plt
import cv2
import skimage
from skimage import data
from skimage.color import rgb2gray, rgba2rgb
from skimage import io
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.image as im
import scipy.ndimage as ndimage
from collections import defaultdict
from PIL import Image


def color_dominant_desc(filename):
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.reshape((img.shape[0] * img.shape[1], 3))
    kmeans = KMeans(3)
    kmeans.fit(img)
    color_result = kmeans.cluster_centers_
    print(f'dominant_color ==> {color_result.astype(int)}')
    return color_result.astype(int)


def color_mean_desc(filename):
    img = cv2.imread(filename)
    avg_color_per_row = np.average(img, axis=0)
    avg_color = np.average(avg_color_per_row, axis=0)
    return avg_color


def color_variance_desc(filename):
    img = cv2.imread(filename)
    lbl, nlbl = ndimage.label(img)
    return ndimage.variance(img, lbl, index=np.arange(1, nlbl+1))


def color_hist_desc(filename):
    img = cv2.imread(filename)
    r = cv2.calcHist([img], [2], mask=None, histSize=[64], ranges=[0, 256])
    g = cv2.calcHist([img], [1], mask=None, histSize=[64], ranges=[0, 256])
    b = cv2.calcHist([img], [0], mask=None, histSize=[64], ranges=[0, 256])
    return np.ravel(b).tolist() + np.ravel(g).tolist() + np.ravel(r).tolist()

def color_layout_desc(filename):
    img = cv2.imread(filename)
    averages = np.zeros((8, 8, 3))
    imgH, imgW, _ = img.shape
    for row in range(8):
        for col in range(8):
            slice = img[imgH // 8 * row: imgH //8 * (row + 1),
                    imgW // 8 * col: imgW // 8 * (col + 1)]
            average_color_per_row = np.mean(slice, axis=0)
            average_color = np.mean(average_color_per_row, axis=0)
            average_color = np.uint8(average_color)
            averages[row][col][0] = average_color[0]
            averages[row][col][1] = average_color[1]
            averages[row][col][2] = average_color[2]
    icon = cv2.cvtColor(
        np.array(averages, dtype=np.uint8), cv2.COLOR_BGR2YCR_CB)
    y, cr, cb = cv2.split(icon)
    dct_y = cv2.dct(np.float64(y))
    dct_cb = cv2.dct(np.float64(cb))
    dct_cr = cv2.dct(np.float64(cr))
    dct_y_zigzag = []
    dct_cb_zigzag = []
    dct_cr_zigzag = []
    flip = True
    flipped_dct_y = np.fliplr(dct_y)
    flipped_dct_cb = np.fliplr(dct_cb)
    flipped_dct_cr = np.fliplr(dct_cr)
    for i in range(8 + 8 - 1):
        k_diag = 8 - 1 - i
        diag_y = np.diag(flipped_dct_y, k=k_diag)
        diag_cb = np.diag(flipped_dct_cb, k=k_diag)
        diag_cr = np.diag(flipped_dct_cr, k=k_diag)
        if flip:
            diag_y = diag_y[::-1]
            diag_cb = diag_cb[::-1]
            diag_cr = diag_cr[::-1]
        dct_y_zigzag.append(diag_y)
        dct_cb_zigzag.append(diag_cb)
        dct_cr_zigzag.append(diag_cr)
        flip = not flip
    return np.concatenate(
        [np.concatenate(dct_y_zigzag), np.concatenate(dct_cb_zigzag), np.concatenate(dct_cr_zigzag)])

def fit_colors(train):
    train_learned = []
    for item in train:
        train_learned.append(color_dominant_desc(item))
        train_learned.append(color_mean_desc(item))
        train_learned.append(color_variance_desc(item))
        train_learned.append(color_hist_desc(item))
        train_learned.append(color_layout_desc(item))
    return train_learned


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def classifications(img, train, name):
    tmp1 = color_dominant_desc(img)
    tmp2 = color_mean_desc(img)
    tmp3 = color_variance_desc(img)
    tmp4 = color_hist_desc(img)
    tmp5 = color_layout_desc(img)
    res = [enc_idx(find_nearest(train[0], tmp1)),
           enc_idx(find_nearest(train[1], tmp2)),
           enc_idx(find_nearest(train[2], tmp3)),
           enc_idx(find_nearest(train[3], tmp4)),
           enc_idx(find_nearest(train[4], tmp5))]
    out = []
    for item in res:
        out.append(1) if item == name else out.append(0)
    return out

def enc_idx(i):
    n = ["sign_1", "sign_3", "sign_4", "sign_5", "sign_6", "sign_7", "sign_8", "sign_9", "sign_10" , "sign_11"]
    return n[i % 9]

def score(scores):
    # print(scores)
    res = [0, 0, 0, 0, 0]
    for i in range(len(scores)):
        res[0] += scores[i][0]
        res[1] += scores[i][1]
        res[2] += scores[i][2]
        res[3] += scores[i][3]
        res[4] += scores[i][4]
    for i in range(len(res)):
        res[i] = res[i] / len(scores) * 100
    return res

if __name__ == "__main__":
    # p = "./TRAIN_WITHOUT_BG/sign_" + str(9) + "_1.png"
    # img_test = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    # res = dominant_color_desc(p)
    # print(res)

    train = []
    for i in range(1, 10):
        p = "./TRAIN_WITHOUT_BG/sign_" + str(i) + "_1.png"
        train.append(p)

    clf = fit_colors(train)

    folders = ["sign_1", "sign_3", "sign_4", "sign_5", "sign_6", "sign_7", "sign_8", "sign_9", "sign_10", "sign_11"]
    results = []
    # enc_dict = enc_idx()
    for name in folders:
        for i in range(4, 14):
            filename = "./TEST_WITHOUT_BG/" + name + "/sign_" + str(i) + ".png"
            results.append(classifications(filename, clf, name))

    scores = score(results)
    descs=["color_dominant_desc", "color_mean_desc", "color_variance_desc", "color_hist_desc", "color_layout_desc"]
    for i, d in enumerate(descs):
        print("-------------------------------------------------------------")
        print("Poprawność - " + d + ": ", scores[i], "%")
