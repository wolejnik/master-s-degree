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
    default_size = find_max_list(train_learned)
    train_learned2 = interpolate_to_fixed_size(train_learned, default_size)
    return train_learned2


def interpolate_to_fixed_size(given_list, default_size):
    for i in range(len(given_list)):
        old_indices = np.arange(0, len(given_list[i]))
        new_indices = np.linspace(0, len(given_list[i]) - 1, default_size)
        spl = UnivariateSpline(old_indices, given_list[i], k=1, s=0)
        given_list[i] = spl(new_indices)
    return given_list


def find_max_list(given_list):
    list_len = [len(i) for i in given_list]
    return(max(list_len))


def classify_signature(img, train, name):
    tmp = signature_desc(img)
    default_size = find_max_list(train)
    img2 = interpolate_to_fixed_size([tmp], default_size)
    res = enc_idx(find_nearest(train, img2))
    out = []
    print("Deskryptor - Sygnatura: ", res)
    out.append(1) if res == name else out.append(0)
    return out


def enc_idx(i):
    n = ["black-and-tan_coonhound", "borzoi", "English_foxhound", "Ibizan_hound", "Irish_water_spaniel",
               "Kerry_blue_terrier", "komondor", "otterhound", "Sealyham_terrier", "whippet"]
    if i == 0 or i == 1 or i == 20 or i == 21 or i == 22:
        return n[0]
    elif i == 2 or i == 3 or i == 23 or i == 24 or i == 25:
        return n[1]
    elif i == 4 or i == 5 or i == 26 or i == 27 or i == 28:
        return n[2]
    elif i == 6 or i == 7 or i == 29 or i == 30 or i == 31:
        return n[3]
    elif i == 8 or i == 9 or i == 32 or i == 33 or i == 34:
        return n[4]
    elif i == 10 or i == 11 or i == 35 or i == 36 or i == 37:
        return n[5]
    elif i == 12 or i == 13 or i == 38 or i == 39 or i == 40:
        return n[6]
    elif i == 14 or i == 15 or i == 41 or i == 42 or i == 43:
        return n[7]
    elif i == 16 or i == 17 or i == 44 or i == 45 or i == 46:
        return n[8]
    elif i == 18 or i == 19 or i == 47 or i == 48 or i == 49:
        return n[9]
    else:
        return None


def find_nearest(array, value):
    dist=[]
    array = np.asarray(array)
    value = np.asarray(value)
    for i in range(len(array)):
        dist.append(np.linalg.norm(array[i] - value))
    dist = np.asarray(dist)
    idx = dist.argmin()
    return idx


def score(scores):
    # print(scores)
    res = 0
    for i in range(len(scores)):
        res += scores[i][0]
    res = res / len(scores) * 100
    return res


if __name__ == "__main__":
    # p = "./BAZA/learning/black-and-tan_coonhound/" + str(6) + ".png"
    # # p = "./BAZA/learning/_TRAIN2/" + str(0) + ".png"
    # img_test = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    # res = signature_desc(img_test)
    # print(res)


    train = []
    for i in range(0, 20):
    # for i in range(0, 50):
        p = "./BAZA/learning/_TRAIN/" + str(i) + ".png"
        # p = "./BAZA/learning/_TRAIN2/" + str(i) + ".png"
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        train.append(img)

    clf = fit_signature(train)
    # print(clf)
    # print(classify_signature(img_test, clf, "black-and-tan_coonhound"))

    folders = ["black-and-tan_coonhound", "borzoi", "English_foxhound", "Ibizan_hound", "Irish_water_spaniel",
               "Kerry_blue_terrier", "komondor", "otterhound", "Sealyham_terrier", "whippet"]
    results = []
    for name in folders:
        for i in range(2, 10):
        # for i in range(5, 10):
            filename = "./BAZA/learning/" + name + "/" + str(i) + ".png"
            print("-------------------------------------------------------------")
            print(name + " - " + str(i) + ".png")
            img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
            results.append(classify_signature(img, clf, name))
            print("-------------------------------------------------------------")

    scores = score(results)
    print("-------------------------------------------------------------")
    print("Poprawność - Sygnatura: ", scores, "%")
    print("-------------------------------------------------------------")

    # plt.imshow(img_test, cmap=plt.cm.gray)
    # plt.show()
    # print(img)
    #example_of_descs(img)


