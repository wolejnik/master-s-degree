import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy import stats
import matplotlib.pyplot as plt
from sklearn import datasets
import cv2 as cv

import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('DB1_102_2.tif',0)
img = cv2.medianBlur(img,5)

ret,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)

th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,11,2)
skel = cv.ximgproc.thinning(255-th2,thinningType=cv2.ximgproc.THINNING_GUOHALL)

titles = ['Original Image', 'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding', 'To detection']
images = [img, th1, th2, skel]

for i in range(4):
    plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
    plt.show()

    ends = []
    forks = []

    w, h = skel.shape
    for i in range(1, w - 2):
        for j in range(1, h - 2):
            sum = 0
            for k in range(3):
                for l in range(3):
                    sum = sum + skel[i + k, j + l]

            if sum / 255 == 2:
                if skel[i + 1, j + 1] == 255:
                    ends.append((j + 1, i + 1))
            if sum / 255 == 4:
                forks.append((j, i))

    plt.figure()
    plt.imshow(skel, 'gray')
    plt.plot(*zip(*ends), 'r.')
    plt.plot(*zip(*forks), 'g1')
plt.show()
