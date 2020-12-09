import numpy as np
import cv2
import glob
from matplotlib import pyplot as plt
from os import listdir
from os.path import isfile, join

def remove_background(image):
    """
    :param image: image from which will remove the background
    """

    # Load the Image
    imgo = cv2.imread(image)
    height, width = imgo.shape[:2]
    im_name = image[:-4]
    # Create a mask holder
    mask = np.zeros(imgo.shape[:2], np.uint8)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=2, sigmaY=2, borderType=cv2.BORDER_DEFAULT)

    # Grab Cut the object
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    # Hard Coding the Rect The object must lie within this rect.
    rect = (10, 10, width - 30, height - 30)
    cv2.grabCut(imgo, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    img1 = imgo * mask[:, :, np.newaxis]

    # Get the background
    background = imgo - img1

    # Change all pixels in the background that are not black to white
    background[np.where((background > [0, 0, 0]).all(axis=2))] = [255, 255, 255]

    # Add the background and the image
    final = background + img1

    # To be done - Smoothening the edges
    cv2.imwrite(f'{im_name}_not_bg.png', final)


def get_filenames(folder, is_bg=False):
    """
    :param is_bg: images with background or not
    :param folder: selected name folder of which  loading a files
    :return: array with names files of which  selected a folder
    """

    if is_bg:
        filenames = [img for img in glob.glob(f'data/{folder}/*_not_bg.png')]
    else:
        filenames = [img for img in glob.glob(f'data/{folder}/*.png')]

    filenames.sort()

    images = []
    for img in filenames:
        images.append(img)

    return images

def binarize_image(iamge_src):

    # https://docs.opencv.org/master/d9/d61/tutorial_py_morphological_ops.html

    image = cv2.imread(iamge_src, 0)
    im_name = iamge_src[:-4]
    img = cv2.medianBlur(image, 5)
    binarized_image = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                               cv2.THRESH_BINARY, 13, 3)

    cv2.imwrite(f'{im_name}_binarized.png', binarized_image)

    # kernel = np.ones((5, 5), np.uint8)
    # gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)

    # titles = ['Image without background', 'Image binarized', 'gradient']
    # images = [img, th3]
    #
    #
    # for i in range():
    #     plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
    #     plt.title(titles[i])
    #     plt.xticks([]),plt.yticks([])
    # plt.show()


# Descriptors
def area_desc(img):
    ret, thresh = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # return np.count_nonzero(img == 255)
    area = cv2.contourArea(contours[0])
    return area


def perimeter_desc(img):
    # https: // opencv - python - tutroals.readthedocs.io / en / latest / py_tutorials / py_imgproc / py_canny / py_canny.html  #
    edges = cv2.Canny(img, 10, 200)

    plt.subplot(121), plt.imshow(img, cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(edges, cmap='gray')
    plt.title('Perimeter'), plt.xticks([]), plt.yticks([])

    plt.show()
    cv2.waitKey(0)

    return np.count_nonzero(edges == 255)


def roundness_desc(img):
    area = np.count_nonzero(img == 255)
    edged = cv2.Canny(img, 30, 200)
    perimeter = np.count_nonzero(edged == 255)
    return np.power(perimeter, 2) / (4 * np.pi * area)


def compactness_desc(img):
    area = np.count_nonzero(img == 0)
    edged = cv2.Canny(img, 30, 200)
    perimeter = np.count_nonzero(edged == 255)
    return np.power(perimeter, 2) / area

def diameter_desc(img):
    ret, thresh = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    (x, y), radius = cv2.minEnclosingCircle(contours[0])
    return radius * 2


# https://www.learnopencv.com/convex-hull-using-opencv-in-python-and-c/
def convex_hull_desc(img):
    ret, thresh = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    hull = []
    # calculate points for each contour
    for i in range(len(contours)):
        hull.append(cv2.convexHull(contours[i], False))
    drawing = np.zeros((thresh.shape[0], thresh.shape[1], 3), np.uint8)
    for i in range(len(contours)):
        color_contours = (0, 255, 0)
        color = (255, 0, 0)
        # draw ith contour
        cv2.drawContours(drawing, contours, i, color_contours, 1, 8, hierarchy)
        # draw ith convex hull object
        cv2.drawContours(drawing, hull, i, color, 1, 8)
    # plt.imshow(drawing, cmap=plt.cm.gray)
    # plt.show()
    edged = cv2.Canny(drawing, 30, 200)
    return np.count_nonzero(edged == 255)


def example_of_descs(img):
    print("------Wartości deskryptorów------")
    print("Pole powierzchni: ", area_desc(img))
    print("Obwód: ", perimeter_desc(img))
    print("Kołowość: ", roundness_desc(img))
    print("Zwartość: ", compactness_desc(img))
    print("Mimośród: ", diameter_desc(img))
    print("Obwód powłoki wypukłej: ", convex_hull_desc(img))
    print("------Wartości deskryptorów------")


def fit(train):
    train_learned = []
    area = []
    per = []
    round = []
    compact = []
    convex = []
    diameter = []
    for i, item in enumerate(train):
        area.append(area_desc(item))
        per.append(perimeter_desc(item))
        round.append(roundness_desc(item))
        compact.append(compactness_desc(item))
        convex.append(convex_hull_desc(item))
        diameter.append(diameter_desc(item))
    train_learned.append(area)
    train_learned.append(per)
    train_learned.append(round)
    train_learned.append(compact)
    train_learned.append(convex)
    train_learned.append(diameter)
    return train_learned

def enc_idx(i):
    n = ["sign_1", "sign_2", "sign_3", "sign_4", "sign_5", "sign_6", "sign_7", "sign_8", "sign_9", "sign_10"]
    return n[i]


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def classify(img, train, name):
    tmp = [area_desc(img), perimeter_desc(img), roundness_desc(img), compactness_desc(img), convex_hull_desc(img), diameter_desc(img)]
    res = [enc_idx(find_nearest(train[0], tmp[0])), enc_idx(find_nearest(train[1], tmp[1])),
          enc_idx(find_nearest(train[2], tmp[2])), enc_idx(find_nearest(train[3], tmp[3])),
          enc_idx(find_nearest(train[4], tmp[4])), enc_idx(find_nearest(train[5], tmp[5]))]
    out = []
    # print("Deskryptor - pole powierzchni: ", res[0])
    # print("Deskryptor - obwód: ", res[1])
    # print("Deskryptor - kołowość: ", res[2])
    # print("Deskryptor - zwartość: ", res[3])
    # print("Deskryptor - obwód powłoki wypukłej: ", res[4])
    # print("Mimośród: ", res[5])

    out.append(1) if res[0] == name else out.append(0)
    out.append(1) if res[1] == name else out.append(0)
    out.append(1) if res[2] == name else out.append(0)
    out.append(1) if res[3] == name else out.append(0)
    out.append(1) if res[4] == name else out.append(0)
    out.append(1) if res[5] == name else out.append(0)
    return out


def score(scores):
    # print(scores)
    area = 0
    peri = 0
    roun = 0
    comp = 0
    conv = 0
    diame = 0
    for i in range(len(scores)):
        area += scores[i][0]
        peri += scores[i][1]
        roun += scores[i][2]
        comp += scores[i][3]
        conv += scores[i][4]
        diame += scores[i][5]

    area = area / len(scores) * 100
    peri = peri / len(scores) * 100
    roun = roun / len(scores) * 100
    comp = comp / len(scores) * 100
    conv = conv / len(scores) * 100
    diame = diame / len(scores) * 100
    return [area, peri, roun, comp, conv, diame]


if __name__ == "__main__":

    train = []
    results = []
    subDirs = ["sign_1", "sign_2", "sign_3", "sign_4", "sign_5", "sign_6", "sign_7", "sign_8", "sign_9", "sign_10"]
    binPath = './TEST_DATA'
    for dir in subDirs:
        pathToBinDir = f'{binPath}/{dir}'
        files = [f for f in listdir(pathToBinDir) if isfile(join(pathToBinDir, f))]
        for idx, file in enumerate(files):
            pathToBinFile = f'{pathToBinDir}/{file}'
            img = cv2.imread(pathToBinFile, 0)
            if idx == 0:
                train.append(img)
                break

    clf = fit(train)
    # print(clf)
    # classify(img_test, clf)

    for dir in subDirs:
        pathToBinDir = f'{binPath}/{dir}'
        files = [f for f in listdir(pathToBinDir) if isfile(join(pathToBinDir, f))]
        for idx, file in enumerate(files):
            pathToBinFile = f'{pathToBinDir}/{file}'
            img = cv2.imread(pathToBinFile, 0)
            if idx != 0:
                results.append(classify(img, clf, dir))

    print(f' results : {results}')
    scores = score(results)
    print("Pole powierzchni: ", scores[0], "%")
    print("Długość obwodu: ", scores[1], "%")
    print("Kołowość: ", scores[2], "%")
    print("Zwartość: ", scores[3], "%")
    # print("Poprawność - obwód powłoki wypukłej: ", scores[4], "%")
    print("Mimośród: ", scores[5], "%")

    #plt.imshow(img, cmap=plt.cm.gray)
    #plt.show()
    #print(img)
    #example_of_descs(img)