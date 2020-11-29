import numpy as np
import cv2
import glob
from matplotlib import pyplot as plt

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


if __name__ == "__main__":

    # images = get_filenames('sign_10')
    # print(images)
    # for image in images:
    #     remove_background(image)
    #     print(f'FINISHED : {image} ')

    images = get_filenames('sign_10', True)
    print(f'{images}')
    for image in images:
        binarize_image(image)
        print(f'FINISHED : {image} ')
