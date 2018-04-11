import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def skindetection(img):
    # define the upper and lower boundaries of the HSV pixel
    # intensities to be considered 'skin'
    lower = np.array([0, 48, 80], dtype = "uint8")
    upper = np.array([20, 255, 255], dtype = "uint8")

    # resize the frame, convert it to the HSV color space,
    # and determine the HSV pixel intensities that fall into
    # the speicifed upper and lower boundaries
    converted = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    skinMask = cv2.inRange(converted, lower, upper)

    # apply a series of erosions and dilations to the mask
    # using an elliptical kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    skinMask = cv2.erode(skinMask, kernel, iterations = 2)
    skinMask = cv2.dilate(skinMask, kernel, iterations = 2)

    # blur the mask to help remove noise, then apply the
    # mask to the frame
    skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
    skin = cv2.bitwise_and(img, img, mask = skinMask)

    return(skin)


def kmeans(img, K):
    #Reshape img to list of pixels
    Z = img.reshape((-1, 3))
    # convert to np.float32
    Z = np.float32(Z)

    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))

    return(res2)

PATH = 'C:/Users/titou/Desktop/Centrale/Option OSY/Deep Learning/data/train2017/train2017'
PATH_exp = 'C:/Users/titou/Desktop/Centrale/Option OSY/Deep Learning/data/train2017/'


files = os.listdir(PATH)
for file in files:
    print(len(files),files.index(file))
    #Opening image
    img = cv2.imread(os.path.join(PATH,file), 1)

    #Canny filter
    edge = cv2.Canny(img, 0.66 * cv2.mean(img)[0], 1.33 * cv2.mean(img)[0])

    #Skin detection
    skin = skindetection(img)

    #Color clustering with K-means, K = 8
    cluster = kmeans(img, 8)

    #Export results
    cv2.imwrite(os.path.join(PATH_exp,'edge/')+ file[:-4]+"_edge.jpg",edge)
    cv2.imwrite(os.path.join(PATH_exp, 'skin/')+ file[:-4] + "_skin.jpg", skin)
    cv2.imwrite(os.path.join(PATH_exp, 'cluster/')+ file[:-4] + "_cluster.jpg", cluster)


