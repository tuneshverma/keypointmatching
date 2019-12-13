import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


def keypoint_match(path_of_image1, path_of_image2):

    img1 = cv.imread(path_of_image1, cv.IMREAD_GRAYSCALE)
    img2 = cv.imread(path_of_image2, cv.IMREAD_GRAYSCALE)

    # Initiate SIFT detector
    sift = cv.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # Need to draw only good matches, so create a mask
    matchesMask = [[0, 0] for i in range(len(matches))]
    # ratio test as per Lowe's paper
    c = 0
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:
            matchesMask[i] = [1, 0]
            c = c + 1
    # print(matchesMask)
    print(c)
    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=(255, 0, 0),
                       matchesMask=matchesMask,
                       flags=cv.DrawMatchesFlags_DEFAULT)
    img3 = cv.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)
    plt.imshow(img3, ), plt.show()


keypoint_match('D:/MintM/deo_classifier/data/fogg/40001571_2-fogg-fragrance-body-spray-paradise_.jpg',
               'D:/MintM/deo_classifier/data/fogg/40001570_2-fogg-fragrance-body-spray-royal.jpg')
