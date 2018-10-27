import cv2
import pickle
import imutils
import numpy as np
from imutils.object_detection import non_max_suppression

import video_utils
from tensorflow_human_detection import DetectorAPI


class ShapeDetector:
    def __init__(self):
        pass

    def detect(self, c):
        # initialize the shape name and approximate the contour
        shape = "unidentified"
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)

        # if the shape is a triangle, it will have 3 vertices
        if len(approx) == 3:
            shape = "triangle"

        # if the shape has 4 vertices, it is either a square or
        # a rectangle
        elif len(approx) == 4:
            # compute the bounding box of the contour and use the
            # bounding box to compute the aspect ratio
            (x, y, w, h) = cv2.boundingRect(approx)
            ar = w / float(h)

            # a square will have an aspect ratio that is approximately
            # equal to one, otherwise, the shape is a rectangle
            shape = "square" if ar >= 0.95 and ar <= 1.05 else "rectangle"

        # if the shape is a pentagon, it will have 5 vertices
        elif len(approx) == 5:
            shape = "pentagon"

        # otherwise, we assume the shape is a circle
        else:
            shape = "circle"

        # return the name of the shape
        return shape


def detect_contour_corner(img):
    # img = cv2.imread(image_path)

    ##(2) convert to hsv-space, then split the channels
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    ##(3) threshold the S channel using adaptive method(`THRESH_OTSU`) or fixed thresh
    th, threshed = cv2.threshold(s, 50, 255, cv2.THRESH_BINARY_INV)

    ##(4) find all the external contours on the threshed S
    _, cnts, _ = cv2.findContours(threshed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    canvas = img.copy()
    # cv2.drawContours(canvas, cnts, -1, (0,255,0), 1)

    ## sort and choose the largest contour
    cnts = sorted(cnts, key=cv2.contourArea)
    cnt = cnts[-1]

    ## approx the contour, so the get the corner points
    arclen = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02 * arclen, True)
    cv2.drawContours(canvas, [cnt], -1, (255, 0, 0), 1, cv2.LINE_AA)
    cv2.drawContours(canvas, [approx], -1, (0, 0, 255), 1, cv2.LINE_AA)

    ## Ok, you can see the result as tag(6)
    cv2.imshow("Detect Contour Corners", canvas)
    cv2.waitKey(0)


def detect_haar(img):
    # img = cv2.imread(image_path, 0)

    lowerBody_cascade = cv2.CascadeClassifier(
        'D:\\Continuum\\Anaconda3\\Lib\\site-packages\\cv2\\data\\haarcascade_fullbody.xml')
    arr_lower_body = lowerBody_cascade.detectMultiScale(img)
    if arr_lower_body != ():
        for (x, y, w, h) in arr_lower_body:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), 2)
        print('body found')

    cv2.imshow('Detect using HAAR', img)
    cv2.waitKey(0)


def detect_shape(image):
    # load the image and resize it to a smaller factor so that
    # the shapes can be approximated better
    # image = cv2.imread(image_path)
    resized = imutils.resize(image, width=300)
    ratio = image.shape[0] / float(resized.shape[0])

    # convert the resized image to grayscale, blur it slightly,
    # and threshold it
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY_INV)[1]

    # find contours in the thresholded image and initialize the
    # shape detector
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    sd = ShapeDetector()

    # loop over the contours
    for c in cnts:
        # compute the center of the contour, then detect the name of the
        # shape using only the contour
        M = cv2.moments(c)
        cX = int((M["m10"] / M["m00"]) * ratio)
        cY = int((M["m01"] / M["m00"]) * ratio)
        shape = sd.detect(c)

        # multiply the contour (x, y)-coordinates by the resize ratio,
        # then draw the contours and the name of the shape on the image
        c = c.astype("float")
        c *= ratio
        c = c.astype("int")
        cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
        cv2.putText(image, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 2)

        # show the output image
        cv2.imshow("Detect Shape", image)
        cv2.waitKey(0)


def find_people(img):
    '''
        Detect people in image
        :param img: numpy.ndarray
        :return: count of rectangles after non-maxima suppression, corresponding to number of people detected in picture
        '''

    BOX_COLOR = (0, 255, 0)  # Green
    CALIBRATION_MODE_1 = (400, (3, 3), (32, 32), 1.01, 0.999)  # People very small size and close together in image
    CALIBRATION_MODE_2 = (400, (3, 3), (32, 32), 1.01, 0.8)  # People very small size
    CALIBRATION_MODE_3 = (400, (4, 4), (32, 32), 1.015, 0.999)  # People small size and close together
    CALIBRATION_MODE_4 = (400, (4, 4), (32, 32), 1.015, 0.8)  # People small size
    CALIBRATION_MODE_5 = (400, (4, 4), (32, 32), 1.02, 0.999)  # People medium size and close together
    CALIBRATION_MODE_6 = (400, (4, 4), (32, 32), 1.02, 0.8)  # People medium size
    CALIBRATION_MODE_7 = (400, (8, 8), (32, 32), 1.03, 0.999)  # People large size and close together
    CALIBRATION_MODE_8 = (400, (8, 8), (32, 32), 1.03, 0.8)  # People large size
    CALIBRATION_MODES = (CALIBRATION_MODE_1, CALIBRATION_MODE_2, CALIBRATION_MODE_3, CALIBRATION_MODE_4,
                         CALIBRATION_MODE_5, CALIBRATION_MODE_6, CALIBRATION_MODE_7, CALIBRATION_MODE_8)

    MIN_IMAGE_WIDTH, WIN_STRIDE, PADDING, SCALE, OVERLAP_THRESHOLD = CALIBRATION_MODE_2
    # img = cv2.imread(image_path)
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    # Chooses whichever size is less
    image = imutils.resize(img, width=min(300, img.shape[1]))
    # detect people in the image
    (rects, wghts) = hog.detectMultiScale(image, winStride=WIN_STRIDE,
                                          padding=PADDING, scale=SCALE)
    # apply non-maxima suppression to the bounding boxes using a
    # fairly large overlap threshold to try to maintain overlapping boxes that are still people
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    pick = non_max_suppression(rects, probs=None, overlapThresh=OVERLAP_THRESHOLD)

    # draw the final bounding boxes
    for (xA, yA, xB, yB) in pick:
        # Tighten the rectangle around each person by a small margin
        cv2.rectangle(image, (xA + 5, yA + 5), (xB - 5, yB - 10), BOX_COLOR, 2)

    cv2.imshow("Find People detection", image)
    cv2.waitKey()

    return len(pick)


def find_extreme_points(image):
    # load the image, convert it to grayscale, and blur it slightly
    # image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # threshold the image, then perform a series of erosions +
    # dilations to remove any small regions of noise
    thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)[1]
    thresh = cv2.erode(thresh, None, iterations=3)
    thresh = cv2.dilate(thresh, None, iterations=3)

    # find contours in thresholded image, then grab the largest
    # one
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    c = max(cnts, key=cv2.contourArea)

    # determine the most extreme points along the contour
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])
    a = [extBot, extLeft, extRight, extTop]
    l = set(a)
    # draw the outline of the object, then draw each of the
    # extreme points, where the left-most is red, right-most
    # is green, top-most is blue, and bottom-most is teal
    cv2.drawContours(image, [c], -1, (0, 255, 255), 2)
    cv2.circle(image, extLeft, 8, (0, 0, 255), -1)
    cv2.circle(image, extRight, 8, (0, 255, 0), -1)
    cv2.circle(image, extTop, 8, (255, 0, 0), -1)
    cv2.circle(image, extBot, 8, (255, 255, 0), -1)

    # show the output image
    cv2.imshow("Find Extreme Points", image)
    cv2.waitKey(0)
    return np.float32(list(l))
    # return np.float32([])


def detect_using_tf(img):
    model_path = 'faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb'
    odapi = DetectorAPI(path_to_ckpt=model_path)
    threshold = 0.7

    boxes, scores, classes, num = odapi.processFrame(img)

    # Visualization of the results of a detection.
    pts2 = np.float32([])
    for i in range(len(boxes)):
        # Class 1 represents human
        if classes[i] == 1 and scores[i] > threshold:
            box = boxes[i]
            cv2.rectangle(img, (box[1], box[0]), (box[3], box[2]), (255, 0, 0), 2)
            pts2 = np.float32([[box[1], box[0]],
                               [box[1], box[2]],
                               [box[3], box[2]]])

    # cv2.imshow("preview", img)
    # cv2.waitKey()
    return pts2


if __name__ == '__main__':
    bottom_images = video_utils.load_images_from_folder("./images/bottom/",True)
    data = []
    for img in bottom_images:
        for scale_factor in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            height_b, width_b, channels = img[0].shape
            scaled_bottom = cv2.resize(img[0], (int(width_b * scale_factor), int(height_b * scale_factor)), fx=scale_factor,
                                       fy=scale_factor, interpolation=cv2.INTER_AREA)
            pts = detect_using_tf(scaled_bottom)
            data.append([img[1],scale_factor, pts])
    with open('human_points.pickle', 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)