import argparse
import ast

import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

import common
from estimator import TfPoseEstimator
from networks import get_graph_path, model_wh


def create_affined_image(image, pts_src, pts_dst):
    rows, cols, ch = image.shape
    M = cv2.getAffineTransform(pts_src, pts_dst)
    return cv2.warpAffine(image, M, (cols, rows))


def compare_images(imageA, imageB, rmseX, rmseY, totalRMSE, referenceValue, title):
    # setup the figure
    fig = plt.figure(title)
    plt.suptitle("RMSE X: %.2f, RMSE Y: %.2f, TOTAL RMSE: %.2f \n\n Reference Value: %.2f" % (rmseX, rmseY, totalRMSE,
                                                                                              referenceValue))

    # show first image
    fig.add_subplot(1, 2, 1)
    plt.imshow(imageA)
    plt.axis("off")

    # show the second image
    fig.add_subplot(1, 2, 2)
    plt.imshow(imageB)
    plt.axis("off")

    # show the images
    plt.show()


def skeletonize(given_image, hip, image_name, model='mobilenet_thin'):
    if not os.path.exists(".\\images\\results"):
        os.makedirs(".\\images\\results")

    w = 432
    h = 368
    scales = None
    estimator = TfPoseEstimator(get_graph_path(model), target_size=(w, h))

    # Load 2 images
    dummy_image = common.read_imgfile('./images/full_body1.png', None, None)

    # Get dummy image skeleton
    dummy_image_parts = estimator.inference(dummy_image, scales=scales)

    # Display the dummy image's skeleton
    # image = TfPoseEstimator.draw_humans(dummy_image, dummy_image_parts, imgcopy=True)
    # cv2.imshow('dummy image result', image)
    # cv2.waitKey()

    # Merge two images
    pts1 = np.float32(
        [[int(dummy_image_parts[0].body_parts[8].x * h), int(dummy_image_parts[0].body_parts[8].y * w)],
         [int(dummy_image_parts[0].body_parts[11].x * h), int(dummy_image_parts[0].body_parts[11].y * w)],
         [int(dummy_image_parts[0].body_parts[17].x * h), int(dummy_image_parts[0].body_parts[17].y * w)]])
    pts2 = np.float32([hip[0],
                       hip[1],
                       hip[2]])
    affined_dummy_image = create_affined_image(dummy_image, pts1, pts2)
    affined_dummy_image = cv2.flip(affined_dummy_image, 0)
    # cv2.imshow("affined", affined_dummy_image)
    # cv2.waitKey()
    # Get each image skeleton
    dummy_image_parts = estimator.inference(affined_dummy_image, scales=scales)
    # image = TfPoseEstimator.draw_humans(affined_dummy_image, dummy_image_parts, imgcopy=True)
    # cv2.imshow('dummy person result', image)
    # cv2.waitKey()
    # Hip coordinates
    firstPersonHipX = dummy_image_parts[0].body_parts[11].x
    # secondPersonHipX = hip[0][0]
    # Find the max between hip X's
    # maxV = max(firstPersonHipX, secondPersonHipX)
    # Combine the two images
    hipX = int(firstPersonHipX * h)

    # create_merged_image(first_image, second_image, h, w, hip)
    merged_image = np.zeros((h * 2, w, 3), np.uint8)
    merged_image[0:hipX, :] = affined_dummy_image[0:hipX, :]
    merged_image[hipX:hipX + h, :] = given_image[:, :]
    # cv2.imshow('Merged Image', merged_image)
    # cv2.waitKey()

    # Find the merge image's skeleton
    merged_image_parts = estimator.inference(merged_image, scales=scales)
    merged_image_skeleton = TfPoseEstimator.draw_humans(merged_image, merged_image_parts, imgcopy=False)
    # cv2.imshow('merged person result', merged_image_skeleton)
    # cv2.waitKey()

    # Take only legs and show them
    legs_image = np.zeros((h, w, 3), np.uint8)
    legs_image[:] = 255
    # hipMergedX = int(merged_image_parts[0].body_parts[11].x *(h*2) + 0.5)
    # delta= int(np.abs(h*2 - hipMergedX))
    legs_image[:, :] = merged_image_skeleton[245: 245 + h, :]
    # cv2.imshow('Legs', legs_image)
    # cv2.waitKey()
    cv2.destroyAllWindows()
    cv2.imwrite(".\\images\\results\\{}.png".format(image_name), legs_image)

    # xSource = []
    # ySource = []
    # xDest = []
    # yDest = []
    # for i in [8, 9, 10, 11, 12, 13]:
    #     x1 = merged_image_parts[0].body_parts[i].x * h
    #     xDest.append(x1)
    #     x2 = second_image_parts[0].body_parts[i].x * h
    #     xSource.append(x2)
    #     y1 = merged_image_parts[0].body_parts[i].y * w
    #     yDest.append(y1)
    #     y2 = second_image_parts[0].body_parts[i].y * w
    #     ySource.append(y2)
    #
    # mseX = ((np.array(xSource) - np.array(xDest)) ** 2).mean()
    # mseY = ((np.array(ySource) - np.array(yDest)) ** 2).mean()
    # rmseX = np.sqrt(mseX)
    # rmseY = np.sqrt(mseY)
    # totalRMSE = np.sqrt(1 / xSource.__len__() * (mseX + mseX))
    # LKneeX = second_image_parts[0].body_parts[12].x * h
    # LKneeY = second_image_parts[0].body_parts[12].y * w
    # LAnkleX = second_image_parts[0].body_parts[13].x * h
    # LAnkleY = second_image_parts[0].body_parts[13].y * w
    # referenceValue = np.sqrt((LKneeX - LAnkleX) ** 2 + (LKneeY - LAnkleY) ** 2)
    #
    # # Display the two images for comparision
    # compare_images(legs_image, second_image_skeleton, rmseX, rmseY, totalRMSE, referenceValue, "Legs VS Original")

    # return [[int(merged_image_parts[0].body_parts[8].x * h), int(merged_image_parts[0].body_parts[8].y * w)],
    #         [int(merged_image_parts[0].body_parts[11].x * h), int(merged_image_parts[0].body_parts[11].y * w)],
    #         [int(merged_image_parts[0].body_parts[17].x * h), int(merged_image_parts[0].body_parts[17].y * w)]]
    return [hip[0], hip[1], hip[2]]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='partial pose run')
    parser.add_argument('--image1', type=str, default='./images/p1.jpg')
    parser.add_argument('--image2', type=str, default='./images/p1.jpg')
    parser.add_argument('--resolution', type=str, default='432x368', help='network input resolution. default=432x368')
    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin')
    parser.add_argument('--scales', type=str, default='[None]', help='for multiple scales, eg. [1.0, (1.1, 0.05)]')
    args = parser.parse_args()
    scales = ast.literal_eval(args.scales)

    w, h = model_wh(args.resolution)
    estimator = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))

    # Load 2 images
    first_image = common.read_imgfile(args.image1, None, None)
    second_image = common.read_imgfile(args.image2, None, None)

    # Get each image skeleton
    first_image_parts = estimator.inference(first_image, scales=scales)
    second_image_parts = estimator.inference(second_image, scales=scales)

    # Display the two skeleton on images
    image = TfPoseEstimator.draw_humans(first_image, first_image_parts, imgcopy=True)
    cv2.imshow('first person result', image)
    cv2.waitKey()
    image = second_image_skeleton = TfPoseEstimator.draw_humans(second_image, second_image_parts, imgcopy=True)
    cv2.imshow('second person result', image)
    cv2.waitKey()

    # Merge two images
    merged_image = np.zeros((h, w, 3), np.uint8)

    pts1 = np.float32([[first_image_parts[0].body_parts[8].x * h, first_image_parts[0].body_parts[8].y * w],
                       [first_image_parts[0].body_parts[11].x * h, first_image_parts[0].body_parts[11].y * w],
                       [first_image_parts[0].body_parts[17].x * h, first_image_parts[0].body_parts[17].y * w]])
    pts2 = np.float32([[second_image_parts[0].body_parts[8].x * h, second_image_parts[0].body_parts[8].y * w],
                       [second_image_parts[0].body_parts[11].x * h, second_image_parts[0].body_parts[11].y * w],
                       [second_image_parts[0].body_parts[17].x * h, second_image_parts[0].body_parts[17].y * w]])

    dst = create_affined_image(first_image, pts1, pts2)
    # Get each image skeleton
    first_image_parts = estimator.inference(dst, scales=scales)
    # Hip coordinates
    firstPersonHipX = first_image_parts[0].body_parts[8].x
    secondPersonHipX = second_image_parts[0].body_parts[8].x
    # Find the max between hip X's
    maxV = max(firstPersonHipX, secondPersonHipX)
    # Combine the two images
    hip = int(maxV * h)

    # create_merged_image(first_image, second_image, h, w, hip)
    merged_image[0:hip, :] = dst[0:hip, :]
    merged_image[hip:h, :] = second_image[hip:h, :]
    cv2.imshow('Merged Image', merged_image)
    cv2.waitKey()

    # Find the merge image's skeleton
    merged_image_parts = estimator.inference(merged_image, scales=scales)
    merged_image_skeleton = TfPoseEstimator.draw_humans(merged_image, merged_image_parts, imgcopy=False)
    cv2.imshow('merged person result', merged_image_skeleton)
    cv2.waitKey()

    # Take only legs and show them
    legs_image = np.zeros((h, w, 3), np.uint8)
    legs_image[:] = 255
    legs_image[hip:h, :] = merged_image_skeleton[hip:h, :]
    cv2.imshow('Legs', legs_image)
    cv2.waitKey()

    xSource = []
    ySource = []
    xDest = []
    yDest = []
    for i in [8, 9, 10, 11, 12, 13]:
        x1 = merged_image_parts[0].body_parts[i].x * h
        xDest.append(x1)
        x2 = second_image_parts[0].body_parts[i].x * h
        xSource.append(x2)
        y1 = merged_image_parts[0].body_parts[i].y * w
        yDest.append(y1)
        y2 = second_image_parts[0].body_parts[i].y * w
        ySource.append(y2)

    mseX = ((np.array(xSource) - np.array(xDest)) ** 2).mean()
    mseY = ((np.array(ySource) - np.array(yDest)) ** 2).mean()
    rmseX = np.sqrt(mseX)
    rmseY = np.sqrt(mseY)
    totalRMSE = np.sqrt(1 / xSource.__len__() * (mseX + mseX))
    LKneeX = second_image_parts[0].body_parts[12].x * h
    LKneeY = second_image_parts[0].body_parts[12].y * w
    LAnkleX = second_image_parts[0].body_parts[13].x * h
    LAnkleY = second_image_parts[0].body_parts[13].y * w
    referenceValue = np.sqrt((LKneeX - LAnkleX) ** 2 + (LKneeY - LAnkleY) ** 2)

    # Display the two images for comparision
    compare_images(legs_image, second_image_skeleton, rmseX, rmseY, totalRMSE, referenceValue, "Legs VS Original")
