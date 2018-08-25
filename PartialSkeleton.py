import argparse
import ast
import cv2
import numpy as np
import matplotlib.pyplot as plt

import common
from estimator import TfPoseEstimator
from networks import get_graph_path, model_wh


def compare_images(imageA, imageB, rmseX, rmseY, totalRMSE, title):
    # compute the mean squared error

    # setup the figure
    fig = plt.figure(title)
    plt.suptitle("RMSE X: %.2f, RMSE Y: %.2f, TOTAL RMSE: %.2f" % (rmseX, rmseY, totalRMSE))

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


# def compare_images(img1, img2):
#     h1, w1 = img1.shape[:2]
#     h2, w2 = img2.shape[:2]
#     # create empty matrix
#     vis = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)
#     # combine 2 images
#     vis[:h1, :w1, :3] = img1
#     vis[:h2, w1:w1 + w2, :3] = img2
#     cv2.imshow("Compare", vis)
#     cv2.waitKey()


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
    # image = TfPoseEstimator.draw_humans(first_image, first_image_parts, imgcopy=False)
    # cv2.imshow('first person result', image)
    # cv2.waitKey()
    second_image_skeleton = TfPoseEstimator.draw_humans(second_image, second_image_parts, imgcopy=True)
    # cv2.imshow('second person result', image)
    # cv2.waitKey()

    # Merge two images
    # TODO: build smart copy of first image into second image
    merged_image = np.zeros((h, w, 3), np.uint8)
    firstPersonHipX = first_image_parts[0].body_parts[8].x
    secondPersonHipX = second_image_parts[0].body_parts[8].x
    # Find the max between hip X's
    maxV = max(firstPersonHipX, secondPersonHipX)
    # Combine the two images
    hip = int(maxV * h)
    merged_image[0:hip, :] = first_image[0:hip, :]
    merged_image[hip:h, :] = second_image[hip:h, :]
    cv2.imshow('Merged Image', merged_image)
    cv2.waitKey()

    # Find the merge image's skeleton
    merged_image_parts = estimator.inference(merged_image, scales=scales)
    merged_image_skeleton = TfPoseEstimator.draw_humans(merged_image, merged_image_parts, imgcopy=False)

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
        # dist.append(np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))

    mseX = ((np.array(xSource) - np.array(xDest)) ** 2).mean()
    mseY = ((np.array(ySource) - np.array(yDest)) ** 2).mean()
    rmseX = np.sqrt(mseX)
    rmseY = np.sqrt(mseY)
    totalRMSE = np.sqrt(1 / xSource.__len__() * (mseX + mseX))

    # Display the two images for comparision
    compare_images(legs_image, second_image_skeleton, rmseX, rmseY, totalRMSE, "Legs VS Original")
