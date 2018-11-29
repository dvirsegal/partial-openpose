import argparse
import ast

import cv2
import numpy as np

import common
from PartialSkeleton import create_affined_image, compare_images
from estimator import TfPoseEstimator
from networks import get_graph_path, model_wh


def calculate_rmse(merged_image_parts, second_image_parts):
    """
    Calculate RMSE between two skeletons
    :param merged_image_parts:
    :param second_image_parts:
    :return:
    """
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
    return rmseX, rmseY, totalRMSE


if __name__ == '__main__':
    # This main purpose is to demonstrate the accuracy of partial OpenPose method
    # Initialize TF Pose Estimator - based on given args
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

    # "Wisely" Merge the two images (using affine transform)
    merged_image = np.zeros((h, w, 3), np.uint8)

    pts1 = np.float32([[first_image_parts[0].body_parts[8].x * h, first_image_parts[0].body_parts[8].y * w],
                       [first_image_parts[0].body_parts[11].x * h, first_image_parts[0].body_parts[11].y * w],
                       [first_image_parts[0].body_parts[17].x * h, first_image_parts[0].body_parts[17].y * w]])
    pts2 = np.float32([[second_image_parts[0].body_parts[8].x * h, second_image_parts[0].body_parts[8].y * w],
                       [second_image_parts[0].body_parts[11].x * h, second_image_parts[0].body_parts[11].y * w],
                       [second_image_parts[0].body_parts[17].x * h, second_image_parts[0].body_parts[17].y * w]])

    dst = create_affined_image(first_image, pts1, pts2)
    # Get dst image skeleton
    first_image_parts = estimator.inference(dst, scales=scales)
    # Hip coordinates
    firstPersonHipX = first_image_parts[0].body_parts[8].x
    secondPersonHipX = second_image_parts[0].body_parts[8].x
    # Find the max between hip X's
    maxV = max(firstPersonHipX, secondPersonHipX)
    # Combine the two images
    hip = int(maxV * h)

    # Merge the two images until the hip coordinate
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

    # Calculate Root MSE score between original and merged skeletons
    rmseX, rmseY, totalRMSE = calculate_rmse(merged_image_parts, second_image_parts)
    LKneeX = second_image_parts[0].body_parts[12].x * h
    LKneeY = second_image_parts[0].body_parts[12].y * w
    LAnkleX = second_image_parts[0].body_parts[13].x * h
    LAnkleY = second_image_parts[0].body_parts[13].y * w

    # Calculate knee -- ankle distance for reference
    referenceValue = np.sqrt((LKneeX - LAnkleX) ** 2 + (LKneeY - LAnkleY) ** 2)

    # Display the two images for comparision
    compare_images(legs_image, second_image_skeleton, rmseX, rmseY, totalRMSE, referenceValue, "Legs VS Original")
