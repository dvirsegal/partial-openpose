import os

import cv2

import PartialSkeleton
import video_utils
from estimator import TfPoseEstimator
from networks import get_graph_path

class CoordinateStore:
    def __init__(self):
        self.points = []

    def select_point(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDBLCLK:
            cv2.circle(img, (x, y), 3, (255, 0, 0), -1)
            self.points.append((x, y))


if __name__ == '__main__':
    input_folder = "./videos/walking/"
    results_folder = "./images/results/"
    input_video = "./videos/walking.mp4"
    output_folder = "./videos"
    print("Start splitting video")
    video_utils.split_video(input_video, input_folder)
    print("Video was splited")
    print("Loading all images")
    images = video_utils.load_images_from_folder(input_folder)
    print("Loaded {} images from {} folder".format(images.__len__(), input_folder))
    first_image = images[0]
    # instantiate class
    coordinateStore1 = CoordinateStore()

    # Bind the function to window
    img = images[0]
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', coordinateStore1.select_point)

    while 1:
        cv2.imshow('image', first_image)
        k = cv2.waitKey(20) & 0xFF
        if k == 27:  # ESC
            break
    cv2.destroyAllWindows()

    print("Selected Coordinates: ")
    for i in coordinateStore1.points:
        print(i)

    hip = coordinateStore1.points

    w = 432
    h = 368
    estimator = TfPoseEstimator(get_graph_path('mobilenet_thin'), target_size=(w, h))
    count = 0
    for img in images:
        PartialSkeleton.skeletonize(estimator,img, hip, count)
        count += 1

    print("Creating output video...")
    video_utils.create_video(input_video, results_folder, output_folder)
    print("Video was created.")
