import cv2
import os

import video_utils
import PartialSkeleton


class CoordinateStore:
    def __init__(self):
        self.points = []

    def select_point(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDBLCLK:
            cv2.circle(img, (x, y), 3, (255, 0, 0), -1)
            self.points.append((x, y))


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        image = cv2.imread(os.path.join(folder, filename))
        if image is not None:
            images.append(image)
    return images


if __name__ == '__main__':
    input_folder = "./videos/walking/"
    results_folder = "./images/results/"
    input_video = "./videos/walking.mp4"
    output_folder = "./videos"
    print("Start splitting video")
    # video_utils.split_video(input_video, input_folder)
    print("Video was splited")
    images = load_images_from_folder(input_folder)
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

    count = 0
    for img in images:
        hip = PartialSkeleton.skeletonize(img,hip,count)
        count +=1

    video_utils.create_video(input_video, results_folder, output_folder)