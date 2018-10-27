import cv2
import os


def load_images_from_folder(folder, save_path=False):
    images = []
    for filename in os.listdir(folder):
        image = cv2.imread(os.path.join(folder, filename))
        if image is not None:
            if save_path:
                images.append([image, os.path.join(folder, filename)])
            else:
                images.append(image)
    return images


def split_video(vid_path, out_path):
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    vidcap = cv2.VideoCapture(vid_path)
    success, image = vidcap.read()
    count = 0
    # success = True
    while success:
        success, image = vidcap.read()
        if success:
            resized_image = cv2.resize(image, (432, 368), interpolation=cv2.INTER_AREA)
            cv2.imwrite("{}\\{}.png".format(out_path, count), resized_image)
            count += 1


def create_video(orig_video, images_path, out_path):
    vidcap = cv2.VideoCapture(orig_video)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    vidcap.release()
    os.system(
        "ffmpeg -f image2 -r {} -i {}/%01d.png -vcodec libx264 -y {}/output.mp4".format(fps, images_path, out_path))


if __name__ == '__main__':
    # split_video("./videos/walking.mp4", "./videos/walking/")
    create_video("./videos/walking.mp4", "./videos/walking/", "./videos")
