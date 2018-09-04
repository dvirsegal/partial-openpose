# Partial OpenPose

The purpose of this project is to use [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) system capabilities on partial human images.
One of prerequisites of OpenPose system is to have a full body image, since it reconstructs the skeleton based on 16 points which generating the whole body parts.

Our method is to wisely add the missing body part into the image and then run the OpenPose framework on it.
The result is then manipulated to show the skeleton on the original body.

We demonstarte our pruposed method on a video displaying a walking person (legs only).

As can be seen in the following GIFs:

Original Video            |  Skeletonized Video
:-------------------------:|:-------------------------:
<img src="site/walking.gif" width="480" height="360" /> |  <img src="site/walking-skeleton.gif" width="480" height="360" />

**Project requirements:**

* python 3.5 or above
* numpy
* opencv
* download [tf-pose-estimation](https://github.com/ildoonet/tf-pose-estimation) - release date 16.02.18
* make sure the tf-pose-estimation folder is on the same level of partial-openpose folder
* make sure to change all relative paths

To run the project, use python generate_partial_skeleton_from_video.py

