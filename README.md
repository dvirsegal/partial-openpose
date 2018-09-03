# Partial OpenPose

The purpose of this project is to use ![OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) system capabilities on partial human images.
One of prerequisites of OpenPose system is to have a full body image, since it reconstructs the skeleton is based on 16 points generating the whole body parts.

Our method is to wisely add the missing body part into the image and then run the OpenPose framework on it.
The result is then manipulated to show the skeleton on the original body.

We demonstarte our pruposed method on a video displaying a walking person (legs only).

As can be seen in the following GIFs:

Original Video            |  Skeletonized Video
:-------------------------:|:-------------------------:
<img src="https://github.com/DeJaVoo/partial-openpose/blob/master/site/walking.gif" width="480" height="360" /> |  <img src="https://github.com/DeJaVoo/partial-openpose/blob/master/site/waling-skeleton.gif" width="480" height="360" />
