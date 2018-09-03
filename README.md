# Partial OpenPose

The purpose of this project is to use OpenPose capabilities on partial human images.
One of prerequisite of OpenPose system is to have full body image since it reconstructs the skeleton based on 16 points generating the whole body parts.

Our method is to wisely add the missing body part into the image and then run the OpenPose.
The result is then manipulated to show the skeleton on the original body.

We demonstarte our pruposed method on a video displaying legs only.

As can be seen in the following GIFs:

Original Video            |  Skeletonized Video
:-------------------------:|:-------------------------:
![](https://github.com/DeJaVoo/partial-openpose/blob/master/site/walking.gif) |  ![](https://github.com/DeJaVoo/partial-openpose/blob/master/site/waling-skeleton.gif)
