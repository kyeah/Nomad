Nomad
=========

A motion tweening and 3D augmented reality system design to utilize 3D information recovered from a single 2D image to insert painted structures and mesh models automatically into the image and into subsequent frames of video. 

![Longhorn Detection](/res/longhorn_detection.png)

The goals of this project were twofold:

1. To replicate the parallel tracking behavior demonstrated in the [2007 Klein-Murray PTAM paper](http://www.robots.ox.ac.uk/~gk/publications/KleinMurray2007ISMAR.pdf) using analysis of a single two-dimensional image for multiple-surface detection.
2. To replicate the nonplanar motion tracking behavior demonstrated in [Disney’s Paperman short](https://www.youtube.com/watch?v=OKl9mpGMCiA#t=1m41s), which utilizes dense optical flow methods as part of the Meander system pipeline for motion tweening.

Our system consists of a primary user interface that iterates over a stream of static or real-time images and dispatches actions to several independent modules. Our main modules comprise a suite of detection, tracking, graphics, vector math, and contour merging classes and methods that make it easy to combine different algorithms for a variety of different video and image cases that may arise. Another module, smoothing and filtering, is included but not completed. This module is intended to reduce the error contributed by bad per-frame tracking measurements.

This diagram shows the structure visually:

![Nomad System Pipeline](/res/nomad_system.png)

More information about our technical system design can be found in OVERVIEW.pdf, and our user interface design can be found at src/README.md.

# Sparse Flow Surface Tracking

![](/res/blank_sparse_flow.gif)

![](/res/longhorn.gif)

![](/res/sidewalk.gif)

# Feature-based Surface Tracking

![](/res/cello_map_small.gif)

# Dense Optical Flow for Motion Tracking

![](/res/r2d2.gif)

![](/res/r2d2_2.gif)

![](/res/paperman.gif)

:------------------------------:|:-----------------------------:
![](/res/blank_sparse_flow.gif) | ![](/res/cello_map_small.gif)
![](/res/r2d2.gif)              | ![](/res/r2d2_2.gif)
![](/res/sidewalk.gif)          | ![](/res/paperman.gif)
![]()                           | ![]()

---------------------------------

Kevin Yeh, Matt Broussard, Kaelin Hooper, Conner Collins (3D Reconstruction with Computer Vision 2014)