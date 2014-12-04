# Motion Tweening: Code Structure

## What this document is

This document describes the structure of our code and how it is used. If you're looking for an explanation of the technical details of the various approaches employed, see our final report.

As we've attempted a number of different approaches, some of which have worked better than others, the structure of the code is a bit confusing (particularly at the top level track_planar.py script, which controls which behaviors are in use through command line arguments), but we'll try to explain it here and in the final report as best as possible.

## Files Overview

- track_planar.py: this is the main executable file. It parses the command line options and makes calls into the other modules.
- detection.py: this contains the implementation of ArbitraryPlaneDetector, which uses contour detection to look for rectangular regions in the frame. It implements several cost functions which can be minimized over to choose a contour from the set of contours detected in a frame.
- tracking.py: this contains the implementation of three classes.
  - GenericTracker: this is naive feature-matching tracking (very similar to assignment 1) that requires a training image.
  - OpticalFlowTracker: A superclass of OpticalFlowHomographyTracker and OpticalFlowPointTracker containing some shared parameters for the optical flow tracking
  - OpticalFlowHomographyTracker: Uses optical flow to compute a homography representing the movement of the tracked surface from an initial frame to the current frame.
  - OpticalFlowPointTracker: Uses optical flow to compute the new positions of a set of points specified from an initial frame in the current frame.
- graphics.py: this handles drawing various things on the frame, including 3D models
- contour_merge.py: this implements logic to combine multiple contours in an effort to find a more rectangular contour. Since the contours are represented as ordered lists of points, it combines them oriented such that the distance between the newly adjacent points is minimized.
- filtering.py: this contains an implementation of a Kalman filter, mostly stripped from project 3.
- miscmath.py: this contains some miscellaneous utility math functions used elsewhere (namely averaging and Euclidean distance)
- obj.py: this is mostly imported code (attribution within) with some adaptation. It loads a 3D model from the OBJ file format.
- vectormath.py: this contains some miscellaneous utility math functions related to vectors, edges, and quadrilaterals
- test_suite.py: this contains unit tests for the files listed above.

## How to Run

$ track_planar.py [options] [video] [trainingFrame]

If the --stream option is not in use, both the video and trainingFrame parameters are required. This does the very naive tracking that requires a flat image of the planar surface.

If the --stream option is in use, the trainingFrame parameter is ignored and the video parameter is optional. If a video is provided, the video file will be used. If it isn't, the index-0 webcam attached to the system will be used.

Options:
  -s, --stream          stream live video and auto-detect planar surfaces
  -o FILE, --object=FILE
                        the 3D OBJ file to overlay
  -d DRAWSTYLE, --drawstyle=DRAWSTYLE
                        3D Model draw style [line, line_shader, face_shader]
  -c, --corners         show the corners of the tracked planar surface
  -v, --no-viz          hide focused contour outline
  -a, --all             show all contours detected (not just the focused one)
  -n, --no-write        skip writing video file (for systems that don't
                        support it, or performance reasons)
  -k, --kalman          use Kalman Filter to smooth motion of predicted corners
  -m, --merge-contours  if on, attempt to merge pairs of contours that become
                        more rectangular when merged
  -f COSTMODE, --costfunc=COSTMODE
                        which cost function to use to evaluate contours
  -t TRACKMODE, --tracker=TRACKMODE
                        which tracker to use for corner tracking [features,
                        flow, pointFlow, naive]
  -l, --stall           Stall video on each frame when not tracking

With the image window focused, the user can draw on the frame using the mouse.

With the image window focused, the following keyboard commands exist:

l: toggle the "stalled" state (-l on the command line)
p: "commit" the current drawing and start tracking it.
t: begin using the specified tracker to track the plane with corners detected in the current frame. The 3D model will start being rendered over it.
q: quit prematurely
any other key: while in the stalled state, advance by one frame

## Other Files

There are also a number of shell scripts for running preset configurations with provided videos. The results of these are discussed in the final report.

- run_blankpaper.sh
- run_blankpaper_flow.sh
- run_hemingway.sh
- run_humanoid.sh
- run_r2d2.sh