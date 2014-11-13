#!/usr/bin/env python

import cv2

class VideoTracker:
    def __init__(self, videoFilename, overlayFilename, overlayBounds):
        pass

    def frames():
        cap = cv2.VideoCapture(0)
        retval, frame = cap.read()

		while retval != 0:
		    # return captured frame to the user
		    yield frame

		    # Capture next frame
		    retval, frame = cap.read()