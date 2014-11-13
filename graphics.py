#!/usr/bin/env python

import cv2

def drawCorners(frame, corners):
	"""
	Draws the corners of a tracked rectangluar planar surface denoted by the
	list of corners as (x, y) coordinates in the frames. Returns the
	frame (though there is no promise of not modifying the frame passed in).
	"""

	for x, y in corners:
		x, y = int(x), int(y)
		cv2.circle(frame, (x, y), 10, (0, 0, 255), 3, 8, 0)

	pass