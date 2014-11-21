#!/usr/bin/env python

from miscmath import *
import numpy as np
import cv2

def appendContours(a, b, flipFirst=False, flipSecond=False):
	"""
	Helper function for mergeContours() doing the actual array combination
	"""
	ac = a[::-1] if flipFirst else a
	bc = b[::-1] if flipSecond else b
	return np.append(ac, bc, axis=0)

def mergeContours(a, b):
	"""
	Returns a contour with the provided contours combined. This looks for the
	arrangement that minimizes distance between adjacent points in the vector
	in order to maintain correct ordering in the resulting contour.
	"""

	a1 = a[0, 0, 0], a[0, 0, 1]
	a2 = a[-1, 0, 0], a[-1, 0, 1]
	b1 = b[0, 0, 0], b[0, 0, 1]
	b2 = b[-1, 0, 0], b[-1, 0, 1]
	
	# TODO Matt: write a comment explaining in more detail what's going on here.
	exprs = {
		dist(a2[0], a2[1], b1[0], b1[1]):
			lambda a, b: appendContours(a, b),
		dist(b2[0], b2[1], a1[0], a1[1]):
			lambda a, b: appendContours(b, a),
		dist(a1[0], a1[1], b1[0], b1[1]):
			lambda a, b: appendContours(b, a, flipFirst=True),
		dist(b2[0], b2[1], a2[0], a2[1]):
			lambda a, b: appendContours(b, a, flipSecond=True)
	}

	return min(exprs.items(), key=lambda it: it[0])[1](a, b)

def contourRectangularity(contour):
	"""
	Uses shape matching to compute a measure (Hu moment) of a contour. The
	smaller this number is, the more rectangular the contour is.
	"""

	rect = np.array([[[0, 0]], [[150, 0]], [[150, 300]], [[0, 300]]])
	return cv2.matchShapes(contour, rect, 2, 0.0)

def contoursMergeable(a, b, combined):
	"""
	Returns true if two contours become more rectangular when combined.
	"""

	ar = contourRectangularity(a)
	br = contourRectangularity(b)
	cr = contourRectangularity(combined)

	# todo: have a threshold instead of just seeing if it's less than both
	return cr < ar and cr < br

def collapseContours(contours):
	"""
	Takes a list of contours and attempts to collapse it into a shorter one by
	combining some contours that seem to go together to make rectangles.
	"""

	i = 1
	ret = contours[:1]
	for i in range(len(contours)):
		a, b = ret[-1], contours[i]
		combined = mergeContours(a, b)
		good = contoursMergeable(a, b, combined)
		if good:
			print "combining contours %d and %d" % (len(ret)-1, i)
			ret[-1] = combined
		else:
			ret.append(b)

	return ret
