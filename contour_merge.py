#!/usr/bin/env python

from miscmath import *
import numpy as np
import cv2


def appendContours(a, b, flipFirst=False, flipSecond=False):
    """
    Helper function for mergeContours() doing the actual array combination.
    a and b are appended and the resulting list of points is returned. If
    flipFirst is True, a will be reversed before being appended. If flipSecond
    is True, b will be reversed before being appended.
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

    # This is a bit confusing so hold on tight.
    # For each possible orientation of the two contours, we map the distance
    # between the points that would be adjacent in the resulting list to a
    # lambda that appends the contours in that orientation. Then, we take a min
    # over that map and execute the selected lambda against the contours.
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
    Uses shape matching to compute a scalar measure (Hu moment) of a contour.
    The smaller this number is, the more rectangular the contour is. The number
    is returned directly.
    """

    # This is a sample rectangle. The cv2.matchShapes procedure is scale and
    # rotation invariant so all that matters is that this is a rectangle of
    # some sort.
    rect = np.array([[[0, 0]], [[150, 0]], [[150, 300]], [[0, 300]]])

    return cv2.matchShapes(contour, rect, 2, 0.0)


def contoursMergeable(a, b, combined):
    """
    Returns True if two contours become more rectangular when combined. Both
    the original contours and the combined contour are required as arguments to
    save doing the combination twice in collapseContours if this function ends
    up returning True.
    """

    ar = contourRectangularity(a)
    br = contourRectangularity(b)
    cr = contourRectangularity(combined)

    # todo: have a threshold instead of just seeing if it's less than both
    return cr < ar and cr < br


def collapseContours(contours):
    """
    Takes a list of contours and attempts to collapse it into a shorter one by
    combining groups of contours adjacent in the list that seem to go together.
    This uses the contoursMergeable procedure above to determine if combining
    the contours would result in a more rectangular contour. Returns a list of
    contours of size less than or equal to the size of the list of contours
    passed in.
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
