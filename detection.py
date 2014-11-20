#!/usr/bin/env python

import cv2
import itertools
import numpy as np
import vectormath as vmath
from miscmath import *
from contour_merge import *
import math

# utility generator function that yields all the points in a contour as (x, y) tuples
def pointsFromContour(cnt):
    for pt in cnt:
        yield (pt[0, 0], pt[0, 1])

class ArbitraryPlaneDetector:

    previouslyReturned = None
    costMode = "rect"
    mergeMode = False

    def __init__(self, costMode="rect", mergeMode=False):
        self.costMode = costMode
        self.mergeMode = mergeMode

    def filter_corners(self, approxCurve, alpha):
        """
        Removes duplicate corners by pairwise distance test.
        """

        pt_pairs = itertools.combinations(approxCurve, 2)
        rejects = []
            
        for pair in pt_pairs:                
            # Tuple conversion allows us to use 'pair in rejects' check.
            tpair = map(lambda pt: tuple(pt), pair)
            if tpair[0] in rejects or tpair[1] in rejects:
                continue
                    
            # Distance test
            d = np.linalg.norm(pair[1] - pair[0])
            if d < alpha:
                rejects.append(tpair[1])

        return rejects        
        
    def filter_edges(self, approxCurve, beta):
        """
        Combines common edges by pairwise edge check.
        Should this check be done if we already have 4 points?
        """

        pt_triplets = itertools.combinations(approxCurve, 3)
        rejects = []

        for triplet in pt_triplets:
            tuplet = map(lambda x: tuple(x), triplet)
            if any(map(lambda pt: pt in rejects, tuplet)):
                continue

            is_coplanar, midpt = vmath.test_coplanar(triplet, beta)
            if is_coplanar:
                rejects.append(tuple(midpt))
            
            # Short-circuit if done
            if len(approxCurve) - len(rejects) == 4:
                break

        return rejects

    def order_corners_clockwise(self, approxCurve):
        """
        Organize corners into clockwise rotation
        Assume the two corners with largest yvals = top corners
        This will fail on rotation...should only use this method to initialize.
        Afterwards, track corners using features/small window?
        """

        y_vals = map(lambda p: p[1], approxCurve)
        y_sorted = np.argsort(y_vals)

        corners = map(lambda idx: approxCurve[idx], y_sorted)
        c1, c2, c3, c4 = corners
        if c1[0] > c2[0]: c1, c2 = c2, c1
        if c4[0] > c3[0]: c3, c4 = c4, c3
        return [c1, c2, c3, c4]

    def detect(self, frame, gaussian_kernel=(15, 15), alpha=20, beta=0.52, viz=True):
        """
        Detects arbitrary planes in a frame.
          1. Canny Edge Detection on blurred grayscale image
          2. Find external contours on edge image
          3. Take the planar contour with largest estimated area
          4. Approximate contour and fit to quad region
        
        Arguments:
          frame: The frame to detect the largest plane in.
          gaussian_kernal: Kernal to blur image by for Canny detection
          alpha: Minimum distance between two points to consider them different features
          beta: Minimum angle between two vectors to consider them different sides (default: ~30 degrees)
          viz: Modifies the input frame to draw visual debuggers

        Returns:
          An array of x-y pairs indicating the corners in CW order, starting from top-left
        """

        # 1. Canny Edge Detection
        gframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gframe = cv2.GaussianBlur(gframe, gaussian_kernel, 0)
        edges = cv2.Canny(gframe, 50, 150)
        edges = cv2.dilate(edges, (-1,-1))

        # 2. Find External contours
        contours, hierarchy = cv2.findContours(edges,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if viz:
            for i, contour in enumerate(contours):
                # draw each separate contour a different color
                c = int(float(i)/len(contours) * 200)
                color = (255, c, 0)

                cv2.drawContours(frame, contour, -1, color, 10)

                points = list(pointsFromContour(contour))
                avgx = int(avg((x for x, y in points)))
                avgy = int(avg((y for x, y in points)))
                inv = tuple(map(lambda x: 255-x, color))
                cv2.circle(frame, (avgx, avgy), 100, color, 5)

        # Return previous contour state if no contours found
        if not contours:
            print "returning previous corners because no contours found"
            if self.previouslyReturned:
                return self.previouslyReturned
            else:
                return np.array([[0, 0], [150, 0], [150, 300], [0, 300]])

        # attempt to merge contours
        if self.mergeMode:
            contours = collapseContours(contours)

        # 3. Estimate the most rectangular contour
        approxCurve = []
        rect = np.array([[[0, 0]], [[150, 0]], [[150, 300]], [[0, 300]]])
        
        # this cost function estimates how similar a contour is to a rectangle
        def rectCost(contour):
            return cv2.matchShapes(contour, rect, 2, 0.0)

        # this cost function is an average of the distances to the center of the frame for all points in the contour
        cx, cy = frame.shape[1]/2.0, frame.shape[0]/2.0
        dim = max(frame.shape[0], frame.shape[1])
        def midDistCost(contour):
            return avg((dist(x, y, cx, cy) / dim for x, y in pointsFromContour(contour)))

        # computes linear combination of features
        # note that the weights are inverted because we're doing a minimization. Large weights imply greater importance.
        def combinedCostFunction(*args):
            def fn(cnt):
                total = 0
                print "---"
                for weight, subFunc in args:
                    value = subFunc(cnt)
                    total += value / float(weight)
                    print "%s: %8.5f (weight=%.3f)" % (subFunc.__name__, value, weight)
                return total
            return fn

        costFuncs = {
            "rect": rectCost,
            "middist": midDistCost,
            "combined1": combinedCostFunction((1, midDistCost), (1, rectCost)),
        }

        costFunction = costFuncs[self.costMode]
        costs = [costFunction(contour) for contour in contours]

        while len(approxCurve) < 4 and contours:
            cnt_idx = costs.index(min(costs))
            bestContour = contours.pop(cnt_idx)
            costs.pop(cnt_idx)

            # Approximate contour as a polygon
            approxCurve = cv2.approxPolyDP(bestContour, epsilon=3, closed=True)            
            approxCurve = map(lambda x: x[0], approxCurve)
            
            rejects = self.filter_corners(approxCurve, alpha)
            approxCurve = np.array([pt for pt in approxCurve if tuple(pt) not in rejects])

        if viz:
            cv2.drawContours(frame, bestContour, -1, (0, 255, 0), 10)
            for pt in approxCurve:
                cv2.circle(frame, tuple(pt), 4, (255, 0, 0))

        # Filter edges and approximate to best quadrilateral
        rejects = self.filter_edges(approxCurve, beta)
        approxCurve = np.array([pt for pt in approxCurve if tuple(pt) not in rejects])
        approxCurve = vmath.approx_quadrilateral(approxCurve, alpha)

        self.previouslyReturned = self.order_corners_clockwise(approxCurve)
        return self.previouslyReturned
