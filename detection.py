#!/usr/bin/env python

import cv2
import itertools
import numpy as np
import vectormath as vmath

# this is closely adapted from project 1
class PlaneDetector:

    def __init__(self, trainingFrame):
        self.trainingFrame = trainingFrame
        self.trainingFeatures = self.findFeatures(trainingFrame)

    def findFeatures(self, frame):
        kp, desc = cv2.SIFT().detectAndCompute(frame, None)
        return kp, desc

    def matchFeatures(self, descQuery, descTraining, ratio=0.7):
        def filterKNN(matches, ratio):
            def criterion(match):
                a, b = match
                return a.distance < ratio * b.distance
            return [m[0] for m in filter(criterion, matches)]

        FLANN_INDEX_KDTREE = 0
        index_params = {
            "algorithm": FLANN_INDEX_KDTREE,
            "trees": 5
        }
        search_params = {
            "checks": 50
        }
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(descQuery, descTraining, k=2)

        return filterKNN(matches, ratio)

    def detect(self, frame):
        """
        Return something, probably. A homography, perhaps?
        """
        kpA, descA = self.trainingFeatures
        kpB, descB = self.findFeatures(frame)

        # match the features
        matches = []
        for ratio in [0.7, 0.75, 0.8, 0.85]:
            matches = self.matchFeatures(descA, descB, ratio)
            if len(matches) >= 4:
                break
        assert len(matches) >= 4

        # compute homography
        srcPoints = np.array([kpA[match.queryIdx].pt for match in matches])
        dstPoints = np.array([kpB[match.trainIdx].pt for match in matches])
        homography, _ = cv2.findHomography(srcPoints, dstPoints, cv2.RANSAC, 5.0)

        return homography


class ArbitraryPlaneDetector:

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

    def detect(self, frame, gaussian_kernel=(15, 15), alpha=20, beta=0.52, viz=False):
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
            for contour in contours:
                cv2.drawContours(frame, contour, -1, (255, 255, 0), 10)

        # Return default rectangle if no contours found; todo: retain previous contour state
        if not contours:
            return np.array([[0, 0], [150, 0], [150, 300], [0, 300]])

        # 3. Estimate the most rectangular contour
        approxCurve = []
        rect = np.array([[[0, 0]], [[150, 0]], [[150, 300]], [[0, 300]]])
        hu_moments = [cv2.matchShapes(contour, rect, 2, 0.0) for contour in contours]
        
        while len(approxCurve) < 4 and contours:
            cnt_idx = hu_moments.index(min(hu_moments))
            bestContour = contours.pop(cnt_idx)
            hu_moments.pop(cnt_idx)

            if viz:
                cv2.drawContours(frame, bestContour, -1, (0, 255, 0), 10)
            
            # Approximate contour as a polygon
            approxCurve = cv2.approxPolyDP(bestContour, epsilon=3, closed=True)            
            approxCurve = map(lambda x: x[0], approxCurve)
            
            rejects = self.filter_corners(approxCurve, alpha)
            approxCurve = np.array([pt for pt in approxCurve if tuple(pt) not in rejects])

        if viz:
            for pt in approxCurve:
                cv2.circle(frame, tuple(pt), 4, (255, 0, 0))

        # Filter edges and approximate to best quadrilateral
        rejects = self.filter_edges(approxCurve, beta)
        approxCurve = np.array([pt for pt in approxCurve if tuple(pt) not in rejects])
        approxCurve = vmath.approx_quadrilateral(approxCurve, alpha)

        return self.order_corners_clockwise(approxCurve)
