#!/usr/bin/env python

import cv2
import numpy as np

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

    def detect(self, frame):
        """
        Detects arbitrary planes in a frame.
          1. Canny Edge Detection on blurred grayscale image
          2. Find external contours on edge image
          3. Take the planar contour with largest estimated area
          4. Approximate contour and fit to quad region
        
        """

        # 1. Canny Edge Detection
        gframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gframe = cv2.GaussianBlur(gframe, (15,15), 0)
        edges = cv2.Canny(gframe, 50, 150)

        # 2. External contours
        contours, hierarchy = cv2.findContours(edges,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
 
        # 3. Estimate the largest contour
        bounds = [cv2.boundingRect(cnt) for cnt in contours]
        areas = [b[2] * b[3] for b in bounds]
        cnt_idx = areas.index(max(areas))
        largestContour = contours[cnt_idx]

        # Approximate contour as quad region with 4 corners
        cntframe = np.zeros_like(gframe)
        approxCurve = cv2.approxPolyDP(largestContour, 3, True)

        # while len(approxCurve) != 4:
        # for idx in xrange(len(approxCurve - 2)):
        # print p1, p2, p3

        cv2.drawContours(frame,approxCurve,-1,(0,0,255), 10)
        
        cv2.imshow('f', frame)
        cv2.waitKey()
        cv2.destroyAllWindows()

        # Todo: approximate to 4 corners and determine clockwise order
        return map(lambda x: x[0], approxCurve)
