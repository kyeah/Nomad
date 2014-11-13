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