#!/usr/bin/env python

import cv2
import itertools
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


def unit_vector(vector):
    """
    Returns the unit vector of the vector.
    """
    return vector / np.linalg.norm(vector)
    
def angle_between(v1, v2):
    """ 
    Returns the angle in radians between vectors 'v1' and 'v2'::
    """
    
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    angle = np.arccos(np.dot(v1_u, v2_u))
    if np.isnan(angle):
        if (v1_u == v2_u).all():
            return 0.0
        else:
            return np.pi
    return angle



class ArbitraryPlaneDetector:

    def detect(self, frame, same_corner_threshold_dist=20, same_side_threshold_angle=0.52, viz=False):
        """
        Detects arbitrary planes in a frame.
          1. Canny Edge Detection on blurred grayscale image
          2. Find external contours on edge image
          3. Take the planar contour with largest estimated area
          4. Approximate contour and fit to quad region
        
        Arguments:
          frame: The frame to detect the largest plane in.
          same_corner_threshhold_dist: Minimum distance to consider two points the same feature
          same_side_threshold_angle: Minimum angle between two vectors to consider them the same side (default: ~30 degrees)
        """

        # TODO: Modularize mathematical computations to make code easier to read

        # 1. Canny Edge Detection
        # Todo: Manual gaussian kernel modification to account for different environments
        gframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gframe = cv2.GaussianBlur(gframe, (5, 5), 0)
        edges = cv2.Canny(gframe, 50, 150)
        edges = cv2.dilate(edges, (-1,-1))

        # 2. External contours
        contours, hierarchy = cv2.findContours(edges,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if viz:
            for contour in contours:
                cv2.drawContours(frame, contour, -1, (255, 255, 0), 10)

        # 3. Estimate the most rectangular contour
        rect = np.array([[[0, 0]], [[150, 0]], [[150, 300]], [[0, 300]]])

        # Return default rectangle if no contours found; todo: retain previous contour state
        if not contours:
            return rect

        approxCurve = []
        hu_moments = [cv2.matchShapes(contour, rect, 2, 0.0) for contour in contours]
        
        while len(approxCurve) < 4 and contours:
            cnt_idx = hu_moments.index(min(hu_moments))
            hu_moments.pop(cnt_idx)

            bestContour = contours.pop(cnt_idx)

            # Approximate contour as quad region with 4 corners
            approxCurve = cv2.approxPolyDP(bestContour, epsilon=3, closed=True)
            
            if viz:
                cv2.drawContours(frame, bestContour, -1, (0, 255, 0), 10)
            
            approxCurve = map(lambda x: x[0], approxCurve)
            
            # Pairwise distance check; rejects contain rejected points.
            # Tuple conversion allows us to use 'pair in rejects' check.
            pt_pairs = itertools.combinations(approxCurve, 2)
            rejects = []
            
            for pair in pt_pairs:
                tpair = map(lambda pt: tuple(pt), pair)
                if tpair[0] in rejects or tpair[1] in rejects:
                    continue
                    
                # Distance test
                d = np.linalg.norm(pair[1] - pair[0])
                if d < same_corner_threshold_dist:
                    rejects.append(tpair[1])

            approxCurve = np.array([pt for pt in approxCurve if tuple(pt) not in rejects])

        if viz:
            for pt in approxCurve:
                cv2.circle(frame, tuple(pt), 4, (255, 0, 0))

        # Check each pair of connected sides for commonality.
        # Again, tuple conversion allows us to check array containment easily.
        # Should this check be done if we already have 4 points? Maybe. Might get rid of false quads.
        pt_triplets = itertools.combinations(approxCurve, 3)
        rejects = []
            
        for triplet in pt_triplets:
            tuplet = map(lambda x: tuple(x), triplet)
            if any(map(lambda pt: pt in rejects, tuplet)):
                continue
                
            p1, p2, p3 = triplet
            v1, v2 = unit_vector(p2 - p1), unit_vector(p3 - p2)
            angle = angle_between(v1, v2)

            # Angle, Reverse-angle Test
            # If the three lines are roughly coplanar, find the longest line and remove the middle point.
            if angle < same_side_threshold_angle or abs(np.pi - angle) < same_side_threshold_angle:
                distances = map(lambda idx: np.linalg.norm((idx+1) % 3 - (idx+2) % 3), xrange(3))
                rejects.append(distances.index(max(distances)))
            
            # Short-circuit if done
            if len(approxCurve) - len(rejects) == 4:
                break

        approxCurve = np.array([pt for pt in approxCurve if tuple(pt) not in rejects])

        if len(approxCurve) == 2:
            # Find dx, dy, and infer head-on plane if the two corners look like they are opposite from each other
            c1, c2 = approxCurve[0], approxCurve[1]
            dx, dy = c2[0] - c1[0], c2[1] - c1[1]
            if abs(dx) > same_corner_threshold_dist / 4 and abs(dy) > same_corner_threshold_dist / 4:
                if dx < 0: c2, c1 = c1, c2
                approxCurve = np.vstack([approxCurve, [[c1[0] + dx, c1[1]], [c1[0], c1[1] + dy]]])

        elif len(approxCurve) == 3:
            # Find hypotenuse, then find last corner
            c1, c2, c3 = approxCurve[0], approxCurve[1], approxCurve[2]
            d1, d2, d3 = np.linalg.norm(c2 - c1), np.linalg.norm(c3 - c1), np.linalg.norm(c3 - c2)
            # todo: ...finish this

        # Last resorts: if not enough points, replicate; if too many points, truncate.
        while len(approxCurve) < 4:
            approxCurve = np.vstack([approxCurve, approxCurve[len(approxCurve) - 1]])

        approxCurve = approxCurve[:4]

        # Organize corners into clockwise rotation
        # Assume the two corners with largest yvals = top corners
        # This will fail on rotation...should only use this method to initialize.
        # Afterwards, track using features?
        y_vals = map(lambda p: p[1], approxCurve)
        y_sorted = np.argsort(y_vals)

        corners = map(lambda idx: approxCurve[idx], y_sorted)
        c1, c2, c3, c4 = corners
        if c1[0] > c2[0]: c1, c2 = c2, c1
        if c4[0] > c3[0]: c3, c4 = c4, c3

        return [c1, c2, c3, c4]
