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

    def detect(self, frame, same_corner_threshold_dist=20, same_side_threshold_angle=0.52):
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

        # 1. Canny Edge Detection
        gframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gframe = cv2.GaussianBlur(gframe, (5, 5), 0)
        edges = cv2.Canny(gframe, 50, 150)

        # 2. External contours
        contours, hierarchy = cv2.findContours(edges,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            cv2.drawContours(frame, contour, -1, (255, 255, 0), 10)

        # 3. Estimate the largest contour
        #bounds = [cv2.boundingRect(cnt) for cnt in contours]
        #areas = [b[2] * b[3] for b in bounds]
        if not contours:
            return [[1,1], [2,1], [2,2], [1,2]]

        #cnt_idx = areas.index(max(areas))
        #num_pts = [len(contour) for contour in contours]
        #cnt_idx = num_pts.index(max(num_pts))
        rect = np.array([[[0, 0]], [[200, 0]], [[200, 200]], [[0, 200]]])
        hu_moments = [cv2.matchShapes(contour, rect, 1, 0.0) for contour in contours]
        cnt_idx = hu_moments.index(min(hu_moments))
        largestContour = contours[cnt_idx]  # May be better to continuously pick and do pairwise dist check until you get contour w/ at least 4 pts

        # Approximate contour as quad region with 4 corners
        epsilon = 3
        approxCurve = cv2.approxPolyDP(largestContour, epsilon, True)

        cv2.drawContours(frame, largestContour, -1, (0, 255, 0), 10)

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

        for pt in approxCurve:
            cv2.circle(frame, tuple(pt), 4, (255, 0, 0))

        # Check each pair of connected sides for commonality.
        # Again, tuple conversion allows us to check array containment easily.
        pt_triplets = itertools.combinations(approxCurve, 3)
        rejects = []
            
        for triplet in pt_triplets:
            tuplet = map(lambda x: tuple(x), triplet)
            if tuplet[0] in rejects or \
               tuplet[1] in rejects or \
               tuplet[2] in rejects:
                continue
                
            p1, p2, p3 = triplet[0], triplet[1], triplet[2]            
            v1, v2 = unit_vector(p2 - p1), unit_vector(p3 - p2)
            angle = angle_between(v1, v2)

            # Angle Test
            if angle < same_side_threshold_angle:
                rejects.append(tuplet[1])

            # Reverse-Angle Test
            elif abs(np.pi - angle) < same_side_threshold_angle:
                rejects.append(tuplet[2])
            
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

        # Fill in with bad points for now
        while len(approxCurve) < 4:
            approxCurve = np.vstack([approxCurve, approxCurve[len(approxCurve) - 1]])

        # Truncate for now
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
