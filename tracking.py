import cv2
import numpy as np

class TrackedPlane:
    
    def __init__( self, init_corners, corners ):
        self.init_corners = init_corners
        self.corners = corners
        
        # Planarize corners to estimate head-on plane
        p0, p1, p3 = init_corners[0], init_corners[1], init_corners[3]
        w,h = np.linalg.norm(p1[0] - p0[0]), np.linalg.norm(p3[1] - p0[1])

        self.planarized_corner_map = np.float32([
            p0,
            (p0[0] + w, p0[1]),
            (p0[0] + w, p0[1] + h),
            (p0[0], p0[1] + h)
        ])

# this is closely adapted from project 1
class GenericTracker:

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

    def track(self, frame):
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

class OpticalFlowTracker:

    def __init__(self, old_gframe, pts=[]):
        self.old_gframe = old_gframe
        self.init_pts = np.float32(pts)
        self.pts = np.float32(pts)

        # Termination criteria: 10 iterations or moved at least 1pt
        self.term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )

        # Parameters for lucas kanade optical flow
        self.lk_params = dict( winSize  = (15,15),
                               maxLevel = 2,
                               criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


    def track(self, gframe):
        p1, st, err = cv2.calcOpticalFlowPyrLK(self.old_gframe, gframe, self.pts, None, **self.lk_params)
        homography, _ = cv2.findHomography(self.init_pts, p1, cv2.RANSAC, 5.0)
        self.pts = p1
        self.old_gframe = gframe
        return homography
