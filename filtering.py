#!/usr/bin/env python

import cv2
from cv2 import cv

class Filter:
	def observe(self, measurement):
		pass
	def predict(self):
		pass

class MaxDifferenceFilter(Filter):
	pass

class KalmanFilter(Filter):
    inited = False
    kf = None
    measurement = None
    timestep = 0

    def __init__(self, useProgressivePNC=False, pnc=1e-4):
        # state vector:
        #  0-7: x, y for each corner
        #  8-16: derivatives of 0-7 respectively
        self.useProgressivePNC = useProgressivePNC
        self.kf = cv.CreateKalman(16, 8, 0)

        self.setOldCvMat(self.kf.transition_matrix, [
            [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
			[0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
			[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
			[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
			[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
			[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
			[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
			[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
			[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
			[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
			[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
			[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
			[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
			[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
			[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
			[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        ])

        self.measurement = cv.CreateMat(8, 1, cv.CV_32FC1)

        cv.SetIdentity(self.kf.measurement_matrix, cv.RealScalar(1))
        cv.SetIdentity(self.kf.process_noise_cov, cv.RealScalar(pnc))
        cv.SetIdentity(self.kf.measurement_noise_cov, cv.RealScalar(1e-1))
        cv.SetIdentity(self.kf.error_cov_post, cv.RealScalar(1))

    def setOldCvMat(self, cvmat, arr):
        for r, row in enumerate(arr):
            for c, v in enumerate(row):
                cvmat[r, c] = v

    # as suggested at http://dsp.stackexchange.com/questions/3039
    #                 /kalman-filter-implementation-and-deciding-parameters
    # this progressively increases the processNoiseCov parameter. This controls
    # what the Kalman filter perceives as the amount stochastic noise in the
    # model to mitigate the issue of it becoming more trusting of itself and
    # less trusting of new observations over time.
    def updateProcessNoiseCov(self):
        if not self.useProgressivePNC:
            return

        self.timestep += 1

        startTS = 0
        endTS = 400
        startCov = 1e-4
        endCov = 1e-3

        covSlope = (endCov - startCov) / (endTS - startTS)
        covIntercept = startCov

        cov = covSlope * (self.timestep - 30) + covIntercept
        print "frame %d: cov=%.4g" % (self.timestep, cov)

        cv.SetIdentity(self.kf.process_noise_cov, cv.RealScalar(cov))

    def observeInternal(self, measurement):

        measurementMatrix = map(lambda v: [v], measurement)

        if not self.inited:
            self.inited = True

            self.setOldCvMat(self.kf.state_pre, measurementMatrix)

        else:

            self.setOldCvMat(self.measurement, measurementMatrix)
            cv.KalmanCorrect(self.kf, self.measurement)

    def predictInternal(self):

        self.updateProcessNoiseCov()

        prediction = cv.KalmanPredict(self.kf)
        return tuple([int(prediction[i, 0]) for i in range(8)])

    # the main code uses a list of points rather than a flat measurement vector
    def observe(self, corners):
    	self.observeInternal([coord for point in corners for coord in point])
    def predict(self):
    	prediction = self.predictInternal()
    	return [(prediction[2*i], prediction[2*i+1]) for i in range(4)]