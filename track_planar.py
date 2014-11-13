#!/usr/bin/env python

import sys
import cv2
import numpy as np

from detection import *
import graphics

def framesFromVideo(video):
	while True:
		ret, frame = video.read()
		if not ret:
			break
		yield frame

def outputFilename(inputFilename):
	dot = inputFilename.rfind(".")
	return "%s.out.%s" % (inputFilename[:dot], inputFilename[dot+1:])

def applyHomography(homography, (x, y)):
	trans = np.dot(homography, [x, y, 1])
	return trans[:2] / trans[2]

def main():
	if len(sys.argv) < 3:
		print "Usage: python track_planar.py <video> <trainingFrame>"
		return
	videoFilename, trainingFrameFilename = sys.argv[1:]

	video = cv2.VideoCapture(videoFilename)
	codec = cv2.cv.CV_FOURCC(*"mp4v")
	writer = None
	trainingFrame = cv2.imread(trainingFrameFilename)
	detector = PlaneDetector(trainingFrame)

	for frameIndex, frame in enumerate(framesFromVideo(video)):
		
		print "processing frame %d" % frameIndex

		# need the dimensions of the first frame to initialize the video writer
		if writer is None:
			dim = tuple(frame.shape[:2][::-1])
			writer = cv2.VideoWriter(outputFilename(videoFilename), codec, 15.0, dim)
			print "initializing writer with dimensions %d x %d" % dim

		homography = detector.detect(frame)
		
		def getCorners(image):
			h, w = image.shape[:2]
			for x in (0, w-1):
				for y in (0, h-1):
					yield (x, y)
		corners = [applyHomography(homography, point) for point in getCorners(trainingFrame)]

		graphics.drawCorners(frame, corners)

		writer.write(frame)
		cv2.imshow("frame", frame)
		
		if cv2.waitKey(1) & 0xFF == ord('q'):
			print "quitting early!"
			break

	video.release()
	video = None
	writer.release()
	writer = None
	cv2.destroyAllWindows()

if __name__ == '__main__':
	main()