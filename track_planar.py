#!/usr/bin/env python

import sys
import cv2

from detection import *

def framesFromVideo(video):
	while True:
		ret, frame = video.read()
		if not ret:
			break
		yield frame

def outputFilename(inputFilename):
	dot = inputFilename.rfind(".")
	return "%s.out.%s" % (inputFilename[:dot], inputFilename[dot+1:])

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

		writer.write(frame)
		cv2.imshow("frame", frame)
		cv2.waitKey(1)

	video.release()
	video = None
	writer.release()
	writer = None
	cv2.destroyAllWindows()

if __name__ == '__main__':
	main()