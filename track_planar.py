#!/usr/bin/env python

import sys
import cv2
import numpy as np
from optparse import OptionParser

from detection import *
from obj import OBJ
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

    parser = OptionParser(usage="usage: %prog [options] [video] [trainingFrame] (video and trainingFrame required if not streaming)")
    parser.add_option("-o", "--object", metavar="FILE", dest="obj", help="the 3D OBJ file to overlay")
    parser.add_option("-c", "--corners", dest="corners", action="store_true", help="show the corners of the tracked planar surface")
    parser.add_option("-s", "--stream", dest="stream", action="store_true", help="stream live video and auto-detect planar surfaces")
    options, args = parser.parse_args()

    streaming = options.stream is not None

    videoSource = detector = None
    if streaming:
        videoSource = 0
        detector = ArbitraryPlaneDetector()

    else:
        if len(args) != 2:
            parser.print_help()
            return

        videoSource, trainingFrameFilename = args
        trainingFrame = cv2.imread(trainingFrameFilename)
        detector = PlaneDetector(trainingFrame)

    video = cv2.VideoCapture(videoSource)
    codec = cv2.cv.CV_FOURCC(*"mp4v")
    writer = None

    overlay = OBJ(options.obj) if options.obj is not None else None
    showCorners = options.corners is not None

    for frameIndex, frame in enumerate(framesFromVideo(video)):
        
        print "processing frame %d" % frameIndex

        # need the dimensions of the first frame to initialize the video writer
        if writer is None and not streaming:
            dim = tuple(frame.shape[:2][::-1])
            writer = cv2.VideoWriter(outputFilename(videoSource), codec, 15.0, dim)
            print "initializing writer with dimensions %d x %d" % dim

        corners = None
        if streaming:
            corners = detector.detect(frame)
        else:
            homography = detector.detect(frame)
        
            if len(np.flatnonzero(homography)) == 0:
                print "encountered zero homography! Skipping frame."
                continue

            def getCorners(image):
                h, w = image.shape[:2]
                for x in (0, w-1):
                    for y in (0, h-1):
                        yield (x, y)

            h, w = trainingFrame.shape[:2]
            corners = [applyHomography(homography, point) for point in getCorners(trainingFrame)]

            # Remap to define corners clockwise
            corners = [corners[0], corners[2], corners[3],corners[1]]

        # draw tracked corners
        if options.corners:
            graphics.drawCorners(frame, corners)

        # Planarize corners to estimate head-on plane
        p0, p1, p3 = corners[0], corners[1], corners[3]
        w,h = np.linalg.norm(p1[0] - p0[0]), np.linalg.norm(p3[1] - p0[1])
        
        planarized_corners = np.float32([
            p0,
            (p0[0] + w, p0[1]),
            (p0[0] + w, p0[1] + h),
            (p0[0], p0[1] + h)
        ])
                
        # Draw 3D object overlay
        if overlay is not None:
            graphics.drawOverlay(frame, planarized_corners, corners, overlay)

        if not streaming:
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
