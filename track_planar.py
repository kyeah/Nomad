#!/usr/bin/env python

import sys
import cv2
import numpy as np
from optparse import OptionParser

from detection import *
from tracking import *
from obj import OBJ
from filtering import *
import graphics


drawing = False
drawingOverlay = None

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

def null_callback(x):
    pass

def paint_mouse(event, x, y, flags, param):
    global drawing, drawingOverlay

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

    if drawing:
        cv2.circle(drawingOverlay, (x,y), 3, (0,255,255), -1)

def main():
    global drawing, drawingOverlay

    parser = OptionParser(usage="usage: %prog [options] [video] [trainingFrame] (video and trainingFrame required if not streaming)")
    parser.add_option("-o", "--object", metavar="FILE", dest="obj", help="the 3D OBJ file to overlay")
    parser.add_option("-c", "--corners", dest="corners", action="store_true", help="show the corners of the tracked planar surface")
    parser.add_option("-s", "--stream", dest="stream", action="store_true", help="stream live video and auto-detect planar surfaces")
    parser.add_option("-n", "--no-write", dest="nowrite", action="store_true", help="skip writing video file (for systems that don't support it)")
    parser.add_option("-k", "--kalman", dest="kalman", action="store_true", help="use a Kalman Filter to smooth predicted corners")
    parser.add_option("-f", "--costfunc", dest="costMode", default="rect", help="which cost function to use to evaluate contours")
    parser.add_option("-m", "--merge-contours", dest="mergeMode", action="store_true", help="if on, attempt to merge pairs of contours that become more rectangular when merged")
    options, args = parser.parse_args()

    videoSource = None
    detector = None
    tracker = None
    writer = None
    plane = None
    corners = None
    kalman = KalmanFilter(useProgressivePNC=True) if options.kalman else None

    codec = cv2.cv.CV_FOURCC(*"mp4v")
    overlay = OBJ(options.obj) if options.obj is not None else None

    if options.stream:
        videoSource = args[0] if args else 0
        detector = ArbitraryPlaneDetector(costMode=options.costMode, mergeMode=options.mergeMode)
        cv2.namedWindow("Stream Options")
        cv2.createTrackbar("Gaussian Kernel", 'Stream Options', 7, 15, null_callback)

    else:
        if len(args) != 2:
            parser.print_help()
            return

        videoSource, trainingFrameFilename = args
        trainingFrame = cv2.imread(trainingFrameFilename)
        tracker = GenericTracker(trainingFrame)

    video = cv2.VideoCapture(videoSource)
    cv2.namedWindow("frame")
    cv2.setMouseCallback("frame", paint_mouse)

    last_gframe = None

    for frameIndex, frame in enumerate(framesFromVideo(video)):
        
        print "processing frame %d" % frameIndex
        frame_copy = np.array(frame)
        gframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if frameIndex == 0:
            drawingOverlay = np.zeros_like(frame)

        # need the dimensions of the first frame to initialize the video writer
        if writer is None and not options.nowrite:
            dim = tuple(frame.shape[:2][::-1])
            writer = cv2.VideoWriter(outputFilename(videoSource), codec, 15.0, dim)
            print "initializing writer with dimensions %d x %d" % dim

        if options.stream:
            kernel = 2 * cv2.getTrackbarPos("Gaussian Kernel", 'Stream Options') + 1
            if not plane:
                # Detect a plane
                corners = detector.detect(frame, gaussian_kernel=(kernel, kernel))
            else:
                # Track current plane
                homography = tracker.track(frame)
                
                if len(np.flatnonzero(homography)) == 0:
                    print "encountered zero homography! Skipping frame."
                    continue

                corners = [applyHomography(homography, point) for point in plane.init_corners]

        else:
            homography = tracker.track(frame)
        
            if len(np.flatnonzero(homography)) == 0:
                print "encountered zero homography! Skipping frame."
                continue

            def getCorners(image):
                h, w = image.shape[:2]
                for x in (0, w-1):
                    for y in (0, h-1):
                        yield (x, y)

            corners = [applyHomography(homography, point) for point in getCorners(trainingFrame)]

            # Remap to define corners clockwise
            corners = [corners[0], corners[2], corners[3],corners[1]]

        # Use Kalman filter to smooth corners
        if options.kalman:
            kalman.observe(corners)
            kalmanCorners = kalman.predict()

        # Draw tracked corners
        if options.corners:
            graphics.drawCorners(frame, corners, (0, 0, 255))
            if options.kalman:
                graphics.drawCorners(frame, kalmanCorners, (0, 255, 255))

        # after drawing, overwrite corners with kalman corners if using kalman filter
        if options.kalman:
            corners = kalmanCorners

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
        if plane and overlay is not None:
            graphics.drawOverlay(frame, planarized_corners, corners, overlay)

        # write the frame number in the corner so the video can be matched to command line output
        textCoords = frame.shape[1]-100, frame.shape[0]-40
        cv2.putText(frame, str(frameIndex), textCoords, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

        # Draw paint overlay
        f = np.zeros_like(frame)

        xs, ys, zs = np.where(drawingOverlay > 0)
        overlayPts = zip(xs, ys)

        if last_gframe is not None and overlayPts:
            flow_tracker = OpticalFlowTracker(last_gframe, np.float32(overlayPts))  # todo: overlayPts are pts in curr gframe, not last!
            overlayPts = flow_tracker.track(gframe)
            drawingOverlay = np.zeros_like(drawingOverlay)
            for (x,y) in overlayPts:
                try:
                    drawingOverlay[x,y] = [0, 255, 255] 
                except:
                    print "ok"

        f += drawingOverlay
        for c in range(0,3):
            f[:,:, c] += frame[:,:, c] * (1 - drawingOverlay[:,:,2]/255.0)

        if not options.nowrite:
            writer.write(f)

        last_gframe = gframe

        cv2.imshow("frame", f)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('t'):
            if plane:
                # Remove tracking plane
                plane = None
            else:
                # Track highlighted plane
                plane = TrackedPlane(corners)
                tracker = GenericTracker(frame_copy)

        if key == ord('q'):
            print "quitting early!"
            break

    video.release()
    video = None
    if not options.nowrite:
        writer.release()
        writer = None
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
