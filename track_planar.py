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
recreateFlowTracker = False

def framesFromVideo(video):
    while True:
        ret, frame = video.read()
        if not ret:
            break
        yield frame

def outputFilename(inputFilename):
    if inputFilename == 0:
        return "test.avi"

    dot = inputFilename.rfind(".")
    return "%s.out.%s" % (inputFilename[:dot], inputFilename[dot+1:])

def applyHomography(homography, (x, y)):
    trans = np.dot(homography, [x, y, 1])
    return trans[:2] / trans[2]

def null_callback(x):
    pass

def paint_mouse(event, x, y, flags, param):
    global drawing, drawingOverlay, recreateFlowTracker

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        recreateFlowTracker = True

    if drawing:
        cv2.circle(drawingOverlay, (x,y), 3, (0,255,255), -1)

def main():
    global drawing, drawingOverlay, recreateFlowTracker

    overlayTest = False

    parser = OptionParser(usage="usage: %prog [options] [video] [trainingFrame] (video and trainingFrame required if not streaming)")
    parser.add_option("-o", "--object", metavar="FILE", dest="obj", help="the 3D OBJ file to overlay")
    parser.add_option("-c", "--corners", dest="corners", action="store_true", help="show the corners of the tracked planar surface")
    parser.add_option("-s", "--stream", dest="stream", action="store_true", help="stream live video and auto-detect planar surfaces")
    parser.add_option("-n", "--no-write", dest="nowrite", action="store_true", help="skip writing video file (for systems that don't support it)")
    parser.add_option("-k", "--kalman", dest="kalman", action="store_true", help="use a Kalman Filter to smooth predicted corners")
    parser.add_option("-f", "--costfunc", dest="costMode", default="rect", help="which cost function to use to evaluate contours")
    parser.add_option("-m", "--merge-contours", dest="mergeMode", action="store_true", help="if on, attempt to merge pairs of contours that become more rectangular when merged")
    parser.add_option("-t", "--tracker", dest="trackMode", default="features", help="which tracker to use for corner tracking [features, flow, pointFlow]")
    parser.add_option("-l", "--stall", dest="stall", action="store_true", help="Stall video on each frame when not tracking")

    options, args = parser.parse_args()

    videoSource = None
    detector = None
    tracker = None
    writer = None
    plane = None
    corners = None
    contour = None

    # Paint variables
    flow_tracker = None
    first_flow_pts = []

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

        # Use the dimensions of the first frame to initialize the video writer
        if writer is None and not options.nowrite:
            dim = tuple(frame.shape[:2][::-1])
            writer = cv2.VideoWriter(outputFilename(videoSource), codec, 15.0, dim)
            print "initializing writer with dimensions %d x %d" % dim

        if options.stream:
            if not plane:
                # Detect a plane
                kernel = 2 * cv2.getTrackbarPos("Gaussian Kernel", 'Stream Options') + 1
                contour, corners = detector.detect(frame, gaussian_kernel=(kernel, kernel))
            else:
                # Track current plane
                homography = None
                if options.trackMode == 'pointFlow':
                    # Track and update corners directly
                    corners = tracker.track(gframe)
                else:
                    # Map initialization corners with new homography
                    if options.trackMode == 'flow':
                        homography = tracker.track(gframe)
                    else:
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

        # After drawing, overwrite corners with kalman corners if using kalman filter
        if options.kalman:
            corners = kalmanCorners

        # Draw 3D object overlay
        if plane and overlay is not None:
            graphics.drawOverlay(frame, plane.planarized_corner_map, corners, overlay)

        # write the frame number in the corner so the video can be matched to command line output
        textCoords = frame.shape[1]-100, frame.shape[0]-40
        cv2.putText(frame, str(frameIndex), textCoords, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

        # Draw paint overlay
        f = np.zeros_like(frame)

        """
        The paint overlay is implemented using the initial bounding rect as the mapping for our painted object to the scene.
        Dense optical flow is applied to the corners to get the homography mapping the original drawing to its scene position.
        """
        if recreateFlowTracker:

            # Grab all pixels of overlay
            xs, ys, zs = np.where(drawingOverlay > 0)
            overlayPts = zip(xs, ys)
            
            # Grab bounding rect
            y, x, h, w = cv2.boundingRect(np.float32(map(lambda x: [x], overlayPts)))
            x1, y1, x2, y2 = x, y, x + w, y + h

            # Create dense optical flow tracker
            first_flow_pts = np.float32([[x1,y1], [x2,y1], [x2,y2], [x1,y2]])
            flow_tracker = OpticalFlowHomographyTracker(last_gframe, first_flow_pts)
            recreateFlowTracker = False
        
        nextOverlay = drawingOverlay

        # Map original drawing to scene position
        if flow_tracker and not drawing:
            homography = flow_tracker.track(gframe)
            if len(np.flatnonzero(homography)) > 0:
                nextOverlay = cv2.warpPerspective(drawingOverlay, homography, (drawingOverlay.shape[1], drawingOverlay.shape[0]))

        # Draw overlay onto new canvas 'f'
        f += nextOverlay
        for c in range(0,3):
            f[:,:, c] += frame[:,:, c] * (1 - nextOverlay[:,:,2]/255.0)

        if not options.nowrite:
            writer.write(f)

        last_gframe = gframe

        cv2.imshow("frame", f)

        if not plane and options.stall:
            key = cv2.waitKey(0) & 0xFF
        else:
            key = cv2.waitKey(1) & 0xFF

        if key == ord('t'):
            if plane:
                # Remove tracking plane
                plane = None
            else:
                # Track highlighted plane
                if options.trackMode == 'flow':
                    # Track all points in contour, using initial brect to map models
                    x, y, w, h = cv2.boundingRect(contour)
                    x1, y1, x2, y2 = x, y, x + w, y + h
                    contour = map(lambda x: x[0], contour)
                    c = 10  # Number of pixels to condense rect by to avoid tracking on boundary
                    plane = TrackedPlane(np.float32([[x1+c,y1+c], [x2-c,y1+c], [x2-c,y2-c], [x1+c,y2-c]]), contour)
                    tracker = OpticalFlowHomographyTracker(gframe, contour)

                elif options.trackMode == 'pointFlow':
                    # Track using the four approximated corners under sparse optical flow
                    plane = TrackedPlane(corners, corners)
                    tracker = OpticalFlowPointTracker(gframe, corners)

                else:
                    # Track a generic 4-corner rectangular plane
                    plane = TrackedPlane(corners, corners)
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
