#!/usr/bin/env python

import cv2
import numpy as np
import math
import itertools
from tracking import OpticalFlowHomographyTracker


class PaintedObject():
    """
    Paint overlays are implemented using the initial bounding rect as
    the mapping for our painted object to the scene. Dense optical flow
    is applied to the corners to get the homography mapping the original
    drawing to its scene position.
    """

    def __init__(self, drawingOverlay, last_gframe):

        # convert red pixels to yellow for the overlay
        self.drawingOverlay = np.zeros_like(drawingOverlay)
        cv2.mixChannels(
            [drawingOverlay],
            [self.drawingOverlay],
            (0, 0, 2, 1, 2, 2)
        )

        self.shape = (drawingOverlay.shape[1], drawingOverlay.shape[0])

        # Grab all pixels of overlay
        xs, ys, zs = np.where(drawingOverlay > 0)
        self.overlayPts = zip(xs, ys)

        # Grab bounding rect
        ptsMat = np.float32(map(lambda x: [x], self.overlayPts))
        y, x, h, w = cv2.boundingRect(ptsMat)
        x1, y1 = x, y
        x2, y2 = x + w, y + h

        # Create dense optical flow tracker
        self.first_flow_pts = np.float32([
            [x1, y1],
            [x2, y1],
            [x2, y2],
            [x1, y2]
        ])
        self.flow_tracker = OpticalFlowHomographyTracker(
            last_gframe,
            self.first_flow_pts
        )

    def track(self, gframe):
        """
        Uses the optical flow tracker on the given frame to determine how to
        warp the drawing. Warps the drawing accordingly and returns the result
        """

        homography = self.flow_tracker.track(gframe)

        if len(np.flatnonzero(homography)) == 0:
            return np.zeros_like(gframe)

        return cv2.warpPerspective(self.drawingOverlay, homography, self.shape)


def drawCorners(frame, corners, color):
    """
    Draws the corners of a tracked rectangluar planar surface denoted by the
    list of corners as (x, y) coordinates in the frames.
    """

    for x, y in corners:
        x, y = int(x), int(y)
        cv2.circle(frame, (x, y), 10, color, 3, 8, 0)


def drawOverlay(frame, init_corners, corners, obj, focal=0.5, scale=0.5,
                draw_style="line_shader"):
    """
    Draws an overlay onto a tracked rectangular planar surface.

    Arguments:
      frame: The frame to draw on.
      init_corners: The corners of a head-on plane representing the initial
                    state, going clockwise.
      corners: The corners of the perspective planar surface in 'frame',
               going clockwise.
      obj: An OBJ structure to be drawn onto the plane.

    """

    (xi0, yi0), (xi1, yi1), (xi2, yi2), (xi3, yi3) = init_corners
    (x0, y0), (x1, y1), (x2, y2), (x3, y3) = corners

    quad_3d = np.float32([[xi0, yi0, 0], [xi1, yi1, 0],
                          [xi2, yi2, 0], [xi3, yi3, 0]])

    quad_2d = np.float32(corners)

    # Calculate the mapping from head-on plane estimation to
    # perspective plane
    h, w = frame.shape[:2]
    H = np.float32([[focal*w, 0, 0.5*(w-1)],
                    [0, focal*w, 0.5*(h-1)],
                    [0, 0, 1]])

    distort_coeff = np.zeros(4)
    _, rot, trans = cv2.solvePnP(quad_3d, quad_2d, H, distort_coeff)

    # Project object points onto the current frame
    surface_w, surface_h = xi1 - xi0, yi3 - yi0

    # Map normalized verts onto surface plane
    verts = np.float32(obj.vertices)
    scale = min(surface_w, surface_h) * scale
    scaled_verts = verts * [scale, scale, 0.5 * scale]
    surface_verts = [xi0 + (surface_w - scale), yi0 + (surface_h - scale), 0]
    obj_verts = scaled_verts + surface_verts
    mapped_verts = cv2.projectPoints(
        obj_verts,
        rot,
        trans,
        H,
        distort_coeff
    )[0].reshape(-1, 2)

    # Draw object faces
    def faceToVert(idx):
        x, y = mapped_verts[idx - 1]
        if math.isnan(x) or math.isnan(y):
                return None
        return (int(x), int(y))

    for face_obj in obj.faces:
        vert_ids, saturation = face_obj[0], face_obj[4]
        v = np.array(map(faceToVert, vert_ids))

        if all(vert is not None for vert in v):
            if draw_style == "face_shader":
                cv2.fillConvexPoly(frame, v, (0, saturation * 255, 0))
            else:
                if draw_style == "line":
                    saturation = 1

                for idx in xrange(len(v)):
                    color = (0, saturation * 255, 0)
                    src = tuple(v[idx])
                    dst = tuple(v[(idx+1) % len(v)])
                    cv2.line(frame, src, dst, color)
