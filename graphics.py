#!/usr/bin/env python

import cv2
import numpy as np
import math

def drawCorners(frame, corners, color):
    """
    Draws the corners of a tracked rectangluar planar surface denoted by the
    list of corners as (x, y) coordinates in the frames. Returns the
    frame (though there is no promise of not modifying the frame passed in).
    """

    for x, y in corners:
        x, y = int(x), int(y)
        cv2.circle(frame, (x, y), 10, color, 3, 8, 0)

    pass


def drawOverlay(frame, init_corners, corners, obj, focal=0.5):
        """
        Draws an overlay onto a tracked rectangular planar surface.

        Arguments:
          frame: The frame to draw on.
          init_corners: The corners of a head-on plane representing the initial state, going clockwise.
          corners: The corners of the perspective planar surface in 'frame', going clockwise.
          obj: An OBJ structure to be drawn onto the plane.

        """
        (xi0, yi0), (xi1, yi1), (xi2, yi2), (xi3, yi3) = init_corners
        (x0, y0), (x1, y1), (x2, y2), (x3, y3) = corners

        quad_3d = np.float32([[xi0, yi0, 0], [xi1, yi1, 0],
                              [xi2, yi2, 0], [xi3, yi3, 0]])
        
        quad_2d = np.float32(corners)
        
        # Calculate the mapping from head-on plane estimation to perspective plane
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
        scale = max(surface_w, surface_h)
        obj_verts = verts * [scale, scale, 0.5 * scale] + [xi0, yi0, 0]
        mapped_verts = cv2.projectPoints(obj_verts, rot, trans, H, distort_coeff)[0].reshape(-1, 2)
        
        # Draw object faces
        def faceToVert(idx):
                x, y = mapped_verts[idx - 1]
                if math.isnan(x) or math.isnan(y):
                        return None
                return (int(x), int(y))

        for face_obj in obj.faces:
                vert_ids = face_obj[0]
                v = map(faceToVert, vert_ids)

                if all(vert is not None for vert in v):
                        cv2.line(frame, v[0], v[1], (0, 255, 0))
                        cv2.line(frame, v[0], v[2], (0, 255, 0))
                        cv2.line(frame, v[1], v[2], (0, 255, 0))
