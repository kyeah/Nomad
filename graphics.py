#!/usr/bin/env python

import cv2

def drawCorners(frame, corners):
	"""
	Draws the corners of a tracked rectangluar planar surface denoted by the
	list of corners as (x, y) coordinates in the frames. Returns the
	frame (though there is no promise of not modifying the frame passed in).
	"""

	for x, y in corners:
		x, y = int(x), int(y)
		cv2.circle(frame, (x, y), 10, (0, 0, 255), 3, 8, 0)

	pass


def drawOverlay(frame, init_corners, corners, obj, focal=0.5):
        """
        Draws an overlay onto a tracked rectangular planar surface.

        Arguments:
          frame: The frame to draw on.
          init_corners: The initial corners of the planar surface.
          corners: The corners of the surface in 'frame'.
          obj: The object to draw, defined by an array of 3D coordinate tuples.

        """
        (xi0, yi0), (xi1, yi1), (xi2, yi2), (xi3, yi3) = init_corners
        (x0, y0), (x1, y1), (x2, y2), (x3, y3) = corners

        quad_3d = np.float32([[x0, y0, 0], [x1, y1, 0],
                              [x2, y2, 0], [x3, y3, 0]])
        
        # Calculate the mapping from init corners to corners
        h, w = frame.shape[:2]
        H = np.float32([[focal*w, 0, 0.5*(w-1)],
                        [0, focal*w, 0.5*(h-1)],
                        [0, 0, 1]])
        
        dist_coeff = np.zeros(4)
        _, rot, trans = cv2.solvePnP(quad_3d, init_corners, H, dist_coeff)
        
        # Project object points onto the current frame
        surface_w, surface_h = xi1 - xi0, yi1 - yi0
        obj_verts = obj * [surface_w, surface_h, -0.3 * surface_w]
        mapped_verts = cv2.projectPoints(obj_verts, rot, trans, H, dist_coeff)[0].reshape(-1, 2)
        
        # Draw mapped vertices onto frame as triangular mesh
        #for i in xrange(len(mapped_verts), 0, 3):
        #        (x1, y1), (x2, y2), (x3, y3) = vert[i], vert[i+1], vert[i+2]
        #        cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 0))
        #        cv2.line(frame, (int(x1), int(y1)), (int(x3), int(y3)), (255, 255, 0))
        #        cv2.line(frame, (int(x3), int(y3)), (int(x2), int(y2)), (255, 255, 0))
        for vert in mapped_verts:
                cv2.circle(frame, (int(vert[0]), int(vert[1])), 5, (255, 255, 0))
