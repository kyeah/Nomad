import numpy as np


def unit_vector(vector):
    """
    Returns the unit vector of the vector.
    """
    if (len(vector) == 2 and vector[0] + vector[1] == 0):
        print("ZERO VECTOR")
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """
    Returns the angle in radians between vectors 'v1' and 'v2'::
    """

    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    angle = np.arccos(np.dot(v1_u, v2_u))
    if np.isnan(angle):
        if (v1_u == v2_u).all():
            return 0.0
        else:
            return np.pi
    return angle


def test_colinear(triplet, beta=0.52):
    p1, p2, p3 = triplet
    v1, v2 = unit_vector(p2 - p1), unit_vector(p3 - p2)
    angle = angle_between(v1, v2)

    # Angle, Reverse-angle Test
    # If the three lines are roughly colinear, find the
    # longest line and return the middle point.
    if angle < beta or abs(np.pi - angle) < beta:
        distances = map(lambda idx: np.linalg.norm(
            triplet[(idx+1) % 3] - triplet[(idx+2) % 3]), xrange(3))
        return True, triplet[distances.index(max(distances))]

    return False, None


def approx_quadrilateral(corners, alpha):
    """
    Approximate missing corners of quadrilateral. Modifies the input curve.
    """

    if len(corners) == 2:
        # Find dx, dy, and infer head-on plane if the two corners
        # look like they are opposite from each other
        c1, c2 = corners[0], corners[1]
        dx, dy = c2[0] - c1[0], c2[1] - c1[1]
        if abs(dx) > alpha and abs(dy) > alpha:
            if dx < 0:
                c2, c1 = c1, c2
            corners = np.vstack(
                [corners, [[c1[0] + dx, c1[1]], [c1[0], c1[1] + dy]]])

    elif len(corners) == 3:
        # Find hypotenuse, then estimate last corner
        c1, c2, c3 = corners[0], corners[1], corners[2]
        d1, d2, d3 = np.linalg.norm(
            c2 - c1), np.linalg.norm(c3 - c1), np.linalg.norm(c3 - c2)
        # todo: ...finish this

    # Last resorts: if not enough points, replicate;
    # if too many points, truncate.
    while len(corners) < 4:
        corners = np.vstack([corners, corners[len(corners) - 1]])

    corners = corners[:4]
    return corners
