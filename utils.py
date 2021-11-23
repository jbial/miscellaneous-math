"""
Utility file containing helpers functions and stuff
"""


def is_collinear(p, q, r):
    """Checks if three points are collinear by shoelace formula
    """
    return p[0]*(q[1]-r[1]) + q[0]*(r[1]-p[1]) + r[0]*(r[1]-p[1]) == 0


