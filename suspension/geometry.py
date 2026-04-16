"""
Geometric primitives for suspension kinematics.

Conventions (SAE J670):
    x: forward (toward front of car)
    y: right (passenger side on LHD)
    z: up
    Origin: on ground, below front axle centerline (when referenced globally)

Units: SI throughout (meters, radians). Convert only at display boundaries.
"""

import numpy as np


def rotate_about_axis(point, axis_origin, axis_direction, angle):
    """
    Rotate a point about an arbitrary axis in 3D space.

    Used to swing an A-arm's outboard ball joint around its inboard pivot axis.

    Parameters
    ----------
    point : ndarray, shape (3,)
        The point to rotate.
    axis_origin : ndarray, shape (3,)
        A point on the rotation axis.
    axis_direction : ndarray, shape (3,)
        Direction vector of the axis (will be normalized internally).
    angle : float
        Rotation angle in radians, right-hand rule about axis_direction.

    Returns
    -------
    ndarray, shape (3,)
        The rotated point.
    """
    axis = axis_direction / np.linalg.norm(axis_direction)
    p = point - axis_origin

    # Rodrigues' rotation formula
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    rotated = (p * cos_a
               + np.cross(axis, p) * sin_a
               + axis * np.dot(axis, p) * (1 - cos_a))

    return rotated + axis_origin


def distance(p1, p2):
    """Euclidean distance between two 3D points."""
    return np.linalg.norm(np.asarray(p2) - np.asarray(p1))