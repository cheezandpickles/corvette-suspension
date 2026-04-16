"""
Front-view kinematic solver for a double-wishbone suspension corner.

Given a set of hardpoints and a wheel-travel input, solve for the positions
and orientations of all suspension components.

Key insight: this is a one-DOF four-bar linkage in the front view. We
parameterize by the lower A-arm angle (equivalent to ride height), then
solve a nonlinear constraint to place the upper ball joint such that the
upright length is preserved.
"""

import numpy as np
from scipy.optimize import brentq

from .geometry import rotate_about_axis, distance
from .hardpoints import DoubleWishboneHardpoints


def solve_corner(hp: DoubleWishboneHardpoints, lower_arm_angle: float):
    """
    Solve a double-wishbone corner given a lower A-arm angle.

    Parameters
    ----------
    hp : DoubleWishboneHardpoints
        Static hardpoint definitions.
    lower_arm_angle : float
        Rotation of the lower A-arm about its inboard pivot axis, radians.
        Positive = bump (wheel moves up relative to chassis).
        Zero = static ride height.

    Returns
    -------
    dict with keys:
        'lower_ball_joint': ndarray, new position
        'upper_ball_joint': ndarray, new position
        'upper_arm_angle' : float, radians
        'wheel_center'    : ndarray, rotated with the upright
        'contact_patch'   : ndarray, rotated with the upright
        'kingpin_axis'    : ndarray, unit vector from lower to upper BJ
    """
    # 1. Rotate the lower ball joint about the lower inboard pivot axis.
    new_lower_bj = rotate_about_axis(
        hp.lower_ball_joint,
        hp.lower_front_pivot,      # any point on the axis works
        hp.lower_pivot_axis(),
        lower_arm_angle,
    )

    # 2. The upper ball joint lies on a circle (upper A-arm arc) AND must be
    #    at distance = upright_length from the new lower ball joint.
    #    Solve for the upper arm angle that satisfies this constraint.
    target_length = hp.upright_length()

    def length_error(upper_angle):
        candidate_upper = rotate_about_axis(
            hp.upper_ball_joint,
            hp.upper_front_pivot,
            hp.upper_pivot_axis(),
            upper_angle,
        )
        return distance(candidate_upper, new_lower_bj) - target_length

    # Bracket the solution: the static position has error = 0 at upper_angle = 0.
    # As the lower arm moves, the upper needs to move in the same direction
    # (both in bump, both in rebound) but by a slightly different amount.
    # Search in a generous range around the lower arm's angle.
    search_range = abs(lower_arm_angle) + np.deg2rad(5)
    lo = lower_arm_angle - search_range
    hi = lower_arm_angle + search_range

    # Brent's method: robust 1D root finder, needs a sign change in the bracket.
    upper_angle = brentq(length_error, lo, hi, xtol=1e-8)

    new_upper_bj = rotate_about_axis(
        hp.upper_ball_joint,
        hp.upper_front_pivot,
        hp.upper_pivot_axis(),
        upper_angle,
    )

    # 3. Transform the wheel center and contact patch with the upright.
    new_wc, new_cp = _transform_with_upright(
        hp, new_lower_bj, new_upper_bj,
    )

    kingpin = new_upper_bj - new_lower_bj
    kingpin_unit = kingpin / np.linalg.norm(kingpin)

    return {
        'lower_ball_joint': new_lower_bj,
        'upper_ball_joint': new_upper_bj,
        'upper_arm_angle' : upper_angle,
        'wheel_center'    : new_wc,
        'contact_patch'   : new_cp,
        'kingpin_axis'    : kingpin_unit,
    }


def _transform_with_upright(hp, new_lower_bj, new_upper_bj):
    """
    Given new ball joint positions, transform wheel center and contact patch.

    Treats the upright as a rigid body rotating about the lower ball joint such
    that the kingpin axis goes from its static orientation to the new one.
    Minimal 1-rotation transform (no spin of upright about its own axis yet —
    that's what steering adds later).
    """
    old_kingpin = hp.upper_ball_joint - hp.lower_ball_joint
    new_kingpin = new_upper_bj - new_lower_bj

    old_unit = old_kingpin / np.linalg.norm(old_kingpin)
    new_unit = new_kingpin / np.linalg.norm(new_kingpin)

    # Axis of rotation: perpendicular to both old and new kingpin directions.
    # Angle: between them.
    rot_axis = np.cross(old_unit, new_unit)
    axis_norm = np.linalg.norm(rot_axis)

    if axis_norm < 1e-12:
        # No rotation needed (pure translation of upright).
        translation = new_lower_bj - hp.lower_ball_joint
        return (hp.wheel_center + translation,
                hp.contact_patch + translation)

    rot_axis = rot_axis / axis_norm
    rot_angle = np.arcsin(np.clip(axis_norm, -1.0, 1.0))

    def transform(point):
        local = point - hp.lower_ball_joint
        rotated = rotate_about_axis(local, np.zeros(3), rot_axis, rot_angle)
        return rotated + new_lower_bj

    return transform(hp.wheel_center), transform(hp.contact_patch)


def compute_camber(wheel_center, contact_patch):
    """
    Camber angle in degrees from wheel center and contact patch positions.

    Camber is the angle between the wheel plane's vertical axis and true
    vertical, measured in the front view (y-z plane). Negative camber = top
    of tire leans toward car centerline (standard race setup).

    For the LEFT wheel: top leaning right (toward centerline, +y) = negative
    camber. For the RIGHT wheel the sign would flip; we handle that in a
    mirror helper later.
    """
    vec = wheel_center - contact_patch
    yz = np.array([vec[1], vec[2]])
    yz_unit = yz / np.linalg.norm(yz)

    # Signed angle from vertical (+z axis), measured in the y-z plane.
    angle_rad = np.arctan2(yz_unit[0], yz_unit[1])

    # Left wheel convention: positive atan2 = top leans toward centerline
    # = negative camber.
    return -np.rad2deg(angle_rad)


def wheel_travel(hp, result):
    """Vertical travel of the wheel center, meters. Positive = bump."""
    return result['wheel_center'][2] - hp.wheel_center[2]