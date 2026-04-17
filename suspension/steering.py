"""
Steering geometry and bump steer analysis.

Bump steer occurs when the tie rod and the lower A-arm don't arc together
through suspension travel. If the tie rod's arc doesn't match the lower
arm's arc, the wheel steers itself as it goes through bump and rebound.
This is the single biggest reason street-converted race cars feel twitchy
over bumps.

The fix: place the tie rod inner pivot so that it lies on (or very close to)
the lower A-arm's instant axis. In practice, this means the tie rod inner
pickup point must be at a specific height and fore-aft location relative
to the lower arm pivots.

This module adds:
    - Tie rod as an additional constraint
    - Bump steer calculation (toe change vs. wheel travel)
    - Steering ratio and Ackermann geometry (basic)

Conventions:
    Toe-out = positive (front of wheel points away from centerline)
    Toe-in = negative
    For left wheel: toe-out means the wheel points toward -y (further left)
"""

import numpy as np
from typing import Optional

from .geometry import rotate_about_axis, distance


def compute_bump_steer(
    hp,
    tie_rod_inner: np.ndarray,
    tie_rod_outer: np.ndarray,
    lower_arm_angles: np.ndarray,
    solve_corner_func,
):
    """
    Compute bump steer (toe change) through a sweep of wheel travel.

    The tie rod is treated as a rigid link. As the suspension moves, the
    outer tie rod end moves with the upright. We check whether the tie rod
    length can be maintained — if not, the upright must rotate about the
    kingpin axis to accommodate, which is bump steer.

    Simplified approach: compute the toe angle at each travel position by
    checking where the tie rod outer end WOULD be (moving with the upright)
    vs. where it NEEDS to be (on the arc of the tie rod inner pivot at
    the correct length). The angular difference is the bump steer.

    Parameters
    ----------
    hp : DoubleWishboneHardpoints
    tie_rod_inner : ndarray, shape (3,) — inner tie rod pivot (on rack)
    tie_rod_outer : ndarray, shape (3,) — outer tie rod end (on upright), static
    lower_arm_angles : ndarray — sweep angles in radians
    solve_corner_func : callable — the solve_corner function

    Returns
    -------
    dict with:
        'toe_change_deg' : ndarray — toe change from static, degrees
        'travel_mm'      : ndarray — wheel travel, mm
        'tie_rod_length_error_mm' : ndarray — how much the tie rod is
            too long (+) or too short (-) at each position, mm
    """
    static_tie_rod_length = distance(tie_rod_inner, tie_rod_outer)

    n = len(lower_arm_angles)
    toe_change = np.zeros(n)
    travel_mm = np.zeros(n)
    length_error = np.zeros(n)

    # Static positions for reference
    static_lower_bj = hp.lower_ball_joint.copy()
    static_upper_bj = hp.upper_ball_joint.copy()

    for i, angle in enumerate(lower_arm_angles):
        result = solve_corner_func(hp, angle)

        # Where the upright moved the outer tie rod end to
        # (same rigid body transform as wheel center / contact patch)
        moved_outer = _transform_point_with_upright(
            tie_rod_outer, hp, result
        )

        # Actual distance from inner pivot to moved outer point
        actual_dist = distance(tie_rod_inner, moved_outer)
        length_error[i] = (actual_dist - static_tie_rod_length) * 1000  # mm

        # The toe change is approximately the arc length error divided by
        # the tie rod's moment arm about the kingpin axis
        # Simplified: use the y-z plane projection
        kingpin = result['kingpin_axis']

        # Moment arm of tie rod about kingpin (approximate)
        tr_vec = moved_outer - result['lower_ball_joint']
        moment_arm = np.linalg.norm(np.cross(tr_vec, kingpin))

        if moment_arm > 1e-6:
            # Angular error in radians
            toe_change[i] = np.rad2deg(
                (actual_dist - static_tie_rod_length) / moment_arm
            )

        travel_mm[i] = (result['wheel_center'][2] - hp.wheel_center[2]) * 1000

    # Zero-reference to static position
    mid = n // 2
    toe_change -= toe_change[mid]

    return {
        'toe_change_deg': toe_change,
        'travel_mm': travel_mm,
        'tie_rod_length_error_mm': length_error,
    }


def _transform_point_with_upright(point, hp, result):
    """
    Transform an arbitrary point attached to the upright using the same
    rigid body rotation that moved the ball joints.

    Uses the rotation from old kingpin direction to new kingpin direction,
    pivoting about the lower ball joint (same as kinematics_front).
    """
    old_kingpin = hp.upper_ball_joint - hp.lower_ball_joint
    new_kingpin = result['upper_ball_joint'] - result['lower_ball_joint']

    old_unit = old_kingpin / np.linalg.norm(old_kingpin)
    new_unit = new_kingpin / np.linalg.norm(new_kingpin)

    rot_axis = np.cross(old_unit, new_unit)
    axis_norm = np.linalg.norm(rot_axis)

    if axis_norm < 1e-12:
        translation = result['lower_ball_joint'] - hp.lower_ball_joint
        return point + translation

    rot_axis = rot_axis / axis_norm
    rot_angle = np.arcsin(np.clip(axis_norm, -1.0, 1.0))

    local = point - hp.lower_ball_joint
    rotated = rotate_about_axis(local, np.zeros(3), rot_axis, rot_angle)
    return rotated + result['lower_ball_joint']


def ideal_tie_rod_inner_height(hp):
    """
    Compute the ideal inner tie rod pivot height for zero bump steer.

    For zero bump steer, the tie rod inner pivot must lie on the
    instant axis of the lower A-arm (in the simplest approximation).
    This means the inner tie rod height should be such that the tie rod
    is parallel to the lower A-arm in the front view.

    More precisely: the tie rod inner pivot should be at a height where
    the tie rod's arc matches the lower A-arm's arc through travel.
    This happens when the tie rod inner pivot is at the same height as
    the lower A-arm pivot axis, projected to the tie rod's lateral position.

    Returns
    -------
    float — ideal z-coordinate for the inner tie rod pivot, meters
    """
    lower_pc = hp.lower_pivot_center()

    # The lower arm line in front view goes from lower_pivot_center to
    # lower_ball_joint. The tie rod inner pivot should be on this line
    # (or its extension) at whatever y-position the rack is at.

    # Slope of lower arm in y-z plane
    dy = hp.lower_ball_joint[1] - lower_pc[1]
    dz = hp.lower_ball_joint[2] - lower_pc[2]

    if abs(dy) < 1e-10:
        return lower_pc[2]

    slope_yz = dz / dy
    return lower_pc[2], slope_yz


def example_steering_hardpoints(hp):
    """
    Generate example tie rod pickup points for a front-steer rack setup.

    Front-steer means the rack is ahead of the front axle centerline.
    The inner pivot is on the rack; the outer pivot is on the steering arm
    (part of the upright).

    These are rough starting points — bump steer optimization will adjust
    the inner pivot height.
    """
    # Rack position: ahead of axle, between the frame rails
    # Inner tie rod pivot
    tie_rod_inner = np.array([0.120, -0.300, 0.170])

    # Outer tie rod end: on the steering arm, slightly ahead of and
    # below the lower ball joint
    tie_rod_outer = np.array([0.100, -0.680, 0.140])

    return tie_rod_inner, tie_rod_outer
