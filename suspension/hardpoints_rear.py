"""
Rear suspension hardpoint definitions for a double-wishbone IRS.

The rear uses the same DoubleWishboneHardpoints dataclass as the front.
The coordinate system is the same SAE J670 frame, but now the hardpoints
are behind the CG (negative x relative to the rear axle origin if you
set up a local frame, but we keep everything in the global vehicle frame).

For the rear double-wishbone:
    - The upper A-arm is typically shorter than the lower (same as front)
    - The pivot axes may be angled differently for anti-squat
    - There's no steering (unless you add rear steer — we won't)
    - The toe link replaces the tie rod, and its geometry matters for
      bump steer just like the front

The halfshaft connects the diff output to the hub but does NOT carry any
suspension loads. It transmits torque only, through CV joints that
accommodate the suspension travel.
"""

import numpy as np
from .hardpoints import DoubleWishboneHardpoints


def example_rear_hardpoints():
    """
    Representative rear-suspension hardpoints for initial testing.

    These are rough ballpark numbers for a C3-width car with a custom
    double-wishbone IRS conversion. NOT measured values.

    Coordinate frame: SAE J670, origin at ground below FRONT axle
    centerline. The rear axle is at x = -wheelbase = -2.489 m.

    These describe the LEFT (driver's side) rear corner, so y is negative.
    """
    # Rear axle x-position (negative = behind front axle)
    x_rear = -2.489  # wheelbase

    return DoubleWishboneHardpoints(
        # Upper A-arm inboard pivots
        # Slightly angled for anti-squat: front pivot higher than rear
        upper_front_pivot=np.array([x_rear + 0.150, -0.280, 0.420]),
        upper_rear_pivot =np.array([x_rear - 0.150, -0.280, 0.400]),

        # Upper ball joint
        upper_ball_joint =np.array([x_rear + 0.000, -0.600, 0.440]),

        # Lower A-arm inboard pivots
        # Also angled for anti-squat
        lower_front_pivot=np.array([x_rear + 0.200, -0.220, 0.200]),
        lower_rear_pivot =np.array([x_rear - 0.200, -0.220, 0.175]),

        # Lower ball joint
        lower_ball_joint =np.array([x_rear + 0.000, -0.680, 0.140]),

        # Wheel center (center of rear wheel/tire)
        wheel_center     =np.array([x_rear + 0.000, -0.720, 0.320]),

        # Contact patch
        contact_patch    =np.array([x_rear + 0.000, -0.720, 0.000]),
    )


def example_rear_toe_link():
    """
    Example toe link (rear equivalent of a tie rod) hardpoints.

    The toe link controls rear toe. For zero bump steer, it needs to be
    at the correct height and location relative to the lower A-arm,
    same principle as the front tie rod.

    Returns
    -------
    (toe_link_inner, toe_link_outer) — ndarray, shape (3,) each
    """
    x_rear = -2.489

    # Inner pivot: on the subframe/diff housing
    toe_link_inner = np.array([x_rear - 0.180, -0.300, 0.160])

    # Outer pivot: on the upright, behind and below the lower BJ
    toe_link_outer = np.array([x_rear - 0.120, -0.680, 0.130])

    return toe_link_inner, toe_link_outer
