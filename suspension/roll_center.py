"""
Front-view instant center and roll center calculations.

The instant center (IC) is the point where the upper and lower A-arm lines
intersect when viewed from the front of the car. It's the instantaneous
center of rotation of the wheel relative to the chassis.

The roll center (RC) is where the line from the contact patch through the
instant center crosses the vehicle centerline (y=0). The roll center height
determines what fraction of lateral load transfer goes through the springs
(elastic transfer) vs. through the linkage geometry (geometric transfer).

Key relationships (Carroll Smith, "Tune to Win"):
    - Low RC → more body roll, more elastic transfer, gentler tire loading
    - High RC → less body roll, more geometric transfer, snappier response
    - RC that moves a lot through travel → inconsistent handling
    - IC that moves toward the wheel → short FVSA → fast camber gain
    - IC that moves away from wheel → long FVSA → slow camber gain

FVSA = front-view swing arm length = horizontal distance from contact patch
to the instant center. This controls the camber change rate.

Conventions:
    Everything is computed for ONE corner (typically left/driver side).
    The roll center is found by mirroring this corner's IC line to the
    opposite side and finding where it crosses centerline y=0.
    For a symmetric car, both sides produce the same RC, so one corner
    is sufficient.
"""

import numpy as np
from typing import Optional


def front_view_instant_center(
    upper_pivot_center: np.ndarray,
    upper_ball_joint: np.ndarray,
    lower_pivot_center: np.ndarray,
    lower_ball_joint: np.ndarray,
) -> Optional[np.ndarray]:
    """
    Find the front-view instant center by intersecting the upper and lower
    A-arm lines in the y-z plane.

    Parameters
    ----------
    upper_pivot_center : ndarray
        Midpoint of the upper A-arm inboard pivot axis.
    upper_ball_joint : ndarray
        Current position of the upper outboard ball joint.
    lower_pivot_center : ndarray
        Midpoint of the lower A-arm inboard pivot axis.
    lower_ball_joint : ndarray
        Current position of the lower outboard ball joint.

    Returns
    -------
    ndarray, shape (2,) — [y, z] coordinates of the instant center, or
    None if the lines are parallel (infinite IC = zero camber change).
    """
    # Project everything into the y-z plane (front view)
    # Upper arm line: from upper_pivot_center to upper_ball_joint
    uy1, uz1 = upper_pivot_center[1], upper_pivot_center[2]
    uy2, uz2 = upper_ball_joint[1], upper_ball_joint[2]

    # Lower arm line: from lower_pivot_center to lower_ball_joint
    ly1, lz1 = lower_pivot_center[1], lower_pivot_center[2]
    ly2, lz2 = lower_ball_joint[1], lower_ball_joint[2]

    # Solve the 2x2 system for the intersection
    # Line 1: P = (uy1, uz1) + t * (uy2-uy1, uz2-uz1)
    # Line 2: P = (ly1, lz1) + s * (ly2-ly1, lz2-lz1)
    dy_upper = uy2 - uy1
    dz_upper = uz2 - uz1
    dy_lower = ly2 - ly1
    dz_lower = lz2 - lz1

    det = dy_upper * dz_lower - dz_upper * dy_lower

    if abs(det) < 1e-12:
        # Lines are parallel — IC is at infinity (equal-length parallel arms)
        return None

    # Cramer's rule
    t = ((ly1 - uy1) * dz_lower - (lz1 - uz1) * dy_lower) / det

    ic_y = uy1 + t * dy_upper
    ic_z = uz1 + t * dz_upper

    return np.array([ic_y, ic_z])


def roll_center_height(
    instant_center_yz: np.ndarray,
    contact_patch: np.ndarray,
) -> float:
    """
    Roll center height from one corner's instant center and contact patch.

    The roll center is where the line from the contact patch through the
    instant center crosses the vehicle centerline (y=0).

    For a LEFT (driver-side) corner, the contact patch has y < 0, and the IC
    is also at y < 0 (typically). The line slopes up toward centerline.

    Parameters
    ----------
    instant_center_yz : ndarray, shape (2,) — [y, z] of the IC
    contact_patch : ndarray, shape (3,) — full 3D contact patch position

    Returns
    -------
    float — z-coordinate of the roll center (height above ground), meters.
    """
    cp_y = contact_patch[1]
    cp_z = contact_patch[2]

    ic_y = instant_center_yz[0]
    ic_z = instant_center_yz[1]

    dy = ic_y - cp_y
    if abs(dy) < 1e-12:
        # IC is directly above/below contact patch — RC height = IC height
        return ic_z

    dz = ic_z - cp_z

    # Parametric: at y=0, t = (0 - cp_y) / (ic_y - cp_y)
    t = -cp_y / dy
    rc_z = cp_z + t * dz

    return rc_z


def front_view_swing_arm_length(
    instant_center_yz: np.ndarray,
    contact_patch: np.ndarray,
) -> float:
    """
    Front-view swing arm length (FVSA).

    This is the horizontal distance from the contact patch center to the
    instant center. It controls the rate of camber change:
        camber_gain ≈ 1 / FVSA  (in rad/m of travel)

    Short FVSA (IC close to wheel) = aggressive camber gain.
    Long FVSA (IC far inboard) = gentle camber gain.

    Returns
    -------
    float — FVSA in meters. Positive means IC is inboard of the contact
    patch (normal). Negative would mean the IC crossed to the wrong side
    (bad geometry).
    """
    cp_y = contact_patch[1]
    ic_y = instant_center_yz[0]

    # For a left corner (cp_y < 0), IC inboard means ic_y > cp_y (closer to 0)
    # FVSA = |ic_y - cp_y|, but we preserve sign: positive = IC is inboard
    # For left corner: inboard = toward +y, so (ic_y - cp_y) > 0 is normal
    return ic_y - cp_y


def compute_roll_geometry_sweep(hp, results_list, angles_list):
    """
    Compute IC, roll center, and FVSA for every position in a sweep.

    Parameters
    ----------
    hp : DoubleWishboneHardpoints
        Static hardpoint definitions.
    results_list : list of dict
        Output from solve_corner() at each sweep position.
    angles_list : array-like
        The lower arm angles used in the sweep (for reference).

    Returns
    -------
    dict with keys:
        'ic_y'   : ndarray — instant center y-coordinate (meters)
        'ic_z'   : ndarray — instant center z-coordinate (meters)
        'rc_z'   : ndarray — roll center height (meters)
        'fvsa'   : ndarray — front-view swing arm length (meters)
        'valid'  : ndarray of bool — whether IC exists (non-parallel arms)
    """
    n = len(results_list)
    ic_y = np.full(n, np.nan)
    ic_z = np.full(n, np.nan)
    rc_z = np.full(n, np.nan)
    fvsa = np.full(n, np.nan)
    valid = np.zeros(n, dtype=bool)

    upper_pc = hp.upper_pivot_center()
    lower_pc = hp.lower_pivot_center()

    for i, result in enumerate(results_list):
        ic = front_view_instant_center(
            upper_pc,
            result['upper_ball_joint'],
            lower_pc,
            result['lower_ball_joint'],
        )
        if ic is not None:
            ic_y[i] = ic[0]
            ic_z[i] = ic[1]
            rc_z[i] = roll_center_height(ic, result['contact_patch'])
            fvsa[i] = front_view_swing_arm_length(ic, result['contact_patch'])
            valid[i] = True

    return {
        'ic_y': ic_y,
        'ic_z': ic_z,
        'rc_z': rc_z,
        'fvsa': fvsa,
        'valid': valid,
    }
