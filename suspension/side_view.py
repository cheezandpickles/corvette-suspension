"""
Side-view geometry: anti-dive, anti-squat, and side-view instant center.

The side-view instant center (SVIC) is found by extending the upper and lower
A-arm pivot axes (viewed from the side, i.e., the x-z plane) until they
intersect. Unlike the front-view IC which uses the projected arm lines,
the side-view IC uses the actual pivot axis inclination angles.

For each A-arm, the pivot axis is defined by the front and rear inboard
pivots. When viewed from the side, this axis has a slope — if the rear
pivot is higher than the front pivot, the axis slopes upward toward the
rear of the car, which generates anti-dive (front) or pro-squat (rear).

Anti-dive % (front, under braking):
    Measures how much the front suspension geometry resists dive.
    = tan(θ_svic) / (h_cg / L_front) * brake_bias_front * 100

Anti-squat % (rear, under acceleration):
    Measures how much the rear geometry resists squat.
    = tan(θ_svic) / (h_cg / L_rear) * 100

Where θ_svic is the angle from the contact patch to the SVIC, h_cg is the
CG height, L_front/L_rear are the distances from the respective axle to CG,
and brake_bias_front is the fraction of braking on the front axle.

Conventions (SAE J670):
    x: forward (positive toward front of car)
    z: up
    Side view = x-z plane

References:
    Milliken & Milliken, "Race Car Vehicle Dynamics", ch. 17
    Carroll Smith, "Tune to Win", ch. 5
"""

import numpy as np
from typing import Optional


def side_view_instant_center(
    front_pivot: np.ndarray,
    rear_pivot: np.ndarray,
    ball_joint: np.ndarray,
) -> tuple:
    """
    Compute the side-view instant center contribution from one A-arm.

    Parameters
    ----------
    front_pivot : ndarray, shape (3,)
    rear_pivot : ndarray, shape (3,)
    ball_joint : ndarray, shape (3,)

    Returns
    -------
    (slope, x_intercept, z_intercept) — the line in x-z space
    """
    dx = rear_pivot[0] - front_pivot[0]
    dz = rear_pivot[2] - front_pivot[2]

    if abs(dx) < 1e-10:
        return (np.inf, front_pivot[0], None)

    slope = dz / dx
    x0 = front_pivot[0]
    z0 = front_pivot[2]

    return (slope, x0, z0)


def find_side_view_ic(
    upper_front_pivot: np.ndarray,
    upper_rear_pivot: np.ndarray,
    upper_ball_joint: np.ndarray,
    lower_front_pivot: np.ndarray,
    lower_rear_pivot: np.ndarray,
    lower_ball_joint: np.ndarray,
) -> Optional[np.ndarray]:
    """
    Find the side-view instant center by intersecting the upper and lower
    pivot axis lines in the x-z plane.

    Returns
    -------
    ndarray, shape (2,) — [x, z] of the side-view IC, or None if parallel.
    """
    ux1, uz1 = upper_front_pivot[0], upper_front_pivot[2]
    ux2, uz2 = upper_rear_pivot[0], upper_rear_pivot[2]

    lx1, lz1 = lower_front_pivot[0], lower_front_pivot[2]
    lx2, lz2 = lower_rear_pivot[0], lower_rear_pivot[2]

    dx_upper = ux2 - ux1
    dz_upper = uz2 - uz1
    dx_lower = lx2 - lx1
    dz_lower = lz2 - lz1

    det = dx_upper * dz_lower - dz_upper * dx_lower
    if abs(det) < 1e-12:
        return None

    t = ((lx1 - ux1) * dz_lower - (lz1 - uz1) * dx_lower) / det

    ic_x = ux1 + t * dx_upper
    ic_z = uz1 + t * dz_upper

    return np.array([ic_x, ic_z])


def anti_dive_percent(
    svic_xz: np.ndarray,
    contact_patch: np.ndarray,
    cg_height: float,
    wheelbase: float,
    front_weight_fraction: float,
    brake_bias_front: float = 0.60,
) -> float:
    """
    Anti-dive percentage for the FRONT suspension.

    Parameters
    ----------
    svic_xz : ndarray, shape (2,) — [x, z] of side-view instant center
    contact_patch : ndarray, shape (3,) — 3D contact patch position
    cg_height : float — CG height above ground, meters
    wheelbase : float — wheelbase, meters
    front_weight_fraction : float — fraction of weight on front axle (0-1)
    brake_bias_front : float — fraction of braking on front axle (0-1)

    Returns
    -------
    float — anti-dive percentage.
    """
    cp_x = contact_patch[0]
    cp_z = contact_patch[2]

    dx = svic_xz[0] - cp_x
    dz = svic_xz[1] - cp_z

    if abs(dx) < 1e-10:
        return 0.0

    tan_theta = dz / abs(dx)

    rear_weight_fraction = 1.0 - front_weight_fraction
    l_front = wheelbase * rear_weight_fraction

    if l_front < 1e-10:
        return 0.0

    tan_ref = (brake_bias_front * cg_height) / l_front

    if abs(tan_ref) < 1e-10:
        return 0.0

    return (tan_theta / tan_ref) * 100.0


def anti_squat_percent(
    svic_xz: np.ndarray,
    contact_patch: np.ndarray,
    cg_height: float,
    wheelbase: float,
    front_weight_fraction: float,
) -> float:
    """
    Anti-squat percentage for the REAR suspension (under acceleration).

    Parameters
    ----------
    svic_xz : ndarray, shape (2,) — [x, z] of rear side-view instant center
    contact_patch : ndarray, shape (3,) — rear contact patch position
    cg_height : float — CG height above ground, meters
    wheelbase : float — wheelbase, meters
    front_weight_fraction : float — fraction of weight on front axle

    Returns
    -------
    float — anti-squat percentage.
    """
    cp_x = contact_patch[0]
    cp_z = contact_patch[2]

    dx = svic_xz[0] - cp_x
    dz = svic_xz[1] - cp_z

    if abs(dx) < 1e-10:
        return 0.0

    tan_theta = dz / abs(dx)

    l_rear = wheelbase * front_weight_fraction

    if l_rear < 1e-10:
        return 0.0

    tan_ref = cg_height / l_rear

    if abs(tan_ref) < 1e-10:
        return 0.0

    return (tan_theta / tan_ref) * 100.0


def compute_side_view_sweep(hp, results_list, vehicle_params):
    """
    Compute side-view geometry for every position in a kinematic sweep.

    Parameters
    ----------
    hp : DoubleWishboneHardpoints
    results_list : list of dict from solve_corner()
    vehicle_params : dict with keys:
        'cg_height', 'wheelbase', 'front_weight_fraction',
        'brake_bias_front' (front only), 'is_front'

    Returns
    -------
    dict with 'svic_x', 'svic_z', 'anti_percent', 'valid'
    """
    n = len(results_list)
    svic_x = np.full(n, np.nan)
    svic_z = np.full(n, np.nan)
    anti_pct = np.full(n, np.nan)
    valid = np.zeros(n, dtype=bool)

    for i, result in enumerate(results_list):
        svic = find_side_view_ic(
            hp.upper_front_pivot, hp.upper_rear_pivot,
            result['upper_ball_joint'],
            hp.lower_front_pivot, hp.lower_rear_pivot,
            result['lower_ball_joint'],
        )

        if svic is not None:
            svic_x[i] = svic[0]
            svic_z[i] = svic[1]
            valid[i] = True

            if vehicle_params.get('is_front', True):
                anti_pct[i] = anti_dive_percent(
                    svic, result['contact_patch'],
                    vehicle_params['cg_height'],
                    vehicle_params['wheelbase'],
                    vehicle_params['front_weight_fraction'],
                    vehicle_params.get('brake_bias_front', 0.60),
                )
            else:
                anti_pct[i] = anti_squat_percent(
                    svic, result['contact_patch'],
                    vehicle_params['cg_height'],
                    vehicle_params['wheelbase'],
                    vehicle_params['front_weight_fraction'],
                )

    return {
        'svic_x': svic_x,
        'svic_z': svic_z,
        'anti_percent': anti_pct,
        'valid': valid,
    }
