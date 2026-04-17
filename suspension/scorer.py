"""
Geometry scorer — grades a set of hardpoints against design targets.

Used in notebook 05 to quickly evaluate whether a geometry is in the
ballpark. Not an optimizer — just a scorecard that turns red/yellow/green
for each metric.

Each metric is scored 0-100:
    100 = dead center of target range
    50  = at the edge of target range
    0   = far outside target range
"""

import numpy as np
from .kinematics_front import solve_corner, compute_camber, wheel_travel
from .roll_center import compute_roll_geometry_sweep
from .side_view import compute_side_view_sweep


def score_geometry(hp, vehicle_params, targets=None):
    """
    Score a set of hardpoints against design targets.

    Parameters
    ----------
    hp : DoubleWishboneHardpoints
    vehicle_params : dict — from VehicleConfig.front_vehicle_params() or rear
    targets : dict, optional — override default targets. Keys:
        'rc_height_mm'     : (min, max)
        'camber_gain'      : (min, max) in deg/mm
        'anti_pct'         : (min, max)
        'fvsa_mm'          : (min, max)
        'rc_migration_mm'  : max change through ±50mm travel

    Returns
    -------
    dict with scores and values for each metric
    """
    if targets is None:
        targets = {
            'rc_height_mm': (25.0, 75.0),
            'camber_gain': (-0.04, -0.015),
            'anti_pct': (15.0, 25.0) if vehicle_params.get('is_front', True)
                        else (15.0, 50.0),
            'fvsa_mm': (1500.0, 3000.0),
            'rc_migration_mm': 30.0,
        }

    # Run kinematic sweep
    angles_rad = np.deg2rad(np.linspace(-8, 8, 65))
    results = []
    camber_list = []
    travel_list = []

    for angle in angles_rad:
        r = solve_corner(hp, angle)
        results.append(r)
        camber_list.append(compute_camber(r['wheel_center'], r['contact_patch']))
        travel_list.append(wheel_travel(hp, r) * 1000)

    camber = np.array(camber_list)
    travel = np.array(travel_list)
    mid = len(results) // 2

    # Roll geometry
    roll_geom = compute_roll_geometry_sweep(hp, results, angles_rad)

    # Side view
    side_geom = compute_side_view_sweep(hp, results, vehicle_params)

    # Extract values
    rc_static_mm = roll_geom['rc_z'][mid] * 1000
    fvsa_static_mm = roll_geom['fvsa'][mid] * 1000

    # Camber gain: slope near static (central difference)
    if mid > 0 and mid < len(travel) - 1:
        dt = travel[mid+1] - travel[mid-1]
        dc = camber[mid+1] - camber[mid-1]
        camber_gain = dc / dt if abs(dt) > 1e-6 else 0.0
    else:
        camber_gain = 0.0

    # Anti-dive/squat at static
    anti_static = side_geom['anti_percent'][mid] if side_geom['valid'][mid] else 0.0

    # RC migration: range of RC height within ±50mm of travel
    in_range = np.abs(travel) <= 50
    if np.any(in_range & roll_geom['valid']):
        rc_in_range = roll_geom['rc_z'][in_range & roll_geom['valid']] * 1000
        rc_migration = rc_in_range.max() - rc_in_range.min()
    else:
        rc_migration = 999.0

    # Score each metric
    scores = {}

    scores['rc_height'] = {
        'value': rc_static_mm,
        'unit': 'mm',
        'target': targets['rc_height_mm'],
        'score': _range_score(rc_static_mm, *targets['rc_height_mm']),
    }

    scores['camber_gain'] = {
        'value': camber_gain,
        'unit': 'deg/mm',
        'target': targets['camber_gain'],
        'score': _range_score(camber_gain, *targets['camber_gain']),
    }

    scores['anti_percent'] = {
        'value': anti_static,
        'unit': '%',
        'target': targets['anti_pct'],
        'score': _range_score(anti_static, *targets['anti_pct']),
    }

    scores['fvsa'] = {
        'value': fvsa_static_mm,
        'unit': 'mm',
        'target': targets['fvsa_mm'],
        'score': _range_score(fvsa_static_mm, *targets['fvsa_mm']),
    }

    scores['rc_migration'] = {
        'value': rc_migration,
        'unit': 'mm',
        'target': ('< ', targets['rc_migration_mm']),
        'score': _max_score(rc_migration, targets['rc_migration_mm']),
    }

    # Overall score (weighted average)
    weights = {
        'rc_height': 1.0,
        'camber_gain': 1.5,  # most important
        'anti_percent': 0.8,
        'fvsa': 0.8,
        'rc_migration': 1.2,
    }

    total_weight = sum(weights.values())
    overall = sum(scores[k]['score'] * weights[k] for k in scores) / total_weight

    return {
        'scores': scores,
        'overall': overall,
        'raw': {
            'camber': camber,
            'travel': travel,
            'roll_geom': roll_geom,
            'side_geom': side_geom,
        },
    }


def _range_score(value, lo, hi):
    """Score 0-100 for a value that should be within [lo, hi]."""
    if lo <= value <= hi:
        # Inside range: score based on how centered
        mid = (lo + hi) / 2.0
        half_range = (hi - lo) / 2.0
        dist_from_center = abs(value - mid)
        return 100.0 * (1.0 - 0.5 * dist_from_center / half_range)
    else:
        # Outside range: score drops off
        if value < lo:
            dist = lo - value
            margin = hi - lo
        else:
            dist = value - hi
            margin = hi - lo

        if margin < 1e-10:
            return 0.0

        return max(0.0, 50.0 * (1.0 - dist / margin))


def _max_score(value, max_val):
    """Score 0-100 for a value that should be below max_val."""
    if value <= max_val:
        return 100.0 * (1.0 - 0.5 * value / max_val)
    else:
        return max(0.0, 50.0 * (1.0 - (value - max_val) / max_val))


def print_scorecard(result):
    """Pretty-print the geometry scorecard."""
    print("\n" + "=" * 55)
    print("  GEOMETRY SCORECARD")
    print("=" * 55)

    for name, data in result['scores'].items():
        score = data['score']
        if score >= 75:
            grade = "GOOD"
        elif score >= 50:
            grade = "OK  "
        else:
            grade = "BAD "

        if isinstance(data['target'], tuple) and len(data['target']) == 2:
            if isinstance(data['target'][0], str):
                tgt_str = f"{data['target'][0]}{data['target'][1]:.1f}"
            else:
                tgt_str = f"{data['target'][0]:.3f} – {data['target'][1]:.3f}"
        else:
            tgt_str = str(data['target'])

        print(f"  [{grade}] {name:18s}  {data['value']:8.3f} {data['unit']:8s}"
              f"  target: {tgt_str}  ({score:.0f}/100)")

    print(f"\n  OVERALL: {result['overall']:.0f}/100")
    print("=" * 55)
