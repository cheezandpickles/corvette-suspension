"""
Hardpoint I/O — read and write hardpoint definitions from/to files.

This module provides the bridge between CAD (Onshape) and the Python sim.
The workflow is:

    1. In Onshape, create a Part Studio with sketch points at each hardpoint.
       Name them consistently (e.g., "upper_front_pivot", "lower_ball_joint").
    2. Export the coordinates to a CSV (manually at first, later via Onshape
       API or FeatureScript).
    3. Load the CSV here → get a DoubleWishboneHardpoints object → run sims.
    4. After tuning in Python, write new coordinates back → update Onshape.

File formats supported:
    - CSV: one row per hardpoint, columns = name, x, y, z
    - JSON: dictionary of hardpoint names → [x, y, z] lists

Units in files: MILLIMETERS (more natural for CAD and human reading).
Units in Python: METERS (SI, used throughout the sim).

The conversion happens at the I/O boundary — you never need to think about
it inside the sim code.
"""

import csv
import json
from pathlib import Path
import numpy as np

from .hardpoints import DoubleWishboneHardpoints


# The canonical names, in the order they appear in the dataclass.
# These must match what you name your sketch points in Onshape.
HARDPOINT_NAMES = [
    'upper_front_pivot',
    'upper_rear_pivot',
    'upper_ball_joint',
    'lower_front_pivot',
    'lower_rear_pivot',
    'lower_ball_joint',
    'wheel_center',
    'contact_patch',
]


def to_dict(hp: DoubleWishboneHardpoints, units='mm') -> dict:
    """
    Convert a DoubleWishboneHardpoints to a dictionary.

    Parameters
    ----------
    hp : DoubleWishboneHardpoints
    units : str, 'mm' or 'm'
        Output units. 'mm' multiplies by 1000 for CAD-friendly values.

    Returns
    -------
    dict — hardpoint name → [x, y, z] list
    """
    scale = 1000.0 if units == 'mm' else 1.0
    fields = [
        hp.upper_front_pivot, hp.upper_rear_pivot, hp.upper_ball_joint,
        hp.lower_front_pivot, hp.lower_rear_pivot, hp.lower_ball_joint,
        hp.wheel_center, hp.contact_patch,
    ]
    return {
        name: (coord * scale).tolist()
        for name, coord in zip(HARDPOINT_NAMES, fields)
    }


def from_dict(d: dict, units='mm') -> DoubleWishboneHardpoints:
    """
    Create a DoubleWishboneHardpoints from a dictionary.

    Parameters
    ----------
    d : dict — hardpoint name → [x, y, z]
    units : str, 'mm' or 'm'
        What units the values in d are in.

    Returns
    -------
    DoubleWishboneHardpoints
    """
    scale = 0.001 if units == 'mm' else 1.0
    arrays = {}
    for name in HARDPOINT_NAMES:
        if name not in d:
            raise KeyError(f"Missing hardpoint: '{name}'. "
                           f"Expected: {HARDPOINT_NAMES}")
        arrays[name] = np.array(d[name], dtype=float) * scale

    return DoubleWishboneHardpoints(**arrays)


def write_csv(hp: DoubleWishboneHardpoints, path: str, units='mm'):
    """
    Write hardpoints to a CSV file.

    Format:
        name,x,y,z
        upper_front_pivot,150.0,-250.0,500.0
        ...

    Parameters
    ----------
    hp : DoubleWishboneHardpoints
    path : str or Path
    units : str, 'mm' or 'm'
    """
    d = to_dict(hp, units=units)
    path = Path(path)
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['name', 'x', 'y', 'z'])
        for name in HARDPOINT_NAMES:
            x, y, z = d[name]
            writer.writerow([name, f'{x:.3f}', f'{y:.3f}', f'{z:.3f}'])


def read_csv(path: str, units='mm') -> DoubleWishboneHardpoints:
    """
    Read hardpoints from a CSV file.

    Parameters
    ----------
    path : str or Path
    units : str — what units the CSV values are in ('mm' or 'm')

    Returns
    -------
    DoubleWishboneHardpoints
    """
    path = Path(path)
    d = {}
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row['name'].strip()
            d[name] = [float(row['x']), float(row['y']), float(row['z'])]
    return from_dict(d, units=units)


def write_json(hp: DoubleWishboneHardpoints, path: str, units='mm'):
    """
    Write hardpoints to a JSON file.

    Parameters
    ----------
    hp : DoubleWishboneHardpoints
    path : str or Path
    units : str, 'mm' or 'm'
    """
    d = to_dict(hp, units=units)
    path = Path(path)
    with open(path, 'w') as f:
        json.dump(d, f, indent=2)


def read_json(path: str, units='mm') -> DoubleWishboneHardpoints:
    """
    Read hardpoints from a JSON file.

    Parameters
    ----------
    path : str or Path
    units : str — what units the JSON values are in

    Returns
    -------
    DoubleWishboneHardpoints
    """
    path = Path(path)
    with open(path, 'r') as f:
        d = json.load(f)
    return from_dict(d, units=units)


def print_summary(hp: DoubleWishboneHardpoints):
    """
    Print a human-readable summary of a hardpoint set with key derived values.
    Good for a quick sanity check after loading from file.
    """
    d = to_dict(hp, units='mm')

    print("=== Hardpoint summary (mm) ===")
    for name in HARDPOINT_NAMES:
        x, y, z = d[name]
        print(f"  {name:25s}  X={x:8.1f}  Y={y:8.1f}  Z={z:8.1f}")

    print(f"\n  Upper arm length (pivot ctr → BJ): "
          f"{np.linalg.norm(hp.upper_ball_joint - hp.upper_pivot_center())*1000:.1f} mm")
    print(f"  Lower arm length (pivot ctr → BJ): "
          f"{np.linalg.norm(hp.lower_ball_joint - hp.lower_pivot_center())*1000:.1f} mm")
    print(f"  Upright length (BJ → BJ):          "
          f"{hp.upright_length()*1000:.1f} mm")
    print(f"  Arm length ratio (upper/lower):     "
          f"{np.linalg.norm(hp.upper_ball_joint - hp.upper_pivot_center()) / np.linalg.norm(hp.lower_ball_joint - hp.lower_pivot_center()):.3f}")
