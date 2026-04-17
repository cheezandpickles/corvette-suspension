"""
Design package export — freeze geometry and produce reference documents.

Once you're happy with the hardpoints, this module exports everything
a machinist (you) needs to build the parts:

    1. Hardpoint CSV files (front and rear)
    2. Key dimensions summary (arm lengths, ball joint spacing, etc.)
    3. Kinematic curves at the frozen geometry
    4. Vehicle parameter summary

The output goes into a design_package/ directory that you can reference
while doing CAD work in Onshape.
"""

import json
from pathlib import Path
import numpy as np

from .hardpoints import DoubleWishboneHardpoints
from .hardpoint_io import write_csv, write_json, to_dict, HARDPOINT_NAMES
from .kinematics_front import solve_corner, compute_camber, wheel_travel
from .roll_center import compute_roll_geometry_sweep
from .side_view import compute_side_view_sweep
from .scorer import score_geometry, print_scorecard


def export_design_package(
    front_hp: DoubleWishboneHardpoints,
    rear_hp: DoubleWishboneHardpoints,
    vehicle_config,
    output_dir: str = "design_package",
    front_tie_rod: tuple = None,
    rear_toe_link: tuple = None,
):
    """
    Export a complete design package to a directory.

    Parameters
    ----------
    front_hp, rear_hp : DoubleWishboneHardpoints
    vehicle_config : VehicleConfig
    output_dir : str
    front_tie_rod : tuple of (inner, outer) ndarray, optional
    rear_toe_link : tuple of (inner, outer) ndarray, optional
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1. Hardpoint files
    write_csv(front_hp, out / "front_left_hardpoints.csv")
    write_csv(rear_hp, out / "rear_left_hardpoints.csv")
    write_json(front_hp, out / "front_left_hardpoints.json")
    write_json(rear_hp, out / "rear_left_hardpoints.json")

    # 2. Key dimensions
    dims = _compute_dimensions(front_hp, rear_hp)
    with open(out / "key_dimensions.json", 'w') as f:
        json.dump(dims, f, indent=2)

    # 3. Kinematic data
    front_data = _sweep_and_collect(front_hp, vehicle_config.front_vehicle_params())
    rear_data = _sweep_and_collect(rear_hp, vehicle_config.rear_vehicle_params())

    _write_sweep_csv(front_data, out / "front_kinematic_curves.csv")
    _write_sweep_csv(rear_data, out / "rear_kinematic_curves.csv")

    # 4. Scorecard
    front_score = score_geometry(front_hp, vehicle_config.front_vehicle_params())
    rear_score = score_geometry(rear_hp, vehicle_config.rear_vehicle_params())

    # 5. Summary text file
    _write_summary(out / "DESIGN_SUMMARY.txt", front_hp, rear_hp,
                   vehicle_config, dims, front_score, rear_score)

    print(f"\nDesign package exported to: {out.resolve()}")
    print(f"  front_left_hardpoints.csv/json")
    print(f"  rear_left_hardpoints.csv/json")
    print(f"  key_dimensions.json")
    print(f"  front_kinematic_curves.csv")
    print(f"  rear_kinematic_curves.csv")
    print(f"  DESIGN_SUMMARY.txt")


def _compute_dimensions(front_hp, rear_hp):
    """Compute key dimensions for reference."""
    def corner_dims(hp, label):
        upper_len = np.linalg.norm(hp.upper_ball_joint - hp.upper_pivot_center())
        lower_len = np.linalg.norm(hp.lower_ball_joint - hp.lower_pivot_center())
        upright_len = hp.upright_length()
        bj_spread_y = abs(hp.upper_ball_joint[1] - hp.lower_ball_joint[1])
        bj_spread_z = abs(hp.upper_ball_joint[2] - hp.lower_ball_joint[2])

        return {
            f'{label}_upper_arm_length_mm': round(upper_len * 1000, 1),
            f'{label}_lower_arm_length_mm': round(lower_len * 1000, 1),
            f'{label}_upright_length_mm': round(upright_len * 1000, 1),
            f'{label}_arm_ratio': round(upper_len / lower_len, 3),
            f'{label}_bj_lateral_spread_mm': round(bj_spread_y * 1000, 1),
            f'{label}_bj_vertical_spread_mm': round(bj_spread_z * 1000, 1),
        }

    dims = {}
    dims.update(corner_dims(front_hp, 'front'))
    dims.update(corner_dims(rear_hp, 'rear'))
    return dims


def _sweep_and_collect(hp, vehicle_params):
    """Run a kinematic sweep and collect all data."""
    angles_rad = np.deg2rad(np.linspace(-10, 10, 81))

    results = []
    camber = []
    travel = []

    for angle in angles_rad:
        r = solve_corner(hp, angle)
        results.append(r)
        camber.append(compute_camber(r['wheel_center'], r['contact_patch']))
        travel.append(wheel_travel(hp, r) * 1000)

    roll_geom = compute_roll_geometry_sweep(hp, results, angles_rad)
    side_geom = compute_side_view_sweep(hp, results, vehicle_params)

    return {
        'travel_mm': np.array(travel),
        'camber_deg': np.array(camber),
        'rc_height_mm': roll_geom['rc_z'] * 1000,
        'fvsa_mm': roll_geom['fvsa'] * 1000,
        'ic_y_mm': roll_geom['ic_y'] * 1000,
        'ic_z_mm': roll_geom['ic_z'] * 1000,
        'anti_pct': side_geom['anti_percent'],
    }


def _write_sweep_csv(data, path):
    """Write kinematic sweep data to CSV."""
    import csv
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['travel_mm', 'camber_deg', 'rc_height_mm',
                         'fvsa_mm', 'ic_y_mm', 'ic_z_mm', 'anti_pct'])
        for i in range(len(data['travel_mm'])):
            writer.writerow([
                f"{data['travel_mm'][i]:.2f}",
                f"{data['camber_deg'][i]:.4f}",
                f"{data['rc_height_mm'][i]:.2f}",
                f"{data['fvsa_mm'][i]:.1f}",
                f"{data['ic_y_mm'][i]:.1f}",
                f"{data['ic_z_mm'][i]:.1f}",
                f"{data['anti_pct'][i]:.2f}",
            ])


def _write_summary(path, front_hp, rear_hp, vc, dims, front_score, rear_score):
    """Write a human-readable design summary."""
    with open(path, 'w') as f:
        f.write("=" * 65 + "\n")
        f.write("  1972 CORVETTE C3 SUSPENSION DESIGN PACKAGE\n")
        f.write("=" * 65 + "\n\n")

        f.write("VEHICLE PARAMETERS\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Total weight:    {vc.total_weight_kg:.0f} kg "
                f"({vc.total_weight_kg * 2.205:.0f} lbs)\n")
        f.write(f"  F/R split:       {vc.front_weight_fraction*100:.0f}/"
                f"{(1-vc.front_weight_fraction)*100:.0f}\n")
        f.write(f"  Wheelbase:       {vc.wheelbase*1000:.0f} mm\n")
        f.write(f"  Front track:     {vc.front_track*1000:.0f} mm\n")
        f.write(f"  Rear track:      {vc.rear_track*1000:.0f} mm\n")
        f.write(f"  CG height:       {vc.cg_height*1000:.0f} mm\n")
        f.write(f"  Front tire:      {vc.front_tire}\n")
        f.write(f"  Rear tire:       {vc.rear_tire}\n\n")

        f.write("KEY DIMENSIONS\n")
        f.write("-" * 40 + "\n")
        for k, v in dims.items():
            f.write(f"  {k}: {v}\n")
        f.write("\n")

        f.write("FRONT GEOMETRY SCORECARD\n")
        f.write("-" * 40 + "\n")
        for name, data in front_score['scores'].items():
            score = data['score']
            grade = "GOOD" if score >= 75 else ("OK" if score >= 50 else "BAD")
            f.write(f"  [{grade:4s}] {name:18s} = {data['value']:.3f} {data['unit']}\n")
        f.write(f"  Overall: {front_score['overall']:.0f}/100\n\n")

        f.write("REAR GEOMETRY SCORECARD\n")
        f.write("-" * 40 + "\n")
        for name, data in rear_score['scores'].items():
            score = data['score']
            grade = "GOOD" if score >= 75 else ("OK" if score >= 50 else "BAD")
            f.write(f"  [{grade:4s}] {name:18s} = {data['value']:.3f} {data['unit']}\n")
        f.write(f"  Overall: {rear_score['overall']:.0f}/100\n\n")

        f.write("HARDPOINT COORDINATES (mm, SAE J670)\n")
        f.write("-" * 40 + "\n")
        f.write("See CSV files for exact values.\n")
        f.write("Mirror about y=0 for right side.\n")
