"""
Quasi-static full-car model: weight transfer, roll, and tire loading.

This module computes what happens to the car as a whole when you corner,
brake, or accelerate. It ties together the individual corner kinematics
with the vehicle-level parameters to predict:

    - Lateral weight transfer (how much load shifts to the outside tires)
    - Body roll angle
    - Individual tire loads (all four corners)
    - Roll angle and its effect on camber

The key concept: total lateral weight transfer is fixed by physics (it
depends only on CG height, track width, total weight, and lateral g).
But the DISTRIBUTION of that transfer between front and rear axles is
what you control through suspension design — specifically through roll
center heights, spring rates, and anti-roll bars.

This is where tire load sensitivity matters: tires make less grip per
unit of load as load increases. So the tire with MORE load is less
efficient than the one with LESS load. This means you want to put more
of the weight transfer on the axle that has MORE total grip margin,
which is usually the axle you want to break away first (rear for
oversteer, front for understeer).

References:
    Milliken & Milliken, "Race Car Vehicle Dynamics", ch. 18-19
    Carroll Smith, "Tune to Win", ch. 6-7
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class WeightTransferResult:
    """Results from a quasi-static weight transfer calculation."""
    lateral_g: float              # input lateral acceleration
    total_lateral_wt: float       # total lateral weight transfer, N
    front_lateral_wt: float       # front axle lateral weight transfer, N
    rear_lateral_wt: float        # rear axle lateral weight transfer, N
    geometric_wt_front: float     # geometric (through links) portion, front, N
    geometric_wt_rear: float      # geometric portion, rear, N
    elastic_wt_front: float       # elastic (through springs) portion, front, N
    elastic_wt_rear: float        # elastic portion, rear, N
    roll_angle_deg: float         # body roll angle, degrees
    tire_loads: dict               # corner loads: 'fl', 'fr', 'rl', 'rr' in N

    def print_summary(self):
        print(f"\n{'='*50}")
        print(f"Weight transfer at {self.lateral_g:.2f} g lateral")
        print(f"{'='*50}")
        print(f"  Total lateral WT:    {self.total_lateral_wt:.0f} N "
              f"({self.total_lateral_wt/4.448:.0f} lbs)")
        print(f"  Front lateral WT:    {self.front_lateral_wt:.0f} N "
              f"({self.front_lateral_wt/4.448:.0f} lbs)")
        print(f"  Rear lateral WT:     {self.rear_lateral_wt:.0f} N "
              f"({self.rear_lateral_wt/4.448:.0f} lbs)")
        print(f"  Front/rear WT split: {self.front_lateral_wt/self.total_lateral_wt*100:.1f}/"
              f"{self.rear_lateral_wt/self.total_lateral_wt*100:.1f}")
        print(f"\n  Roll angle:          {self.roll_angle_deg:.2f} deg")
        print(f"\n  Tire loads (cornering to the right, left = outside):")
        print(f"    FL (outside): {self.tire_loads['fl']:.0f} N "
              f"({self.tire_loads['fl']/4.448:.0f} lbs)")
        print(f"    FR (inside):  {self.tire_loads['fr']:.0f} N "
              f"({self.tire_loads['fr']/4.448:.0f} lbs)")
        print(f"    RL (outside): {self.tire_loads['rl']:.0f} N "
              f"({self.tire_loads['rl']/4.448:.0f} lbs)")
        print(f"    RR (inside):  {self.tire_loads['rr']:.0f} N "
              f"({self.tire_loads['rr']/4.448:.0f} lbs)")


def compute_lateral_weight_transfer(
    vehicle_config,
    lateral_g: float,
    rc_height_front: float,
    rc_height_rear: float,
) -> WeightTransferResult:
    """
    Quasi-static lateral weight transfer for a given cornering load.

    Parameters
    ----------
    vehicle_config : VehicleConfig
    lateral_g : float — lateral acceleration in g's (positive = right turn)
    rc_height_front : float — front roll center height, meters
    rc_height_rear : float — rear roll center height, meters

    Returns
    -------
    WeightTransferResult
    """
    vc = vehicle_config
    W = vc.total_weight_n
    m = vc.total_weight_kg
    ay = lateral_g * 9.81  # m/s^2

    # Total lateral weight transfer (this is fixed by physics)
    total_lat_wt = m * ay * vc.cg_height / (
        (vc.front_track + vc.rear_track) / 2.0
    )

    # Roll center height at the CG location (linear interpolation)
    rc_at_cg = (rc_height_front * (1 - vc.front_weight_fraction) +
                rc_height_rear * vc.front_weight_fraction)

    # Roll moment arm = CG height - roll axis height at CG
    roll_arm = vc.cg_height - rc_at_cg

    # Geometric weight transfer (direct, through the links)
    geo_wt_front = (vc.front_axle_load_n * ay / 9.81 *
                    rc_height_front / vc.front_track)
    geo_wt_rear = (vc.rear_axle_load_n * ay / 9.81 *
                   rc_height_rear / vc.rear_track)

    # Roll moment
    roll_moment = m * ay * roll_arm  # N·m

    # Roll stiffness (at the wheel)
    # K_roll = 0.5 * track^2 * (k_spring * MR^2 + k_arb)
    # The 0.5 is because roll compresses one side and extends the other
    k_roll_front = 0.5 * vc.front_track**2 * (
        vc.front_spring_rate_nmm * 1000 * vc.front_motion_ratio**2 +
        vc.front_arb_rate_nmm * 1000
    )
    k_roll_rear = 0.5 * vc.rear_track**2 * (
        vc.rear_spring_rate_nmm * 1000 * vc.rear_motion_ratio**2 +
        vc.rear_arb_rate_nmm * 1000
    )

    total_roll_stiffness = k_roll_front + k_roll_rear

    if total_roll_stiffness < 1e-6:
        roll_angle_rad = 0.0
    else:
        roll_angle_rad = roll_moment / total_roll_stiffness

    roll_angle_deg = np.rad2deg(roll_angle_rad)

    # Elastic weight transfer (distributed by roll stiffness ratio)
    elastic_moment = roll_moment  # total moment going through springs
    if total_roll_stiffness > 1e-6:
        front_roll_fraction = k_roll_front / total_roll_stiffness
    else:
        front_roll_fraction = 0.5

    elastic_wt_front = (elastic_moment * front_roll_fraction /
                        vc.front_track)
    elastic_wt_rear = (elastic_moment * (1 - front_roll_fraction) /
                       vc.rear_track)

    # Total per-axle weight transfer
    front_lat_wt = geo_wt_front + elastic_wt_front
    rear_lat_wt = geo_wt_rear + elastic_wt_rear

    # Individual tire loads (right turn: left = outside)
    static_fl = vc.front_corner_load_n
    static_fr = vc.front_corner_load_n
    static_rl = vc.rear_corner_load_n
    static_rr = vc.rear_corner_load_n

    tire_loads = {
        'fl': static_fl + front_lat_wt,   # outside front
        'fr': static_fr - front_lat_wt,   # inside front
        'rl': static_rl + rear_lat_wt,    # outside rear
        'rr': static_rr - rear_lat_wt,    # inside rear
    }

    return WeightTransferResult(
        lateral_g=lateral_g,
        total_lateral_wt=total_lat_wt,
        front_lateral_wt=front_lat_wt,
        rear_lateral_wt=rear_lat_wt,
        geometric_wt_front=geo_wt_front,
        geometric_wt_rear=geo_wt_rear,
        elastic_wt_front=elastic_wt_front,
        elastic_wt_rear=elastic_wt_rear,
        roll_angle_deg=roll_angle_deg,
        tire_loads=tire_loads,
    )


def compute_longitudinal_weight_transfer(
    vehicle_config,
    longitudinal_g: float,
) -> dict:
    """
    Longitudinal weight transfer under braking or acceleration.

    Parameters
    ----------
    vehicle_config : VehicleConfig
    longitudinal_g : float — longitudinal accel in g's
        Positive = accelerating, negative = braking

    Returns
    -------
    dict with 'front_load_n' and 'rear_load_n' (per axle, not per corner)
    """
    vc = vehicle_config
    m = vc.total_weight_kg
    ax = longitudinal_g * 9.81

    # Weight transfer
    delta_wt = m * ax * vc.cg_height / vc.wheelbase  # positive = rearward

    front_total = vc.front_axle_load_n - delta_wt
    rear_total = vc.rear_axle_load_n + delta_wt

    return {
        'front_axle_load_n': front_total,
        'rear_axle_load_n': rear_total,
        'front_corner_load_n': front_total / 2.0,
        'rear_corner_load_n': rear_total / 2.0,
        'weight_transfer_n': delta_wt,
    }


def roll_angle_for_lateral_g(vehicle_config, lateral_g, rc_height_front, rc_height_rear):
    """Quick helper to get just the roll angle."""
    result = compute_lateral_weight_transfer(
        vehicle_config, lateral_g, rc_height_front, rc_height_rear
    )
    return result.roll_angle_deg


def spring_rate_recommendation(vehicle_config, target_roll_deg_at_1g=1.5):
    """
    Suggest spring rates to achieve a target roll angle at 1g.

    This is a starting point — you'll refine on track. Most street/track
    cars aim for 1.0-2.0 degrees of roll at 1g lateral.

    Parameters
    ----------
    vehicle_config : VehicleConfig
    target_roll_deg_at_1g : float

    Returns
    -------
    dict with suggested spring rates and ARB rates
    """
    vc = vehicle_config

    # Roll moment at 1g
    rc_avg = vc.cg_height * 0.15  # assume RC is ~15% of CG height
    roll_arm = vc.cg_height - rc_avg
    roll_moment_1g = vc.total_weight_kg * 9.81 * roll_arm

    # Required total roll stiffness
    target_roll_rad = np.deg2rad(target_roll_deg_at_1g)
    required_k_roll = roll_moment_1g / target_roll_rad

    # Split 60/40 front/rear for mild understeer balance
    k_roll_front = required_k_roll * 0.55
    k_roll_rear = required_k_roll * 0.45

    # Back-calculate spring rates (assume 70% of roll stiffness from springs)
    spring_fraction = 0.70

    k_spring_front = (k_roll_front * spring_fraction /
                      (0.5 * vc.front_track**2 * vc.front_motion_ratio**2)) / 1000
    k_spring_rear = (k_roll_rear * spring_fraction /
                     (0.5 * vc.rear_track**2 * vc.rear_motion_ratio**2)) / 1000

    k_arb_front = (k_roll_front * (1 - spring_fraction) /
                   (0.5 * vc.front_track**2)) / 1000
    k_arb_rear = (k_roll_rear * (1 - spring_fraction) /
                  (0.5 * vc.rear_track**2)) / 1000

    return {
        'front_spring_nmm': k_spring_front,
        'rear_spring_nmm': k_spring_rear,
        'front_arb_nmm': k_arb_front,
        'rear_arb_nmm': k_arb_rear,
        'total_roll_stiffness': required_k_roll,
        'target_roll_deg': target_roll_deg_at_1g,
    }
