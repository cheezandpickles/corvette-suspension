"""
Master vehicle configuration for the 1972 Corvette C3.

This is the single source of truth for all car-level parameters. Every
notebook imports from here instead of defining its own constants.

Edit this file as you measure/weigh the actual car. Values marked
(ESTIMATED) should be replaced with measured data when available.

Units: SI throughout (meters, kg, radians). Convert only at display.
"""

import json
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional
import numpy as np


@dataclass
class TireSpec:
    """Tire specification for one axle."""
    section_width_mm: float      # e.g. 295
    aspect_ratio: float          # e.g. 0.35 (35-series)
    wheel_diameter_in: float     # e.g. 18
    treadwear: int = 200         # UTQG treadwear rating
    compound: str = "street"     # "street", "200tw", "r_compound"

    @property
    def section_width(self):
        """Section width in meters."""
        return self.section_width_mm / 1000.0

    @property
    def sidewall_height(self):
        """Sidewall height in meters."""
        return self.section_width * self.aspect_ratio

    @property
    def wheel_radius(self):
        """Wheel (rim) radius in meters."""
        return self.wheel_diameter_in * 0.0254 / 2.0

    @property
    def tire_radius(self):
        """Loaded tire radius in meters (approximate, no deflection)."""
        return self.wheel_radius + self.sidewall_height

    @property
    def tire_diameter(self):
        """Overall tire diameter in meters."""
        return self.tire_radius * 2.0

    def __repr__(self):
        return (f"{self.section_width_mm:.0f}/"
                f"{self.aspect_ratio*100:.0f}R"
                f"{self.wheel_diameter_in:.0f} "
                f"({self.treadwear}tw {self.compound})")


@dataclass
class VehicleConfig:
    """
    All vehicle-level parameters for the C3 project.

    These feed into every analysis notebook. When a value is estimated,
    the comment says so — replace with measured values as you get them.
    """

    # ---- Mass & inertia ----
    curb_weight_kg: float = 1588.0          # ~3500 lbs (ESTIMATED)
    driver_weight_kg: float = 82.0          # ~180 lbs, with helmet/gear
    fuel_weight_kg: float = 30.0            # ~8 gallons, half tank

    # ---- Geometry ----
    wheelbase: float = 2.489                # 98.0 inches (C3 spec)
    front_track: float = 1.499              # 59.0 inches (ESTIMATED)
    rear_track: float = 1.524               # 60.0 inches (ESTIMATED)
    cg_height: float = 0.460                # ~18 inches (ESTIMATED)
    front_weight_fraction: float = 0.52     # slightly nose-heavy with SBC (ESTIMATED)

    # ---- Brakes ----
    brake_bias_front: float = 0.60          # 60% front (ESTIMATED, adjustable)

    # ---- Tires ----
    front_tire: TireSpec = field(default_factory=lambda: TireSpec(
        section_width_mm=295, aspect_ratio=0.35,
        wheel_diameter_in=18, treadwear=200, compound="200tw"
    ))
    rear_tire: TireSpec = field(default_factory=lambda: TireSpec(
        section_width_mm=335, aspect_ratio=0.30,
        wheel_diameter_in=18, treadwear=200, compound="200tw"
    ))

    # ---- Suspension (springs & dampers, for quasi-static model) ----
    front_spring_rate_nmm: float = 53.0     # N/mm (~300 lb/in) (ESTIMATED)
    rear_spring_rate_nmm: float = 44.0      # N/mm (~250 lb/in) (ESTIMATED)
    front_motion_ratio: float = 0.70        # spring travel / wheel travel (ESTIMATED)
    rear_motion_ratio: float = 0.75         # (ESTIMATED)
    front_arb_rate_nmm: float = 20.0        # anti-roll bar, N/mm at wheel (ESTIMATED)
    rear_arb_rate_nmm: float = 10.0         # (ESTIMATED)

    # ---- Aero (minimal for now) ----
    frontal_area: float = 1.85              # m^2 (ESTIMATED)
    cd: float = 0.44                        # drag coefficient (ESTIMATED, C3 is not slippery)
    cl_front: float = 0.0                   # lift coefficient, front (no downforce)
    cl_rear: float = 0.0                    # lift coefficient, rear

    # ---- Future powertrain (for weight transfer calcs) ----
    engine: str = "SBC 350"                 # current engine
    power_hp: float = 250.0                 # current (ESTIMATED)
    future_engine: str = "LS7 NA"
    future_power_hp: float = 750.0          # target

    # ---- Targets (what you're designing toward) ----
    target_anti_dive_pct: tuple = (15.0, 25.0)   # min, max
    target_anti_squat_pct: tuple = (15.0, 50.0)
    target_rc_height_front_mm: tuple = (25.0, 75.0)
    target_rc_height_rear_mm: tuple = (50.0, 125.0)
    target_camber_gain_degpmm: tuple = (-0.04, -0.015)  # deg per mm bump
    target_bump_steer_degpmm: float = 0.001  # max acceptable, deg per mm

    # ---- Derived properties ----

    @property
    def total_weight_kg(self):
        return self.curb_weight_kg + self.driver_weight_kg + self.fuel_weight_kg

    @property
    def total_weight_n(self):
        return self.total_weight_kg * 9.81

    @property
    def front_axle_load_n(self):
        return self.total_weight_n * self.front_weight_fraction

    @property
    def rear_axle_load_n(self):
        return self.total_weight_n * (1.0 - self.front_weight_fraction)

    @property
    def front_corner_load_n(self):
        return self.front_axle_load_n / 2.0

    @property
    def rear_corner_load_n(self):
        return self.rear_axle_load_n / 2.0

    @property
    def cg_to_front_axle(self):
        """Distance from CG to front axle, meters."""
        return self.wheelbase * (1.0 - self.front_weight_fraction)

    @property
    def cg_to_rear_axle(self):
        """Distance from CG to rear axle, meters."""
        return self.wheelbase * self.front_weight_fraction

    def front_vehicle_params(self):
        """Dict formatted for side_view.compute_side_view_sweep (front)."""
        return {
            'cg_height': self.cg_height,
            'wheelbase': self.wheelbase,
            'front_weight_fraction': self.front_weight_fraction,
            'brake_bias_front': self.brake_bias_front,
            'is_front': True,
        }

    def rear_vehicle_params(self):
        """Dict formatted for side_view.compute_side_view_sweep (rear)."""
        return {
            'cg_height': self.cg_height,
            'wheelbase': self.wheelbase,
            'front_weight_fraction': self.front_weight_fraction,
            'is_front': False,
        }

    def print_summary(self):
        """Print a human-readable summary."""
        print("=" * 60)
        print("1972 Corvette C3 — Vehicle Configuration")
        print("=" * 60)
        print(f"\n  Total weight:       {self.total_weight_kg:.0f} kg "
              f"({self.total_weight_kg * 2.205:.0f} lbs)")
        print(f"  Front/rear split:   {self.front_weight_fraction*100:.0f}/"
              f"{(1-self.front_weight_fraction)*100:.0f}")
        print(f"  Front corner load:  {self.front_corner_load_n:.0f} N "
              f"({self.front_corner_load_n / 4.448:.0f} lbs)")
        print(f"  Rear corner load:   {self.rear_corner_load_n:.0f} N "
              f"({self.rear_corner_load_n / 4.448:.0f} lbs)")
        print(f"\n  Wheelbase:          {self.wheelbase*1000:.0f} mm "
              f"({self.wheelbase / 0.0254:.1f} in)")
        print(f"  Front track:        {self.front_track*1000:.0f} mm "
              f"({self.front_track / 0.0254:.1f} in)")
        print(f"  Rear track:         {self.rear_track*1000:.0f} mm "
              f"({self.rear_track / 0.0254:.1f} in)")
        print(f"  CG height:          {self.cg_height*1000:.0f} mm "
              f"({self.cg_height / 0.0254:.1f} in)")
        print(f"  CG to front axle:   {self.cg_to_front_axle*1000:.0f} mm")
        print(f"  CG to rear axle:    {self.cg_to_rear_axle*1000:.0f} mm")
        print(f"\n  Front tire:         {self.front_tire}")
        print(f"  Rear tire:          {self.rear_tire}")
        print(f"  Front tire radius:  {self.front_tire.tire_radius*1000:.0f} mm")
        print(f"  Rear tire radius:   {self.rear_tire.tire_radius*1000:.0f} mm")
        print(f"\n  Brake bias:         {self.brake_bias_front*100:.0f}% front")
        print(f"\n  Front spring:       {self.front_spring_rate_nmm:.0f} N/mm "
              f"({self.front_spring_rate_nmm / 0.17513:.0f} lb/in)")
        print(f"  Rear spring:        {self.rear_spring_rate_nmm:.0f} N/mm "
              f"({self.rear_spring_rate_nmm / 0.17513:.0f} lb/in)")
        print(f"\n  Current engine:     {self.engine} ({self.power_hp:.0f} hp)")
        print(f"  Future engine:      {self.future_engine} ({self.future_power_hp:.0f} hp)")


# Default configuration — import this in notebooks
C3_CONFIG = VehicleConfig()
