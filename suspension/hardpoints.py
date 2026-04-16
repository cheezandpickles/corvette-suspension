"""
Hardpoint definitions for a double-wishbone suspension corner.

A "hardpoint" is a specific 3D location on the car that defines suspension
geometry — typically a pivot center, ball joint, or attachment. For a
double-wishbone corner we need 7 primary hardpoints, plus auxiliary points
for wheel and contact patch.

All coordinates are in the vehicle coordinate system (SAE J670), meters.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class DoubleWishboneHardpoints:
    """
    The seven hardpoints that define a double-wishbone corner's kinematics,
    plus wheel center and contact patch.

    For the A-arms, "front" and "rear" refer to the front and rear inboard
    pivots (fore and aft along the car), NOT the front axle. Both pivots
    together define the pivot axis of the arm.
    """
    # Upper A-arm inboard pivots (define the upper pivot axis)
    upper_front_pivot: np.ndarray
    upper_rear_pivot: np.ndarray

    # Upper outboard ball joint
    upper_ball_joint: np.ndarray

    # Lower A-arm inboard pivots
    lower_front_pivot: np.ndarray
    lower_rear_pivot: np.ndarray

    # Lower outboard ball joint
    lower_ball_joint: np.ndarray

    # Wheel center (geometric center of the wheel, on the spin axis)
    wheel_center: np.ndarray

    # Tire contact patch center (directly below wheel center in static position,
    # on the ground plane)
    contact_patch: np.ndarray

    def upper_pivot_axis(self):
        """Direction vector from front to rear upper inboard pivot."""
        return self.upper_rear_pivot - self.upper_front_pivot

    def lower_pivot_axis(self):
        """Direction vector from front to rear lower inboard pivot."""
        return self.lower_rear_pivot - self.lower_front_pivot

    def upper_pivot_center(self):
        """Midpoint of the upper inboard pivot axis."""
        return 0.5 * (self.upper_front_pivot + self.upper_rear_pivot)

    def lower_pivot_center(self):
        """Midpoint of the lower inboard pivot axis."""
        return 0.5 * (self.lower_front_pivot + self.lower_rear_pivot)

    def upright_length(self):
        """Static distance between ball joints (the upright's effective length)."""
        return np.linalg.norm(self.upper_ball_joint - self.lower_ball_joint)

    def static_kingpin_axis(self):
        """Unit vector from lower to upper ball joint in static position."""
        v = self.upper_ball_joint - self.lower_ball_joint
        return v / np.linalg.norm(v)


def example_frontend_hardpoints():
    """
    Representative front-suspension hardpoints for initial testing.

    These are rough ballpark numbers for a C3-sized car, NOT measured values.
    Used for sanity-checking the solver before real measurements are available.

    Coordinate frame: SAE J670 (x forward, y right, z up), origin at ground
    below front axle centerline, meters. These describe the LEFT (driver's
    side on LHD) front corner, so y is negative.
    """
    return DoubleWishboneHardpoints(
        # Upper A-arm inboard pivots (on the frame, inboard and high)
        upper_front_pivot=np.array([ 0.150, -0.250, 0.500]),
        upper_rear_pivot =np.array([-0.150, -0.250, 0.500]),

        # Upper ball joint (tucked in and up at the top of the upright)
        upper_ball_joint =np.array([ 0.000, -0.620, 0.530]),

        # Lower A-arm inboard pivots
        lower_front_pivot=np.array([ 0.180, -0.200, 0.180]),
        lower_rear_pivot =np.array([-0.180, -0.200, 0.180]),

        # Lower ball joint (wider out, near the bottom of the upright)
        lower_ball_joint =np.array([ 0.000, -0.680, 0.160]),

        # Wheel center (center of a ~24" diameter tire)
        wheel_center     =np.array([ 0.000, -0.720, 0.320]),

        # Contact patch (directly below wheel center at static)
        contact_patch    =np.array([ 0.000, -0.720, 0.000]),
    )