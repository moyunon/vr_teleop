"""Scan the RM75 elbow singularity using the runtime Jacobian and IK."""

import numpy as np

from vr_rm75_teleop.rm75_fk import forward_kinematics
from vr_rm75_teleop.rm75_ik import solve_ik
from vr_rm75_teleop.rm75_model import RM75Model
from vr_rm75_teleop.target_feasibility import (
    minimum_singular_value,
    singularity_region,
    singularity_speed_scale,
)


SIGMA_STOP = 0.010
SIGMA_WARN = 0.020
Q4_MAGNITUDES_DEG = (30.0, 25.0, 20.0, 15.0, 12.5, 10.0,
                     7.5, 5.0, 2.5, 0.0)
REFERENCE_Q_DEG = {
    "left": (-40.0, -25.0, 15.0, -30.0, 10.0, -35.0, 80.0),
    "right": (20.0, 35.0, 25.0, 30.0, 15.0, 40.0, -120.0),
}


def scan_side(side):
    """Return q4, sigma, region, scale, and IK status for one arm."""
    model = RM75Model(side=side)
    q_reference = np.deg2rad(REFERENCE_Q_DEG[side])
    q_seed = q_reference.copy()
    q4_sign = -1.0 if side == "left" else 1.0
    rows = []

    for magnitude_deg in Q4_MAGNITUDES_DEG:
        q = q_reference.copy()
        q4_deg = q4_sign * magnitude_deg
        q[3] = np.deg2rad(q4_deg)
        T_target = forward_kinematics(q, model=model)
        sigma_min = minimum_singular_value(q, model)
        ik_result = solve_ik(
            T_target=T_target,
            q_seed=q_seed,
            model=model,
            max_iterations=20,
            position_tolerance=1e-4,
            orientation_tolerance=1e-3,
            damping=0.02,
            step_gain=0.7,
            max_joint_step=np.deg2rad(2.0),
            preferred_posture=q_reference,
            preferred_posture_gain=1.0,
            max_null_step=np.deg2rad(0.10),
        )
        if ik_result["success"]:
            q_seed = ik_result["q"].copy()

        rows.append({
            "side": side,
            "q4_deg": q4_deg,
            "sigma_min": sigma_min,
            "ik_success": bool(ik_result["success"]),
            "ik_iterations": int(ik_result["iterations"]),
            "region": singularity_region(
                sigma_min,
                sigma_stop=SIGMA_STOP,
                sigma_warn=SIGMA_WARN,
            ),
            "speed_scale": singularity_speed_scale(
                sigma_min,
                sigma_stop=SIGMA_STOP,
                sigma_warn=SIGMA_WARN,
            ),
        })

    return rows


def validate_scan(rows):
    """Check the expected threshold crossings in the current RM75 model."""
    sigmas = np.array([row["sigma_min"] for row in rows])
    if np.any(np.diff(sigmas) > 1e-12):
        raise AssertionError("sigma_min did not decrease monotonically")

    by_magnitude = {
        abs(row["q4_deg"]): row
        for row in rows
    }
    expected_regions = {
        15.0: "safe",
        12.5: "warning",
        5.0: "stop",
        0.0: "stop",
    }
    for magnitude_deg, expected in expected_regions.items():
        actual = by_magnitude[magnitude_deg]["region"]
        if actual != expected:
            raise AssertionError(
                f"|q4|={magnitude_deg} expected {expected}, got {actual}"
            )

    if by_magnitude[0.0]["sigma_min"] > 1e-12:
        raise AssertionError("q4=0 should expose the elbow singularity")
    if not all(row["ik_success"] for row in rows):
        raise AssertionError("runtime IK unexpectedly failed during scan")


def main():
    """Print and validate the model-specific threshold evidence table."""
    print("RM75 q4 -> 0 singularity scan")
    print("Jacobian SVD: unweighted [Jv in m; Jw in rad]")
    print(f"sigma_stop={SIGMA_STOP:.3f}, sigma_warn={SIGMA_WARN:.3f}")
    print()
    print("side   q4_deg   sigma_min   IK status  iter  region   rate_scale")

    for side in ("left", "right"):
        rows = scan_side(side)
        validate_scan(rows)
        for row in rows:
            ik_status = "OK" if row["ik_success"] else "FAIL"
            print(
                f"{side:5s} {row['q4_deg']:8.1f} "
                f"{row['sigma_min']:11.8f} {ik_status:9s} "
                f"{row['ik_iterations']:4d} {row['region']:8s} "
                f"{row['speed_scale']:10.6f}"
            )

    print()
    print("PASS: 0.010/0.020 partition stop/warning/safe as expected.")
    print("Defaults retained; hardware commissioning is still required.")


if __name__ == "__main__":
    main()
