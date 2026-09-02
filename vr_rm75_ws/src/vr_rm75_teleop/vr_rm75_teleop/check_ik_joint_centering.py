import numpy as np

from scipy.spatial.transform import Rotation

from vr_rm75_teleop.rm75_model import RM75Model
from vr_rm75_teleop.rm75_fk import forward_kinematics
from vr_rm75_teleop.rm75_ik import solve_ik
from vr_rm75_teleop.rm75_nullspace import preferred_posture_cost


np.set_printoptions(
    precision=6,
    suppress=True,
)


def independent_pose_error(
    T_target,
    T_solution,
):
    """Return independent position and orientation errors."""

    position_error = np.linalg.norm(
        T_target[:3, 3]
        - T_solution[:3, 3]
    )

    R_error = (
        T_target[:3, :3]
        @ T_solution[:3, :3].T
    )

    orientation_error = np.linalg.norm(
        Rotation
        .from_matrix(R_error)
        .as_rotvec()
    )

    return (
        position_error,
        orientation_error,
    )


def evaluate_result(
    name,
    result,
    T_target,
    q_preferred,
    model,
):
    """Print and return the independent result metrics."""

    q = result["q"]

    T_solution = forward_kinematics(
        q,
        model=model,
    )

    p_error, r_error = independent_pose_error(
        T_target,
        T_solution,
    )

    posture_cost = preferred_posture_cost(
        q,
        q_preferred,
        model,
    )

    print("")
    print("=" * 70)
    print(name)
    print("=" * 70)
    print("")
    print("Success:", result["success"])
    print("Iterations:", result["iterations"])
    print("")
    print("q [deg]:")
    print(np.rad2deg(q))
    print("")
    print("Position error [mm]:", p_error * 1000.0)
    print("Orientation error [deg]:", np.rad2deg(r_error))
    print("Preferred posture cost:", posture_cost)

    return {
        "position_error": p_error,
        "orientation_error": r_error,
        "posture_cost": posture_cost,
    }


def main():
    """Compare baseline DLS with the current preferred-posture nullspace API."""

    model = RM75Model(
        side="right",
    )

    q_true = np.deg2rad(
        [
            15.0,
            -15.0,
            25.0,
            45.0,
            -20.0,
            30.0,
            20.0,
        ]
    )

    q_seed = np.deg2rad(
        [
            10.0,
            -20.0,
            30.0,
            40.0,
            -25.0,
            35.0,
            15.0,
        ]
    )

    # A safe preferred posture on the same positive-J4 elbow branch.
    q_preferred = np.deg2rad(
        [
            20.0,
            -30.0,
            40.0,
            60.0,
            -30.0,
            40.0,
            30.0,
        ]
    )

    T_target = forward_kinematics(
        q_true,
        model=model,
    )

    common_kwargs = {
        "T_target": T_target,
        "q_seed": q_seed,
        "model": model,
        "max_iterations": 300,
        "damping": 0.02,
        "step_gain": 0.5,
    }

    baseline = solve_ik(
        **common_kwargs,
    )

    preferred = solve_ik(
        **common_kwargs,
        preferred_posture=q_preferred,
        preferred_posture_gain=1.0,
        max_null_step=np.deg2rad(0.5),
    )

    baseline_metrics = evaluate_result(
        "A: BASELINE DLS",
        baseline,
        T_target,
        q_preferred,
        model,
    )

    preferred_metrics = evaluate_result(
        "B: DLS + PREFERRED POSTURE",
        preferred,
        T_target,
        q_preferred,
        model,
    )

    cost_reduction = (
        baseline_metrics["posture_cost"]
        - preferred_metrics["posture_cost"]
    )

    task_error_ok = (
        baseline_metrics["position_error"] < 1e-4
        and baseline_metrics["orientation_error"] < 1e-3
        and preferred_metrics["position_error"] < 1e-4
        and preferred_metrics["orientation_error"] < 1e-3
    )

    passed = (
        baseline["success"]
        and preferred["success"]
        and task_error_ok
        and cost_reduction > 1e-5
    )

    print("")
    print("=" * 70)
    print("COMPARISON")
    print("=" * 70)
    print("")
    print("Cost reduction:", cost_reduction)
    print(
        "PREFERRED POSTURE TEST:",
        "PASS" if passed else "FAIL",
    )

    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
