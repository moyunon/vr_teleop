"""Compare fixed and experimental adaptive DLS damping offline.

The production solver and teleoperation node remain configured with the
validated fixed damping value.  This harness temporarily replaces only the
DLS step function while calling the same ``solve_ik`` implementation, then
restores it after every benchmark case.
"""

from contextlib import contextmanager
import time

import numpy as np

import vr_rm75_teleop.rm75_ik as ik_module
from vr_rm75_teleop.rm75_fk import forward_kinematics
from vr_rm75_teleop.rm75_ik import (
    damped_least_squares_step as fixed_dls_step,
)
from vr_rm75_teleop.rm75_model import RM75Model
from vr_rm75_teleop.target_feasibility import (
    minimum_singular_value,
    singularity_speed_scale,
    validate_singularity_thresholds,
)


FIXED_DAMPING = 0.020
ADAPTIVE_MAX_DAMPING = 0.025
SIGMA_STOP = 0.010
SIGMA_WARN = 0.020
MODES = ("fixed", "adaptive")
SIDES = ("left", "right")

REFERENCE_Q_DEG = {
    "left": np.array([-40.0, -25.0, 15.0, -30.0,
                      10.0, -35.0, 80.0]),
    "right": np.array([20.0, 35.0, 25.0, 30.0,
                       15.0, 40.0, -120.0]),
}

# center |q4| [deg], sinusoidal amplitude [deg]
CONTINUOUS_SCENARIOS = {
    "normal": (30.0, 2.0),
    "sigma_warn": (10.5, 1.5),
    "sigma_stop": (6.5, 1.0),
    "near_q4_zero": (1.5, 1.5),
}

# One reachable target per region for deterministic seed-recovery stress.
RECOVERY_Q4_DEG = {
    "normal": 30.0,
    "sigma_warn": 12.5,
    "sigma_stop": 6.5,
    "near_q4_zero": 1.0,
}


def adaptive_damping_value(
    sigma_min,
    base_damping=FIXED_DAMPING,
    max_damping=ADAPTIVE_MAX_DAMPING,
    sigma_stop=SIGMA_STOP,
    sigma_warn=SIGMA_WARN,
):
    """Return a smooth experimental damping value from the current sigma."""
    sigma_stop, sigma_warn = validate_singularity_thresholds(
        sigma_stop,
        sigma_warn,
    )
    base_damping = float(base_damping)
    max_damping = float(max_damping)
    if not np.isfinite(base_damping) or base_damping <= 0.0:
        raise ValueError("base_damping must be finite and positive")
    if not np.isfinite(max_damping) or max_damping < base_damping:
        raise ValueError(
            "max_damping must be finite and at least base_damping"
        )

    safe_rate_scale = singularity_speed_scale(
        sigma_min,
        sigma_stop=sigma_stop,
        sigma_warn=sigma_warn,
    )
    singularity_risk = 1.0 - safe_rate_scale
    return float(
        base_damping
        + (max_damping - base_damping) * singularity_risk
    )


def adaptive_dls_step(
    J,
    error,
    damping=FIXED_DAMPING,
):
    """Evaluate one DLS step with the experimental sigma-based schedule."""
    J = np.asarray(J, dtype=float)
    sigma_min = float(np.linalg.svd(J, compute_uv=False)[-1])
    adaptive_damping = adaptive_damping_value(
        sigma_min,
        base_damping=damping,
    )
    return fixed_dls_step(
        J,
        error,
        damping=adaptive_damping,
    )


@contextmanager
def damping_mode(mode):
    """Temporarily select fixed or adaptive DLS inside the shared solver."""
    if mode not in MODES:
        raise ValueError(f"unknown damping mode: {mode}")
    original_step = ik_module.damped_least_squares_step
    if mode == "adaptive":
        ik_module.damped_least_squares_step = adaptive_dls_step
    try:
        yield
    finally:
        ik_module.damped_least_squares_step = original_step


def solve_runtime_ik(T_target, q_seed, q_preferred, model):
    """Run the same fixed-budget IK settings used by dual-arm fusion."""
    return ik_module.solve_ik(
        T_target=T_target,
        q_seed=q_seed,
        model=model,
        max_iterations=20,
        position_tolerance=1e-4,
        orientation_tolerance=1e-3,
        damping=FIXED_DAMPING,
        step_gain=0.7,
        max_joint_step=np.deg2rad(2.0),
        preferred_posture=q_preferred,
        preferred_posture_gain=1.0,
        max_null_step=np.deg2rad(0.10),
    )


def make_continuous_path(side, scenario, frames):
    """Generate one reachable, closed target path around a q4 region."""
    if side not in SIDES:
        raise ValueError(f"unknown side: {side}")
    if scenario not in CONTINUOUS_SCENARIOS:
        raise ValueError(f"unknown scenario: {scenario}")
    frames = int(frames)
    if frames < 3:
        raise ValueError("frames must be at least 3")

    center_deg, amplitude_deg = CONTINUOUS_SCENARIOS[scenario]
    q4_sign = -1.0 if side == "left" else 1.0
    path = []
    for index in range(frames):
        phase = 2.0 * np.pi * index / (frames - 1)
        q_deg = REFERENCE_Q_DEG[side].copy()
        q_deg[0] += 0.6 * np.sin(phase)
        q_deg[1] += 0.4 * np.sin(2.0 * phase)
        q_deg[2] += 0.5 * np.sin(phase)
        q_deg[3] = q4_sign * (
            center_deg + amplitude_deg * np.sin(phase)
        )
        q_deg[4] += 0.4 * np.sin(phase)
        q_deg[5] += 0.3 * np.sin(2.0 * phase)
        q_deg[6] += 0.6 * np.sin(phase)
        path.append(np.deg2rad(q_deg))
    return path


def _joint_metrics(q_history):
    """Return maximum command step and frame-to-frame step discontinuity."""
    q_history = np.asarray(q_history, dtype=float)
    joint_steps = np.diff(q_history, axis=0)
    max_step_deg = float(
        np.max(np.abs(np.rad2deg(joint_steps)))
    )
    step_changes = np.diff(joint_steps, axis=0)
    max_continuity_deg = float(
        np.max(np.linalg.norm(np.rad2deg(step_changes), axis=1))
    )
    return max_step_deg, max_continuity_deg


def run_continuous_case(side, scenario, mode, frames=121):
    """Measure one previous-solution-seeded reachable target sequence."""
    model = RM75Model(side=side)
    q_targets = make_continuous_path(side, scenario, frames)
    q_preferred = q_targets[0].copy()
    q_current = q_preferred.copy()
    q_history = [q_current.copy()]
    iterations = []
    position_errors = []
    orientation_errors = []
    solve_times_ms = []
    failures = 0

    with damping_mode(mode):
        for q_target in q_targets:
            T_target = forward_kinematics(q_target, model=model)
            solve_start = time.perf_counter()
            result = solve_runtime_ik(
                T_target,
                q_current,
                q_preferred,
                model,
            )
            solve_times_ms.append(
                (time.perf_counter() - solve_start) * 1000.0
            )
            iterations.append(result["iterations"])
            position_errors.append(result["position_error"])
            orientation_errors.append(result["orientation_error"])
            if result["success"]:
                q_current = result["q"].copy()
            else:
                failures += 1
            q_history.append(q_current.copy())

    sigma_values = [
        minimum_singular_value(q_target, model)
        for q_target in q_targets
    ]
    max_step_deg, max_continuity_deg = _joint_metrics(q_history)
    return {
        "kind": "continuous",
        "side": side,
        "scenario": scenario,
        "mode": mode,
        "samples": len(q_targets),
        "failures": failures,
        "mean_iterations": float(np.mean(iterations)),
        "max_joint_step_deg": max_step_deg,
        "max_step_change_deg": max_continuity_deg,
        "max_position_error_mm": float(max(position_errors) * 1000.0),
        "max_orientation_error_deg": float(
            np.rad2deg(max(orientation_errors))
        ),
        "median_solve_ms": float(np.median(solve_times_ms)),
        "sigma_min": float(min(sigma_values)),
        "sigma_max": float(max(sigma_values)),
    }


def make_recovery_seeds(q_target, side, scenario, trials):
    """Return identical deterministic 2-degree seed perturbations per mode."""
    scenario_index = tuple(RECOVERY_Q4_DEG).index(scenario)
    random_seed = 7500 + (0 if side == "left" else 100) + scenario_index
    generator = np.random.default_rng(random_seed)
    return [
        q_target + np.deg2rad(generator.normal(0.0, 2.0, q_target.size))
        for _ in range(int(trials))
    ]


def run_recovery_case(side, scenario, mode, trials=32):
    """Stress convergence from deterministic seeds near each region."""
    if scenario not in RECOVERY_Q4_DEG:
        raise ValueError(f"unknown scenario: {scenario}")
    trials = int(trials)
    if trials <= 0:
        raise ValueError("trials must be positive")
    model = RM75Model(side=side)
    q_target_deg = REFERENCE_Q_DEG[side].copy()
    q4_sign = -1.0 if side == "left" else 1.0
    q_target_deg[3] = q4_sign * RECOVERY_Q4_DEG[scenario]
    q_target = np.deg2rad(q_target_deg)
    T_target = forward_kinematics(q_target, model=model)
    seeds = make_recovery_seeds(q_target, side, scenario, trials)
    iterations = []
    position_errors = []
    orientation_errors = []
    solve_times_ms = []
    corrections_deg = []
    failures = 0

    with damping_mode(mode):
        for q_seed in seeds:
            solve_start = time.perf_counter()
            result = solve_runtime_ik(
                T_target,
                q_seed,
                q_target,
                model,
            )
            solve_times_ms.append(
                (time.perf_counter() - solve_start) * 1000.0
            )
            iterations.append(result["iterations"])
            position_errors.append(result["position_error"])
            orientation_errors.append(result["orientation_error"])
            corrections_deg.append(
                np.max(np.abs(np.rad2deg(result["q"] - q_seed)))
            )
            if not result["success"]:
                failures += 1

    return {
        "kind": "recovery",
        "side": side,
        "scenario": scenario,
        "mode": mode,
        "samples": len(seeds),
        "failures": failures,
        "mean_iterations": float(np.mean(iterations)),
        "max_joint_step_deg": float(max(corrections_deg)),
        "max_position_error_mm": float(max(position_errors) * 1000.0),
        "max_orientation_error_deg": float(
            np.rad2deg(max(orientation_errors))
        ),
        "median_solve_ms": float(np.median(solve_times_ms)),
        "sigma_min": minimum_singular_value(q_target, model),
        "sigma_max": minimum_singular_value(q_target, model),
    }


def run_benchmark(continuous_frames=121, recovery_trials=32):
    """Run both policies on both arms in all required singularity regions."""
    continuous_rows = []
    recovery_rows = []
    for side in SIDES:
        for scenario in CONTINUOUS_SCENARIOS:
            for mode in MODES:
                continuous_rows.append(
                    run_continuous_case(
                        side,
                        scenario,
                        mode,
                        frames=continuous_frames,
                    )
                )
                recovery_rows.append(
                    run_recovery_case(
                        side,
                        scenario,
                        mode,
                        trials=recovery_trials,
                    )
                )
    return continuous_rows, recovery_rows


def evaluate_integration(continuous_rows, recovery_rows):
    """Apply conservative evidence gates before enabling adaptive damping."""
    reasons = []
    all_rows = continuous_rows + recovery_rows
    indexed = {
        (row["kind"], row["side"], row["scenario"], row["mode"]): row
        for row in all_rows
    }

    for kind in ("continuous", "recovery"):
        for side in SIDES:
            for scenario in CONTINUOUS_SCENARIOS:
                fixed = indexed[(kind, side, scenario, "fixed")]
                adaptive = indexed[(kind, side, scenario, "adaptive")]
                if adaptive["failures"] > fixed["failures"]:
                    reasons.append(
                        f"{kind}/{side}/{scenario}: failures "
                        f"{fixed['failures']} -> {adaptive['failures']}"
                    )

    singular_scenarios = (
        "sigma_warn",
        "sigma_stop",
        "near_q4_zero",
    )
    fixed_worst_step = max(
        row["max_joint_step_deg"]
        for row in continuous_rows
        if row["mode"] == "fixed"
        and row["scenario"] in singular_scenarios
    )
    adaptive_worst_step = max(
        row["max_joint_step_deg"]
        for row in continuous_rows
        if row["mode"] == "adaptive"
        and row["scenario"] in singular_scenarios
    )
    fixed_worst_continuity = max(
        row["max_step_change_deg"]
        for row in continuous_rows
        if row["mode"] == "fixed"
        and row["scenario"] in singular_scenarios
    )
    adaptive_worst_continuity = max(
        row["max_step_change_deg"]
        for row in continuous_rows
        if row["mode"] == "adaptive"
        and row["scenario"] in singular_scenarios
    )
    material_improvement = (
        adaptive_worst_step <= 0.90 * fixed_worst_step
        or adaptive_worst_continuity <= 0.90 * fixed_worst_continuity
    )
    if not material_improvement:
        reasons.append(
            "no >=10% worst-case joint-step or continuity improvement"
        )

    return len(reasons) == 0, reasons


def print_rows(title, rows):
    """Print one compact comparison table."""
    print(title)
    print(
        "side  scenario       mode      sigma range       fail/n "
        "iter  joint_deg  pos_mm  rot_deg  median_ms  continuity_deg"
    )
    for row in rows:
        continuity = row.get("max_step_change_deg")
        continuity_text = "-" if continuity is None else f"{continuity:.4f}"
        print(
            f"{row['side']:5s} {row['scenario']:14s} "
            f"{row['mode']:8s} "
            f"{row['sigma_min']:.4f}-{row['sigma_max']:.4f} "
            f"{row['failures']:2d}/{row['samples']:<3d} "
            f"{row['mean_iterations']:5.2f} "
            f"{row['max_joint_step_deg']:9.4f} "
            f"{row['max_position_error_mm']:7.4f} "
            f"{row['max_orientation_error_deg']:8.4f} "
            f"{row['median_solve_ms']:9.4f} {continuity_text:>14s}"
        )
    print()


def main():
    """Run the complete evidence gate and print the integration decision."""
    print("RM75 fixed-vs-adaptive damping evaluation")
    print(
        f"fixed={FIXED_DAMPING:.3f}; adaptive="
        f"{FIXED_DAMPING:.3f}..{ADAPTIVE_MAX_DAMPING:.3f}; "
        f"sigma_stop={SIGMA_STOP:.3f}; sigma_warn={SIGMA_WARN:.3f}"
    )
    print()
    continuous_rows, recovery_rows = run_benchmark()
    print_rows("CONTINUOUS PREVIOUS-SEED TRAJECTORIES", continuous_rows)
    print_rows("PERTURBED-SEED RECOVERY STRESS", recovery_rows)
    recommended, reasons = evaluate_integration(
        continuous_rows,
        recovery_rows,
    )
    if recommended:
        print("DECISION: adaptive damping satisfies the integration gates.")
    else:
        print("DECISION: keep production fixed damping=0.020.")
        for reason in reasons:
            print(f"- {reason}")


if __name__ == "__main__":
    main()
