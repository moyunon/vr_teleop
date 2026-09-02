"""Tests for the offline fixed-versus-adaptive damping evidence gate."""

import numpy as np
import pytest

import vr_rm75_teleop.rm75_ik as ik_module
from vr_rm75_teleop.check_adaptive_damping import (
    adaptive_damping_value,
    damping_mode,
    evaluate_integration,
    run_benchmark,
)


def test_adaptive_candidate_is_continuous_and_leaves_safe_region_unchanged():
    """Use fixed damping in safe space and smoothly increase near stop."""
    assert adaptive_damping_value(0.030) == pytest.approx(0.020)
    assert adaptive_damping_value(0.020) == pytest.approx(0.020)
    assert adaptive_damping_value(0.015) == pytest.approx(0.0225)
    assert adaptive_damping_value(0.010) == pytest.approx(0.025)
    assert adaptive_damping_value(0.001) == pytest.approx(0.025)
    assert adaptive_damping_value(0.010 + 1e-8) == pytest.approx(
        0.025,
        abs=1e-12,
    )


@pytest.mark.parametrize(
    "base_damping,max_damping",
    [(0.0, 0.025), (np.nan, 0.025), (0.02, 0.019), (0.02, np.inf)],
)
def test_invalid_adaptive_damping_bounds_are_rejected(
    base_damping,
    max_damping,
):
    """Reject nonfinite, nonpositive, or reversed damping bounds."""
    with pytest.raises(ValueError):
        adaptive_damping_value(
            0.015,
            base_damping=base_damping,
            max_damping=max_damping,
        )


def test_experimental_mode_always_restores_production_dls_function():
    """Keep the benchmark hook isolated from later solver calls."""
    original = ik_module.damped_least_squares_step
    with damping_mode("adaptive"):
        assert ik_module.damped_least_squares_step is not original
    assert ik_module.damped_least_squares_step is original


def test_current_evidence_rejects_adaptive_integration():
    """Adaptive damping adds failures near stop under the runtime budget."""
    continuous_rows, recovery_rows = run_benchmark(
        continuous_frames=31,
        recovery_trials=16,
    )
    recommended, reasons = evaluate_integration(
        continuous_rows,
        recovery_rows,
    )

    assert recommended is False
    assert any("failures" in reason for reason in reasons)
    fixed_stop_failures = sum(
        row["failures"]
        for row in recovery_rows
        if row["mode"] == "fixed" and row["scenario"] == "sigma_stop"
    )
    adaptive_stop_failures = sum(
        row["failures"]
        for row in recovery_rows
        if row["mode"] == "adaptive" and row["scenario"] == "sigma_stop"
    )
    assert adaptive_stop_failures > fixed_stop_failures
