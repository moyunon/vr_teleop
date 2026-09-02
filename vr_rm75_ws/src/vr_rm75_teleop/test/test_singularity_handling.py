"""Tests for model-specific sigma classification and smooth rate scaling."""

import numpy as np
import pytest

from vr_rm75_teleop.rm75_jacobian import geometric_jacobian
from vr_rm75_teleop.rm75_model import RM75Model
from vr_rm75_teleop.target_feasibility import (
    minimum_singular_value,
    singularity_region,
    singularity_speed_scale,
    validate_singularity_thresholds,
)


@pytest.mark.parametrize("side,q4_sign", [("left", -1.0), ("right", 1.0)])
def test_q4_scan_crosses_model_specific_warning_and_stop_regions(
    side,
    q4_sign,
):
    """Confirm 0.010/0.020 against the current metre/radian Jacobian."""
    model = RM75Model(side=side)
    q = np.deg2rad([
        -40.0 if side == "left" else 20.0,
        -25.0 if side == "left" else 35.0,
        15.0 if side == "left" else 25.0,
        q4_sign * 30.0,
        10.0 if side == "left" else 15.0,
        -35.0 if side == "left" else 40.0,
        80.0 if side == "left" else -120.0,
    ])

    samples = {}
    for magnitude_deg in (15.0, 12.5, 5.0, 0.0):
        q[3] = np.deg2rad(q4_sign * magnitude_deg)
        samples[magnitude_deg] = minimum_singular_value(q, model)

    assert samples[15.0] >= 0.020
    assert 0.010 < samples[12.5] < 0.020
    assert samples[5.0] <= 0.010
    assert samples[0.0] < 1e-12
    assert singularity_region(samples[15.0]) == "safe"
    assert singularity_region(samples[12.5]) == "warning"
    assert singularity_region(samples[5.0]) == "stop"


def test_stacked_geometric_jacobian_uses_metre_and_radian_blocks():
    """Check unit axes and that sigma is the direct stacked-Jacobian SVD."""
    model = RM75Model(side="left")
    q = np.deg2rad([10.0, -20.0, 30.0, -40.0, -25.0, 35.0, 15.0])
    J = geometric_jacobian(q, model=model)

    assert J.shape == (6, model.DOF)
    assert np.allclose(np.linalg.norm(J[3:, :], axis=0), 1.0)
    assert np.max(np.abs(J[:3, :])) < np.sum(np.abs(model.d))
    assert minimum_singular_value(q, model) == pytest.approx(
        np.linalg.svd(J, compute_uv=False)[-1]
    )


def test_warning_scale_is_monotonic_smoothstep_with_exact_boundaries():
    """Avoid a Cartesian velocity jump at either sigma threshold."""
    assert singularity_speed_scale(0.005) == 0.0
    assert singularity_speed_scale(0.010) == 0.0
    assert singularity_speed_scale(0.015) == pytest.approx(0.5)
    assert singularity_speed_scale(0.020) == 1.0
    assert singularity_speed_scale(0.030) == 1.0

    sigmas = np.linspace(0.010, 0.020, 101)
    scales = np.array([singularity_speed_scale(value) for value in sigmas])
    assert np.all(np.diff(scales) >= 0.0)
    assert singularity_speed_scale(0.010 + 1e-8) < 1e-10
    assert 1.0 - singularity_speed_scale(0.020 - 1e-8) < 1e-10


@pytest.mark.parametrize(
    "sigma_stop,sigma_warn",
    [(-0.01, 0.02), (0.01, 0.01), (0.02, 0.01), (np.nan, 0.02)],
)
def test_invalid_singularity_thresholds_are_rejected(sigma_stop, sigma_warn):
    """Reject unordered or nonfinite policy thresholds at startup."""
    with pytest.raises(ValueError):
        validate_singularity_thresholds(sigma_stop, sigma_warn)


@pytest.mark.parametrize("sigma_min", [-0.1, np.nan, np.inf])
def test_invalid_sigma_is_rejected(sigma_min):
    """Never convert invalid Jacobian telemetry into a speed command."""
    with pytest.raises(ValueError):
        singularity_speed_scale(sigma_min)
