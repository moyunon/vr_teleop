"""Validate the conservative, still-disabled first-motion profile."""

from pathlib import Path

import pytest
import yaml


CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"


def load_parameters(filename):
    """Load one fusion-node ROS parameter mapping."""
    with (CONFIG_DIR / filename).open(encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    return document["quest_dual_ik_fusion"]["ros__parameters"]


def test_default_and_first_motion_profiles_keep_actuation_disabled():
    """Require a separate explicit operator override after loading YAML."""
    normal = load_parameters("quest_dual_ik_fusion.yaml")
    first = load_parameters("safe_first_motion.yaml")

    assert normal["enable_robot_motion"] is False
    assert first["enable_robot_motion"] is False
    assert first["require_robot_state"] is True
    assert first["collision_protection_enabled"] is True


def test_first_motion_profile_uses_requested_conservative_rates():
    """Lock the initial Cartesian, qdot, qddot, and jump budgets."""
    parameters = load_parameters("safe_first_motion.yaml")

    assert parameters["max_cartesian_translation_rate_m_s"] == pytest.approx(
        0.05
    )
    assert parameters["max_cartesian_rotation_rate_rad_s"] == pytest.approx(
        0.30
    )
    assert parameters["joint_velocity_scale"] == pytest.approx(0.10)
    assert parameters["max_robot_command_delta_deg"] == pytest.approx(0.25)
    assert parameters["joint_acceleration_limit_deg_s2"] == pytest.approx(
        [30.0] * 7
    )


def test_first_motion_profile_keeps_all_runtime_guards_enabled():
    """Require deadman, collision, singularity, limits, and watchdog values."""
    parameters = load_parameters("safe_first_motion.yaml")

    assert parameters["collision_distance_timeout_s"] > 0.0
    assert parameters["command_timeout_s"] > 0.0
    assert parameters["deadman_input_timeout_s"] > 0.0
    assert parameters["deadman_grip_on_threshold"] > (
        parameters["deadman_grip_off_threshold"]
    )
    assert parameters["sigma_stop"] > 0.0
    assert parameters["sigma_warn"] > parameters["sigma_stop"]
    assert parameters["joint_soft_limit_margin_deg"] > 0.0
    assert parameters["max_consecutive_ik_failures"] >= 1
    assert parameters["following_warning_deg"] == pytest.approx([2.0] * 7)
    assert parameters["following_stop_deg"] == pytest.approx([5.0] * 7)
    assert parameters["following_persistence_s"] > 0.0


def test_unified_launch_keeps_motion_and_bag_default_off():
    """Statically lock the two side-effecting launch defaults."""
    launch_path = (
        Path(__file__).resolve().parents[1]
        / "launch"
        / "commissioning_dry_run.launch.py"
    )
    source = launch_path.read_text(encoding="utf-8")
    assert '"enable_robot_motion", default_value="false"' in source
    assert '"enable_bag_recording", default_value="false"' in source
    assert '"movej_canfd"' not in source
