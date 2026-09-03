import numpy as np

from scipy.spatial.transform import Rotation

from vr_rm75_teleop.rm75_model import RM75Model
from vr_rm75_teleop.rm75_fk import (
    forward_kinematics,
)


def rotation_y(angle_deg):

    angle = np.deg2rad(
        angle_deg
    )

    c = np.cos(angle)
    s = np.sin(angle)

    return np.array(
        [
            [ c, 0.0,  s],
            [0.0, 1.0, 0.0],
            [-s, 0.0,  c],
        ],
        dtype=float,
    )


def controller_pose_to_transform(
    pose,
):

    pose = np.asarray(
        pose,
        dtype=float,
    )

    # position:
    #
    # controller unit:
    # 0.001 mm
    #
    # 1 unit = 1e-6 m

    p = (
        pose[:3]
        * 1e-6
    )

    # orientation:
    #
    # controller unit:
    # 0.001 rad

    euler_xyz = (
        pose[3:]
        * 1e-3
    )

    R = (
        Rotation
        .from_euler(
            "xyz",
            euler_xyz,
        )
        .as_matrix()
    )

    T = np.eye(
        4,
        dtype=float,
    )

    T[:3, :3] = R
    T[:3, 3] = p

    return T


def check_arm(
    name,
    side,
    joint_raw,
    pose_raw,
    install_pitch_deg,
):
    # =========================================================
    # Build model from controller DH data
    # =========================================================

    model = RM75Model(
        side=side,
    )

    # get_DH_data actual value:
    #
    # d7 = 161.199 mm


    # =========================================================
    # Controller joint:
    #
    # unit = 0.001 deg
    # =========================================================

    q = np.deg2rad(
        np.asarray(
            joint_raw,
            dtype=float,
        )
        * 1e-3
    )

    # =========================================================
    # Our FK in SDH base
    # =========================================================

    T_sdh = forward_kinematics(
        q,
        model=model,
    )

    # =========================================================
    # Controller installation transform
    #
    # Current RealBot:
    #
    # LEFT  pitch = -90 deg
    # RIGHT pitch = +90 deg
    # =========================================================

    R_install = rotation_y(
        install_pitch_deg
    )

    T_install = np.eye(
        4,
        dtype=float,
    )

    T_install[:3, :3] = (
        R_install
    )

    T_predicted = (
        T_install
        @ T_sdh
    )

    # =========================================================
    # Controller reported pose
    # =========================================================

    T_controller = (
        controller_pose_to_transform(
            pose_raw
        )
    )

    # =========================================================
    # Position error
    # =========================================================

    position_error = np.linalg.norm(
        T_predicted[:3, 3]
        -
        T_controller[:3, 3]
    )

    # =========================================================
    # Orientation error
    # =========================================================

    R_error = (
        T_controller[:3, :3]
        @ T_predicted[:3, :3].T
    )

    orientation_error = np.linalg.norm(
        Rotation
        .from_matrix(
            R_error
        )
        .as_rotvec()
    )

    # =========================================================
    # Print
    # =========================================================

    print("")
    print("=" * 70)
    print(name)

    print("")
    print(
        "q [deg]:"
    )
    print(
        np.rad2deg(q)
    )

    print("")
    print(
        "FK in SDH base position [m]:"
    )
    print(
        T_sdh[:3, 3]
    )

    print("")
    print(
        "Predicted controller position [m]:"
    )
    print(
        T_predicted[:3, 3]
    )

    print("")
    print(
        "Reported controller position [m]:"
    )
    print(
        T_controller[:3, 3]
    )

    print("")
    print(
        "Position error [mm]:",
        position_error * 1000.0,
    )

    print(
        "Orientation error [deg]:",
        np.rad2deg(
            orientation_error
        ),
    )


def main():

    check_arm(
        name="LEFT ARM",

        side="left",

        joint_raw=[
            -64323,
            -33346,
            -53,
            -80689,
            8423,
            -47092,
            111354,
        ],

        pose_raw=[
            -217440,
            339793,
            -182204,
            2984,
            1328,
            -3045,
        ],

        install_pitch_deg=-90.0,
    )

    check_arm(
        name="RIGHT ARM",

        side="right",

        joint_raw=[
            21181,
            48006,
            32464,
            74966,
            21515,
            54393,
            -158266,
        ],

        pose_raw=[
            201052,
            318880,
            -227732,
            -105,
            1222,
            2372,
        ],

        install_pitch_deg=90.0,
    )


if __name__ == "__main__":
    main()