import xml.etree.ElementTree as ET

import numpy as np

from scipy.spatial.transform import Rotation

from vr_rm75_teleop.rm75_model import RM75Model
from vr_rm75_teleop.rm75_fk import forward_kinematics


URDF_PATH = (
    "/home/moyu/workspace/vr_teleop/vr_rm75_ws/"
    "src/lsrx_rm75_dual_description/"
    "urdf/LSRX_RM75_DUAL.urdf"
)


def origin_to_transform(origin):

    xyz = np.fromstring(
        origin.attrib.get(
            "xyz",
            "0 0 0",
        ),
        sep=" ",
    )

    rpy = np.fromstring(
        origin.attrib.get(
            "rpy",
            "0 0 0",
        ),
        sep=" ",
    )

    T = np.eye(
        4,
        dtype=float,
    )

    T[:3, :3] = (
        Rotation
        .from_euler(
            "xyz",
            rpy,
        )
        .as_matrix()
    )

    T[:3, 3] = xyz

    return T


def rotation_z_transform(
    angle,
):

    c = np.cos(angle)
    s = np.sin(angle)

    T = np.eye(
        4,
        dtype=float,
    )

    T[:3, :3] = np.array(
        [
            [c, -s, 0.0],
            [s,  c, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    return T


def urdf_arm_fk(
    root,
    side,
    q,
):

    prefix = (
        "l"
        if side == "left"
        else "r"
    )

    T = np.eye(
        4,
        dtype=float,
    )

    for index in range(
        1,
        8,
    ):

        joint_name = (
            f"{prefix}_rm75_joint_{index}"
        )

        joint = root.find(
            f"./joint[@name='{joint_name}']"
        )

        if joint is None:
            raise RuntimeError(
                f"Joint not found: {joint_name}"
            )

        T_origin = (
            origin_to_transform(
                joint.find("origin")
            )
        )

        T_joint = (
            rotation_z_transform(
                q[index - 1]
            )
        )

        T = (
            T
            @ T_origin
            @ T_joint
        )

    return T


def pose_error(
    T_a,
    T_b,
):

    position_error = np.linalg.norm(
        T_a[:3, 3]
        -
        T_b[:3, 3]
    )

    R_error = (
        T_a[:3, :3]
        @ T_b[:3, :3].T
    )

    orientation_error = np.linalg.norm(
        Rotation
        .from_matrix(
            R_error
        )
        .as_rotvec()
    )

    return (
        position_error,
        orientation_error,
    )


def check_side(
    root,
    side,
):

    model = RM75Model(
        side=side,
    )

    test_configs_deg = [
        [
            0, 0, 0, 0, 0, 0, 0
        ],
        [
            10, -20, 30,
            40, -50, 60, 70
        ],
        [
            -45, 25, -35,
            55, 20, -40, 110
        ],
    ]

    print("")
    print("=" * 70)
    print(
        f"{side.upper()} ARM"
    )

    all_pass = True

    for index, q_deg in enumerate(
        test_configs_deg
    ):

        q = np.deg2rad(
            q_deg
        )

        T_model = (
            forward_kinematics(
                q,
                model=model,
            )
        )

        T_urdf = (
            urdf_arm_fk(
                root,
                side,
                q,
            )
        )

        pos_err, ori_err = (
            pose_error(
                T_model,
                T_urdf,
            )
        )

        pos_err_mm = (
            pos_err
            * 1000.0
        )

        ori_err_deg = (
            np.rad2deg(
                ori_err
            )
        )

        print("")
        print(
            f"Config {index + 1}"
        )

        print(
            "Position error [mm]:",
            pos_err_mm,
        )

        print(
            "Orientation error [deg]:",
            ori_err_deg,
        )

        if (
            pos_err_mm > 1e-6
            or
            ori_err_deg > 1e-6
        ):
            all_pass = False

    print("")

    if all_pass:
        print(
            f"{side.upper()} URDF FK: PASS"
        )
    else:
        print(
            f"{side.upper()} URDF FK: FAIL"
        )

    return all_pass


def main():

    tree = ET.parse(
        URDF_PATH
    )

    root = tree.getroot()

    left_pass = check_side(
        root,
        "left",
    )

    right_pass = check_side(
        root,
        "right",
    )

    print("")
    print("=" * 70)

    if (
        left_pass
        and right_pass
    ):
        print(
            "URDF / RM75 MODEL CONSISTENCY: PASS"
        )
    else:
        print(
            "URDF / RM75 MODEL CONSISTENCY: FAIL"
        )


if __name__ == "__main__":
    main()