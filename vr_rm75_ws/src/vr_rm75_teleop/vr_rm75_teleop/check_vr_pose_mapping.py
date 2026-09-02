import numpy as np

from scipy.spatial.transform import Rotation

from vr_rm75_teleop.vr_pose_mapping import (
    C_VR_TO_LEFT_ARM,
    C_VR_TO_RIGHT_ARM,
    map_vr_pose_to_robot_target,
)


np.set_printoptions(
    precision=6,
    suppress=True,
)


def make_pose(
    position,
    rotation=None,
):

    T = np.eye(
        4,
        dtype=float,
    )

    T[:3, 3] = np.asarray(
        position,
        dtype=float,
    )

    if rotation is not None:
        T[:3, :3] = np.asarray(
            rotation,
            dtype=float,
        )

    return T


def check_translation(
    side,
    vr_delta,
    expected_arm_delta,
):

    T_vr_anchor = np.eye(
        4,
        dtype=float,
    )

    T_vr_current = make_pose(
        vr_delta
    )

    T_ee_anchor = np.eye(
        4,
        dtype=float,
    )

    T_target = (
        map_vr_pose_to_robot_target(
            T_vr_anchor=
                T_vr_anchor,

            T_vr_current=
                T_vr_current,

            T_ee_anchor=
                T_ee_anchor,

            side=side,

            position_scale=1.0,
        )
    )

    actual = (
        T_target[:3, 3]
    )

    expected = np.asarray(
        expected_arm_delta,
        dtype=float,
    )

    passed = np.allclose(
        actual,
        expected,
        atol=1e-12,
    )

    print(
        f"{side.upper():5s}",
        "actual =",
        actual,
        "expected =",
        expected,
        "PASS" if passed else "FAIL",
    )

    return passed


def check_orientation(
    side,
    label,
    vr_rotvec,
    expected_arm_rotvec,
):

    T_vr_anchor = np.eye(
        4,
        dtype=float,
    )

    T_vr_current = make_pose(
        position=[0.0, 0.0, 0.0],
        rotation=(
            Rotation
            .from_rotvec(vr_rotvec)
            .as_matrix()
        ),
    )

    T_target = map_vr_pose_to_robot_target(
        T_vr_anchor=T_vr_anchor,
        T_vr_current=T_vr_current,
        T_ee_anchor=np.eye(4, dtype=float),
        side=side,
        orientation_scale=1.0,
    )

    actual = (
        Rotation
        .from_matrix(T_target[:3, :3])
        .as_rotvec()
    )

    expected = np.asarray(
        expected_arm_rotvec,
        dtype=float,
    )

    passed = np.allclose(
        actual,
        expected,
        atol=1e-12,
        rtol=0.0,
    )

    print(
        f"{side.upper():5s}",
        f"{label:7s}",
        "actual =",
        actual,
        "expected =",
        expected,
        "PASS" if passed else "FAIL",
    )

    return passed


def main():

    print("")
    print(
        "C_VR_TO_LEFT_ARM ="
    )
    print(
        C_VR_TO_LEFT_ARM
    )

    print("")
    print(
        "C_VR_TO_RIGHT_ARM ="
    )
    print(
        C_VR_TO_RIGHT_ARM
    )

    print("")
    print("=" * 70)
    print(
        "TEST 1: physical forward"
    )
    print(
        "VR delta = [-0.1, 0, 0]"
    )

    results = []

    results.append(
        check_translation(
            side="left",
            vr_delta=[
                 0.1,
                 0.0,
                 0.0,
            ],
            expected_arm_delta=[
                0.0,
                0.1,
                0.0,
            ],
        )
    )

    results.append(
        check_translation(
            side="right",
            vr_delta=[
                 0.1,
                 0.0,
                 0.0,
            ],
            expected_arm_delta=[
                0.0,
                0.1,
                0.0,
            ],
        )
    )

    print("")
    print("=" * 70)
    print(
        "TEST 2: physical right"
    )
    print(
        "VR delta = [0, +0.1, 0]"
    )

    results.append(
        check_translation(
            side="left",
            vr_delta=[
                0.0,
                -0.1,
                0.0,
            ],
            expected_arm_delta=[
                 0.0,
                 0.0,
                -0.1,
            ],
        )
    )

    results.append(
        check_translation(
            side="right",
            vr_delta=[
                0.0,
                -0.1,
                0.0,
            ],
            expected_arm_delta=[
                0.0,
                0.0,
                0.1,
            ],
        )
    )

    print("")
    print("=" * 70)
    print(
        "TEST 3: physical up"
    )
    print(
        "VR delta = [0, 0, +0.1]"
    )

    results.append(
        check_translation(
            side="left",
            vr_delta=[
                0.0,
                0.0,
                0.1,
            ],
            expected_arm_delta=[
                0.1,
                0.0,
                0.0,
            ],
        )
    )

    results.append(
        check_translation(
            side="right",
            vr_delta=[
                0.0,
                0.0,
                0.1,
            ],
            expected_arm_delta=[
               -0.1,
                0.0,
                0.0,
            ],
        )
    )

    print("")
    print("=" * 70)

    print(
        "TEST 4: orientation axis and sign"
    )

    angle = np.deg2rad(10.0)

    orientation_cases = {
        "left": [
            ("+roll", [angle, 0.0, 0.0], [0.0, angle, 0.0]),
            ("-roll", [-angle, 0.0, 0.0], [0.0, -angle, 0.0]),
            ("+pitch", [0.0, angle, 0.0], [0.0, 0.0, angle]),
            ("-pitch", [0.0, -angle, 0.0], [0.0, 0.0, -angle]),
            ("+yaw", [0.0, 0.0, angle], [angle, 0.0, 0.0]),
            ("-yaw", [0.0, 0.0, -angle], [-angle, 0.0, 0.0]),
        ],
        "right": [
            ("+roll", [angle, 0.0, 0.0], [0.0, angle, 0.0]),
            ("-roll", [-angle, 0.0, 0.0], [0.0, -angle, 0.0]),
            ("+pitch", [0.0, angle, 0.0], [0.0, 0.0, -angle]),
            ("-pitch", [0.0, -angle, 0.0], [0.0, 0.0, angle]),
            ("+yaw", [0.0, 0.0, angle], [-angle, 0.0, 0.0]),
            ("-yaw", [0.0, 0.0, -angle], [angle, 0.0, 0.0]),
        ],
    }

    for side, cases in orientation_cases.items():
        for label, vr_rotvec, expected_arm_rotvec in cases:
            results.append(
                check_orientation(
                    side=side,
                    label=label,
                    vr_rotvec=vr_rotvec,
                    expected_arm_rotvec=expected_arm_rotvec,
                )
            )

    print("")
    print("=" * 70)

    if all(
        results
    ):
        print(
            "VR POSITION MAPPING TEST: PASS"
        )
    else:
        print(
            "VR POSITION MAPPING TEST: FAIL"
        )

        raise SystemExit(1)


if __name__ == "__main__":
    main()
