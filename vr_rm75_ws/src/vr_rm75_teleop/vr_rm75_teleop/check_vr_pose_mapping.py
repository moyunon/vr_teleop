import numpy as np

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
):

    T = np.eye(
        4,
        dtype=float,
    )

    T[:3, 3] = np.asarray(
        position,
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


if __name__ == "__main__":
    main()