import numpy as np

from vr_rm75_teleop.rm75_model import RM75Model
from vr_rm75_teleop.rm75_fk import forward_kinematics


np.set_printoptions(
    precision=6,
    suppress=True,
)


def main():

    model = RM75Model(
        side="left",
    )

    # ============================================================
    # Test 1:
    # 所有关节角 = 0
    # ============================================================

    q_zero = np.zeros(7)

    T_07, transforms = forward_kinematics(
        q_zero,
        model=model,
        return_all=True,
    )

    print("")
    print("==============================")
    print("RM75 FK - ZERO CONFIGURATION")
    print("==============================")

    for i, T in enumerate(transforms):

        print("")
        print(f"T_0{i} =")
        print(T)

    print("")
    print("Final T_07:")
    print(T_07)

    # ============================================================
    # 标准 RM75-B 零位理论结果
    # ============================================================

    expected_zero = np.array(
        [
            [-1.0,  0.0,  0.0, 0.0],
            [ 0.0, -1.0,  0.0, 0.0],
            [ 0.0,  0.0,  1.0, 0.867699],
            [ 0.0,  0.0,  0.0, 1.0],
        ],
        dtype=float,
    )

    if np.allclose(
        T_07,
        expected_zero,
        atol=1e-9,
    ):
        print("")
        print("ZERO TEST: PASS")

    else:
        print("")
        print("ZERO TEST: FAIL")
        print("Expected:")
        print(expected_zero)

    # ============================================================
    # Test 2:
    # 非零关节角
    # ============================================================

    q_test = np.deg2rad(
        [
            0.0,
            30.0,
            0.0,
            60.0,
            0.0,
            30.0,
            0.0,
        ]
    )

    T_test = forward_kinematics(
        q_test,
        model=model,
    )

    print("")
    print("==============================")
    print("RM75 FK - TEST CONFIGURATION")
    print("==============================")

    print("")
    print(
        "q [deg] = "
        "[0, 30, 0, 60, 0, 30, 0]"
    )

    print("")
    print("T_07 =")
    print(T_test)


if __name__ == "__main__":
    main()