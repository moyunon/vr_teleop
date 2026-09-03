import numpy as np

from scipy.spatial.transform import Rotation

from vr_rm75_teleop.se3_rate_limiter import (
    limit_pose_step,
)


def main():

    T_current = np.eye(
        4,
        dtype=float,
    )

    T_desired = np.eye(
        4,
        dtype=float,
    )

    # 希望一次跳 10 cm
    T_desired[0, 3] = 0.10

    # 希望一次旋转 90 deg
    T_desired[:3, :3] = (
        Rotation
        .from_euler(
            "z",
            90.0,
            degrees=True,
        )
        .as_matrix()
    )

    result = (
        limit_pose_step(
            T_current=
                T_current,

            T_desired=
                T_desired,

            max_translation_step=
                0.005,

            max_rotation_step=
                np.deg2rad(
                    2.0
                ),
        )
    )

    print(
        "Requested translation [mm]:",
        result[
            "translation_distance"
        ] * 1000.0,
    )

    print(
        "Actual translation step [mm]:",
        result[
            "translation_step"
        ] * 1000.0,
    )

    print(
        "Requested rotation [deg]:",
        np.rad2deg(
            result[
                "rotation_distance"
            ]
        ),
    )

    print(
        "Actual rotation step [deg]:",
        np.rad2deg(
            result[
                "rotation_step"
            ]
        ),
    )

    translation_pass = np.isclose(
        result[
            "translation_step"
        ],
        0.005,
    )

    rotation_pass = np.isclose(
        result[
            "rotation_step"
        ],
        np.deg2rad(
            2.0
        ),
    )

    if (
        translation_pass
        and rotation_pass
    ):

        print(
            "SE3 RATE LIMITER TEST: PASS"
        )

    else:

        print(
            "SE3 RATE LIMITER TEST: FAIL"
        )


if __name__ == "__main__":
    main()