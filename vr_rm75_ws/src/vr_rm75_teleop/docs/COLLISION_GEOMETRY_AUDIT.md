# Collision geometry audit

Source audited: `lsrx_rm75_dual_description/urdf/LSRX_RM75_DUAL.urdf` and its
repository meshes. This is a software-model audit, not a dimensional survey of
the assembled robot. “Mesh reused” means visual and collision reference the
same repository STL; it does not claim that the STL is a validated collision
approximation.

| Link(s) | Visual | Collision | Audit result |
|---|---|---|---|
| `base_link` | `base_link.STL` | 0.005 × 0.475 × 0.185 m box | Simplified primitive; assembled accuracy unverified |
| `wr_link`, `wl_link` | wheel STL | radius 0.085 m × length 0.0465 m cylinder | Simplified primitive |
| `xt_link_1` | STL | 0.135 × 0.144 × 0.255 m box | Simplified primitive |
| `xt_link_2` | STL | 0.13 × 0.023 × 0.3 m box | Simplified primitive |
| `dt_link`, `yb_link`, `xb_link`, `jb_link`, `tb_link` | STL | mesh reused | Present; dimensional survey pending |
| `tb_camera_link` | repository camera STL | **missing** | NOT READY; do not assume visual mesh is collision-certified |
| `xb_camera_link_1..3` | repository camera STLs | **missing** | NOT READY |
| `base_camera_link_1..2` | repository camera STLs | **missing** | NOT READY |
| `l_rm75_base_link`, `r_rm75_base_link` | RM75 base STL | mesh reused | Present |
| `l_rm75_link_1`, `r_rm75_link_1` | link 1 STL | mesh reused | Shoulder present |
| `l_rm75_link_2`, `r_rm75_link_2` | link 2 STL | mesh reused | Upper arm present |
| `l_rm75_link_3`, `r_rm75_link_3` | link 3 STL | mesh reused | Present |
| `l_rm75_link_4`, `r_rm75_link_4` | link 4 STL | mesh reused | Forearm/elbow present |
| `l_rm75_link_5`, `r_rm75_link_5` | link 5 STL | mesh reused | Wrist present |
| `l_rm75_link_6`, `r_rm75_link_6` | link 6 STL | mesh reused | Wrist present |
| `l_rm75_link_7`, `r_rm75_link_7` | link 7 STL | mesh reused | Flange present |
| `l_rm75_camera_rolink`, `r_rm75_camera_rolink` | camera bracket STL | mesh reused | End camera bracket present |
| `l_rm75_camera_link`, `r_rm75_camera_link` | camera STL | mesh reused | End camera present |
| `base_link_2` | STL | mesh reused | Present |
| four `wc_*_link` links | caster STL | radius 0.05 m sphere | Simplified primitive |
| `ltool_base_link`, `rtool_base_link` | tool base STL | mesh reused | End effector mount present |
| all `ltool_*_1/_2/Support_Link` and `rtool_*_1/_2/Support_Link` | tool STL | mesh reused | Gripper structure present; joint state source still needed |
| four `*Pad_Link` links | missing | missing | Frame/physical-pad intent must be verified on hardware |

The left/right arm mounts have explicit fixed transforms from the common
`xb_link` in the URDF. The backend evaluates these transforms rather than
using a guessed arm separation. Agreement with the physical side mounts is
`HARDWARE_PENDING`.

Current readiness blockers:

1. Six chassis camera links have visual meshes but no collision elements.
2. No surveyed table, wall, pedestal, or other environment geometry is
   configured in `collision_geometry.yaml`.
3. Non-arm movable body/gripper joints lack a measured state source or an
   externally verified fixed value.
4. `python-fcl` is not installed in the audited environment.

Therefore the collision backend intentionally publishes no distance snapshot
and `backend_ready=false`; the existing collision watchdog keeps the Safety
Supervisor closed. Do not set readiness manually. Resolve the four blockers,
then visually compare closest link pairs/points with the assembled system.

