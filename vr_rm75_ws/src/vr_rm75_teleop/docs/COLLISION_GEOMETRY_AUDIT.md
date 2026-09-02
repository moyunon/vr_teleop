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
| `tb_camera_link` | repository camera STL | **missing** | Out of current RM75-only scope |
| `xb_camera_link_1..3` | repository camera STLs | **missing** | Out of current RM75-only scope |
| `base_camera_link_1..2` | repository camera STLs | **missing** | Out of current RM75-only scope |
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

## Current RM75-only commissioning scope

`collision_geometry.yaml` explicitly enables `left_self`, `right_self`, and
`inter_arm`. It explicitly disables `environment` and `robot_body`. The only
monitored links are `l_rm75_base_link`, `l_rm75_link_1..7`,
`r_rm75_base_link`, and `r_rm75_link_1..7`.

Accordingly, geometry readiness applies only to those 16 links. The six
missing chassis-camera collision elements, empty environment, body/head/base,
end cameras, and modeled tool/gripper branches do not block this narrowed
backend. Online transforms use the arms' common `xb_link` frame and require
only `l_rm75_joint_1..7` and `r_rm75_joint_1..7`. Parent-child and explicitly
ignored pair filtering remains active.

The audited runtime imports `python-fcl`, parses 56 existing collision
geometries with no load failure, and reports the narrowed backend geometry
ready. Disabled categories have no numerical placeholder; diagnostics report
`DISABLED_BY_CONFIGURATION`.

This readiness is not a motion-clearance result. A 2026-09-02 read-only static
state sample produced `left_self=0.053036446 m`,
`right_self=0.053015256 m`, and `inter_arm=0.000000000 m`; the limiting pair was
`l_rm75_base_link:0` ↔ `r_rm75_base_link:0`. Under the configured 0.05 m stop
threshold this is STOP. Do not ignore that pair unless its permanent overlap
and structural relationship are verified against the assembled hardware.
