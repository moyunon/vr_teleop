# RM75 command interface safety decision

## Selected command

The actuator boundary uses the RealMan JSON `movej_canfd` command with
`follow=false` (low-follow mode). High-follow mode is intentionally absent
from the API exposed by this project.

The reasons are:

- `quest_dual_ik_fusion` normally runs at 50 Hz, or one point every 20 ms.
- RealMan requires high-follow `movej_canfd` points at intervals no longer
  than 10 ms. The current loop cannot satisfy that contract.
- RealMan documents `movej_follow` as controller-planned motion, but also
  describes it as unsuitable for high-frequency pass-through control.
- `movej_canfd` accepts the already limited target stream without controller
  trajectory planning. This project therefore performs Cartesian, IK,
  singularity, soft-limit, command-delta, and qdot shaping before transmission.

Official references:

- <https://develop.realman-robotics.com/robot/apipython/classes/movePlan/>
- <https://develop.realman-robotics.com/robot/json/motionConfig/>
- <https://develop.realman-robotics.com/robot/json/readme/>
- <https://develop.realman-robotics.com/robot/demo/python/movejCANFD/>

Every installation must confirm its controller model, firmware/API generation,
and low-follow behavior against both physical RM75 controllers before motion.
A version mismatch is a commissioning blocker.

Gate C raw-protocol commissioning confirmed RM75-6FB controller generation
4.0, firmware V1.0.4.t11: successful `movej_canfd` with `follow=false` emits no
per-frame TCP response, while `set_arm_stop` returns its documented ACK. The
production response mode for this firmware is therefore `send_only`.

## Timing and transport behavior

TCP connect/write operations are bounded by
`robot_command_transport_timeout_s` (default 10 ms). With the default
`movej_response_mode=send_only`, LEFT and RIGHT are written in that order and
successful `sendall()` calls report only transport-send success. They do not
mean the controller accepted a target or that the robot physically reached
it. Direct read-only RM75 feedback and the following-error monitor provide the
independent motion supervision.

`movej_response_mode=joint_state_ack` remains available as an explicit
diagnostic/compatibility mode for firmware that returns the documented
`joint_state`. It uses the separate provisional
`movej_response_timeout_s=50 ms`; it is not part of the current production
command cadence. Software stops always retain their independent ACK and use
`stop_response_timeout_s=10 ms`.

The command payload contains seven controller-joint angles in integer 0.001
degrees and always includes `"follow": false`. Both complete messages are
encoded and validated before either socket is written.

The response frame, seven reported joints, and integer `arm_err` are validated.
In optional movej ACK mode, an ACK receive timeout leaves that socket
temporarily open because the controller may already have executed the command.
The dispatcher immediately attempts `SAFETY_STOP` on both still-open channels,
collects the stop ACKs, then closes both channels and latches a global FAULT.
The same stop-before-close sequence applies to a send failure wherever a
channel remains usable. EOF, reset, or another broken-socket error closes the
unavailable side immediately; the peer is still stopped. Any incomplete stop
result explicitly requires the physical E-stop.

The two robot controllers are independent network endpoints, so a truly atomic
dual-arm network commit is impossible. If one write succeeds and its peer
fails, both sockets are closed and the global FAULT prevents any continued
one-arm following. An operator reset with the deadman released is required
before reconnecting.

## Motion gates

`enable_robot_motion=false` is the default. In that state:

- no command socket is opened;
- no motion payload can be sent;
- real state input, IK, safety evaluation, RViz output, and dry-run command
  generation continue normally.

When the parameter is explicitly true, startup additionally rejects unsafe
static combinations: real state and collision protection must be enabled, the
joint velocity scale must be at most 10%, and the control rate must be at least
50 Hz. Runtime transmission still requires the Safety Supervisor to be
`ENGAGED`, including fresh robot/VR/collision data, valid numerics, deadman,
soft-limit, singularity, velocity, safe-command watchdog, command-channel, and
output-watchdog checks.

The final pre-send boundary independently rejects:

- malformed, non-finite, or hard-limit joint values;
- command delta greater than `max_robot_command_delta_deg`;
- joint velocity greater than the configured per-joint limit;
- joint acceleration greater than `joint_acceleration_limit_deg_s2`;
- stale or repeated safe targets;
- a closed Safety Supervisor command gate.

An abnormal joint jump is rejected into HOLD. It is not clipped into a hidden
trajectory. During normal engaged motion, a stateful limiter bounds the change
in commanded joint velocity on every cycle and brakes toward a fixed target.
The Safety Supervisor and final command boundary independently recompute qdot
and qddot and reject any violation. Emergency safety gates still stop command
transmission immediately; software acceleration shaping never delays a
deadman, collision, watchdog, numeric, or communication stop.

## First-motion profile

`config/safe_first_motion.yaml` is the conservative commissioning profile. It
sets Cartesian rates to 0.05 m/s and 0.30 rad/s, joint velocity scale to 10%,
and the commanded joint acceleration budget to 30 deg/s^2. Collision,
deadman, singularity, soft-limit, robot-state, safe-command, and collision-data
watchdogs remain enabled.

Loading the profile does not authorize actuator output because it retains
`enable_robot_motion=false`. An operator must first complete dry-run checks and
confirm the physical controller/API behavior, then make a separate explicit
override to enable motion. The normal project configuration also defaults to
motion disabled.

## Commissioning status

Zero-delta commissioning must measure LEFT/RIGHT `sendall()` duration and the
independent feedback delay. `send_only` removes a nonexistent movej ACK from
the 50 Hz cadence; it does not remove feedback freshness, following-error, or
any other safety requirement. Stop ACK latency remains separately measured.

This implementation and its automated tests use fake sockets only. No test in
this stage connects to either real controller or transmits a motion command.
