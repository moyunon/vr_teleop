# Dual RM75 commissioning checklist

Stop immediately for unexpected motion, incorrect joint order/sign, stale
feedback, missing geometry, incomplete software-stop ACK, or any person/object
inside the exclusion zone. The physical E-stop is the authoritative emergency
stop; software stop is not a substitute.

## Gate A — read-only hardware check

Keep `enable_robot_motion=false`. Do not start a command socket.

- Verify left/right controller IP, TCP port, controller generation, firmware,
  and current official API documentation.
- Compare all seven joint positions against the teach pendant: order, radians
  conversion, sign, and plausible hard limits.
- Confirm connected, stale, enable, arm/joint error, and brake/status meaning.
- If passive UDP is enabled, confirm sender IP/port, seven `joint_speed` fields,
  actual sequence progression, period/effective rate, and unit conversion.
- Compare UDP velocity with a pendant-observed slow manual motion. The Gen-4
  0.02 rpm scale is `HARDWARE_PENDING` until this check passes.
- Record TCP query latency and verify stale detection by a controlled network
  disconnect while no commanded motion is active.

Pass evidence: bag/log, screenshots, firmware/API version, and operator signoff.

## Gate B — complete dry-run

Start:

```bash
ros2 launch vr_rm75_teleop commissioning_dry_run.launch.py \
  enable_robot_motion:=false enable_bag_recording:=true
```

- Confirm startup explicitly says dry-run and no command sockets are opened.
- Verify Quest poses, both grips, input freshness, measured joint state,
  qdot/qddot provenance, IK output, singularity, and safety diagnostics.
- Test grip release, VR loss, state stale, collision snapshot stale, malformed
  input, and following-error injection in a fake/offline test source.
- Confirm only one stop event appears per ENGAGED exit and dry-run reports zero
  network attempt.
- Confirm `timing_diagnostics` reports measured mean/max/p95/p99, effective
  frequency, jitter, and deadline misses; configuration at 50 Hz is not proof.
- Inspect the recorded closest collision pair against RViz/physical geometry.

Gate B cannot pass while `collision/backend_ready=false`.

## Gate C — stationary software-stop verification

Robot workspace clear, physical E-stop tested and held by a second operator,
speed limits at the approved minimum, and no teleoperation motion requested.

- Confirm official stop command/response fields on the installed controller.
- Enable command output only under the site-approved procedure.
- From a stationary state, verify controlled stop and safety stop separately.
- Require positive ACK from both arms and measure each ACK RTT.
- Disconnect one test path only under an approved safe procedure and confirm
  incomplete ACK latches the software fault and requires physical intervention.

This gate requires on-site authorization and is not completed by unit tests.

## Gate D — first extremely slow motion

- Use `safe_first_motion.yaml`; do not increase its 0.05 m/s translation,
  0.30 rad/s rotation, 0.10 joint velocity scale, or 30 deg/s² qddot limits.
- Start with one arm, a tiny displacement, full clearance, spotter at E-stop,
  and bag recording. Keep the other arm stationary and monitored.
- Validate direction, scale, latency, following error, stop response, and
  collision closest pair before attempting the second arm.
- Calibrate provisional following-error thresholds from measured latency and
  interpolation data. Do not loosen thresholds merely to suppress a stop.
- Progress to dual-arm tests only after documented single-arm acceptance.

## Recorder topics

The unified launch records Quest pose/grip/freshness, deadman, per-arm and dual
measured joints, measured qdot/qddot, safe command joints, following error,
IK/sigma/limiter diagnostics, five collision distances and backend quality,
safety state, stop events/ACK,
robot command status, and timing diagnostics when
`enable_bag_recording:=true`.

The launch default is false for both motion and bag recording. Bag recording
does not authorize motion.
