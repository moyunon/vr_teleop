# Commissioning implementation status

## Software complete

- Task A: strict TCP/UDP feedback parsing, sender validation, UDP direct joint
  speed, measurement sequence/period/effective rate, freshness and diagnostics.
- Task B: controlled versus safety software stops, ENGAGED-exit edge trigger,
  both-arm send/ACK/error/latency, dry-run zero-network behavior, and best-effort
  armed shutdown.
- Task C implementation: URDF parser/kinematics, FCL mesh/primitives, all five
  atomic distance categories, structural pair filtering, environment config,
  quality/closest-pair/timing diagnostics, and fail-closed ROS adapter.
- Task D: timestamp-aware following error with persistence/hysteresis, direct
  qdot plus explicit finite-difference fallback, actual-dt filtered qddot,
  rolling timing/jitter/deadline metrics, unified launch, recorder, and checklist.

## Simulated test pass

Pure-Python protocol, state parser, stop policy, feedback estimator, following
error, collision category/filter/readiness, consumer threshold, Supervisor, and
timing tests pass without opening real network sockets.

## Dry-run readiness blockers

- Install/validate the declared `python-fcl` runtime through the approved system
  dependency process; it was not installed automatically.
- Add audited collision elements for six chassis camera links.
- Measure/configure the environment and source all non-arm movable joint poses.
- Verify the launch in a ROS environment containing `rclpy` and required nodes.

Until those items are resolved, collision stays NOT READY and Safety Supervisor
correctly prevents ENGAGED motion output.

## Hardware pending

- Installed RM75 generation/firmware and stop response compatibility.
- UDP `joint_speed` unit/scale and actual packet rate.
- Joint order/sign/enable/error/brake comparison with both teach pendants.
- Physical arm-mount transform and every collision mesh/dimension.
- Stop ACK versus observed physical arrest and measured stop distance/time.
- Following-error thresholds, command ACK RTT, timing distribution, and first
  single-arm/dual-arm motion evidence.

No hardware validation or real motion was performed by this software task.

## Repository hygiene note

`build/`, `install/`, and `log/` currently contain tracked artifacts. They were
not deleted because that would be destructive and unrelated user history may
depend on them. Before running, source only the intended workspace prefix and
verify executable/package paths with `ros2 pkg prefix vr_rm75_teleop`; rebuild
in a clean reviewed environment if the prefix is stale.
