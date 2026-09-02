# Commissioning implementation status

## Software complete

- Task A: strict TCP/UDP feedback parsing, sender validation, UDP direct joint
  speed, measurement sequence/period/effective rate, freshness and diagnostics.
- Task B: controlled versus safety software stops, ENGAGED-exit edge trigger,
  both-arm send/ACK/error/latency, dry-run zero-network behavior, and best-effort
  armed shutdown.
- Task C implementation: URDF parser/kinematics, FCL mesh/primitives, retained
  five-category architecture with an explicit enabled subset, structural pair
  filtering, environment config, quality/closest-pair/timing diagnostics, and
  fail-closed ROS adapter.
- Task D: timestamp-aware following error with persistence/hysteresis, direct
  qdot plus explicit finite-difference fallback, actual-dt filtered qddot,
  rolling timing/jitter/deadline metrics, unified launch, recorder, and checklist.

## Simulated test pass

Pure-Python protocol, state parser, stop policy, feedback estimator, following
error, collision category/filter/readiness, consumer threshold, Supervisor, and
timing tests pass without opening real network sockets.

## RM75-only collision commissioning result

- `python-fcl` imports successfully; the real URDF parses 56 collision
  geometries and the configured 16-link RM75-only backend is geometry-ready.
- Enabled: left self, right self, and inter-arm. Disabled by configuration:
  environment and robot body. Six chassis-camera collision omissions and an
  empty environment are outside this explicitly narrowed claim.
- A read-only live static state supplied only the 14 arm joints and produced
  0.053036446 m left self, 0.053015256 m right self, and 0.000000000 m
  inter-arm. The last value is the left-base/right-base pair and is a collision
  STOP under current thresholds.
- First mesh-backed solve measured about 710 ms; cached solves measured about
  93–100 ms on this host. That is borderline for the 100 ms collision watchdog
  and insufficient for a 50 Hz update. Timing/architecture optimization and an
  on-site decision about the base-base geometry remain commissioning blockers.

The narrowed backend is therefore configuration/geometry ready, but the
current static state is not collision-clear and the measured update rate is not
yet qualified for real motion.

## Hardware pending

- Installed RM75 generation/firmware and stop response compatibility.
- UDP `joint_speed` unit/scale and actual packet rate.
- Joint order/sign/enable/error/brake comparison with both teach pendants.
- Physical arm-mount transform and every in-scope RM75 collision mesh.
- Determine whether the base-base zero distance is a true unsafe overlap or a
  verified permanent structural pair before changing any ignore configuration.
- Stop ACK versus observed physical arrest and measured stop distance/time.
- Following-error thresholds, command ACK RTT, timing distribution, and first
  single-arm/dual-arm motion evidence.

No command socket was opened and no real motion was sent. Hardware interaction
was limited to one already-published, read-only static state sample.

## Repository hygiene note

`build/`, `install/`, and `log/` currently contain tracked artifacts. They were
not deleted because that would be destructive and unrelated user history may
depend on them. Before running, source only the intended workspace prefix and
verify executable/package paths with `ros2 pkg prefix vr_rm75_teleop`; rebuild
in a clean reviewed environment if the prefix is stale.
