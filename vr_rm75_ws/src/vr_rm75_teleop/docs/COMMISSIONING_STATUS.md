# Commissioning implementation status

## Software complete

- Task A: strict TCP/UDP feedback parsing, sender validation, UDP direct joint
  speed, measurement sequence/period/effective rate, freshness and diagnostics.
- Task B: controlled versus safety software stops, ENGAGED-exit edge trigger,
  both-arm send/ACK/error/latency, dry-run zero-network behavior, and best-effort
  armed shutdown.
- Task C implementation: URDF parser/kinematics, FCL mesh/primitives, retained
  five-category architecture with an explicit enabled subset, structural pair
  filtering, environment config, per-category/global closest-pair diagnostics,
  source-age timing diagnostics, persistent FCL objects with exact AABB
  pruning, and a fail-closed latest-state ROS adapter.
- Task D: timestamp-aware following error with persistence/hysteresis, direct
  qdot plus explicit finite-difference fallback, actual-dt filtered qddot,
  rolling timing/jitter/deadline metrics, unified launch, recorder, and checklist.
- Following-error re-engagement hardening: every `READY -> ENGAGED` edge uses
  `robot_command_gate_open_since` as a new epoch. Each arm independently waits
  for its first safe command timestamp from that epoch; previous-epoch commands
  cannot enter the comparison. Once admitted, the existing command/measurement
  freshness, timestamp-skew, persistence, and tracking-error protections remain
  fail-closed.
- Actuator engagement bootstrap uses that same epoch. While either arm lacks a
  current-engagement safe command, dispatch reports
  `AWAITING_FIRST_SAFE_COMMAND` with per-arm readiness, sends no `movej_canfd`,
  and does not request a stop. The unchanged command-output watchdog remains
  bounded by `command_timeout_s`; expiry still causes SAFETY_STOP and HOLD.
  Only safe commands genuinely generated after anchor capture and IK/limiting
  can carry a current-epoch timestamp into the first dual-arm dispatch.

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
  0.053036446 m left self, 0.053015256 m right self, and 0.137900000 m
  inter-arm after formally excluding only the fixed left-base/right-base pair.
  Opposite-arm base-to-moving-link pairs remain monitored.
- The same static sample reports closest pairs left link 4 ↔ left link 6,
  right link 4 ↔ right link 6, and left link 1 ↔ right base. It yields WARNING
  (right self), not STOP, with the unchanged 0.05/0.15 m thresholds.
- On 2026-09-03, cold initialization plus first evaluation measured 666.707 ms
  (628.778 ms FCL preload; 26.709 ms first evaluation). Across 100 warm exact
  evaluations: mean 27.281 ms, p50 26.994 ms, p95 28.657 ms, p99 31.986 ms,
  max 38.110 ms. A separate exhaustive all-pair evaluation matched every
  optimized category distance exactly. This is not a 50 Hz collision solve,
  but latest-state depth-1 input and sub-100 ms output-age rejection allow the
  asynchronous collision stream to support the 50 Hz controller watchdog on
  this measured host.

The narrowed backend is configuration/geometry ready and its current static
state is outside STOP, but it remains inside WARNING with only about 3 mm above
the stop boundary. That is not authorization for real motion.

## Hardware pending

- Installed RM75 generation/firmware and stop response compatibility.
- UDP `joint_speed` unit/scale and actual packet rate.
- Joint order/sign/enable/error/brake comparison with both teach pendants.
- Physical arm-mount transform and every in-scope RM75 collision mesh.
- Confirm the fixed base-base structural relationship and inspect the three
  reported closest pairs against the assembled mechanism.
- Calibrate collision thresholds only after the close link-4/link-6 self pairs
  are physically confirmed; do not loosen the watchdog or thresholds to clear
  commissioning.
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
