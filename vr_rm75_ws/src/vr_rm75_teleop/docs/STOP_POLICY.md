# RM75 dual-arm software stop policy

This policy is edge-triggered. A stop request is issued once when the Safety
Supervisor leaves `ENGAGED`; periodic `HOLD` evaluations do not resend it.
Both arm requests are written before waiting for either acknowledgement.

| Trigger while ENGAGED | Project class | RM75 JSON command |
|---|---|---|
| Intentional dual-grip deadman release | `CONTROLLED_STOP` | `set_arm_slow_stop` |
| VR/tracking/state stale, collision stop, following-error stop, watchdog, numeric/IK/limit fault, transport fault | `SAFETY_STOP` | `set_arm_stop` |

The expected positive responses are respectively
`{"command":"set_arm_slow_stop","arm_slow_stop":true}` and
`{"command":"set_arm_stop","arm_stop":true}`. Per-arm attempt, ACK, error,
and ACK latency are published in `/vr_rm75/stop_event`; the aggregate ACK is
published in `/vr_rm75/stop_acknowledged`.

In dry-run, the intended stop is logged and published, but no command socket
is created and no bytes are sent. Shutdown attempts `CONTROLLED_STOP` only if
this process previously acknowledged a motion command and an existing socket
is still available. It never reconnects during shutdown or stop handling.

`set_arm_stop` is a controller software stop, not the physical emergency-stop
circuit. A missing/malformed/negative/timeout ACK is an unresolved dangerous
condition: use the physical E-stop, keep clear of the workspace, and inspect
both controllers and networks. Software ACK does not prove physical arrest.

Protocol sources checked for this implementation:

- [RealMan motion configuration JSON protocol](https://develop.realman-robotics.com/en/robot/json/motionConfig/)
- [RealMan fourth-generation UDP configuration](https://develop.realman-robotics.com/en/robot4th/json/udpConfig/)

The installed controller generation/firmware must still be confirmed on site.

