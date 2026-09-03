"""Meta Quest Reader."""

import os
import math
import select
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Literal

import numpy as np
from ppadb.client import Client as AdbClient
from scipy.spatial.transform import Rotation

from meta_quest_teleop.buttons_parser import parse_buttons


def eprint(*args: Any, **kwargs: Any) -> None:
    """Print error messages to stderr."""
    RED = "\033[1;31m"
    sys.stderr.write(RED)
    print(*args, file=sys.stderr, **kwargs)
    RESET = "\033[0;0m"
    sys.stderr.write(RESET)


class MetaQuestReader:
    """Meta Quest Reader class with high-level APIs for transforms and button callbacks.

    This class handles the Meta Quest device connection, data reading, and provides
    clean APIs to access hand controller transforms in OpenXR and ROS coordinate
    systems, button event callbacks, and analog input values.
    """

    def __init__(
        self,
        ip_address: str | None = None,
        port: int = 5555,
        APK_name: str = "com.rail.oculus.teleop",
        run: bool = True,
        axis_mask: list[int] | None = None,
    ) -> None:
        """Initialize the MetaQuestReader.

        Args:
            ip_address: IP address to the device. If None, USB connection is used.
            port: Port number for connection. Defaults to 5555.
            APK_name: Android package name. Defaults to "com.rail.oculus.teleop".
            run: Whether to start reader immediately. Defaults to True.
            axis_mask: Mask for axes [x, y, z, roll, pitch, yaw]. 1 = enabled, 0 = disabled.
                       Masked axes (x, y, z, roll, pitch, yaw) will be zeroed.
        """
        self.running = False
        self.last_transforms: dict[str, Any] | None = {}
        self.last_buttons: dict[str, Any] | None = {}
        self._lock = threading.Lock()
        self.tag = "wE9ryARX"

        self.ip_address = ip_address
        self.port = port
        self.APK_name = APK_name

        # Validate axis mask
        if axis_mask is not None:
            assert (
                len(axis_mask) == 6
            ), "axis_mask must have 6 elements [x, y, z, roll, pitch, yaw]"
            assert np.all(np.isin(axis_mask, [0, 1])), "axis_mask values must be 0 or 1"
            # NOTE: Because we are reading in openxr coordinates, we need to resort the mask for ROS coordinates
            # x -> z, y -> -x, z -> -y , roll -> -pitch, pitch -> -roll, yaw -> yaw
            self.axis_mask = np.array(
                [
                    axis_mask[1],
                    axis_mask[2],
                    axis_mask[0],
                    axis_mask[4],
                    axis_mask[5],
                    axis_mask[3],
                ],
                dtype=int,
            )
        else:
            self.axis_mask = None

        # Button state tracking for edge detection
        self._prev_button_states: dict[str, bool] = {}

        # Callback system
        # TODO: add more button event callbacks.
        self._callbacks: dict[str, list[Callable]] = {
            "button_b_pressed": [],
            "button_a_pressed": [],
            "button_x_pressed": [],
            "button_y_pressed": [],
            "button_rj_pressed": [],
            "button_lj_pressed": [],
        }

        self._callbacks_locks: dict[str, threading.Lock] = {
            "button_b_pressed": threading.Lock(),
            "button_a_pressed": threading.Lock(),
            "button_x_pressed": threading.Lock(),
            "button_y_pressed": threading.Lock(),
            "button_rj_pressed": threading.Lock(),
            "button_lj_pressed": threading.Lock(),
        }

        # Cache latest transforms and button values (validated)
        self._latest_transforms: dict[str, np.ndarray] = {}
        self._latest_buttons: dict[str, Any] = {}
        self._last_sample_monotonic: float | None = None

        # Stream diagnostics.  A "commit" is one coherent latest-state
        # update, not one logcat line parsed from an accumulated backlog.
        self.lines_received = 0
        self.lines_committed = 0
        self.backlog_lines_dropped = 0
        self.last_batch_line_count = 0

        self.device = self.get_device()
        self.install(verbose=False)
        if run:
            self.run()

    def __del__(self) -> None:
        """Destructor."""
        self.stop()

    def run(self) -> None:
        """Start reading data from the Meta Quest device."""
        self.running = True
        self.device.shell(
            'am start -n "com.rail.oculus.teleop/com.rail.oculus.teleop.MainActivity" '
            "-a android.intent.action.MAIN -c android.intent.category.LAUNCHER"
        )
        serial = self.device.serial
        # Two different "-s" flags here, for two different programs:
        #   adb -s <serial>   selects the target device (adb global option).
        #   logcat -s <tag>   silences everything except the teleop app's pose
        #                     lines, so the reader needn't drain the full device
        #                     log (this source-side filter is half the latency fix).
        # We shell out to the adb binary on purpose: that separate OS process
        # drains the device socket into the kernel pipe independently of this
        # process's GIL, so under heavy thread contention (the dual-teleop case)
        # the Python reader only has to keep up with a local pipe. A ppadb
        # in-process socket reader was benchmarked and degraded ~40-60% worse
        # under that contention. See Misc/logcat-reader-latency.md.
        cmd = [
            "adb",
            "-s",
            serial,
            "shell",
            "logcat",
            "-T",
            "0",
            "-s",
            self.tag,
        ]
        self.thread = threading.Thread(
            target=self._read_logcat_subprocess,
            args=(cmd,),
            daemon=True,
        )
        self.thread.start()

    def _read_logcat_subprocess(self, cmd: list[str]) -> None:
        """Read logcat output from adb shell subprocess.

        Each select wake-up drains every byte currently available from the
        local pipe, keeps any incomplete trailing line, and commits only the
        newest valid complete teleop sample.  The adb child may continue
        filling this pipe while Python is stalled, so processing every line
        after recovery would replay an old controller trajectory as fresh.
        """
        try:
            with subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=0,
                close_fds=True,
                start_new_session=True,
            ) as proc:
                assert proc.stdout is not None
                fd = proc.stdout.fileno()
                os.set_blocking(fd, False)
                buffer = b""
                while self.running:
                    # Timeout so self.running is re-checked while logcat is quiet
                    readable, _, _ = select.select([fd], [], [], 1.0)
                    if not readable:
                        continue
                    chunk, pipe_closed = self._drain_logcat_pipe(fd)
                    if chunk:
                        buffer = self._consume_logcat_bytes(buffer, chunk)
                    if pipe_closed:
                        break
                proc.terminate()
                proc.wait(timeout=5)
        except FileNotFoundError:
            eprint(
                "⚠️ adb binary not found. Install android-tools-adb in the container."
            )
        except OSError as e:
            eprint(f"⚠️ Failed to start adb logcat subprocess: {e}")
        except subprocess.SubprocessError as e:
            eprint(f"⚠️ adb subprocess error: {e}")

    @staticmethod
    def _drain_logcat_pipe(fd: int) -> tuple[bytes, bool]:
        """Drain all bytes immediately available on a nonblocking pipe."""
        chunks = []
        pipe_closed = False
        while True:
            try:
                chunk = os.read(fd, 65536)
            except BlockingIOError:
                break
            if not chunk:
                pipe_closed = True
                break
            chunks.append(chunk)
            readable, _, _ = select.select([fd], [], [], 0.0)
            if not readable:
                break
        return b"".join(chunks), pipe_closed

    def _consume_logcat_bytes(self, pending: bytes, chunk: bytes) -> bytes:
        """Consume complete lines and return an unmodified trailing fragment."""
        combined = pending + chunk
        if b"\n" not in combined:
            self._process_complete_logcat_lines([])
            return combined
        complete, trailing = combined.rsplit(b"\n", 1)
        self._process_complete_logcat_lines(complete.split(b"\n"))
        return trailing

    def _process_complete_logcat_lines(self, raw_lines: list[bytes]) -> None:
        """Parse one drained batch and commit only its newest valid sample."""
        latest_sample = None
        valid_sample_count = 0
        for raw_line in raw_lines:
            data = self.extract_data(
                raw_line.decode("utf-8", errors="replace").strip()
            )
            if not data:
                continue
            try:
                transforms, buttons = MetaQuestReader.process_data(data)
            except (TypeError, ValueError):
                # A torn or malformed logcat line is not a valid candidate;
                # retain the newest earlier valid sample from this batch.
                continue
            if transforms is None or buttons is None:
                continue
            latest_sample = (transforms, buttons)
            valid_sample_count += 1

        with self._lock:
            self.lines_received += len(raw_lines)
            self.last_batch_line_count = len(raw_lines)
            self.backlog_lines_dropped += max(0, valid_sample_count - 1)

        if latest_sample is None:
            return

        transforms, buttons = latest_sample
        validated_transforms = {}
        for key, matrix in transforms.items():
            validated = self._validate_transform(matrix)
            if validated is not None:
                validated_transforms[key] = validated

        # This timestamp is deliberately sampled once, at the only commit in
        # the batch.  It represents when the latest state became visible to
        # the PC process; APK payloads currently contain no reliable source
        # timestamp or sequence suitable for cross-device age calculations.
        committed_monotonic = time.monotonic()
        with self._lock:
            self.last_transforms, self.last_buttons = transforms, buttons
            self._latest_transforms = validated_transforms
            self._latest_buttons = buttons
            self._last_sample_monotonic = committed_monotonic
            self.lines_committed += 1

        # Grip/button input is state, not an event stream: only the latest
        # state may drive callbacks after a stall.
        self._handle_button_events(buttons)

    def get_stream_diagnostics(self) -> dict[str, int]:
        """Return a thread-safe snapshot of latest-state stream counters."""
        with self._lock:
            return {
                "lines_received": self.lines_received,
                "lines_committed": self.lines_committed,
                "backlog_lines_dropped": self.backlog_lines_dropped,
                "last_batch_line_count": self.last_batch_line_count,
            }

    def stop(self) -> None:
        """Stop reading data from the Meta Quest device."""
        self.running = False
        if hasattr(self, "thread"):
            self.thread.join()

    def get_network_device(self, client: AdbClient, retry: int = 0) -> Any:
        """Get the Meta Quest device over the network.

        Args:
            client: ADB client.
            retry: Retry count.

        Returns:
            The Meta Quest device.
        """
        try:
            client.remote_connect(self.ip_address, self.port)
        except RuntimeError as e:
            eprint(f"⚠️ Failed to connect to device over network: {e}")
            os.system("adb devices")
            client.remote_connect(self.ip_address, self.port)
        assert self.ip_address is not None
        device = client.device(self.ip_address + ":" + str(self.port))

        if device is None:
            if retry == 1:
                os.system("adb tcpip " + str(self.port))
            if retry == 2:
                eprint(
                    "Make sure that device is running and is available at the "
                    "IP address specified as the OculusReader argument `ip_address`."
                )
                eprint("Currently provided IP address:", self.ip_address)
                eprint("Run `adb shell ip route` to verify the IP address.")
                exit(1)
            else:
                self.get_device()
                raise RuntimeError("Could not connect to device.")
        return device

    def get_usb_device(self, client: AdbClient) -> Any:
        """Get the Meta Quest device over USB.

        Args:
            client: ADB client.

        Returns:
            The Meta Quest device.
        """
        try:
            devices = client.devices()
        except RuntimeError as e:
            eprint(f"⚠️ Failed to get USB devices: {e}")
            os.system("adb devices")
            devices = client.devices()
        for device in devices:
            if device.serial.count(".") < 3:
                return device
        eprint(
            "Device not found. Make sure that device is running "
            "and is connected over USB"
        )
        eprint("Run `adb devices` to verify that the device is visible.")
        exit(1)

    def get_device(self) -> Any:
        """Get the Meta Quest device.

        Returns:
            The Meta Quest device.
        """
        # Default is "127.0.0.1" and 5037
        client = AdbClient(host="127.0.0.1", port=5037)
        if self.ip_address is not None:
            return self.get_network_device(client)
        else:
            return self.get_usb_device(client)

    def install(
        self, APK_path: str | None = None, verbose: bool = True, reinstall: bool = False
    ) -> None:
        """Install the APK on the Meta Quest device.

        Args:
            APK_path: Path to the APK file. If None, the default path is used.
            verbose: Whether to print messages. Defaults to True.
            reinstall: Whether to reinstall the APK if it is already installed.
                Defaults to False.
        """
        try:
            installed = self.device.is_installed(self.APK_name)
            if not installed or reinstall:
                if APK_path is None:
                    APK_path = os.path.join(
                        os.path.dirname(os.path.realpath(__file__)),
                        "APK",
                        "teleop-pointer-frame-relative.apk",
                    )
                success = self.device.install(APK_path, test=True, reinstall=reinstall)
                installed = self.device.is_installed(self.APK_name)
                if installed and success:
                    print("APK installed successfully.")
                else:
                    eprint("APK install failed.")
            elif verbose:
                print("APK is already installed.")
        except RuntimeError:
            eprint("Device is visible but could not be accessed.")
            eprint(
                "Run `adb devices` to verify that the device is visible and accessible."
            )
            eprint(
                'If you see "no permissions" next to the device serial, '
                "please put on the Meta Quest and allow the access."
            )
            exit(1)

    def uninstall(self, verbose: bool = True) -> None:
        """Uninstall the APK from the Meta Quest device.

        Args:
            verbose: Whether to print messages. Defaults to True.
        """
        try:
            installed = self.device.is_installed(self.APK_name)
            if installed:
                success = self.device.uninstall(self.APK_name)
                installed = self.device.is_installed(self.APK_name)
                if not installed and success:
                    print("APK uninstall finished.")
                    print(
                        "Please verify if the app disappeared from the "
                        'list as described in "UNINSTALL.md".'
                    )
                    print(
                        "For the resolution of this issue, please follow "
                        "https://github.com/Swind/pure-python-adb/issues/71."
                    )
                else:
                    eprint("APK uninstall failed")
            elif verbose:
                print("APK is not installed.")
        except RuntimeError:
            eprint("Device is visible but could not be accessed.")
            eprint(
                "Run `adb devices` to verify that the device is visible and accessible."
            )
            eprint(
                'If you see "no permissions" next to the device serial, '
                "please put on the Oculus Quest and allow the access."
            )
            exit(1)

    @staticmethod
    def process_data(
        string: str,
    ) -> tuple[dict[str, np.ndarray] | None, dict[str, Any] | None]:
        """Parse transforms, tracking confidence and controller buttons."""
        try:
            transforms_string, buttons_string = string.split("&", 1)
        except ValueError as e:
            eprint(f"⚠️ Failed to split data string by '&': {e}")
            return None, None

        transforms: dict[str, np.ndarray] = {}
        tracking_high: dict[str, bool] = {}

        for pair_string in transforms_string.split("|"):
            pair = pair_string.split(":", 1)
            if len(pair) != 2:
                continue

            key = pair[0].strip()
            payload = pair[1].strip()

            # New world-space APK:
            # lc:1 / rc:1 -> tracking confidence HIGH
            if key in ("lc", "rc"):
                try:
                    tracking_high[key[0]] = bool(int(payload))
                except ValueError:
                    tracking_high[key[0]] = False
                continue

            # Transform fields:
            # lg/lm/lp and rg/rm/rp
            try:
                values = [float(v) for v in payload.split() if v]
            except ValueError:
                continue

            if len(values) == 16:
                transforms[key] = np.asarray(values, dtype=float).reshape(4, 4)

        buttons = parse_buttons(buttons_string)

        # Keep tracking state alongside the existing button/state dictionary.
        buttons["leftTrackingHigh"] = tracking_high.get("l", False)
        buttons["rightTrackingHigh"] = tracking_high.get("r", False)

        return transforms, buttons

    def extract_data(self, line: str) -> str:
        """Extract data from a line of logcat output.

        Args:
            line: Line of logcat output.

        Returns:
            Extracted data.
        """
        output = ""
        if self.tag in line:
            try:
                output += line.split(self.tag + ": ")[1]
            except ValueError as e:
                eprint(f"⚠️ Failed to extract data from logcat line: {e}")
        return output

    def get_transformations_and_buttons(
        self,
    ) -> tuple[dict[str, np.ndarray] | None, dict[str, Any] | None]:
        """Get the latest transformations and button states.

        Returns:
            Tuple of transformations and button states.
        """
        with self._lock:
            return self.last_transforms, self.last_buttons

    def get_data_age_s(self, now_monotonic=None) -> float:
        """Return age of the latest parsed APK sample, or infinity."""
        if now_monotonic is None:
            now_monotonic = time.monotonic()
        now_monotonic = float(now_monotonic)
        if not math.isfinite(now_monotonic):
            raise ValueError("now_monotonic must be finite")
        with self._lock:
            received = self._last_sample_monotonic
        if received is None:
            return math.inf
        return max(0.0, now_monotonic - received)

    def data_is_fresh(self, timeout_s, now_monotonic=None) -> bool:
        """Return whether the APK/logcat source produced a recent sample."""
        timeout_s = float(timeout_s)
        if not math.isfinite(timeout_s) or timeout_s <= 0.0:
            raise ValueError("timeout_s must be finite and positive")
        return self.get_data_age_s(now_monotonic) <= timeout_s

    def _apply_axis_mask(self, transform: np.ndarray) -> np.ndarray:
        """Apply axis mask to transform, zeroing masked axes.

        Args:
            transform: Current 4x4 transformation matrix (OpenXR coordinates)

        Returns:
            Masked 4x4 transformation matrix (OpenXR coordinates)
        """
        # Start with current transform

        transform_translation = transform[:3, 3]
        transform_translation_masked = transform_translation * self.axis_mask[:3]
        transform_rotation = transform[:3, :3]
        transform_rotation_euler = Rotation.from_matrix(transform_rotation).as_euler(
            "xyz"
        )
        transform_rotation_euler_masked = transform_rotation_euler * self.axis_mask[3:]
        transform_rotation_masked = Rotation.from_euler(
            "xyz", transform_rotation_euler_masked
        ).as_matrix()

        transform_masked = np.eye(4)
        transform_masked[:3, 3] = transform_translation_masked
        transform_masked[:3, :3] = transform_rotation_masked

        return transform_masked

    def get_hand_controller_transform_openxr(
        self,
        hand: Literal["left", "right", "l", "r"] = "right",
        pose_type: Literal["grip", "model", "pointer"] = "grip",
    ) -> np.ndarray | None:
        """
        Get one controller transform in OpenXR coordinates.

        pose_type:
            grip    -> g
            model   -> m
            pointer -> p
        """

        hand_key = self._normalize_hand_key(hand)

        suffix_map = {
            "grip": "g",
            "model": "m",
            "pointer": "p",
        }

        if pose_type not in suffix_map:
            raise ValueError(
                "pose_type must be 'grip', 'model', or 'pointer'"
            )

        key = hand_key + suffix_map[pose_type]

        with self._lock:
            transform = self._latest_transforms.get(key)

            if transform is None:
                return None

            transform_openxr = transform.copy()

        if self.axis_mask is not None:
            transform_openxr = self._apply_axis_mask(
                transform_openxr
            )

        return transform_openxr

    def get_hand_controller_transform_ros(
        self,
        hand: Literal["left", "right", "l", "r"] = "right",
        pose_type: Literal["grip", "model", "pointer"] = "grip",
    ) -> np.ndarray | None:
        """
        Get one controller transform expressed in ROS coordinates.

        OpenXR:
            +X right
            +Y up
            +Z backward

        ROS:
            +X forward
            +Y left
            +Z up
        """

        transform_openxr = (
            self.get_hand_controller_transform_openxr(
                hand=hand,
                pose_type=pose_type,
            )
        )

        if transform_openxr is None:
            return None

        Q = Rotation.from_quat(
            [0.5, -0.5, -0.5, 0.5]
        )

        T_static = np.eye(4)
        T_static[:3, :3] = Q.as_matrix()

        return T_static @ transform_openxr

    def get_tracking_valid(
        self,
        hand: Literal["left", "right", "l", "r"] = "right",
    ) -> bool:
        """Return whether controller tracking confidence is HIGH."""

        hand_key = self._normalize_hand_key(hand)

        key = (
            "leftTrackingHigh"
            if hand_key == "l"
            else "rightTrackingHigh"
        )

        with self._lock:
            return bool(
                self._latest_buttons.get(
                    key,
                    False,
                )
            )

    def get_button_state(self, button_name: str) -> bool:
        """Get current state of a button.

        Args:
            button_name: Button name (e.g., 'A', 'B', 'X', 'Y', 'RJ',
                'LJ')

        Returns:
            True if button is pressed, False otherwise
        """
        with self._lock:
            return self._latest_buttons.get(button_name, False)

    def get_grip_value(
        self, hand: Literal["left", "right", "l", "r"] = "right"
    ) -> float:
        """Get the continuous grip value (analog trigger).

        Args:
            hand: Which hand ('left', 'right', 'l', or 'r')

        Returns:
            Float value in range [0.0, 1.0] where 0.0 is not pressed and
            1.0 is fully pressed
        """
        hand_key = self._normalize_hand_key(hand)
        button_name = "leftGrip" if hand_key == "l" else "rightGrip"
        with self._lock:
            value = self._latest_buttons.get(button_name, 0.0)

        # Handle case where value might be a tuple from parsing
        if isinstance(value, tuple):
            return float(value[0]) if len(value) > 0 else 0.0
        return float(value) if value else 0.0

    def get_trigger_value(
        self, hand: Literal["left", "right", "l", "r"] = "right"
    ) -> float:
        """Get the continuous trigger value (index finger trigger).

        Args:
            hand: Which hand ('left', 'right', 'l', or 'r')

        Returns:
            Float value in range [0.0, 1.0] where 0.0 is not pressed and
            1.0 is fully pressed
        """
        hand_key = self._normalize_hand_key(hand)
        button_name = "leftTrig" if hand_key == "l" else "rightTrig"
        with self._lock:
            value = self._latest_buttons.get(button_name, 0.0)

        # Handle case where value might be a tuple from parsing
        if isinstance(value, tuple):
            return float(value[0]) if len(value) > 0 else 0.0
        return float(value) if value else 0.0

    def get_joystick_value(
        self, hand: Literal["left", "right", "l", "r"] = "right"
    ) -> tuple[float, float]:
        """Get the joystick position.

        Args:
            hand: Which hand ('left', 'right', 'l', or 'r')

        Returns:
            Tuple (x, y) where both x and y are in range [-1.0, 1.0]
            Returns (0.0, 0.0) if not available
        """
        hand_key = self._normalize_hand_key(hand)
        button_name = "leftJS" if hand_key == "l" else "rightJS"
        with self._lock:
            value = self._latest_buttons.get(button_name, (0.0, 0.0))

        if isinstance(value, tuple) and len(value) >= 2:
            return (float(value[0]), float(value[1]))
        return (0.0, 0.0)

    def on(self, event: str, callback: Callable) -> None:
        """Register a callback for an event.

        Available events:
        - 'button_b_pressed': Called when Button B is pressed
        - 'button_a_pressed': Called when Button A is pressed
        - 'button_x_pressed': Called when Button X is pressed
        - 'button_y_pressed': Called when Button Y is pressed
        - 'button_rj_pressed': Called when Right Joystick is pressed
        - 'button_lj_pressed': Called when Left Joystick is pressed

        Args:
            event: Event name
            callback: Function to call when event occurs
        """
        # make sure the event is a valid event
        if event not in self._callbacks:
            raise ValueError(
                f"Invalid event: {event}. Must be one of: "
                f"{list(self._callbacks.keys())}"
            )

        self._callbacks[event].append(callback)

    def _validate_transform(self, matrix: np.ndarray) -> np.ndarray | None:
        """Validate transformation matrix.

        Args:
            matrix: 4x4 transformation matrix

        Returns:
            The same matrix if valid, None if invalid
        """
        if np.allclose(matrix, 0.0):
            return None

        det = np.linalg.det(matrix[:3, :3])
        if abs(abs(det) - 1.0) > 0.1:
            return None

        return matrix

    def _normalize_hand_key(self, hand: Literal["left", "right", "l", "r"]) -> str:
        """Normalize hand identifier to 'l' or 'r'.

        Args:
            hand: Hand identifier ('left', 'right', 'l', or 'r')

        Returns:
            'l' or 'r'
        """
        if hand in ("left", "l"):
            return "l"
        elif hand in ("right", "r"):
            return "r"
        else:
            raise ValueError(
                f"Invalid hand: {hand}. Must be 'left', 'right', " f"'l', or 'r'"
            )

    def _handle_button_events(self, buttons: dict) -> None:
        """Handle button press events and trigger callbacks.

        Args:
            buttons: Dictionary of button states
        """
        # Use lock to prevent race conditions when called from multiple threads
        callbacks_to_trigger = []
        with self._lock:
            # Check for button presses (rising edge detection)
            button_map = {
                "B": "button_b_pressed",
                "A": "button_a_pressed",
                "X": "button_x_pressed",
                "Y": "button_y_pressed",
                "RJ": "button_rj_pressed",
                "LJ": "button_lj_pressed",
            }

            for button_key, event_name in button_map.items():
                current_state = buttons.get(button_key, False)
                prev_state = self._prev_button_states.get(button_key, False)

                # Rising edge detected
                if current_state and not prev_state:
                    if not self._callbacks_locks[event_name].locked():
                        self._callbacks_locks[event_name].acquire()
                    else:
                        continue
                    # Update state BEFORE triggering callbacks to prevent double-trigger
                    self._prev_button_states[button_key] = current_state
                    # Collect callbacks to trigger (release lock before calling to avoid blocking)
                    callbacks_to_trigger.extend(
                        [(event_name, cb) for cb in self._callbacks[event_name]]
                    )
                else:
                    self._prev_button_states[button_key] = current_state

        # Trigger callbacks outside the lock to avoid blocking other threads
        for event_name, callback in callbacks_to_trigger:
            try:
                callback()
            finally:
                self._callbacks_locks[event_name].release()


def main() -> None:
    """Main function to test the MetaQuestReader."""
    oculus_reader = MetaQuestReader()

    while True:
        time.sleep(0.3)
        print(oculus_reader.get_transformations_and_buttons())


if __name__ == "__main__":
    main()
