"""
VLA inference runner — NO ROS 2 DEPENDENCY.

Runs a VLA policy served by OmniRobot (pi05, GR00T-N1.5, qwenpi, ...) against the Sonic whole-body control stack.
All communication uses ZMQ:
  1. Robot state  -> ZMQ SUB on ``g1_debug`` topic (from C++ zmq_output_handler)
  2. Actions out  -> ZMQ PUB (latent protocol v4: motion token + hand joints)
  3. Camera       -> ZMQ SUB via RealsenseZMQSubscriber (g1_camera_publisher.py :5620)
  4. Keyboard     -> ZMQ SUB via ZMQKeyboardSubscriber

Uses OmniRobotAdapter (OmniRobot WebSocket client) to communicate with a running
OmniRobot policy server (scripts/serve.py --transport websocket); the model behind it is
selected by the server, and the client adapts to it via the metadata handshake.

Keyboard commands (received via ZMQ from the standalone keyboard publisher):
  p  -> pause / resume the policy loop
  k  -> start / stop the C++ control loop
  i  -> blend smoothly to initial pose (or snap if no prior token) and switch to POSE mode
  t  -> change prompt at runtime (publisher sends ``prompt:<text>``)
  [  -> toggle left hand open/closed for initial pose
  ]  -> toggle right hand open/closed for initial pose
  c  -> start recording (handled by data exporter if running)
  s  -> stop recording success (handled by data exporter)
  f  -> stop recording failure (handled by data exporter)
"""

from dataclasses import dataclass
import queue
import threading
from collections import deque
import time

import numpy as np
import tyro
import zmq

from gear_sonic.data.robot_model.instantiation.g1 import instantiate_g1_robot_model
from gear_sonic.utils.data_collection.keyboard_subscriber import (
    DEFAULT_ZMQ_KEYBOARD_PORT,
    ZMQKeyboardSubscriber,
)
from gear_sonic.utils.data_collection.telemetry import Telemetry
from gear_sonic.utils.data_collection.transforms import compute_projected_gravity
from gear_sonic.utils.data_collection.zmq_state_subscriber import ZMQStateSubscriber
from gear_sonic.utils.inference.initial_poses import LATENT_INITIAL_MOTION_TOKEN
from gear_sonic.utils.inference.vla_utils import (
    calculate_latency_compensated_index,
    concat_action,
    prepare_observation_for_eval,
    should_trigger_new_inference,
)
from gear_sonic.utils.teleop.solver.hand.g1_gripper_ik_solver import (
    G1GripperInverseKinematicsSolver,
)
from gear_sonic.utils.teleop.zmq.zmq_planner_sender import (
    build_command_message,
    pack_pose_message,
)


class OmniRobotAdapter:
    """OmniRobot WebSocket policy server behind GR00T's PolicyClient interface.

    Model-agnostic: every checkpoint served by OmniRobot (pi05, GR00T-N1.5,
    qwenpi, ...) speaks the same wire format, and the per-model differences
    (action chunk length, image-history depth/stride) are read from the server's
    metadata handshake below rather than hardcoded here.

    Bridges the two wire formats: takes the GR00T-shaped observation that
    ``run_vla_inference`` builds and re-packages it into the flat
    ``{images, states, text, embodiment_tag}`` obs the OmniRobot ``Policy.infer``
    expects, then slices the flat 78-dim action back into the
    ``motion_token`` / ``left_hand`` / ``right_hand`` keys the SONIC action
    publisher consumes.
    """

    def __init__(self, host: str, port: int, embodiment_tag: str = "real_g1"):
        from gear_sonic.utils.inference.openpi_client.websocket_client_policy import (
            WebsocketClientPolicy,
        )

        self.ws = WebsocketClientPolicy(host=host, port=port)
        self.embodiment_tag = embodiment_tag

        # Server contract from the metadata handshake (OmniRobot Policy.server_metadata):
        #  - video_history: models trained with img_history_size H > 1 expect a stack of
        #    H frames per camera at temporal stride S env-steps (dataset fps), oldest ->
        #    newest; a single frame puts them out of distribution (e.g. qwenpi: 3 @ 10).
        #  - action_horizon: the chunk length the server returns; --action-horizon MUST
        #    match it (the client indexes/clamps by it).
        md = self.ws.get_server_metadata() or {}
        vh = md.get("video_history") or {}
        self.history_frames = int(vh.get("frames", 1) or 1)
        self.history_stride_steps = int(vh.get("stride_steps", 1) or 1)
        self.server_action_horizon = None
        embs = md.get("embodiments") or {}
        for _tag, entry in embs.items():
            if isinstance(entry, dict) and "action_horizon" in entry:
                self.server_action_horizon = int(entry["action_horizon"])
                break
        # Some policies PREDICT a long chunk but are meant to be re-queried sooner:
        # the server advertises how many actions to execute before replanning under
        # control.replan_after_actions (== execute_horizon). None = execute the full
        # chunk. Sync execution honors this (see --execute-horizon).
        ctrl = md.get("control") or {}
        reh = ctrl.get("replan_after_actions", ctrl.get("execute_horizon"))
        self.server_execute_horizon = int(reh) if reh else None
        print_green(
            f"Policy server metadata: action_horizon={self.server_action_horizon}, "
            f"replan_after_actions={self.server_execute_horizon}, "
            f"video_history={self.history_frames} frame(s) @ stride {self.history_stride_steps} step(s)"
        )

    def ping(self) -> bool:
        # WebsocketClientPolicy blocks in __init__ until the server is up.
        return True

    def get_action(self, obs: dict):
        img = np.asarray(obs["video"]["ego_view"][0, 0], dtype=np.uint8)  # (H, W, 3)
        # Frame history (T, H, W, 3), oldest -> newest, provided by the camera
        # subscriber when the server asked for it; otherwise the single frame.
        hist = (obs.get("video_history") or {}).get("ego_view")
        images = np.asarray(hist, dtype=np.uint8) if hist is not None else img
        state43 = np.asarray(obs["pi05_state43"], dtype=np.float32)       # (43,)
        prompt = obs["language"]["annotation.human.task_description"][0][0]

        out = self.ws.infer(
            {
                "images": {"ego_view": images},
                "states": {"state": state43},
                "text": prompt,
                "embodiment_tag": self.embodiment_tag,
            }
        )
        a = np.asarray(out["action"])  # (action_horizon, 78)
        return {
            "motion_token": a[:, :64],
            "left_hand": a[:, 64:71],
            "right_hand": a[:, 71:78],
        }, {}


class RealsenseZMQSubscriber:
    """Reads frames from teleop_yiqi_robot/g1_camera_publisher.py.

    Drop-in for ``ComposedCameraClientSensor``: exposes ``read()`` returning
    ``{"images": {"ego_view": img}, "timestamps": {"ego_view": ts}}`` (or None
    if no frame has arrived yet).

    The robot publisher does ``sock.send_pyobj({"timestamp", "image"})`` on a
    ZMQ PUB socket, where ``image`` is BGR (H, W, 3) uint8. We convert to RGB
    because the pi05 image processor expects RGB.

    NOTE: confirm the RGB/BGR convention matches how ``observation.images.
    egocentric`` was stored when g1_pnp_pour_v3 was built — a silent swap here
    degrades the policy with no error.
    """

    def __init__(self, host: str, port: int, history_frames: int = 1, history_stride_s: float = 0.0):
        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.SUB)
        self._sock.setsockopt_string(zmq.SUBSCRIBE, "")
        self._sock.setsockopt(zmq.CONFLATE, 1)  # keep only the freshest frame
        self._sock.connect(f"tcp://{host}:{port}")
        self._poller = zmq.Poller()
        self._poller.register(self._sock, zmq.POLLIN)
        print(f"[RealsenseZMQSubscriber] Connected to tcp://{host}:{port}")

        # Optional frame history (server metadata video_history): keep a ring buffer
        # of recent frames via a capture thread, so read() can return a (T, H, W, 3)
        # stack at offsets [-(T-1)*S, ..., -S, 0] seconds regardless of how rarely
        # the inference worker polls us. T == 1 keeps the legacy single-frame path.
        self._hist_n = max(1, int(history_frames))
        self._hist_stride_s = float(history_stride_s)
        self._buf: deque = deque()
        self._lock = threading.Lock()
        if self._hist_n > 1:
            keep_s = (self._hist_n - 1) * self._hist_stride_s + 1.0
            self._keep_s = keep_s
            t = threading.Thread(target=self._capture_loop, daemon=True)
            t.start()
            print(
                f"[RealsenseZMQSubscriber] frame history: {self._hist_n} frames @ "
                f"{self._hist_stride_s:.2f}s stride (buffer {keep_s:.1f}s)"
            )

    def _recv_frame(self):
        msg = self._sock.recv_pyobj()
        bgr = np.asarray(msg["image"])          # (H, W, 3) BGR uint8
        rgb = np.ascontiguousarray(bgr[:, :, ::-1])
        return msg.get("timestamp", time.time()), rgb

    def _capture_loop(self):
        while True:
            if not self._poller.poll(timeout=100):
                continue
            ts, rgb = self._recv_frame()
            now = time.time()
            with self._lock:
                self._buf.append((ts, rgb))
                while self._buf and now - self._buf[0][0] > self._keep_s:
                    self._buf.popleft()

    def _history_stack(self):
        """(T, H, W, 3) oldest->newest, nearest buffered frame to each target time,
        clamped to the oldest frame at start-up (mirrors the training loader)."""
        with self._lock:
            frames = list(self._buf)
        if not frames:
            return None
        t_now = frames[-1][0]
        ts = np.array([f[0] for f in frames])
        stack = []
        for k in range(self._hist_n - 1, -1, -1):
            target = t_now - k * self._hist_stride_s
            idx = int(np.argmin(np.abs(ts - target)))
            stack.append(frames[idx][1])
        return t_now, frames[-1][1], np.stack(stack)

    def read(self):
        if self._hist_n > 1:
            h = self._history_stack()
            if h is None:
                return None
            ts, rgb, stack = h
            return {
                "images": {"ego_view": rgb},
                "history": {"ego_view": stack},
                "timestamps": {"ego_view": ts},
            }
        if not self._poller.poll(timeout=0):  # non-blocking
            return None
        ts, rgb = self._recv_frame()
        return {
            "images": {"ego_view": rgb},
            "timestamps": {"ego_view": ts},
        }


@dataclass
class InferenceConfig:
    """CLI config for the VLA inference runner."""

    # Policy server (Isaac-GR00T PolicyServer)
    host: str = "localhost"
    """The host address of the Isaac-GR00T PolicyServer."""

    port: int = 5550
    """The port of the Isaac-GR00T PolicyServer."""

    # Control
    action_publish_rate: int = 50
    """Rate at which individual actions are published to the C++ control loop (Hz)."""

    action_horizon: int = 40
    """Action horizon of the VLA policy (number of future actions per inference)."""

    rate: float = 1 / 0.4
    """Rate at which we run the forward pass of the VLA policy (Hz)."""

    # Camera
    camera_host: str = "localhost"
    """Camera server host (the robot's IP running g1_camera_publisher.py)."""

    camera_port: int = 5620
    """Camera publisher port (g1_camera_publisher.py default)."""

    sim: bool = False
    """Running against MuJoCo sim: use ComposedCameraClientSensor (matches
    the sim's SensorServer image protocol) instead of RealsenseZMQSubscriber
    (matches the real robot's g1_camera_publisher.py pickle protocol)."""

    # ZMQ: Robot state (from C++ zmq_output_handler, g1_debug topic)
    state_zmq_host: str = "localhost"
    """ZMQ host for robot state (g1_debug topic from C++ deploy)."""

    state_zmq_port: int = 5557
    """ZMQ port for robot state (same socket as robot_config topic)."""

    # ZMQ: Action output (latent actions to C++ control loop)
    action_zmq_host: str = "localhost"
    """ZMQ host for action output (PUB socket)."""

    action_zmq_port: int = 5556
    """ZMQ port for action output."""

    # ZMQ: Keyboard input
    keyboard_zmq_host: str = "localhost"
    """ZMQ host for keyboard input."""

    keyboard_zmq_port: int = DEFAULT_ZMQ_KEYBOARD_PORT
    """ZMQ port for keyboard input."""

    # Embodiment
    embodiment_tag: str = "real_g1"
    """Embodiment tag sent to the pi05 policy server — must match the tag the
    server was started with (serve.py --embodiment). Default preserves the
    previous behavior (the adapter used to hardcode "real_g1" and ignore this
    flag); pass it explicitly to serve a differently-tagged checkpoint."""

    # Prompt / eval
    prompt: str = "demo"
    """The language prompt for the VLA policy."""

    # Initial pose
    initial_pose_blend_duration: float = 1.0
    """Duration (seconds) for smooth interpolation to initial pose. The robot
    blends from its current motion token to the initial pose token over this
    period. Set to 0 to snap instantly (no blend)."""

    init_episode: str = ""
    """Optional path to a recorded episode; if set, 'i' inits to that episode's
    FIRST motion token instead of the generic LATENT_INITIAL_MOTION_TOKEN — an
    in-distribution start pose that matches the checkpoint's training data.
    Accepts an aligned npz (data/g1/staging/<task>/aligned/episode_N.npz, key
    'motion_token') or a token npz (tokens_episode_N.npz, key 'token_state').
    Empty = the default standing pose. NOTE: the token lives in the SONIC
    decoder's latent space, so use an episode recorded with the SAME SONIC deploy
    you are running (hands still follow the '['/']' toggles)."""

    # Chunk handoff
    execution: str = "async"
    """Chunk handoff style: "async" or "sync".
    async (GR00T default): re-infer on a timer (`--rate`, counted from chunk
    ARRIVAL) and enter the new chunk at the latency-compensated index, so
    chunks overlap and the current one is replaced mid-way (a jump at the seam).
    sync: query only on the tick that sends the current chunk's LAST action,
    hold that action while the server thinks (~inference latency), then start
    the new chunk at index 0. Every action of every chunk is executed and the
    robot pauses ~latency at each boundary; `--rate` is ignored. Matches the
    offline evaluator / Dexmate live driver. The hold-last keeps the WBC fed at
    the publish rate, but validate in sim before the real robot."""

    execute_horizon: int = 0
    """SYNC only: execute this many actions of each chunk before replanning, for a
    policy that PREDICTS a long chunk but is meant to be re-queried sooner (e.g.
    predict 32, execute 16). 0 = use the server's advertised replan_after_actions
    if any, else the full action_horizon. Must be 1..action_horizon. Ignored in
    async (use `--rate` there). --action-horizon still matches the FULL chunk the
    server returns."""

    # Debug
    verbose_timing: bool = False
    """Whether to always print timing info (not just when loop is slow)."""


def print_green(x):
    print(f"\033[92m{x}\033[0m")


# ---------------------------------------------------------------------------
# Action packing (latent protocol v4)
# ---------------------------------------------------------------------------


def pack_latent_action_message(
    motion_token: np.ndarray,
    frame_index: np.ndarray,
    left_hand_joints: np.ndarray = None,
    right_hand_joints: np.ndarray = None,
) -> bytes:
    """Pack a single motion-token action into a ZMQ message (Protocol v4).

    Args:
        motion_token: Shape ``[64]`` (flat) or ``[1, 64]``.
        frame_index:  Shape ``[1]``.
        left_hand_joints:  Shape ``[7]`` or ``[1, 7]``, optional.
        right_hand_joints: Shape ``[7]`` or ``[1, 7]``, optional.

    Returns:
        Packed ZMQ message bytes.
    """
    motion_token = np.asarray(motion_token, dtype=np.float32)
    frame_index = np.asarray(frame_index, dtype=np.int64)

    if frame_index.ndim == 0:
        frame_index = np.array([frame_index], dtype=np.int64)
    elif frame_index.shape[0] != 1:
        frame_index = frame_index[:1]

    if motion_token.ndim == 1:
        motion_token = motion_token.reshape(1, -1)

    pose_data = {
        "token_state": motion_token,
        "frame_index": frame_index,
    }

    if left_hand_joints is not None:
        left_hand_joints = np.asarray(left_hand_joints, dtype=np.float32)
        if left_hand_joints.ndim == 1:
            if left_hand_joints.shape[0] != 7:
                raise ValueError(
                    f"left_hand_joints must have shape [7], got {left_hand_joints.shape}"
                )
            left_hand_joints = left_hand_joints.reshape(1, 7)
        pose_data["left_hand_joints"] = left_hand_joints

    if right_hand_joints is not None:
        right_hand_joints = np.asarray(right_hand_joints, dtype=np.float32)
        if right_hand_joints.ndim == 1:
            if right_hand_joints.shape[0] != 7:
                raise ValueError(
                    f"right_hand_joints must have shape [7], got {right_hand_joints.shape}"
                )
            right_hand_joints = right_hand_joints.reshape(1, 7)
        pose_data["right_hand_joints"] = right_hand_joints

    return pack_pose_message(pose_data, topic="pose", version=4)


def get_action_field(action_dict: dict, key: str):
    """Get action field from dict, checking both with and without 'action.' prefix."""
    value = action_dict.get(key)
    if value is not None:
        return value
    value = action_dict.get(f"action.{key}")
    if value is not None:
        return value
    raise AssertionError(
        f"Required action field '{key}' (or 'action.{key}') not found in processed_action. "
        f"Available keys: {list(action_dict.keys())}"
    )


# ---------------------------------------------------------------------------
# Observation / inference helpers
# ---------------------------------------------------------------------------


def prepare_observation_from_sensors(
    camera_subscriber,
    state_subscriber,
    robot_model,
    language_prompt: str,
    log_errors: bool = False,
):
    """Read sensors and prepare observation for the VLA policy.

    Returns:
        observation dict, or None if sensor data not yet available.
    """
    camera_msg = camera_subscriber.read()
    if camera_msg is None:
        if log_errors:
            print("[DEBUG] prepare_observation: waiting for camera msg..", flush=True)
        return None

    state_msg = state_subscriber.get_msg()
    if state_msg is None:
        if log_errors:
            print("[DEBUG] prepare_observation: waiting for state msg..", flush=True)
        return None

    cam_img = camera_msg["images"]["ego_view"]

    # Copy index finger data to middle finger (hardware coupling)
    state_msg["left_hand_q"][5] = state_msg["left_hand_q"][3]
    state_msg["left_hand_q"][6] = state_msg["left_hand_q"][4]

    qpos = robot_model.get_configuration_from_actuated_joints(
        body_actuated_joint_values=state_msg["body_q"],
        left_hand_actuated_joint_values=state_msg["left_hand_q"],
        right_hand_actuated_joint_values=state_msg["right_hand_q"],
    )

    video = {"ego_view": cam_img[np.newaxis, np.newaxis]}
    if "left_wrist" in camera_msg["images"]:
        video["left_wrist"] = camera_msg["images"]["left_wrist"][np.newaxis, np.newaxis]
    if "right_wrist" in camera_msg["images"]:
        video["wrist_view"] = camera_msg["images"]["right_wrist"][np.newaxis, np.newaxis]

    observation = {
        "video": video,
        "state": {},
        "language": {
            "annotation.human.task_description": [[language_prompt]],
        },
        "q": np.asarray(qpos, dtype=np.float32)[np.newaxis, np.newaxis],
        "timestamps": camera_msg["timestamps"]["ego_view"],
    }

    observation = prepare_observation_for_eval(robot_model, observation)

    # Frame history stack for servers that ask for it (see OmniRobotAdapter); kept out
    # of observation["video"] so prepare_observation_for_eval sees a single frame.
    if camera_msg.get("history"):
        observation["video_history"] = dict(camera_msg["history"])

    # Projected gravity for Sonic latent embodiment
    assert "base_quat" in state_msg, "base_quat not found in state_msg"
    base_quat = np.asarray(state_msg["base_quat"], dtype=np.float64)
    assert base_quat.shape == (4,), "base_quat must have shape (4,)"
    projected_gravity = compute_projected_gravity(base_quat)
    observation["state"]["projected_gravity"] = np.asarray(
        projected_gravity, dtype=np.float32
    )[np.newaxis, np.newaxis]

    # pi05 flat state (order per build_lerobot_v3.py: qpos(29)+lhand(7)+rhand(7)).
    # Hand dims use the COMMANDED values (last_*_hand_action), NOT the measured
    # left/right_hand_q: the training datasets' state-hands are bit-identical to
    # the action-hands (verified on g1_pnp/pour_v3 parquets 2026-07-28) because
    # the recording pipeline stores last_*_hand_action for both. Measured hand_q
    # diverges from commanded exactly during grasps (fingers blocked by the
    # object), which would feed the policy an out-of-distribution state at the
    # manipulation-critical moments. (The index->middle copy above only affects
    # the measured hand_q used for the GR00T FK fields, not this state.)
    observation["pi05_state43"] = np.concatenate(
        [
            np.asarray(state_msg["body_q"], dtype=np.float32),              # 29
            np.asarray(state_msg["last_left_hand_action"], dtype=np.float32),   # 7 (commanded)
            np.asarray(state_msg["last_right_hand_action"], dtype=np.float32),  # 7 (commanded)
        ]
    )

    return observation


def run_policy_inference_and_process(policy, observation, robot_model):
    """Run policy inference via Isaac-GR00T PolicyClient and process results.

    Returns:
        processed_action dict or None on error.
    """
    try:
        action, _info = policy.get_action(observation)

        action.pop("task_progress", None)
        action.pop("action.task_progress", None)

        motion_key = "motion_token" if "motion_token" in action else "action.motion_token"
        if np.abs(action[motion_key]).max() > 1.25:
            print(
                f"[Warning] action['{motion_key}'] max "
                f"({np.abs(action[motion_key]).max():.4f}) > 1.25. "
                "Exceeds action bound, skipping."
            )
            return None

        processed_action = concat_action(robot_model, action)
        return processed_action
    except Exception as e:
        print(f"Error in inference: {e}")
        import traceback

        traceback.print_exc()
        return None


def _inference_worker_loop(
    inference_queue: queue.Queue,
    result_queue: queue.Queue,
    stop_event: threading.Event,
    busy_event: threading.Event,
    prepare_obs_fn,
    inference_fn,
):
    """Persistent worker thread for async inference."""
    while not stop_event.is_set():
        try:
            try:
                inference_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            busy_event.set()
            try:
                observation = prepare_obs_fn()
                if observation is None:
                    print("[DEBUG] Worker thread: Observation is None, skipping", flush=True)
                    continue

                inference_start_time = time.monotonic()
                processed_action = inference_fn(observation)

                if processed_action is not None:
                    try:
                        result_queue.put_nowait((processed_action, inference_start_time))
                    except queue.Full:
                        try:
                            result_queue.get_nowait()
                            result_queue.put_nowait((processed_action, inference_start_time))
                        except queue.Empty:
                            result_queue.put_nowait((processed_action, inference_start_time))
            finally:
                busy_event.clear()
        except Exception as e:
            print(f"Error in inference worker thread: {e}")
            import traceback

            traceback.print_exc()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _compute_closed_hand_joints(side: str) -> np.ndarray:
    """Compute closed hand joint positions using G1GripperInverseKinematicsSolver."""
    side_str = "left" if side.upper() == "L" else "right"
    solver = G1GripperInverseKinematicsSolver(side=side_str)
    return solver._get_middle_close_q_desired().astype(np.float32)


def _load_init_token_from_episode(path: str) -> np.ndarray:
    """Return a recorded episode's FIRST motion token [64], for use as the init
    pose (see InferenceConfig.init_episode). Accepts an aligned npz (key
    'motion_token', from data_pipeline/g1/align_episode.py) or a token npz (key
    'token_state', from the PC recorder). Fails loud on a bad file/shape."""
    d = np.load(path)
    key = "motion_token" if "motion_token" in d else ("token_state" if "token_state" in d else None)
    if key is None:
        raise SystemExit(
            f"--init-episode {path!r}: no 'motion_token' or 'token_state' key "
            f"(found {list(d.keys())}). Pass an aligned episode npz or a tokens_episode_N.npz."
        )
    tok = np.asarray(d[key], dtype=np.float32)
    if tok.ndim != 2 or tok.shape[1] != 64:
        raise SystemExit(f"--init-episode {path!r}: expected (T, 64) tokens, got {tok.shape}.")
    return tok[0].copy()  # the episode's first-frame motion token


def main(config: InferenceConfig):
    pause_loop = True

    robot_model = instantiate_g1_robot_model(waist_location="lower_and_upper_body")

    # pi05 policy over OmniRobot WebSocket (replaces GR00T's ZMQ PolicyClient)
    n1_policy = OmniRobotAdapter(
        host=config.host, port=config.port, embodiment_tag=config.embodiment_tag
    )

    print(f"Connecting to PolicyServer at {config.host}:{config.port}...")
    if n1_policy.ping():
        print_green("PolicyServer is reachable.")
    else:
        print("WARNING: PolicyServer not reachable. Inference will fail until server is up.")

    if (
        n1_policy.server_action_horizon is not None
        and n1_policy.server_action_horizon != config.action_horizon
    ):
        raise SystemExit(
            f"--action-horizon {config.action_horizon} does not match the policy server's "
            f"chunk length {n1_policy.server_action_horizon}. Pass "
            f"--action-horizon {n1_policy.server_action_horizon} (and recompute --rate: "
            f"quasi-sync = 1/((H-4)/{config.action_publish_rate} - 0.4))."
        )
    # Frame-history stride is given in env steps at the dataset fps, which is the
    # action publish rate (10 Hz for the G1 datasets).
    history_stride_s = n1_policy.history_stride_steps / float(config.action_publish_rate)

    state_subscriber = ZMQStateSubscriber(
        host=config.state_zmq_host,
        port=config.state_zmq_port,
    )

    if config.sim:
        from gear_sonic.camera.composed_camera import ComposedCameraClientSensor

        camera_subscriber = ComposedCameraClientSensor(
            server_ip=config.camera_host, port=config.camera_port
        )
        if n1_policy.history_frames > 1:
            print(
                f"WARNING: server expects {n1_policy.history_frames}-frame history but the "
                "sim camera sends single frames -- policy runs out of distribution."
            )
    else:
        camera_subscriber = RealsenseZMQSubscriber(
            host=config.camera_host,
            port=config.camera_port,
            history_frames=n1_policy.history_frames,
            history_stride_s=history_stride_s,
        )

    zmq_context = zmq.Context()
    zmq_socket = zmq_context.socket(zmq.PUB)
    zmq_socket.bind(f"tcp://{config.action_zmq_host}:{config.action_zmq_port}")
    time.sleep(0.1)
    print_green(
        f"ZMQ action socket bound to tcp://{config.action_zmq_host}:{config.action_zmq_port}"
    )
    print_green(f"Using embodiment tag: {config.embodiment_tag}")

    keyboard_listener = ZMQKeyboardSubscriber(
        port=config.keyboard_zmq_port, host=config.keyboard_zmq_host
    )

    telemetry = Telemetry(window_size=100)

    loop_rate = config.action_publish_rate
    loop_period = 1.0 / loop_rate

    # Track C++ control loop state
    cpp_loop_running = False
    cpp_mode = "OFF"  # "OFF", "PLANNER", or "POSE"

    # Track initial pose hand states
    initial_pose_left_hand_closed = False
    initial_pose_right_hand_closed = False

    # Init-pose target token: a recorded episode's first frame (in-distribution
    # warm start) or the generic standing pose. Resolved once at startup.
    if config.init_episode:
        init_motion_token = _load_init_token_from_episode(config.init_episode)
        print_green(f"Init pose: first motion token from episode {config.init_episode}")
    else:
        init_motion_token = LATENT_INITIAL_MOTION_TOKEN

    def publish_initial_pose():
        """Publish initial pose command to move robot to starting position."""
        print("Moving to initial pose")
        left_hand = (
            _compute_closed_hand_joints("L")
            if initial_pose_left_hand_closed
            else np.zeros(7, dtype=np.float32)
        )
        right_hand = (
            _compute_closed_hand_joints("R")
            if initial_pose_right_hand_closed
            else np.zeros(7, dtype=np.float32)
        )
        zmq_message = pack_latent_action_message(
            motion_token=init_motion_token,
            frame_index=np.array([0], dtype=np.int64),
            left_hand_joints=left_hand,
            right_hand_joints=right_hand,
        )
        zmq_socket.send(zmq_message)
        print_green("Sent latent initial pose via ZMQ")
        time.sleep(1.0)
        print("Initial pose done.")

    def blend_to_initial_pose(duration_s: float) -> bool:
        """Smoothly interpolate from the last sent motion token to the initial pose.

        Linearly blends over ``duration_s`` seconds at the action publish rate,
        sending intermediate tokens each loop iteration. Returns True if blend
        was performed, False if skipped (no previous token available).
        """
        nonlocal last_sent_motion_token
        if last_sent_motion_token is None:
            print("No previous motion token — snapping to initial pose instead.")
            publish_initial_pose()
            return False

        start_token = last_sent_motion_token.copy()
        target_token = np.asarray(init_motion_token, dtype=np.float32).copy()
        num_steps = max(1, round(config.action_publish_rate * duration_s))
        step_period = 1.0 / config.action_publish_rate

        left_hand = (
            _compute_closed_hand_joints("L")
            if initial_pose_left_hand_closed
            else np.zeros(7, dtype=np.float32)
        )
        right_hand = (
            _compute_closed_hand_joints("R")
            if initial_pose_right_hand_closed
            else np.zeros(7, dtype=np.float32)
        )

        print(
            f"Blending to initial pose over {duration_s:.2f}s "
            f"({num_steps} steps at {config.action_publish_rate} Hz)"
        )

        for step in range(num_steps):
            t_step_start = time.monotonic()
            alpha = (step + 1) / num_steps
            blended_token = ((1.0 - alpha) * start_token + alpha * target_token).astype(
                np.float32
            )
            zmq_message = pack_latent_action_message(
                motion_token=blended_token,
                frame_index=np.array([0], dtype=np.int64),
                left_hand_joints=left_hand,
                right_hand_joints=right_hand,
            )
            zmq_socket.send(zmq_message)
            last_sent_motion_token = blended_token.copy()

            elapsed = time.monotonic() - t_step_start
            remaining = step_period - elapsed
            if remaining > 0:
                time.sleep(remaining)

        print_green("Initial pose blend complete.")
        return True

    def send_cpp_control_command(start: bool, planner: bool = False):
        """Send C++ control loop start/stop commands via ZMQ."""
        nonlocal cpp_loop_running, cpp_mode
        try:
            cmd_msg = build_command_message(start=start, stop=not start, planner=planner)
            zmq_socket.send(cmd_msg)
            time.sleep(0.01)
            action_str = "start" if start else "stop"
            mode_str = "planner" if planner else "pose"
            cpp_loop_running = start
            if start:
                cpp_mode = "PLANNER" if planner else "POSE"
            else:
                cpp_mode = "OFF"
            print_green(f"Sent ZMQ command: {action_str} control loop ({mode_str} mode)")
            return True
        except Exception as e:
            action_str = "start" if start else "stop"
            print(f"Warning: Failed to send {action_str} command message: {e}")
            return False

    if config.execution not in ("async", "sync"):
        raise SystemExit(f"--execution must be 'async' or 'sync', got {config.execution!r}")
    sync_execution = config.execution == "sync"
    # How many actions of each chunk sync executes before replanning. Priority:
    # explicit --execute-horizon, else the server's advertised replan_after_actions,
    # else the full chunk. Clamped to 1..action_horizon. async always uses the full
    # chunk (its cadence is --rate), so this equals action_horizon there.
    if sync_execution:
        exec_h = (
            config.execute_horizon
            or n1_policy.server_execute_horizon
            or config.action_horizon
        )
        exec_h = max(1, min(int(exec_h), config.action_horizon))
    else:
        exec_h = config.action_horizon
    if sync_execution:
        print_green(
            f"Execution: SYNC — execute {exec_h} of {config.action_horizon} actions per "
            "chunk, hold the last during inference, restart at index 0 (--rate ignored)."
        )
    else:
        print_green(
            f"Execution: ASYNC — re-infer every {1.0 / config.rate:.2f}s after chunk "
            "arrival, enter at the latency-compensated index."
        )

    # Inference state
    cached_action_chunk = None
    action_chunk_index = 0
    last_inference_time = 0.0
    inference_interval = 1.0 / config.rate

    zmq_frame_counter = 0
    last_sent_motion_token: np.ndarray | None = None

    PROMPT_MSG_PREFIX = "prompt:"

    def check_keyboard_input():
        nonlocal pause_loop, cpp_loop_running, cpp_mode
        nonlocal initial_pose_left_hand_closed, initial_pose_right_hand_closed
        nonlocal cached_action_chunk, action_chunk_index, last_inference_time
        nonlocal zmq_frame_counter, last_sent_motion_token

        key = keyboard_listener.read_msg()
        if key is None:
            return

        if key.startswith(PROMPT_MSG_PREFIX):
            new_prompt = key[len(PROMPT_MSG_PREFIX):]
            if new_prompt:
                old_prompt = language_prompt_ref[0]
                language_prompt_ref[0] = new_prompt
                print_green(f'Inference prompt changed: "{old_prompt}" -> "{new_prompt}"')
            else:
                print("Received empty prompt change -- ignoring.")
            return

        if key == "c":
            print("Keyboard: 'c' (start recording -- handled by data exporter)")
        elif key == "s":
            print("Keyboard: 's' (stop recording success -- handled by data exporter)")
        elif key == "f":
            print("Keyboard: 'f' (stop recording failure -- handled by data exporter)")
        elif key == "i":
            if cpp_loop_running and cpp_mode == "PLANNER":
                if send_cpp_control_command(start=True, planner=False):
                    print("Switched to POSE mode (from PLANNER mode)")
                else:
                    print("Warning: Failed to switch to POSE mode")
            elif not cpp_loop_running:
                print("Note: C++ loop not running - press 'k' to start")

            pause_loop = True
            if config.initial_pose_blend_duration > 0 and last_sent_motion_token is not None:
                blend_to_initial_pose(config.initial_pose_blend_duration)
            else:
                publish_initial_pose()

            zmq_frame_counter = 0
            cached_action_chunk = None
            action_chunk_index = 0
            print("Cleared cached action chunk, reset frame counter")
        elif key == "p":
            pause_loop = not pause_loop
            print(f"{'Paused' if pause_loop else 'Resumed'} policy loop")
            if pause_loop:
                print("Policy loop paused (C++ loop still running - press 'k' to stop)")
            else:
                # Discard any chunk computed while paused: its observation may
                # predate an 'i' init-pose move, so its actions belong to a pose
                # the robot is no longer in — executing it causes a sudden large
                # first motion. Clearing forces an immediate fresh inference from
                # the CURRENT pose (the deploy holds the last token for the
                # ~0.4 s this takes). Observed on hardware 2026-07-28.
                cached_action_chunk = None
                action_chunk_index = 0
                print("Policy loop resumed - cleared stale chunk, inferring fresh "
                      "from current pose (brief hold)")
        elif key == "k":
            if cpp_loop_running:
                current_planner = cpp_mode == "PLANNER"
                print(f"Stopping C++ control loop (from {cpp_mode} mode)...")
                if send_cpp_control_command(start=False, planner=current_planner):
                    print("Stopped C++ control loop")
            else:
                print("Starting C++ control loop in PLANNER mode...")
                if send_cpp_control_command(start=True, planner=True):
                    print("Started C++ control loop in PLANNER mode")
                    print("Press 'i' to send initial pose and switch to POSE mode")
                    if pause_loop:
                        print("Note: Policy loop is paused - press 'p' to resume")
        elif key == "[":
            initial_pose_left_hand_closed = not initial_pose_left_hand_closed
            print(
                f"Initial pose left hand: {'closed' if initial_pose_left_hand_closed else 'open'}"
            )
        elif key == "]":
            initial_pose_right_hand_closed = not initial_pose_right_hand_closed
            print(
                f"Initial pose right hand: "
                f"{'closed' if initial_pose_right_hand_closed else 'open'}"
            )

    # Mutable prompt container (single-writer from keyboard, single-reader from inference)
    language_prompt_ref: list[str] = [config.prompt]
    print(f"Starting the policy loop with language prompt: {language_prompt_ref[0]}")

    inference_queue = queue.Queue(maxsize=1)
    result_queue = queue.Queue(maxsize=1)
    inference_stop_event = threading.Event()
    inference_busy_event = threading.Event()

    inference_worker_thread = threading.Thread(
        target=_inference_worker_loop,
        args=(
            inference_queue,
            result_queue,
            inference_stop_event,
            inference_busy_event,
            lambda: prepare_observation_from_sensors(
                camera_subscriber=camera_subscriber,
                state_subscriber=state_subscriber,
                robot_model=robot_model,
                language_prompt=language_prompt_ref[0],
                log_errors=True,
            ),
            lambda obs: run_policy_inference_and_process(
                policy=n1_policy,
                observation=obs,
                robot_model=robot_model,
            ),
        ),
        daemon=True,
    )
    inference_worker_thread.start()

    try:
        while True:
            t_start = time.monotonic()
            check_keyboard_input()

            # Consume result first so last_inference_time is fresh before trigger check
            try:
                processed_action, inference_start_time = result_queue.get_nowait()
                inference_delay = time.monotonic() - inference_start_time
                if sync_execution:
                    # The robot held still while the server thought, so the
                    # observation is still current: use the chunk from index 0.
                    action_chunk_index = 0
                else:
                    action_chunk_index = calculate_latency_compensated_index(
                        inference_delay, config.action_publish_rate, config.action_horizon
                    )
                cached_action_chunk = processed_action
                last_inference_time = time.monotonic()
                print_green(
                    f'New action chunk (prompt: "{language_prompt_ref[0]}", '
                    f"latency: {inference_delay:.3f}s)"
                )
            except queue.Empty:
                pass

            worker_is_busy = inference_busy_event.is_set()
            if sync_execution:
                # Query on the tick that sends the chunk's last EXECUTED action (or
                # when nothing is cached). exec_h may be < action_horizon (execute a
                # prefix of a longer prediction, then replan). While the server
                # thinks, the clamp below keeps re-sending that action.
                # `result_queue.empty()` closes the race where the worker finished
                # after this tick's consume step: without it we would query twice.
                should_start = (
                    (not worker_is_busy)
                    and result_queue.empty()
                    and (
                        cached_action_chunk is None
                        or action_chunk_index >= exec_h - 1
                    )
                )
            else:
                should_start = should_trigger_new_inference(
                    cached_chunk_exists=(cached_action_chunk is not None),
                    inference_thread_running=worker_is_busy,
                    time_since_last_inference=(time.monotonic() - last_inference_time),
                    inference_interval=inference_interval,
                )

            if should_start:
                try:
                    inference_queue.put_nowait(None)
                except queue.Full:
                    pass

            if pause_loop:
                print("Pausing...", end="", flush=True)
                time.sleep(0.2)
                print(".", end="", flush=True)
                continue

            with telemetry.timer("total_loop"):
                if cached_action_chunk is None:
                    print("[DEBUG] No cached chunk yet, waiting...", flush=True)
                    _sleep_remaining(t_start, loop_period)
                    continue

                processed_action = cached_action_chunk

                if processed_action is None or not processed_action:
                    print("[DEBUG] processed_action is None or empty, skipping", flush=True)
                else:
                    motion_token = np.asarray(
                        get_action_field(processed_action, "motion_token"),
                        dtype=np.float32,
                    )
                    left_hand_joints = np.asarray(
                        get_action_field(processed_action, "left_hand"),
                        dtype=np.float32,
                    )
                    right_hand_joints = np.asarray(
                        get_action_field(processed_action, "right_hand"),
                        dtype=np.float32,
                    )

                    # Action arrays arrive as (B, T, D) from the model.
                    # Squeeze batch dim to get (T, D), then index by time step.
                    if motion_token.ndim == 3:
                        motion_token = motion_token[0]
                    if left_hand_joints.ndim == 3:
                        left_hand_joints = left_hand_joints[0]
                    if right_hand_joints.ndim == 3:
                        right_hand_joints = right_hand_joints[0]

                    horizon = motion_token.shape[0] if motion_token.ndim == 2 else 1
                    current_idx = min(action_chunk_index, horizon - 1)

                    if motion_token.ndim == 2:
                        motion_token = motion_token[current_idx]
                    if left_hand_joints.ndim == 2:
                        left_hand_joints = left_hand_joints[current_idx]
                    if right_hand_joints.ndim == 2:
                        right_hand_joints = right_hand_joints[current_idx]

                    frame_index = np.array([zmq_frame_counter], dtype=np.int64)
                    zmq_frame_counter += 1

                    zmq_message = pack_latent_action_message(
                        motion_token,
                        frame_index,
                        left_hand_joints=left_hand_joints,
                        right_hand_joints=right_hand_joints,
                    )
                    zmq_socket.send(zmq_message)
                    last_sent_motion_token = motion_token.copy()
                    if zmq_frame_counter % 50 == 0:
                        print_green(
                            f"ZMQ: Sent latent action - "
                            f"frame: {frame_index[0]}, "
                            f"token shape: {motion_token.shape}"
                        )

                # Clamp at exec_h-1: in sync this parks the index on the last EXECUTED
                # action (holding it while inference runs, never advancing into the
                # unexecuted tail of a longer prediction); in async exec_h ==
                # action_horizon, so this is the usual full-chunk clamp.
                action_chunk_index = min(action_chunk_index + 1, exec_h - 1)

            end_time = time.monotonic()

            if config.verbose_timing:
                telemetry.log_timing_info(context="VLA Inference Loop", threshold=0.0)
            elif (end_time - t_start) > (1 / config.rate):
                telemetry.log_timing_info(
                    context="VLA Inference Loop Missed", threshold=0.001
                )

            _sleep_remaining(t_start, loop_period)

    except KeyboardInterrupt:
        print("VLA inference loop terminated by user")

    finally:
        inference_stop_event.set()
        inference_worker_thread.join(timeout=1.0)
        zmq_socket.close()
        zmq_context.term()
        state_subscriber.close()
        keyboard_listener.close()
        print("Shutdown complete.")


def _sleep_remaining(t_start: float, loop_period: float):
    """Sleep for the remainder of the loop period."""
    elapsed = time.monotonic() - t_start
    remaining = loop_period - elapsed
    if remaining > 0:
        time.sleep(remaining)


if __name__ == "__main__":
    config = tyro.cli(InferenceConfig)
    main(config)
