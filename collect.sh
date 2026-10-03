#!/usr/bin/env bash
# Start ROS bringup, VR publisher, and recorder inside Docker or from the host.
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
cd "$repo_root"

usage() {
  cat <<'EOF'
Usage: ./collect.sh [config/collect/franka.yaml]

Starts ROS bringup, VR topics, and vr_collect.py. On the host, uses Docker
containers; inside Humble Docker, uses local processes. Press Ctrl+C to stop
all components. Build the image and overlay once as described in
README.md before using this command.
EOF
}

if [[ ${1:-} == --help || ${1:-} == -h ]]; then
  usage
  exit 0
fi
if (( $# > 1 )); then
  usage >&2
  exit 2
fi

config_arg="${1:-config/collect/franka.yaml}"
if ! config_path="$(realpath -e -- "$config_arg")" || [[ ! -f "$config_path" ]]; then
  echo "[collect] Config not found: $config_arg" >&2
  exit 1
fi
if [[ "$config_path" != "$repo_root/"* ]]; then
  echo "[collect] Config must be inside the repository: $config_arg" >&2
  exit 1
fi
config_relative="${config_path#"$repo_root/"}"

docker_dir="$repo_root/ros2_ws/docker/franka_humble"
inside_docker=false
if [[ -f /.dockerenv ]] || grep -q docker /proc/self/cgroup 2>/dev/null; then
  inside_docker=true
fi
if [[ "$inside_docker" == true ]]; then
  if [[ ! -f /opt/ros/humble/setup.bash || ! -x /opt/uv/venv/bin/python ]]; then
    echo "[collect] Container collection requires the project's Humble image." >&2
    exit 1
  fi
  # ROS setup scripts may reference unset variables.
  set +u
  source /opt/ros/humble/setup.bash
  if [[ -f /opt/panda_ws/install/setup.bash ]]; then
    source /opt/panda_ws/install/setup.bash
    export LD_LIBRARY_PATH="/opt/panda_libfranka/lib:${LD_LIBRARY_PATH:-}"
  elif [[ -f /opt/franka_ros2_ws/install/setup.bash ]]; then
    source /opt/franka_ros2_ws/install/setup.bash
  fi
  if [[ -f /opt/overlay_ws/install/local_setup.bash ]]; then
    source /opt/overlay_ws/install/local_setup.bash
  fi
  set -u
  export PATH="/opt/uv/venv/bin:$PATH"
  if [[ -z "${FRANKA_ROBOT_IP:-}" ]]; then
    echo "[collect] FRANKA_ROBOT_IP is missing; start with the project's Compose wrapper." >&2
    exit 1
  fi
  if ! command -v setsid >/dev/null; then
    echo "[collect] setsid is required to manage collection processes." >&2
    exit 1
  fi
else
  if [[ ! -f "$docker_dir/.env" ]]; then
    echo "[collect] Create $docker_dir/.env from .env.example first." >&2
    exit 1
  fi
  if ! command -v docker >/dev/null 2>&1 || ! docker info >/dev/null 2>&1; then
    echo "[collect] Docker is unavailable or this user cannot access the daemon." >&2
    exit 1
  fi
  if ! docker image inspect data-collect/franka-ros2-humble:2.5.1-uv >/dev/null 2>&1; then
    echo "[collect] Build the image first: bash ros2_ws/docker/franka_humble/scripts/compose_safe.sh build franka_humble" >&2
    exit 1
  fi
fi

safe_compose="$docker_dir/scripts/compose_safe.sh"
device_compose="$docker_dir/scripts/compose_devices.sh"
dh5_compose="$docker_dir/scripts/compose_dh5.sh"
config_values="$(mktemp)"
containers=()
processes=()
process_names=()

cleanup() {
  trap - EXIT INT TERM
  if (( ${#containers[@]} )); then
    echo "[collect] Stopping collection containers..." >&2
    docker stop --time 5 "${containers[@]}" >/dev/null 2>&1 || true
    docker rm -f "${containers[@]}" >/dev/null 2>&1 || true
  fi
  if (( ${#processes[@]} )); then
    echo "[collect] Stopping collection processes..." >&2
    for pid in "${processes[@]}"; do
      kill -INT -- "-$pid" 2>/dev/null || true
    done
    for (( attempt=0; attempt<50; attempt++ )); do
      local alive=false
      for pid in "${processes[@]}"; do
        if kill -0 -- "-$pid" 2>/dev/null; then alive=true; fi
      done
      [[ "$alive" == true ]] || break
      sleep 0.1
    done
    for pid in "${processes[@]}"; do
      kill -TERM -- "-$pid" 2>/dev/null || true
    done
    sleep 1
    for pid in "${processes[@]}"; do
      kill -KILL -- "-$pid" 2>/dev/null || true
      wait "$pid" 2>/dev/null || true
    done
  fi
  rm -f -- "$config_values"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# Parse YAML with the container's Python in either environment.
config_python() {
  if [[ "$inside_docker" == true ]]; then
    python "$@"
  else
    bash "$safe_compose" run --rm -T --no-deps franka_humble python "$@"
  fi
}
config_python -c '
import pathlib
import sys
from control.robot_config import load_mapping
from control.motion_config import workflow_motion_config

path = pathlib.Path(sys.argv[1])
config = load_mapping(path)
workflow_motion_config(config, control_mode=config.get("control", {}).get("control_mode", "ee"))
if not isinstance(config, dict):
    raise ValueError(f"Invalid config: {path}")
if not pathlib.Path("/opt/overlay_ws/install/setup.bash").is_file():
    raise RuntimeError("ROS overlay is missing; run scripts/build_overlay.sh as in README.md")

robot = config.get("robot", {})
camera = config.get("camera", {})
gripper = config.get("gripper", {})
dataset = config.get("dataset", {})
dataset_format = str(dataset.get("lerobot_format", "v2")).strip().lower()
if dataset_format != "v2":
    raise ValueError(
        "collect.sh starts the Humble image and requires dataset.lerobot_format: v2; "
        "the Jazzy image does not have robot bringup yet"
    )
backend = str(camera.get("camera_backend", "none")).lower()
kind = str(gripper.get("gripper_type", "franka")).lower()
if backend not in {"ros", "none"}:
    raise ValueError(f"Unsupported camera_backend: {backend}")
if kind not in {"franka", "dh5", "none"}:
    raise ValueError(f"Unsupported gripper_type: {kind}")
values = [
    str(robot.get("robot_type", "panda")),
    str(bool(robot.get("use_fake_hardware", True))).lower(),
    backend,
    str(camera.get("external_camera_serial", "")),
    str(camera.get("wrist_camera_serial", "")),
    kind,
    str(bool(gripper.get("soft_gripper_use_fake", False))).lower(),
    str(gripper.get("soft_gripper_port", "/dev/ttyUSB0")),
    str(bool(gripper.get("enable_soft_gripper_cameras", True))).lower(),
    str(bool(dataset.get("enable_logging", False))).lower(),
]
sys.stdout.buffer.write("\0".join(values).encode() + b"\0")
' "$config_relative" > "$config_values"
mapfile -d '' -t settings < "$config_values"
if (( ${#settings[@]} != 10 )); then
  echo "[collect] Could not read collection settings." >&2
  exit 1
fi

robot_type="${settings[0]}"
fake_hardware="${settings[1]}"
camera_backend="${settings[2]}"
external_serial="${settings[3]}"
wrist_serial="${settings[4]}"
gripper_type="${settings[5]}"
dh5_fake="${settings[6]}"
dh5_port="${settings[7]}"
dh5_cameras="${settings[8]}"
logging="${settings[9]}"

bringup_compose="$safe_compose"
start_cameras=false
load_gripper=false
start_dh5=false
if [[ "$camera_backend" == ros ]]; then
  start_cameras=true
  bringup_compose="$device_compose"
fi
if [[ "$gripper_type" == franka ]]; then
  load_gripper=true
elif [[ "$gripper_type" == dh5 ]]; then
  start_dh5=true
  bringup_compose="$dh5_compose"
fi

echo "[collect] Config: $config_relative | fake_hardware=$fake_hardware | cameras=$start_cameras | logging=$logging"
if [[ "$logging" == false ]]; then
  echo "[collect] Dataset logging is disabled in the config; this run will not save episodes." >&2
fi

if [[ "$inside_docker" == true ]]; then
  start_process() {
    local name="$1"
    shift
    # Background shell jobs inherit ignored SIGINT. Reset it before exec so
    # Ctrl+C lets ROS and the recorder run their normal shutdown handlers.
    setsid /opt/uv/venv/bin/python -c '
import os
import signal
import sys
signal.signal(signal.SIGINT, signal.SIG_DFL)
os.execvp(sys.argv[1], sys.argv[1:])
' "$@" &
    processes+=("$!")
    process_names+=("$name")
  }
  start_process ros ros2 launch data_collect_franka franka_data_collect.launch.py \
    "robot_ip:=$FRANKA_ROBOT_IP" \
    "robot_type:=$robot_type" "use_fake_hardware:=$fake_hardware" \
    "load_gripper:=$load_gripper" "start_cameras:=$start_cameras" \
    "external_serial:=$external_serial" "wrist_serial:=$wrist_serial" \
    "start_dh5_gripper:=$start_dh5" "dh5_use_fake:=$dh5_fake" \
    "dh5_port:=$dh5_port" "start_dh5_cameras:=$dh5_cameras"
  echo "[collect] Waiting for /joint_states from ROS bringup..."
  start_process readiness timeout 120 ros2 topic echo /joint_states sensor_msgs/msg/JointState --once >/dev/null
  readiness_pid="${processes[1]}"
  while kill -0 "$readiness_pid" 2>/dev/null; do
    if ! kill -0 "${processes[0]}" 2>/dev/null; then
      echo "[collect] ROS bringup exited before becoming ready." >&2
      exit 1
    fi
    sleep 0.2
  done
  wait "$readiness_pid" || {
    echo "[collect] ROS bringup readiness check failed." >&2
    exit 1
  }
  # The readiness process has finished; only supervise persistent components.
  unset 'processes[1]' 'process_names[1]'
  processes=("${processes[@]}")
  process_names=("${process_names[@]}")
  start_process vr python -m teleop_xr.ros2 --mode teleop
  start_process recorder python vr_collect.py --config "$config_relative"
  echo "[collect] Running inside Humble Docker. Press Ctrl+C to stop all components."
  while :; do
    for index in "${!processes[@]}"; do
      if ! kill -0 "${processes[$index]}" 2>/dev/null; then
        status=0
        wait "${processes[$index]}" || status=$?
        echo "[collect] Process exited: ${process_names[$index]} (status $status)." >&2
        if [[ "${process_names[$index]}" == recorder && "$status" == 0 ]]; then
          exit 0
        fi
        exit 1
      fi
    done
    sleep 0.2
  done
fi

run_id="${UID:-$(id -u)}-$$"
ros_name="franka-collect-ros-$run_id"
vr_name="franka-collect-vr-$run_id"
recorder_name="franka-collect-recorder-$run_id"

start_container() {
  local name="$1" compose_script="$2"
  shift 2
  bash "$compose_script" run -d -T --no-deps --name "$name" franka_humble "$@" >/dev/null
  containers+=("$name")
  docker logs -f "$name" 2>&1 | sed -u "s/^/[$name] /" &
}

start_container "$ros_name" "$bringup_compose" bash -c '
  exec ros2 launch data_collect_franka franka_data_collect.launch.py \
    "robot_ip:=$FRANKA_ROBOT_IP" "$@"
' _ \
  "robot_type:=$robot_type" "use_fake_hardware:=$fake_hardware" \
  "load_gripper:=$load_gripper" "start_cameras:=$start_cameras" \
  "external_serial:=$external_serial" "wrist_serial:=$wrist_serial" \
  "start_dh5_gripper:=$start_dh5" "dh5_use_fake:=$dh5_fake" \
  "dh5_port:=$dh5_port" "start_dh5_cameras:=$dh5_cameras"

echo "[collect] Waiting for /joint_states from ROS bringup..."
docker exec "$ros_name" bash -c \
  'source /opt/ros/humble/setup.bash; timeout 120 ros2 topic echo /joint_states sensor_msgs/msg/JointState --once >/dev/null'

start_container "$vr_name" "$safe_compose" python -m teleop_xr.ros2 --mode teleop
start_container "$recorder_name" "$safe_compose" python vr_collect.py --config "$config_relative"
echo "[collect] Running. Press Ctrl+C to stop all three containers."

while :; do
  for name in "${containers[@]}"; do
    state="$(docker inspect --format '{{.State.Running}} {{.State.ExitCode}}' "$name")" || {
      echo "[collect] Container disappeared: $name" >&2
      exit 1
    }
    if [[ "$state" != true\ * ]]; then
      echo "[collect] Container exited: $name (status ${state#* })." >&2
      if [[ "$name" == "$recorder_name" && "${state#* }" == 0 ]]; then
        exit 0
      fi
      exit 1
    fi
  done
  sleep 2
done
