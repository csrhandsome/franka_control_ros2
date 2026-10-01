#!/usr/bin/env bash
# Start the ROS bringup, VR publisher, and recorder with the existing Docker image.
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
cd "$repo_root"

usage() {
  cat <<'EOF'
Usage: ./collect.sh [config/collect/franka.yaml]

Starts ROS bringup, VR topics, and vr_collect.py in Docker. Press Ctrl+C to
stop all three containers. Build the image and overlay once as described in
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

safe_compose="$docker_dir/scripts/compose_safe.sh"
device_compose="$docker_dir/scripts/compose_devices.sh"
dh5_compose="$docker_dir/scripts/compose_dh5.sh"
config_values="$(mktemp)"
containers=()

cleanup() {
  trap - EXIT INT TERM
  if (( ${#containers[@]} )); then
    echo "[collect] Stopping collection containers..." >&2
    docker stop --time 5 "${containers[@]}" >/dev/null 2>&1 || true
    docker rm -f "${containers[@]}" >/dev/null 2>&1 || true
  fi
  rm -f -- "$config_values"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# Parse YAML with the container's Python. The host does not need a .venv.
bash "$safe_compose" run --rm -T --no-deps franka_humble \
  python -c '
import pathlib
import sys
import yaml

path = pathlib.Path(sys.argv[1])
config = yaml.safe_load(path.read_text(encoding="utf-8"))
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
    str(robot.get("robot_type", "fr3")),
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
  'source /opt/ros/humble/setup.bash; timeout 120 ros2 topic echo /joint_states --once >/dev/null'

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
