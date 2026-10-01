# Jazzy + LeRobot 0.6.1 experiment

This is an isolated ROS 2 Jazzy / Python 3.12 image for LeRobot 0.6.1
recording. It does not replace the Humble robot bringup or start a robot.
The synthetic smoke test checks ROS Python and `cv_bridge` imports, creates a
LeRobot v3 dataset using the collector's core feature schema, reads it, then
resumes it for another episode. The image rebuilds `cv_bridge` 4.1.0 against
NumPy 2 because Jazzy's prebuilt extension uses the NumPy 1 ABI.

From the repository root:

```bash
bash ros2_ws/docker/franka_jazzy/scripts/compose_safe.sh build
bash ros2_ws/docker/franka_jazzy/scripts/compose_safe.sh run --rm franka_jazzy \
  python ros2_ws/docker/franka_jazzy/scripts/smoke_test.py
bash ros2_ws/docker/franka_jazzy/scripts/compose_safe.sh run --rm franka_jazzy \
  python tests/docker_dataset_adapter_smoke.py --format v3
bash ros2_ws/docker/franka_jazzy/scripts/compose_safe.sh run --rm franka_jazzy \
  python tests/docker_recording_smoke.py --backend jazzy
bash ros2_ws/docker/franka_jazzy/scripts/compose_safe.sh run --rm franka_jazzy \
  python tests/docker_collect_pipeline_smoke.py --backend jazzy
```

The wrapper reuses the host settings in `franka_humble/.env`, but its image,
Compose project, Python environment, and temporary test datasets are separate.
The project mount is writable for future recordings, using the configured host
UID/GID. It has no USB device mapping and does not launch `franka_ros2`.

`vr_collect.py` is the sole collector entry point. Select `v2` or `v3` with
`dataset.lerobot_format` in the collection YAML, or override it with
`--lerobot-format v3`. Both formats share the same frame schema and recording
interface. The tests exercise that interface (including resume, discard, and
DH5 features), the production writer, and the collector with synthetic frames;
they do not connect to a robot. The Jazzy image does not yet contain the Franka
ROS packages, overlay, VR publisher, or camera/USB bringup required for live
Jazzy collection. A v3 dataset also needs a new `repo_id` or date; the adapter
refuses to append v3 episodes to a Humble dataset.
