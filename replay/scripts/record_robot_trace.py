"""Record a real Panda trace in Humble; --move enables a small ROS round trip."""

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from control.robotic_arm_controller_ros import RoboticArmControlerRos


def record_robot_trace(
    output: Path, *, move: bool = False, config_path: Path = Path("config/planer/panda.yaml")
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    # Create exclusively before connecting or commanding the robot.
    with output.open("x", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["timestamp"]
            + [f"q{i}" for i in range(7)]
            + [f"transform{i}" for i in range(16)]
            + [f"command{i}" for i in range(7)]
        )
        arm = RoboticArmControlerRos(config_path=config_path)
        try:
            arm.connect()
            arm.wait_ready(required=("joints", "ee"))
            start = arm.get_state()["joint_positions"].copy()
            target = start.copy()
            target[0] += 0.02
            if abs(target[0]) > 2.7:
                raise ValueError("Joint 1 is too near its limit for this small test")
            print(arm.get_status(), flush=True)
            begin = time.monotonic()
            count = 0

            def sample(state):
                nonlocal count
                arm.get_state(required=("joints", "ee"))
                elapsed = time.monotonic() - begin
                command = start.copy()
                if move and elapsed < 8.0:
                    phase = elapsed / 4.0 if elapsed < 4.0 else (8.0 - elapsed) / 4.0
                    command[0] += 0.02 * (10 * phase**3 - 15 * phase**4 + 6 * phase**5)
                writer.writerow(
                    [elapsed]
                    + state["joint_positions"].tolist()
                    + state["end_effector_pose"].flatten(order="F").tolist()
                    + command.tolist()
                )
                count += 1

            if move:
                print("ROS round trip: joint 1 +0.02 rad then return, 8 seconds", flush=True)
                points = [
                    {
                        "time_s": t,
                        "positions": q,
                        "velocities": np.zeros(7),
                        "accelerations": np.zeros(7),
                    }
                    for t, q in ((0.0, start), (4.0, target), (8.0, start))
                ]
                result = arm.execute_joint_trajectory(points, on_sample=sample)
                print("Trajectory result:", result, flush=True)
            duration = 10.0 if move else 2.0
            while time.monotonic() - begin < duration:
                sample(arm.get_state(required=("joints", "ee")))
                time.sleep(0.01)
            print(f"Saved {count} measured samples: {output}", flush=True)
            print("Return error (rad):", arm.get_state()["joint_positions"] - start, flush=True)
        finally:
            arm.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--move", action="store_true")
    parser.add_argument("--config", type=Path, default=Path("config/planer/panda.yaml"))
    args = parser.parse_args()
    record_robot_trace(args.output, move=args.move, config_path=args.config)
