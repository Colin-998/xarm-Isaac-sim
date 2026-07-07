"""Package recorded xArm conveyor episodes into RLDS-shaped JSONL shards."""

import argparse
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("--dataset-root", type=Path, required=True)
parser.add_argument("--output-root", type=Path, required=True)
parser.add_argument("--instruction", default="Pick up the red cube and place it at the conveyor start while avoiding obstacles.")
parser.add_argument("--expected-episodes", type=int, default=500)
parser.add_argument("--shard-size", type=int, default=50)
parser.add_argument("--validation-episodes", type=int, default=50)
parser.add_argument(
    "--copy-mode",
    choices=("hardlink", "copy"),
    default="hardlink",
    help="Use hardlinks by default to keep the package self-contained without duplicating disk blocks.",
)
args = parser.parse_args()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_rows(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def link_or_copy(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return
    if args.copy_mode == "hardlink":
        try:
            os.link(source, target)
            return
        except OSError:
            pass
    shutil.copy2(source, target)


def flatten_state(observation):
    gripper = observation["gripper_joint_positions"]
    return (
        observation["arm_joint_positions"]
        + [gripper["drive_joint"]]
        + observation["tcp_position"]
        + observation["cube_position"]
    )


def make_step(episode_dir, row, index, last_index, image_target_dir):
    image_source = episode_dir / row["image"]
    image_target = image_target_dir / row["image"]
    link_or_copy(image_source, image_target)
    relative_image = image_target.relative_to(args.output_root).as_posix()
    action = row["action"]["arm_joint_positions"] + [
        row["action"]["gripper_joint_position"]
    ]
    is_last = index == last_index
    return {
        "observation": {
            "image": relative_image,
            "natural_language_instruction": args.instruction,
            "state": flatten_state(row["observation"]),
            "arm_joint_positions": row["observation"]["arm_joint_positions"],
            "arm_joint_velocities": row["observation"]["arm_joint_velocities"],
            "gripper_joint_positions": row["observation"]["gripper_joint_positions"],
            "tcp_position": row["observation"]["tcp_position"],
            "tcp_rotation_matrix": row["observation"]["tcp_rotation_matrix"],
            "cube_position": row["observation"]["cube_position"],
            "cube_orientation_wxyz": row["observation"]["cube_orientation_wxyz"],
            "obstacles": row["observation"]["obstacles"],
            "phase": row["phase"],
            "time_seconds": row["time_seconds"],
        },
        "action": action,
        "action_dict": row["action"],
        "reward": 1.0 if is_last else 0.0,
        "discount": 0.0 if is_last else 1.0,
        "is_first": index == 0,
        "is_last": is_last,
        "is_terminal": is_last,
    }


dataset_root = args.dataset_root.resolve()
args.output_root = args.output_root.resolve()
episodes_dir = args.output_root / "episodes"
images_dir = args.output_root / "images"
episodes_dir.mkdir(parents=True, exist_ok=True)
images_dir.mkdir(parents=True, exist_ok=True)

episode_dirs = sorted(
    path
    for path in dataset_root.glob("episode_[0-9][0-9][0-9][0-9][0-9]")
    if path.is_dir()
)
if len(episode_dirs) != args.expected_episodes:
    raise RuntimeError(
        f"Expected {args.expected_episodes} source episodes, found {len(episode_dirs)}"
    )
if args.shard_size <= 0:
    raise ValueError("--shard-size must be positive")

manifest = read_json(dataset_root / "dataset_manifest.json")
num_shards = (len(episode_dirs) + args.shard_size - 1) // args.shard_size
total_steps = 0
shards = []

for shard_index in range(num_shards):
    start = shard_index * args.shard_size
    shard_episode_dirs = episode_dirs[start : start + args.shard_size]
    shard_name = f"rlds-{shard_index:05d}-of-{num_shards:05d}.jsonl"
    shard_path = episodes_dir / shard_name
    with shard_path.open("w", encoding="utf-8") as stream:
        for episode_dir in shard_episode_dirs:
            metadata = read_json(episode_dir / "metadata.json")
            rows = read_rows(episode_dir / "actions.jsonl")
            if not metadata.get("success"):
                raise RuntimeError(f"Refusing failed episode: {episode_dir.name}")
            if len(rows) != metadata["frames_written"]:
                raise RuntimeError(f"Frame count mismatch: {episode_dir.name}")
            image_target_dir = images_dir / episode_dir.name
            steps = [
                make_step(
                    episode_dir,
                    row,
                    index,
                    len(rows) - 1,
                    image_target_dir,
                )
                for index, row in enumerate(rows)
            ]
            total_steps += len(steps)
            stream.write(
                json.dumps(
                    {
                        "episode_id": episode_dir.name,
                        "episode_metadata": metadata,
                        "steps": steps,
                    },
                    separators=(",", ":"),
                )
                + "\n"
            )
    shards.append(
        {
            "path": shard_path.relative_to(args.output_root).as_posix(),
            "episodes": len(shard_episode_dirs),
        }
    )

train_episodes = max(len(episode_dirs) - args.validation_episodes, 0)
dataset_info = {
    "dataset_name": "xarm6_curobo_conveyor_rlds",
    "schema": "RLDS JSONL",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "source_dataset": str(dataset_root),
    "source_manifest": manifest,
    "language_instruction": args.instruction,
    "num_episodes": len(episode_dirs),
    "num_steps": total_steps,
    "capture_fps": manifest["capture_fps"],
    "resolution": manifest["resolution"],
    "shards": shards,
    "splits": {
        "train": {
            "episodes": train_episodes,
            "range": [0, train_episodes],
        },
        "validation": {
            "episodes": len(episode_dirs) - train_episodes,
            "range": [train_episodes, len(episode_dirs)],
        },
    },
}
(args.output_root / "dataset_info.json").write_text(
    json.dumps(dataset_info, indent=2),
    encoding="utf-8",
)
(args.output_root / "features.json").write_text(
    json.dumps(
        {
            "episode": {
                "episode_id": "string",
                "episode_metadata": "dict",
                "steps": "sequence<RLDSStep>",
            },
            "step": {
                "observation": {
                    "image": "relative PNG path, RGB 256x256",
                    "natural_language_instruction": "string",
                    "state": "float[13]: arm6 + gripper + tcp3 + cube3",
                    "arm_joint_positions": "float[6]",
                    "arm_joint_velocities": "float[6]",
                    "gripper_joint_positions": "dict",
                    "tcp_position": "float[3]",
                    "tcp_rotation_matrix": "float[3,3]",
                    "cube_position": "float[3]",
                    "cube_orientation_wxyz": "float[4]",
                    "obstacles": "sequence<dict>",
                    "phase": "string",
                    "time_seconds": "float",
                },
                "action": "float[7]: arm6 + gripper",
                "reward": "float",
                "discount": "float",
                "is_first": "bool",
                "is_last": "bool",
                "is_terminal": "bool",
            },
        },
        indent=2,
    ),
    encoding="utf-8",
)
print(
    f"Packaged {len(episode_dirs)} episodes, {total_steps} steps into {args.output_root}",
    flush=True,
)
