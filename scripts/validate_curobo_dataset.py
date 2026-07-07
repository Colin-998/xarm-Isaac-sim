"""Validate cuRobo conveyor episodes before V-JEPA2 training."""

import argparse
import json
from pathlib import Path

from PIL import Image


parser = argparse.ArgumentParser()
parser.add_argument("dataset_root", type=Path)
parser.add_argument("--expected-episodes", type=int)
args = parser.parse_args()

root = args.dataset_root.resolve()
incomplete_dirs = sorted(root.glob(".episode_*.tmp"))
if incomplete_dirs:
    raise RuntimeError(
        "Incomplete temporary episodes found: "
        + ", ".join(path.name for path in incomplete_dirs)
    )
episode_dirs = sorted(
    path
    for path in root.glob("episode_[0-9][0-9][0-9][0-9][0-9]")
    if path.is_dir()
)
if args.expected_episodes is not None:
    if len(episode_dirs) != args.expected_episodes:
        raise RuntimeError(
            f"Expected {args.expected_episodes} episodes, "
            f"found {len(episode_dirs)}"
        )
if not episode_dirs:
    raise RuntimeError(f"No episodes found below {root}")

total_frames = 0
for episode_dir in episode_dirs:
    metadata_path = episode_dir / "metadata.json"
    actions_path = episode_dir / "actions.jsonl"
    if not metadata_path.exists() or not actions_path.exists():
        raise RuntimeError(f"Incomplete episode: {episode_dir.name}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for line in actions_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not metadata.get("success"):
        raise RuntimeError(f"Failed episode included: {episode_dir.name}")
    if metadata["capture_fps"] != 4:
        raise RuntimeError(f"Unexpected FPS: {episode_dir.name}")
    if metadata["resolution"] != [256, 256]:
        raise RuntimeError(f"Unexpected resolution: {episode_dir.name}")
    if len(rows) != metadata["frames_written"] or len(rows) < 64:
        raise RuntimeError(f"Invalid row count: {episode_dir.name}")

    expected_step = (
        metadata["physics_fps"] // metadata["capture_fps"]
    )
    for previous, current in zip(rows, rows[1:]):
        if current["frame"] - previous["frame"] != expected_step:
            raise RuntimeError(
                f"Non-constant frame spacing: {episode_dir.name}"
            )
    for row in rows:
        action = row["action"]
        observation = row["observation"]
        if len(action["arm_joint_positions"]) != 6:
            raise RuntimeError(f"Invalid action: {episode_dir.name}")
        if len(observation["arm_joint_positions"]) != 6:
            raise RuntimeError(f"Invalid observation: {episode_dir.name}")
        image_path = episode_dir / row["image"]
        if not image_path.exists():
            raise RuntimeError(f"Missing image: {image_path}")
        with Image.open(image_path) as image:
            if image.mode != "RGB" or image.size != (256, 256):
                raise RuntimeError(
                    f"Invalid image {image_path}: "
                    f"mode={image.mode}, size={image.size}"
                )
    metrics = metadata["metrics"]
    if not metrics["placed_at_start"] or not metrics["returned_to_end"]:
        raise RuntimeError(
            f"Physical validation failed: {episode_dir.name}"
        )
    total_frames += len(rows)

print(
    f"Validated {len(episode_dirs)} episodes, "
    f"{total_frames} RGB/action pairs, 4 FPS, 256x256.",
    flush=True,
)
