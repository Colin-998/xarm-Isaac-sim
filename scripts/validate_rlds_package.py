"""Validate RLDS-shaped JSONL package produced from xArm conveyor episodes."""

import argparse
import json
from pathlib import Path

from PIL import Image


parser = argparse.ArgumentParser()
parser.add_argument("package_root", type=Path)
parser.add_argument("--expected-episodes", type=int, default=500)
parser.add_argument("--expected-steps", type=int, default=32000)
args = parser.parse_args()

root = args.package_root.resolve()
info = json.loads((root / "dataset_info.json").read_text(encoding="utf-8"))
if info["num_episodes"] != args.expected_episodes:
    raise RuntimeError(
        f"Expected {args.expected_episodes} episodes, found {info['num_episodes']}"
    )
if info["num_steps"] != args.expected_steps:
    raise RuntimeError(
        f"Expected {args.expected_steps} steps, found {info['num_steps']}"
    )

episode_count = 0
step_count = 0
for shard in info["shards"]:
    shard_path = root / shard["path"]
    if not shard_path.exists():
        raise RuntimeError(f"Missing shard: {shard_path}")
    shard_episodes = 0
    with shard_path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            episode = json.loads(line)
            steps = episode["steps"]
            if len(steps) < 64:
                raise RuntimeError(f"Episode too short: {episode['episode_id']}")
            for index, step in enumerate(steps):
                observation = step["observation"]
                if len(step["action"]) != 7:
                    raise RuntimeError(f"Bad action in {episode['episode_id']}")
                if len(observation["state"]) != 13:
                    raise RuntimeError(f"Bad state in {episode['episode_id']}")
                if step["is_first"] != (index == 0):
                    raise RuntimeError(f"Bad is_first in {episode['episode_id']}")
                if step["is_last"] != (index == len(steps) - 1):
                    raise RuntimeError(f"Bad is_last in {episode['episode_id']}")
                if step["is_terminal"] != step["is_last"]:
                    raise RuntimeError(f"Bad is_terminal in {episode['episode_id']}")
                expected_reward = 1.0 if step["is_last"] else 0.0
                expected_discount = 0.0 if step["is_last"] else 1.0
                if step["reward"] != expected_reward:
                    raise RuntimeError(f"Bad reward in {episode['episode_id']}")
                if step["discount"] != expected_discount:
                    raise RuntimeError(f"Bad discount in {episode['episode_id']}")
                image_path = root / observation["image"]
                if not image_path.exists():
                    raise RuntimeError(f"Missing image: {image_path}")
                with Image.open(image_path) as image:
                    if image.mode != "RGB" or image.size != (256, 256):
                        raise RuntimeError(
                            f"Bad image {image_path}: {image.mode} {image.size}"
                        )
            shard_episodes += 1
            episode_count += 1
            step_count += len(steps)
    if shard_episodes != shard["episodes"]:
        raise RuntimeError(f"Shard count mismatch: {shard_path.name}")

if episode_count != args.expected_episodes or step_count != args.expected_steps:
    raise RuntimeError(
        f"Count mismatch: episodes={episode_count}, steps={step_count}"
    )

print(
    f"Validated RLDS package: {episode_count} episodes, {step_count} steps.",
    flush=True,
)
