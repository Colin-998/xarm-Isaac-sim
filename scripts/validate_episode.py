import argparse
import json
from pathlib import Path

from PIL import Image


parser = argparse.ArgumentParser()
parser.add_argument("episode_dir", type=Path)
args = parser.parse_args()

episode_dir = args.episode_dir.resolve()
metadata = json.loads((episode_dir / "metadata.json").read_text(encoding="utf-8"))
rows = [
    json.loads(line)
    for line in (episode_dir / "actions.jsonl").read_text(encoding="utf-8").splitlines()
    if line.strip()
]

if not metadata["success"]:
    raise RuntimeError("Episode metadata reports a failed grasp")
if len(rows) != metadata["frames_written"]:
    raise RuntimeError("JSONL row count does not match metadata")
if len(rows) < 2:
    raise RuntimeError("Episode has too few observations")

frame_step = metadata["physics_fps"] // metadata["capture_fps"]
for previous, current in zip(rows, rows[1:]):
    if current["frame"] - previous["frame"] != frame_step:
        raise RuntimeError("Observation spacing is not a constant 4 FPS")

for row in rows:
    image_path = episode_dir / row["image"]
    if not image_path.exists():
        raise RuntimeError(f"Missing image: {image_path.name}")
    with Image.open(image_path) as image:
        if list(image.size) != metadata["resolution"]:
            raise RuntimeError(f"Unexpected resolution for {image_path.name}: {image.size}")

print(
    f"Validated {episode_dir}: {len(rows)} frames, "
    f"{metadata['capture_fps']} FPS, resolution={metadata['resolution']}, "
    f"final_z={metadata['cube_final_position'][2]:.4f} m"
)
