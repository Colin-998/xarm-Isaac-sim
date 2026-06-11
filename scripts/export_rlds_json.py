import argparse
import json
from pathlib import Path


parser = argparse.ArgumentParser(
    description="Convert one recorded episode to an RLDS-shaped JSON document."
)
parser.add_argument("episode_dir", type=Path)
parser.add_argument("--instruction", default="Pick up the red cube.")
args = parser.parse_args()

episode_dir = args.episode_dir.resolve()
metadata = json.loads((episode_dir / "metadata.json").read_text(encoding="utf-8"))
source_rows = [
    json.loads(line)
    for line in (episode_dir / "actions.jsonl").read_text(encoding="utf-8").splitlines()
    if line.strip()
]

steps = []
last_index = len(source_rows) - 1
for index, row in enumerate(source_rows):
    steps.append(
        {
            "observation": {
                **row["observation"],
                "image_path": row["image"],
                "language_instruction": args.instruction,
            },
            "action": row["action"],
            "reward": 1.0 if index == last_index and metadata["success"] else 0.0,
            "discount": 1.0,
            "is_first": index == 0,
            "is_last": index == last_index,
            "is_terminal": index == last_index,
        }
    )

output = {
    "episode_metadata": metadata,
    "steps": steps,
}
output_path = episode_dir / "rlds_episode.json"
output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
print(f"Wrote {output_path}")
