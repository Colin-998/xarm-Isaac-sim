"""Build Stage-1 projector-alignment samples from MLLM annotations."""

import argparse
import json
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("--annotations", type=Path, required=True)
parser.add_argument("--output", type=Path, default=Path("outputs/stage1_projector_alignment.jsonl"))
parser.add_argument("--max-episodes", type=int, default=0)
parser.add_argument("--stride", type=int, default=4)
args = parser.parse_args()


def iter_annotations(path):
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


args.output.parent.mkdir(parents=True, exist_ok=True)
sample_count = 0
episode_count = 0
with args.output.open("w", encoding="utf-8") as stream:
    for episode in iter_annotations(args.annotations):
        if args.max_episodes and episode_count >= args.max_episodes:
            break
        episode_count += 1
        for step in episode["steps"][:: max(1, args.stride)]:
            qa_text = " ".join(
                f"Question: {qa['question']} Answer: {qa['answer']}"
                for qa in step.get("question_answer", [])
            )
            text = (
                f"Instruction: {episode['language_instruction']} "
                f"Observation: {step['caption']} "
                f"Goal: {step['phase_goal']}. {qa_text}"
            )
            stream.write(
                json.dumps(
                    {
                        "episode_id": episode["episode_id"],
                        "step_index": step["step_index"],
                        "image": step["image"],
                        "phase": step["phase"],
                        "text": text,
                    },
                    separators=(",", ":"),
                )
                + "\n"
            )
            sample_count += 1

manifest = {
    "annotations": str(args.annotations.resolve()),
    "output": str(args.output.resolve()),
    "episodes": episode_count,
    "samples": sample_count,
    "stride": args.stride,
    "schema": "stage1_projector_alignment_jsonl",
}
args.output.with_suffix(".manifest.json").write_text(
    json.dumps(manifest, indent=2),
    encoding="utf-8",
)
print(f"Wrote {sample_count} projector samples from {episode_count} episodes to {args.output}")
