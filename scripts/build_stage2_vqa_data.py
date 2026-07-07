"""Build Stage-2 image-text QA samples from deterministic annotations."""

import argparse
import json
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("--annotations", type=Path, required=True)
parser.add_argument("--output", type=Path, default=Path("outputs/stage2_vqa.jsonl"))
parser.add_argument("--max-episodes", type=int, default=0)
args = parser.parse_args()


def iter_annotations(path):
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


args.output.parent.mkdir(parents=True, exist_ok=True)
sample_count = 0
episode_count = 0
answer_vocab = {}
with args.output.open("w", encoding="utf-8") as stream:
    for episode in iter_annotations(args.annotations):
        if args.max_episodes and episode_count >= args.max_episodes:
            break
        episode_count += 1
        for step in episode["steps"]:
            for qa in step.get("question_answer", []):
                answer = qa["answer"].strip()
                answer_vocab.setdefault(answer, len(answer_vocab))
                stream.write(
                    json.dumps(
                        {
                            "episode_id": episode["episode_id"],
                            "step_index": step["step_index"],
                            "image": step["image"],
                            "phase": step["phase"],
                            "caption": step["caption"],
                            "question": qa["question"],
                            "answer": answer,
                            "answer_id": answer_vocab[answer],
                            "prompt": (
                                f"Instruction: {episode['language_instruction']} "
                                f"Question: {qa['question']}"
                            ),
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
    "answers": answer_vocab,
    "schema": "stage2_image_text_qa_jsonl",
}
args.output.with_suffix(".manifest.json").write_text(
    json.dumps(manifest, indent=2),
    encoding="utf-8",
)
print(f"Wrote {sample_count} VQA samples from {episode_count} episodes to {args.output}")
