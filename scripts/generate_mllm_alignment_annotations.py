"""Generate deterministic MLLM alignment annotations for xArm RLDS episodes."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


PHASE_CAPTIONS = {
    "observe_cube": "The robot observes the red cube at the conveyor pickup end.",
    "approach_cube": "The gripper moves toward the visible red cube.",
    "descend_to_cube": "The gripper descends to align with the red cube.",
    "close_gripper": "The gripper closes around the red cube.",
    "lift_cube": "The robot lifts the red cube away from the conveyor.",
    "carry_to_start": "The arm carries the red cube around the yellow obstacles.",
    "place_cube": "The robot lowers the red cube at the conveyor start.",
    "open_gripper": "The gripper opens to release the red cube.",
    "retreat_after_release": "The arm retreats after releasing the red cube.",
    "conveyor_return": "The conveyor moves the red cube back toward the pickup end.",
    "conveyor_settle": "The cube settles near the conveyor pickup end.",
    "observe_returned_cube": "The robot observes the returned red cube for the next cycle.",
}

PHASE_GOALS = {
    "observe_cube": "locate cube",
    "approach_cube": "approach cube",
    "descend_to_cube": "align gripper",
    "close_gripper": "grasp cube",
    "lift_cube": "lift cube",
    "carry_to_start": "avoid obstacles and carry",
    "place_cube": "place cube",
    "open_gripper": "release cube",
    "retreat_after_release": "clear workspace",
    "conveyor_return": "wait for conveyor return",
    "conveyor_settle": "wait for cube stability",
    "observe_returned_cube": "confirm returned cube",
}


parser = argparse.ArgumentParser()
parser.add_argument("--rlds-root", type=Path, required=True)
parser.add_argument(
    "--output",
    type=Path,
    default=Path("outputs/mllm_alignment_annotations.jsonl"),
)
args = parser.parse_args()


def load_episodes(root):
    info = json.loads((root / "dataset_info.json").read_text(encoding="utf-8"))
    for shard in info["shards"]:
        shard_path = root / shard["path"]
        with shard_path.open("r", encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    yield json.loads(line)


def cube_relation(step):
    cube = step["observation"]["cube_position"]
    if cube[1] > 0.20:
        return "near the conveyor pickup end"
    if cube[1] < -0.20:
        return "near the conveyor start"
    if cube[2] > 0.16:
        return "held above the conveyor"
    return "moving along the curved conveyor"


root = args.rlds_root.resolve()
args.output.parent.mkdir(parents=True, exist_ok=True)
rows_written = 0
with args.output.open("w", encoding="utf-8") as output_stream:
    for episode in load_episodes(root):
        steps = []
        for index, step in enumerate(episode["steps"]):
            phase = step["observation"]["phase"]
            caption = PHASE_CAPTIONS.get(
                phase,
                f"The robot is in phase {phase}.",
            )
            relation = cube_relation(step)
            steps.append(
                {
                    "step_index": index,
                    "image": step["observation"]["image"],
                    "phase": phase,
                    "phase_goal": PHASE_GOALS.get(phase, phase),
                    "caption": f"{caption} The red cube is {relation}.",
                    "question_answer": [
                        {
                            "question": "What should the robot focus on now?",
                            "answer": PHASE_GOALS.get(phase, phase),
                        },
                        {
                            "question": "Where is the red cube?",
                            "answer": relation,
                        },
                    ],
                }
            )
        output_stream.write(
            json.dumps(
                {
                    "episode_id": episode["episode_id"],
                    "language_instruction": (
                        "Pick up the red cube, avoid the yellow obstacles, "
                        "place it at the conveyor start, and wait for it to return."
                    ),
                    "instruction_variants": [
                        "Move the red cube from the pickup end to the conveyor start while avoiding obstacles.",
                        "Grasp the red cube, carry it around the yellow blocks, and release it at the start of the belt.",
                        "Complete one safe conveyor cycle with the red cube and the xArm6 gripper.",
                    ],
                    "episode_summary": (
                        "The xArm6 observes, grasps, lifts, carries, places, "
                        "and then waits for the conveyor to return the red cube."
                    ),
                    "steps": steps,
                },
                separators=(",", ":"),
            )
            + "\n"
        )
        rows_written += 1

manifest = {
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "source": str(root),
    "output": str(args.output.resolve()),
    "episodes": rows_written,
    "annotation_policy": "deterministic phase templates with cube spatial relation",
}
(args.output.with_suffix(".manifest.json")).write_text(
    json.dumps(manifest, indent=2),
    encoding="utf-8",
)
print(f"Wrote {rows_written} alignment annotation episodes to {args.output}")
