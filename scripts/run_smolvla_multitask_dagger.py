"""Collect and train multi-task SmolVLA DAgger rollouts for xArm6.

The first task preserves the original conveyor loop.  Additional tasks share
the same cube grasp skill but change the language instruction and place target,
which is the smallest honest step toward testing generalization.
"""

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ISAAC_PYTHON = Path.home() / "isaac_sim_5.1" / "python.bat"
DEFAULT_POLICY = ROOT / "outputs/smolvla_xarm6_level2/best_smolvla_policy.pt"
DEFAULT_RLDS_ROOT = ROOT / "outputs/rlds_xarm6_curobo_500_v2"


TASKS = [
    {
        "name": "conveyor_cycle",
        "instruction": (
            "Pick up the red cube and place it at the conveyor start while "
            "avoiding obstacles, then let the conveyor return it."
        ),
        "place_position": None,
        "disable_conveyor_return": False,
    },
    {
        "name": "basket_drop",
        "instruction": (
            "Pick up the red cube and throw it into the far basket target while "
            "avoiding obstacles."
        ),
        # The robot releases before the basket. The cube inherits velocity
        # from a fast gripper swing instead of receiving a fixed release
        # velocity.
        "place_position": [0.18, -0.48, 0.24],
        "basket": {
            "center": [0.52, -0.58, 0.08],
            "inner_size": [0.20, 0.20],
            "wall_height": 0.08,
        },
        "basket_velocity_mode": "gripper",
        "disable_conveyor_return": True,
    },
    {
        "name": "place_on_platform",
        "instruction": (
            "Pick up the red cube and place it on the low platform target "
            "while avoiding obstacles."
        ),
        "place_position": [0.16, -0.30, 0.125],
        "disable_conveyor_return": True,
    },
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--isaac-python", type=Path, default=DEFAULT_ISAAC_PYTHON)
    parser.add_argument("--seed-policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--rlds-root", type=Path, default=DEFAULT_RLDS_ROOT)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "outputs/smolvla_multitask_dagger",
    )
    parser.add_argument("--iteration-name", default=None)
    parser.add_argument("--episodes-per-task", type=int, default=1)
    parser.add_argument("--cycles-per-task", type=int, default=1)
    parser.add_argument("--conveyor-speed", type=float, default=0.25)
    parser.add_argument("--max-samples", type=int, default=2048)
    parser.add_argument("--max-episodes", type=int, default=240)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--task", action="append", default=[])
    parser.add_argument("--skip-collect", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def selected_tasks(names):
    if not names:
        return TASKS
    wanted = set(names)
    tasks = [task for task in TASKS if task["name"] in wanted]
    missing = sorted(wanted - {task["name"] for task in tasks})
    if missing:
        raise ValueError(f"Unknown task(s): {missing}")
    return tasks


def run_command(command, cwd, dry_run=False):
    text = subprocess.list2cmdline([str(item) for item in command])
    print(f"RUN {text}", flush=True)
    if dry_run:
        return {"command": text, "returncode": None, "dry_run": True}
    env = os.environ.copy()
    env["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    completed = subprocess.run([str(item) for item in command], cwd=str(cwd), env=env)
    return {"command": text, "returncode": completed.returncode}


def write_task_metadata(path, task, base_metadata_path):
    metadata = json.loads(base_metadata_path.read_text(encoding="utf-8"))
    metadata["task_name"] = task["name"]
    metadata["task_instruction"] = task["instruction"]
    if task["place_position"] is not None:
        metadata["place_position"] = task["place_position"]
    if task.get("basket") is not None:
        metadata["basket"] = task["basket"]
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def plan_command(args, task, plan_path):
    command = [
        args.isaac_python.resolve(),
        ROOT / "scripts/curobo_dynamic_pick_planner.py",
        "--output",
        plan_path,
    ]
    if task["place_position"] is not None:
        command.append(
            "--place-position="
            + ",".join(str(value) for value in task["place_position"])
        )
    return command


def collect_command(args, task, plan_path, metadata_path, rollout_root, report_path):
    command = [
        args.isaac_python.resolve(),
        ROOT / "scripts/play_curobo_dynamic_pick.py",
        "--headless",
        "--use-static-plan",
        "--plan",
        plan_path,
        "--metadata",
        metadata_path,
        "--episodes",
        str(max(1, args.episodes_per_task)),
        "--cycles",
        str(max(1, args.cycles_per_task)),
        "--conveyor-speed",
        str(args.conveyor_speed),
        "--brain-control",
        "filtered",
        "--brain-policy",
        args.seed_policy.resolve(),
        "--brain-local-files-only",
        "--brain-blend",
        "0.70",
        "--brain-max-teacher-delta",
        "0.35",
        "--brain-max-step-delta",
        "0.03",
        "--brain-terminal-servo",
        "--brain-terminal-servo-step",
        "0.055",
        "--brain-terminal-servo-max-joint-delta",
        "0.08",
        "--brain-terminal-servo-align-frames",
        "900",
        "--brain-terminal-servo-phases",
        "close_gripper",
        "--brain-place-servo",
        "--brain-place-servo-frames",
        "1200",
        "--brain-place-servo-hover-height",
        "0.18",
        "--grasp-mode",
        "relative",
        "--grasp-attach-distance",
        "0.025",
        "--grasp-outward-offset",
        "0.015",
        "--link-clearance-action",
        "warn",
        "--task-name",
        task["name"],
        "--task-instruction",
        task["instruction"],
        "--record-root",
        rollout_root,
        "--keep-failed-episodes",
        "--brain-run-report",
        report_path,
    ]
    if task["disable_conveyor_return"]:
        command.append("--disable-conveyor-return")
    if task.get("basket_velocity_mode") is not None:
        command.extend(
            [
                "--basket-velocity-mode",
                task["basket_velocity_mode"],
                "--basket-release-policy",
                ROOT / "outputs/smolvla_multitask_dagger/ballistic_throw_policy.pt",
            ]
        )
    return command


def train_command(args, output_dir, dagger_roots):
    command = [
        args.isaac_python.resolve(),
        ROOT / "scripts/train_smolvla_policy.py",
        "--rlds-root",
        args.rlds_root.resolve(),
        "--base-checkpoint",
        args.seed_policy.resolve(),
        "--output-dir",
        output_dir,
        "--max-episodes",
        str(args.max_episodes),
        "--max-samples",
        str(args.max_samples),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--learning-rate",
        str(args.learning_rate),
        "--action-representation",
        "delta",
        "--include-success-dagger",
        "--local-files-only",
    ]
    for root in dagger_roots:
        command.extend(["--dagger-root", root])
    return command


def main():
    args = parse_args()
    tasks = selected_tasks(args.task)
    if not args.seed_policy.exists():
        raise FileNotFoundError(args.seed_policy)
    if not args.rlds_root.exists():
        raise FileNotFoundError(args.rlds_root)

    iteration_name = args.iteration_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = (args.output_root / iteration_name).resolve()
    if not args.dry_run:
        run_root.mkdir(parents=True, exist_ok=True)

    commands = {}
    dagger_roots = []
    task_summaries = []
    for task in tasks:
        task_root = run_root / task["name"]
        plan_path = task_root / "plan.npz"
        metadata_path = task_root / "plan.json"
        rollout_root = task_root / "rollouts"
        report_path = rollout_root / "brain_run.json"
        if not args.dry_run:
            task_root.mkdir(parents=True, exist_ok=True)
            rollout_root.mkdir(parents=True, exist_ok=True)

        if not args.skip_collect:
            commands[f"{task['name']}_plan"] = run_command(
                plan_command(args, task, plan_path),
                ROOT,
                args.dry_run,
            )
            if not args.dry_run:
                write_task_metadata(metadata_path, task, plan_path.with_suffix(".json"))
            commands[f"{task['name']}_collect"] = run_command(
                collect_command(
                    args,
                    task,
                    plan_path,
                    metadata_path,
                    rollout_root,
                    report_path,
                ),
                ROOT,
                args.dry_run,
            )
        dagger_roots.append(rollout_root)
        task_summaries.append(
            {
                "task_name": task["name"],
                "instruction": task["instruction"],
                "place_position": task["place_position"],
                "rollout_root": str(rollout_root),
                "report": str(report_path),
            }
        )

    policy_dir = run_root / "policy"
    if not args.skip_train:
        commands["train_smolvla_multitask"] = run_command(
            train_command(args, policy_dir, dagger_roots),
            ROOT,
            args.dry_run,
        )

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "run_root": str(run_root),
        "seed_policy": str(args.seed_policy.resolve()),
        "tasks": task_summaries,
        "policy": str((policy_dir / "best_smolvla_policy.pt").resolve()),
        "commands": commands,
    }
    if not args.dry_run:
        (run_root / "summary.json").write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
