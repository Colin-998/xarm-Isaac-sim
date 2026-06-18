"""Run one DAgger-style autonomy correction iteration for direct xArm control."""

import argparse
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ISAAC_PYTHON = Path.home() / "isaac_sim_5.1" / "python.bat"
DEFAULT_RLDS_ROOT = ROOT / "outputs/rlds_xarm6_curobo_500_v2"
DEFAULT_SEED_POLICY = (
    ROOT
    / "outputs/dagger_autonomy_iterations/dagger_batch_003_repeat/policy/latest_stage3_policy.pt"
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--isaac-python", type=Path, default=DEFAULT_ISAAC_PYTHON)
    parser.add_argument("--seed-policy", type=Path, default=DEFAULT_SEED_POLICY)
    parser.add_argument("--rlds-root", type=Path, default=DEFAULT_RLDS_ROOT)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "outputs/dagger_autonomy_iterations",
    )
    parser.add_argument("--iteration-name", default=None)
    parser.add_argument("--collect-cycles", type=int, default=8)
    parser.add_argument("--max-step-delta", type=float, default=0.03)
    parser.add_argument("--conveyor-speed", type=float, default=0.25)
    parser.add_argument(
        "--no-static-plan",
        action="store_true",
        help="Use live cuRobo planning instead of the saved static plan.",
    )
    parser.add_argument(
        "--grasp-attach-distance",
        type=float,
        default=0.025,
        help=(
            "Maximum TCP-to-cube distance allowed before a relative grasp is "
            "attached. The default is intentionally tight to avoid remote-looking grasps."
        ),
    )
    parser.add_argument(
        "--vision-grasp",
        action="store_true",
        help=(
            "Use V-JEPA/Llama predictions during the final cube approach. "
            "By default, DAgger runs use simulator cube pose plus IK for the "
            "terminal grasp so the pickup is stable and visibly in contact."
        ),
    )
    parser.add_argument(
        "--terminal-servo-step",
        type=float,
        default=0.055,
        help="Cartesian TCP step used by the no-teleport cube approach servo.",
    )
    parser.add_argument(
        "--terminal-servo-max-joint-delta",
        type=float,
        default=0.08,
        help="Per-frame joint delta limit for the terminal cube approach servo.",
    )
    parser.add_argument("--terminal-servo-align-frames", type=int, default=900)
    parser.add_argument(
        "--terminal-servo-phases",
        default="close_gripper",
        help=(
            "Comma-separated phases where the no-vision terminal servo is allowed. "
            "The default keeps the calibrated finger-pocket grasp aligned "
            "while the fingers close."
        ),
    )
    parser.add_argument("--place-servo-frames", type=int, default=1200)
    parser.add_argument("--place-servo-hover-height", type=float, default=0.18)
    parser.add_argument(
        "--allow-phase-mismatch",
        action="store_true",
        help="Allow direct actions even when the predicted phase disagrees.",
    )
    parser.add_argument("--max-rlds-episodes", type=int, default=200)
    parser.add_argument("--max-dagger-episodes", type=int, default=0)
    parser.add_argument("--dagger-repeat", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--action-loss-weight", type=float, default=3.0)
    parser.add_argument("--phase-loss-weight", type=float, default=0.2)
    parser.add_argument("--skip-collect", action="store_true")
    parser.add_argument(
        "--dagger-root",
        type=Path,
        action="append",
        default=[],
        help="Use existing failed rollout roots. Implies these roots are added to training.",
    )
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--trained-policy", type=Path)
    parser.add_argument("--skip-gate", action="store_true")
    parser.add_argument(
        "--show-isaac",
        action="store_true",
        help=(
            "Open the Isaac Sim window for collection and gate runs. This is "
            "slower, but avoids headless RGB recorder issues and lets the "
            "operator watch the robot."
        ),
    )
    parser.add_argument(
        "--fail-on-gate",
        action="store_true",
        help="Exit non-zero when the direct no-teleport gate does not pass.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def timestamp_name():
    return datetime.now().strftime("iter_%Y%m%d_%H%M%S")


def run_command(command, cwd, dry_run=False):
    command_text = subprocess.list2cmdline([str(item) for item in command])
    print(f"RUN {command_text}", flush=True)
    if dry_run:
        return {"command": command_text, "returncode": None, "dry_run": True}
    env = os.environ.copy()
    env["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    completed = subprocess.run(
        [str(item) for item in command],
        cwd=str(cwd),
        env=env,
        text=True,
    )
    return {"command": command_text, "returncode": completed.returncode}


def read_json(path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def extract_failure_distance(text):
    if not text:
        return None
    match = re.search(r"tcp_to_cube_distance_m=([0-9.]+)", text)
    if match:
        return float(match.group(1))
    return None


def summarize_episode_root(root):
    manifest = read_json(root / "dataset_manifest.json") or {}
    brain_report = read_json(root / "brain_run.json") or {}
    episodes = []
    for metadata_file in sorted(root.glob("episode_*/metadata.json")):
        metadata = read_json(metadata_file)
        if metadata is None:
            continue
        metrics = metadata.get("metrics", {})
        runtime_stats = (
            metrics.get("brain_runtime", {})
            .get("stats", {})
        )
        distance = metrics.get("grasp_attach_distance_m")
        if distance is None:
            distance = extract_failure_distance(metrics.get("payload_failure"))
        episodes.append(
            {
                "episode": metadata_file.parent.name,
                "success": bool(metadata.get("success", True)),
                "frames_written": metadata.get("frames_written"),
                "grasp_attach_distance_m": distance,
                "start_place_distance_m": metrics.get("start_place_distance_m"),
                "payload_failure": metrics.get("payload_failure"),
                "lifted_without_teleport": metrics.get("lifted_without_teleport"),
                "teleport_shortcut_used": metrics.get("teleport_shortcut_used"),
                "placed_at_start": metrics.get("placed_at_start"),
                "returned_to_end": metrics.get("returned_to_end"),
                "brain_prediction_count": int(
                    runtime_stats.get("predictions", 0) or 0
                ),
            }
        )
    failed_distances = [
        item["grasp_attach_distance_m"]
        for item in episodes
        if not item["success"] and item["grasp_attach_distance_m"] is not None
    ]
    brain_stats = (
        brain_report.get("brain_runtime", {})
        .get("stats", {})
    )
    prediction_count = max(
        [int(brain_stats.get("predictions", 0) or 0)]
        + [int(item.get("brain_prediction_count", 0) or 0) for item in episodes]
    )
    total_episodes = len(episodes)
    invalid_reasons = []
    if total_episodes == 0:
        invalid_reasons.append("no_recorded_episodes")
    if brain_report and prediction_count == 0:
        invalid_reasons.append("no_brain_predictions")
    return {
        "root": str(root.resolve()),
        "manifest": manifest,
        "brain_report": (
            str((root / "brain_run.json").resolve())
            if (root / "brain_run.json").exists()
            else None
        ),
        "brain_prediction_count": prediction_count,
        "episodes": episodes,
        "failed_episode_count": len([item for item in episodes if not item["success"]]),
        "successful_episode_count": len([item for item in episodes if item["success"]]),
        "total_episode_count": total_episodes,
        "valid_for_training": total_episodes > 0,
        "invalid_reasons": invalid_reasons,
        "best_failed_grasp_distance_m": (
            min(failed_distances) if failed_distances else None
        ),
        "best_place_distance_m": min(
            [
                item["start_place_distance_m"]
                for item in episodes
                if item["start_place_distance_m"] is not None
            ],
            default=None,
        ),
    }


def summarize_gate(root):
    brain_report = read_json(root / "brain_run.json") or {}
    episode_summary = summarize_episode_root(root)
    successful_no_teleport = [
        item
        for item in episode_summary["episodes"]
        if item["success"]
        and item.get("lifted_without_teleport")
        and not item.get("teleport_shortcut_used")
        and item.get("returned_to_end")
    ]
    failures = brain_report.get("failures", [])
    failure_distances = [
        failure.get("grasp_attach_distance_m")
        for failure in failures
        if failure.get("grasp_attach_distance_m") is not None
    ]
    brain_stats = (
        brain_report.get("brain_runtime", {})
        .get("stats", {})
    )
    failure_prediction_counts = [
        int(
            failure.get("brain_runtime", {})
            .get("stats", {})
            .get("predictions", 0)
            or 0
        )
        for failure in failures
    ]
    prediction_count = max(
        [int(brain_stats.get("predictions", 0) or 0)]
        + [int(item.get("brain_prediction_count", 0) or 0) for item in episode_summary["episodes"]]
        + failure_prediction_counts
    )
    invalid_reasons = list(episode_summary.get("invalid_reasons", []))
    if prediction_count == 0:
        invalid_reasons.append("gate_had_no_brain_predictions")
    return {
        **episode_summary,
        "brain_report": str((root / "brain_run.json").resolve()),
        "brain_failures": failures,
        "brain_prediction_count": prediction_count,
        "valid_for_gate": prediction_count > 0 and bool(
            episode_summary["episodes"] or failures
        ),
        "invalid_reasons": invalid_reasons,
        "passed_direct_no_teleport": bool(successful_no_teleport),
        "best_failure_grasp_distance_m": (
            min(failure_distances) if failure_distances else None
        ),
    }


def main():
    args = parse_args()
    iteration_name = args.iteration_name or timestamp_name()
    iteration_root = (args.output_root / iteration_name).resolve()
    collect_root = iteration_root / "failed_rollouts"
    train_root = iteration_root / "policy"
    gate_root = iteration_root / "direct_gate"
    iteration_root.mkdir(parents=True, exist_ok=True)

    commands = {}
    dagger_roots = [path.resolve() for path in args.dagger_root]

    def isaac_brain_command(cycles, record_root, report_path, policy_path):
        command = [
            args.isaac_python.resolve(),
            ROOT / "scripts/play_curobo_dynamic_pick.py",
            "--conveyor-speed",
            str(args.conveyor_speed),
            "--brain-control",
            "direct",
            "--brain-policy",
            policy_path.resolve(),
            "--brain-local-files-only",
            "--brain-max-step-delta",
            str(args.max_step_delta),
            "--brain-terminal-servo",
            "--brain-terminal-servo-step",
            str(args.terminal_servo_step),
            "--brain-terminal-servo-max-joint-delta",
            str(args.terminal_servo_max_joint_delta),
            "--brain-terminal-servo-align-frames",
            str(args.terminal_servo_align_frames),
            "--brain-terminal-servo-phases",
            args.terminal_servo_phases,
            "--brain-place-servo-frames",
            str(args.place_servo_frames),
            "--brain-place-servo-hover-height",
            str(args.place_servo_hover_height),
            "--grasp-mode",
            "relative",
            "--grasp-attach-distance",
            str(args.grasp_attach_distance),
            "--record-root",
            record_root,
            "--keep-failed-episodes",
            "--brain-run-report",
            report_path,
        ]
        if not args.vision_grasp:
            command.append("--brain-terminal-servo-without-vision")
        if args.show_isaac:
            command[2:2] = ["--episodes", str(max(1, cycles))]
        else:
            command[2:2] = ["--headless", "--episodes", str(max(1, cycles))]
        if not args.no_static_plan:
            command.insert(2, "--use-static-plan")
        if args.allow_phase_mismatch:
            command.append("--brain-allow-phase-mismatch")
        return command

    if not args.skip_collect:
        collect_report = collect_root / "brain_run.json"
        commands["collect"] = run_command(
            isaac_brain_command(
                args.collect_cycles,
                collect_root,
                collect_report,
                args.seed_policy,
            ),
            ROOT,
            args.dry_run,
        )
        dagger_roots.append(collect_root)

    trained_policy = (
        args.trained_policy.resolve()
        if args.trained_policy is not None
        else train_root / "latest_stage3_policy.pt"
    )
    if not args.skip_train:
        train_command = [
            args.isaac_python.resolve(),
            ROOT / "scripts/train_stage3_direct_correction.py",
            "--rlds-root",
            args.rlds_root.resolve(),
            "--base-checkpoint",
            args.seed_policy.resolve(),
            "--output-dir",
            train_root,
            "--state-conditioned",
            "--max-episodes",
            str(args.max_rlds_episodes),
            "--max-samples",
            str(args.max_samples),
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--learning-rate",
            str(args.learning_rate),
            "--action-loss-weight",
            str(args.action_loss_weight),
            "--phase-loss-weight",
            str(args.phase_loss_weight),
            "--local-files-only",
        ]
        if args.max_dagger_episodes:
            train_command.extend(["--max-dagger-episodes", str(args.max_dagger_episodes)])
        train_command.extend(["--dagger-repeat", str(max(1, args.dagger_repeat))])
        for dagger_root in dagger_roots:
            train_command.extend(["--dagger-root", dagger_root])
        commands["train"] = run_command(train_command, ROOT, args.dry_run)

    if not args.skip_gate:
        gate_report = gate_root / "brain_run.json"
        commands["gate"] = run_command(
            isaac_brain_command(1, gate_root, gate_report, trained_policy),
            ROOT,
            args.dry_run,
        )

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "iteration": iteration_name,
        "seed_policy": str(args.seed_policy.resolve()),
        "trained_policy": str(trained_policy.resolve()),
        "dagger_roots": [str(path.resolve()) for path in dagger_roots],
        "commands": commands,
        "collection": (
            summarize_episode_root(collect_root)
            if collect_root.exists() and not args.dry_run
            else None
        ),
        "training_metrics": read_jsonl(train_root / "metrics.jsonl")
        if train_root.exists() and not args.dry_run
        else [],
        "gate": (
            summarize_gate(gate_root)
            if gate_root.exists() and not args.dry_run
            else None
        ),
    }
    invalid_reasons = []
    if summary["collection"] is not None and not summary["collection"].get(
        "valid_for_training",
        False,
    ):
        invalid_reasons.extend(
            f"collection:{reason}"
            for reason in summary["collection"].get("invalid_reasons", [])
        )
    if summary["gate"] is not None and not summary["gate"].get(
        "valid_for_gate",
        False,
    ):
        invalid_reasons.extend(
            f"gate:{reason}"
            for reason in summary["gate"].get("invalid_reasons", [])
        )
    summary["invalid_reasons"] = invalid_reasons
    summary["passed"] = bool(
        summary["gate"]
        and summary["gate"].get("valid_for_gate")
        and summary["gate"]["passed_direct_no_teleport"]
    )
    summary_path = iteration_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if args.fail_on_gate and not args.dry_run and not summary["passed"]:
        raise SystemExit(1)


def read_jsonl(path):
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
    return rows


if __name__ == "__main__":
    main()
