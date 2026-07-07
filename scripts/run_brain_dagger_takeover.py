"""Run staged DAgger iterations toward brain-only xArm control.

This is the replacement DAgger entrypoint.  cuRobo and the terminal/place
servo are treated as teachers during collection; the gate at the end uses
direct brain control plus the runtime clearance monitor as the safety brake.
"""

import argparse
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ISAAC_PYTHON = Path.home() / "isaac_sim_5.1" / "python.bat"
DEFAULT_RLDS_ROOT = ROOT / "outputs/rlds_xarm6_curobo_500_v2"
DEFAULT_SEED_POLICY = ROOT / "outputs/stage3_expert_match_500_001/best_stage3_policy.pt"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs/brain_dagger_takeover"
OLD_DAGGER_ROOT = ROOT / "outputs/dagger_autonomy_iterations"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--isaac-python", type=Path, default=DEFAULT_ISAAC_PYTHON)
    parser.add_argument("--seed-policy", type=Path, default=DEFAULT_SEED_POLICY)
    parser.add_argument("--rlds-root", type=Path, default=DEFAULT_RLDS_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--reset-output", action="store_true")
    parser.add_argument(
        "--delete-old-dagger",
        action="store_true",
        help="Delete outputs/dagger_autonomy_iterations before running.",
    )
    parser.add_argument("--iteration-name", default=None)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--collect-cycles", type=int, default=4)
    parser.add_argument("--gate-cycles", type=int, default=2)
    parser.add_argument("--conveyor-speed", type=float, default=0.25)
    parser.add_argument("--max-step-delta", type=float, default=0.03)
    parser.add_argument("--max-rlds-episodes", type=int, default=200)
    parser.add_argument("--max-samples", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--dagger-repeat", type=int, default=2)
    parser.add_argument("--action-loss-weight", type=float, default=3.0)
    parser.add_argument("--dagger-loss-weight", type=float, default=2.0)
    parser.add_argument("--close-contact-loss-weight", type=float, default=2.5)
    parser.add_argument("--phase-loss-weight", type=float, default=0.2)
    parser.add_argument("--show-isaac", action="store_true")
    parser.add_argument(
        "--dagger-root",
        type=Path,
        action="append",
        default=[],
        help="Existing visible rollout roots to include as DAgger correction data.",
    )
    parser.add_argument("--skip-collect", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-gate", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-on-gate", action="store_true")
    return parser.parse_args()


def timestamp_name():
    return datetime.now().strftime("takeover_%Y%m%d_%H%M%S")


def require_inside_root(path):
    resolved = path.resolve()
    root = ROOT.resolve()
    if resolved == root or root not in resolved.parents:
        raise RuntimeError(f"Refusing to delete outside workspace: {resolved}")
    return resolved


def reset_dir(path, dry_run=False):
    resolved = require_inside_root(path)
    if not resolved.exists():
        return {"path": str(resolved), "deleted": False, "reason": "missing"}
    print(f"DELETE {resolved}", flush=True)
    if dry_run:
        return {"path": str(resolved), "deleted": False, "dry_run": True}
    shutil.rmtree(resolved)
    return {"path": str(resolved), "deleted": True}


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


def read_jsonl(path):
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def summarize_episode_root(root):
    episodes = []
    for metadata_file in sorted(root.glob("episode_*/metadata.json")):
        metadata = read_json(metadata_file)
        if metadata is None:
            continue
        metrics = metadata.get("metrics", {})
        runtime_stats = metrics.get("brain_runtime", {}).get("stats", {})
        violations = metrics.get("link_clearance_violations") or []
        episodes.append(
            {
                "episode": metadata_file.parent.name,
                "success": bool(metadata.get("success", True)),
                "frames_written": metadata.get("frames_written"),
                "lifted_without_teleport": metrics.get("lifted_without_teleport"),
                "teleport_shortcut_used": metrics.get("teleport_shortcut_used"),
                "returned_to_end": metrics.get("returned_to_end"),
                "grasp_attach_distance_m": metrics.get("grasp_attach_distance_m"),
                "minimum_robot_obstacle_clearance_m": metrics.get(
                    "minimum_robot_obstacle_clearance_m"
                ),
                "minimum_robot_cube_clearance_m": metrics.get(
                    "minimum_robot_cube_clearance_m"
                ),
                "link_clearance_violation_count": len(violations),
                "brain_prediction_count": int(runtime_stats.get("predictions", 0) or 0),
                "brain_direct_steps": int(runtime_stats.get("direct_steps", 0) or 0),
                "brain_filtered_steps": int(runtime_stats.get("filtered_steps", 0) or 0),
                "payload_failure": metrics.get("payload_failure"),
            }
        )
    return episodes


def summarize_run(root):
    brain_report = read_json(root / "brain_run.json") or {}
    episodes = summarize_episode_root(root)
    failures = brain_report.get("failures", [])
    prediction_counts = [
        int(brain_report.get("brain_runtime", {}).get("stats", {}).get("predictions", 0) or 0)
    ]
    prediction_counts.extend(item["brain_prediction_count"] for item in episodes)
    prediction_counts.extend(
        int(failure.get("brain_runtime", {}).get("stats", {}).get("predictions", 0) or 0)
        for failure in failures
    )
    successful = [
        item
        for item in episodes
        if item["success"]
        and item.get("lifted_without_teleport")
        and not item.get("teleport_shortcut_used")
        and item.get("returned_to_end")
        and item.get("link_clearance_violation_count") == 0
    ]
    return {
        "root": str(root.resolve()),
        "brain_report": str((root / "brain_run.json").resolve())
        if (root / "brain_run.json").exists()
        else None,
        "episodes": episodes,
        "failures": failures,
        "episode_count": len(episodes),
        "success_count": len(successful),
        "failure_count": len([item for item in episodes if not item["success"]])
        + len(failures),
        "brain_prediction_count": max(prediction_counts or [0]),
        "passed": len(successful) > 0,
    }


def require_nonempty_rollout(summary, label):
    if summary["episode_count"] <= 0:
        raise RuntimeError(f"{label} wrote no DAgger episodes")
    if summary["brain_prediction_count"] <= 0:
        raise RuntimeError(f"{label} recorded no brain predictions")


def isaac_command(args, cycles, record_root, report_path, policy_path, mode, teacher_level):
    command = [
        args.isaac_python.resolve(),
        ROOT / "scripts/play_curobo_dynamic_pick.py",
        "--use-static-plan",
        "--conveyor-speed",
        str(args.conveyor_speed),
        "--brain-control",
        mode,
        "--brain-policy",
        policy_path.resolve(),
        "--brain-local-files-only",
        "--brain-max-step-delta",
        str(args.max_step_delta),
        "--grasp-mode",
        "relative",
        "--grasp-attach-distance",
        "0.025",
        "--grasp-outward-offset",
        "0.015",
        "--record-root",
        record_root,
        "--keep-failed-episodes",
        "--brain-run-report",
        report_path,
    ]
    if teacher_level != "brain_only":
        command.extend(
            [
                "--brain-terminal-servo",
                "--brain-terminal-servo-step",
                "0.055",
                "--brain-terminal-servo-max-joint-delta",
                "0.08",
                "--brain-terminal-servo-align-frames",
                "900",
                "--brain-terminal-servo-phases",
                "close_gripper",
                "--brain-place-servo-frames",
                "1200",
                "--brain-place-servo-hover-height",
                "0.18",
            ]
        )
    if teacher_level == "reduced":
        command.extend(["--brain-blend", "0.70", "--brain-max-teacher-delta", "0.35"])
    elif teacher_level == "brain_only":
        command.extend(
            [
                "--disable-vertical-grasp-servo",
                "--brain-strict-direct",
                "--brain-phase-hold-frames",
                "1200",
            ]
        )
    if args.show_isaac:
        command[2:2] = [
            "--episodes",
            str(max(1, cycles)),
            "--cycles",
            str(max(1, cycles)),
            "--stop-after-cycles",
        ]
    else:
        command[2:2] = [
            "--headless",
            "--episodes",
            str(max(1, cycles)),
            "--cycles",
            str(max(1, cycles)),
        ]
    return command


def train_command(args, train_root, current_policy, dagger_roots):
    command = [
        args.isaac_python.resolve(),
        ROOT / "scripts/train_stage3_direct_correction.py",
        "--rlds-root",
        args.rlds_root.resolve(),
        "--base-checkpoint",
        current_policy.resolve(),
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
        "--dagger-loss-weight",
        str(args.dagger_loss_weight),
        "--close-contact-loss-weight",
        str(args.close_contact_loss_weight),
        "--phase-loss-weight",
        str(args.phase_loss_weight),
        "--dagger-repeat",
        str(max(1, args.dagger_repeat)),
        "--local-files-only",
    ]
    for dagger_root in dagger_roots:
        command.extend(["--dagger-root", dagger_root])
    return command


def main():
    args = parse_args()
    if not args.seed_policy.exists():
        raise FileNotFoundError(args.seed_policy)
    if not args.rlds_root.exists():
        raise FileNotFoundError(args.rlds_root)

    iteration_name = args.iteration_name or timestamp_name()
    output_root = args.output_root.resolve()
    iteration_root = output_root / iteration_name

    resets = []
    if args.delete_old_dagger:
        resets.append(reset_dir(OLD_DAGGER_ROOT, args.dry_run))
    if args.reset_output:
        resets.append(reset_dir(output_root, args.dry_run))
    if not args.dry_run:
        iteration_root.mkdir(parents=True, exist_ok=True)

    rounds = [
        {
            "name": "round1_terminal_teacher",
            "mode": "direct",
            "teacher_level": "terminal_only",
            "purpose": "brain handles global motion; terminal teacher corrects the final 5-10 cm.",
        },
        {
            "name": "round2_reduced_teacher",
            "mode": "filtered",
            "teacher_level": "reduced",
            "purpose": "increase brain authority while keeping teacher rejection and correction labels.",
        },
        {
            "name": "round3_brain_only_gate",
            "mode": "direct",
            "teacher_level": "brain_only",
            "purpose": "direct multimodal brain control; clearance monitor remains the safety brake.",
        },
    ][: max(1, min(int(args.rounds), 3))]

    current_policy = args.seed_policy.resolve()
    dagger_roots = [path.resolve() for path in args.dagger_root]
    commands = {}
    summaries = []

    for index, round_cfg in enumerate(rounds, start=1):
        round_root = iteration_root / f"{index:02d}_{round_cfg['name']}"
        collect_root = round_root / "rollouts"
        policy_root = round_root / "policy"
        report_path = collect_root / "brain_run.json"
        if not args.dry_run:
            collect_root.mkdir(parents=True, exist_ok=True)

        if not args.skip_collect:
            collect_command = isaac_command(
                args,
                args.collect_cycles if index < len(rounds) else args.gate_cycles,
                collect_root,
                report_path,
                current_policy,
                round_cfg["mode"],
                round_cfg["teacher_level"],
            )
            commands[f"{round_cfg['name']}_collect"] = run_command(
                collect_command,
                ROOT,
                args.dry_run,
            )
            dagger_roots.append(collect_root)
            if not args.dry_run:
                collect_summary = summarize_run(collect_root)
                require_nonempty_rollout(collect_summary, round_cfg["name"])

        if index < len(rounds) and not args.skip_train:
            train = train_command(args, policy_root, current_policy, dagger_roots)
            commands[f"{round_cfg['name']}_train"] = run_command(
                train,
                ROOT,
                args.dry_run,
            )
            candidate = policy_root / "best_stage3_policy.pt"
            current_policy = candidate if candidate.exists() or args.dry_run else policy_root / "latest_stage3_policy.pt"

        if not args.dry_run:
            summaries.append(
                {
                    **round_cfg,
                    "policy_after_round": str(current_policy.resolve()),
                    "rollout_summary": (
                        summarize_run(collect_root)
                        if collect_root.exists()
                        else None
                    ),
                    "training_metrics": read_jsonl(policy_root / "metrics.jsonl"),
                }
            )

    final_gate = None
    if not args.skip_gate:
        gate_root = iteration_root / "final_direct_gate"
        gate_report = gate_root / "brain_run.json"
        if not args.dry_run:
            gate_root.mkdir(parents=True, exist_ok=True)
        gate_command = isaac_command(
            args,
            args.gate_cycles,
            gate_root,
            gate_report,
            current_policy,
            "direct",
            "brain_only",
        )
        commands["final_direct_gate"] = run_command(gate_command, ROOT, args.dry_run)
        if not args.dry_run:
            final_gate = summarize_run(gate_root)
            require_nonempty_rollout(final_gate, "final_direct_gate")

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "iteration": iteration_name,
        "objective": "DAgger toward multimodal brain-only xArm conveyor control.",
        "seed_policy": str(args.seed_policy.resolve()),
        "final_policy": str(current_policy.resolve()),
        "resets": resets,
        "rounds": summaries,
        "final_gate": final_gate,
        "commands": commands,
        "view_command": subprocess.list2cmdline(
            [
                str(args.isaac_python.resolve()),
                str(ROOT / "scripts/play_curobo_dynamic_pick.py"),
                "--use-static-plan",
                "--cycles",
                str(args.gate_cycles),
                "--stop-after-cycles",
                "--conveyor-speed",
                str(args.conveyor_speed),
                "--brain-control",
                "direct",
                "--brain-policy",
                str(current_policy.resolve()),
                "--brain-local-files-only",
                "--brain-max-step-delta",
                str(args.max_step_delta),
                "--show-brain-ee-path",
                "--disable-vertical-grasp-servo",
                "--grasp-mode",
                "relative",
                "--grasp-attach-distance",
                "0.025",
                "--grasp-outward-offset",
                "0.015",
                "--brain-run-report",
                str((iteration_root / "visible_brain_only_run.json").resolve()),
            ]
        ),
    }
    summary["passed"] = bool(final_gate and final_gate.get("passed"))
    if not args.dry_run:
        summary_path = iteration_root / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if args.fail_on_gate and not args.dry_run and not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
