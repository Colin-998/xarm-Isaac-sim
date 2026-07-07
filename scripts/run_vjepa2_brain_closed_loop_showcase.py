"""Prepare or launch a safety-filtered V-JEPA2 brain closed-loop showcase."""

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ISAAC_PYTHON = Path.home() / "isaac_sim_5.1" / "python.bat"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1-projector", type=Path, default=ROOT / "outputs/stage1_projector_llama31_smoke/latest_projector.pt")
    parser.add_argument("--stage2-vqa", type=Path, default=ROOT / "outputs/stage2_vqa_llama31_4096/latest_stage2_vqa.pt")
    parser.add_argument("--stage2-eval", type=Path, default=ROOT / "outputs/stage2_vqa_llama31_4096/eval_metrics_4096.json")
    parser.add_argument("--stage3-policy", type=Path, default=ROOT / "outputs/stage3_video_sft_500ep_w4/latest_stage3_policy.pt")
    parser.add_argument("--stage3-eval", type=Path, default=ROOT / "outputs/stage3_video_sft_500ep_w4/eval_metrics_500_w4.json")
    parser.add_argument("--cycles", type=int, default=1)
    parser.add_argument("--launch-isaac", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--brain-control",
        choices=["observe", "filtered", "direct"],
        default="filtered",
    )
    parser.add_argument("--brain-max-step-delta", type=float, default=0.03)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/vjepa2_brain_closed_loop_showcase.json")
    return parser.parse_args()


def require_file(path):
    if not path.exists():
        raise FileNotFoundError(path)
    return str(path.resolve())


def main():
    args = parse_args()
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "vjepa2_llama31_brain_online_closed_loop_showcase",
        "brain_stack": {
            "stage1_projector": require_file(args.stage1_projector),
            "stage2_vqa": require_file(args.stage2_vqa),
            "stage2_eval": require_file(args.stage2_eval),
            "stage3_policy": require_file(args.stage3_policy),
            "stage3_eval": require_file(args.stage3_eval),
        },
        "executor": {
            "isaac_script": str((ROOT / "scripts/play_curobo_dynamic_pick.py").resolve()),
            "safety_layer": (
                "filtered mode lets Stage-3 targets act only when they agree "
                "with the collision-aware cuRobo fallback; direct mode is an "
                "experimental raw policy stress test"
            ),
            "brain_control": args.brain_control,
            "grasp_mode": "relative",
            "teleport_shortcut": False,
            "cycles": args.cycles,
        },
        "closed_loop_claim": {
            "brain_outputs": "VQA answers, phase intent, and 14D robot action target",
            "control_authority": (
                "filtered mode shares target authority with the safety planner; "
                "direct mode sends rate-limited Stage-3 targets without teacher blending"
            ),
            "direct_motor_control": args.brain_control == "direct",
            "no_teleport_grasp": True,
        },
    }
    if args.stage2_eval.exists():
        report["stage2_metrics"] = json.loads(args.stage2_eval.read_text(encoding="utf-8"))
    if args.stage3_eval.exists():
        report["stage3_metrics"] = json.loads(args.stage3_eval.read_text(encoding="utf-8"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote showcase report to {args.output}")
    print(
        "Brain stack ready: V-JEPA2 + Llama 3.1 projector + Stage2 VQA + Stage3 video policy"
    )
    if not args.launch_isaac:
        print("Use --launch-isaac to open the Isaac Sim online brain control showcase.")
        return

    command = [
        str(ISAAC_PYTHON),
        str(ROOT / "scripts/play_curobo_dynamic_pick.py"),
        "--cycles",
        str(args.cycles),
        "--conveyor-speed",
        "0.25",
        "--brain-control",
        args.brain_control,
        "--brain-policy",
        str(args.stage3_policy),
        "--brain-local-files-only",
        "--brain-max-step-delta",
        str(args.brain_max_step_delta),
        "--grasp-mode",
        "relative",
    ]
    if args.headless:
        command.append("--headless")
    print("Launching Isaac Sim showcase:")
    print(subprocess.list2cmdline(command))
    subprocess.run(command, cwd=str(ROOT), check=True)


if __name__ == "__main__":
    main()
