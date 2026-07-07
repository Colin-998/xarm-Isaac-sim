"""Run the no-teleport V-JEPA2 brain control gate in Isaac Sim."""

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ISAAC_PYTHON = Path.home() / "isaac_sim_5.1" / "python.bat"
DEFAULT_POLICY = ROOT / "outputs/stage3_video_sft_500ep_w4/latest_stage3_policy.pt"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--isaac-python",
        type=Path,
        default=DEFAULT_ISAAC_PYTHON,
    )
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument(
        "--mode",
        choices=["all", "direct", "filtered"],
        default="all",
    )
    parser.add_argument("--cycles", type=int, default=1)
    parser.add_argument("--max-step-delta", type=float, default=0.03)
    parser.add_argument("--conveyor-speed", type=float, default=0.25)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/no_teleport_brain_gate",
    )
    return parser.parse_args()


def run_mode(args, mode):
    report_path = args.output_dir / f"{mode}_run.json"
    command = [
        str(args.isaac_python.resolve()),
        str(ROOT / "scripts/play_curobo_dynamic_pick.py"),
        "--headless",
        "--cycles",
        str(args.cycles),
        "--conveyor-speed",
        str(args.conveyor_speed),
        "--brain-control",
        mode,
        "--brain-policy",
        str(args.policy.resolve()),
        "--brain-local-files-only",
        "--brain-max-step-delta",
        str(args.max_step_delta),
        "--grasp-mode",
        "relative",
        "--brain-run-report",
        str(report_path),
    ]
    env = os.environ.copy()
    env["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    print("Running:", subprocess.list2cmdline(command), flush=True)
    completed = subprocess.run(
        command,
        cwd=str(ROOT),
        env=env,
        text=True,
    )
    if not report_path.exists():
        raise FileNotFoundError(f"Missing run report: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    cycles = report.get("cycles", [])
    failures = report.get("failures", [])
    successful_cycles = [
        cycle
        for cycle in cycles
        if cycle.get("metrics", {}).get("lifted_without_teleport")
        and not cycle.get("metrics", {}).get("teleport_shortcut_used")
        and cycle.get("metrics", {}).get("returned_to_end")
    ]
    return {
        "mode": mode,
        "returncode": completed.returncode,
        "report": str(report_path.resolve()),
        "cycles": len(cycles),
        "successful_no_teleport_cycles": len(successful_cycles),
        "failures": failures,
        "passed": len(successful_cycles) >= max(args.cycles, 1),
    }


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    modes = ["direct", "filtered"] if args.mode == "all" else [args.mode]
    results = [run_mode(args, mode) for mode in modes]
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "gate": "no_teleport_brain_control",
        "policy": str(args.policy.resolve()),
        "cycles_requested": args.cycles,
        "grasp_mode": "relative",
        "teleport_shortcut_allowed": False,
        "results": results,
        "passed": all(result["passed"] for result in results),
        "interpretation": (
            "direct must pass before claiming full autonomous target control; "
            "filtered is the current safety-backed no-teleport baseline."
        ),
    }
    output = args.output_dir / "summary.json"
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
