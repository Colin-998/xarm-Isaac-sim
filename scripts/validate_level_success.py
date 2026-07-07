"""Validate xArm conveyor run reports against the two-level demo criteria."""

import argparse
import json
from pathlib import Path


DEFAULT_CLEARANCE_THRESHOLD = 0.012
DEFAULT_CLEARANCE_WARNING_TOLERANCE = 0.001


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("--level2-cycles", type=int, default=3)
    parser.add_argument("--clearance-threshold", type=float, default=DEFAULT_CLEARANCE_THRESHOLD)
    parser.add_argument(
        "--clearance-warning-tolerance",
        type=float,
        default=DEFAULT_CLEARANCE_WARNING_TOLERANCE,
        help=(
            "Clearance misses within this many meters below the threshold are "
            "warnings, not major collisions."
        ),
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def min_clearance(metrics):
    values = [
        metrics.get("minimum_robot_obstacle_clearance_m"),
        metrics.get("minimum_robot_cube_clearance_m"),
    ]
    values.extend(
        item.get("clearance_m")
        for item in metrics.get("link_clearance_violations", [])
        if isinstance(item, dict)
    )
    finite = [float(value) for value in values if value is not None]
    return min(finite) if finite else None


def cycle_verdict(cycle, threshold, warning_tolerance):
    metrics = cycle.get("metrics", {})
    clearance = min_clearance(metrics)
    violations = metrics.get("link_clearance_violations") or []
    clearance_warning = False
    major_collision = False
    if clearance is not None and clearance < threshold:
        if clearance >= threshold - warning_tolerance:
            clearance_warning = True
        else:
            major_collision = True
    if any(
        isinstance(item, dict)
        and item.get("kind") == "cube"
        and float(item.get("clearance_m", 1.0)) < -0.003
        for item in violations
    ):
        major_collision = True

    checks = {
        "grasp_success": bool(metrics.get("lifted_without_teleport")),
        "place_success": bool(metrics.get("placed_at_start")),
        "cube_returned": bool(metrics.get("returned_to_end")),
        "no_teleport": not bool(metrics.get("teleport_shortcut_used")),
        "cube_not_knocked_away": (
            metrics.get("max_cube_height_m") is None
            or float(metrics.get("max_cube_height_m")) < 0.45
        ),
        "no_major_collision": not major_collision,
    }
    passed = all(checks.values())
    return {
        "cycle": cycle.get("cycle"),
        "passed": passed,
        "checks": checks,
        "minimum_clearance_m": clearance,
        "clearance_warning": clearance_warning,
        "major_collision": major_collision,
        "warnings": [
            (
                "link clearance slightly below threshold; treated as warning "
                "under the current demo criteria"
            )
        ]
        if clearance_warning
        else [],
        "metrics": {
            "grasp_attach_distance_m": metrics.get("grasp_attach_distance_m"),
            "start_place_distance_m": metrics.get("start_place_distance_m"),
            "final_end_distance_m": metrics.get("final_end_distance_m"),
            "max_cube_height_m": metrics.get("max_cube_height_m"),
            "brain_control": metrics.get("brain_control"),
        },
    }


def main():
    args = parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    cycles = report.get("cycles") or []
    cycle_results = [
        cycle_verdict(
            cycle,
            args.clearance_threshold,
            args.clearance_warning_tolerance,
        )
        for cycle in cycles
    ]
    passed_cycles = [item for item in cycle_results if item["passed"]]
    failures = report.get("failures") or []
    longest_consecutive = 0
    current_streak = 0
    for item in cycle_results:
        if item["passed"]:
            current_streak += 1
            longest_consecutive = max(longest_consecutive, current_streak)
        else:
            current_streak = 0
    major_failures = [
        item
        for item in failures
        if "slightly below threshold" not in str(item)
    ]
    result = {
        "report": str(args.report.resolve()),
        "cycles_observed": len(cycles),
        "failure_count": len(failures),
        "level1_passed": len(passed_cycles) >= 1,
        "level2_passed": (
            longest_consecutive >= max(3, int(args.level2_cycles))
            and not major_failures
        ),
        "passed_cycle_count": len(passed_cycles),
        "longest_consecutive_passed_cycles": longest_consecutive,
        "minimum_link_clearance_m": min(
            (
                item["minimum_clearance_m"]
                for item in cycle_results
                if item["minimum_clearance_m"] is not None
            ),
            default=None,
        ),
        "major_collision": any(item["major_collision"] for item in cycle_results),
        "clearance_warning_count": sum(
            1 for item in cycle_results if item["clearance_warning"]
        ),
        "cycle_results": cycle_results,
        "failures": failures,
    }
    text = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)
    if not result["level1_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
