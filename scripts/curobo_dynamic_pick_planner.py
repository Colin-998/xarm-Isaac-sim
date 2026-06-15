"""Plan a dynamic xArm6 conveyor pick-and-place sequence with cuRobo V2."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from curobo_bootstrap import ROOT, configure_curobo_imports


configure_curobo_imports()

from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.scene import Scene
from curobo.types import GoalToolPose, JointState


ROBOT_CONFIG = ROOT / "config" / "xarm6_curobo.yml"
BELT_RADIUS = 0.43
BELT_HEIGHT = 0.045
BELT_START_ANGLE = math.radians(-105.0)
DEFAULT_CUBE_SIZE = 0.04
DOWNWARD_QUATERNION = [0.0, 1.0, 0.0, 0.0]


def arc_position(angle, z):
    return [
        BELT_RADIUS * math.cos(angle),
        BELT_RADIUS * math.sin(angle),
        z,
    ]


def parse_vector(text, expected_length):
    values = [float(value.strip()) for value in text.split(",")]
    if len(values) != expected_length:
        raise argparse.ArgumentTypeError(
            f"Expected {expected_length} comma-separated values, got {text!r}"
        )
    return values


def load_obstacles(path):
    if path is None:
        return [
            {
                "name": "obstacle_0",
                "position": [0.24, -0.06, 0.19],
                "dims": [0.10, 0.12, 0.34],
            },
            {
                "name": "obstacle_1",
                "position": [0.28, 0.11, 0.13],
                "dims": [0.08, 0.10, 0.24],
            },
        ]

    data = json.loads(path.read_text(encoding="utf-8"))
    obstacles = data.get("obstacles", data)
    if not isinstance(obstacles, list):
        raise ValueError("Obstacle JSON must be a list or contain an 'obstacles' list")
    return obstacles


def scene_from_obstacles(obstacles, payload_padding=0.0):
    cuboids = {}
    for index, obstacle in enumerate(obstacles):
        name = obstacle.get("name", f"obstacle_{index}")
        position = obstacle["position"]
        dims = obstacle.get("dims", obstacle.get("scale"))
        if dims is None:
            raise ValueError(f"Obstacle {name!r} has no dims or scale")
        expanded = [float(value) + 2.0 * payload_padding for value in dims]
        cuboids[name] = {
            "dims": expanded,
            "pose": [
                float(position[0]),
                float(position[1]),
                float(position[2]),
                1.0,
                0.0,
                0.0,
                0.0,
            ],
        }
    return {"cuboid": cuboids}


def goal_pose(position):
    return GoalToolPose(
        tool_frames=["link_tcp"],
        position=torch.tensor(
            [[[[position]]]],
            device="cuda",
            dtype=torch.float32,
        ),
        quaternion=torch.tensor(
            [[[[DOWNWARD_QUATERNION]]]],
            device="cuda",
            dtype=torch.float32,
        ),
    )


def final_arm_state(plan, joint_names):
    ordered = plan.reorder(joint_names)
    positions = ordered.position
    while positions.ndim > 3:
        positions = positions.squeeze(1)
    return JointState(
        position=positions[:, -1, :],
        joint_names=joint_names,
    )


def plan_phase(planner, phase_name, target, current_state, gripper, output):
    result = planner.plan_pose(goal_pose(target), current_state, max_attempts=8)
    if result is None or not bool(result.success.any()):
        raise RuntimeError(
            f"cuRobo failed during phase {phase_name!r} for target {target}"
        )
    plan = result.get_interpolated_plan()
    ordered = plan.reorder(planner.joint_names)
    positions = ordered.position
    while positions.ndim > 3:
        positions = positions.squeeze(1)
    output[phase_name] = positions.squeeze(0).detach().cpu().numpy()
    print(
        f"{phase_name}: waypoints={output[phase_name].shape[0]} "
        f"target={[round(value, 4) for value in target]}"
    )
    return final_arm_state(plan, planner.joint_names), {
        "name": phase_name,
        "target_tcp": target,
        "gripper": gripper,
        "waypoints": int(output[phase_name].shape[0]),
    }


parser = argparse.ArgumentParser()
parser.add_argument(
    "--cube-position",
    type=lambda value: parse_vector(value, 3),
    default=None,
    help="Detected cube center as x,y,z in metres.",
)
parser.add_argument(
    "--obstacles-json",
    type=Path,
    help="JSON list with obstacle name, position and dims/scale.",
)
parser.add_argument("--cube-size", type=float, default=DEFAULT_CUBE_SIZE)
parser.add_argument("--clearance", type=float, default=0.012)
parser.add_argument(
    "--output",
    type=Path,
    default=ROOT / "outputs" / "curobo_dynamic_pick_plan.npz",
)
args = parser.parse_args()

cube_position = args.cube_position or arc_position(
    math.radians(105.0),
    BELT_HEIGHT + args.cube_size,
)
place_position = arc_position(
    BELT_START_ANGLE,
    BELT_HEIGHT + args.cube_size,
)
obstacles = load_obstacles(args.obstacles_json)
normal_scene = scene_from_obstacles(obstacles)
payload_padding = args.cube_size * 0.5 + args.clearance
carry_scene = scene_from_obstacles(obstacles, payload_padding=payload_padding)

planner_config = MotionPlannerCfg.create(
    robot=str(ROBOT_CONFIG),
    scene_model=normal_scene,
    collision_cache={"cuboid": max(4, len(obstacles) + 2)},
    self_collision_check=True,
    num_ik_seeds=32,
    num_trajopt_seeds=8,
    use_cuda_graph=False,
)
planner = MotionPlanner(planner_config)
planner.warmup(enable_graph=False, num_warmup_iterations=1)
current_state = JointState.from_position(
    planner.default_joint_state.position.unsqueeze(0),
    joint_names=planner.joint_names,
)

approach_height = max(cube_position[2] + 0.16, 0.28)
carry_height = max(cube_position[2] + 0.22, 0.34)
place_approach_height = max(place_position[2] + 0.20, 0.32)
phase_specs = [
    ("approach_cube", [cube_position[0], cube_position[1], approach_height], "open"),
    ("descend_to_cube", list(cube_position), "open"),
]

trajectories = {}
metadata_phases = []
for phase_name, target, gripper in phase_specs:
    current_state, phase_metadata = plan_phase(
        planner,
        phase_name,
        target,
        current_state,
        gripper,
        trajectories,
    )
    metadata_phases.append(phase_metadata)

# Conservatively account for the grasped cube by expanding every obstacle.
planner.update_world(Scene.create(carry_scene))
carry_specs = [
    ("lift_cube", [cube_position[0], cube_position[1], carry_height], "closed"),
    (
        "carry_to_start",
        [place_position[0], place_position[1], place_approach_height],
        "closed",
    ),
    ("place_cube", list(place_position), "closed"),
]
for phase_name, target, gripper in carry_specs:
    current_state, phase_metadata = plan_phase(
        planner,
        phase_name,
        target,
        current_state,
        gripper,
        trajectories,
    )
    metadata_phases.append(phase_metadata)

planner.update_world(Scene.create(normal_scene))
current_state, phase_metadata = plan_phase(
    planner,
    "retreat_after_release",
    [place_position[0], place_position[1], place_approach_height],
    current_state,
    "open",
    trajectories,
)
metadata_phases.append(phase_metadata)

args.output = args.output.resolve()
args.output.parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(args.output, **trajectories)
metadata = {
    "robot_config": str(ROBOT_CONFIG),
    "cube_position": cube_position,
    "cube_size": args.cube_size,
    "place_position": place_position,
    "payload_obstacle_padding": payload_padding,
    "obstacles": obstacles,
    "joint_names": planner.joint_names,
    "phases": metadata_phases,
    "trajectory_file": str(args.output),
}
metadata_path = args.output.with_suffix(".json")
metadata_path.write_text(
    json.dumps(metadata, indent=2),
    encoding="utf-8",
)
print(f"Saved trajectory: {args.output}")
print(f"Saved metadata: {metadata_path}")
