"""Reusable cuRobo runtime for dynamic xArm6 conveyor pick planning."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from curobo_bootstrap import ROOT, configure_curobo_imports


configure_curobo_imports()

from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.scene import Scene
from curobo.types import GoalToolPose, JointState


ROBOT_CONFIG = ROOT / "config" / "xarm6_curobo.yml"
DOWNWARD_QUATERNION = [0.0, 1.0, 0.0, 0.0]


def scene_from_obstacles(obstacles, payload_padding=0.0):
    cuboids = {}
    for index, obstacle in enumerate(obstacles):
        name = obstacle.get("name", f"obstacle_{index}")
        dims = obstacle.get("dims", obstacle.get("scale"))
        if dims is None:
            raise ValueError(f"Obstacle {name!r} has no dims or scale")
        expanded = [float(value) + 2.0 * payload_padding for value in dims]
        position = obstacle["position"]
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


class DynamicPickPlanner:
    def __init__(self, obstacles, cube_size=0.04, clearance=0.012):
        self.obstacles = obstacles
        self.cube_size = float(cube_size)
        self.clearance = float(clearance)
        self.normal_scene = scene_from_obstacles(obstacles)
        self.carry_scene = scene_from_obstacles(
            obstacles,
            payload_padding=self.cube_size * 0.5 + self.clearance,
        )
        config = MotionPlannerCfg.create(
            robot=str(ROBOT_CONFIG),
            scene_model=self.normal_scene,
            collision_cache={"cuboid": max(4, len(obstacles) + 2)},
            self_collision_check=True,
            num_ik_seeds=32,
            num_trajopt_seeds=8,
            use_cuda_graph=False,
        )
        self.planner = MotionPlanner(config)
        self.planner.warmup(
            enable_graph=False,
            num_warmup_iterations=1,
        )

    def _plan_phase(self, name, target, current_state, trajectories):
        result = self.planner.plan_pose(
            goal_pose(target),
            current_state,
            max_attempts=8,
        )
        if result is None or not bool(result.success.any()):
            raise RuntimeError(
                f"cuRobo failed during phase {name!r} for target {target}"
            )
        plan = result.get_interpolated_plan()
        ordered = plan.reorder(self.planner.joint_names)
        positions = ordered.position
        while positions.ndim > 3:
            positions = positions.squeeze(1)
        trajectories[name] = positions.squeeze(0).detach().cpu().numpy()
        return final_arm_state(plan, self.planner.joint_names)

    def plan(self, cube_position, place_position, start_arm_positions):
        cube_position = np.asarray(cube_position, dtype=float)
        place_position = np.asarray(place_position, dtype=float)
        current_state = JointState.from_position(
            torch.tensor(
                [start_arm_positions],
                device="cuda",
                dtype=torch.float32,
            ),
            joint_names=self.planner.joint_names,
        )
        approach_height = max(cube_position[2] + 0.16, 0.28)
        carry_height = max(cube_position[2] + 0.22, 0.34)
        place_approach_height = max(place_position[2] + 0.20, 0.32)
        phases = [
            (
                "approach_cube",
                [cube_position[0], cube_position[1], approach_height],
            ),
            ("descend_to_cube", cube_position.tolist()),
        ]
        trajectories = {}
        self.planner.update_world(Scene.create(self.normal_scene))
        for name, target in phases:
            current_state = self._plan_phase(
                name,
                target,
                current_state,
                trajectories,
            )

        self.planner.update_world(Scene.create(self.carry_scene))
        carry_phases = [
            (
                "lift_cube",
                [cube_position[0], cube_position[1], carry_height],
            ),
            (
                "carry_to_start",
                [
                    place_position[0],
                    place_position[1],
                    place_approach_height,
                ],
            ),
            ("place_cube", place_position.tolist()),
        ]
        for name, target in carry_phases:
            current_state = self._plan_phase(
                name,
                target,
                current_state,
                trajectories,
            )

        self.planner.update_world(Scene.create(self.normal_scene))
        current_state = self._plan_phase(
            "retreat_after_release",
            [
                place_position[0],
                place_position[1],
                place_approach_height,
            ],
            current_state,
            trajectories,
        )
        return trajectories


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    request = json.loads(args.request.read_text(encoding="utf-8"))
    planner = DynamicPickPlanner(
        obstacles=request["obstacles"],
        cube_size=request["cube_size"],
        clearance=request.get("clearance", 0.012),
    )
    trajectories = planner.plan(
        cube_position=request["cube_position"],
        place_position=request["place_position"],
        start_arm_positions=request["start_arm_positions"],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **trajectories)
    print(
        "runtime_plan_complete "
        + " ".join(
            f"{name}={len(trajectory)}"
            for name, trajectory in trajectories.items()
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
