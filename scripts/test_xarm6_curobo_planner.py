"""Run a minimal cuRobo V2 motion-planning test for the local xArm6 model."""

import argparse
import torch

from curobo_bootstrap import ROOT, configure_curobo_imports


configure_curobo_imports()

from curobo.kinematics import Kinematics, KinematicsCfg
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.types import GoalToolPose, JointState


ROBOT_CONFIG = ROOT / "config" / "xarm6_curobo.yml"

parser = argparse.ArgumentParser()
parser.add_argument(
    "--disable-self-collision",
    action="store_true",
    help="Diagnostic mode that disables robot self-collision checking.",
)
parser.add_argument(
    "--with-obstacle",
    action="store_true",
    help="Place a cuboid across the direct TCP corridor.",
)
args = parser.parse_args()

kinematics = Kinematics(KinematicsCfg.from_robot_yaml_file(str(ROBOT_CONFIG)))
q_start = JointState.from_position(
    kinematics.default_joint_state.position.unsqueeze(0),
    joint_names=kinematics.joint_names,
)
fk_state = kinematics.compute_kinematics(q_start)
tcp_pose = fk_state.tool_poses.get_link_pose("link_tcp")

goal_position = tcp_pose.position.reshape(1, 1, 1, 1, 3).clone()
goal_position[..., 0] -= 0.14
goal_position[..., 1] += 0.20
goal_quaternion = tcp_pose.quaternion.reshape(1, 1, 1, 1, 4).clone()
goal = GoalToolPose(
    tool_frames=["link_tcp"],
    position=goal_position,
    quaternion=goal_quaternion,
)

scene_model = None
if args.with_obstacle:
    start = tcp_pose.position.reshape(3)
    target = goal_position.reshape(3)
    midpoint = 0.5 * (start + target)
    scene_model = {
        "cuboid": {
            "transfer_block": {
                "dims": [0.07, 0.10, 0.18],
                "pose": [
                    float(midpoint[0]),
                    float(midpoint[1]),
                    float(midpoint[2] - 0.09),
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                ],
            },
        }
    }

planner_config = MotionPlannerCfg.create(
    robot=str(ROBOT_CONFIG),
    scene_model=scene_model,
    collision_cache={"cuboid": 4},
    self_collision_check=not args.disable_self_collision,
    num_ik_seeds=24,
    num_trajopt_seeds=4,
    use_cuda_graph=False,
)
planner = MotionPlanner(planner_config)
planner.warmup(enable_graph=False, num_warmup_iterations=1)
ik_result = planner.ik_solver.solve_pose(
    goal,
    return_seeds=planner.trajopt_solver.config.num_seeds,
    current_state=q_start,
)
print(f"IK success: {ik_result.success.tolist()}")
print(f"IK feasible: {ik_result.feasible.tolist()}")
print(f"IK position error: {ik_result.position_error.tolist()}")
print(f"IK rotation error: {ik_result.rotation_error.tolist()}")
if ik_result.metrics is not None:
    print(f"IK metrics: {ik_result.metrics}")
result = planner.plan_pose(goal, q_start)

if result is None or not bool(result.success.any()):
    if result is not None:
        print(f"Trajectory success: {result.success.tolist()}")
        print(
            "Trajectory feasible: "
            f"{None if result.feasible is None else result.feasible.tolist()}"
        )
        print(f"Trajectory position error: {result.position_error.tolist()}")
        print(f"Trajectory rotation error: {result.rotation_error.tolist()}")
    raise RuntimeError("cuRobo failed to plan the xArm6 test motion")

plan = result.get_interpolated_plan()
ordered_plan = plan.reorder(kinematics.joint_names)
plan_fk = kinematics.compute_kinematics(
    JointState(
        position=ordered_plan.position.squeeze(1),
        joint_names=kinematics.joint_names,
    )
)
tcp_path = plan_fk.tool_poses.get_link_pose("link_tcp").position.reshape(-1, 3)
line_start = tcp_path[0]
line_vector = tcp_path[-1] - line_start
line_length_sq = torch.dot(line_vector, line_vector)
if float(line_length_sq) > 0.0:
    alpha = ((tcp_path - line_start) @ line_vector / line_length_sq).clamp(0.0, 1.0)
    line_points = line_start + alpha.unsqueeze(-1) * line_vector
    max_line_deviation = torch.linalg.vector_norm(
        tcp_path - line_points, dim=-1
    ).max()
else:
    max_line_deviation = torch.tensor(0.0, device=tcp_path.device)
print(f"Planning succeeded: waypoints={plan.position.shape[-2]}")
print(f"Joint names: {planner.joint_names}")
print(f"Start TCP: {tcp_pose.position.squeeze(0).tolist()}")
print(f"Goal TCP: {goal_position.reshape(3).tolist()}")
print(f"Scene obstacle enabled: {args.with_obstacle}")
print(f"Max TCP deviation from direct line: {float(max_line_deviation):.4f} m")
