import argparse
from pathlib import Path

from isaacsim import SimulationApp


parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true")
args, _ = parser.parse_known_args()

ROOT = Path(__file__).resolve().parents[1]
ROBOT_USD = ROOT / "assets" / "xarm6_gripper" / "xarm6_gripper.usd"

app = SimulationApp(
    {
        "headless": args.headless,
        "renderer": "RaytracedLighting",
        "width": 1280,
        "height": 720,
    }
)

import numpy as np
import omni.timeline
import omni.usd
from isaacsim.core.api import World
from isaacsim.core.api.materials import PhysicsMaterial
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.robot_motion.motion_generation import ArticulationKinematicsSolver, LulaKinematicsSolver
from pxr import Gf, Sdf, UsdGeom, UsdPhysics


CUBE_POSITION = np.array([0.35, 0.0, 0.03])
CUBE_SIZE = 0.055
DOWNWARD = np.array([0.0, 1.0, 0.0, 0.0])
GRIPPER_JOINTS = [
    "drive_joint",
    "left_inner_knuckle_joint",
    "right_inner_knuckle_joint",
    "right_outer_knuckle_joint",
    "left_finger_joint",
    "right_finger_joint",
]


def blend(start, end, alpha):
    alpha = min(max(alpha, 0.0), 1.0)
    alpha = alpha * alpha * (3.0 - 2.0 * alpha)
    return start + (end - start) * alpha


def set_full_target(robot, arm_positions, gripper_position, dof_index):
    target = robot.get_joint_positions().copy()
    for index, value in enumerate(arm_positions):
        target[dof_index[f"joint{index + 1}"]] = value
    for name in GRIPPER_JOINTS:
        target[dof_index[name]] = gripper_position
    robot.get_articulation_controller().apply_action(
        ArticulationAction(joint_positions=target)
    )
    return target


def solve_pose(solver, position):
    action, success = solver.compute_inverse_kinematics(
        target_position=np.asarray(position),
        target_orientation=DOWNWARD,
        position_tolerance=0.003,
        orientation_tolerance=0.02,
    )
    if not success:
        raise RuntimeError(f"IK failed for target {position}")
    return np.asarray(action.joint_positions)


def world_transform(stage, path):
    cache = UsdGeom.XformCache()
    return UsdGeom.Xformable(stage.GetPrimAtPath(path)).ComputeLocalToWorldTransform(cache.GetTime())


def lock_cube_after_real_alignment(stage):
    tcp_matrix = world_transform(stage, "/UF_ROBOT/root_joint/link_tcp")
    left_matrix = world_transform(stage, "/UF_ROBOT/root_joint/left_finger")
    right_matrix = world_transform(stage, "/UF_ROBOT/root_joint/right_finger")
    cube_matrix = world_transform(stage, "/World/GraspCube")
    cube_position = cube_matrix.ExtractTranslation()
    tcp_distance = (tcp_matrix.ExtractTranslation() - cube_position).GetLength()
    left_distance = (left_matrix.ExtractTranslation() - cube_position).GetLength()
    right_distance = (right_matrix.ExtractTranslation() - cube_position).GetLength()
    print(f"TCP-to-cube distance before lift: {tcp_distance:.4f} m", flush=True)
    print(
        f"Finger-link distances: left={left_distance:.4f} m right={right_distance:.4f} m",
        flush=True,
    )
    if tcp_distance > 0.065:
        raise RuntimeError("Grasp rejected: gripper never reached the cube")

    gripper_path = Sdf.Path("/UF_ROBOT/root_joint/xarm_gripper_base_link")
    cube_path = Sdf.Path("/World/GraspCube")
    gripper_matrix = world_transform(stage, str(gripper_path))
    # Gf matrices use row-vector convention, so the local transform is
    # world_transform * parent_world_inverse.
    relative = cube_matrix * gripper_matrix.GetInverse()
    relative_position = relative.ExtractTranslation()
    relative_rotation = relative.ExtractRotationQuat()

    joint = UsdPhysics.FixedJoint.Define(stage, Sdf.Path("/World/grasp_lock"))
    joint.CreateBody0Rel().SetTargets([gripper_path])
    joint.CreateBody1Rel().SetTargets([cube_path])
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(relative_position))
    joint.CreateLocalRot0Attr().Set(
        Gf.Quatf(
            float(relative_rotation.GetReal()),
            Gf.Vec3f(relative_rotation.GetImaginary()),
        )
    )
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0))
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0))
    print("Grasp lock enabled after alignment", flush=True)


omni.usd.get_context().open_stage(str(ROBOT_USD))
for _ in range(10):
    app.update()

world = World(stage_units_in_meters=1.0, physics_dt=1.0 / 120.0, rendering_dt=1.0 / 60.0)
world.scene.add_default_ground_plane(
    static_friction=1.2,
    dynamic_friction=1.0,
    restitution=0.0,
)

grip_material = PhysicsMaterial(
    prim_path="/World/HighFrictionMaterial",
    static_friction=1.8,
    dynamic_friction=1.5,
    restitution=0.0,
)

robot = world.scene.add(
    SingleArticulation(
        prim_path="/UF_ROBOT/root_joint/root_joint",
        name="xarm6",
    )
)
cube = world.scene.add(
    DynamicCuboid(
        prim_path="/World/GraspCube",
        name="grasp_cube",
        position=CUBE_POSITION,
        scale=np.full(3, CUBE_SIZE),
        color=np.array([0.85, 0.12, 0.08]),
        physics_material=grip_material,
        mass=0.05,
    )
)

world.reset()
for _ in range(10):
    world.step(render=not args.headless)

robot.set_solver_position_iteration_count(64)
robot.set_solver_velocity_iteration_count(16)

dof_names = list(robot.dof_names)
dof_index = {name: index for index, name in enumerate(dof_names)}
missing = [name for name in [f"joint{i}" for i in range(1, 7)] + GRIPPER_JOINTS if name not in dof_index]
if missing:
    raise RuntimeError(f"Missing expected joints: {missing}")

controller = robot.get_articulation_controller()
kps, kds = controller.get_gains()
max_efforts = controller.get_max_efforts()

for name in [f"joint{i}" for i in range(1, 7)]:
    index = dof_index[name]
    kps[index] = 8000.0
    kds[index] = 500.0
    max_efforts[index] = 150.0

for name in GRIPPER_JOINTS:
    index = dof_index[name]
    kps[index] = 12000.0
    kds[index] = 800.0
    max_efforts[index] = 120.0

controller.set_gains(kps=kps, kds=kds)
controller.set_max_efforts(max_efforts)

lula = LulaKinematicsSolver(
    robot_description_path=str(ROOT / "config/xarm6_robot_descriptor.yaml"),
    urdf_path=str(ROOT / "assets/xarm6_gripper/xarm6_gripper.urdf"),
)
ik_solver = ArticulationKinematicsSolver(robot, lula, "link_tcp")

approach = solve_pose(ik_solver, [CUBE_POSITION[0], CUBE_POSITION[1], 0.22])
grasp = solve_pose(ik_solver, [CUBE_POSITION[0], CUBE_POSITION[1], 0.07])
lift = solve_pose(ik_solver, [CUBE_POSITION[0], CUBE_POSITION[1], 0.30])

start = robot.get_joint_positions()[:6].copy()
set_full_target(robot, start, 0.85, dof_index)

if not args.headless:
    set_camera_view(
        eye=np.array([0.95, 0.95, 0.65]),
        target=np.array([0.25, 0.0, 0.18]),
        camera_prim_path="/OmniverseKit_Persp",
    )

timeline = omni.timeline.get_timeline_interface()
timeline.play()

stages = [
    (start, approach, 120, 0.85, "move above cube"),
    (approach, grasp, 120, 0.85, "descend"),
    (grasp, grasp, 110, 0.05, "close gripper"),
    (grasp, grasp, 50, 0.05, "confirm grasp"),
    (grasp, lift, 160, 0.05, "lift"),
]

frame = 0
lock_created = False
for arm_start, arm_end, duration, gripper_target, label in stages:
    print(label, flush=True)
    for local_frame in range(duration):
        arm_target = blend(arm_start, arm_end, local_frame / max(duration - 1, 1))
        set_full_target(robot, arm_target, gripper_target, dof_index)

        if label == "confirm grasp" and local_frame == duration - 5 and not lock_created:
            lock_cube_after_real_alignment(omni.usd.get_context().get_stage())
            lock_created = True

        world.step(render=not args.headless)
        frame += 1

timeline.stop()
cube_position, _ = cube.get_world_pose()
print(f"Final cube position: {cube_position}", flush=True)
print("Demo finished", flush=True)

if args.headless:
    app.close()
else:
    while app.is_running():
        app.update()
    app.close()
