import argparse
import json
from pathlib import Path

from isaacsim import SimulationApp


parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true")
parser.add_argument(
    "--random-seed",
    type=int,
    help="Randomize the cube position within the calibrated grasp region.",
)
parser.add_argument(
    "--record-dir",
    type=Path,
    help="Write 4 FPS 256x256 RGB frames and action JSONL to this directory.",
)
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
import carb.settings
import omni.physx
import omni.replicator.core as rep
import omni.timeline
import omni.usd
from isaacsim.core.api import World
from isaacsim.core.api.materials import PhysicsMaterial
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.robot_motion.motion_generation import ArticulationKinematicsSolver, LulaKinematicsSolver
from PIL import Image
from pxr import PhysicsSchemaTools, PhysxSchema, Usd, UsdGeom, UsdPhysics, UsdShade


CUBE_POSITION = np.array([0.33, 0.0, 0.020])
CUBE_SIZE = 0.040
if args.random_seed is not None:
    rng = np.random.default_rng(args.random_seed)
    CUBE_POSITION[:2] += rng.uniform(
        low=np.array([-0.006, -0.012]),
        high=np.array([0.006, 0.012]),
    )
GRASP_CENTER_OFFSET = np.zeros(3)
DOWNWARD = np.array([0.0, 1.0, 0.0, 0.0])
GRIPPER_JOINTS = [
    "drive_joint",
    "left_inner_knuckle_joint",
    "right_inner_knuckle_joint",
    "right_outer_knuckle_joint",
    "left_finger_joint",
    "right_finger_joint",
]
GRIPPER_CONTROL_JOINTS = GRIPPER_JOINTS
contact_stats = {
    "left_finger": {"count": 0, "impulse": 0.0},
    "right_finger": {"count": 0, "impulse": 0.0},
    "ground": {"count": 0, "impulse": 0.0},
}
record_dir = args.record_dir.resolve() if args.record_dir else None
record_rows = []
rgb_annotator = None


def capture_observation(frame, label, arm_target, gripper_target, robot, cube):
    if rgb_annotator is None or frame % 30 != 0:
        return

    rep.orchestrator.step(rt_subframes=2, delta_time=0.0, pause_timeline=False)
    image_name = f"rgb_{frame:06d}.png"
    rgba = rgb_annotator.get_data()
    Image.fromarray(rgba).convert("RGB").save(record_dir / image_name)

    cube_position, cube_orientation = cube.get_world_pose()
    record_rows.append(
        {
            "frame": frame,
            "time_seconds": frame / 120.0,
            "stage": label,
            "image": image_name,
            "action": {
                "arm_joint_positions": np.asarray(arm_target).tolist(),
                "gripper_joint_position": float(gripper_target),
            },
            "observation": {
                "joint_positions": robot.get_joint_positions().tolist(),
                "cube_position": cube_position.tolist(),
                "cube_orientation_wxyz": cube_orientation.tolist(),
            },
        }
    )


def on_contact_report(contact_headers, contact_data):
    for header in contact_headers:
        actor_paths = (
            str(PhysicsSchemaTools.intToSdfPath(header.actor0)),
            str(PhysicsSchemaTools.intToSdfPath(header.actor1)),
        )
        if "/World/GraspCube" not in actor_paths:
            continue

        other_path = actor_paths[1] if actor_paths[0] == "/World/GraspCube" else actor_paths[0]
        category = None
        for name in contact_stats:
            if name in other_path.lower():
                category = name
                break
        if category is None:
            continue

        start = header.contact_data_offset
        end = start + header.num_contact_data
        contact_stats[category]["count"] += header.num_contact_data
        contact_stats[category]["impulse"] += sum(
            float(np.linalg.norm(contact_data[index].impulse))
            for index in range(start, end)
        )


def blend(start, end, alpha):
    alpha = min(max(alpha, 0.0), 1.0)
    alpha = alpha * alpha * (3.0 - 2.0 * alpha)
    return start + (end - start) * alpha


def set_full_target(robot, arm_positions, gripper_position, dof_index):
    target = robot.get_joint_positions().copy()
    for index, value in enumerate(arm_positions):
        target[dof_index[f"joint{index + 1}"]] = value
    for name in GRIPPER_CONTROL_JOINTS:
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


def world_bounds(stage, path):
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [
            UsdGeom.Tokens.default_,
            UsdGeom.Tokens.render,
            UsdGeom.Tokens.proxy,
            UsdGeom.Tokens.guide,
        ],
    )
    bounds = cache.ComputeWorldBound(stage.GetPrimAtPath(path)).ComputeAlignedRange()
    return np.asarray(bounds.GetMin()), np.asarray(bounds.GetMax())


def axis_gap(first_min, first_max, second_min, second_max):
    return np.maximum(0.0, np.maximum(first_min - second_max, second_min - first_max))


def confirm_physical_grasp(stage, robot, dof_index):
    cube_min, cube_max = world_bounds(stage, "/World/GraspCube")
    left_min, left_max = world_bounds(stage, "/UF_ROBOT/root_joint/left_finger")
    right_min, right_max = world_bounds(stage, "/UF_ROBOT/root_joint/right_finger")
    left_gap = axis_gap(left_min, left_max, cube_min, cube_max)
    right_gap = axis_gap(right_min, right_max, cube_min, cube_max)

    print(f"Cube bounds: min={cube_min} max={cube_max}", flush=True)
    print(f"Left finger bounds: min={left_min} max={left_max}", flush=True)
    print(f"Right finger bounds: min={right_min} max={right_max}", flush=True)
    print(f"Left finger gap xyz: {left_gap}", flush=True)
    print(f"Right finger gap xyz: {right_gap}", flush=True)
    joint_positions = robot.get_joint_positions()
    actual_gripper = {
        name: float(joint_positions[dof_index[name]]) for name in GRIPPER_JOINTS
    }
    print(f"Actual gripper joints: {actual_gripper}", flush=True)
    print(f"Contact report: {contact_stats}", flush=True)

    contact_tolerance = 0.006
    if np.linalg.norm(left_gap) > contact_tolerance:
        raise RuntimeError("Grasp rejected: left finger is not touching the cube")
    if np.linalg.norm(right_gap) > contact_tolerance:
        raise RuntimeError("Grasp rejected: right finger is not touching the cube")
    print("Bilateral finger contact confirmed; lifting without an attachment joint", flush=True)


def bind_finger_material(stage, physics_material):
    for path in (
        "/UF_ROBOT/root_joint/left_finger",
        "/UF_ROBOT/root_joint/right_finger",
    ):
        finger = stage.GetPrimAtPath(path)
        collision_group = stage.GetPrimAtPath(f"{path}/collisions")
        if collision_group.IsInstance():
            collision_group.SetInstanceable(False)
        colliders = [
            prim
            for prim in Usd.PrimRange(finger)
            if prim.HasAPI(UsdPhysics.CollisionAPI)
        ]
        if not colliders:
            descendants = [
                f"{prim.GetPath()} type={prim.GetTypeName()} "
                f"schemas={list(prim.GetAppliedSchemas())}"
                for prim in Usd.PrimRange(finger)
            ]
            print(f"No CollisionAPI prims below {path}: {descendants}", flush=True)
            colliders = [finger]
        for collider in colliders:
            binding = UsdShade.MaterialBindingAPI.Apply(collider)
            binding.Bind(
                physics_material.material,
                bindingStrength=UsdShade.Tokens.strongerThanDescendants,
                materialPurpose="physics",
            )
            print(f"Bound high-friction material to {collider.GetPath()}", flush=True)


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
    static_friction=3.0,
    dynamic_friction=2.5,
    restitution=0.0,
)
bind_finger_material(omni.usd.get_context().get_stage(), grip_material)

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
        size=CUBE_SIZE,
        color=np.array([0.85, 0.12, 0.08]),
        physics_material=grip_material,
        mass=0.05,
    )
)
if record_dir:
    record_dir.mkdir(parents=True, exist_ok=True)
    carb.settings.get_settings().set("/omni/replicator/captureOnPlay", False)
    camera = rep.create.camera(
        position=(1.15, 1.15, 0.82),
        look_at=(0.25, 0.0, 0.22),
    )
    rep.create.light(rotation=(315, 0, 0), intensity=2500, light_type="distant")
    rep.create.light(intensity=500, light_type="dome")
    render_product = rep.create.render_product(camera, (256, 256))
    rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
    rgb_annotator.attach(render_product)
    rep.orchestrator.preview()

contact_report = PhysxSchema.PhysxContactReportAPI.Apply(
    omni.usd.get_context().get_stage().GetPrimAtPath("/World/GraspCube")
)
contact_report.CreateThresholdAttr().Set(0.0)
contact_subscription = (
    omni.physx.get_physx_simulation_interface().subscribe_contact_report_events(
        on_contact_report
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

for name in GRIPPER_CONTROL_JOINTS:
    index = dof_index[name]
    kps[index] = 12000.0
    kds[index] = 800.0
    max_efforts[index] = 120.0

controller.set_gains(kps=kps, kds=kds)
controller.set_max_efforts(max_efforts)

lula = LulaKinematicsSolver(
    robot_description_path=str(ROOT / "config/xarm6_robot_descriptor.yaml"),
    urdf_path=str(ROOT / "assets/xarm6_gripper/xarm6_gripper_control.urdf"),
)
ik_solver = ArticulationKinematicsSolver(robot, lula, "link_tcp")

grasp_xy = CUBE_POSITION[:2] + GRASP_CENTER_OFFSET[:2]
approach = solve_pose(ik_solver, [grasp_xy[0], grasp_xy[1], 0.22])
grasp = solve_pose(ik_solver, [grasp_xy[0], grasp_xy[1], 0.020])
lift = solve_pose(ik_solver, [grasp_xy[0], grasp_xy[1], 0.30])

start = robot.get_joint_positions()[:6].copy()
set_full_target(robot, start, 0.0, dof_index)

if not args.headless:
    set_camera_view(
        eye=np.array([0.95, 0.95, 0.65]),
        target=np.array([0.25, 0.0, 0.18]),
        camera_prim_path="/OmniverseKit_Persp",
    )

timeline = omni.timeline.get_timeline_interface()
timeline.play()

stages = [
    (start, approach, 120, 0.0, "move above cube"),
    (approach, grasp, 120, 0.0, "descend"),
    (grasp, grasp, 180, 0.70, "close gripper"),
    (grasp, grasp, 80, 0.70, "confirm grasp"),
    (grasp, lift, 240, 0.70, "lift"),
]

frame = 0
grasp_confirmed = False
max_cube_z = float(CUBE_POSITION[2])
for arm_start, arm_end, duration, gripper_target, label in stages:
    print(label, flush=True)
    if label == "lift":
        for stats in contact_stats.values():
            stats["count"] = 0
            stats["impulse"] = 0.0
    for local_frame in range(duration):
        arm_target = blend(arm_start, arm_end, local_frame / max(duration - 1, 1))
        set_full_target(robot, arm_target, gripper_target, dof_index)

        if label == "confirm grasp" and local_frame == duration - 5 and not grasp_confirmed:
            confirm_physical_grasp(
                omni.usd.get_context().get_stage(), robot, dof_index
            )
            grasp_confirmed = True

        world.step(render=not args.headless)
        capture_observation(
            frame,
            label,
            arm_target,
            gripper_target,
            robot,
            cube,
        )
        if label == "lift":
            current_cube_position, _ = cube.get_world_pose()
            max_cube_z = max(max_cube_z, float(current_cube_position[2]))
        frame += 1

timeline.stop()
cube_position, _ = cube.get_world_pose()
print(f"Final cube position: {cube_position}", flush=True)
print(f"Maximum cube height during lift: {max_cube_z}", flush=True)
print(f"Lift contact report: {contact_stats}", flush=True)
for side in ("left_finger", "right_finger"):
    finger_min, finger_max = world_bounds(
        omni.usd.get_context().get_stage(), f"/UF_ROBOT/root_joint/{side}"
    )
    print(f"Final {side} bounds: min={finger_min} max={finger_max}", flush=True)
if cube_position[2] < 0.10:
    raise RuntimeError("Grasp failed: cube did not leave the ground")
print("Demo finished", flush=True)

if record_dir:
    with (record_dir / "actions.jsonl").open("w", encoding="utf-8") as stream:
        for row in record_rows:
            stream.write(json.dumps(row, ensure_ascii=True) + "\n")
    metadata = {
        "random_seed": args.random_seed,
        "capture_fps": 4,
        "physics_fps": 120,
        "resolution": [256, 256],
        "cube_size_m": CUBE_SIZE,
        "cube_initial_position": CUBE_POSITION.tolist(),
        "cube_final_position": cube_position.tolist(),
        "success": True,
        "frames_written": len(record_rows),
    }
    (record_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    rep.orchestrator.wait_until_complete()
    print(f"Recorded episode to {record_dir}", flush=True)

if args.headless:
    app.close()
else:
    while app.is_running():
        app.update()
    app.close()
