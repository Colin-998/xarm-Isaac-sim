"""Visualize and physically execute a saved cuRobo xArm6 pick plan in Isaac Sim."""

import argparse
import json
import math
import os
import subprocess
from pathlib import Path

from isaacsim import SimulationApp


parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true")
parser.add_argument(
    "--plan",
    type=Path,
    default=Path("outputs/curobo_dynamic_pick_plan.npz"),
)
parser.add_argument(
    "--metadata",
    type=Path,
    default=Path("outputs/curobo_dynamic_pick_plan.json"),
)
parser.add_argument(
    "--steps-per-waypoint",
    type=int,
    default=2,
    help="Physics steps used to track each cuRobo waypoint.",
)
parser.add_argument(
    "--physical-grasp",
    action="store_true",
    help="Use finger friction only instead of the default payload attachment.",
)
parser.add_argument("--conveyor-speed", type=float, default=0.25)
parser.add_argument(
    "--cycles",
    type=int,
    default=1,
    help="Number of cycles in headless mode; the GUI loops until End is pressed.",
)
args, _ = parser.parse_known_args()

ROOT = Path(__file__).resolve().parents[1]
ROBOT_USD = ROOT / "assets/xarm6_gripper/xarm6_gripper.usd"
ISAAC_PYTHON = Path.home() / "isaac_sim_5.1" / "python.bat"
RUNTIME_PLANNER = ROOT / "scripts/xarm6_curobo_runtime.py"
RUNTIME_REQUEST = ROOT / "outputs/curobo_runtime_request.json"
RUNTIME_PLAN = ROOT / "outputs/curobo_runtime_plan.npz"
PHYSICS_FPS = 120
BELT_RADIUS = 0.43
BELT_HEIGHT = 0.045
BELT_WIDTH = 0.13
BELT_SEGMENTS = 17
BELT_START_ANGLE = math.radians(-105.0)
BELT_END_ANGLE = math.radians(105.0)
OPEN_GRIPPER = 0.0
CLOSED_GRIPPER = 0.70
GRIPPER_JOINTS = [
    "drive_joint",
    "left_inner_knuckle_joint",
    "right_inner_knuckle_joint",
    "right_outer_knuckle_joint",
    "left_finger_joint",
    "right_finger_joint",
]

app = SimulationApp(
    {
        "headless": args.headless,
        "renderer": "RaytracedLighting",
        "width": 1280,
        "height": 720,
    }
)

import numpy as np
import omni.kit.commands
import omni.timeline
import omni.ui as ui
import omni.usd
from pxr import Gf, PhysxSchema, Usd, UsdPhysics, UsdShade
from isaacsim.core.api import World
from isaacsim.core.api.materials import PhysicsMaterial
from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaacsim.core.utils.extensions import enable_extension
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.robot_motion.motion_generation import (
    ArticulationKinematicsSolver,
    LulaKinematicsSolver,
)


def arc_position(angle, z=BELT_HEIGHT):
    return np.array(
        [
            BELT_RADIUS * math.cos(angle),
            BELT_RADIUS * math.sin(angle),
            z,
        ]
    )


def create_arc_conveyor(world):
    segment_length = (
        BELT_RADIUS
        * (BELT_END_ANGLE - BELT_START_ANGLE)
        / (BELT_SEGMENTS - 1)
    )
    segments = []
    graph_nodes = []
    belt_material = PhysicsMaterial(
        prim_path="/World/Materials/ConveyorPhysics",
        static_friction=1.5,
        dynamic_friction=1.2,
        restitution=0.0,
    )
    enable_extension("isaacsim.asset.gen.conveyor")
    app.update()

    for index, angle in enumerate(
        np.linspace(BELT_START_ANGLE, BELT_END_ANGLE, BELT_SEGMENTS)
    ):
        segment = world.scene.add(
            FixedCuboid(
                prim_path=f"/World/Conveyor/Segment_{index:02d}",
                name=f"conveyor_segment_{index:02d}",
                position=arc_position(angle),
                orientation=euler_angles_to_quat(
                    np.array([0.0, 0.0, angle + math.pi / 2.0])
                ),
                scale=np.array(
                    [segment_length * 1.10, BELT_WIDTH, 0.035]
                ),
                color=np.array([0.08, 0.12, 0.16]),
                physics_material=belt_material,
            )
        )
        segments.append(segment)

        rigid_body = UsdPhysics.RigidBodyAPI.Apply(segment.prim)
        rigid_body.CreateRigidBodyEnabledAttr().Set(True)
        rigid_body.CreateKinematicEnabledAttr().Set(True)
        _, graph_node = omni.kit.commands.execute(
            "CreateConveyorBelt",
            conveyor_prim=segment.prim,
        )
        graph_node.GetAttribute("inputs:direction").Set((1.0, 0.0, 0.0))
        graph_node.GetAttribute("inputs:enabled").Set(False)
        surface_velocity = PhysxSchema.PhysxSurfaceVelocityAPI.Apply(
            segment.prim
        )
        surface_velocity.GetSurfaceVelocityLocalSpaceAttr().Set(False)
        surface_velocity.GetSurfaceVelocityAttr().Set(Gf.Vec3f(0.0))
        velocity_attr = graph_node.GetParent().GetAttribute(
            "graph:variable:Velocity"
        )
        if not velocity_attr:
            raise RuntimeError(
                f"Conveyor graph has no Velocity variable for {segment.prim_path}"
            )
        velocity_attr.Set(0.0)
        graph_nodes.append(graph_node)

    end_tangent = np.array(
        [-math.sin(BELT_END_ANGLE), math.cos(BELT_END_ANGLE), 0.0]
    )
    world.scene.add(
        FixedCuboid(
            prim_path="/World/Conveyor/EndStop",
            name="conveyor_end_stop",
            position=(
                arc_position(BELT_END_ANGLE)
                + end_tangent * 0.065
                + np.array([0.0, 0.0, 0.045])
            ),
            orientation=euler_angles_to_quat(
                np.array([0.0, 0.0, BELT_END_ANGLE + math.pi / 2.0])
            ),
            scale=np.array([0.018, BELT_WIDTH, 0.09]),
            color=np.array([0.7, 0.72, 0.75]),
            physics_material=belt_material,
        )
    )
    return segments, graph_nodes


def set_conveyor_speed(graph_nodes, speed):
    angles = np.linspace(
        BELT_START_ANGLE,
        BELT_END_ANGLE,
        BELT_SEGMENTS,
    )
    for graph_node, angle in zip(graph_nodes, angles):
        targets = graph_node.GetRelationship("inputs:conveyorPrim").GetTargets()
        if len(targets) != 1:
            raise RuntimeError(
                f"Invalid conveyor target at {graph_node.GetPath()}"
            )
        conveyor_prim = omni.usd.get_context().get_stage().GetPrimAtPath(
            targets[0]
        )
        tangent = np.array([-math.sin(angle), math.cos(angle), 0.0])
        surface_velocity = PhysxSchema.PhysxSurfaceVelocityAPI.Apply(
            conveyor_prim
        )
        surface_velocity.GetSurfaceVelocityLocalSpaceAttr().Set(False)
        surface_velocity.GetSurfaceVelocityAttr().Set(
            Gf.Vec3f(*(tangent * float(speed)))
        )
        graph_node.GetParent().GetAttribute(
            "graph:variable:Velocity"
        ).Set(float(speed))


def bind_finger_material(stage, physics_material):
    bound = []
    for prim in stage.TraverseAll():
        if prim.GetName() not in {"left_finger", "right_finger"}:
            continue
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            physics_material.material,
            bindingStrength=UsdShade.Tokens.strongerThanDescendants,
            materialPurpose="physics",
        )
        bound.append(str(prim.GetPath()))
    return bound


def configure_robot(robot, dof_index):
    robot.set_solver_position_iteration_count(64)
    robot.set_solver_velocity_iteration_count(16)
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


def complete_target(robot, dof_index, arm_positions, gripper_position):
    target = robot.get_joint_positions().copy()
    for joint_number, value in enumerate(arm_positions, start=1):
        target[dof_index[f"joint{joint_number}"]] = value
    for name in GRIPPER_JOINTS:
        target[dof_index[name]] = gripper_position
    return target


def set_pose_immediately(robot, dof_index, arm_positions, gripper_position):
    target = complete_target(
        robot, dof_index, arm_positions, gripper_position
    )
    robot.set_joint_positions(target)
    robot.set_joint_velocities(np.zeros_like(target))


def apply_target(robot, dof_index, arm_positions, gripper_position):
    target = complete_target(
        robot, dof_index, arm_positions, gripper_position
    )
    robot.get_articulation_controller().apply_action(
        ArticulationAction(joint_positions=target)
    )


plan_path = args.plan.resolve()
metadata_path = args.metadata.resolve()
if not plan_path.exists():
    raise FileNotFoundError(f"Plan not found: {plan_path}")
if not metadata_path.exists():
    raise FileNotFoundError(f"Metadata not found: {metadata_path}")

plan_data = np.load(plan_path)
metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
phases = metadata["phases"]
phase_names = [phase["name"] for phase in phases]

omni.usd.get_context().open_stage(str(ROBOT_USD))
for _ in range(10):
    app.update()

world = World(
    stage_units_in_meters=1.0,
    physics_dt=1.0 / PHYSICS_FPS,
    rendering_dt=1.0 / 60.0,
)
world.scene.add(
    FixedCuboid(
        prim_path="/World/Ground",
        name="ground",
        position=np.array([0.0, 0.0, -0.025]),
        scale=np.array([2.0, 2.0, 0.05]),
        color=np.array([0.28, 0.42, 0.55]),
    )
)
conveyor_segments, conveyor_graph_nodes = create_arc_conveyor(world)
for index, obstacle in enumerate(metadata["obstacles"]):
    world.scene.add(
        FixedCuboid(
            prim_path=f"/World/Obstacles/Obstacle_{index}",
            name=f"obstacle_{index}",
            position=np.asarray(obstacle["position"], dtype=float),
            scale=np.asarray(
                obstacle.get("dims", obstacle.get("scale")),
                dtype=float,
            ),
            color=np.array([0.95, 0.55 - index * 0.15, 0.05]),
        )
    )

grip_material = PhysicsMaterial(
    prim_path="/World/Materials/GripperPhysics",
    static_friction=3.0,
    dynamic_friction=2.5,
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
        prim_path="/World/Objects/CycleCube",
        name="cycle_cube",
        position=np.asarray(metadata["cube_position"], dtype=float),
        size=float(metadata["cube_size"]),
        color=np.array([0.82, 0.08, 0.05]),
        mass=0.05,
        physics_material=grip_material,
    )
)
if not bind_finger_material(
    omni.usd.get_context().get_stage(), grip_material
):
    raise RuntimeError("Could not bind physics material to gripper fingers")

world.reset()
dof_index = {name: index for index, name in enumerate(robot.dof_names)}
required_joints = [f"joint{i}" for i in range(1, 7)] + GRIPPER_JOINTS
missing = [name for name in required_joints if name not in dof_index]
if missing:
    raise RuntimeError(f"Missing expected joints: {missing}")
configure_robot(robot, dof_index)
lula = LulaKinematicsSolver(
    robot_description_path=str(
        ROOT / "config/xarm6_robot_descriptor.yaml"
    ),
    urdf_path=str(
        ROOT / "assets/xarm6_gripper/xarm6_gripper_control.urdf"
    ),
)
fk_solver = ArticulationKinematicsSolver(robot, lula, "link_tcp")

first_positions = np.asarray(plan_data[phases[0]["name"]])[0]
set_pose_immediately(
    robot,
    dof_index,
    first_positions,
    OPEN_GRIPPER,
)
cube_initial_position = np.asarray(metadata["cube_position"], dtype=float)
cube.set_world_pose(position=cube_initial_position)
cube.set_linear_velocity(np.zeros(3))
cube.set_angular_velocity(np.zeros(3))

if not args.headless:
    set_camera_view(
        eye=np.array([1.15, 1.15, 0.82]),
        target=np.array([0.0, 0.0, 0.20]),
        camera_prim_path="/OmniverseKit_Persp",
    )

timeline = omni.timeline.get_timeline_interface()
timeline.play()
control_state = {"retry_requested": False, "end_requested": False}
status_label = None
if not args.headless:
    window = ui.Window("cuRobo Conveyor Cycle", width=340, height=135)
    with window.frame:
        with ui.VStack(spacing=8):
            status_label = ui.Label("Starting...")
            with ui.HStack(spacing=8):
                ui.Button(
                    "Run Again",
                    height=38,
                    clicked_fn=lambda: control_state.update(
                        retry_requested=True
                    ),
                )
                ui.Button(
                    "End",
                    height=38,
                    clicked_fn=lambda: control_state.update(
                        end_requested=True
                    ),
                )


payload_state = {"held": False}


def update_attached_payload():
    if args.physical_grasp or not payload_state["held"]:
        return
    tcp_position, _ = fk_solver.compute_end_effector_pose()
    cube.set_world_pose(position=np.asarray(tcp_position))
    cube.set_linear_velocity(np.zeros(3))
    cube.set_angular_velocity(np.zeros(3))


def step_for(frames, arm_positions, gripper_position):
    for _ in range(frames):
        if (
            control_state["end_requested"]
            or control_state["retry_requested"]
            or not app.is_running()
        ):
            return False
        apply_target(
            robot,
            dof_index,
            arm_positions,
            gripper_position,
        )
        world.step(render=not args.headless)
        update_attached_payload()
    return True


def reset_cycle():
    set_conveyor_speed(conveyor_graph_nodes, 0.0)
    configure_robot(robot, dof_index)
    set_pose_immediately(
        robot,
        dof_index,
        first_positions,
        OPEN_GRIPPER,
    )
    cube.set_world_pose(position=cube_initial_position)
    cube.set_linear_velocity(np.zeros(3))
    cube.set_angular_velocity(np.zeros(3))
    payload_state["held"] = False
    for _ in range(30):
        world.step(render=not args.headless)


def arm_positions():
    positions = robot.get_joint_positions()
    return np.asarray(
        [positions[dof_index[f"joint{i}"]] for i in range(1, 7)],
        dtype=float,
    )


def detect_and_plan():
    for _ in range(PHYSICS_FPS // 4):
        world.step(render=not args.headless)
    detected_position = np.asarray(cube.get_world_pose()[0], dtype=float)
    if status_label is not None:
        status_label.text = (
            "Planning from cube "
            f"({detected_position[0]:.3f}, "
            f"{detected_position[1]:.3f}, "
            f"{detected_position[2]:.3f})"
        )
    print(
        f"detected_cube={detected_position.round(4).tolist()}",
        flush=True,
    )
    request = {
        "cube_position": detected_position.tolist(),
        "place_position": metadata["place_position"],
        "start_arm_positions": arm_positions().tolist(),
        "cube_size": metadata["cube_size"],
        "clearance": 0.012,
        "obstacles": metadata["obstacles"],
    }
    RUNTIME_REQUEST.parent.mkdir(parents=True, exist_ok=True)
    RUNTIME_REQUEST.write_text(
        json.dumps(request, indent=2),
        encoding="utf-8",
    )
    command = subprocess.list2cmdline(
        [
            str(ISAAC_PYTHON),
            str(RUNTIME_PLANNER),
            "--request",
            str(RUNTIME_REQUEST),
            "--output",
            str(RUNTIME_PLAN),
        ]
    )
    planner_environment = os.environ.copy()
    planner_environment["WARP_CACHE_PATH"] = str(
        ROOT / "outputs/warp_curobo_cache"
    )
    result = subprocess.run(
        command,
        cwd=ROOT,
        shell=True,
        env=planner_environment,
        capture_output=True,
        text=True,
        timeout=180,
    )
    if result.returncode != 0:
        details = "\n".join(
            (result.stdout + "\n" + result.stderr).splitlines()[-30:]
        )
        raise RuntimeError(
            "Dynamic cuRobo planner process failed:\n" + details
        )
    for line in result.stdout.splitlines():
        if "runtime_plan_complete" in line:
            print(line, flush=True)
    with np.load(RUNTIME_PLAN) as runtime_data:
        trajectories = {
            phase_name: np.asarray(runtime_data[phase_name]).copy()
            for phase_name in phase_names
        }
    print(
        "dynamic_plan_complete "
        + " ".join(
            f"{name}={len(trajectories[name])}"
            for name in phase_names
        ),
        flush=True,
    )
    return trajectories


def run_pick_and_place(trajectories):
    gripper_position = OPEN_GRIPPER
    for phase_name in phase_names:
        trajectory = np.asarray(trajectories[phase_name])
        if status_label is not None:
            status_label.text = f"Running: {phase_name}"
        print(f"phase={phase_name} waypoints={len(trajectory)}", flush=True)

        if phase_name == "lift_cube":
            gripper_position = CLOSED_GRIPPER
            if not step_for(100, trajectory[0], gripper_position):
                return False
            payload_state["held"] = True
            update_attached_payload()
        elif phase_name == "retreat_after_release":
            if payload_state["held"] and not args.physical_grasp:
                cube.set_world_pose(
                    position=np.asarray(metadata["place_position"])
                )
                cube.set_linear_velocity(np.zeros(3))
                cube.set_angular_velocity(np.zeros(3))
            payload_state["held"] = False
            gripper_position = OPEN_GRIPPER
            if not step_for(90, trajectory[0], gripper_position):
                return False

        for arm_positions in trajectory:
            if not step_for(
                max(args.steps_per_waypoint, 1),
                arm_positions,
                gripper_position,
            ):
                return False

    final_cube = np.asarray(cube.get_world_pose()[0])
    place_position = np.asarray(metadata["place_position"])
    place_distance = float(
        np.linalg.norm(final_cube[:2] - place_position[:2])
    )
    print(
        f"playback_complete cube={final_cube.round(4).tolist()} "
        f"place_distance={place_distance:.4f}m",
        flush=True,
    )
    if status_label is not None:
        status_label.text = "Conveyor returning cube..."
    return True


def return_cube_on_conveyor(hold_positions):
    set_conveyor_speed(conveyor_graph_nodes, args.conveyor_speed)
    pickup_position = np.asarray(metadata["cube_position"], dtype=float)
    max_frames = PHYSICS_FPS * 15
    for frame in range(max_frames):
        if (
            control_state["end_requested"]
            or control_state["retry_requested"]
            or not app.is_running()
        ):
            set_conveyor_speed(conveyor_graph_nodes, 0.0)
            return False
        apply_target(
            robot,
            dof_index,
            hold_positions,
            OPEN_GRIPPER,
        )
        world.step(render=not args.headless)
        cube_position = np.asarray(cube.get_world_pose()[0])
        end_distance = float(
            np.linalg.norm(cube_position[:2] - pickup_position[:2])
        )
        if frame % PHYSICS_FPS == 0:
            print(
                f"conveyor_return t={frame / PHYSICS_FPS:.1f}s "
                f"distance={end_distance:.3f}m",
                flush=True,
            )
        if end_distance < 0.065:
            set_conveyor_speed(conveyor_graph_nodes, 0.0)
            for _ in range(PHYSICS_FPS // 3):
                apply_target(
                    robot,
                    dof_index,
                    hold_positions,
                    OPEN_GRIPPER,
                )
                world.step(render=not args.headless)
            print("conveyor_return complete", flush=True)
            return True

    set_conveyor_speed(conveyor_graph_nodes, 0.0)
    raise RuntimeError("Cube did not return to the conveyor pickup point")


try:
    reset_cycle()
    completed_cycles = 0
    while app.is_running():
        if control_state["end_requested"]:
            break
        if control_state["retry_requested"]:
            control_state["retry_requested"] = False
            reset_cycle()
            completed_cycles = 0

        cycle_number = completed_cycles + 1
        if status_label is not None:
            status_label.text = f"Cycle {cycle_number}: detecting cube"
        print(f"cycle={cycle_number} start", flush=True)
        trajectories = detect_and_plan()
        if not run_pick_and_place(trajectories):
            continue
        hold_positions = np.asarray(
            trajectories["retreat_after_release"]
        )[-1]
        if not return_cube_on_conveyor(hold_positions):
            continue
        completed_cycles += 1
        if status_label is not None:
            status_label.text = f"Cycle {completed_cycles} complete"
        print(f"cycle={completed_cycles} complete", flush=True)
        if args.headless and completed_cycles >= max(args.cycles, 1):
            break
finally:
    set_conveyor_speed(conveyor_graph_nodes, 0.0)
    timeline.stop()
    app.close()
