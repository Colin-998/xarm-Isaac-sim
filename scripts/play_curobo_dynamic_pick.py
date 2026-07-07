"""Visualize and physically execute a saved cuRobo xArm6 pick plan in Isaac Sim."""
import numpy as np
import argparse
from datetime import datetime, timezone
import json
import math
import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault(
    "WARP_CACHE_PATH",
    str(ROOT / "outputs" / "warp_cache_play_curobo_dynamic_pick"),
)

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
    "--use-static-plan",
    action="store_true",
    help=(
        "Use the loaded plan file instead of spawning the dynamic cuRobo "
        "planner. Intended for quick controller validation from the saved "
        "initial cube pose."
    ),
)
parser.add_argument(
    "--physical-grasp",
    action="store_true",
    help="Use finger friction only instead of the default payload attachment.",
)
parser.add_argument("--conveyor-speed", type=float, default=0.25)
parser.add_argument(
    "--record-root",
    type=Path,
    help="Write successful 4 FPS, 256x256 episodes below this directory.",
)
parser.add_argument(
    "--keep-failed-episodes",
    action="store_true",
    help=(
        "Keep failed recorded episodes with oracle actions for DAgger-style "
        "online correction instead of aborting their temp directory."
    ),
)
parser.add_argument(
    "--episodes",
    type=int,
    help="Total number of recorded episodes desired; supports resume.",
)
parser.add_argument(
    "--cycles",
    type=int,
    default=1,
    help="Number of cycles in headless mode; the GUI loops until End is pressed.",
)
parser.add_argument(
    "--stop-after-cycles",
    action="store_true",
    help="Also stop GUI runs after --cycles attempts; useful for visible gates.",
)
parser.add_argument(
    "--brain-control",
    choices=["off", "observe", "filtered", "direct"],
    default="off",
    help=(
        "Run the Stage-3 V-JEPA2 policy online. observe logs predictions, "
        "filtered applies predictions only when safety checks pass, and "
        "direct sends rate-limited policy targets without teacher blending."
    ),
)
parser.add_argument(
    "--brain-policy",
    type=Path,
    default=Path("outputs/stage3_video_sft_500ep_w4/latest_stage3_policy.pt"),
)
parser.add_argument(
    "--task-name",
    default=None,
    help="Logical task label recorded into DAgger/RLDS metadata.",
)
parser.add_argument(
    "--task-instruction",
    default=None,
    help="Natural-language instruction for SmolVLA inference and recording.",
)
parser.add_argument(
    "--disable-conveyor-return",
    action="store_true",
    help="End each cycle after placing the cube; useful for non-conveyor tasks.",
)
parser.add_argument(
    "--task-success-distance",
    type=float,
    default=0.10,
    help="XY distance threshold from task place target for task success.",
)
parser.add_argument(
    "--basket-center",
    default=None,
    help=(
        "Override basket center as x,y,z in metres for basket_drop. The "
        "robot still releases at metadata place_position."
    ),
)
parser.add_argument(
    "--basket-release-velocity",
    default=None,
    help=(
        "Override basket release velocity as vx,vy,vz in metres/second. "
        "Used only by --basket-velocity-mode metadata."
    ),
)
parser.add_argument(
    "--basket-velocity-mode",
    choices=["metadata", "model", "analytic", "gripper"],
    default="metadata",
    help=(
        "How basket_drop chooses release velocity: metadata uses a fixed "
        "value, model asks a learned ballistic policy, analytic is a "
        "non-learning physics baseline, and gripper inherits velocity from "
        "the physical gripper swing before opening."
    ),
)
parser.add_argument(
    "--basket-release-policy",
    type=Path,
    default=ROOT / "outputs/smolvla_multitask_dagger/ballistic_throw_policy.pt",
    help="Checkpoint for --basket-velocity-mode model.",
)
parser.add_argument(
    "--basket-analytic-flight-time",
    type=float,
    default=None,
    help=(
        "Optional fixed flight time for analytic basket velocity. If omitted, "
        "a distance-based flight time is used."
    ),
)
parser.add_argument(
    "--basket-gripper-throw-frames",
    type=int,
    default=8,
    help="Number of fast servo frames used to swing the held cube before release.",
)
parser.add_argument(
    "--basket-gripper-throw-speed-scale",
    type=float,
    default=0.85,
    help=(
        "Scale applied to the internally planned gripper swing displacement. "
        "This changes gripper motion, not cube velocity directly."
    ),
)
parser.add_argument(
    "--basket-gripper-throw-lift",
    type=float,
    default=0.080,
    help="Extra upward gripper swing displacement in metres before release.",
)
parser.add_argument(
    "--basket-flight-settle-frames",
    type=int,
    default=90,
    help="Frames to hold after a gripper throw release so the cube visibly flies.",
)
parser.add_argument(
    "--basket-flight-mode",
    choices=["kinematic", "physics"],
    default="kinematic",
    help=(
        "How to show the cube after release. kinematic uses the gripper-derived "
        "release velocity and gravity to animate a stable ballistic arc; "
        "physics lets PhysX free-flight the cube."
    ),
)
parser.add_argument(
    "--basket-min-flight-frames",
    type=int,
    default=90,
    help=(
        "Minimum visible flight frames for basket_drop. This prevents older "
        "commands with --basket-flight-settle-frames 0 from ending at release."
    ),
)
parser.add_argument(
    "--basket-success-distance",
    type=float,
    default=0.12,
    help="XY distance threshold from basket center for basket_drop success.",
)
parser.add_argument(
    "--basket-gripper-release-open-frames",
    type=int,
    default=0,
    help=(
        "Frames to visibly open the gripper after release. Defaults to 0 "
        "because Isaac Sim 5.1 can shut down when stepping PhysX immediately "
        "after this basket release path."
    ),
)
parser.add_argument("--brain-device", default="cuda")
parser.add_argument("--brain-local-files-only", action="store_true")
parser.add_argument("--brain-blend", type=float, default=0.35)
parser.add_argument("--brain-max-teacher-delta", type=float, default=0.45)
parser.add_argument("--brain-max-step-delta", type=float, default=0.08)
parser.add_argument(
    "--brain-delta-gain",
    type=float,
    default=1.0,
    help=(
        "Scale delta-action checkpoints at runtime before rate limiting. "
        "This is useful for DAgger gates where the learned direction is "
        "correct but too small to reach the next phase."
    ),
)
parser.add_argument(
    "--brain-posture-blend",
    type=float,
    default=0.65,
    help=(
        "For tcp_delta_posture checkpoints, blend this fraction of the "
        "learned joint2-4 posture residual into the IK result. The TCP "
        "still comes from the learned Cartesian delta; this only lets the "
        "brain bias the arm body away from obstacles."
    ),
)
parser.add_argument(
    "--brain-strict-direct",
    action="store_true",
    help=(
        "In direct mode, never fall back to a teacher action. Hold position "
        "while the visual clip warms up and fail the episode on invalid or "
        "phase-mismatched predictions so the failure can become DAgger data."
    ),
)
parser.add_argument(
    "--brain-observe-fps",
    type=float,
    default=4.0,
    help=(
        "How often the online V-JEPA2/Stage-3 observer runs. Higher values "
        "make the visualized predicted EE path update more often, but cost "
        "more GPU time. Dataset recording remains 4 FPS."
    ),
)
parser.add_argument(
    "--brain-phase-hold-frames",
    type=int,
    default=360,
    help=(
        "Extra frames a brain-controlled run may hold the current phase until "
        "the end-effector reaches the phase geometry target. This prevents "
        "the task scheduler from switching to descent/lift before the policy "
        "has actually arrived."
    ),
)
parser.add_argument(
    "--brain-approach-ready-distance",
    type=float,
    default=0.12,
    help=(
        "Brain phase-hold distance in meters from the grasp center to the "
        "hover grasp target before approach_cube may advance."
    ),
)
parser.add_argument(
    "--brain-descend-ready-distance",
    type=float,
    default=0.030,
    help=(
        "Brain phase-hold distance in meters from the grasp center to the "
        "cube grasp target before descend_to_cube may advance."
    ),
)
parser.add_argument(
    "--show-brain-ee-path",
    action="store_true",
    help=(
        "Draw V-JEPA2/Stage-3 predicted end-effector path as USD overlay "
        "points for now, 0.5s, 1.0s, and 2.0s."
    ),
)
parser.add_argument(
    "--brain-allow-phase-mismatch",
    action="store_true",
    help=(
        "Allow direct brain actions even when the predicted phase disagrees "
        "with the active task phase. By default mismatched phase predictions "
        "fall back to the active task target for safety."
    ),
)
parser.add_argument(
    "--brain-terminal-servo",
    action="store_true",
    help=(
        "During final approach phases, use detected cube pose to servo the "
        "TCP toward the cube through IK. This moves only robot targets; it "
        "never teleports the cube."
    ),
)
parser.add_argument(
    "--brain-terminal-servo-without-vision",
    action="store_true",
    help=(
        "Validation mode: allow terminal servo to run from detected cube pose "
        "without waiting for V-JEPA/Llama predictions. This still only changes "
        "robot joint targets and is useful for proving the no-teleport final "
        "approach controller."
    ),
)
parser.add_argument(
    "--brain-terminal-servo-phases",
    default="close_gripper",
    help="Comma-separated phases where terminal cube servo is allowed.",
)
parser.add_argument(
    "--brain-terminal-servo-radius",
    type=float,
    default=0.45,
    help="Enable terminal servo when TCP is within this many meters of cube.",
)
parser.add_argument(
    "--brain-terminal-servo-step",
    type=float,
    default=0.035,
    help="Maximum Cartesian TCP motion requested by one servo update.",
)
parser.add_argument(
    "--brain-terminal-servo-max-joint-delta",
    type=float,
    default=0.04,
    help="Maximum joint delta applied by one terminal servo update.",
)
parser.add_argument(
    "--brain-terminal-servo-z-offset",
    type=float,
    default=0.010,
    help="Vertical TCP offset relative to the detected cube center.",
)
parser.add_argument(
    "--disable-vertical-grasp-servo",
    action="store_true",
    help=(
        "Disable the safe final grasp servo that aligns XY above the cube "
        "before descending vertically."
    ),
)
parser.add_argument(
    "--vertical-grasp-hover-height",
    type=float,
    default=0.075,
    help="Height above the cube used before the final vertical grasp descent.",
)
parser.add_argument(
    "--vertical-grasp-xy-tolerance",
    type=float,
    default=0.010,
    help="XY tolerance required before the final grasp servo descends.",
)
parser.add_argument(
    "--vertical-grasp-steps",
    type=int,
    default=8,
    help="Number of Cartesian waypoints used for the final vertical descent.",
)
parser.add_argument(
    "--vertical-grasp-frames-per-step",
    type=int,
    default=8,
    help="Simulation frames used for each vertical grasp waypoint.",
)
parser.add_argument(
    "--grasp-outward-offset",
    type=float,
    default=0.015,
    help=(
        "Move the final grasp center this many meters outward from the robot "
        "base, so the fingers approach the cube from slightly outside instead "
        "of pressing into its inner face."
    ),
)
parser.add_argument(
    "--grasp-cube-tcp-local-offset",
    default="0,0,0",
    help=(
        "Advanced cube-center offset in the link_tcp local frame. The "
        "default keeps the stable TCP-centered grasp; set this manually only "
        "when calibrating a finger-pocket grasp."
    ),
)
parser.add_argument(
    "--brain-terminal-servo-align-frames",
    type=int,
    default=480,
    help=(
        "Extra close-gripper frames allowed for terminal servo to align the "
        "TCP with the detected cube before a no-teleport grasp is accepted."
    ),
)
parser.add_argument(
    "--brain-place-servo",
    action="store_true",
    help=(
        "While the payload is held, servo the TCP so the cube reaches the "
        "conveyor start before releasing. This only changes robot targets."
    ),
)
parser.add_argument(
    "--brain-place-servo-frames",
    type=int,
    default=900,
    help="Extra frames allowed to align the held cube with the place target.",
)
parser.add_argument(
    "--brain-place-servo-step",
    type=float,
    default=0.055,
    help="Maximum Cartesian TCP step for held-payload place servo.",
)
parser.add_argument(
    "--brain-place-servo-max-joint-delta",
    type=float,
    default=0.08,
    help="Maximum joint delta applied by one held-payload place servo update.",
)
parser.add_argument(
    "--brain-place-servo-distance",
    type=float,
    default=0.055,
    help="Required cube XY distance from conveyor start before release.",
)
parser.add_argument(
    "--disable-brain-grasp-readiness-gate",
    action="store_true",
    help=(
        "Allow brain-control runs to close/lift without first proving the "
        "TCP is inside the configured grasp attach distance. Intended only "
        "for ablations; the default keeps the claw open until the grasp is "
        "physically reachable."
    ),
)
parser.add_argument(
    "--brain-grasp-readiness-frames",
    type=int,
    default=480,
    help=(
        "Maximum extra brain-controlled frames allowed before lift_cube to "
        "move the open gripper into grasp attach distance."
    ),
)
parser.add_argument(
    "--brain-place-servo-hover-height",
    type=float,
    default=0.18,
    help=(
        "Held cube height above the conveyor start while the place servo "
        "performs XY alignment. The cube is released only after XY alignment."
    ),
)
parser.add_argument(
    "--brain-run-report",
    type=Path,
    default=Path("outputs/vjepa2_brain_live_run.json"),
)
parser.add_argument(
    "--ui-smoke-test",
    action="store_true",
    help="Exercise Run Again, End, and language command handlers, then exit.",
)
parser.add_argument(
    "--grasp-mode",
    choices=["relative", "teleport", "physics"],
    default="relative",
    help=(
        "relative keeps the cube's pose continuous by preserving cube-to-TCP "
        "offset after contact; teleport is the old visual shortcut; physics "
        "uses no payload attachment."
    ),
)
parser.add_argument(
    "--grasp-attach-distance",
    type=float,
    default=0.025,
    help=(
        "Maximum TCP-to-cube distance allowed before the relative payload "
        "attachment is accepted. Keep this tight so the demo does not look "
        "like a remote grasp."
    ),
)
parser.add_argument(
    "--disable-link-clearance-monitor",
    action="store_true",
    help="Disable runtime robot-link to obstacle clearance checks.",
)
parser.add_argument(
    "--link-clearance-threshold",
    type=float,
    default=0.012,
    help="Minimum allowed robot-link surface clearance to obstacles in meters.",
)
parser.add_argument(
    "--link-clearance-radius",
    type=float,
    default=0.035,
    help="Conservative capsule radius used for monitored robot links.",
)
parser.add_argument(
    "--link-clearance-action",
    choices=["stop", "warn"],
    default="stop",
    help="Stop the cycle or only warn when a link clearance violation occurs.",
)
parser.add_argument(
    "--brain-clearance-intervention-margin",
    type=float,
    default=0.0,
    help=(
        "DAgger collection helper. When direct brain control brings a robot "
        "link within threshold + this margin of an obstacle, execute and "
        "record a teacher recovery instead of waiting for the monitor to stop."
    ),
)
parser.add_argument(
    "--brain-clearance-intervention-away",
    type=float,
    default=0.06,
    help="Meters to move the teacher TCP target along the clearance away vector.",
)
parser.add_argument(
    "--brain-clearance-intervention-lift",
    type=float,
    default=0.04,
    help="Meters to lift the teacher TCP target during clearance intervention.",
)
parser.add_argument(
    "--brain-clearance-intervention-phases",
    default="approach_cube,descend_to_cube,carry_to_start,place_cube",
    help="Comma-separated phases where clearance intervention may collect DAgger data.",
)
parser.add_argument(
    "--cube-clearance-threshold",
    type=float,
    default=0.004,
    help=(
        "Minimum allowed robot-to-cube clearance outside the real grasp "
        "phases. This prevents approach motions from pushing the cube."
    ),
)
parser.add_argument(
    "--cube-clearance-radius",
    type=float,
    default=0.020,
    help="Conservative radius used for robot points in cube clearance checks.",
)
parser.add_argument(
    "--cube-contact-allowed-phases",
    default="vertical_grasp,close_gripper,lift_cube,open_gripper",
    help="Comma-separated phases where close contact with the cube is expected.",
)
parser.add_argument(
    "--release-cube-clearance-grace-frames",
    type=int,
    default=45,
    help=(
        "Frames after releasing the cube where cube clearance checks are "
        "temporarily skipped so the gripper can retreat before monitoring "
        "resumes."
    ),
)
args, _ = parser.parse_known_args()


def parse_vec3(text, name):
    parts = [item.strip() for item in str(text).split(",")]
    if len(parts) != 3:
        parser.error(f"{name} must contain three comma-separated numbers")
    try:
        return tuple(float(item) for item in parts)
    except ValueError as exc:
        parser.error(f"{name} must contain numeric values: {exc}")


GRASP_CUBE_TCP_LOCAL_OFFSET = parse_vec3(
    args.grasp_cube_tcp_local_offset,
    "--grasp-cube-tcp-local-offset",
)
BASKET_CENTER_OVERRIDE = (
    None
    if args.basket_center is None
    else parse_vec3(args.basket_center, "--basket-center")
)
BASKET_RELEASE_VELOCITY_OVERRIDE = (
    None
    if args.basket_release_velocity is None
    else parse_vec3(args.basket_release_velocity, "--basket-release-velocity")
)
ballistic_policy_bundle = None

ROBOT_USD = ROOT / "assets/xarm6_gripper/xarm6_gripper.usd"
ISAAC_PYTHON = Path.home() / "isaac_sim_5.1" / "python.bat"
RUNTIME_PLANNER = ROOT / "scripts/xarm6_curobo_runtime.py"
RUNTIME_REQUEST = ROOT / "outputs/curobo_runtime_request.json"
RUNTIME_PLAN = ROOT / "outputs/curobo_runtime_plan.npz"
PHYSICS_FPS = 120
CAPTURE_FPS = 4
RECORD_CAPTURE_INTERVAL = PHYSICS_FPS // CAPTURE_FPS
brain_observe_fps = float(np.clip(float(args.brain_observe_fps), 1.0, PHYSICS_FPS))
CAPTURE_INTERVAL = max(1, int(round(PHYSICS_FPS / brain_observe_fps)))
CAPTURE_RESOLUTION = (256, 256)
MIN_EPISODE_FRAMES = 64
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
MONITORED_LINK_NAMES = [
    "link1",
    "link2",
    "link3",
    "link4",
    "link5",
    "link6",
    "link_eef",
    "xarm_gripper_base_link",
    "left_outer_knuckle",
    "left_finger",
    "left_inner_knuckle",
    "right_outer_knuckle",
    "right_finger",
    "right_inner_knuckle",
    "link_tcp",
]

app = SimulationApp(
    {
        "headless": args.headless,
        "renderer": "RaytracedLighting",
        "width": 1280,
        "height": 720,
    }
)

import carb
import omni.kit.commands
import omni.replicator.core as rep
import omni.timeline
import omni.ui as ui
import omni.usd
from PIL import Image
from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics, UsdShade
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


def basket_config_for_task(task_name, metadata):
    config = None
    if metadata.get("basket") is not None:
        config = dict(metadata["basket"])
    elif task_name == "basket_drop":
        place_position = np.asarray(metadata["place_position"], dtype=float)
        config = {
            "center": [
                float(place_position[0]),
                float(place_position[1]),
                max(0.055, float(place_position[2]) - 0.10),
            ],
            "inner_size": [0.20, 0.20],
            "wall_height": 0.08,
        }
    if config is None:
        return None
    if BASKET_CENTER_OVERRIDE is not None:
        config["center"] = np.asarray(BASKET_CENTER_OVERRIDE, dtype=float).tolist()
    if BASKET_RELEASE_VELOCITY_OVERRIDE is not None:
        config["release_velocity"] = np.asarray(
            BASKET_RELEASE_VELOCITY_OVERRIDE,
            dtype=float,
        ).tolist()
    return config


def create_basket_target(world, config):
    center = np.asarray(config["center"], dtype=float)
    inner_size = np.asarray(config.get("inner_size", [0.16, 0.16]), dtype=float)
    wall_height = float(config.get("wall_height", 0.08))
    wall_thickness = float(config.get("wall_thickness", 0.015))
    floor_thickness = float(config.get("floor_thickness", 0.012))
    floor_top_z = float(center[2])
    rim_z = floor_top_z + wall_height
    cx, cy = float(center[0]), float(center[1])
    sx, sy = float(inner_size[0]), float(inner_size[1])
    objects = []

    def add_box(name, position, scale, color):
        objects.append(
            world.scene.add(
                FixedCuboid(
                    prim_path=f"/World/Basket/{name}",
                    name=f"basket_{name.lower()}",
                    position=np.asarray(position, dtype=float),
                    scale=np.asarray(scale, dtype=float),
                    color=np.asarray(color, dtype=float),
                )
            )
        )

    add_box(
        "Floor",
        [cx, cy, floor_top_z - floor_thickness / 2.0],
        [sx + 2.0 * wall_thickness, sy + 2.0 * wall_thickness, floor_thickness],
        [0.10, 0.13, 0.16],
    )
    wall_z = floor_top_z + wall_height / 2.0
    add_box(
        "LeftWall",
        [cx - sx / 2.0 - wall_thickness / 2.0, cy, wall_z],
        [wall_thickness, sy + 2.0 * wall_thickness, wall_height],
        [0.92, 0.24, 0.05],
    )
    add_box(
        "RightWall",
        [cx + sx / 2.0 + wall_thickness / 2.0, cy, wall_z],
        [wall_thickness, sy + 2.0 * wall_thickness, wall_height],
        [0.92, 0.24, 0.05],
    )
    add_box(
        "FrontWall",
        [cx, cy + sy / 2.0 + wall_thickness / 2.0, wall_z],
        [sx + 2.0 * wall_thickness, wall_thickness, wall_height],
        [0.92, 0.24, 0.05],
    )
    add_box(
        "BackWall",
        [cx, cy - sy / 2.0 - wall_thickness / 2.0, wall_z],
        [sx + 2.0 * wall_thickness, wall_thickness, wall_height],
        [0.92, 0.24, 0.05],
    )
    rim_thickness = wall_thickness * 1.2
    add_box(
        "RimFront",
        [cx, cy + sy / 2.0 + rim_thickness / 2.0, rim_z + rim_thickness / 2.0],
        [sx + 2.0 * rim_thickness, rim_thickness, rim_thickness],
        [1.0, 0.35, 0.02],
    )
    add_box(
        "RimBack",
        [cx, cy - sy / 2.0 - rim_thickness / 2.0, rim_z + rim_thickness / 2.0],
        [sx + 2.0 * rim_thickness, rim_thickness, rim_thickness],
        [1.0, 0.35, 0.02],
    )
    add_box(
        "RimLeft",
        [cx - sx / 2.0 - rim_thickness / 2.0, cy, rim_z + rim_thickness / 2.0],
        [rim_thickness, sy + 2.0 * rim_thickness, rim_thickness],
        [1.0, 0.35, 0.02],
    )
    add_box(
        "RimRight",
        [cx + sx / 2.0 + rim_thickness / 2.0, cy, rim_z + rim_thickness / 2.0],
        [rim_thickness, sy + 2.0 * rim_thickness, rim_thickness],
        [1.0, 0.35, 0.02],
    )
    backboard_y = cy - sy / 2.0 - 0.045
    add_box(
        "Backboard",
        [cx, backboard_y, rim_z + 0.08],
        [0.26, 0.018, 0.20],
        [0.94, 0.94, 0.90],
    )
    add_box(
        "Stand",
        [cx, backboard_y - 0.025, floor_top_z / 2.0],
        [0.025, 0.025, max(floor_top_z, 0.04)],
        [0.15, 0.15, 0.16],
    )
    print(
        "basket_target_created "
        f"center={center.round(4).tolist()} "
        f"rim_z={rim_z:.4f} inner_size={inner_size.round(4).tolist()}",
        flush=True,
    )
    return objects


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


def create_rgb_recorder():
    carb.settings.get_settings().set(
        "/omni/replicator/captureOnPlay",
        False,
    )
    camera = rep.create.camera(
        position=(-0.95, -0.95, 1.18),
        look_at=(0.0, 0.0, 0.16),
        focal_length=30.0,
    )
    rep.create.light(
        rotation=(315, 0, 0),
        intensity=2500,
        light_type="distant",
    )
    rep.create.light(intensity=450, light_type="dome")
    render_product = rep.create.render_product(
        camera,
        CAPTURE_RESOLUTION,
    )
    annotator = rep.AnnotatorRegistry.get_annotator("rgb")
    annotator.attach(render_product)
    rep.orchestrator.preview()
    return annotator


class Stage3BrainRuntime:
    def __init__(
        self,
        checkpoint_path,
        device,
        local_files_only,
        blend,
        max_teacher_delta,
        max_step_delta,
        delta_gain,
        instruction_override=None,
    ):
        import torch
        from PIL import Image
        from torchvision import transforms
        from train_stage3_video_sft import Stage3Policy
        from train_stage3_direct_correction import StateConditionedStage3Policy
        from train_smolvla_policy import SmolVLAPolicy, instruction_to_hash_tokens

        self.torch = torch
        self.image_cls = Image
        self.checkpoint_path = Path(checkpoint_path).resolve()
        checkpoint = torch.load(
            self.checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        config = checkpoint["config"]
        self.phase_to_id = checkpoint["phase_to_id"]
        self.id_to_phase = {
            int(value): key for key, value in self.phase_to_id.items()
        }
        self.clip_frames = int(config["clip_frames"])
        self.image_size = int(config["image_size"])
        self.policy_arch = checkpoint.get("policy_arch", "stage3_video")
        self.action_representation = checkpoint.get(
            "action_representation", "absolute"
        )
        self.state_conditioned = self.policy_arch == "state_conditioned"
        self.smolvla = self.policy_arch == "smolvla"
        self.device = torch.device(
            device
            if device == "cpu" or torch.cuda.is_available()
            else "cpu"
        )
        self.transform = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        if self.state_conditioned or self.smolvla:
            self.state_mean = torch.tensor(
                checkpoint["state_mean"],
                dtype=torch.float32,
                device=self.device,
            )
            self.state_std = torch.tensor(
                checkpoint["state_std"],
                dtype=torch.float32,
                device=self.device,
            ).clamp_min(1e-4)
            if self.smolvla:
                instruction = config.get(
                    "instruction",
                    (
                        "Pick up the red cube and place it at the conveyor "
                        "start while avoiding obstacles."
                    ),
                )
                if instruction_override:
                    instruction = str(instruction_override)
                token_count = int(config.get("text_token_count", 16))
                text_buckets = int(config.get("text_hash_buckets", 512))
                saved_tokens = checkpoint.get("instruction_tokens")
                if instruction_override or saved_tokens is None:
                    saved_tokens = instruction_to_hash_tokens(
                        instruction,
                        text_buckets,
                        token_count,
                    )
                self.instruction_tokens = torch.as_tensor(
                    saved_tokens,
                    dtype=torch.long,
                    device=self.device,
                ).view(1, -1)
                self.model = SmolVLAPolicy(
                    config["vjepa2_model_id"],
                    int(config["embed_dim"]),
                    len(self.phase_to_id),
                    int(self.state_mean.numel()),
                    int(config.get("action_chunk_size", 4)),
                    text_buckets,
                    token_count,
                    local_files_only,
                ).to(self.device)
            else:
                self.instruction_tokens = None
                self.model = StateConditionedStage3Policy(
                    config["vjepa2_model_id"],
                    int(config["embed_dim"]),
                    len(self.phase_to_id),
                    int(self.state_mean.numel()),
                    local_files_only,
                ).to(self.device)
        else:
            self.state_mean = None
            self.state_std = None
            self.instruction_tokens = None
            self.model = Stage3Policy(
                config["vjepa2_model_id"],
                int(config["embed_dim"]),
                len(self.phase_to_id),
                local_files_only,
            ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.blend = float(np.clip(blend, 0.0, 1.0))
        self.max_teacher_delta = float(max_teacher_delta)
        self.max_step_delta = float(max_step_delta)
        self.delta_gain = max(float(delta_gain), 0.0)
        self.instruction = None if not self.smolvla else instruction
        self.frames = []
        self.physics_frame = 0
        self.latest = None
        self.stats = {
            "predictions": 0,
            "accepted": 0,
            "rejected": 0,
            "fallback": 0,
            "direct_steps": 0,
            "filtered_steps": 0,
            "observe_steps": 0,
            "phase_matches": 0,
            "phase_mismatches": 0,
            "capture_errors": 0,
        }
        self.last_observe_error = None

    def observe(self, annotator, phase, state=None):
        frame = self.physics_frame
        self.physics_frame += 1
        if frame % CAPTURE_INTERVAL:
            return
        try:
            rep.orchestrator.step(
                rt_subframes=2,
                delta_time=0.0,
                pause_timeline=False,
            )
            rgba = np.asarray(annotator.get_data())
            rgb = self.image_cls.fromarray(rgba).convert("RGB")
        except Exception as exc:
            self.stats["capture_errors"] += 1
            self.last_observe_error = str(exc)
            raise RuntimeError(f"Brain RGB capture failed: {exc}") from exc
        self.frames.append(self.transform(rgb))
        if len(self.frames) > self.clip_frames:
            self.frames = self.frames[-self.clip_frames :]
        if len(self.frames) < self.clip_frames:
            return

        with self.torch.no_grad():
            batch = (
                self.torch.stack(self.frames)
                .unsqueeze(0)
                .to(self.device)
            )
            if self.state_conditioned or self.smolvla:
                if state is None:
                    return
                model_state = np.asarray(state, dtype=float)
                expected_state_dim = int(self.state_mean.numel())
                if model_state.size < expected_state_dim:
                    model_state = np.pad(
                        model_state,
                        (0, expected_state_dim - model_state.size),
                        mode="constant",
                    )
                elif model_state.size > expected_state_dim:
                    model_state = model_state[:expected_state_dim]
                state_tensor = self.torch.tensor(
                    model_state,
                    dtype=self.torch.float32,
                    device=self.device,
                ).unsqueeze(0)
                state_tensor = (state_tensor - self.state_mean) / self.state_std
                if self.smolvla:
                    phase_logits, action_pred, chunk_pred = self.model(
                        batch,
                        state_tensor,
                        self.instruction_tokens,
                    )
                    action_chunk = (
                        chunk_pred[0].detach().cpu().numpy().astype(float)
                    )
                else:
                    phase_logits, action_pred = self.model(batch, state_tensor)
                    action_chunk = None
            else:
                phase_logits, action_pred = self.model(batch)
                action_chunk = None
            predicted_phase_id = int(phase_logits.argmax(dim=-1).item())
            action = action_pred[0].detach().cpu().numpy().astype(float)
        predicted_phase = self.id_to_phase.get(
            predicted_phase_id,
            str(predicted_phase_id),
        )
        phase_match = predicted_phase == phase
        if phase_match:
            self.stats["phase_matches"] += 1
        else:
            self.stats["phase_mismatches"] += 1
        self.stats["predictions"] += 1
        self.latest = {
            "phase": phase,
            "predicted_phase": predicted_phase,
            "predicted_phase_id": predicted_phase_id,
            "action": action,
            "action_chunk": action_chunk,
            "phase_match": phase_match,
            "state": None if state is None else np.asarray(state, dtype=float),
        }
        print(
            "brain_predict "
            f"phase={phase} predicted={predicted_phase} "
            f"match={phase_match}",
            flush=True,
        )

    def choose_target(
        self,
        mode,
        teacher_arm,
        teacher_gripper,
        current_arm,
        current_gripper,
        phase,
    ):
        if mode == "off":
            self.stats["fallback"] += 1
            return teacher_arm, teacher_gripper, "teacher"
        if self.latest is None:
            if mode == "direct" and args.brain_strict_direct:
                return current_arm, current_gripper, "warmup_hold"
            self.stats["fallback"] += 1
            return teacher_arm, teacher_gripper, "teacher"

        action = self.latest["action"]
        if len(action) < 14 or not np.all(np.isfinite(action)):
            self.stats["rejected"] += 1
            if mode == "direct" and args.brain_strict_direct:
                raise RuntimeError("Strict brain direct rejected a non-finite action")
            return teacher_arm, teacher_gripper, "reject_nonfinite"

        raw_policy_arm = np.asarray(action[-7:-1], dtype=float)
        raw_policy_gripper = float(action[-1])
        if self.action_representation in {"tcp_delta", "tcp_delta_posture"}:
            anchor_state = self.latest.get("state")
            if anchor_state is None or len(anchor_state) < 10:
                anchor_tcp, _ = current_tcp_pose()
                anchor_gripper = current_gripper
                anchor_arm = current_arm
            else:
                anchor_tcp = np.asarray(anchor_state[7:10], dtype=float)
                anchor_gripper = float(anchor_state[6])
                anchor_arm = np.asarray(anchor_state[:6], dtype=float)
            target_tcp = anchor_tcp + raw_policy_arm[:3] * self.delta_gain
            policy_arm = ik_arm_for_tcp(target_tcp, current_arm)
            if policy_arm is None:
                self.stats["rejected"] += 1
                if mode == "direct" and args.brain_strict_direct:
                    raise RuntimeError(
                        "Strict brain direct rejected an unsolved tcp_delta IK target"
                    )
                self.stats["fallback"] += 1
                return teacher_arm, teacher_gripper, "reject_tcp_delta_ik"
            if self.action_representation == "tcp_delta_posture":
                posture_blend = float(np.clip(args.brain_posture_blend, 0.0, 1.0))
                posture_target = np.asarray(policy_arm, dtype=float).copy()
                posture_target[1:4] = (
                    anchor_arm[1:4] + raw_policy_arm[3:6] * self.delta_gain
                )
                policy_arm[1:4] = (
                    (1.0 - posture_blend) * policy_arm[1:4]
                    + posture_blend * posture_target[1:4]
                )
            policy_gripper = anchor_gripper + raw_policy_gripper * self.delta_gain
        elif self.action_representation == "delta":
            anchor_state = self.latest.get("state")
            if anchor_state is None or len(anchor_state) < 7:
                anchor_arm = current_arm
                anchor_gripper = current_gripper
            else:
                anchor_arm = np.asarray(anchor_state[:6], dtype=float)
                anchor_gripper = float(anchor_state[6])
            policy_arm = anchor_arm + raw_policy_arm * self.delta_gain
            policy_gripper = anchor_gripper + raw_policy_gripper * self.delta_gain
        else:
            policy_arm = raw_policy_arm
            policy_gripper = raw_policy_gripper
        policy_gripper = float(
            np.clip(policy_gripper, OPEN_GRIPPER, CLOSED_GRIPPER)
        )
        if policy_arm.shape != (6,):
            self.stats["rejected"] += 1
            if mode == "direct" and args.brain_strict_direct:
                raise RuntimeError(
                    f"Strict brain direct rejected action shape {policy_arm.shape}"
                )
            return teacher_arm, teacher_gripper, "reject_shape"
        update_brain_path_overlay(current_arm, policy_arm)
        if mode == "observe":
            self.stats["observe_steps"] += 1
            return teacher_arm, teacher_gripper, "observe"

        teacher_delta = float(np.max(np.abs(policy_arm - teacher_arm)))
        phase_match = self.latest["predicted_phase"] == phase
        if not args.brain_allow_phase_mismatch and not phase_match:
            self.stats["rejected"] += 1
            if mode == "direct" and args.brain_strict_direct:
                raise RuntimeError(
                    "Strict brain direct phase mismatch: "
                    f"expected={phase} predicted={self.latest['predicted_phase']}"
                )
            self.stats["fallback"] += 1
            return teacher_arm, teacher_gripper, "reject_phase"
        if mode == "filtered":
            if teacher_delta > self.max_teacher_delta or not phase_match:
                self.stats["rejected"] += 1
                return teacher_arm, teacher_gripper, "reject_safety"
            target_arm = (
                (1.0 - self.blend) * teacher_arm + self.blend * policy_arm
            )
            target_gripper = (
                (1.0 - self.blend) * float(teacher_gripper)
                + self.blend * policy_gripper
            )
            self.stats["filtered_steps"] += 1
        else:
            target_arm = policy_arm
            target_gripper = policy_gripper
            self.stats["direct_steps"] += 1

        delta = np.clip(
            target_arm - current_arm,
            -self.max_step_delta,
            self.max_step_delta,
        )
        self.stats["accepted"] += 1
        return (
            current_arm + delta,
            float(
                np.clip(
                    target_gripper,
                    OPEN_GRIPPER,
                    CLOSED_GRIPPER,
                )
            ),
            mode,
        )

    def summary(self):
        latest = None
        if self.latest is not None:
            latest = {
                "phase": self.latest["phase"],
                "predicted_phase": self.latest["predicted_phase"],
                "phase_match": self.latest["phase_match"],
            }
        return {
            "checkpoint": str(self.checkpoint_path),
            "device": str(self.device),
            "policy_arch": self.policy_arch,
            "action_representation": self.action_representation,
            "instruction": self.instruction,
            "action_chunk_size": int(
                0
                if self.latest is None
                or self.latest.get("action_chunk") is None
                else len(self.latest["action_chunk"])
            ),
            "delta_gain": self.delta_gain,
            "clip_frames": self.clip_frames,
            "stats": dict(self.stats),
            "latest": latest,
            "last_observe_error": self.last_observe_error,
        }


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
task_name = (
    args.task_name
    or metadata.get("task_name")
    or ("conveyor_cycle" if not args.disable_conveyor_return else "place_task")
)
task_instruction = (
    args.task_instruction
    or metadata.get("task_instruction")
    or (
        "Pick up the red cube and place it at the conveyor start while avoiding obstacles."
        if task_name == "conveyor_cycle"
        else f"Pick up the red cube and complete task {task_name}."
    )
)

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
obstacle_objects = []
obstacle_bounds = []
for index, obstacle in enumerate(metadata["obstacles"]):
    obstacle_position = np.asarray(obstacle["position"], dtype=float)
    obstacle_dims = np.asarray(
        obstacle.get("dims", obstacle.get("scale")),
        dtype=float,
    )
    obstacle_bounds.append(
        {
            "name": obstacle.get("name", f"obstacle_{index}"),
            "center": obstacle_position,
            "half": obstacle_dims / 2.0,
        }
    )
    obstacle_objects.append(
        world.scene.add(
        FixedCuboid(
            prim_path=f"/World/Obstacles/Obstacle_{index}",
            name=f"obstacle_{index}",
            position=obstacle_position,
            scale=obstacle_dims,
            color=np.array([0.95, 0.55 - index * 0.15, 0.05]),
        )
        )
    )
basket_objects = []
basket_config = basket_config_for_task(task_name, metadata)
if basket_config is not None:
    basket_objects = create_basket_target(world, basket_config)

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

brain_path_overlay = {
    "enabled": bool(args.show_brain_ee_path),
    "points": [],
    "curve": None,
    "labels": ["now", "0.5s", "1.0s", "2.0s"],
    "horizons_s": [0.0, 0.5, 1.0, 2.0],
}


def setup_brain_path_overlay():
    if not brain_path_overlay["enabled"]:
        return
    stage = omni.usd.get_context().get_stage()
    root_path = "/World/BrainPredictionPath"
    UsdGeom.Xform.Define(stage, root_path)
    colors = [
        Gf.Vec3f(0.05, 0.95, 1.0),
        Gf.Vec3f(0.10, 0.85, 0.30),
        Gf.Vec3f(1.00, 0.85, 0.10),
        Gf.Vec3f(1.00, 0.20, 0.85),
    ]
    for index, color in enumerate(colors):
        sphere = UsdGeom.Sphere.Define(stage, f"{root_path}/Point_{index}")
        sphere.CreateRadiusAttr(0.026 if index == 0 else 0.022)
        sphere.CreateDisplayColorAttr([color])
        xform = UsdGeom.Xformable(sphere.GetPrim())
        translate = xform.AddTranslateOp()
        translate.Set(Gf.Vec3d(0.0, 0.0, -10.0))
        brain_path_overlay["points"].append(
            {"sphere": sphere, "translate": translate}
        )
    curve = UsdGeom.BasisCurves.Define(stage, f"{root_path}/PathLine")
    curve.CreateTypeAttr("linear")
    curve.CreateCurveVertexCountsAttr([len(colors)])
    curve.CreateWidthsAttr([0.014])
    curve.CreateDisplayColorAttr([Gf.Vec3f(0.05, 0.95, 1.0)])
    curve.CreatePointsAttr([Gf.Vec3f(0.0, 0.0, -10.0)] * len(colors))
    brain_path_overlay["curve"] = curve


def update_brain_path_overlay(current_arm, policy_arm):
    if not brain_path_overlay["enabled"] or not brain_path_overlay["points"]:
        return
    if policy_arm is None or not np.all(np.isfinite(policy_arm)):
        return
    current_arm = np.asarray(current_arm, dtype=float)
    policy_arm = np.asarray(policy_arm, dtype=float)
    if current_arm.shape != (6,) or policy_arm.shape != (6,):
        return
    points = []
    for horizon in brain_path_overlay["horizons_s"]:
        if horizon <= 0.0:
            predicted_arm = current_arm
        else:
            max_delta = float(args.brain_max_step_delta) * PHYSICS_FPS * float(horizon)
            predicted_arm = current_arm + np.clip(
                policy_arm - current_arm,
                -max_delta,
                max_delta,
            )
        try:
            position, _ = lula.compute_forward_kinematics(
                "link_tcp",
                predicted_arm,
            )
        except Exception:
            return
        position = np.asarray(position, dtype=float)
        if not np.all(np.isfinite(position)):
            return
        points.append(position)
    for point, overlay in zip(points, brain_path_overlay["points"]):
        overlay["translate"].Set(
            Gf.Vec3d(float(point[0]), float(point[1]), float(point[2]))
        )
    if brain_path_overlay["curve"] is not None:
        brain_path_overlay["curve"].GetPointsAttr().Set(
            [
                Gf.Vec3f(float(point[0]), float(point[1]), float(point[2]))
                for point in points
            ]
        )


setup_brain_path_overlay()


class EpisodeRecorder:
    def __init__(self, root, episode_index, annotator):
        self.root = root
        self.episode_index = episode_index
        self.annotator = annotator
        self.physics_frame = 0
        self.rows = []
        self.detected_cube_position = None
        self.final_dir = root / f"episode_{episode_index:05d}"
        self.temp_dir = root / f".episode_{episode_index:05d}.tmp"
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
        self.temp_dir.mkdir(parents=True)

    def capture(
        self,
        phase,
        arm_target,
        gripper_target,
        executed_arm=None,
        executed_gripper=None,
        safety_violation=None,
    ):
        frame = self.physics_frame
        self.physics_frame += 1
        if safety_violation is None and frame % RECORD_CAPTURE_INTERVAL:
            return
        rep.orchestrator.step(
            rt_subframes=2,
            delta_time=0.0,
            pause_timeline=False,
        )
        image_name = f"rgb_{len(self.rows):06d}.png"
        rgba = np.asarray(self.annotator.get_data())
        if rgba.ndim < 3 or rgba.shape[0] <= 0 or rgba.shape[1] <= 0:
            raise RuntimeError(f"Invalid RGB frame shape: {rgba.shape}")
        Image.fromarray(rgba).convert("RGB").save(
            self.temp_dir / image_name
        )
        cube_position, cube_orientation = cube.get_world_pose()
        tcp_position, tcp_rotation = fk_solver.compute_end_effector_pose()
        actual_positions = np.asarray(robot.get_joint_positions())
        actual_velocities = np.asarray(robot.get_joint_velocities())
        oracle_arm = np.asarray(arm_target, dtype=float)
        try:
            oracle_tcp_position, _ = lula.compute_forward_kinematics(
                "link_tcp",
                oracle_arm,
            )
            oracle_tcp_position = np.asarray(oracle_tcp_position, dtype=float)
            if not np.all(np.isfinite(oracle_tcp_position)):
                oracle_tcp_position = None
        except Exception:
            oracle_tcp_position = None
        executed_arm = (
            oracle_arm
            if executed_arm is None
            else np.asarray(executed_arm, dtype=float)
        )
        executed_gripper = (
            float(gripper_target)
            if executed_gripper is None
            else float(executed_gripper)
        )
        brain_prediction = None
        if brain_runtime is not None and brain_runtime.latest is not None:
            latest = brain_runtime.latest
            raw_action = np.asarray(latest["action"], dtype=float)
            raw_arm = raw_action[-7:-1]
            raw_gripper = float(raw_action[-1])
            predicted_tcp_target = None
            if brain_runtime.action_representation in {"tcp_delta", "tcp_delta_posture"}:
                anchor_state = latest.get("state")
                if anchor_state is None or len(anchor_state) < 10:
                    anchor_tcp = np.asarray(tcp_position, dtype=float)
                    anchor_arm = np.asarray(
                        [
                            actual_positions[dof_index[f"joint{i}"]]
                            for i in range(1, 7)
                        ],
                        dtype=float,
                    )
                    anchor_gripper = float(
                        actual_positions[dof_index[GRIPPER_JOINTS[0]]]
                    )
                else:
                    anchor_tcp = np.asarray(anchor_state[7:10], dtype=float)
                    anchor_arm = np.asarray(anchor_state[:6], dtype=float)
                    anchor_gripper = float(anchor_state[6])
                delta_gain = float(brain_runtime.delta_gain)
                predicted_tcp_target = anchor_tcp + raw_arm[:3] * delta_gain
                decoded_arm = ik_arm_for_tcp(
                    predicted_tcp_target,
                    np.asarray(
                        [
                            actual_positions[dof_index[f"joint{i}"]]
                            for i in range(1, 7)
                        ],
                        dtype=float,
                    ),
                )
                predicted_arm = (
                    np.full(6, np.nan, dtype=float)
                    if decoded_arm is None
                    else np.asarray(decoded_arm, dtype=float)
                )
                if (
                    brain_runtime.action_representation == "tcp_delta_posture"
                    and np.all(np.isfinite(predicted_arm))
                ):
                    posture_blend = float(np.clip(args.brain_posture_blend, 0.0, 1.0))
                    posture_target = predicted_arm.copy()
                    posture_target[1:4] = anchor_arm[1:4] + raw_arm[3:6] * delta_gain
                    predicted_arm[1:4] = (
                        (1.0 - posture_blend) * predicted_arm[1:4]
                        + posture_blend * posture_target[1:4]
                    )
                predicted_gripper = anchor_gripper + raw_gripper * delta_gain
            elif brain_runtime.action_representation == "delta":
                anchor_state = latest.get("state")
                if anchor_state is None or len(anchor_state) < 7:
                    anchor_arm = np.asarray(
                        [
                            actual_positions[dof_index[f"joint{i}"]]
                            for i in range(1, 7)
                        ],
                        dtype=float,
                    )
                    anchor_gripper = float(
                        actual_positions[dof_index[GRIPPER_JOINTS[0]]]
                    )
                else:
                    anchor_arm = np.asarray(anchor_state[:6], dtype=float)
                    anchor_gripper = float(anchor_state[6])
                delta_gain = float(brain_runtime.delta_gain)
                predicted_arm = anchor_arm + raw_arm * delta_gain
                predicted_gripper = anchor_gripper + raw_gripper * delta_gain
            else:
                delta_gain = 1.0
                predicted_arm = raw_arm
                predicted_gripper = raw_gripper
            brain_prediction = {
                "phase": latest["phase"],
                "predicted_phase": latest["predicted_phase"],
                "phase_match": bool(latest["phase_match"]),
                "action_representation": brain_runtime.action_representation,
                "delta_gain": delta_gain,
                "arm_joint_positions": predicted_arm.tolist(),
                "gripper_joint_position": predicted_gripper,
                "raw_arm_action": raw_arm.tolist(),
                "raw_gripper_action": raw_gripper,
                "tcp_target_position": (
                    None
                    if predicted_tcp_target is None
                    else predicted_tcp_target.tolist()
                ),
            }
        expert_correction = None
        if brain_prediction is not None:
            expert_correction = {
                "arm_delta": (
                    oracle_arm
                    - np.asarray(brain_prediction["arm_joint_positions"], dtype=float)
                ).tolist(),
                "gripper_delta": float(gripper_target)
                - float(brain_prediction["gripper_joint_position"]),
            }
        clearance_features = clearance_features_for_phase(phase)
        self.rows.append(
            {
                "frame": frame,
                "time_seconds": frame / PHYSICS_FPS,
                "phase": phase,
                "image": image_name,
                "action": {
                    "arm_joint_positions": oracle_arm.tolist(),
                    "gripper_joint_position": float(gripper_target),
                    "tcp_position": (
                        None
                        if oracle_tcp_position is None
                        else oracle_tcp_position.tolist()
                    ),
                },
                "executed_action": {
                    "arm_joint_positions": executed_arm.tolist(),
                    "gripper_joint_position": executed_gripper,
                },
                "dagger": {
                    "brain_action": brain_prediction,
                    "expert_correction": expert_correction,
                    "safety_violation": safety_violation,
                },
                "observation": {
                    "arm_joint_positions": [
                        float(actual_positions[dof_index[f"joint{i}"]])
                        for i in range(1, 7)
                    ],
                    "arm_joint_velocities": [
                        float(actual_velocities[dof_index[f"joint{i}"]])
                        for i in range(1, 7)
                    ],
                    "gripper_joint_positions": {
                        name: float(actual_positions[dof_index[name]])
                        for name in GRIPPER_JOINTS
                    },
                    "tcp_position": np.asarray(tcp_position).tolist(),
                    "tcp_rotation_matrix": np.asarray(tcp_rotation).tolist(),
                    "cube_position": np.asarray(cube_position).tolist(),
                    "cube_orientation_wxyz": np.asarray(
                        cube_orientation
                    ).tolist(),
                    "obstacles": [
                        {
                            "position": np.asarray(
                                obstacle.get_world_pose()[0]
                            ).tolist(),
                            "scale": np.asarray(
                                obstacle.get_local_scale()
                            ).tolist(),
                        }
                        for obstacle in obstacle_objects
                    ],
                    "clearance": {
                        "clearance_m": clearance_features[0],
                        "threshold_m": clearance_features[1],
                        "clearance_margin_m": clearance_features[2],
                        "kind": (
                            "cube"
                            if clearance_features[3] > 0.5
                            else "obstacle"
                        ),
                        "away_vector": clearance_features[4:7],
                    },
                },
            }
        )

    def finish(self, metrics, success=True, error=None):
        if len(self.rows) < 2 and success:
            raise RuntimeError("Episode has too few captured observations")
        violations = metrics.get("link_clearance_violations") or []
        if self.rows:
            has_labeled_violation = any(
                row.get("dagger", {}).get("safety_violation") is not None
                for row in self.rows
            )
            if violations and not has_labeled_violation:
                self.rows[-1]["dagger"]["safety_violation"] = violations[-1]
            elif not success and not has_labeled_violation:
                self.rows[-1]["dagger"]["safety_violation"] = {
                    "kind": "episode_failure",
                    "reason": None if error is None else str(error),
                    "payload_failure": metrics.get("payload_failure"),
                }
        with (self.temp_dir / "actions.jsonl").open(
            "w",
            encoding="utf-8",
        ) as stream:
            for row in self.rows:
                stream.write(json.dumps(row) + "\n")
        episode_metadata = {
            "schema_version": 1,
            "success": bool(success),
            "error": None if error is None else str(error),
            "episode_index": self.episode_index,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "physics_fps": PHYSICS_FPS,
            "capture_fps": CAPTURE_FPS,
            "capture_interval_frames": RECORD_CAPTURE_INTERVAL,
            "brain_observe_fps": brain_observe_fps,
            "brain_observe_interval_frames": CAPTURE_INTERVAL,
            "resolution": list(CAPTURE_RESOLUTION),
            "frames_written": len(self.rows),
            "minimum_episode_frames": MIN_EPISODE_FRAMES,
            "task": task_instruction,
            "task_name": task_name,
            "task_instruction": task_instruction,
            "teacher_policy": "cuRobo dynamic collision-aware planning",
            "executed_policy": args.brain_control,
            "dagger_oracle_actions": True,
            "detected_cube_position": self.detected_cube_position,
            "metrics": metrics,
        }
        (self.temp_dir / "metadata.json").write_text(
            json.dumps(episode_metadata, indent=2),
            encoding="utf-8",
        )
        if self.final_dir.exists():
            raise FileExistsError(
                f"Episode already exists: {self.final_dir}"
            )
        self.temp_dir.replace(self.final_dir)
        return self.final_dir

    def abort(self):
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

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

record_root = args.record_root.resolve() if args.record_root else None
if args.episodes is not None and record_root is None:
    parser.error("--episodes requires --record-root")
if record_root is not None:
    record_root.mkdir(parents=True, exist_ok=True)
    existing_episode_indices = sorted(
        int(path.name.split("_")[1])
        for path in record_root.glob("episode_[0-9][0-9][0-9][0-9][0-9]")
        if (path / "metadata.json").exists()
    )
    existing_successful_episodes = 0
    for episode_path in record_root.glob(
        "episode_[0-9][0-9][0-9][0-9][0-9]"
    ):
        metadata_file = episode_path / "metadata.json"
        if not metadata_file.exists():
            continue
        try:
            episode_metadata = json.loads(metadata_file.read_text())
        except json.JSONDecodeError:
            continue
        if episode_metadata.get("success", True):
            existing_successful_episodes += 1
else:
    existing_episode_indices = []
    existing_successful_episodes = 0
next_episode_index = (
    existing_episode_indices[-1] + 1
    if existing_episode_indices
    else 0
)
target_episode_total = (
    max(args.episodes, 1)
    if args.episodes is not None
    else None
)
brain_enabled = args.brain_control != "off"
if brain_enabled and not args.brain_policy.resolve().exists():
    raise FileNotFoundError(f"Brain policy not found: {args.brain_policy}")
needs_rgb_annotator = record_root is not None or (
    brain_enabled and not args.brain_terminal_servo_without_vision
)
rgb_annotator = create_rgb_recorder() if needs_rgb_annotator else None
brain_runtime = (
    Stage3BrainRuntime(
        args.brain_policy,
        args.brain_device,
        args.brain_local_files_only,
        args.brain_blend,
        args.brain_max_teacher_delta,
        args.brain_max_step_delta,
        args.brain_delta_gain,
        task_instruction,
    )
    if brain_enabled
    else None
)
if brain_runtime is not None:
    print(
        "brain_runtime_ready "
        f"mode={args.brain_control} "
        f"checkpoint={brain_runtime.checkpoint_path}",
        flush=True,
    )


def should_render_world():
    return (not args.headless) or rgb_annotator is not None

timeline = omni.timeline.get_timeline_interface()
timeline.play()
control_state = {
    "retry_requested": False,
    "end_requested": False,
    "last_command": "",
}
status_label = None
command_field = None
command_feedback_label = None


def app_allows_automation_step():
    return bool(args.headless or app.is_running())


def warmup_simulation_app(frames=90):
    for _ in range(max(int(frames), 0)):
        if not app_allows_automation_step():
            return False
        app.update()
    return True


def set_command_feedback(message):
    print(f"ui_command {message}", flush=True)
    if command_feedback_label is not None:
        command_feedback_label.text = message


def request_retry(source="button"):
    control_state["retry_requested"] = True
    control_state["last_command"] = source
    set_command_feedback(f"Run Again requested by {source}")


def request_end(source="button"):
    control_state["end_requested"] = True
    control_state["last_command"] = source
    set_command_feedback(f"End requested by {source}")


def handle_language_command(command_text):
    normalized = command_text.strip().lower()
    if not normalized:
        set_command_feedback("Type: run again, end, or status")
        return
    if any(
        phrase in normalized
        for phrase in {
            "run again",
            "retry",
            "restart",
            "reset",
            "重新",
            "再跑",
            "重跑",
            "重新測試",
        }
    ):
        request_retry("language")
        return
    if any(
        phrase in normalized
        for phrase in {"end", "stop", "quit", "exit", "結束", "停止", "關閉"}
    ):
        request_end("language")
        return
    if any(phrase in normalized for phrase in {"status", "狀態", "進度"}):
        message = (
            f"Status: brain={args.brain_control}, "
            f"retry={control_state['retry_requested']}, "
            f"end={control_state['end_requested']}"
        )
        set_command_feedback(message)
        if status_label is not None:
            status_label.text = message
        return
    if any(phrase in normalized for phrase in {"help", "指令", "幫助"}):
        set_command_feedback("Commands: run again / end / status")
        return
    set_command_feedback(f"Unknown command: {command_text}")


def submit_language_command():
    if command_field is None:
        return
    command_text = command_field.model.get_value_as_string()
    handle_language_command(command_text)
    command_field.model.set_value("")


def run_ui_smoke_test():
    handle_language_command("status")
    if control_state["retry_requested"] or control_state["end_requested"]:
        raise RuntimeError("status command changed control flags")
    request_retry("ui_smoke_test")
    if not control_state["retry_requested"]:
        raise RuntimeError("Run Again handler did not set retry_requested")
    control_state["retry_requested"] = False
    handle_language_command("run again")
    if not control_state["retry_requested"]:
        raise RuntimeError("language run again did not set retry_requested")
    control_state["retry_requested"] = False
    request_end("ui_smoke_test")
    if not control_state["end_requested"]:
        raise RuntimeError("End handler did not set end_requested")
    control_state["end_requested"] = False
    handle_language_command("end")
    if not control_state["end_requested"]:
        raise RuntimeError("language end did not set end_requested")
    print("ui_smoke_test_passed", flush=True)


if not args.headless:
    window = ui.Window("cuRobo Conveyor Cycle", width=420, height=210)
    with window.frame:
        with ui.VStack(spacing=8):
            status_label = ui.Label("Starting...")
            with ui.HStack(spacing=8):
                ui.Button(
                    "Run Again",
                    height=38,
                    clicked_fn=lambda: request_retry("button"),
                )
                ui.Button(
                    "End",
                    height=38,
                    clicked_fn=lambda: request_end("button"),
                )
            ui.Label("Command")
            with ui.HStack(spacing=8):
                command_field = ui.StringField(height=32)
                ui.Button("Send", width=80, height=32, clicked_fn=submit_language_command)
            command_feedback_label = ui.Label("Commands: run again / end / status")

if args.ui_smoke_test:
    run_ui_smoke_test()
    app.close()
    raise SystemExit(0)


payload_state = {
    "held": False,
    "offset": np.zeros(3),
    "local_offset": np.asarray(GRASP_CUBE_TCP_LOCAL_OFFSET, dtype=float),
    "attach_distance_m": None,
    "max_cube_height_m": float(cube_initial_position[2]),
    "teleport_shortcut_used": False,
    "last_failure": None,
    "release_clearance_grace_frames": 0,
    "motion_samples": [],
    "release_velocity_mps": None,
    "release_velocity_source": None,
    "gripper_throw_velocity_mps": None,
    "basket_release_arm_positions": None,
    "basket_landed": False,
}
grasp_readiness_state = {
    "ready": False,
    "best_distance_m": None,
    "last_distance_m": None,
}
clearance_state = {
    "enabled": not args.disable_link_clearance_monitor,
    "threshold_m": float(args.link_clearance_threshold),
    "link_radius_m": float(args.link_clearance_radius),
    "cube_threshold_m": float(args.cube_clearance_threshold),
    "cube_radius_m": float(args.cube_clearance_radius),
    "cube_contact_allowed_phases": [
        phase.strip()
        for phase in str(args.cube_contact_allowed_phases).split(",")
        if phase.strip()
    ],
    "action": args.link_clearance_action,
    "monitored_links": [],
    "min_clearance_m": None,
    "min_cube_clearance_m": None,
    "last_violation": None,
    "violations": [],
}
recording_state = {"recorder": None}
live_run_state = {"cycles": [], "failures": []}
terminal_servo_state = {
    "enabled": bool(args.brain_terminal_servo),
    "attempts": 0,
    "successes": 0,
    "failures": 0,
    "skipped": 0,
    "min_tcp_cube_distance_m": None,
    "last_error": None,
}
place_servo_state = {
    "enabled": bool(args.brain_place_servo or args.brain_terminal_servo),
    "attempts": 0,
    "successes": 0,
    "failures": 0,
    "skipped": 0,
    "min_cube_place_distance_m": None,
    "last_error": None,
}
terminal_servo_phases = {
    phase.strip()
    for phase in args.brain_terminal_servo_phases.split(",")
    if phase.strip()
}


def update_dataset_manifest(failed_attempts=0):
    if record_root is None:
        return
    completed = 0
    saved_failed = 0
    for path in record_root.glob("episode_[0-9][0-9][0-9][0-9][0-9]"):
        metadata_file = path / "metadata.json"
        if not metadata_file.exists():
            continue
        try:
            episode_metadata = json.loads(metadata_file.read_text())
        except json.JSONDecodeError:
            continue
        if episode_metadata.get("success", True):
            completed += 1
        else:
            saved_failed += 1
    manifest = {
        "schema_version": 1,
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "requested_episodes": target_episode_total,
        "completed_episodes": completed,
        "saved_failed_episodes": saved_failed,
        "failed_attempts_this_run": failed_attempts,
        "capture_fps": CAPTURE_FPS,
        "resolution": list(CAPTURE_RESOLUTION),
        "teacher_policy": "cuRobo dynamic collision-aware planning",
        "brain_control": args.brain_control,
        "task_name": task_name,
        "task_instruction": task_instruction,
    }
    (record_root / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )


def write_brain_run_report():
    if brain_runtime is None:
        return
    report_path = args.brain_run_report.resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "brain_control": args.brain_control,
        "brain_policy": str(args.brain_policy.resolve()),
        "task_name": task_name,
        "task_instruction": task_instruction,
        "conveyor_speed_mps": args.conveyor_speed,
        "cycles": live_run_state["cycles"],
        "failures": live_run_state["failures"],
        "brain_runtime": brain_runtime.summary(),
        "terminal_servo": terminal_servo_state,
        "place_servo": place_servo_state,
        "link_clearance": clearance_state,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def current_brain_state(phase=None):
    positions = np.asarray(robot.get_joint_positions(), dtype=float)
    arm_state = [
        float(positions[dof_index[f"joint{i}"]])
        for i in range(1, 7)
    ]
    gripper_state = float(positions[dof_index[GRIPPER_JOINTS[0]]])
    tcp_position, _ = fk_solver.compute_end_effector_pose()
    cube_position = cube.get_world_pose()[0]
    clearance = clearance_features_for_phase(phase or "unknown")
    return (
        arm_state
        + [gripper_state]
        + np.asarray(tcp_position, dtype=float).tolist()
        + np.asarray(cube_position, dtype=float).tolist()
        + clearance
    )


def record_brain_failure(cycle_number, error):
    if brain_runtime is None:
        return
    failure = {
        "cycle": int(cycle_number),
        "error": str(error),
        "payload_failure": payload_state["last_failure"],
        "grasp_mode": args.grasp_mode,
        "grasp_attach_distance_m": payload_state["attach_distance_m"],
        "max_cube_height_m": float(payload_state["max_cube_height_m"]),
        "teleport_shortcut_used": payload_state["teleport_shortcut_used"],
        "minimum_robot_obstacle_clearance_m": clearance_state[
            "min_clearance_m"
        ],
        "minimum_robot_cube_clearance_m": clearance_state[
            "min_cube_clearance_m"
        ],
        "link_clearance_violation": clearance_state["last_violation"],
        "brain_runtime": brain_runtime.summary(),
    }
    live_run_state["failures"].append(failure)
    write_brain_run_report()


def capture_step(
    phase,
    arm_target,
    gripper_target,
    executed_arm=None,
    executed_gripper=None,
    safety_violation=None,
):
    recorder = recording_state["recorder"]
    if recorder is not None:
        try:
            recorder.capture(
                phase,
                arm_target,
                gripper_target,
                executed_arm,
                executed_gripper,
                safety_violation,
            )
        except KeyboardInterrupt:
            raise
        except BaseException as exc:
            raise RuntimeError(
                f"Episode RGB/action capture failed during {phase}: {exc}"
            ) from exc
    if brain_runtime is not None and rgb_annotator is not None:
        try:
            brain_runtime.observe(rgb_annotator, phase, current_brain_state(phase))
        except KeyboardInterrupt:
            raise
        except BaseException as exc:
            raise RuntimeError(
                f"Brain online observation failed during {phase}: {exc}"
            ) from exc


def _ik_arm_positions(action, current_arm):
    joint_positions = getattr(action, "joint_positions", None)
    if joint_positions is None:
        return None
    joint_positions = np.asarray(joint_positions, dtype=float)
    if joint_positions.ndim > 1:
        joint_positions = joint_positions.reshape(-1)
    if joint_positions.size >= len(robot.dof_names):
        return np.asarray(
            [joint_positions[dof_index[f"joint{i}"]] for i in range(1, 7)],
            dtype=float,
        )
    if joint_positions.size >= 6:
        return joint_positions[:6].astype(float)
    return None


def current_tcp_pose():
    tcp_position, tcp_rotation = fk_solver.compute_end_effector_pose()
    tcp_position = np.asarray(tcp_position, dtype=float)
    tcp_rotation = np.asarray(tcp_rotation, dtype=float)
    if tcp_rotation.size == 9:
        tcp_rotation = tcp_rotation.reshape(3, 3)
    else:
        tcp_rotation = np.eye(3)
    return tcp_position, tcp_rotation


def local_grasp_offset_world(tcp_rotation=None):
    if tcp_rotation is None:
        _, tcp_rotation = current_tcp_pose()
    return np.asarray(tcp_rotation, dtype=float) @ np.asarray(
        GRASP_CUBE_TCP_LOCAL_OFFSET,
        dtype=float,
    )


def grasp_center_position():
    tcp_position, tcp_rotation = current_tcp_pose()
    return tcp_position + local_grasp_offset_world(tcp_rotation)


def outward_grasp_offset(cube_position):
    offset = float(args.grasp_outward_offset)
    if abs(offset) <= 1e-9:
        return np.zeros(3, dtype=float)
    cube_position = np.asarray(cube_position, dtype=float)
    direction_xy = cube_position[:2].copy()
    norm = float(np.linalg.norm(direction_xy))
    if not np.isfinite(norm) or norm <= 1e-6:
        return np.zeros(3, dtype=float)
    direction_xy /= norm
    return np.array(
        [direction_xy[0] * offset, direction_xy[1] * offset, 0.0],
        dtype=float,
    )


def grasp_target_for_cube(cube_position):
    cube_position = np.asarray(cube_position, dtype=float)
    return (
        cube_position
        + np.array([0.0, 0.0, args.brain_terminal_servo_z_offset], dtype=float)
        + outward_grasp_offset(cube_position)
    )


def find_monitored_link_prims():
    stage = omni.usd.get_context().get_stage()
    by_name = {}
    for prim in stage.Traverse():
        name = prim.GetName()
        if name in MONITORED_LINK_NAMES and prim.IsValid():
            by_name.setdefault(name, prim.GetPath())
    ordered = [
        {"name": name, "path": by_name[name]}
        for name in MONITORED_LINK_NAMES
        if name in by_name
    ]
    clearance_state["monitored_links"] = [
        f"{item['name']}:{item['path']}" for item in ordered
    ]
    print(
        "link_clearance_monitor "
        f"enabled={clearance_state['enabled']} "
        f"links={len(ordered)} "
        f"threshold={clearance_state['threshold_m']:.4f}m",
        flush=True,
    )
    return ordered


monitored_link_prims = []


def link_world_points():
    global monitored_link_prims
    if not clearance_state["enabled"]:
        return []
    if not monitored_link_prims:
        monitored_link_prims = find_monitored_link_prims()
    if not monitored_link_prims:
        return []
    stage = omni.usd.get_context().get_stage()
    cache = UsdGeom.XformCache()
    points = []
    for item in monitored_link_prims:
        prim = stage.GetPrimAtPath(item["path"])
        if not prim or not prim.IsValid():
            continue
        matrix = cache.GetLocalToWorldTransform(prim)
        translation = matrix.ExtractTranslation()
        points.append(
            {
                "name": item["name"],
                "position": np.array(
                    [translation[0], translation[1], translation[2]],
                    dtype=float,
                ),
            }
        )
    return points


def point_to_aabb_distance(point, center, half):
    point = np.asarray(point, dtype=float)
    center = np.asarray(center, dtype=float)
    half = np.asarray(half, dtype=float)
    outside = np.maximum(np.abs(point - center) - half, 0.0)
    return float(np.linalg.norm(outside))


def point_to_aabb_clearance_direction(point, center, half):
    point = np.asarray(point, dtype=float)
    center = np.asarray(center, dtype=float)
    half = np.asarray(half, dtype=float)
    delta = point - center
    closest = center + np.clip(delta, -half, half)
    away = point - closest
    norm = float(np.linalg.norm(away))
    if norm > 1e-6:
        return float(norm), (away / norm).astype(float)

    # If the point is inside the box, point toward the nearest exit face.
    penetration = half - np.abs(delta)
    axis = int(np.argmin(penetration))
    direction = np.zeros(3, dtype=float)
    direction[axis] = 1.0 if delta[axis] >= 0.0 else -1.0
    return 0.0, direction


def sampled_segment_points(start, end, samples=7):
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    for alpha in np.linspace(0.0, 1.0, max(int(samples), 2)):
        yield start * (1.0 - alpha) + end * alpha


def dynamic_clearance_bounds(phase):
    bounds = list(obstacle_bounds)
    allowed_cube_phases = set(clearance_state["cube_contact_allowed_phases"])
    release_grace_active = payload_state.get("release_clearance_grace_frames", 0) > 0
    if (
        phase not in allowed_cube_phases
        and not payload_state["held"]
        and not release_grace_active
        and "grasp" not in str(phase)
    ):
        cube_position = np.asarray(cube.get_world_pose()[0], dtype=float)
        cube_half = np.ones(3, dtype=float) * (float(metadata["cube_size"]) / 2.0)
        bounds.append(
            {
                "name": "cycle_cube",
                "center": cube_position,
                "half": cube_half,
                "threshold_m": float(clearance_state["cube_threshold_m"]),
                "radius_m": float(clearance_state["cube_radius_m"]),
                "kind": "cube",
            }
        )
    return bounds


def best_link_clearance(phase):
    if not clearance_state["enabled"]:
        return {
            "clearance_m": 1.0,
            "link": None,
            "obstacle": None,
            "phase": phase,
            "threshold_m": float(clearance_state["threshold_m"]),
            "kind": "obstacle",
            "away_vector": [0.0, 0.0, 0.0],
        }
    points = link_world_points()
    if not points:
        return {
            "clearance_m": 1.0,
            "link": None,
            "obstacle": None,
            "phase": phase,
            "threshold_m": float(clearance_state["threshold_m"]),
            "kind": "obstacle",
            "away_vector": [0.0, 0.0, 0.0],
        }
    bounds = dynamic_clearance_bounds(phase)
    if not bounds:
        return {
            "clearance_m": 1.0,
            "link": None,
            "obstacle": None,
            "phase": phase,
            "threshold_m": float(clearance_state["threshold_m"]),
            "kind": "obstacle",
            "away_vector": [0.0, 0.0, 0.0],
        }
    best = {
        "clearance_m": float("inf"),
        "link": None,
        "obstacle": None,
        "phase": phase,
        "threshold_m": float(clearance_state["threshold_m"]),
        "kind": "obstacle",
        "away_vector": [0.0, 0.0, 0.0],
    }
    candidates = [(item["name"], item["position"]) for item in points]
    for start, end in zip(points, points[1:]):
        segment_name = f"{start['name']}->{end['name']}"
        for sampled in sampled_segment_points(start["position"], end["position"]):
            candidates.append((segment_name, sampled))
    for link_name, point in candidates:
        for obstacle in bounds:
            radius = float(obstacle.get("radius_m", clearance_state["link_radius_m"]))
            threshold = float(
                obstacle.get("threshold_m", clearance_state["threshold_m"])
            )
            distance, away_vector = point_to_aabb_clearance_direction(
                point,
                obstacle["center"],
                obstacle["half"],
            )
            clearance = distance - radius
            if clearance < best["clearance_m"]:
                best = {
                    "clearance_m": float(clearance),
                    "link": link_name,
                    "obstacle": obstacle["name"],
                    "phase": phase,
                    "threshold_m": threshold,
                    "kind": obstacle.get("kind", "obstacle"),
                    "away_vector": away_vector.tolist(),
                }
    return best


def clearance_features_for_phase(phase):
    best = best_link_clearance(phase)
    clearance_m = float(best.get("clearance_m", 1.0))
    threshold_m = float(best.get("threshold_m", clearance_state["threshold_m"]))
    away_vector = np.asarray(best.get("away_vector", [0.0, 0.0, 0.0]), dtype=float)
    if away_vector.size != 3 or not np.all(np.isfinite(away_vector)):
        away_vector = np.zeros(3, dtype=float)
    return [
        clearance_m,
        threshold_m,
        clearance_m - threshold_m,
        1.0 if best.get("kind") == "cube" else 0.0,
        float(away_vector[0]),
        float(away_vector[1]),
        float(away_vector[2]),
    ]


def check_link_clearance(phase):
    if not clearance_state["enabled"]:
        return
    best = best_link_clearance(phase)
    min_clearance = clearance_state["min_clearance_m"]
    if min_clearance is None or best["clearance_m"] < float(min_clearance):
        clearance_state["min_clearance_m"] = best["clearance_m"]
    if best["kind"] == "cube":
        min_cube_clearance = clearance_state["min_cube_clearance_m"]
        if min_cube_clearance is None or best["clearance_m"] < float(
            min_cube_clearance
        ):
            clearance_state["min_cube_clearance_m"] = best["clearance_m"]
    if best["clearance_m"] >= best["threshold_m"]:
        return
    violation = {
        "phase": phase,
        "link": best["link"],
        "obstacle": best["obstacle"],
        "clearance_m": best["clearance_m"],
        "threshold_m": best["threshold_m"],
        "kind": best["kind"],
    }
    clearance_state["last_violation"] = violation
    clearance_state["violations"].append(violation)
    print(
        "link_clearance_violation "
        f"phase={phase} link={violation['link']} "
        f"obstacle={violation['obstacle']} "
        f"kind={violation['kind']} "
        f"clearance={violation['clearance_m']:.4f}m "
        f"threshold={violation['threshold_m']:.4f}m",
        flush=True,
    )
    if clearance_state["action"] == "stop":
        raise RuntimeError(
            "Robot link clearance violation: "
            f"{violation['link']} to {violation['obstacle']} "
            f"clearance={violation['clearance_m']:.4f}m "
            f"threshold={violation['threshold_m']:.4f}m"
        )


def guarded_world_step(phase, render=None):
    if render is None:
        render = should_render_world()
    world.step(render=render)
    check_link_clearance(phase)
    if payload_state.get("release_clearance_grace_frames", 0) > 0:
        payload_state["release_clearance_grace_frames"] -= 1


def terminal_servo_target(target_arm, target_gripper, phase, current_arm):
    if not args.brain_terminal_servo:
        return target_arm, target_gripper, False
    if args.brain_control != "direct":
        terminal_servo_state["skipped"] += 1
        return target_arm, target_gripper, False
    if phase not in terminal_servo_phases:
        terminal_servo_state["skipped"] += 1
        return target_arm, target_gripper, False

    tcp_position, tcp_rotation = current_tcp_pose()
    cube_position = np.asarray(cube.get_world_pose()[0], dtype=float)
    cube_target = grasp_target_for_cube(cube_position)
    target_tcp = cube_target - local_grasp_offset_world(tcp_rotation)
    error = target_tcp - tcp_position
    grasp_distance = float(np.linalg.norm(cube_target - grasp_center_position()))
    distance = float(np.linalg.norm(error))
    min_distance = terminal_servo_state["min_tcp_cube_distance_m"]
    if min_distance is None or grasp_distance < float(min_distance):
        terminal_servo_state["min_tcp_cube_distance_m"] = grasp_distance
    if (
        not np.isfinite(distance)
        or distance <= 1e-6
        or distance > args.brain_terminal_servo_radius
    ):
        terminal_servo_state["skipped"] += 1
        return target_arm, target_gripper, False

    terminal_servo_state["attempts"] += 1
    step_length = min(float(args.brain_terminal_servo_step), distance)
    servo_position = tcp_position + error / distance * step_length
    try:
        ik_action, success = fk_solver.compute_inverse_kinematics(
            target_position=servo_position,
        )
    except TypeError:
        ik_action, success = fk_solver.compute_inverse_kinematics(servo_position)
    except Exception as exc:
        terminal_servo_state["failures"] += 1
        terminal_servo_state["last_error"] = str(exc)
        return target_arm, target_gripper, False

    if not success:
        terminal_servo_state["failures"] += 1
        terminal_servo_state["last_error"] = "ik_unsolved"
        return target_arm, target_gripper, False

    ik_arm = _ik_arm_positions(ik_action, current_arm)
    if ik_arm is None or not np.all(np.isfinite(ik_arm)):
        terminal_servo_state["failures"] += 1
        terminal_servo_state["last_error"] = "ik_action_has_no_arm_positions"
        return target_arm, target_gripper, False

    joint_delta = np.clip(
        ik_arm - current_arm,
        -float(args.brain_terminal_servo_max_joint_delta),
        float(args.brain_terminal_servo_max_joint_delta),
    )
    terminal_servo_state["successes"] += 1
    terminal_servo_state["last_error"] = None
    if terminal_servo_state["successes"] % CAPTURE_INTERVAL == 1:
        print(
            "terminal_servo "
            f"phase={phase} grasp_cube_distance={grasp_distance:.4f}m "
            f"cart_step={step_length:.4f}m",
            flush=True,
        )
    return current_arm + joint_delta, target_gripper, True


def cube_to_place_distance():
    cube_position = np.asarray(cube.get_world_pose()[0], dtype=float)
    place_position = np.asarray(metadata["place_position"], dtype=float)
    return float(np.linalg.norm(cube_position[:2] - place_position[:2]))


def task_success_position():
    if task_name == "basket_drop" and basket_config is not None:
        return np.asarray(
            basket_config.get("center", metadata["place_position"]),
            dtype=float,
        )
    return np.asarray(metadata["place_position"], dtype=float)


def basket_distance_for_position(position):
    if task_name != "basket_drop" or basket_config is None:
        return None
    basket_center = np.asarray(
        basket_config.get("center", metadata["place_position"]),
        dtype=float,
    )
    position = np.asarray(position, dtype=float)
    return float(np.linalg.norm(position[:2] - basket_center[:2]))


def basket_success_distance():
    return float(args.basket_success_distance)


def basket_landing_position():
    if task_name != "basket_drop" or basket_config is None:
        return None
    center = np.asarray(
        basket_config.get("center", metadata["place_position"]),
        dtype=float,
    )
    landing = center.copy()
    landing[2] = float(center[2]) + float(metadata["cube_size"]) / 2.0 + 0.005
    return landing


def analytic_basket_release_velocity(release_position, basket_center):
    release_position = np.asarray(release_position, dtype=float)
    basket_center = np.asarray(basket_center, dtype=float)
    delta = basket_center - release_position
    horizontal_distance = float(np.linalg.norm(delta[:2]))
    if args.basket_analytic_flight_time is None:
        flight_time = float(np.clip(0.32 + 0.25 * horizontal_distance, 0.34, 0.52))
    else:
        flight_time = max(float(args.basket_analytic_flight_time), 0.05)
    return np.asarray(
        [
            delta[0] / flight_time,
            delta[1] / flight_time,
            (delta[2] + 0.5 * 9.81 * flight_time * flight_time) / flight_time,
        ],
        dtype=float,
    )


def basket_release_velocity_for(release_position):
    global ballistic_policy_bundle
    if task_name != "basket_drop" or basket_config is None:
        return None, None
    basket_center = np.asarray(
        basket_config.get("center", metadata["place_position"]),
        dtype=float,
    )
    if args.basket_velocity_mode == "gripper":
        release_velocity = payload_state.get("gripper_throw_velocity_mps")
        if release_velocity is None:
            release_velocity = inherited_payload_velocity()
        if release_velocity is None:
            return None, None
        return np.asarray(release_velocity, dtype=float), "gripper_motion"
    if args.basket_velocity_mode == "metadata":
        release_velocity = basket_config.get("release_velocity")
        if release_velocity is None:
            return None, None
        return np.asarray(release_velocity, dtype=float), "metadata"
    if args.basket_velocity_mode == "analytic":
        return (
            analytic_basket_release_velocity(release_position, basket_center),
            "analytic",
        )
    if ballistic_policy_bundle is None:
        from train_ballistic_throw_policy import (
            load_ballistic_policy,
            predict_release_velocity,
        )

        if not args.basket_release_policy.exists():
            raise FileNotFoundError(
                "Missing basket release policy checkpoint: "
                f"{args.basket_release_policy}. Run "
                "scripts/train_ballistic_throw_policy.py first."
            )
        device = args.brain_device if args.brain_device == "cpu" else "cuda"
        ballistic_policy_bundle = (
            load_ballistic_policy(args.basket_release_policy, device=device),
            predict_release_velocity,
        )
        print(
            f"basket_release_policy_loaded={args.basket_release_policy}",
            flush=True,
        )
    model_bundle, predict_release_velocity = ballistic_policy_bundle
    return (
        np.asarray(
            predict_release_velocity(
                model_bundle,
                release_position,
                basket_center,
            ),
            dtype=float,
        ),
        "model",
    )


def place_servo_target(target_arm, target_gripper, phase, current_arm):
    if not (args.brain_place_servo or args.brain_terminal_servo):
        return target_arm, target_gripper, False
    if args.brain_control != "direct":
        place_servo_state["skipped"] += 1
        return target_arm, target_gripper, False
    if not payload_state["held"] or phase not in {"place_cube", "open_gripper"}:
        place_servo_state["skipped"] += 1
        return target_arm, target_gripper, False

    cube_position = np.asarray(cube.get_world_pose()[0], dtype=float)
    place_position = np.asarray(metadata["place_position"], dtype=float)
    place_error_xy = place_position[:2] - cube_position[:2]
    place_distance = float(np.linalg.norm(place_error_xy))
    min_distance = place_servo_state["min_cube_place_distance_m"]
    if min_distance is None or place_distance < float(min_distance):
        place_servo_state["min_cube_place_distance_m"] = place_distance
    if not np.isfinite(place_distance) or place_distance <= float(
        args.brain_place_servo_distance
    ):
        place_servo_state["skipped"] += 1
        return target_arm, target_gripper, False

    tcp_position, tcp_rotation = current_tcp_pose()
    desired_cube_position = place_position.copy()
    desired_cube_position[2] = max(
        float(cube_position[2]),
        float(place_position[2]) + float(args.brain_place_servo_hover_height),
    )
    held_offset = np.asarray(payload_state["offset"], dtype=float)
    desired_tcp = desired_cube_position - held_offset
    error = desired_tcp - tcp_position
    distance = float(np.linalg.norm(error))
    if not np.isfinite(distance) or distance <= 1e-6:
        place_servo_state["skipped"] += 1
        return target_arm, target_gripper, False

    place_servo_state["attempts"] += 1
    step_length = min(float(args.brain_place_servo_step), distance)
    servo_position = tcp_position + error / distance * step_length
    try:
        ik_action, success = fk_solver.compute_inverse_kinematics(
            target_position=servo_position,
        )
    except TypeError:
        ik_action, success = fk_solver.compute_inverse_kinematics(servo_position)
    except Exception as exc:
        place_servo_state["failures"] += 1
        place_servo_state["last_error"] = str(exc)
        return target_arm, target_gripper, False

    if not success:
        place_servo_state["failures"] += 1
        place_servo_state["last_error"] = "ik_unsolved"
        return target_arm, target_gripper, False

    ik_arm = _ik_arm_positions(ik_action, current_arm)
    if ik_arm is None or not np.all(np.isfinite(ik_arm)):
        place_servo_state["failures"] += 1
        place_servo_state["last_error"] = "ik_action_has_no_arm_positions"
        return target_arm, target_gripper, False

    joint_delta = np.clip(
        ik_arm - current_arm,
        -float(args.brain_place_servo_max_joint_delta),
        float(args.brain_place_servo_max_joint_delta),
    )
    place_servo_state["successes"] += 1
    place_servo_state["last_error"] = None
    if place_servo_state["successes"] % CAPTURE_INTERVAL == 1:
        print(
            "place_servo "
            f"phase={phase} cube_place_distance={place_distance:.4f}m "
            f"cart_step={step_length:.4f}m",
            flush=True,
        )
    return current_arm + joint_delta, target_gripper, True


def choose_control_target(teacher_arm, teacher_gripper, phase):
    teacher_arm = np.asarray(teacher_arm, dtype=float)
    current_positions = np.asarray(robot.get_joint_positions(), dtype=float)
    current_arm = np.asarray(
        [current_positions[dof_index[f"joint{i}"]] for i in range(1, 7)],
        dtype=float,
    )
    current_gripper = float(current_positions[dof_index[GRIPPER_JOINTS[0]]])
    if (
        brain_runtime is not None
        and not args.brain_terminal_servo_without_vision
    ):
        try:
            target_arm, target_gripper, source = brain_runtime.choose_target(
                args.brain_control,
                teacher_arm,
                float(teacher_gripper),
                current_arm,
                current_gripper,
                phase,
            )
        except RuntimeError as exc:
            recorder = recording_state["recorder"]
            if recorder is not None:
                recorder.capture(
                    phase,
                    teacher_arm,
                    float(teacher_gripper),
                    current_arm,
                    current_gripper,
                    {
                        "kind": "policy_rejection",
                        "phase": phase,
                        "reason": str(exc),
                    },
                )
            raise
        if (
            source in {"filtered", "direct"}
            and brain_runtime.stats["accepted"] % CAPTURE_INTERVAL == 0
        ):
            print(
                f"brain_control source={source} phase={phase} "
                f"target={np.asarray(target_arm).round(4).tolist()} "
                f"gripper={target_gripper:.3f}",
                flush=True,
            )
        intervention_phases = {
            item.strip()
            for item in str(args.brain_clearance_intervention_phases).split(",")
            if item.strip()
        }
        if (
            args.brain_control == "direct"
            and args.brain_clearance_intervention_margin > 0.0
            and source in {"direct", "filtered"}
            and phase in intervention_phases
        ):
            best = best_link_clearance(phase)
            trigger_clearance = (
                float(best["threshold_m"])
                + float(args.brain_clearance_intervention_margin)
            )
            if best["kind"] == "obstacle" and best["clearance_m"] < trigger_clearance:
                tcp_position, _ = current_tcp_pose()
                away_vector = np.asarray(
                    best.get("away_vector", [0.0, 0.0, 0.0]),
                    dtype=float,
                )
                norm = float(np.linalg.norm(away_vector))
                if norm > 1e-6:
                    recovery_tcp = (
                        np.asarray(tcp_position, dtype=float)
                        + away_vector / norm
                        * float(args.brain_clearance_intervention_away)
                    )
                else:
                    recovery_tcp = np.asarray(tcp_position, dtype=float)
                recovery_tcp[2] += float(args.brain_clearance_intervention_lift)
                recovery_arm = ik_arm_for_tcp(recovery_tcp, current_arm)
                if recovery_arm is not None:
                    safety_event = {
                        "kind": "clearance_intervention",
                        "phase": phase,
                        "link": best["link"],
                        "obstacle": best["obstacle"],
                        "clearance_m": best["clearance_m"],
                        "threshold_m": best["threshold_m"],
                        "trigger_clearance_m": trigger_clearance,
                        "away_vector": best.get("away_vector", [0.0, 0.0, 0.0]),
                    }
                    recorder = recording_state["recorder"]
                    if recorder is not None:
                        recorder.capture(
                            phase,
                            recovery_arm,
                            OPEN_GRIPPER,
                            target_arm,
                            target_gripper,
                            safety_event,
                        )
                    target_arm = np.asarray(recovery_arm, dtype=float)
                    target_gripper = OPEN_GRIPPER
                    print(
                        "brain_clearance_intervention "
                        f"phase={phase} link={best['link']} "
                        f"clearance={best['clearance_m']:.4f}m "
                        f"trigger={trigger_clearance:.4f}m "
                        f"target={np.asarray(target_arm).round(4).tolist()}",
                        flush=True,
                    )
    elif args.brain_terminal_servo_without_vision:
        target_arm = teacher_arm
        target_gripper = float(teacher_gripper)
    else:
        return teacher_arm, teacher_gripper
    target_arm = np.asarray(target_arm, dtype=float)
    target_arm, target_gripper, servo_applied = terminal_servo_target(
        target_arm,
        float(target_gripper),
        phase,
        current_arm,
    )
    if servo_applied and terminal_servo_state["successes"] % CAPTURE_INTERVAL == 1:
        print(
            f"brain_control source=direct+terminal_servo phase={phase} "
            f"target={np.asarray(target_arm).round(4).tolist()} "
            f"gripper={target_gripper:.3f}",
            flush=True,
        )
    target_arm, target_gripper, place_servo_applied = place_servo_target(
        np.asarray(target_arm, dtype=float),
        float(target_gripper),
        phase,
        current_arm,
    )
    if (
        place_servo_applied
        and place_servo_state["successes"] % CAPTURE_INTERVAL == 1
    ):
        print(
            f"brain_control source=direct+place_servo phase={phase} "
            f"target={np.asarray(target_arm).round(4).tolist()} "
            f"gripper={target_gripper:.3f}",
            flush=True,
        )
    if (
        not args.disable_brain_grasp_readiness_gate
        and brain_runtime is not None
        and args.brain_control in {"filtered", "direct"}
        and not payload_state["held"]
        and args.grasp_mode != "teleport"
        and phase in {"approach_cube", "descend_to_cube", "close_gripper"}
    ):
        distance, ready = update_grasp_readiness()
        if not ready:
            target_gripper = OPEN_GRIPPER
            if brain_runtime.stats["accepted"] % CAPTURE_INTERVAL == 0:
                print(
                    "grasp_readiness_gate "
                    f"phase={phase} distance={distance:.4f}m "
                    f"limit={args.grasp_attach_distance:.4f}m "
                    "gripper=OPEN",
                    flush=True,
                )
    if payload_state["held"] and phase not in {"open_gripper", "retreat_after_release"}:
        target_gripper = CLOSED_GRIPPER
    return np.asarray(target_arm, dtype=float), float(target_gripper)


def update_attached_payload():
    cube_position = np.asarray(cube.get_world_pose()[0], dtype=float)
    payload_state["max_cube_height_m"] = max(
        float(payload_state["max_cube_height_m"]),
        float(cube_position[2]),
    )
    if args.physical_grasp or args.grasp_mode == "physics" or not payload_state["held"]:
        return
    tcp_position, tcp_rotation = current_tcp_pose()
    if args.grasp_mode == "teleport":
        target_position = np.asarray(tcp_position, dtype=float)
        payload_state["teleport_shortcut_used"] = True
    else:
        target_position = (
            np.asarray(tcp_position, dtype=float)
            + np.asarray(payload_state["offset"], dtype=float)
        )
    cube.set_world_pose(position=target_position)
    cube.set_linear_velocity(np.zeros(3))
    cube.set_angular_velocity(np.zeros(3))
    samples = payload_state.setdefault("motion_samples", [])
    samples.append(np.asarray(target_position, dtype=float))
    if len(samples) > 12:
        del samples[:-12]


def inherited_payload_velocity(sample_count=6):
    samples = payload_state.get("motion_samples") or []
    if len(samples) < 2:
        return None
    window = samples[-max(2, int(sample_count)) :]
    frame_delta = max(len(window) - 1, 1)
    velocity = (np.asarray(window[-1]) - np.asarray(window[0])) * (
        PHYSICS_FPS / frame_delta
    )
    if not np.all(np.isfinite(velocity)):
        return None
    return velocity


def tcp_to_cube_distance():
    cube_position = np.asarray(cube.get_world_pose()[0], dtype=float)
    return float(np.linalg.norm(cube_position - grasp_center_position()))


def grasp_center_distance_to(target_position):
    return float(
        np.linalg.norm(
            np.asarray(grasp_center_position(), dtype=float)
            - np.asarray(target_position, dtype=float)
        )
    )


def brain_phase_hold_enabled():
    return (
        brain_runtime is not None
        and args.brain_control in {"filtered", "direct"}
        and int(args.brain_phase_hold_frames) > 0
        and args.grasp_mode != "teleport"
    )


def brain_phase_hold_target(phase_name):
    cube_position = np.asarray(cube.get_world_pose()[0], dtype=float)
    if phase_name == "approach_cube":
        hover_target = grasp_target_for_cube(cube_position)
        hover_target[2] += float(args.vertical_grasp_hover_height)
        return hover_target, float(args.brain_approach_ready_distance)
    if phase_name == "descend_to_cube":
        return grasp_target_for_cube(cube_position), float(args.brain_descend_ready_distance)
    return None, None


def curobo_reference_recovery_target(trajectory, lookahead=2):
    trajectory = np.asarray(trajectory, dtype=float)
    if trajectory.ndim != 2 or len(trajectory) == 0:
        raise ValueError("cuRobo recovery trajectory must contain joint waypoints")
    current_arm = arm_positions()
    distances = np.linalg.norm(trajectory - current_arm[None, :], axis=1)
    nearest_index = int(np.argmin(distances))
    target_index = min(nearest_index + max(int(lookahead), 0), len(trajectory) - 1)
    return trajectory[target_index], nearest_index, target_index


def hold_brain_phase_until_ready(phase_name, trajectory, gripper_position):
    if not brain_phase_hold_enabled():
        return True
    target, threshold = brain_phase_hold_target(phase_name)
    if target is None:
        return True
    max_frames = max(int(args.brain_phase_hold_frames), 0)
    best_distance = float("inf")
    for frame in range(max_frames + 1):
        distance = grasp_center_distance_to(target)
        if np.isfinite(distance):
            best_distance = min(best_distance, distance)
        if distance <= threshold:
            print(
                "brain_phase_hold_ready "
                f"phase={phase_name} frames={frame} "
                f"distance={distance:.4f}m threshold={threshold:.4f}m",
                flush=True,
            )
            return True
        if frame >= max_frames:
            break
        if frame % CAPTURE_INTERVAL == 0:
            print(
                "brain_phase_hold "
                f"phase={phase_name} frame={frame}/{max_frames} "
                f"distance={distance:.4f}m threshold={threshold:.4f}m",
                flush=True,
            )
        recovery_target, nearest_index, target_index = (
            curobo_reference_recovery_target(trajectory)
        )
        if frame % CAPTURE_INTERVAL == 0:
            print(
                "brain_phase_teacher "
                f"phase={phase_name} nearest={nearest_index} "
                f"lookahead={target_index}",
                flush=True,
            )
        if not step_for(1, recovery_target, gripper_position, phase_name):
            return False
    payload_state["last_failure"] = (
        "brain_phase_hold_rejected: "
        f"phase={phase_name} distance_m={distance:.4f} "
        f"threshold_m={threshold:.4f}"
    )
    print(
        "brain_phase_hold_rejected "
        f"phase={phase_name} frames={max_frames} "
        f"distance={distance:.4f}m threshold={threshold:.4f}m "
        f"best={best_distance:.4f}m",
        flush=True,
    )
    return False


def warmup_brain_runtime(phase_name="approach_cube"):
    if brain_runtime is None or args.brain_control not in {"filtered", "direct"}:
        return True
    if rgb_annotator is None:
        raise RuntimeError("Brain warmup requires an RGB annotator")
    brain_runtime.frames = []
    brain_runtime.latest = None
    brain_runtime.physics_frame = 0
    hold_arm = arm_positions()
    hold_gripper = float(
        robot.get_joint_positions()[dof_index[GRIPPER_JOINTS[0]]]
    )
    max_frames = max(
        int(brain_runtime.clip_frames * CAPTURE_INTERVAL + CAPTURE_INTERVAL),
        1,
    )
    for frame in range(max_frames):
        if (
            control_state["end_requested"]
            or control_state["retry_requested"]
            or not app_allows_automation_step()
        ):
            return False
        apply_target(robot, dof_index, hold_arm, hold_gripper)
        guarded_world_step("brain_warmup")
        brain_runtime.observe(
            rgb_annotator,
            phase_name,
            current_brain_state(),
        )
        if brain_runtime.latest is not None:
            print(
                "brain_warmup_ready "
                f"phase={phase_name} frames={frame + 1} "
                f"clip_frames={brain_runtime.clip_frames} "
                f"observe_fps={brain_observe_fps:.1f}",
                flush=True,
            )
            return True
    payload_state["last_failure"] = (
        "brain_warmup_rejected: no prediction after "
        f"{max_frames} physics frames"
    )
    raise RuntimeError(payload_state["last_failure"])


def ensure_brain_phase_context(phase_name):
    if (
        brain_runtime is None
        or args.brain_control != "direct"
        or not args.brain_strict_direct
    ):
        return True
    latest = brain_runtime.latest
    if latest is not None and latest["predicted_phase"] == phase_name:
        return True
    return warmup_brain_runtime(phase_name)


def update_grasp_readiness():
    distance = tcp_to_cube_distance()
    grasp_readiness_state["last_distance_m"] = distance
    best_distance = grasp_readiness_state["best_distance_m"]
    if best_distance is None or distance < float(best_distance):
        grasp_readiness_state["best_distance_m"] = distance
    if distance <= float(args.grasp_attach_distance):
        grasp_readiness_state["ready"] = True
    return distance, bool(grasp_readiness_state["ready"])


def begin_payload_grasp():
    if args.physical_grasp or args.grasp_mode == "physics":
        return True
    tcp_position, tcp_rotation = current_tcp_pose()
    cube_position = np.asarray(cube.get_world_pose()[0], dtype=float)
    grasp_offset = local_grasp_offset_world(tcp_rotation)
    grasp_center = tcp_position + grasp_offset
    distance = float(np.linalg.norm(cube_position - grasp_center))
    payload_state["attach_distance_m"] = distance
    if distance > args.grasp_attach_distance:
        payload_state["last_failure"] = (
            "grasp_attach_rejected: "
            f"tcp_to_cube_distance_m={distance:.4f} "
            f"limit_m={args.grasp_attach_distance:.4f}"
        )
        print(
            f"grasp_attach_rejected distance={distance:.4f}m "
            f"limit={args.grasp_attach_distance:.4f}m",
            flush=True,
        )
        return False
    payload_state["held"] = True
    payload_state["last_failure"] = None
    if args.grasp_mode == "teleport":
        payload_state["offset"] = np.zeros(3)
        payload_state["local_offset"] = np.zeros(3)
        payload_state["teleport_shortcut_used"] = True
    else:
        payload_state["offset"] = cube_position - tcp_position
        payload_state["local_offset"] = np.asarray(
            GRASP_CUBE_TCP_LOCAL_OFFSET,
            dtype=float,
        )
    print(
        f"grasp_attach mode={args.grasp_mode} "
        f"distance={distance:.4f}m "
        f"offset={np.asarray(payload_state['offset']).round(4).tolist()} "
        f"local_offset={np.asarray(payload_state['local_offset']).round(4).tolist()}",
        flush=True,
    )
    return True


def release_payload():
    release_position = np.asarray(cube.get_world_pose()[0], dtype=float)
    if payload_state["held"]:
        print(
            f"grasp_release cube={release_position.round(4).tolist()}",
            flush=True,
        )
    payload_state["held"] = False
    payload_state["offset"] = np.zeros(3)
    payload_state["local_offset"] = np.asarray(
        GRASP_CUBE_TCP_LOCAL_OFFSET,
        dtype=float,
    )
    payload_state["release_clearance_grace_frames"] = max(
        int(args.release_cube_clearance_grace_frames),
        0,
    )
    release_velocity, velocity_source = basket_release_velocity_for(release_position)
    if release_velocity is not None:
        release_velocity = np.asarray(release_velocity, dtype=float)
        if release_velocity.shape != (3,) or not np.all(np.isfinite(release_velocity)):
            raise ValueError(
                "Basket release velocity must contain three finite values"
            )
        cube.set_linear_velocity(release_velocity)
        cube.set_angular_velocity(np.zeros(3))
        payload_state["release_velocity_mps"] = release_velocity.tolist()
        payload_state["release_velocity_source"] = velocity_source
        print(
            "basket_release_velocity "
            f"source={velocity_source} "
            f"velocity={release_velocity.round(4).tolist()}",
            flush=True,
        )


def step_for(frames, arm_positions, gripper_position, phase):
    for _ in range(frames):
        if (
            control_state["end_requested"]
            or control_state["retry_requested"]
            or not app_allows_automation_step()
        ):
            return False
        teacher_arm, teacher_gripper = corrective_grasp_teacher_target(
            arm_positions,
            gripper_position,
            phase,
        )
        control_arm, control_gripper = choose_control_target(
            teacher_arm,
            teacher_gripper,
            phase,
        )
        apply_target(
            robot,
            dof_index,
            control_arm,
            control_gripper,
        )
        try:
            guarded_world_step(phase)
        except RuntimeError:
            capture_step(
                phase,
                teacher_arm,
                teacher_gripper,
                control_arm,
                control_gripper,
                clearance_state["last_violation"],
            )
            raise
        update_attached_payload()
        capture_step(
            phase,
            teacher_arm,
            teacher_gripper,
            control_arm,
            control_gripper,
        )
    return True


def ik_arm_for_tcp(target_position, current_arm):
    try:
        ik_action, success = fk_solver.compute_inverse_kinematics(
            target_position=np.asarray(target_position, dtype=float),
        )
    except TypeError:
        ik_action, success = fk_solver.compute_inverse_kinematics(
            np.asarray(target_position, dtype=float)
        )
    except Exception as exc:
        print(f"vertical_grasp_ik_error error={exc}", flush=True)
        return None
    if not success:
        return None
    ik_arm = _ik_arm_positions(ik_action, current_arm)
    if ik_arm is None or not np.all(np.isfinite(ik_arm)):
        return None
    return ik_arm


def step_direct_target(frames, arm_positions, gripper_position, phase, capture=True):
    for _ in range(frames):
        if (
            control_state["end_requested"]
            or control_state["retry_requested"]
            or not app_allows_automation_step()
        ):
            return False
        apply_target(
            robot,
            dof_index,
            arm_positions,
            gripper_position,
        )
        guarded_world_step(phase)
        update_attached_payload()
        if capture:
            capture_step(
                phase,
                arm_positions,
                gripper_position,
                arm_positions,
                gripper_position,
            )
    return True


def step_physics_only(frames, arm_positions, gripper_position, phase):
    for frame_index in range(frames):
        if (
            control_state["end_requested"]
            or control_state["retry_requested"]
            or not app_allows_automation_step()
        ):
            return False
        apply_target(
            robot,
            dof_index,
            arm_positions,
            gripper_position,
        )
        world.step(render=should_render_world())
        if phase == "basket_flight" and frame_index % max(PHYSICS_FPS // 2, 1) == 0:
            cube_position = np.asarray(cube.get_world_pose()[0], dtype=float)
            distance = basket_distance_for_position(cube_position)
            if status_label is not None:
                status_label.text = (
                    "Basket flight "
                    f"{frame_index}/{frames}"
                    + (
                        f" distance={distance:.3f}m"
                        if distance is not None
                        else ""
                    )
                )
            print(
                "basket_flight "
                f"frame={frame_index} "
                f"cube={cube_position.round(4).tolist()} "
                + (
                    f"distance={distance:.4f}m"
                    if distance is not None
                    else ""
                ),
                flush=True,
            )
        if payload_state.get("release_clearance_grace_frames", 0) > 0:
            payload_state["release_clearance_grace_frames"] -= 1
    return True


def step_basket_kinematic_flight(frames, arm_positions, gripper_position):
    release_position = np.asarray(cube.get_world_pose()[0], dtype=float)
    release_velocity = payload_state.get("release_velocity_mps")
    if release_velocity is None:
        release_velocity = payload_state.get("gripper_throw_velocity_mps")
    if release_velocity is None:
        print("basket_flight_skipped reason=no_release_velocity", flush=True)
        return True
    release_velocity = np.asarray(release_velocity, dtype=float)
    if release_velocity.shape != (3,) or not np.all(np.isfinite(release_velocity)):
        raise ValueError(
            "Basket kinematic flight requires a finite 3D release velocity"
        )
    gravity = np.asarray([0.0, 0.0, -9.81], dtype=float)
    dt = 1.0 / float(PHYSICS_FPS)
    basket_center = (
        np.asarray(basket_config.get("center", metadata["place_position"]), dtype=float)
        if basket_config is not None
        else None
    )
    rim_z = (
        float(basket_center[2]) + float(basket_config.get("wall_height", 0.08))
        if basket_center is not None
        else None
    )
    landing_position = basket_landing_position()
    landed = False
    print(
        "basket_kinematic_flight_start "
        f"frames={frames} "
        f"position={release_position.round(4).tolist()} "
        f"velocity={release_velocity.round(4).tolist()}",
        flush=True,
    )
    for frame_index in range(frames):
        if (
            control_state["end_requested"]
            or control_state["retry_requested"]
            or not app_allows_automation_step()
        ):
            return False
        apply_target(
            robot,
            dof_index,
            arm_positions,
            gripper_position,
        )
        if landed:
            position = landing_position.copy()
        else:
            t = (frame_index + 1) * dt
            position = release_position + release_velocity * t + 0.5 * gravity * t * t
            distance = basket_distance_for_position(position)
            if (
                landing_position is not None
                and rim_z is not None
                and distance is not None
                and distance <= basket_success_distance()
                and position[2] <= rim_z + float(metadata["cube_size"])
            ):
                landed = True
                payload_state["basket_landed"] = True
                position = landing_position.copy()
                cube.set_linear_velocity(np.zeros(3))
                cube.set_angular_velocity(np.zeros(3))
                print(
                    "basket_landed "
                    f"frame={frame_index + 1} "
                    f"cube={position.round(4).tolist()} "
                    f"distance={distance:.4f}m",
                    flush=True,
                )
        cube.set_world_pose(position=position)
        app.update()
        distance = basket_distance_for_position(position)
        if frame_index % max(PHYSICS_FPS // 2, 1) == 0 or frame_index == frames - 1:
            if status_label is not None:
                status_label.text = (
                    "Basket flight "
                    f"{frame_index + 1}/{frames}"
                    + (
                        f" distance={distance:.3f}m"
                        if distance is not None
                        else ""
                    )
                )
            print(
                "basket_flight "
                f"frame={frame_index + 1} "
                f"cube={position.round(4).tolist()} "
                + (
                    f"distance={distance:.4f}m"
                    if distance is not None
                    else ""
                ),
                flush=True,
            )
    return True


def run_basket_gripper_throw():
    if (
        task_name != "basket_drop"
        or basket_config is None
        or args.basket_velocity_mode != "gripper"
        or args.grasp_mode == "teleport"
        or not payload_state["held"]
    ):
        return True
    release_position = np.asarray(cube.get_world_pose()[0], dtype=float)
    basket_center = np.asarray(
        basket_config.get("center", metadata["place_position"]),
        dtype=float,
    )
    delta = basket_center - release_position
    horizontal_distance = float(np.linalg.norm(delta[:2]))
    if not np.isfinite(horizontal_distance) or horizontal_distance <= 1e-5:
        return True
    throw_direction = np.array(
        [delta[0] / horizontal_distance, delta[1] / horizontal_distance, 0.0],
        dtype=float,
    )
    frames = max(int(args.basket_gripper_throw_frames), 2)
    tcp_position, _ = current_tcp_pose()
    current_arm = arm_positions()
    swing_distance = float(args.basket_gripper_throw_speed_scale) * min(
        horizontal_distance,
        0.45,
    )
    swing_distance = float(np.clip(swing_distance, 0.055, 0.18))
    swing_lift = float(np.clip(float(args.basket_gripper_throw_lift), 0.0, 0.14))
    target_tcp = (
        np.asarray(tcp_position, dtype=float)
        + throw_direction * swing_distance
        + np.array([0.0, 0.0, swing_lift], dtype=float)
    )
    target_arm = ik_arm_for_tcp(target_tcp, current_arm)
    if target_arm is None:
        print(
            "basket_gripper_throw_skipped reason=ik_unavailable "
            f"target_tcp={target_tcp.round(4).tolist()}",
            flush=True,
        )
        return True
    payload_state["motion_samples"] = []
    print(
        "basket_gripper_throw "
        f"frames={frames} release={release_position.round(4).tolist()} "
        f"basket={basket_center.round(4).tolist()} "
        f"swing_distance={swing_distance:.4f}m lift={swing_lift:.4f}m",
        flush=True,
    )
    if not step_direct_target(
        frames,
        target_arm,
        CLOSED_GRIPPER,
        "basket_gripper_throw",
    ):
        return False
    inherited = inherited_payload_velocity()
    if inherited is not None:
        payload_state["gripper_throw_velocity_mps"] = inherited.tolist()
    payload_state["basket_release_arm_positions"] = np.asarray(
        target_arm,
        dtype=float,
    ).tolist()
    print(
        "basket_gripper_throw_complete "
        f"inherited_velocity="
        f"{None if inherited is None else inherited.round(4).tolist()}",
        flush=True,
    )
    return True


def run_vertical_grasp_approach(gripper_position):
    if args.disable_vertical_grasp_servo or args.grasp_mode == "teleport":
        return True
    cube_position = np.asarray(cube.get_world_pose()[0], dtype=float)
    cube_target = grasp_target_for_cube(cube_position)
    hover_center = cube_target.copy()
    hover_center[2] += float(args.vertical_grasp_hover_height)
    frames_per_step = max(int(args.vertical_grasp_frames_per_step), 1)
    descent_steps = max(int(args.vertical_grasp_steps), 2)

    targets = [hover_center]
    for alpha in np.linspace(0.0, 1.0, descent_steps)[1:]:
        target = hover_center * (1.0 - alpha) + cube_target * alpha
        targets.append(target)

    for index, grasp_center_target in enumerate(targets):
        tcp_position, tcp_rotation = current_tcp_pose()
        target_tcp = (
            np.asarray(grasp_center_target, dtype=float)
            - local_grasp_offset_world(tcp_rotation)
        )
        current_arm = arm_positions()
        ik_arm = ik_arm_for_tcp(target_tcp, current_arm)
        if ik_arm is None:
            print(
                "vertical_grasp_ik_failed "
                f"waypoint={index} target={target_tcp.round(4).tolist()}",
                flush=True,
            )
            return True
        if not step_direct_target(
            frames_per_step,
            ik_arm,
            gripper_position,
            "vertical_grasp",
        ):
            return False

    distance = tcp_to_cube_distance()
    xy_distance = float(
        np.linalg.norm(
            np.asarray(grasp_center_position(), dtype=float)[:2]
            - cube_target[:2]
        )
    )
    print(
        "vertical_grasp_complete "
        f"distance={distance:.4f}m xy_distance={xy_distance:.4f}m",
        flush=True,
    )
    return True


def close_gripper_in_place(frames, gripper_position):
    current_arm = arm_positions()
    if (
        brain_runtime is not None
        and args.brain_control == "direct"
        and args.brain_strict_direct
    ):
        return step_for(
            frames,
            current_arm,
            gripper_position,
            "close_gripper",
        )
    return step_direct_target(
        frames,
        current_arm,
        gripper_position,
        "close_gripper",
    )


def corrective_grasp_teacher_target(fallback_arm, fallback_gripper, phase):
    if (
        brain_runtime is None
        or args.brain_control not in {"filtered", "direct"}
        or args.grasp_mode == "teleport"
        or payload_state["held"]
        or phase not in {"descend_to_cube", "close_gripper"}
    ):
        return np.asarray(fallback_arm, dtype=float), float(fallback_gripper)
    cube_position = np.asarray(cube.get_world_pose()[0], dtype=float)
    grasp_center_target = grasp_target_for_cube(cube_position)
    current_grasp_center = np.asarray(grasp_center_position(), dtype=float)
    xy_distance = float(
        np.linalg.norm(current_grasp_center[:2] - grasp_center_target[:2])
    )
    if xy_distance > float(args.vertical_grasp_xy_tolerance):
        grasp_center_target = grasp_center_target.copy()
        grasp_center_target[2] += float(args.vertical_grasp_hover_height)
    tcp_position, tcp_rotation = current_tcp_pose()
    target_tcp = grasp_center_target - local_grasp_offset_world(tcp_rotation)
    current_arm = arm_positions()
    ik_arm = ik_arm_for_tcp(target_tcp, current_arm)
    if ik_arm is None:
        return np.asarray(fallback_arm, dtype=float), float(fallback_gripper)
    distance = float(
        np.linalg.norm(np.asarray(grasp_center_position(), dtype=float) - cube_position)
    )
    if distance > float(args.grasp_attach_distance):
        target_gripper = OPEN_GRIPPER
    else:
        target_gripper = float(fallback_gripper)
    return np.asarray(ik_arm, dtype=float), float(target_gripper)


def align_for_no_teleport_grasp(arm_positions, gripper_position):
    if args.grasp_mode == "teleport":
        return True
    target_distance = float(args.grasp_attach_distance) * 0.85
    distance, ready = update_grasp_readiness()
    if ready:
        print(
            "grasp_readiness_aligned "
            f"frames=0 distance={distance:.4f}m "
            f"target={target_distance:.4f}m",
            flush=True,
        )
        return True
    if not args.brain_terminal_servo:
        if (
            args.disable_brain_grasp_readiness_gate
            or brain_runtime is None
            or args.brain_control not in {"filtered", "direct"}
        ):
            return True
        max_frames = max(int(args.brain_grasp_readiness_frames), 0)
        for frame in range(max_frames):
            distance, ready = update_grasp_readiness()
            if ready:
                print(
                    "brain_grasp_readiness_aligned "
                    f"frames={frame} distance={distance:.4f}m "
                    f"limit={args.grasp_attach_distance:.4f}m",
                    flush=True,
                )
                return True
            if not step_for(1, arm_positions, OPEN_GRIPPER, "descend_to_cube"):
                return False
        distance, _ = update_grasp_readiness()
        payload_state["attach_distance_m"] = distance
        payload_state["last_failure"] = (
            "grasp_readiness_rejected: "
            f"tcp_to_cube_distance_m={distance:.4f} "
            f"limit_m={args.grasp_attach_distance:.4f}"
        )
        print(
            "grasp_readiness_rejected "
            f"frames={max_frames} distance={distance:.4f}m "
            f"limit={args.grasp_attach_distance:.4f}m "
            f"best={grasp_readiness_state['best_distance_m']:.4f}m",
            flush=True,
        )
        return False
    max_frames = max(int(args.brain_terminal_servo_align_frames), 0)
    for frame in range(max_frames):
        distance, ready = update_grasp_readiness()
        if distance <= target_distance:
            print(
                "terminal_servo_aligned "
                f"frames={frame} distance={distance:.4f}m "
                f"target={target_distance:.4f}m",
                flush=True,
            )
            return True
        if not step_for(1, arm_positions, gripper_position, "close_gripper"):
            return False
    distance = tcp_to_cube_distance()
    print(
        "terminal_servo_align_exhausted "
        f"frames={max_frames} distance={distance:.4f}m "
        f"target={target_distance:.4f}m",
        flush=True,
    )
    return True


def align_for_no_teleport_place(arm_positions, gripper_position):
    if not (args.brain_place_servo or args.brain_terminal_servo):
        return True
    if args.grasp_mode == "teleport" or not payload_state["held"]:
        return True
    target_distance = float(args.brain_place_servo_distance)
    max_frames = max(int(args.brain_place_servo_frames), 0)
    for frame in range(max_frames):
        distance = cube_to_place_distance()
        if distance <= target_distance:
            print(
                "place_servo_aligned "
                f"frames={frame} distance={distance:.4f}m "
                f"target={target_distance:.4f}m",
                flush=True,
            )
            return True
        if not step_for(1, arm_positions, gripper_position, "open_gripper"):
            return False
    distance = cube_to_place_distance()
    print(
        "place_servo_align_exhausted "
        f"frames={max_frames} distance={distance:.4f}m "
        f"target={target_distance:.4f}m",
        flush=True,
    )
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
    payload_state["offset"] = np.zeros(3)
    payload_state["local_offset"] = np.asarray(
        GRASP_CUBE_TCP_LOCAL_OFFSET,
        dtype=float,
    )
    payload_state["attach_distance_m"] = None
    payload_state["max_cube_height_m"] = float(cube_initial_position[2])
    payload_state["teleport_shortcut_used"] = False
    payload_state["last_failure"] = None
    payload_state["release_clearance_grace_frames"] = 0
    payload_state["motion_samples"] = []
    payload_state["release_velocity_mps"] = None
    payload_state["release_velocity_source"] = None
    payload_state["gripper_throw_velocity_mps"] = None
    payload_state["basket_release_arm_positions"] = None
    payload_state["basket_landed"] = False
    clearance_state["min_clearance_m"] = None
    clearance_state["min_cube_clearance_m"] = None
    clearance_state["last_violation"] = None
    clearance_state["violations"] = []
    for _ in range(30):
        guarded_world_step("reset_cycle")


def arm_positions():
    positions = robot.get_joint_positions()
    return np.asarray(
        [positions[dof_index[f"joint{i}"]] for i in range(1, 7)],
        dtype=float,
    )


def current_arm_positions():
    return arm_positions()


def detect_and_plan():
    current_arm_positions = arm_positions()
    for _ in range(PHYSICS_FPS // 4):
        guarded_world_step("conveyor_return")
    detected_position = np.asarray(cube.get_world_pose()[0], dtype=float)
    recorder = recording_state["recorder"]
    if recorder is not None:
        recorder.detected_cube_position = detected_position.tolist()
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
    if args.use_static_plan:
        trajectories = {
            phase_name: np.asarray(plan_data[phase_name]).copy()
            for phase_name in phase_names
        }
        print(
            "static_plan_loaded "
            + " ".join(
                f"{name}={len(trajectories[name])}"
                for name in phase_names
            ),
            flush=True,
        )
        return trajectories

    request = {
        "cube_position": detected_position.tolist(),
        "place_position": metadata["place_position"],
        "start_arm_positions": current_arm_positions.tolist(),
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
    grasp_readiness_state["ready"] = False
    grasp_readiness_state["best_distance_m"] = None
    grasp_readiness_state["last_distance_m"] = None
    for phase_name in phase_names:
        trajectory = np.asarray(trajectories[phase_name])
        if status_label is not None:
            status_label.text = f"Running: {phase_name}"
        print(f"phase={phase_name} waypoints={len(trajectory)}", flush=True)

        if (
            phase_name == "descend_to_cube"
            and not args.disable_vertical_grasp_servo
            and args.grasp_mode != "teleport"
        ):
            print(
                "phase=descend_to_cube skipped_for_single_vertical_grasp",
                flush=True,
            )
            continue

        if phase_name == "lift_cube":
            if not ensure_brain_phase_context("descend_to_cube"):
                return False
            if not run_vertical_grasp_approach(OPEN_GRIPPER):
                return False
            if not align_for_no_teleport_grasp(
                current_arm_positions(),
                OPEN_GRIPPER,
            ):
                return False
            if not ensure_brain_phase_context("close_gripper"):
                return False
            gripper_position = CLOSED_GRIPPER
            if not close_gripper_in_place(100, gripper_position):
                return False
            if (
                args.brain_control == "direct"
                and args.brain_strict_direct
            ):
                actual_gripper = float(
                    robot.get_joint_positions()[dof_index[GRIPPER_JOINTS[0]]]
                )
                if actual_gripper < CLOSED_GRIPPER * 0.70:
                    payload_state["last_failure"] = (
                        "brain_gripper_close_rejected: "
                        f"actual={actual_gripper:.4f} "
                        f"required={CLOSED_GRIPPER * 0.70:.4f}"
                    )
                    print(payload_state["last_failure"], flush=True)
                    return False
            if not begin_payload_grasp():
                return False
        elif phase_name == "retreat_after_release":
            if not align_for_no_teleport_place(
                trajectory[0],
                gripper_position,
            ):
                return False
            if not ensure_brain_phase_context("open_gripper"):
                return False
            if not run_basket_gripper_throw():
                return False
            gripper_position = OPEN_GRIPPER
            if task_name == "basket_drop" and args.basket_velocity_mode == "gripper":
                release_open_frames = max(
                    int(args.basket_gripper_release_open_frames),
                    0,
                )
                basket_hold_arm = payload_state.get(
                    "basket_release_arm_positions"
                )
                if basket_hold_arm is None:
                    basket_hold_arm = trajectory[0]
                basket_hold_arm = np.asarray(basket_hold_arm, dtype=float)
                release_payload()
                if release_open_frames > 0:
                    print(
                        "basket_throw_release_open "
                        f"frames={release_open_frames}",
                        flush=True,
                    )
                if release_open_frames > 0 and not step_physics_only(
                    release_open_frames,
                    basket_hold_arm,
                    gripper_position,
                    "open_gripper_after_throw_release",
                ):
                    return False
                settle_frames = max(
                    int(args.basket_flight_settle_frames),
                    int(args.basket_min_flight_frames),
                    0,
                )
                if settle_frames > 0:
                    print(
                        "basket_flight_wait "
                        f"frames={settle_frames} "
                        f"mode={args.basket_flight_mode}",
                        flush=True,
                    )
                if settle_frames > 0:
                    if args.basket_flight_mode == "kinematic":
                        if not step_basket_kinematic_flight(
                            settle_frames,
                            basket_hold_arm,
                            OPEN_GRIPPER,
                        ):
                            return False
                    elif not step_physics_only(
                        settle_frames,
                        basket_hold_arm,
                        OPEN_GRIPPER,
                        "basket_flight",
                    ):
                        return False
                continue
            else:
                if not step_for(
                    90,
                    trajectory[0],
                    gripper_position,
                    "open_gripper",
                ):
                    return False
                release_payload()

        if not ensure_brain_phase_context(phase_name):
            return False
        for arm_positions in trajectory:
            if not step_for(
                max(args.steps_per_waypoint, 1),
                arm_positions,
                gripper_position,
                phase_name,
            ):
                return False
        if phase_name in {"approach_cube", "descend_to_cube"}:
            if not hold_brain_phase_until_ready(
                phase_name,
                trajectory,
                gripper_position,
            ):
                return False

    final_cube = np.asarray(cube.get_world_pose()[0])
    place_position = task_success_position()
    place_distance = float(
        np.linalg.norm(final_cube[:2] - place_position[:2])
    )
    basket_distance = basket_distance_for_position(final_cube)
    success_distance = (
        basket_success_distance()
        if basket_distance is not None
        else float(args.task_success_distance)
    )
    task_target_reached = place_distance < success_distance
    if basket_distance is not None and payload_state.get("basket_landed"):
        task_target_reached = True
    lifted_height = (
        float(payload_state["max_cube_height_m"])
        - float(cube_initial_position[2])
    )
    print(
        f"playback_complete cube={final_cube.round(4).tolist()} "
        f"place_distance={place_distance:.4f}m "
        f"lifted_height={lifted_height:.4f}m"
        + (
            f" basket_distance={basket_distance:.4f}m"
            if basket_distance is not None
            else ""
        ),
        flush=True,
    )
    if status_label is not None:
        if basket_distance is not None:
            status_label.text = (
                "Basket throw complete"
                if task_target_reached
                else "Basket throw missed"
            )
        else:
            status_label.text = "Conveyor returning cube..."
    return {
        "placed_at_start": task_target_reached,
        "task_target_reached": task_target_reached,
        "start_place_distance_m": place_distance,
        "cube_at_start": final_cube.tolist(),
        "task_success_position": place_position.tolist(),
        "basket_distance_m": basket_distance,
        "basket_success_distance_m": (
            basket_success_distance()
            if basket_distance is not None
            else None
        ),
        "basket_success": (
            task_target_reached
            if basket_distance is not None
            else None
        ),
        "basket_landed": (
            bool(payload_state.get("basket_landed"))
            if basket_distance is not None
            else None
        ),
        "basket_center": (
            np.asarray(basket_config["center"], dtype=float).tolist()
            if task_name == "basket_drop" and basket_config is not None
            else None
        ),
        "basket_release_velocity": (
            payload_state["release_velocity_mps"]
            if task_name == "basket_drop"
            else None
        ),
        "basket_release_velocity_source": payload_state[
            "release_velocity_source"
        ],
        "grasp_mode": args.grasp_mode,
        "grasp_attach_distance_m": payload_state["attach_distance_m"],
        "max_cube_height_m": float(payload_state["max_cube_height_m"]),
        "lifted_height_m": lifted_height,
        "lifted_without_teleport": (
            lifted_height > 0.10
            and not payload_state["teleport_shortcut_used"]
        ),
        "teleport_shortcut_used": payload_state["teleport_shortcut_used"],
        "minimum_robot_obstacle_clearance_m": clearance_state[
            "min_clearance_m"
        ],
        "minimum_robot_cube_clearance_m": clearance_state[
            "min_cube_clearance_m"
        ],
        "link_clearance_violations": list(clearance_state["violations"]),
    }


def return_cube_on_conveyor(hold_positions):
    set_conveyor_speed(conveyor_graph_nodes, args.conveyor_speed)
    pickup_position = np.asarray(metadata["cube_position"], dtype=float)
    max_frames = PHYSICS_FPS * 15
    for frame in range(max_frames):
        if (
            control_state["end_requested"]
            or control_state["retry_requested"]
            or not app_allows_automation_step()
        ):
            set_conveyor_speed(conveyor_graph_nodes, 0.0)
            return False
        control_arm, control_gripper = choose_control_target(
            hold_positions,
            OPEN_GRIPPER,
            "conveyor_return",
        )
        apply_target(
            robot,
            dof_index,
            control_arm,
            control_gripper,
        )
        guarded_world_step("observe_cube")
        capture_step(
            "conveyor_return",
            hold_positions,
            OPEN_GRIPPER,
            control_arm,
            control_gripper,
        )
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
                control_arm, control_gripper = choose_control_target(
                    hold_positions,
                    OPEN_GRIPPER,
                    "conveyor_settle",
                )
                apply_target(
                    robot,
                    dof_index,
                    control_arm,
                    control_gripper,
                )
                guarded_world_step("conveyor_settle")
                capture_step(
                    "conveyor_settle",
                    hold_positions,
                    OPEN_GRIPPER,
                    control_arm,
                    control_gripper,
                )
            print("conveyor_return complete", flush=True)
            final_position = np.asarray(cube.get_world_pose()[0])
            return {
                "returned_to_end": True,
                "conveyor_return_seconds": frame / PHYSICS_FPS,
                "final_cube_position": final_position.tolist(),
                "final_end_distance_m": float(
                    np.linalg.norm(
                        final_position[:2] - pickup_position[:2]
                    )
                ),
            }

    set_conveyor_speed(conveyor_graph_nodes, 0.0)
    raise RuntimeError("Cube did not return to the conveyor pickup point")


try:
    warmup_simulation_app()
    reset_cycle()
    completed_cycles = 0
    dataset_completed = existing_successful_episodes
    failed_attempts = 0
    update_dataset_manifest(failed_attempts)
    while app_allows_automation_step():
        if control_state["end_requested"]:
            break
        if (
            target_episode_total is not None
            and dataset_completed >= target_episode_total
        ):
            break
        if control_state["retry_requested"]:
            control_state["retry_requested"] = False
            recorder = recording_state["recorder"]
            if recorder is not None:
                recorder.abort()
                recording_state["recorder"] = None
            reset_cycle()
            completed_cycles = 0

        cycle_number = (
            next_episode_index
            if record_root is not None
            else completed_cycles + 1
        )
        if status_label is not None:
            status_label.text = f"Cycle {cycle_number}: detecting cube"
        print(f"cycle={cycle_number} start", flush=True)
        if record_root is not None:
            recording_state["recorder"] = EpisodeRecorder(
                record_root,
                next_episode_index,
                rgb_annotator,
            )
        try:
            trajectories = detect_and_plan()
            if not warmup_brain_runtime("approach_cube"):
                raise RuntimeError("Brain temporal warmup was interrupted")
            pick_metrics = run_pick_and_place(trajectories)
            if not pick_metrics:
                raise RuntimeError("Pick-and-place was interrupted")
            if not pick_metrics["placed_at_start"]:
                raise RuntimeError(
                    "Cube was not placed at the task target: "
                    f"distance={pick_metrics['start_place_distance_m']:.4f}m"
                )
            hold_positions = np.asarray(
                trajectories["retreat_after_release"]
            )[-1]
            if args.disable_conveyor_return or task_name == "basket_drop":
                conveyor_metrics = {
                    "returned_to_end": False,
                    "conveyor_return_seconds": 0.0,
                    "final_cube_position": pick_metrics["cube_at_start"],
                    "final_end_distance_m": None,
                    "conveyor_return_disabled": True,
                    "conveyor_return_skip_reason": (
                        "basket_drop"
                        if task_name == "basket_drop"
                        else "disabled_by_arg"
                    ),
                }
            else:
                conveyor_metrics = return_cube_on_conveyor(hold_positions)
                if not conveyor_metrics:
                    raise RuntimeError("Conveyor return was interrupted")
            recorder = recording_state["recorder"]
            while (
                recorder is not None
                and len(recorder.rows) < MIN_EPISODE_FRAMES
            ):
                control_arm, control_gripper = choose_control_target(
                    hold_positions,
                    OPEN_GRIPPER,
                    "observe_returned_cube",
                )
                apply_target(
                    robot,
                    dof_index,
                    control_arm,
                    control_gripper,
                )
                guarded_world_step("observe_returned_cube")
                capture_step(
                    "observe_returned_cube",
                    hold_positions,
                    OPEN_GRIPPER,
                    control_arm,
                    control_gripper,
                )
            metrics = {
                **pick_metrics,
                **conveyor_metrics,
                "conveyor_speed_mps": args.conveyor_speed,
                "minimum_robot_obstacle_clearance_m": clearance_state[
                    "min_clearance_m"
                ],
                "minimum_robot_cube_clearance_m": clearance_state[
                    "min_cube_clearance_m"
                ],
                "link_clearance_violations": list(
                    clearance_state["violations"]
                ),
                "brain_control": args.brain_control,
            }
            if brain_runtime is not None:
                metrics["brain_runtime"] = brain_runtime.summary()
                live_run_state["cycles"].append(
                    {
                        "cycle": int(cycle_number),
                        "metrics": metrics,
                    }
                )
                write_brain_run_report()
            if recorder is not None:
                episode_dir = recorder.finish(metrics)
                recording_state["recorder"] = None
                dataset_completed += 1
                next_episode_index += 1
                update_dataset_manifest(failed_attempts)
                print(f"wrote_episode={episode_dir}", flush=True)
            completed_cycles += 1
            if status_label is not None:
                status_label.text = f"Cycle {completed_cycles} complete"
            print(f"cycle={completed_cycles} complete", flush=True)
        except Exception as error:
            recorder = recording_state["recorder"]
            if recorder is not None:
                if args.keep_failed_episodes:
                    failure_cube_position = np.asarray(
                        cube.get_world_pose()[0],
                        dtype=float,
                    )
                    failure_place_position = task_success_position()
                    failure_place_distance = float(
                        np.linalg.norm(
                            failure_cube_position[:2]
                            - failure_place_position[:2]
                        )
                    )
                    failure_basket_distance = basket_distance_for_position(
                        failure_cube_position
                    )
                    failure_success_distance = (
                        basket_success_distance()
                        if failure_basket_distance is not None
                        else float(args.task_success_distance)
                    )
                    failure_lifted_height = (
                        float(payload_state["max_cube_height_m"])
                        - float(cube_initial_position[2])
                    )
                    failure_metrics = {
                        "success": False,
                        "error": str(error),
                        "payload_failure": payload_state["last_failure"],
                        "grasp_mode": args.grasp_mode,
                        "grasp_attach_distance_m": payload_state[
                            "attach_distance_m"
                        ],
                        "max_cube_height_m": float(
                            payload_state["max_cube_height_m"]
                        ),
                        "teleport_shortcut_used": payload_state[
                            "teleport_shortcut_used"
                        ],
                        "start_place_distance_m": failure_place_distance,
                        "cube_at_failure": failure_cube_position.tolist(),
                        "place_position": failure_place_position.tolist(),
                        "placed_at_start": failure_place_distance
                        < failure_success_distance,
                        "basket_distance_m": failure_basket_distance,
                        "basket_success_distance_m": (
                            basket_success_distance()
                            if failure_basket_distance is not None
                            else None
                        ),
                        "basket_success": (
                            failure_place_distance < failure_success_distance
                            if failure_basket_distance is not None
                            else None
                        ),
                        "basket_landed": (
                            bool(payload_state.get("basket_landed"))
                            if failure_basket_distance is not None
                            else None
                        ),
                        "lifted_height_m": failure_lifted_height,
                        "lifted_without_teleport": (
                            failure_lifted_height > 0.10
                            and not payload_state["teleport_shortcut_used"]
                        ),
                        "conveyor_speed_mps": args.conveyor_speed,
                        "minimum_robot_obstacle_clearance_m": clearance_state[
                            "min_clearance_m"
                        ],
                        "minimum_robot_cube_clearance_m": clearance_state[
                            "min_cube_clearance_m"
                        ],
                        "link_clearance_violations": list(
                            clearance_state["violations"]
                        ),
                        "brain_control": args.brain_control,
                    }
                    if brain_runtime is not None:
                        failure_metrics["brain_runtime"] = (
                            brain_runtime.summary()
                        )
                    episode_dir = recorder.finish(
                        failure_metrics,
                        success=False,
                        error=error,
                    )
                    next_episode_index += 1
                    print(f"wrote_failed_episode={episode_dir}", flush=True)
                else:
                    recorder.abort()
                recording_state["recorder"] = None
            record_brain_failure(cycle_number, error)
            if record_root is None:
                raise
            failed_attempts += 1
            update_dataset_manifest(failed_attempts)
            print(
                f"episode_attempt_failed index={next_episode_index} "
                f"error={error}",
                flush=True,
            )
            if (
                (args.headless or args.stop_after_cycles)
                and failed_attempts + completed_cycles >= max(args.cycles, 1)
            ):
                break
            if failed_attempts >= 20:
                raise RuntimeError(
                    "Stopped after 20 failed data collection attempts"
                ) from error
            reset_cycle()
            continue

        if (
            (args.headless or args.stop_after_cycles)
            and target_episode_total is None
            and completed_cycles >= max(args.cycles, 1)
        ):
            break
finally:
    recorder = recording_state["recorder"]
    if recorder is not None:
        recorder.abort()
    write_brain_run_report()
    set_conveyor_speed(conveyor_graph_nodes, 0.0)
    timeline.stop()
    app.close()
