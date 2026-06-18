"""Visualize and physically execute a saved cuRobo xArm6 pick plan in Isaac Sim."""

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
parser.add_argument("--brain-device", default="cuda")
parser.add_argument("--brain-local-files-only", action="store_true")
parser.add_argument("--brain-blend", type=float, default=0.35)
parser.add_argument("--brain-max-teacher-delta", type=float, default=0.45)
parser.add_argument("--brain-max-step-delta", type=float, default=0.08)
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

ROBOT_USD = ROOT / "assets/xarm6_gripper/xarm6_gripper.usd"
ISAAC_PYTHON = Path.home() / "isaac_sim_5.1" / "python.bat"
RUNTIME_PLANNER = ROOT / "scripts/xarm6_curobo_runtime.py"
RUNTIME_REQUEST = ROOT / "outputs/curobo_runtime_request.json"
RUNTIME_PLAN = ROOT / "outputs/curobo_runtime_plan.npz"
PHYSICS_FPS = 120
CAPTURE_FPS = 4
CAPTURE_INTERVAL = PHYSICS_FPS // CAPTURE_FPS
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

import numpy as np
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
    ):
        import torch
        from PIL import Image
        from torchvision import transforms
        from train_stage3_video_sft import Stage3Policy
        from train_stage3_direct_correction import StateConditionedStage3Policy

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
        self.state_conditioned = self.policy_arch == "state_conditioned"
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
        if self.state_conditioned:
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
            if self.state_conditioned:
                if state is None:
                    return
                state_tensor = self.torch.tensor(
                    state,
                    dtype=self.torch.float32,
                    device=self.device,
                ).unsqueeze(0)
                state_tensor = (state_tensor - self.state_mean) / self.state_std
                phase_logits, action_pred = self.model(batch, state_tensor)
            else:
                phase_logits, action_pred = self.model(batch)
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
            "phase_match": phase_match,
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
        if mode == "off" or self.latest is None:
            self.stats["fallback"] += 1
            return teacher_arm, teacher_gripper, "teacher"
        if mode == "observe":
            self.stats["observe_steps"] += 1
            return teacher_arm, teacher_gripper, "observe"

        action = self.latest["action"]
        if len(action) < 14 or not np.all(np.isfinite(action)):
            self.stats["rejected"] += 1
            return teacher_arm, teacher_gripper, "reject_nonfinite"

        policy_arm = np.asarray(action[-7:-1], dtype=float)
        policy_gripper = float(np.clip(action[-1], OPEN_GRIPPER, CLOSED_GRIPPER))
        if policy_arm.shape != (6,):
            self.stats["rejected"] += 1
            return teacher_arm, teacher_gripper, "reject_shape"

        teacher_delta = float(np.max(np.abs(policy_arm - teacher_arm)))
        phase_match = self.latest["predicted_phase"] == phase
        if not args.brain_allow_phase_mismatch and not phase_match:
            self.stats["rejected"] += 1
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
    ):
        frame = self.physics_frame
        self.physics_frame += 1
        if frame % CAPTURE_INTERVAL:
            return
        if not args.headless:
            rep.orchestrator.step(
                rt_subframes=2,
                delta_time=0.0,
                pause_timeline=False,
            )
        image_name = f"rgb_{len(self.rows):06d}.png"
        rgba = np.asarray(self.annotator.get_data())
        Image.fromarray(rgba).convert("RGB").save(
            self.temp_dir / image_name
        )
        cube_position, cube_orientation = cube.get_world_pose()
        tcp_position, tcp_rotation = fk_solver.compute_end_effector_pose()
        actual_positions = np.asarray(robot.get_joint_positions())
        actual_velocities = np.asarray(robot.get_joint_velocities())
        oracle_arm = np.asarray(arm_target, dtype=float)
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
        self.rows.append(
            {
                "frame": frame,
                "time_seconds": frame / PHYSICS_FPS,
                "phase": phase,
                "image": image_name,
                "action": {
                    "arm_joint_positions": oracle_arm.tolist(),
                    "gripper_joint_position": float(gripper_target),
                },
                "executed_action": {
                    "arm_joint_positions": executed_arm.tolist(),
                    "gripper_joint_position": executed_gripper,
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
                },
            }
        )

    def finish(self, metrics, success=True, error=None):
        if len(self.rows) < 2:
            raise RuntimeError("Episode has too few captured observations")
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
            "capture_interval_frames": CAPTURE_INTERVAL,
            "resolution": list(CAPTURE_RESOLUTION),
            "frames_written": len(self.rows),
            "minimum_episode_frames": MIN_EPISODE_FRAMES,
            "task": (
                "Detect and pick the cube at the conveyor end, avoid "
                "obstacles, place it at the conveyor start, and wait for "
                "the conveyor to return it."
            ),
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
        "conveyor_speed_mps": args.conveyor_speed,
        "cycles": live_run_state["cycles"],
        "failures": live_run_state["failures"],
        "brain_runtime": brain_runtime.summary(),
        "terminal_servo": terminal_servo_state,
        "place_servo": place_servo_state,
        "link_clearance": clearance_state,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def current_brain_state():
    positions = np.asarray(robot.get_joint_positions(), dtype=float)
    arm_state = [
        float(positions[dof_index[f"joint{i}"]])
        for i in range(1, 7)
    ]
    gripper_state = float(positions[dof_index[GRIPPER_JOINTS[0]]])
    tcp_position, _ = fk_solver.compute_end_effector_pose()
    cube_position = cube.get_world_pose()[0]
    return (
        arm_state
        + [gripper_state]
        + np.asarray(tcp_position, dtype=float).tolist()
        + np.asarray(cube_position, dtype=float).tolist()
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
):
    recorder = recording_state["recorder"]
    if recorder is not None:
        recorder.capture(
            phase,
            arm_target,
            gripper_target,
            executed_arm,
            executed_gripper,
        )
    if brain_runtime is not None and rgb_annotator is not None:
        brain_runtime.observe(rgb_annotator, phase, current_brain_state())


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


def check_link_clearance(phase):
    if not clearance_state["enabled"]:
        return
    points = link_world_points()
    if not points:
        return
    bounds = dynamic_clearance_bounds(phase)
    if not bounds:
        return
    best = {
        "clearance_m": float("inf"),
        "link": None,
        "obstacle": None,
        "phase": phase,
        "threshold_m": float(clearance_state["threshold_m"]),
        "kind": "obstacle",
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
            distance = point_to_aabb_distance(
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
                }
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
        target_arm, target_gripper, source = brain_runtime.choose_target(
            args.brain_control,
            teacher_arm,
            float(teacher_gripper),
            current_arm,
            current_gripper,
            phase,
        )
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


def tcp_to_cube_distance():
    cube_position = np.asarray(cube.get_world_pose()[0], dtype=float)
    return float(np.linalg.norm(cube_position - grasp_center_position()))


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
    if payload_state["held"]:
        print(
            f"grasp_release cube={np.asarray(cube.get_world_pose()[0]).round(4).tolist()}",
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


def step_for(frames, arm_positions, gripper_position, phase):
    for _ in range(frames):
        if (
            control_state["end_requested"]
            or control_state["retry_requested"]
            or not app.is_running()
        ):
            return False
        control_arm, control_gripper = choose_control_target(
            arm_positions,
            gripper_position,
            phase,
        )
        apply_target(
            robot,
            dof_index,
            control_arm,
            control_gripper,
        )
        guarded_world_step(phase)
        update_attached_payload()
        capture_step(
            phase,
            arm_positions,
            gripper_position,
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


def step_direct_target(frames, arm_positions, gripper_position, phase):
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
        guarded_world_step(phase)
        update_attached_payload()
        capture_step(
            phase,
            arm_positions,
            gripper_position,
            arm_positions,
            gripper_position,
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
    return step_direct_target(
        frames,
        current_arm,
        gripper_position,
        "close_gripper",
    )


def align_for_no_teleport_grasp(arm_positions, gripper_position):
    if not args.brain_terminal_servo or args.grasp_mode == "teleport":
        return True
    target_distance = float(args.grasp_attach_distance) * 0.85
    max_frames = max(int(args.brain_terminal_servo_align_frames), 0)
    for frame in range(max_frames):
        distance = tcp_to_cube_distance()
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
        capture_step(
            "observe_cube",
            current_arm_positions,
            OPEN_GRIPPER,
        )
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
            if not run_vertical_grasp_approach(OPEN_GRIPPER):
                return False
            if not align_for_no_teleport_grasp(
                current_arm_positions(),
                OPEN_GRIPPER,
            ):
                return False
            gripper_position = CLOSED_GRIPPER
            if not close_gripper_in_place(100, gripper_position):
                return False
            if not begin_payload_grasp():
                return False
        elif phase_name == "retreat_after_release":
            if not align_for_no_teleport_place(
                trajectory[0],
                gripper_position,
            ):
                return False
            gripper_position = OPEN_GRIPPER
            if not step_for(
                90,
                trajectory[0],
                gripper_position,
                "open_gripper",
            ):
                return False
            release_payload()

        for arm_positions in trajectory:
            if not step_for(
                max(args.steps_per_waypoint, 1),
                arm_positions,
                gripper_position,
                phase_name,
            ):
                return False

    final_cube = np.asarray(cube.get_world_pose()[0])
    place_position = np.asarray(metadata["place_position"])
    place_distance = float(
        np.linalg.norm(final_cube[:2] - place_position[:2])
    )
    lifted_height = (
        float(payload_state["max_cube_height_m"])
        - float(cube_initial_position[2])
    )
    print(
        f"playback_complete cube={final_cube.round(4).tolist()} "
        f"place_distance={place_distance:.4f}m "
        f"lifted_height={lifted_height:.4f}m",
        flush=True,
    )
    if status_label is not None:
        status_label.text = "Conveyor returning cube..."
    return {
        "placed_at_start": place_distance < 0.10,
        "start_place_distance_m": place_distance,
        "cube_at_start": final_cube.tolist(),
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
            or not app.is_running()
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
    reset_cycle()
    completed_cycles = 0
    dataset_completed = existing_successful_episodes
    failed_attempts = 0
    update_dataset_manifest(failed_attempts)
    while app.is_running():
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
            pick_metrics = run_pick_and_place(trajectories)
            if not pick_metrics:
                raise RuntimeError("Pick-and-place was interrupted")
            if not pick_metrics["placed_at_start"]:
                raise RuntimeError(
                    "Cube was not placed at the conveyor start: "
                    f"distance={pick_metrics['start_place_distance_m']:.4f}m"
                )
            hold_positions = np.asarray(
                trajectories["retreat_after_release"]
            )[-1]
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
                    failure_place_position = np.asarray(
                        metadata["place_position"],
                        dtype=float,
                    )
                    failure_place_distance = float(
                        np.linalg.norm(
                            failure_cube_position[:2]
                            - failure_place_position[:2]
                        )
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
                        "placed_at_start": failure_place_distance < 0.10,
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
                args.headless
                and target_episode_total is None
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
            args.headless
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
