import argparse
import json
import math
import os
from pathlib import Path

from isaacsim import SimulationApp


parser = argparse.ArgumentParser(
    description="Generate xArm6 conveyor-cycle demonstrations for V-JEPA2-AC."
)
parser.add_argument("--headless", action="store_true")
parser.add_argument("--episodes", type=int, default=1)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--record-root", type=Path, default=Path("outputs/conveyor_dataset"))
parser.add_argument("--preview-only", action="store_true")
parser.add_argument(
    "--use-conveyor-graph",
    dest="use_conveyor_graph",
    action="store_true",
    help="Use Isaac Sim's official conveyor surface velocity (enabled by default).",
)
parser.add_argument(
    "--no-conveyor-graph",
    dest="use_conveyor_graph",
    action="store_false",
    help="Disable conveyor surface velocity for scene inspection only.",
)
parser.set_defaults(use_conveyor_graph=True)
parser.add_argument("--conveyor-speed", type=float, default=0.25)
args, _ = parser.parse_known_args()

ROOT = Path(__file__).resolve().parents[1]
ROBOT_USD = ROOT / "assets/xarm6_gripper/xarm6_gripper.usd"
os.environ.setdefault("WP_CACHE_PATH", str(ROOT / "outputs/warp_cache"))

app = SimulationApp(
    {
        "headless": args.headless,
        "renderer": "RaytracedLighting",
        "width": 1280,
        "height": 720,
    }
)

import carb.settings
import numpy as np
import omni.kit.commands
import omni.replicator.core as rep
import omni.timeline
import omni.ui as ui
import omni.usd
from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics, UsdShade
from isaacsim.core.api import World
from isaacsim.core.api.materials import PhysicsMaterial
from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.extensions import enable_extension
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.robot_motion.motion_generation import (
    ArticulationKinematicsSolver,
    LulaKinematicsSolver,
)
from PIL import Image


PHYSICS_FPS = 120
CAPTURE_FPS = 4
CAPTURE_INTERVAL = PHYSICS_FPS // CAPTURE_FPS
BELT_RADIUS = 0.43
BELT_HEIGHT = 0.045
BELT_WIDTH = 0.13
BELT_SEGMENTS = 17
BELT_START_ANGLE = math.radians(-105)
BELT_END_ANGLE = math.radians(105)
GRIP_STATIC_FRICTION = 3.0
GRIP_DYNAMIC_FRICTION = 2.5
DOWNWARD = np.array([0.0, 1.0, 0.0, 0.0])
OPEN_GRIPPER = 0.0
CLOSED_GRIPPER = 0.70
OBSTACLE_CLEARANCE = 0.012
SELF_CLEARANCE = 0.020
GRIPPER_JOINTS = [
    "drive_joint",
    "left_inner_knuckle_joint",
    "right_inner_knuckle_joint",
    "right_outer_knuckle_joint",
    "left_finger_joint",
    "right_finger_joint",
]


def arc_position(angle, z=BELT_HEIGHT):
    return np.array(
        [
            BELT_RADIUS * math.cos(angle),
            BELT_RADIUS * math.sin(angle),
            z,
        ]
    )


def create_arc_conveyor(world):
    """Build an offline semicircle and optionally add official conveyor graphs."""
    segment_length = BELT_RADIUS * (
        BELT_END_ANGLE - BELT_START_ANGLE
    ) / (BELT_SEGMENTS - 1)
    segments = []
    graph_nodes = []
    belt_material = PhysicsMaterial(
        prim_path="/World/Materials/ConveyorPhysics",
        static_friction=1.5,
        dynamic_friction=1.2,
        restitution=0.0,
    )

    if args.use_conveyor_graph:
        enable_extension("isaacsim.asset.gen.conveyor")
        app.update()

    for index, angle in enumerate(
        np.linspace(BELT_START_ANGLE, BELT_END_ANGLE, BELT_SEGMENTS)
    ):
        tangent_yaw = angle + math.pi / 2
        segment = world.scene.add(
            FixedCuboid(
                prim_path=f"/World/Conveyor/Segment_{index:02d}",
                name=f"conveyor_segment_{index:02d}",
                position=arc_position(angle),
                orientation=euler_angles_to_quat(
                    np.array([0.0, 0.0, tangent_yaw])
                ),
                scale=np.array([segment_length * 1.10, BELT_WIDTH, 0.035]),
                color=np.array([0.08, 0.12, 0.16]),
                physics_material=belt_material,
            )
        )
        segments.append(segment)

        if args.use_conveyor_graph:
            # Conveyor surface velocity requires a rigid body. Keep each visual
            # segment kinematic so the graph cannot turn the arc into free
            # dynamic bodies that fall under gravity.
            rigid_body = UsdPhysics.RigidBodyAPI.Apply(segment.prim)
            rigid_body.CreateRigidBodyEnabledAttr().Set(True)
            rigid_body.CreateKinematicEnabledAttr().Set(True)
            _, graph_node = omni.kit.commands.execute(
                "CreateConveyorBelt",
                conveyor_prim=segment.prim,
            )
            graph_node.GetAttribute("inputs:direction").Set(
                (1.0, 0.0, 0.0)
            )
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
                np.array([0.0, 0.0, BELT_END_ANGLE + math.pi / 2])
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
    if len(graph_nodes) != len(angles):
        raise RuntimeError(
            f"Expected {len(angles)} conveyor nodes, got {len(graph_nodes)}"
        )
    for graph_node, angle in zip(graph_nodes, angles):
        targets = graph_node.GetRelationship("inputs:conveyorPrim").GetTargets()
        if len(targets) != 1:
            raise RuntimeError(
                f"Invalid conveyor target at {graph_node.GetPath()}: {targets}"
            )
        conveyor_prim = omni.usd.get_context().get_stage().GetPrimAtPath(
            targets[0]
        )
        surface_velocity = PhysxSchema.PhysxSurfaceVelocityAPI.Apply(
            conveyor_prim
        )
        surface_velocity.GetSurfaceVelocityLocalSpaceAttr().Set(False)
        tangent = np.array([-math.sin(angle), math.cos(angle), 0.0])
        surface_velocity.GetSurfaceVelocityAttr().Set(
            Gf.Vec3f(*(tangent * float(speed)))
        )
        velocity_attr = graph_node.GetParent().GetAttribute(
            "graph:variable:Velocity"
        )
        if not velocity_attr:
            raise RuntimeError(
                f"Missing conveyor Velocity variable at {graph_node.GetPath()}"
            )
        velocity_attr.Set(float(speed))


def verify_conveyor_segments_fixed(segments, expected_positions, tolerance=1e-4):
    errors = []
    for segment, expected in zip(segments, expected_positions):
        actual = np.asarray(segment.get_world_pose()[0])
        error = float(np.linalg.norm(actual - expected))
        if error > tolerance:
            errors.append(f"{segment.name}: drift={error:.6f} m")
    if errors:
        raise RuntimeError("Conveyor segments moved:\n" + "\n".join(errors))


def random_obstacle_states(rng):
    states = []
    obstacle_specs = [
        # These occupy the direct transfer corridor but leave a path on either side.
        (np.array([0.24, -0.06, 0.19]), np.array([0.10, 0.12, 0.34])),
        (np.array([0.28, 0.11, 0.13]), np.array([0.08, 0.10, 0.24])),
    ]
    for index, (base_position, base_scale) in enumerate(obstacle_specs):
        jitter = rng.uniform(
            low=np.array([-0.025, -0.035, 0.0]),
            high=np.array([0.025, 0.035, 0.04]),
        )
        scale = base_scale * rng.uniform(0.80, 1.20, size=3)
        states.append((base_position + jitter, scale))
    return states


def create_random_obstacles(world, rng):
    obstacles = []
    for index, (position, scale) in enumerate(random_obstacle_states(rng)):
        obstacle = world.scene.add(
            FixedCuboid(
                prim_path=f"/World/Obstacles/Obstacle_{index}",
                name=f"obstacle_{index}",
                position=position,
                scale=scale,
                color=np.array([0.95, 0.55 - index * 0.15, 0.05]),
            )
        )
        obstacles.append(obstacle)
    return obstacles


def reset_obstacles(obstacles, rng):
    for obstacle, (position, scale) in zip(
        obstacles,
        random_obstacle_states(rng),
    ):
        obstacle.set_world_pose(position=position)
        obstacle.set_local_scale(scale)


def solve_pose(solver, position):
    action, success = solver.compute_inverse_kinematics(
        target_position=np.asarray(position),
        target_orientation=DOWNWARD,
        position_tolerance=0.004,
        orientation_tolerance=0.03,
    )
    if not success:
        raise RuntimeError(f"IK failed for target {position}")
    return np.asarray(action.joint_positions)


def set_target(robot, arm_positions, gripper_position, dof_index):
    target = robot.get_joint_positions().copy()
    for joint_number, value in enumerate(arm_positions, start=1):
        target[dof_index[f"joint{joint_number}"]] = value
    for name in GRIPPER_JOINTS:
        target[dof_index[name]] = gripper_position
    robot.get_articulation_controller().apply_action(
        ArticulationAction(joint_positions=target)
    )


def set_pose_immediately(robot, arm_positions, gripper_position, dof_index):
    target = robot.get_joint_positions().copy()
    for joint_number, value in enumerate(arm_positions, start=1):
        target[dof_index[f"joint{joint_number}"]] = value
    for name in GRIPPER_JOINTS:
        target[dof_index[name]] = gripper_position
    robot.set_joint_positions(target)
    robot.set_joint_velocities(np.zeros_like(target))


def blend(start, end, alpha):
    alpha = np.clip(alpha, 0.0, 1.0)
    smooth = alpha * alpha * (3.0 - 2.0 * alpha)
    return start + (end - start) * smooth


def choose_teacher_detour(obstacles, safe_height):
    """Route through the negative-X corridor, opposite the obstacle field."""
    del obstacles
    return np.array([-0.24, -0.32, safe_height])


def obstacle_safe_height(obstacles):
    del obstacles
    # The route stays in the negative-X corridor, away from the positive-X
    # obstacle field. This height is reachable without switching IK branches.
    return 0.34


def collision_prims(root_prim):
    rigid_bodies = [
        prim
        for prim in Usd.PrimRange(root_prim)
        if prim.HasAPI(UsdPhysics.RigidBodyAPI)
    ]
    if rigid_bodies:
        return rigid_bodies
    return [
        prim
        for prim in Usd.PrimRange(root_prim)
        if prim.HasAPI(UsdPhysics.CollisionAPI)
    ]


def world_aabb(cache, prim):
    bounds = cache.ComputeWorldBound(prim).ComputeAlignedBox()
    minimum = np.asarray(bounds.GetMin())
    maximum = np.asarray(bounds.GetMax())
    if (
        not np.all(np.isfinite(minimum))
        or not np.all(np.isfinite(maximum))
        or np.any(maximum < minimum)
        or np.max(np.abs(np.concatenate([minimum, maximum]))) > 1e6
    ):
        raise RuntimeError(f"Invalid world bounds for {prim.GetPath()}")
    return minimum, maximum


def aabb_distance(first, second):
    first_min, first_max = first
    second_min, second_max = second
    gap = np.maximum(
        np.maximum(first_min - second_max, second_min - first_max),
        0.0,
    )
    return float(np.linalg.norm(gap))


def robot_obstacle_clearance(stage, robot_colliders, obstacles):
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
        useExtentsHint=False,
    )
    obstacle_prims = [
        stage.GetPrimAtPath(obstacle.prim_path)
        for obstacle in obstacles
    ]
    closest = (float("inf"), None, None)
    valid_robot_bounds = 0
    for robot_prim in robot_colliders:
        try:
            robot_bounds = world_aabb(cache, robot_prim)
        except RuntimeError:
            continue
        valid_robot_bounds += 1
        for obstacle_prim in obstacle_prims:
            distance = aabb_distance(
                robot_bounds,
                world_aabb(cache, obstacle_prim),
            )
            if distance < closest[0]:
                closest = (distance, robot_prim, obstacle_prim)
    if valid_robot_bounds == 0 or closest[1] is None:
        raise RuntimeError("No valid robot link bounds were available")
    return closest


def assert_robot_obstacle_clearance(stage, robot_colliders, obstacles, phase):
    distance, robot_prim, obstacle_prim = robot_obstacle_clearance(
        stage,
        robot_colliders,
        obstacles,
    )
    if distance < OBSTACLE_CLEARANCE:
        raise RuntimeError(
            "Robot entered the obstacle safety envelope: "
            f"phase={phase}, link={robot_prim.GetPath()}, "
            f"obstacle={obstacle_prim.GetPath()}, "
            f"clearance={distance:.4f}m, "
            f"required={OBSTACLE_CLEARANCE:.4f}m"
        )
    return distance


def arm_self_clearance(stage):
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
        useExtentsHint=False,
    )
    chain_names = [
        "link_base",
        "link1",
        "link2",
        "link3",
        "link4",
        "link5",
        "link6",
        "xarm_gripper_base_link",
    ]
    chain = []
    for index, name in enumerate(chain_names):
        prim = stage.GetPrimAtPath(f"/UF_ROBOT/root_joint/{name}")
        if not prim.IsValid():
            continue
        try:
            bounds = world_aabb(cache, prim)
        except RuntimeError:
            continue
        chain.append((index, prim, bounds))

    closest = (float("inf"), None, None)
    for first_index, first_prim, first_bounds in chain:
        for second_index, second_prim, second_bounds in chain:
            # Adjacent links and links separated by one compact wrist joint
            # naturally have overlapping AABBs on this robot.
            if first_index > 2 or second_index - first_index < 3:
                continue
            distance = aabb_distance(first_bounds, second_bounds)
            if distance < closest[0]:
                closest = (distance, first_prim, second_prim)
    if closest[1] is None:
        raise RuntimeError("No valid non-adjacent robot link pairs were available")
    return closest


def assert_arm_self_clearance(stage, phase):
    distance, first_prim, second_prim = arm_self_clearance(stage)
    if distance < SELF_CLEARANCE:
        raise RuntimeError(
            "Robot links entered the self-collision safety envelope: "
            f"phase={phase}, first={first_prim.GetPath()}, "
            f"second={second_prim.GetPath()}, "
            f"clearance={distance:.4f}m, "
            f"required={SELF_CLEARANCE:.4f}m"
        )
    return distance


def assert_gripper_above_ground(stage):
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
        useExtentsHint=False,
    )
    minimum_z = min(
        world_aabb(cache, stage.GetPrimAtPath(path))[0][2]
        for path in (
            "/UF_ROBOT/root_joint/left_finger",
            "/UF_ROBOT/root_joint/right_finger",
        )
    )
    if minimum_z < 0.005:
        raise RuntimeError(
            "Initial gripper pose intersects or nearly touches the ground: "
            f"minimum_z={minimum_z:.4f}m"
        )


def create_recorder():
    carb.settings.get_settings().set("/omni/replicator/captureOnPlay", False)
    camera = rep.create.camera(
        position=(1.25, 1.25, 0.92),
        look_at=(0.15, 0.0, 0.18),
    )
    rep.create.light(rotation=(315, 0, 0), intensity=2500, light_type="distant")
    rep.create.light(intensity=450, light_type="dome")
    render_product = rep.create.render_product(camera, (256, 256))
    annotator = rep.AnnotatorRegistry.get_annotator("rgb")
    annotator.attach(render_product)
    rep.orchestrator.preview()
    return annotator


def bind_finger_material(stage, physics_material):
    bound_colliders = []
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
            colliders = [finger]
        for collider in colliders:
            UsdShade.MaterialBindingAPI.Apply(collider).Bind(
                physics_material.material,
                bindingStrength=UsdShade.Tokens.strongerThanDescendants,
                materialPurpose="physics",
            )
            bound_colliders.append(str(collider.GetPath()))
    return bound_colliders


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


def record_step(
    annotator,
    record_dir,
    rows,
    frame,
    phase,
    arm_target,
    gripper_target,
    robot,
    cube,
    obstacles,
):
    if annotator is None or frame % CAPTURE_INTERVAL:
        return
    rep.orchestrator.step(rt_subframes=2, delta_time=0.0, pause_timeline=False)
    image_name = f"rgb_{frame:06d}.png"
    Image.fromarray(annotator.get_data()).convert("RGB").save(
        record_dir / image_name
    )
    cube_position, cube_orientation = cube.get_world_pose()
    rows.append(
        {
            "frame": frame,
            "time_seconds": frame / PHYSICS_FPS,
            "phase": phase,
            "image": image_name,
            "action": {
                "arm_joint_positions": np.asarray(arm_target).tolist(),
                "gripper_joint_position": float(gripper_target),
            },
            "observation": {
                "joint_positions": robot.get_joint_positions().tolist(),
                "cube_position": cube_position.tolist(),
                "cube_orientation_wxyz": cube_orientation.tolist(),
                "obstacles": [
                    {
                        "position": obstacle.get_world_pose()[0].tolist(),
                        "scale": obstacle.get_local_scale().tolist(),
                    }
                    for obstacle in obstacles
                ],
            },
        }
    )


def write_episode(record_dir, rows, seed, metrics):
    with (record_dir / "actions.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    metadata = {
        "seed": seed,
        "physics_fps": PHYSICS_FPS,
        "capture_fps": CAPTURE_FPS,
        "resolution": [256, 256],
        "task": (
            "move cube from conveyor end to start while avoiding obstacles, "
            "then wait for the conveyor to return it to the end"
        ),
        "teacher_policy": "IK waypoints with randomized geometric detour",
        "frames_written": len(rows),
        "validation": metrics,
    }
    (record_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
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
conveyor_expected_positions = [
    arc_position(angle)
    for angle in np.linspace(
        BELT_START_ANGLE,
        BELT_END_ANGLE,
        BELT_SEGMENTS,
    )
]

grip_material = PhysicsMaterial(
    prim_path="/World/Materials/GripperPhysics",
    static_friction=GRIP_STATIC_FRICTION,
    dynamic_friction=GRIP_DYNAMIC_FRICTION,
    restitution=0.0,
)
robot = world.scene.add(
    SingleArticulation(
        prim_path="/UF_ROBOT/root_joint/root_joint",
        name="xarm6",
    )
)
bound_colliders = bind_finger_material(
    omni.usd.get_context().get_stage(),
    grip_material,
)
if not bound_colliders:
    raise RuntimeError("Could not bind the high-friction material to finger colliders")

annotator = None if args.preview_only else create_recorder()
world.reset()

dof_index = {name: index for index, name in enumerate(robot.dof_names)}
missing = [
    name
    for name in [f"joint{i}" for i in range(1, 7)] + GRIPPER_JOINTS
    if name not in dof_index
]
if missing:
    raise RuntimeError(f"Missing expected joints: {missing}")
configure_robot(robot, dof_index)

lula = LulaKinematicsSolver(
    robot_description_path=str(ROOT / "config/xarm6_robot_descriptor.yaml"),
    urdf_path=str(ROOT / "assets/xarm6_gripper/xarm6_gripper_control.urdf"),
)
ik_solver = ArticulationKinematicsSolver(robot, lula, "link_tcp")
safe_home = solve_pose(
    ik_solver,
    arc_position(BELT_END_ANGLE, BELT_HEIGHT + 0.24),
)
stage = omni.usd.get_context().get_stage()
robot_colliders = collision_prims(
    stage.GetPrimAtPath("/UF_ROBOT/root_joint")
)
if not robot_colliders:
    raise RuntimeError("No robot collision geometry was found for safety checks")

if not args.headless:
    set_camera_view(
        eye=np.array([1.15, 1.15, 0.82]),
        target=np.array([0.18, 0.0, 0.18]),
        camera_prim_path="/OmniverseKit_Persp",
    )

timeline = omni.timeline.get_timeline_interface()
timeline.play()
rng = np.random.default_rng(args.seed)

retry_state = {"requested": False}
retry_window = None
retry_status = None
if not args.headless:
    retry_window = ui.Window("xArm6 Test Control", width=280, height=120)
    with retry_window.frame:
        with ui.VStack(spacing=8):
            retry_status = ui.Label("Running...")
            ui.Button(
                "Run Test Again",
                height=36,
                clicked_fn=lambda: retry_state.update(requested=True),
            )

episode_index = 0
obstacles = None
cube = None
while app.is_running():
    if episode_index >= args.episodes and not retry_state["requested"]:
        if args.headless:
            break
        retry_status.text = "Ready - click the button to run again"
        app.update()
        continue
    retry_state["requested"] = False
    if retry_status is not None:
        retry_status.text = "Running..."
        app.update()
    seed_offset = min(episode_index, max(args.episodes - 1, 0))
    episode_seed = args.seed + seed_offset
    episode_rng = np.random.default_rng(episode_seed)

    conveyor_start = arc_position(BELT_START_ANGLE, BELT_HEIGHT + 0.04)
    conveyor_end = arc_position(BELT_END_ANGLE, BELT_HEIGHT + 0.04)
    if obstacles is None:
        obstacles = create_random_obstacles(world, episode_rng)
    if cube is None:
        cube = world.scene.add(
            DynamicCuboid(
                prim_path="/World/Objects/CycleCube",
                name="cycle_cube",
                position=conveyor_end,
                size=0.04,
                color=np.array([0.82, 0.08, 0.05]),
                mass=0.05,
                physics_material=grip_material,
            )
        )
    world.reset()
    reset_obstacles(obstacles, np.random.default_rng(episode_seed))
    cube.set_world_pose(position=conveyor_end)
    cube.set_linear_velocity(np.zeros(3))
    cube.set_angular_velocity(np.zeros(3))
    configure_robot(robot, dof_index)
    set_pose_immediately(
        robot,
        safe_home,
        OPEN_GRIPPER,
        dof_index,
    )
    set_conveyor_speed(conveyor_graph_nodes, 0.0)
    for _ in range(10):
        world.step(render=not args.headless)
    minimum_robot_clearance = assert_robot_obstacle_clearance(
        stage,
        robot_colliders,
        obstacles,
        "safe_home",
    )
    minimum_self_clearance = assert_arm_self_clearance(stage, "safe_home")
    assert_gripper_above_ground(stage)
    verify_conveyor_segments_fixed(
        conveyor_segments,
        conveyor_expected_positions,
    )

    initial_cube_xy = np.asarray(cube.get_world_pose()[0])[:2]
    for _ in range(PHYSICS_FPS):
        world.step(render=not args.headless)
    settled_cube_xy = np.asarray(cube.get_world_pose()[0])[:2]
    initial_stability_drift = float(
        np.linalg.norm(settled_cube_xy - initial_cube_xy)
    )
    if initial_stability_drift > 0.003:
        raise RuntimeError(
            "Cube did not remain stable at the conveyor end: "
            f"drift={initial_stability_drift:.4f} m"
        )

    safe_height = obstacle_safe_height(obstacles)
    end_above = solve_pose(
        ik_solver,
        conveyor_end + np.array([0.0, 0.0, 0.20]),
    )
    end_grasp = solve_pose(ik_solver, conveyor_end)
    end_lift = solve_pose(
        ik_solver,
        np.array([conveyor_end[0], conveyor_end[1], safe_height]),
    )
    detour = solve_pose(
        ik_solver,
        choose_teacher_detour(obstacles, safe_height),
    )
    start_above = solve_pose(
        ik_solver,
        conveyor_start + np.array([0.0, 0.0, 0.20]),
    )
    start_place = solve_pose(ik_solver, conveyor_start)

    stages = [
        (safe_home, end_above, 180, OPEN_GRIPPER, "safe_approach"),
        (end_above, end_grasp, 100, OPEN_GRIPPER, "descend_end"),
        (end_grasp, end_grasp, 140, CLOSED_GRIPPER, "grasp_at_end"),
        (end_grasp, end_lift, 140, CLOSED_GRIPPER, "lift_from_end"),
        (end_lift, detour, 240, CLOSED_GRIPPER, "avoid_obstacle"),
        (detour, start_above, 240, CLOSED_GRIPPER, "approach_start"),
        (start_above, start_place, 100, CLOSED_GRIPPER, "descend_start"),
        (start_place, start_place, 90, OPEN_GRIPPER, "release_at_start"),
        (start_place, start_above, 120, OPEN_GRIPPER, "clear_start"),
        (start_above, detour, 220, OPEN_GRIPPER, "return_detour"),
        (detour, safe_home, 200, OPEN_GRIPPER, "return_home"),
    ]

    record_dir = (
        args.record_root.resolve() / f"episode_{episode_index:05d}_seed_{episode_seed}"
    )
    if not args.preview_only:
        record_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    frame = 0
    max_cube_height = float(cube.get_world_pose()[0][2])

    for arm_start, arm_end, duration, gripper, phase in stages:
        print(f"episode={episode_index} phase={phase}", flush=True)
        for local_frame in range(duration):
            arm_target = blend(
                arm_start,
                arm_end,
                local_frame / max(duration - 1, 1),
            )
            set_target(robot, arm_target, gripper, dof_index)
            minimum_robot_clearance = min(
                minimum_robot_clearance,
                assert_robot_obstacle_clearance(
                    stage,
                    robot_colliders,
                    obstacles,
                    phase,
                ),
            )
            minimum_self_clearance = min(
                minimum_self_clearance,
                assert_arm_self_clearance(stage, phase),
            )
            world.step(render=not args.headless)
            minimum_robot_clearance = min(
                minimum_robot_clearance,
                assert_robot_obstacle_clearance(
                    stage,
                    robot_colliders,
                    obstacles,
                    phase,
                ),
            )
            minimum_self_clearance = min(
                minimum_self_clearance,
                assert_arm_self_clearance(stage, phase),
            )
            max_cube_height = max(
                max_cube_height,
                float(cube.get_world_pose()[0][2]),
            )
            record_step(
                annotator,
                record_dir,
                rows,
                frame,
                phase,
                arm_target,
                gripper,
                robot,
                cube,
                obstacles,
            )
            frame += 1
        print(
            f"phase_end={phase} "
            f"cube={np.asarray(cube.get_world_pose()[0]).round(3).tolist()}",
            flush=True,
        )

    cube_at_start_position = np.asarray(cube.get_world_pose()[0])
    start_place_distance = float(
        np.linalg.norm(cube_at_start_position[:2] - conveyor_start[:2])
    )
    placed_at_start = start_place_distance < 0.10
    if not placed_at_start:
        raise RuntimeError(
            "Robot did not place the cube at the conveyor start: "
            f"distance={start_place_distance:.3f} m"
        )

    if not conveyor_graph_nodes:
        raise RuntimeError(
            "The cycle requires conveyor surface velocity. "
            "Remove --no-conveyor-graph."
        )

    print(
        f"episode={episode_index} phase=conveyor_return "
        f"speed={args.conveyor_speed:.2f}m/s",
        flush=True,
    )
    set_conveyor_speed(conveyor_graph_nodes, args.conveyor_speed)
    returned_to_end = False
    conveyor_return_frames = 0
    max_return_frames = PHYSICS_FPS * 15
    while conveyor_return_frames < max_return_frames:
        set_target(robot, safe_home, OPEN_GRIPPER, dof_index)
        world.step(render=not args.headless)
        minimum_robot_clearance = min(
            minimum_robot_clearance,
            assert_robot_obstacle_clearance(
                stage,
                robot_colliders,
                obstacles,
                "conveyor_return",
            ),
        )
        minimum_self_clearance = min(
            minimum_self_clearance,
            assert_arm_self_clearance(stage, "conveyor_return"),
        )
        cube_position = np.asarray(cube.get_world_pose()[0])
        max_cube_height = max(max_cube_height, float(cube_position[2]))
        record_step(
            annotator,
            record_dir,
            rows,
            frame,
            "conveyor_return",
            safe_home,
            OPEN_GRIPPER,
            robot,
            cube,
            obstacles,
        )
        frame += 1
        conveyor_return_frames += 1
        if conveyor_return_frames % PHYSICS_FPS == 0:
            print(
                "conveyor_return "
                f"t={conveyor_return_frames / PHYSICS_FPS:.1f}s "
                f"position={cube_position.round(3).tolist()} "
                f"end_distance={np.linalg.norm(cube_position[:2] - conveyor_end[:2]):.3f}m",
                flush=True,
            )
        if np.linalg.norm(cube_position[:2] - conveyor_end[:2]) < 0.065:
            returned_to_end = True
            break

    set_conveyor_speed(conveyor_graph_nodes, 0.0)
    for _ in range(PHYSICS_FPS // 2):
        set_target(robot, safe_home, OPEN_GRIPPER, dof_index)
        world.step(render=not args.headless)
        frame += 1

    verify_conveyor_segments_fixed(
        conveyor_segments,
        conveyor_expected_positions,
    )
    final_cube_position = np.asarray(cube.get_world_pose()[0])
    final_end_distance = float(
        np.linalg.norm(final_cube_position[:2] - conveyor_end[:2])
    )
    lifted = max_cube_height > BELT_HEIGHT + 0.12
    returned_and_stopped = returned_to_end and final_end_distance < 0.10
    metrics = {
        "lifted": lifted,
        "initial_end_stability_drift_m": initial_stability_drift,
        "placed_at_start": placed_at_start,
        "start_place_distance_m": start_place_distance,
        "returned_to_end": returned_and_stopped,
        "conveyor_speed_mps": args.conveyor_speed,
        "conveyor_return_seconds": conveyor_return_frames / PHYSICS_FPS,
        "max_cube_height_m": max_cube_height,
        "final_end_distance_m": final_end_distance,
        "minimum_robot_obstacle_clearance_m": minimum_robot_clearance,
        "required_robot_obstacle_clearance_m": OBSTACLE_CLEARANCE,
        "minimum_arm_self_clearance_m": minimum_self_clearance,
        "required_arm_self_clearance_m": SELF_CLEARANCE,
    }
    if not lifted or not returned_and_stopped:
        raise RuntimeError(
            "Episode failed physical validation: "
            f"lifted={lifted}, returned_to_end={returned_and_stopped}, "
            f"max_z={max_cube_height:.3f}, "
            f"end_distance={final_end_distance:.3f}"
        )
    print(f"Physical validation passed: {metrics}", flush=True)
    if not args.preview_only:
        write_episode(record_dir, rows, episode_seed, metrics)
        print(f"Wrote {record_dir}", flush=True)

    if retry_status is not None:
        retry_status.text = "Ready - click the button to run again"
    episode_index += 1

timeline.stop()
app.close()
