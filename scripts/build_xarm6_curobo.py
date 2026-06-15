"""Build a cuRobo V2 robot configuration for the local xArm6 gripper model."""

import argparse
from pathlib import Path

from curobo_bootstrap import configure_curobo_imports


parser = argparse.ArgumentParser()
parser.add_argument("--sphere-density", type=float, default=1.2)
parser.add_argument("--collision-samples", type=int, default=500)
parser.add_argument(
    "--output",
    type=Path,
    default=Path("config/xarm6_curobo.yml"),
)
args = parser.parse_args()

configure_curobo_imports()

from curobo.robot_builder import RobotBuilder


ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = ROOT / "assets" / "xarm6_gripper"
URDF_PATH = ASSET_ROOT / "xarm6_gripper_control.urdf"

builder = RobotBuilder(
    urdf_path=str(URDF_PATH),
    asset_path=str(ASSET_ROOT),
    tool_frames=["link_tcp"],
)
print(
    f"Parsed xArm6: links={len(builder._link_names)} "
    f"mesh_links={len(builder._mesh_link_names)}"
)

builder.fit_collision_spheres(
    sphere_density=args.sphere_density,
    compute_metrics=True,
)
print(
    f"Fitted {builder.num_spheres} collision spheres across "
    f"{len(builder.collision_link_names)} links"
)

builder.compute_collision_matrix(
    prune_collisions=True,
    num_samples=args.collision_samples,
)
config = builder.build()
kinematics = config["kinematics"]
gripper_joints = [
    "drive_joint",
    "left_finger_joint",
    "left_inner_knuckle_joint",
    "right_outer_knuckle_joint",
    "right_finger_joint",
    "right_inner_knuckle_joint",
]
kinematics["lock_joints"] = {name: 0.425 for name in gripper_joints}

output = args.output.resolve()
output.parent.mkdir(parents=True, exist_ok=True)
builder.save(config, str(output))
print(f"Wrote {output}")
