from pathlib import Path

from isaacsim import SimulationApp


ROOT = Path(__file__).resolve().parents[1]
URDF_PATH = ROOT / "assets" / "xarm6_gripper" / "xarm6_gripper.urdf"
USD_PATH = ROOT / "assets" / "xarm6_gripper" / "xarm6_gripper.usd"

app = SimulationApp({"headless": True})

import omni.kit.commands
from pxr import PhysxSchema, Sdf, Usd, UsdPhysics


status, config = omni.kit.commands.execute("URDFCreateImportConfig")
if not status:
    raise RuntimeError("Could not create URDF import configuration")

config.merge_fixed_joints = False
config.convex_decomp = False
config.import_inertia_tensor = True
config.fix_base = True
config.make_default_prim = True
config.distance_scale = 1.0

status, prim_path = omni.kit.commands.execute(
    "URDFParseAndImportFile",
    urdf_path=str(URDF_PATH),
    import_config=config,
    dest_path=str(USD_PATH),
    get_articulation_root=True,
)
if not status:
    raise RuntimeError("URDF importer failed")

for _ in range(5):
    app.update()

stage = Usd.Stage.Open(str(USD_PATH))
if not stage:
    raise RuntimeError(f"Could not open generated USD: {USD_PATH}")

# A readable initial pose for the arm, in degrees.
joint_targets = {
    "joint1": 0.0,
    "joint2": -30.0,
    "joint3": -60.0,
    "joint4": 0.0,
    "joint5": 90.0,
    "joint6": 0.0,
    # 0.85 rad is the open limit in the UFACTORY gripper URDF.
    "drive_joint": 48.7,
}

for prim in stage.Traverse():
    name = prim.GetName()
    if name not in joint_targets:
        continue

    drive = UsdPhysics.DriveAPI.Get(prim, "angular")
    if drive:
        drive.CreateTargetPositionAttr(joint_targets[name])

    state = PhysxSchema.JointStateAPI.Apply(prim, "angular")
    state.CreatePositionAttr(joint_targets[name])

stage.GetRootLayer().Save()

# Remove visual references emitted for links that have no visual geometry.
base_path = USD_PATH.parent / "configuration" / "xarm6_gripper_base.usd"
base_layer = Sdf.Layer.FindOrOpen(str(base_path))
if base_layer:
    for path in (
        "/UF_ROBOT/world/visuals",
        "/UF_ROBOT/link_eef/visuals",
        "/UF_ROBOT/link_tcp/visuals",
    ):
        visual = base_layer.GetPrimAtPath(path)
        if visual:
            visual.referenceList.ClearEdits()
    base_layer.Save()

print(f"Created {USD_PATH}")
print(f"Articulation root: {prim_path}")
app.close()
