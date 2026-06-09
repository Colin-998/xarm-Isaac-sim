from pathlib import Path

from isaacsim import SimulationApp


ROOT = Path(__file__).resolve().parents[1]
app = SimulationApp({"headless": True})

import numpy as np
import omni.usd
from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation
from isaacsim.robot_motion.motion_generation import ArticulationKinematicsSolver, LulaKinematicsSolver


omni.usd.get_context().open_stage(str(ROOT / "assets/xarm6_gripper/xarm6_gripper.usd"))
for _ in range(10):
    app.update()

world = World(stage_units_in_meters=1.0)
robot = world.scene.add(
    SingleArticulation(prim_path="/UF_ROBOT/root_joint/root_joint", name="xarm6")
)
world.reset()
for _ in range(5):
    world.step(render=False)

lula = LulaKinematicsSolver(
    robot_description_path=str(ROOT / "config/xarm6_robot_descriptor.yaml"),
    urdf_path=str(ROOT / "assets/xarm6_gripper/xarm6_gripper.urdf"),
)
solver = ArticulationKinematicsSolver(robot, lula, "link_tcp")

targets = [
    ("approach", np.array([0.35, 0.0, 0.22]), np.array([0.0, 1.0, 0.0, 0.0])),
    ("grasp_10cm", np.array([0.35, 0.0, 0.10]), np.array([0.0, 1.0, 0.0, 0.0])),
    ("grasp_07cm", np.array([0.35, 0.0, 0.07]), np.array([0.0, 1.0, 0.0, 0.0])),
    ("lift", np.array([0.35, 0.0, 0.30]), np.array([0.0, 1.0, 0.0, 0.0])),
]

results = []
for name, position, orientation in targets:
    action, success = solver.compute_inverse_kinematics(position, orientation)
    results.append(f"{name} success={success} positions={action.joint_positions}")

(ROOT / "ik_results.txt").write_text("\n".join(results), encoding="utf-8")

app.close()
