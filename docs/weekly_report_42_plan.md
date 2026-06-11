# Weekly Report 42 Implementation Plan

## Phase 1: Physical Grasp

- Import xArm6 and the xArm gripper as one articulation.
- Generate a control URDF without importer-broken mimic joints.
- Use convex finger collision meshes and bind friction to the actual collider prims.
- Reject a grasp unless both fingers report contact and the cube rises above 0.10 m.
- Do not use a fixed joint, attachment joint, or teleport for grasping.

Run:

```powershell
& C:\Users\User\isaac_sim_5.1\python.bat scripts\grasp_cube_demo.py
```

## Phase 2: Episode Collection

Record one randomized episode:

```powershell
& C:\Users\User\isaac_sim_5.1\python.bat scripts\grasp_cube_demo.py `
  --headless `
  --random-seed 7 `
  --record-dir outputs\episode_seed7
```

Validate the 4 FPS, 256x256 dataset:

```powershell
& C:\Users\User\isaac_sim_5.1\python.bat scripts\validate_episode.py `
  outputs\episode_seed7
```

Create an RLDS-shaped intermediate document:

```powershell
& C:\Users\User\isaac_sim_5.1\python.bat scripts\export_rlds_json.py `
  outputs\episode_seed7
```

The JSON document preserves the standard RLDS step fields. A later TensorFlow
Datasets builder can convert these records to sharded TFRecord files without
changing the simulator data schema.

## Phase 3: V-JEPA2 and LLM Alignment

- Encode the ordered RGB observations at 4 FPS with V-JEPA2.
- Keep the action, joint state, cube pose, and language instruction aligned by
  step index.
- Train or evaluate the policy head on the recorded action targets.

## Phase 4: Closed-Loop Evaluation

- Replace the scripted action source with model-predicted actions.
- Preserve the same contact and lift success checks.
- Report grasp success rate, maximum cube height, slip events, and inference
  latency across fixed random seeds.
