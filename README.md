# xArm Isaac Sim: V-JEPA2 / SmolVLA Robotic Manipulation Sandbox

> 中文：Isaac Sim xArm6 機械手臂操作研究專案，整合 cuRobo 規劃、SmolVLA / V-JEPA2 多模態大腦控制、DAgger 資料採集、RLDS 資料集打包、輸送帶循環夾取與投籃展示任務。
>
> English: An Isaac Sim xArm6 robotic manipulation research project that integrates cuRobo planning, SmolVLA / V-JEPA2 brain-assisted control, DAgger data collection, RLDS dataset packaging, conveyor-cycle pick-and-place, and basket-throw showcase tasks.

Repository:

```text
https://github.com/Colin-998/xarm-Isaac-sim.git
```

---

## 1. Project Overview / 專案總覽

### 中文

本專案是一個以 **NVIDIA Isaac Sim 5.1** 與 **UFACTORY xArm6** 為核心的具身 AI 研究原型。它不是單純的機械手臂模型匯入範例，而是一套從機器手臂模擬、任務場景設計、cuRobo 專家規劃、資料採集、RLDS 打包、多模態模型訓練，到閉環控制展示的完整研究流程。

目前專案主要研究方向是：

- 如何在 Isaac Sim 中建立可重複的 xArm6 操作任務。
- 如何用 cuRobo 產生安全、可驗證的 expert trajectory。
- 如何把模擬資料整理成適合 V-JEPA2 / VLA / LLM 使用的訓練資料。
- 如何用 DAgger 逐步讓多模態 policy 學習接近 expert 的控制趨勢。
- 如何展示 V-JEPA2 / SmolVLA-style policy 在 Isaac Sim 中產生 action trend。

本專案目前定位為 **simulation research prototype**。它的重點是展示多模態大腦、世界模型特徵、專家規劃與安全執行之間的整合，而不是宣稱已經能直接部署到真實機械手臂。

### English

This project is an embodied AI research prototype built around **NVIDIA Isaac Sim 5.1** and the **UFACTORY xArm6** robotic arm. It is not only a robot-import demo. Instead, it provides an end-to-end workflow that connects robot simulation, task design, cuRobo expert planning, dataset collection, RLDS packaging, multimodal model training, and closed-loop policy visualization.

The current research focus includes:

- Building repeatable xArm6 manipulation tasks in Isaac Sim.
- Using cuRobo to generate safe and verifiable expert trajectories.
- Converting simulation rollouts into training data for V-JEPA2, VLA-style policies, and LLM-aligned models.
- Applying DAgger to gradually improve multimodal policy behavior toward expert trajectories.
- Demonstrating V-JEPA2 / SmolVLA-style policy action trends in Isaac Sim.

This repository is currently a **simulation research prototype**. The goal is to demonstrate the integration of multimodal brain control, world-model-like visual features, expert motion planning, and safety-aware execution. It does not claim to be ready for direct deployment on a real robot.

---

## 2. Main Tasks / 主要任務

### 2.1 Conveyor-Cycle Pick-and-Place / 輸送帶循環夾取

#### 中文

輸送帶循環任務是本專案最主要的基礎任務。場景中放置一個半圓形輸送帶，xArm6 位於輸送帶中央附近。紅色方塊會位於輸送帶終點，手臂需要偵測方塊位置、避開障礙物、垂直對準下降、夾起方塊，並將方塊放到輸送帶起點。接著輸送帶會把方塊送回終點，完成一個循環。

任務流程：

1. 方塊停在輸送帶終點。
2. xArm6 偵測方塊位置。
3. cuRobo 或多模態 policy 產生接近與抓取軌跡。
4. 夾爪垂直下降並在可夾距離內關爪。
5. 手臂夾起方塊並避開障礙物。
6. 方塊被放到輸送帶起點。
7. 輸送帶啟動，將方塊送回終點。
8. 方塊回到終點後，下一輪重新偵測與抓取。

#### English

The conveyor-cycle task is the core baseline task of this project. A semi-circular conveyor is placed around the xArm6. The red cube starts near the conveyor end. The robot must detect the cube, avoid obstacles, descend vertically, grasp the cube, move it to the conveyor start, and then wait for the conveyor to return it to the pickup end.

Task sequence:

1. The cube stays at the conveyor pickup end.
2. The xArm6 detects the cube position.
3. cuRobo or the multimodal policy generates approach and grasp motions.
4. The gripper descends vertically and closes only after reaching graspable distance.
5. The robot lifts the cube while avoiding obstacles.
6. The cube is placed at the conveyor start.
7. The conveyor moves the cube back to the pickup end.
8. After the cube returns, the next cycle starts with fresh detection.

---

### 2.2 Basket-Throw Showcase / 投籃展示任務

#### 中文

投籃展示任務用來測試多任務控制與未來 VLA 泛化能力。手臂需要先夾起紅色方塊，再透過夾爪甩動產生拋投動作，讓方塊落入場景中的籃框。

目前實作中，方塊的 release velocity 來自夾爪甩動過程估計出的速度，而不是直接手動指定固定速度。為了避免 Isaac Sim 5.1 在 release 後使用 PhysX free-flight 時偶發 shutdown 或畫面停在 release frame，本專案提供 `kinematic` basket flight visualization。它會使用夾爪甩動推估速度與重力公式穩定顯示拋物線，並在方塊進入籃框範圍後將方塊穩定落在籃內。

這個任務的目的不是取代真實物理驗證，而是提供一個穩定、可展示、可記錄的多任務具身控制可視化環境。

#### English

The basket-throw task is designed to test multi-task behavior and future VLA generalization. The robot first grasps the red cube, then performs a gripper swing motion to throw the cube into a basket in the scene.

In the current implementation, the cube release velocity is estimated from the gripper swing motion instead of being manually specified as a fixed velocity. To avoid occasional Isaac Sim 5.1 instability when switching to PhysX free-flight immediately after release, the project provides a `kinematic` basket flight visualization mode. This mode uses the gripper-derived velocity and gravity to render a stable ballistic arc, then lands the cube inside the basket when it enters the target area.

This task is not meant to replace physical validation. It is a stable, visual, recordable multi-task embodied-control showcase.

---

## 3. Key Features / 主要功能

### 中文

- **Isaac Sim 5.1 xArm6 模擬**
  - 匯入 xArm6 URDF / USD。
  - 整合 xArm gripper。
  - 建立可用於抓取、放置、避障與投籃的場景。

- **cuRobo expert planning**
  - 產生 xArm6 關節軌跡。
  - 作為 expert policy / teacher data。
  - 用於 DAgger correction 與安全基準。

- **多模態大腦控制**
  - 支援 V-JEPA2 / Stage 3 video policy。
  - 支援 SmolVLA-style action chunk policy。
  - 支援 `observe`、`filtered`、`direct` 等不同控制模式。

- **DAgger 資料採集**
  - 記錄失敗、修正與安全介入資料。
  - 讓 policy 逐步學會接近 expert behavior。
  - 不追求無限制 brain takeover，而是重視穩定展示與安全邊界。

- **安全監控**
  - link clearance monitor。
  - robot-obstacle clearance。
  - robot-cube clearance。
  - warning / stop 模式。

- **RLDS 資料集打包**
  - 4 FPS。
  - 256 x 256 RGB。
  - phase labels。
  - action labels。
  - task metadata。
  - safety metrics。

### English

- **Isaac Sim 5.1 xArm6 simulation**
  - Imports xArm6 URDF / USD assets.
  - Integrates the xArm gripper.
  - Builds manipulation scenes for grasping, placing, obstacle avoidance, and basket throwing.

- **cuRobo expert planning**
  - Generates xArm6 joint trajectories.
  - Serves as expert policy / teacher data.
  - Provides safety baselines for DAgger correction.

- **Multimodal brain-assisted control**
  - Supports V-JEPA2 / Stage 3 video policy.
  - Supports SmolVLA-style action chunk policy.
  - Supports `observe`, `filtered`, and `direct` control modes.

- **DAgger data collection**
  - Records failures, corrections, and safety interventions.
  - Helps the learned policy move closer to expert behavior.
  - Avoids unlimited brain takeover; the focus is stable demonstration with safety boundaries.

- **Safety monitoring**
  - Link clearance monitor.
  - Robot-obstacle clearance.
  - Robot-cube clearance.
  - Warning / stop modes.

- **RLDS dataset packaging**
  - 4 FPS.
  - 256 x 256 RGB.
  - Phase labels.
  - Action labels.
  - Task metadata.
  - Safety metrics.

---

## 4. Project Status / 目前進度

### 中文

已完成：

- xArm6 / gripper asset 匯入與整理。
- Isaac Sim xArm6 pick-and-place demo。
- 半圓形輸送帶場景。
- 障礙物與安全距離監控。
- cuRobo 專家軌跡規劃。
- 動態方塊偵測。
- 輸送帶循環任務。
- 4 FPS、256 x 256 時序影像與動作標籤資料採集流程。
- RLDS-shaped JSONL dataset packaging。
- V-JEPA2 + Llama projector alignment 腳本。
- Stage 2 image-text QA 資料與訓練腳本。
- Stage 3 video policy / direct correction / DAgger 腳本。
- SmolVLA-style policy training entrypoint。
- 投籃任務與籃框展示。

目前仍屬研究原型：

- 不建議直接部署到真實機械手臂。
- 模型權重、訓練輸出與大型資料集不放在 Git repository。
- 部分 Isaac Sim / PhysX 行為使用展示穩定化邏輯。
- 完全由 VLA 低階接管仍需要更多資料與安全驗證。

### English

Completed:

- xArm6 / gripper asset import and cleanup.
- Isaac Sim xArm6 pick-and-place demo.
- Semi-circular conveyor scene.
- Obstacles and clearance monitoring.
- cuRobo expert trajectory planning.
- Dynamic cube detection.
- Conveyor-cycle task.
- 4 FPS, 256 x 256 temporal image and action-label data collection.
- RLDS-shaped JSONL dataset packaging.
- V-JEPA2 + Llama projector alignment scripts.
- Stage 2 image-text QA data and training scripts.
- Stage 3 video policy / direct correction / DAgger scripts.
- SmolVLA-style policy training entrypoint.
- Basket-throw task and basket visualization.

Still research prototype:

- Not recommended for direct real-robot deployment.
- Model weights, training outputs, and large datasets are not stored in Git.
- Some Isaac Sim / PhysX behavior is stabilized for visualization.
- Full low-level VLA takeover still requires more data and stronger safety validation.

---

## 5. Repository Layout / 專案結構

```text
.
|-- assets/
|   |-- xarm6/                         # xArm6 URDF, meshes, generated USD assets
|   `-- xarm6_gripper/                 # Combined xArm6 + gripper asset
|-- config/
|   |-- xarm6_curobo.yml               # cuRobo robot config
|   `-- xarm6_robot_descriptor.yaml    # Robot descriptor
|-- docs/
|   |-- phase2_dataset_collection_zh_tw.md
|   |-- phase3_mllm_alignment_zh_tw.md
|   |-- rlds_packaging_zh_tw.md
|   |-- stage1_projector_alignment_zh_tw.md
|   |-- stage2_stage3_closed_loop_zh_tw.md
|   |-- vjepa2_conveyor_design.md
|   `-- weekly_report_42_plan.md
|-- scripts/
|   |-- conveyor_cycle_scene.py
|   |-- curobo_dynamic_pick_planner.py
|   |-- play_curobo_dynamic_pick.py
|   |-- run_smolvla_multitask_dagger.py
|   |-- train_smolvla_policy.py
|   |-- train_stage3_direct_correction.py
|   |-- train_vjepa2_llama_projector.py
|   |-- package_rlds_dataset.py
|   `-- validate_level_success.py
`-- README.md
```

### 中文

`outputs/`、`build/`、`vendor/`、`tools/`、`__pycache__/` 等資料夾已在 `.gitignore` 中排除。這些資料夾通常包含模型權重、資料集、cache、臨時輸出或第三方工具，不應直接推到 GitHub。

### English

The `outputs/`, `build/`, `vendor/`, `tools/`, and `__pycache__/` directories are ignored through `.gitignore`. These directories usually contain model weights, datasets, caches, temporary outputs, or third-party tools and should not be pushed directly to GitHub.

---

## 6. Environment / 執行環境

### 中文

主要開發環境：

- OS: Windows 11
- Simulator: NVIDIA Isaac Sim 5.1
- Robot: UFACTORY xArm6
- Planner: cuRobo
- Python runtime: Isaac Sim bundled Python
- GPU: NVIDIA RTX GPU recommended

本專案多數腳本依賴 Isaac Sim runtime，建議使用 Isaac Sim 內建 Python：

```powershell
& C:\Users\User\isaac_sim_5.1\python.bat <script.py> <args>
```

### English

Primary development environment:

- OS: Windows 11
- Simulator: NVIDIA Isaac Sim 5.1
- Robot: UFACTORY xArm6
- Planner: cuRobo
- Python runtime: Isaac Sim bundled Python
- GPU: NVIDIA RTX GPU recommended

Most scripts depend on the Isaac Sim runtime. Use the bundled Isaac Sim Python:

```powershell
& C:\Users\User\isaac_sim_5.1\python.bat <script.py> <args>
```

---

## 7. xArm6 Assets / xArm6 模型資產

### 中文

Isaac Sim-ready xArm6 asset:

```text
assets/xarm6/xarm6.urdf
```

Combined xArm6 + gripper asset:

```text
assets/xarm6_gripper/xarm6_gripper.usd
assets/xarm6_gripper/xarm6_gripper.urdf
```

模型來源：

```text
https://github.com/xArm-Developer/xarm_ros2
```

本專案使用 UFACTORY 官方 `xarm_ros2` / `xarm_description` 相關資產整理出 Isaac Sim 可匯入版本，並移除 ROS 2 control、Gazebo、transmission 等不需要於 Isaac Sim URDF importer 的元素。

### English

Isaac Sim-ready xArm6 asset:

```text
assets/xarm6/xarm6.urdf
```

Combined xArm6 + gripper asset:

```text
assets/xarm6_gripper/xarm6_gripper.usd
assets/xarm6_gripper/xarm6_gripper.urdf
```

Original asset source:

```text
https://github.com/xArm-Developer/xarm_ros2
```

This project reorganizes UFACTORY's official `xarm_ros2` / `xarm_description` assets into an Isaac Sim compatible format. ROS 2 control, Gazebo, and transmission elements that are unnecessary for Isaac Sim's URDF importer are removed.

---

## 8. Quick Start / 快速開始

### 8.1 Rebuild Combined Arm + Gripper USD / 重建手臂與夾爪 USD

```powershell
cd C:\Users\User\Documents\XArm
& C:\Users\User\isaac_sim_5.1\python.bat scripts\import_xarm6_gripper.py
```

### 8.2 Physical Grasp Demo / 實體接觸抓取展示

```powershell
cd C:\Users\User\Documents\XArm
& C:\Users\User\isaac_sim_5.1\python.bat scripts\grasp_cube_demo.py
```

中文：此 demo 使用 gripper 與 cube 的 PhysX contact，不建立固定 joint。若方塊沒有離地，demo 會視為失敗。

English: This demo uses PhysX contact between the gripper and the cube. It does not create a fixed joint. If the cube does not leave the ground, the demo fails.

---

## 9. Conveyor Cycle Demo / 輸送帶循環展示

### Preview / 預覽場景

```powershell
cd C:\Users\User\Documents\XArm
& C:\Users\User\isaac_sim_5.1\python.bat scripts\conveyor_cycle_scene.py --preview-only --use-conveyor-graph
```

### Record Episodes / 錄製資料

```powershell
cd C:\Users\User\Documents\XArm
& C:\Users\User\isaac_sim_5.1\python.bat scripts\conveyor_cycle_scene.py `
  --episodes 500 `
  --conveyor-speed 0.25 `
  --record-root outputs\conveyor_dataset
```

---

## 10. Dynamic cuRobo Pick-and-Place / 動態 cuRobo 夾取與放置

### Plan / 產生規劃

```powershell
cd C:\Users\User\Documents\XArm
& C:\Users\User\isaac_sim_5.1\python.bat scripts\curobo_dynamic_pick_planner.py
```

### Replay / 播放與驗證

```powershell
cd C:\Users\User\Documents\XArm
& C:\Users\User\isaac_sim_5.1\python.bat scripts\play_curobo_dynamic_pick.py `
  --use-static-plan `
  --cycles 1 `
  --conveyor-speed 0.25 `
  --grasp-mode relative `
  --grasp-attach-distance 0.025 `
  --grasp-outward-offset 0.015 `
  --link-clearance-action warn
```

---

## 11. Basket Throw Showcase / 投籃展示

```powershell
cd C:\Users\User\Documents\XArm
& C:\Users\User\isaac_sim_5.1\python.bat scripts\play_curobo_dynamic_pick.py `
  --use-static-plan `
  --plan outputs\smolvla_multitask_dagger\basket_throw_smoke\basket_drop\plan.npz `
  --metadata outputs\smolvla_multitask_dagger\basket_throw_smoke\basket_drop\plan.json `
  --cycles 1 `
  --stop-after-cycles `
  --conveyor-speed 0.25 `
  --brain-control filtered `
  --brain-policy outputs\smolvla_xarm6_level2\best_smolvla_policy.pt `
  --brain-local-files-only `
  --brain-blend 0.70 `
  --brain-max-teacher-delta 0.35 `
  --brain-max-step-delta 0.03 `
  --brain-terminal-servo `
  --brain-terminal-servo-step 0.055 `
  --brain-terminal-servo-max-joint-delta 0.08 `
  --brain-terminal-servo-align-frames 900 `
  --brain-terminal-servo-phases close_gripper `
  --brain-place-servo `
  --brain-place-servo-frames 1200 `
  --brain-place-servo-hover-height 0.18 `
  --grasp-mode relative `
  --grasp-attach-distance 0.025 `
  --grasp-outward-offset 0.015 `
  --link-clearance-action warn `
  --task-name basket_drop `
  --task-instruction "Pick up the red cube and throw it into the far basket target with a gripper swing while avoiding obstacles." `
  --disable-conveyor-return `
  --basket-center=0.52,-0.58,0.08 `
  --basket-velocity-mode gripper `
  --basket-flight-mode kinematic `
  --basket-min-flight-frames 240
```

### 中文說明

- `--basket-velocity-mode gripper`：方塊初速度來自夾爪甩動，不是手動固定速度。
- `--basket-flight-mode kinematic`：使用穩定拋物線可視化，避免 release 後 PhysX free-flight 不穩。
- `--basket-min-flight-frames 240`：保留足夠畫面時間觀察飛行與進籃。
- `--disable-conveyor-return`：投籃任務不進入輸送帶回流流程。

### English Notes

- `--basket-velocity-mode gripper`: the cube release velocity comes from the gripper swing motion, not from a manually fixed velocity.
- `--basket-flight-mode kinematic`: uses stable ballistic visualization to avoid unstable PhysX free-flight immediately after release.
- `--basket-min-flight-frames 240`: keeps enough visible frames to observe the flight and basket landing.
- `--disable-conveyor-return`: disables conveyor return for the basket task.

---

## 12. Dataset Collection / 資料採集

### 中文

目標資料格式：

- 4 FPS
- 256 x 256 RGB
- 時序影像
- phase labels
- joint / gripper action labels
- task metadata
- safety metrics

資料通常輸出到：

```text
outputs/
```

此資料夾不進 Git。若要分享 dataset，建議使用 Hugging Face Dataset、GitHub Releases、雲端硬碟或 Git LFS。

### English

Target data format:

- 4 FPS
- 256 x 256 RGB
- Temporal image sequences
- Phase labels
- Joint / gripper action labels
- Task metadata
- Safety metrics

Recorded data usually goes to:

```text
outputs/
```

This directory is ignored by Git. To share datasets, use Hugging Face Dataset, GitHub Releases, cloud storage, or Git LFS.

---

## 13. RLDS Packaging / RLDS 資料集打包

```powershell
cd C:\Users\User\Documents\XArm
& C:\Users\User\isaac_sim_5.1\python.bat scripts\package_rlds_dataset.py `
  --record-root outputs\conveyor_dataset `
  --output-root outputs\rlds_xarm6_curobo
```

Validation:

```powershell
cd C:\Users\User\Documents\XArm
& C:\Users\User\isaac_sim_5.1\python.bat scripts\validate_rlds_package.py `
  --dataset-root outputs\rlds_xarm6_curobo
```

中文：RLDS-shaped package 會保留 episode metadata、step-level observation、action、phase、terminal flags 與資料路徑。

English: The RLDS-shaped package preserves episode metadata, step-level observations, actions, phases, terminal flags, and data paths.

---

## 14. Multimodal Training Pipeline / 多模態訓練流程

### Stage 1: Projector Alignment / 投影層對齊

中文：凍結 V-JEPA2 視覺 encoder 與 Llama-family language model，只訓練中間 projector，讓視覺特徵可以對齊語言 embedding space。

English: Freeze the V-JEPA2 visual encoder and a Llama-family language model, then train only the projector so visual features can align with the language embedding space.

```powershell
cd C:\Users\User\Documents\XArm
& C:\Users\User\isaac_sim_5.1\python.bat scripts\train_vjepa2_llama_projector.py
```

### Stage 2: Image-Text QA / 靜態圖文問答

中文：訓練模型理解單張場景影像、物體位置、phase、相對關係與任務語意。

English: Train the model to understand static scene images, object locations, phases, spatial relations, and task semantics.

```powershell
cd C:\Users\User\Documents\XArm
& C:\Users\User\isaac_sim_5.1\python.bat scripts\build_stage2_vqa_data.py
& C:\Users\User\isaac_sim_5.1\python.bat scripts\train_stage2_vqa.py
```

### Stage 3: Video SFT / 時序動作微調

中文：輸入多幀影像、機器人狀態與任務文字，輸出下一步 action 或 action chunk。

English: Use multi-frame video, robot state, and task text to predict the next action or action chunk.

```powershell
cd C:\Users\User\Documents\XArm
& C:\Users\User\isaac_sim_5.1\python.bat scripts\train_stage3_direct_correction.py
```

### SmolVLA-style Policy / SmolVLA 風格策略

中文：以較小的 VLA-style policy 在本機 GPU 上進行多任務行為學習，保留未來接 LeRobot / SmolVLA backend 的空間。

English: Train a smaller VLA-style policy on a local GPU for multi-task behavior learning, while keeping room for future LeRobot / SmolVLA backend integration.

```powershell
cd C:\Users\User\Documents\XArm
& C:\Users\User\isaac_sim_5.1\python.bat scripts\train_smolvla_policy.py
```

---

## 15. DAgger Strategy / DAgger 策略

### 中文

本專案使用 DAgger 的方向來改善多模態大腦控制，但不無限制追求完全 brain takeover。策略是：

1. 先用 cuRobo / terminal servo / place servo 產生 expert behavior。
2. 讓 brain policy 嘗試控制。
3. 當抓取偏移、放置不準、過度接近障礙物、phase mismatch 或任務失敗時，記錄 correction samples。
4. 將 correction samples 混回下一輪訓練。
5. 以 Level 1 / Level 2 驗收條件評估展示穩定性。

### English

This project uses DAgger to improve multimodal brain control, but it does not blindly pursue unlimited brain takeover. The strategy is:

1. Use cuRobo / terminal servo / place servo to generate expert behavior.
2. Let the brain policy attempt control.
3. Record correction samples when grasping drifts, placement misses, obstacle clearance becomes risky, phase mismatch occurs, or the task fails.
4. Mix correction samples into the next training iteration.
5. Evaluate demonstration stability using Level 1 / Level 2 criteria.

Entry point:

```powershell
cd C:\Users\User\Documents\XArm
& C:\Users\User\isaac_sim_5.1\python.bat scripts\run_smolvla_multitask_dagger.py
```

---

## 16. Validation Criteria / 驗收條件

### Level 1: Single-cycle success / 單輪成功

中文條件：

1. 一次抓取成功。
2. 一次放置成功。
3. 無重大碰撞。
4. 方塊沒有被明顯撞飛。
5. 輸送帶流程能完成一輪。
6. link clearance 若只有極小幅接近門檻，記為 warning，不直接視為整體失敗。

English criteria:

1. One successful grasp.
2. One successful placement.
3. No major collision.
4. The cube is not obviously knocked away.
5. The conveyor cycle completes one full round.
6. Minor link-clearance threshold proximity is logged as a warning instead of being treated as full task failure.

### Level 2: Multi-cycle success / 連續多輪成功

中文條件：

1. 連續 3 輪以上。
2. 每輪都能抓取、放置、回流。
3. 方塊回到終點後能重新偵測並再次抓取。
4. 不發生重大碰撞或明顯任務失敗。

English criteria:

1. At least 3 consecutive cycles.
2. Each cycle completes grasp, placement, and return.
3. After returning to the pickup end, the cube is detected and grasped again.
4. No major collision or obvious task failure occurs.

Validation:

```powershell
cd C:\Users\User\Documents\XArm
& C:\Users\User\isaac_sim_5.1\python.bat scripts\validate_level_success.py
```

---

## 17. What Is Not Included / 沒有放進 Git 的內容

### 中文

以下內容通常不會放進 Git repository：

- `outputs/` 裡的 recorded episodes。
- RLDS shards。
- model checkpoints。
- policy weights。
- Hugging Face model cache。
- Isaac Sim runtime cache。
- 大型影片與實驗輸出。

如果要分享這些資料，建議使用：

- GitHub Releases
- Hugging Face Hub / Dataset
- Git LFS
- Google Drive / OneDrive

### English

The following are usually not stored in the Git repository:

- Recorded episodes under `outputs/`.
- RLDS shards.
- Model checkpoints.
- Policy weights.
- Hugging Face model cache.
- Isaac Sim runtime cache.
- Large videos and experiment outputs.

To share these artifacts, use:

- GitHub Releases
- Hugging Face Hub / Dataset
- Git LFS
- Google Drive / OneDrive

---

## 18. Research Direction / 研究方向

### 中文

本專案的長期目標是探索：

```text
V-JEPA2 world-model-style visual prediction
+ SmolVLA / VLA-style action policy
+ cuRobo expert planning
+ DAgger correction
+ Isaac Sim safety validation
= safer multimodal robotic manipulation prototype
```

短期方向：

- 保留輸送帶循環任務穩定性。
- 擴充投籃、放置到障礙物上、多目標放置等任務。
- 用 DAgger 收集多任務 correction data。
- 讓 SmolVLA-style policy 學到跨任務 action trend。

中期方向：

- 導入更完整的 VLA backbone。
- 增加 action chunk prediction。
- 提升非訓練場景中的泛化能力。
- 保留 end-effector path line 作為未來動作預測可視化。

長期方向：

- 讓多模態大腦在安全約束內負責更高比例的控制。
- 將 Isaac Sim 中的 VLA policy 與真實 xArm6 實驗流程銜接。
- 建立可重複、可驗證、可擴充的具身 AI manipulation benchmark。

### English

The long-term research goal is to explore:

```text
V-JEPA2 world-model-style visual prediction
+ SmolVLA / VLA-style action policy
+ cuRobo expert planning
+ DAgger correction
+ Isaac Sim safety validation
= safer multimodal robotic manipulation prototype
```

Short-term directions:

- Preserve conveyor-cycle task stability.
- Add basket throwing, placing objects on obstacles, and multi-target placement tasks.
- Use DAgger to collect multi-task correction data.
- Help the SmolVLA-style policy learn cross-task action trends.

Mid-term directions:

- Integrate a more complete VLA backbone.
- Improve action chunk prediction.
- Improve generalization to unseen scene variations.
- Keep the end-effector path line as future action prediction visualization.

Long-term directions:

- Let the multimodal brain handle a larger portion of control under safety constraints.
- Connect Isaac Sim VLA policies to real xArm6 experiments.
- Build a repeatable, verifiable, extensible embodied AI manipulation benchmark.

---

## 19. Safety Disclaimer / 安全聲明

### 中文

本專案目前僅作為 Isaac Sim 模擬與研究用途。請勿在沒有完整安全驗證、硬體限位、急停機制、碰撞檢查與人工監督的情況下，直接部署到真實機械手臂。

### English

This repository is a simulation research prototype. Do not directly deploy the generated policies on a real robot without independent safety validation, hardware limits, emergency stop handling, collision checking, and supervised testing.

---

## 20. Acknowledgements / 致謝

- UFACTORY xArm ROS 2 assets:

```text
https://github.com/xArm-Developer/xarm_ros2
```

- NVIDIA Isaac Sim
- cuRobo
- V-JEPA2 research direction
- SmolVLA / VLA-style embodied policy direction

