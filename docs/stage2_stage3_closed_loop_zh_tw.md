# Stage 2 / Stage 3 / 閉環展示說明

本文件整理目前 V-JEPA 2 + Llama 3.1 多模態訓練成果，以及 Isaac Sim xArm6 閉環展示的執行方式。

## Stage 2: 大規模多模態問答

資料集：

```text
outputs/stage2_vqa.jsonl
samples = 64000
episodes = 500
answer classes = 16
```

最新訓練輸出：

```text
outputs/stage2_vqa_llama31_4096/latest_stage2_vqa.pt
```

設定：

- Vision: frozen V-JEPA 2 features
- Language target: Llama 3.1-8B-Instruct token embedding
- Stage 1 projector: `outputs/stage1_projector_llama31_smoke/latest_projector.pt`
- Training samples: 4096
- Epochs: 3
- Device: CUDA

訓練 loss：

| Epoch | Loss |
| --- | ---: |
| 1 | 1.7734 |
| 2 | 1.2451 |
| 3 | 0.7044 |

評估結果：

```text
outputs/stage2_vqa_llama31_4096/eval_metrics_4096.json
accuracy = 0.840087890625
loss = 0.4612685663596494
```

這代表模型已能從視覺狀態回答方塊位置、目前任務階段、夾取/放置/等待輸送帶返回等語意問題。

## Stage 3: 影片時序指令微調

最新訓練輸出：

```text
outputs/stage3_video_sft_500ep_w4/latest_stage3_policy.pt
```

設定：

- Backbone: frozen V-JEPA 2
- Input: 4-frame video clip
- Output: phase id + 14D action target
- Source episodes: 500
- Windows per episode: 4
- Effective training samples: 2000
- Epochs: 3
- Device: CUDA

訓練 loss：

| Epoch | Loss |
| --- | ---: |
| 1 | 0.0835 |
| 2 | 0.0042 |
| 3 | 0.0036 |

評估結果：

```text
outputs/stage3_video_sft_500ep_w4/eval_metrics_500_w4.json
phase_accuracy = 0.994
mean_action_abs_error = 0.04082483244687319
```

這代表 Stage 3 已能根據影片時序預測目前動作階段，並輸出接近專家資料的 14 維動作目標。

## 目前閉環能力

目前展示支援兩種閉環模式：

- `direct`: Stage 3 policy 直接輸出 xArm6 關節/夾爪 target，僅做小步長 rate limit。
- `filtered`: Stage 3 policy 輸出 target，通過安全檢查才採用，否則回退到 cuRobo target。

目前成果展示預設使用 `direct`，並用 `--brain-max-step-delta 0.03` 限制每一步最大關節變化。這不是完全無保護的真機裸控，而是在 Isaac Sim 中由 V-JEPA 2 / Llama 3.1 多模態大腦接管 xArm target control。

```text
V-JEPA 2 視覺特徵
-> Stage 1 Llama 3.1 projector
-> Stage 2 Image-Text QA
-> Stage 3 video policy phase/action intention
-> online xArm6 joint/gripper target control
-> optional safety-filtered cuRobo fallback
-> xArm6 conveyor pick cycle
```

已驗證 direct 模式一圈成果：

```text
outputs/vjepa2_brain_live_run_direct.json
brain_control = direct
direct_steps = 1584
accepted = 1584
rejected = 0
placed_at_start = true
start_place_distance_m = 0.000010777218006403105
returned_to_end = true
final_end_distance_m = 0.06434393636437706
```

可以誠實稱為：

```text
V-JEPA 2 / Llama 3.1 multimodal brain direct target-control closed loop in Isaac Sim
```

注意：目前仍保留 cuRobo 規劃與 phase loop 作為任務腳手架，讓 demo 穩定完成 pick-place-return cycle。下一步若要做更嚴格的研究版，可以把 phase loop 也交給 Stage 3/LLM policy 決策。

## 成果展示指令

只產生展示報告：

```powershell
cd C:\Users\User\Documents\XArm
& C:\Users\User\isaac_sim_5.1\python.bat scripts\run_vjepa2_brain_closed_loop_showcase.py
```

啟動 Isaac Sim 展示：

```powershell
cd C:\Users\User\Documents\XArm
& C:\Users\User\isaac_sim_5.1\python.bat scripts\run_vjepa2_brain_closed_loop_showcase.py --launch-isaac --cycles 1
```

背景執行 Isaac Sim 展示：

```powershell
cd C:\Users\User\Documents\XArm
& C:\Users\User\isaac_sim_5.1\python.bat scripts\run_vjepa2_brain_closed_loop_showcase.py --launch-isaac --headless --cycles 3
```

如果要改回更保守的安全濾波展示：

```powershell
cd C:\Users\User\Documents\XArm
& C:\Users\User\isaac_sim_5.1\python.bat scripts\run_vjepa2_brain_closed_loop_showcase.py --launch-isaac --cycles 1 --brain-control filtered
```

## 下一個技術關卡

若要達到更嚴格的「大腦完全自治」版本，下一步不是再只做離線訓練，而是要把 phase loop 也交給 Stage 3/LLM policy：

1. 從 Isaac Sim camera 每 0.25 秒取得 4 FPS 時序影像。
2. 由 Stage 2 VQA 判斷物件位置與任務語意。
3. 由 Stage 3 policy 即時輸出 phase id 與 14D action target。
4. 不再固定跑 cuRobo phase sequence，而是由 policy/LLM 決定下一個 phase。
5. 只保留 joint limit、速度限制、碰撞距離、障礙物距離與 emergency stop。

這會是比目前展示更強的 fully autonomous closed loop 版本。
