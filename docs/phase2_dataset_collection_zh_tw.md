# 階段二：V-JEPA2 高純度資料採集

## 資料規格

- 每個成功的抓取循環為一個 episode。
- RGB 時序影像固定為 4 FPS、256 x 256、PNG。
- 每個 episode 至少包含 64 幀，對齊 `fpc64-256` 模型輸入。
- 每張影像對應 `actions.jsonl` 中的一筆標籤。
- 只有完成抓取、放置及輸送帶送回的 episode 才會正式保存。
- cuRobo 每輪都會依方塊實際位置和手臂目前姿態重新規劃。

每個 episode 內容：

```text
episode_00000/
  rgb_000000.png
  rgb_000001.png
  ...
  actions.jsonl
  metadata.json
```

動作標籤包含：

- 六軸手臂目標關節角度。
- 夾爪目標位置。
- 任務階段與模擬時間。

觀測標籤包含：

- 六軸實際關節角度與速度。
- 每個夾爪關節的實際位置。
- TCP 位置與旋轉矩陣。
- 方塊位置與姿態。
- 障礙物位置與尺寸。

## 採集 500 組

```powershell
cd C:\Users\User\Documents\XArm

& C:\Users\User\isaac_sim_5.1\python.bat `
  scripts\play_curobo_dynamic_pick.py `
  --headless `
  --episodes 500 `
  --record-root outputs\vjepa2_curobo_500 `
  --steps-per-waypoint 2 `
  --conveyor-speed 0.25
```

`--episodes 500` 代表資料夾內最終需要有 500 組。若程式中斷，
重新執行相同指令即可從下一個 episode 繼續，不會覆寫已完成資料。

進度記錄於：

```text
outputs/vjepa2_curobo_500/dataset_manifest.json
```

## 驗證資料集

```powershell
& C:\Users\User\isaac_sim_5.1\python.bat `
  scripts\validate_curobo_dataset.py `
  outputs\vjepa2_curobo_500 `
  --expected-episodes 500
```

驗證器會檢查 episode 數量、成功狀態、影像存在性、RGB 格式、
256 x 256 解析度、4 FPS 間距、動作維度與完整物理循環。
