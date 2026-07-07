# RLDS 資料集打包說明

## 輸入資料集

```text
outputs/vjepa2_curobo_500_v2
```

此資料集包含：

- 500 個成功 episode。
- 每個 episode 64 張 RGB 影像。
- 4 FPS、256 x 256。
- 共 32,000 組 RGB/action step。

## RLDS 輸出資料集

```text
outputs/rlds_xarm6_curobo_500_v2
```

資料夾結構：

```text
dataset_info.json
features.json
episodes/
  rlds-00000-of-00010.jsonl
  ...
  rlds-00009-of-00010.jsonl
images/
  episode_00000/
    rgb_000000.png
    ...
```

每個 JSONL shard 的每一行是一個 episode，格式為：

```json
{
  "episode_id": "episode_00000",
  "episode_metadata": {},
  "steps": [
    {
      "observation": {},
      "action": [],
      "reward": 0.0,
      "discount": 1.0,
      "is_first": true,
      "is_last": false,
      "is_terminal": false
    }
  ]
}
```

## 主要欄位

- `observation.image`：相對於 RLDS package root 的 RGB PNG 路徑。
- `observation.natural_language_instruction`：語言任務指令。
- `observation.state`：`arm6 + gripper + tcp3 + cube3`，共 13 維。
- `action`：`arm6 + gripper`，共 7 維。
- `reward`：最後一步為 `1.0`，其他為 `0.0`。
- `discount`：最後一步為 `0.0`，其他為 `1.0`。
- `is_first`、`is_last`、`is_terminal`：RLDS step 標準旗標。

## 重新打包

```powershell
cd C:\Users\User\Documents\XArm

& C:\Users\User\isaac_sim_5.1\python.bat `
  scripts\package_rlds_dataset.py `
  --dataset-root outputs\vjepa2_curobo_500_v2 `
  --output-root outputs\rlds_xarm6_curobo_500_v2 `
  --expected-episodes 500 `
  --shard-size 50 `
  --validation-episodes 50
```

## 驗證

```powershell
& C:\Users\User\isaac_sim_5.1\python.bat `
  scripts\validate_rlds_package.py `
  outputs\rlds_xarm6_curobo_500_v2 `
  --expected-episodes 500 `
  --expected-steps 32000
```
