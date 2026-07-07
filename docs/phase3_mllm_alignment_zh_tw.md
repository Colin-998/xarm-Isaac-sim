# 第三階段：V-JEPA2 + LLM 多模態對齊

本階段目標是把第二階段的專家示範資料轉成可用於多模態混訓的格式，讓模型同時看懂：

- 影像時序：4 FPS、256x256 的 xArm6 任務畫面。
- 動作標籤：手臂 6 軸關節目標加夾爪目標。
- 語言標註：任務指令、階段描述、方塊位置問答。
- 任務狀態：phase、方塊位置、障礙物位置、TCP 與 joint state。

## 目前決策

先採用「可立即執行的 alignment bootstrap」：

1. 從 RLDS JSONL 產生 deterministic 語言標註。
2. 用輕量 tiny vision backbone 訓練影片、文字、動作共同對齊。
3. 等本機安裝 `transformers`、`tokenizers`、`safetensors` 並下載 V-JEPA2 權重後，再把 tiny backbone 換成 frozen V-JEPA2 feature extractor。

這樣做的好處是資料格式、標籤、checkpoint、loss 都先跑通，不會把風險集中在套件安裝與大型模型下載。

## 已產出檔案

- `outputs/mllm_alignment_annotations.jsonl`
- `outputs/mllm_alignment_annotations.manifest.json`
- `outputs/mllm_alignment_tiny_full/latest.pt`
- `outputs/mllm_alignment_tiny_full/vocab.json`
- `outputs/mllm_alignment_tiny_full/phase_to_id.json`
- `outputs/mllm_alignment_tiny_full/training_config.json`
- `outputs/mllm_alignment_tiny_full/metrics.jsonl`

## 語言標註產生

```powershell
cd C:\Users\User\Documents\XArm
& C:\Users\User\isaac_sim_5.1\python.bat scripts\generate_mllm_alignment_annotations.py `
  --rlds-root outputs\rlds_xarm6_curobo_500_v2 `
  --output outputs\mllm_alignment_annotations.jsonl
```

## Dry Run

```powershell
cd C:\Users\User\Documents\XArm
& C:\Users\User\isaac_sim_5.1\python.bat scripts\train_mllm_alignment.py `
  --rlds-root outputs\rlds_xarm6_curobo_500_v2 `
  --annotations outputs\mllm_alignment_annotations.jsonl `
  --dry-run `
  --max-episodes 8 `
  --clip-frames 16 `
  --batch-size 2 `
  --image-size 128
```

## 已啟動的 Bootstrap 訓練

```powershell
cd C:\Users\User\Documents\XArm
& C:\Users\User\isaac_sim_5.1\python.bat scripts\train_mllm_alignment.py `
  --rlds-root outputs\rlds_xarm6_curobo_500_v2 `
  --annotations outputs\mllm_alignment_annotations.jsonl `
  --output-dir outputs\mllm_alignment_tiny_full `
  --epochs 1 `
  --clip-frames 16 `
  --batch-size 2 `
  --image-size 128 `
  --device cuda
```

目前完整 500 組資料已跑過 1 個 bootstrap epoch，平均 loss 約為 `0.8871`。

## 真正 V-JEPA2 混訓結果

已安裝 Isaac Sim Python 需要的 Hugging Face 套件：

- `transformers`
- `tokenizers`
- `safetensors`
- `huggingface_hub`
- `tqdm`

並已下載/快取 `facebook/vjepa2-vitl-fpc64-256` 到：

```text
outputs/hf_cache
```

目前已完成 frozen V-JEPA2 backbone 的短訓練：

```powershell
cd C:\Users\User\Documents\XArm
$env:HF_HOME='C:\Users\User\Documents\XArm\outputs\hf_cache'
& C:\Users\User\isaac_sim_5.1\python.bat scripts\train_mllm_alignment.py `
  --rlds-root outputs\rlds_xarm6_curobo_500_v2 `
  --annotations outputs\mllm_alignment_annotations.jsonl `
  --output-dir outputs\mllm_alignment_vjepa2_128ep `
  --epochs 3 `
  --max-episodes 128 `
  --clip-frames 4 `
  --batch-size 2 `
  --image-size 256 `
  --embed-dim 256 `
  --device cuda `
  --vision-backbone vjepa2 `
  --local-files-only
```

訓練 loss：

| Epoch | Loss |
| --- | ---: |
| 1 | 1.3400 |
| 2 | 0.9268 |
| 3 | 0.5236 |

評估指令：

```powershell
cd C:\Users\User\Documents\XArm
$env:HF_HOME='C:\Users\User\Documents\XArm\outputs\hf_cache'
& C:\Users\User\isaac_sim_5.1\python.bat scripts\evaluate_mllm_alignment.py `
  --rlds-root outputs\rlds_xarm6_curobo_500_v2 `
  --annotations outputs\mllm_alignment_annotations.jsonl `
  --checkpoint outputs\mllm_alignment_vjepa2_128ep\latest.pt `
  --output outputs\mllm_alignment_vjepa2_128ep\eval_metrics.json `
  --max-episodes 64 `
  --batch-size 2 `
  --device cuda `
  --local-files-only
```

目前 64 episode 評估結果：

| Metric | Value |
| --- | ---: |
| Phase accuracy | 0.96875 |
| Mean action absolute error | 0.19685 |
| Video-text retrieval top-1 | 0.75 |

## 後續升級建議

目前已完成 frozen V-JEPA2 feature extractor。下一步建議先擴大到 500 episode、增加 clip frames 到 8 或 16，再考慮 LoRA 或少量解凍。不要太早全量微調 V-JEPA2，否則 500 組資料很容易過擬合。

## 對第四階段的意義

這個階段不是直接取代控制器，而是學出「影片、語言、動作」的共同表徵。後續可用於：

- 讓 LLM 根據畫面與語言目標選擇任務 phase。
- 讓 policy 使用 V-JEPA2 視覺特徵，而不是只靠手工 state。
- 比較障礙物、方塊位置變化時，模型是否維持正確抓取與避障策略。
