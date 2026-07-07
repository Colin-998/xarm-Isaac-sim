# Stage 1：正式 Projector Alignment

本階段目標是把 V-JEPA2 的視覺特徵投影到 Llama 系列語言模型的 embedding space。

目前已完成：

- 產生 Stage 1 projector alignment JSONL。
- 新增 V-JEPA2 + Llama projector 訓練腳本。
- 安裝 `accelerate`、`peft`、`sentencepiece`、`protobuf`。
- 驗證 `meta-llama/Llama-3.1-8B-Instruct` 目前因 Hugging Face gated 權限尚未登入，無法下載。
- 使用公開 tiny Llama 架構模型完成 smoke training，證明訓練管線可行。

## 資料產生

```powershell
cd C:\Users\User\Documents\XArm
& C:\Users\User\isaac_sim_5.1\python.bat scripts\build_projector_alignment_data.py `
  --annotations outputs\mllm_alignment_annotations.jsonl `
  --output outputs\stage1_projector_alignment.jsonl `
  --stride 4
```

輸出：

```text
outputs/stage1_projector_alignment.jsonl
outputs/stage1_projector_alignment.manifest.json
```

目前資料量：

```text
episodes = 500
samples = 8000
```

## Smoke Training

因為 Llama 3.1 尚未登入授權，目前先用公開 tiny Llama 架構模型驗證：

```powershell
cd C:\Users\User\Documents\XArm
$env:HF_HOME='C:\Users\User\Documents\XArm\outputs\hf_cache'
$env:HF_HUB_DISABLE_SYMLINKS_WARNING='1'
& C:\Users\User\isaac_sim_5.1\python.bat scripts\train_vjepa2_llama_projector.py `
  --rlds-root outputs\rlds_xarm6_curobo_500_v2 `
  --samples outputs\stage1_projector_alignment.jsonl `
  --output-dir outputs\stage1_projector_tiny_llama `
  --llm-model-id hf-internal-testing/tiny-random-LlamaForCausalLM `
  --max-samples 128 `
  --batch-size 4 `
  --epochs 3 `
  --device cuda `
  --local-files-only
```

訓練結果：

| Epoch | Loss |
| --- | ---: |
| 1 | 0.5205 |
| 2 | 0.2737 |
| 3 | 0.2394 |

輸出：

```text
outputs/stage1_projector_tiny_llama/latest_projector.pt
outputs/stage1_projector_tiny_llama/metrics.jsonl
outputs/stage1_projector_tiny_llama/training_config.json
```

## 切換到 Llama 3.1

目前 `meta-llama/Llama-3.1-8B-Instruct` 回傳 `401 Unauthorized / gated repo`。需要先完成：

1. 在 Hugging Face 網站申請並取得 Llama 3.1 模型存取權。
2. 在本機登入：

```powershell
& C:\Users\User\isaac_sim_5.1\python.bat -m huggingface_hub.cli login
```

或使用 Hugging Face CLI：

```powershell
C:\Users\User\isaac_sim_5.1\kit\python\Scripts\huggingface-cli.exe login
```

登入後可改用：

```powershell
cd C:\Users\User\Documents\XArm
$env:HF_HOME='C:\Users\User\Documents\XArm\outputs\hf_cache'
$env:HF_HUB_DISABLE_SYMLINKS_WARNING='1'
& C:\Users\User\isaac_sim_5.1\python.bat scripts\train_vjepa2_llama_projector.py `
  --rlds-root outputs\rlds_xarm6_curobo_500_v2 `
  --samples outputs\stage1_projector_alignment.jsonl `
  --output-dir outputs\stage1_projector_llama31 `
  --llm-model-id meta-llama/Llama-3.1-8B-Instruct `
  --max-samples 8000 `
  --batch-size 1 `
  --epochs 1 `
  --device cuda
```

注意：Llama 3.1 8B 權重很大，若顯存不足，建議改用較小的 Llama-family 模型先做 projector alignment，或使用量化/CPU offload。

## Llama 3.1 正式 Smoke Training

Llama 3.1 授權通過後，已使用 embedding-only 模式完成正式 smoke training。

這個模式只載入：

```text
model.embed_tokens.weight
```

而不是整個 8B 模型，因此能避免把完整 Llama 3.1 塞進 GPU。

執行指令：

```powershell
cd C:\Users\User\Documents\XArm
$env:HF_HUB_DISABLE_SYMLINKS_WARNING='1'
& C:\Users\User\isaac_sim_5.1\python.bat scripts\train_vjepa2_llama_projector.py `
  --rlds-root outputs\rlds_xarm6_curobo_500_v2 `
  --samples outputs\stage1_projector_alignment.jsonl `
  --output-dir outputs\stage1_projector_llama31_smoke `
  --llm-model-id meta-llama/Llama-3.1-8B-Instruct `
  --llm-embedding-mode safetensors `
  --max-samples 8 `
  --batch-size 2 `
  --epochs 1 `
  --device cuda
```

結果：

```text
llm_hidden_size = 4096
embedding_source = safetensors:model-00001-of-00004.safetensors
loss = 1.1905
checkpoint = outputs/stage1_projector_llama31_smoke/latest_projector.pt
```

## 目前狀態判定

嚴格來說：

- Stage 1 架構與流程已完成。
- Stage 1 with tiny Llama 已完成 smoke training。
- Stage 1 with Llama 3.1 已完成 embedding-only smoke training。

下一步可以擴大到更多 samples，例如 128、512、8000，並觀察 projector loss 是否穩定下降。
