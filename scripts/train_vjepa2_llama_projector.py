"""Stage 1 projector alignment: V-JEPA2 visual features to Llama embeddings.

The script freezes V-JEPA2 and a Llama-family language model, then trains only a
small projector that maps visual features into the language embedding space.
Use a tiny public Llama model for smoke tests, then switch --llm-model-id to
meta-llama/Llama-3.1-8B-Instruct after Hugging Face authentication is available.
"""

import argparse
import json
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


@dataclass
class ProjectorConfig:
    rlds_root: str
    samples: str
    output_dir: str
    vjepa2_model_id: str
    llm_model_id: str
    epochs: int
    batch_size: int
    max_samples: int
    image_size: int
    learning_rate: float
    device: str
    local_files_only: bool
    llm_embedding_mode: str


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rlds-root", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/stage1_vjepa2_llama_projector"))
    parser.add_argument("--vjepa2-model-id", default="facebook/vjepa2-vitl-fpc64-256")
    parser.add_argument("--llm-model-id", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--llm-embedding-mode",
        choices=["auto", "full_model", "safetensors"],
        default="auto",
        help="Use full model embeddings or load only the safetensors embed_tokens shard.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


class ProjectorDataset(Dataset):
    def __init__(self, rlds_root, samples_path, tokenizer, max_samples, image_size):
        self.rlds_root = rlds_root
        self.tokenizer = tokenizer
        self.rows = []
        with samples_path.open("r", encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    self.rows.append(json.loads(line))
                    if max_samples and len(self.rows) >= max_samples:
                        break
        self.image_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image = Image.open(self.rlds_root / row["image"]).convert("RGB")
        encoded = self.tokenizer(
            row["text"],
            max_length=128,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "image": self.image_transform(image),
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0).float(),
            "episode_id": row["episode_id"],
            "step_index": row["step_index"],
            "text": row["text"],
        }


class VJEPA2Projector(nn.Module):
    def __init__(self, vjepa2_model_id, llm_hidden_size, local_files_only=False):
        super().__init__()
        from transformers import AutoConfig, VJEPA2Model

        self.vjepa2_config = AutoConfig.from_pretrained(
            vjepa2_model_id,
            local_files_only=local_files_only,
        )
        self.vjepa2 = VJEPA2Model.from_pretrained(
            vjepa2_model_id,
            local_files_only=local_files_only,
        )
        for param in self.vjepa2.parameters():
            param.requires_grad = False
        hidden_size = int(getattr(self.vjepa2_config, "hidden_size", 1024))
        self.projector = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, llm_hidden_size),
        )

    def forward(self, images):
        videos = images.unsqueeze(1).repeat(1, 2, 1, 1, 1)
        self.vjepa2.eval()
        with torch.no_grad():
            output = self.vjepa2(pixel_values_videos=videos, skip_predictor=True)
        visual = output.last_hidden_state.mean(dim=1)
        return self.projector(visual)


def mean_pool_embeddings(embeddings, attention_mask):
    masked = embeddings * attention_mask.unsqueeze(-1)
    denom = attention_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    return masked.sum(dim=1) / denom


def contrastive_loss(projected, target):
    if projected.shape[0] < 2:
        return projected.new_tensor(0.0)
    logits = F.normalize(projected, dim=-1) @ F.normalize(target, dim=-1).t()
    labels = torch.arange(projected.shape[0], device=projected.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


class FrozenTextEmbedding(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(weight, freeze=True)

    @property
    def embedding_dim(self):
        return self.embedding.embedding_dim

    def forward(self, input_ids):
        return self.embedding(input_ids)


def load_safetensors_embedding(model_id, local_files_only=False):
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model_id, local_files_only=local_files_only)
    index_path = hf_hub_download(
        model_id,
        "model.safetensors.index.json",
        local_files_only=local_files_only,
    )
    with open(index_path, "r", encoding="utf-8") as stream:
        index = json.load(stream)
    shard_name = index["weight_map"].get("model.embed_tokens.weight")
    if not shard_name:
        raise RuntimeError(f"model.embed_tokens.weight not found in {index_path}")
    shard_path = hf_hub_download(
        model_id,
        shard_name,
        local_files_only=local_files_only,
    )
    with safe_open(shard_path, framework="pt", device="cpu") as shard:
        weight = shard.get_tensor("model.embed_tokens.weight")
    hidden_size = int(getattr(config, "hidden_size", weight.shape[1]))
    if weight.shape[1] != hidden_size:
        raise RuntimeError(
            f"Unexpected embedding shape {tuple(weight.shape)} for hidden_size={hidden_size}"
        )
    return FrozenTextEmbedding(weight), hidden_size, shard_name


def load_text_embedding(model_id, mode, device, local_files_only=False):
    if mode in ("auto", "safetensors"):
        try:
            embedding, hidden_size, source = load_safetensors_embedding(
                model_id,
                local_files_only=local_files_only,
            )
            return embedding.to(device), hidden_size, f"safetensors:{source}"
        except Exception:
            if mode == "safetensors":
                raise
    from transformers import AutoModelForCausalLM

    llm = AutoModelForCausalLM.from_pretrained(
        model_id,
        local_files_only=local_files_only,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    ).to(device)
    llm.eval()
    for param in llm.parameters():
        param.requires_grad = False
    return llm.get_input_embeddings(), int(llm.get_input_embeddings().embedding_dim), "full_model"


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.llm_model_id,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    text_embedding, llm_hidden_size, embedding_source = load_text_embedding(
        args.llm_model_id,
        args.llm_embedding_mode,
        device,
        local_files_only=args.local_files_only,
    )
    text_embedding.eval()

    dataset = ProjectorDataset(
        rlds_root=args.rlds_root.resolve(),
        samples_path=args.samples.resolve(),
        tokenizer=tokenizer,
        max_samples=args.max_samples,
        image_size=args.image_size,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    sample = dataset[0]
    print(
        "projector_dataset_ready "
        f"samples={len(dataset)} image={tuple(sample['image'].shape)} "
        f"llm_hidden_size={llm_hidden_size} llm={args.llm_model_id} "
        f"embedding_source={embedding_source}"
    )
    if args.dry_run:
        print(f"sample_text={sample['text'][:180]}")
        return

    model = VJEPA2Projector(
        vjepa2_model_id=args.vjepa2_model_id,
        llm_hidden_size=llm_hidden_size,
        local_files_only=args.local_files_only,
    ).to(device)
    optimizer = torch.optim.AdamW(model.projector.parameters(), lr=args.learning_rate, weight_decay=1e-4)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = ProjectorConfig(
        rlds_root=str(args.rlds_root.resolve()),
        samples=str(args.samples.resolve()),
        output_dir=str(args.output_dir.resolve()),
        vjepa2_model_id=args.vjepa2_model_id,
        llm_model_id=args.llm_model_id,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        image_size=args.image_size,
        learning_rate=args.learning_rate,
        device=str(device),
        local_files_only=args.local_files_only,
        llm_embedding_mode=args.llm_embedding_mode,
    )
    (args.output_dir / "training_config.json").write_text(
        json.dumps(asdict(config), indent=2),
        encoding="utf-8",
    )

    metrics_path = args.output_dir / "metrics.jsonl"
    total_steps = 0
    with metrics_path.open("w", encoding="utf-8") as metrics_stream:
        for epoch in range(args.epochs):
            totals = {"loss": 0.0, "mse": 0.0, "cosine": 0.0, "contrastive": 0.0}
            for step, batch in enumerate(loader):
                images = batch["image"].to(device)
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                with torch.no_grad():
                    token_embeddings = text_embedding(input_ids)
                    target = mean_pool_embeddings(token_embeddings, attention_mask)
                projected = model(images)
                mse = F.mse_loss(projected.float(), target.float())
                cosine = 1.0 - F.cosine_similarity(projected.float(), target.float(), dim=-1).mean()
                clip = contrastive_loss(projected.float(), target.float())
                loss = mse + cosine + 0.1 * clip

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.projector.parameters(), 1.0)
                optimizer.step()

                total_steps += 1
                values = {
                    "loss": float(loss.detach().cpu()),
                    "mse": float(mse.detach().cpu()),
                    "cosine": float(cosine.detach().cpu()),
                    "contrastive": float(clip.detach().cpu()),
                }
                for key, value in values.items():
                    totals[key] += value
                if step % 10 == 0:
                    print(
                        f"epoch={epoch + 1}/{args.epochs} step={step + 1}/{len(loader)} "
                        f"loss={values['loss']:.4f} mse={values['mse']:.4f} "
                        f"cosine={values['cosine']:.4f} contrastive={values['contrastive']:.4f}"
                    )

            denom = max(1, len(loader))
            row = {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "epoch": epoch + 1,
                "steps": total_steps,
                **{key: value / denom for key, value in totals.items()},
            }
            metrics_stream.write(json.dumps(row, separators=(",", ":")) + "\n")
            metrics_stream.flush()
            torch.save(
                {
                    "projector_state_dict": model.projector.state_dict(),
                    "config": asdict(config),
                    "llm_hidden_size": llm_hidden_size,
                    "epoch": epoch + 1,
                    "metrics": row,
                },
                args.output_dir / "latest_projector.pt",
            )
            print(
                f"saved epoch={epoch + 1} checkpoint={args.output_dir / 'latest_projector.pt'} "
                f"loss={row['loss']:.4f}"
            )


if __name__ == "__main__":
    main()
