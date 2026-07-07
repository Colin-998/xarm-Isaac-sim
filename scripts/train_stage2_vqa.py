"""Stage 2 image-text QA adapter over V-JEPA2 and Llama embeddings."""

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

from train_vjepa2_llama_projector import (
    VJEPA2Projector,
    load_text_embedding,
    mean_pool_embeddings,
)


@dataclass
class Stage2Config:
    rlds_root: str
    vqa_data: str
    output_dir: str
    stage1_projector: str
    vjepa2_model_id: str
    llm_model_id: str
    epochs: int
    batch_size: int
    max_samples: int
    image_size: int
    learning_rate: float
    device: str
    llm_embedding_mode: str
    local_files_only: bool


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rlds-root", type=Path, required=True)
    parser.add_argument("--vqa-data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/stage2_vqa_llama31"))
    parser.add_argument("--stage1-projector", type=Path, required=True)
    parser.add_argument("--vjepa2-model-id", default="facebook/vjepa2-vitl-fpc64-256")
    parser.add_argument("--llm-model-id", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--llm-embedding-mode", choices=["auto", "full_model", "safetensors"], default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


class VQADataset(Dataset):
    def __init__(self, root, vqa_data, tokenizer, max_samples, image_size):
        self.root = root
        self.tokenizer = tokenizer
        self.rows = []
        self.answer_to_id = {}
        with vqa_data.open("r", encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    row = json.loads(line)
                    self.answer_to_id.setdefault(row["answer"], len(self.answer_to_id))
                    row["answer_id"] = self.answer_to_id[row["answer"]]
                    self.rows.append(row)
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
        image = Image.open(self.root / row["image"]).convert("RGB")
        question = self.tokenizer(
            row["prompt"],
            max_length=96,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        answer = self.tokenizer(
            row["answer"],
            max_length=16,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "image": self.image_transform(image),
            "question_ids": question["input_ids"].squeeze(0),
            "question_mask": question["attention_mask"].squeeze(0).float(),
            "answer_ids": answer["input_ids"].squeeze(0),
            "answer_mask": answer["attention_mask"].squeeze(0).float(),
            "answer_id": torch.tensor(row["answer_id"], dtype=torch.long),
        }


class Stage2VQAModel(nn.Module):
    def __init__(self, stage1_projector, vjepa2_model_id, llm_hidden_size, num_answers, local_files_only):
        super().__init__()
        self.vision_projector = VJEPA2Projector(vjepa2_model_id, llm_hidden_size, local_files_only)
        ckpt = torch.load(stage1_projector, map_location="cpu", weights_only=False)
        self.vision_projector.projector.load_state_dict(ckpt["projector_state_dict"])
        for param in self.vision_projector.parameters():
            param.requires_grad = False
        self.fusion = nn.Sequential(
            nn.LayerNorm(llm_hidden_size * 2),
            nn.Linear(llm_hidden_size * 2, llm_hidden_size),
            nn.SiLU(),
            nn.Linear(llm_hidden_size, llm_hidden_size),
        )
        self.answer_head = nn.Linear(llm_hidden_size, num_answers)

    def forward(self, image, question_embedding):
        visual = self.vision_projector(image)
        fused = self.fusion(torch.cat([visual.float(), question_embedding.float()], dim=-1))
        return fused, self.answer_head(fused)


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.llm_model_id, local_files_only=args.local_files_only)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    text_embedding, llm_hidden_size, embedding_source = load_text_embedding(
        args.llm_model_id,
        args.llm_embedding_mode,
        device,
        local_files_only=args.local_files_only,
    )
    text_embedding.eval()

    dataset = VQADataset(
        root=args.rlds_root.resolve(),
        vqa_data=args.vqa_data.resolve(),
        tokenizer=tokenizer,
        max_samples=args.max_samples,
        image_size=args.image_size,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    print(
        f"stage2_vqa_ready samples={len(dataset)} answers={len(dataset.answer_to_id)} "
        f"llm_hidden={llm_hidden_size} embedding_source={embedding_source}"
    )
    if args.dry_run:
        return

    model = Stage2VQAModel(
        stage1_projector=args.stage1_projector,
        vjepa2_model_id=args.vjepa2_model_id,
        llm_hidden_size=llm_hidden_size,
        num_answers=len(dataset.answer_to_id),
        local_files_only=args.local_files_only,
    ).to(device)
    optimizer = torch.optim.AdamW(
        list(model.fusion.parameters()) + list(model.answer_head.parameters()),
        lr=args.learning_rate,
        weight_decay=1e-4,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = Stage2Config(
        rlds_root=str(args.rlds_root.resolve()),
        vqa_data=str(args.vqa_data.resolve()),
        output_dir=str(args.output_dir.resolve()),
        stage1_projector=str(args.stage1_projector.resolve()),
        vjepa2_model_id=args.vjepa2_model_id,
        llm_model_id=args.llm_model_id,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        image_size=args.image_size,
        learning_rate=args.learning_rate,
        device=str(device),
        llm_embedding_mode=args.llm_embedding_mode,
        local_files_only=args.local_files_only,
    )
    (args.output_dir / "training_config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
    (args.output_dir / "answer_to_id.json").write_text(json.dumps(dataset.answer_to_id, indent=2), encoding="utf-8")

    metrics_path = args.output_dir / "metrics.jsonl"
    total_steps = 0
    with metrics_path.open("w", encoding="utf-8") as metrics_stream:
        for epoch in range(args.epochs):
            totals = {"loss": 0.0, "ce": 0.0, "cosine": 0.0, "accuracy": 0.0}
            for step, batch in enumerate(loader):
                images = batch["image"].to(device)
                question_ids = batch["question_ids"].to(device)
                question_mask = batch["question_mask"].to(device)
                answer_ids = batch["answer_ids"].to(device)
                answer_mask = batch["answer_mask"].to(device)
                answer_id = batch["answer_id"].to(device)
                with torch.no_grad():
                    q_emb = mean_pool_embeddings(text_embedding(question_ids), question_mask)
                    a_emb = mean_pool_embeddings(text_embedding(answer_ids), answer_mask)
                fused, logits = model(images, q_emb)
                ce = F.cross_entropy(logits, answer_id)
                cosine = 1.0 - F.cosine_similarity(fused.float(), a_emb.float(), dim=-1).mean()
                loss = ce + 0.25 * cosine
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_steps += 1
                acc = (logits.argmax(dim=-1) == answer_id).float().mean()
                values = {
                    "loss": float(loss.detach().cpu()),
                    "ce": float(ce.detach().cpu()),
                    "cosine": float(cosine.detach().cpu()),
                    "accuracy": float(acc.detach().cpu()),
                }
                for key, value in values.items():
                    totals[key] += value
                if step % 10 == 0:
                    print(
                        f"epoch={epoch + 1}/{args.epochs} step={step + 1}/{len(loader)} "
                        f"loss={values['loss']:.4f} acc={values['accuracy']:.3f}"
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
                    "model_state_dict": model.state_dict(),
                    "config": asdict(config),
                    "answer_to_id": dataset.answer_to_id,
                    "epoch": epoch + 1,
                    "metrics": row,
                },
                args.output_dir / "latest_stage2_vqa.pt",
            )
            print(f"saved epoch={epoch + 1} checkpoint={args.output_dir / 'latest_stage2_vqa.pt'} loss={row['loss']:.4f}")


if __name__ == "__main__":
    main()
