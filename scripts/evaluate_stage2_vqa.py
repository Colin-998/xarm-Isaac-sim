"""Evaluate Stage-2 image-text QA adapter."""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from train_stage2_vqa import Stage2VQAModel, VQADataset
from train_vjepa2_llama_projector import load_text_embedding, mean_pool_embeddings


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rlds-root", type=Path, required=True)
    parser.add_argument("--vqa-data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    answer_to_id = checkpoint["answer_to_id"]

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config["llm_model_id"], local_files_only=args.local_files_only)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    text_embedding, llm_hidden_size, embedding_source = load_text_embedding(
        config["llm_model_id"],
        config["llm_embedding_mode"],
        device,
        local_files_only=args.local_files_only,
    )
    text_embedding.eval()
    dataset = VQADataset(
        root=args.rlds_root.resolve(),
        vqa_data=args.vqa_data.resolve(),
        tokenizer=tokenizer,
        max_samples=args.max_samples,
        image_size=int(config["image_size"]),
    )
    filtered = []
    for row in dataset.rows:
        if row["answer"] in answer_to_id:
            row["answer_id"] = answer_to_id[row["answer"]]
            filtered.append(row)
    dataset.rows = filtered
    dataset.answer_to_id = answer_to_id
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = Stage2VQAModel(
        stage1_projector=Path(config["stage1_projector"]),
        vjepa2_model_id=config["vjepa2_model_id"],
        llm_hidden_size=llm_hidden_size,
        num_answers=len(answer_to_id),
        local_files_only=args.local_files_only,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    total = 0
    correct = 0
    loss_sum = 0.0
    preview = []
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            question_ids = batch["question_ids"].to(device)
            question_mask = batch["question_mask"].to(device)
            answer_id = batch["answer_id"].to(device)
            q_emb = mean_pool_embeddings(text_embedding(question_ids), question_mask)
            _, logits = model(images, q_emb)
            loss = F.cross_entropy(logits, answer_id)
            pred = logits.argmax(dim=-1)
            total += answer_id.numel()
            correct += (pred == answer_id).sum().item()
            loss_sum += float(loss.cpu()) * answer_id.numel()
            for p, t in zip(pred.cpu().tolist(), answer_id.cpu().tolist()):
                if len(preview) < 12:
                    preview.append({"predicted_answer_id": p, "target_answer_id": t})

    metrics = {
        "checkpoint": str(args.checkpoint.resolve()),
        "samples": total,
        "accuracy": correct / max(1, total),
        "loss": loss_sum / max(1, total),
        "answers": answer_to_id,
        "embedding_source": embedding_source,
        "predictions_preview": preview,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
