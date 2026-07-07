"""Evaluate an MLLM alignment checkpoint on RLDS episodes."""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from train_mllm_alignment import (
    AlignmentAdapter,
    RLDSEpisodeDataset,
    collate,
    load_annotations,
    load_episodes,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rlds-root", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-episodes", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    vocab = checkpoint["vocab"]
    phase_to_id = checkpoint["phase_to_id"]

    root = args.rlds_root.resolve()
    annotations = load_annotations(args.annotations.resolve())
    episodes = load_episodes(root, args.max_episodes)
    dataset = RLDSEpisodeDataset(
        root=root,
        episodes=episodes,
        annotations=annotations,
        vocab=vocab,
        clip_frames=int(config["clip_frames"]),
        image_size=int(config["image_size"]),
    )
    dataset.phase_to_id = phase_to_id
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate,
    )

    model = AlignmentAdapter(
        vocab_size=len(vocab),
        num_phases=len(phase_to_id),
        embed_dim=int(config["embed_dim"]),
        vision_backbone=config["vision_backbone"],
        vjepa2_model_id=config["vjepa2_model_id"],
        freeze_vision=config["freeze_vision"],
        local_files_only=args.local_files_only,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    total = 0
    correct = 0
    action_abs_error = 0.0
    retrieval_correct = 0
    retrieval_total = 0
    rows = []
    with torch.no_grad():
        for batch in loader:
            output = model(
                batch["frames"].to(device),
                batch["token_ids"].to(device),
                batch["token_mask"].to(device),
            )
            action_target = batch["action_target"].to(device)
            phase_id = batch["phase_id"].to(device)
            phase_pred = output["phase"].argmax(dim=-1)
            action_abs_error += torch.abs(output["action"] - action_target).mean(dim=1).sum().item()
            correct += (phase_pred == phase_id).sum().item()
            total += phase_id.numel()

            if output["video"].shape[0] > 1:
                logits = output["video"] @ output["text"].t()
                retrieval_pred = logits.argmax(dim=-1)
                labels = torch.arange(logits.shape[0], device=device)
                retrieval_correct += (retrieval_pred == labels).sum().item()
                retrieval_total += labels.numel()

            for episode_id, pred, target in zip(
                batch["episode_id"],
                phase_pred.detach().cpu().tolist(),
                phase_id.detach().cpu().tolist(),
            ):
                rows.append(
                    {
                        "episode_id": episode_id,
                        "predicted_phase_id": pred,
                        "target_phase_id": target,
                    }
                )

    metrics = {
        "checkpoint": str(args.checkpoint.resolve()),
        "episodes": len(dataset),
        "phase_accuracy": correct / max(1, total),
        "mean_action_abs_error": action_abs_error / max(1, total),
        "retrieval_top1": retrieval_correct / retrieval_total if retrieval_total else None,
        "phase_to_id": phase_to_id,
        "predictions_preview": rows[:10],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
