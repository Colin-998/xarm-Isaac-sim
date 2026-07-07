"""Evaluate Stage-3 video SFT policy on held-out RLDS episodes."""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from train_stage3_video_sft import Stage3Policy, VideoSFTDataset, collate, load_episodes


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rlds-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-episodes", type=int, default=64)
    parser.add_argument("--windows-per-episode", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    phase_to_id = checkpoint["phase_to_id"]
    episodes = load_episodes(args.rlds_root.resolve(), args.max_episodes)
    dataset = VideoSFTDataset(
        args.rlds_root.resolve(),
        episodes,
        int(config["clip_frames"]),
        int(config["image_size"]),
        args.windows_per_episode,
    )
    dataset.phase_to_id = phase_to_id
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate)
    model = Stage3Policy(
        config["vjepa2_model_id"],
        int(config["embed_dim"]),
        len(phase_to_id),
        args.local_files_only,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    total = 0
    correct = 0
    action_abs = 0.0
    final_arm_abs = 0.0
    final_gripper_abs = 0.0
    preview = []
    with torch.no_grad():
        for batch in loader:
            phase_logits, action_pred = model(batch["frames"].to(device))
            phase_id = batch["phase_id"].to(device)
            action_target = batch["action_target"].to(device)
            pred = phase_logits.argmax(dim=-1)
            abs_error = torch.abs(action_pred - action_target)
            correct += (pred == phase_id).sum().item()
            total += phase_id.numel()
            action_abs += abs_error.mean(dim=1).sum().item()
            final_arm_abs += abs_error[:, -7:-1].mean(dim=1).sum().item()
            final_gripper_abs += abs_error[:, -1].sum().item()
            for episode_id, p, t in zip(batch["episode_id"], pred.cpu().tolist(), phase_id.cpu().tolist()):
                if len(preview) < 12:
                    preview.append({"episode_id": episode_id, "predicted_phase_id": p, "target_phase_id": t})

    metrics = {
        "checkpoint": str(args.checkpoint.resolve()),
        "episodes": len(dataset),
        "source_episodes": len(episodes),
        "windows_per_episode": args.windows_per_episode,
        "phase_accuracy": correct / max(1, total),
        "mean_action_abs_error": action_abs / max(1, total),
        "expert_action_mae": action_abs / max(1, total),
        "expert_final_arm_mae_rad": final_arm_abs / max(1, total),
        "expert_final_gripper_mae": final_gripper_abs / max(1, total),
        "phase_to_id": phase_to_id,
        "predictions_preview": preview,
        "closed_loop_readiness": {
            "policy_outputs": "phase_id and 14D action target",
            "recommended_executor": "safety-filtered Isaac Sim executor with cuRobo fallback",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
