"""Train a lightweight SmolVLA-compatible policy for the xArm conveyor task.

This is intentionally a local adapter rather than a hard dependency on the
LeRobot package.  The checkpoint format is consumed by play_curobo_dynamic_pick
through policy_arch="smolvla" and keeps the same runtime contract as the
existing Stage-3 policy: phase logits plus a 14D action target.  Internally it
uses a VLA-style action chunk head so a future LeRobot SmolVLA backend can be
swapped behind the same interface.
"""

import argparse
import hashlib
import json
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from train_stage3_video_sft import load_episodes
from train_stage3_direct_correction import (
    DirectCorrectionDataset,
    compute_state_stats,
    load_dagger_episodes,
    pad_state_features,
)


DEFAULT_POLICY = Path(
    "outputs/brain_dagger_takeover/"
    "takeover_fresh_ee_path_001/01_round1_terminal_teacher/policy/"
    "best_stage3_policy.pt"
)
DEFAULT_INSTRUCTION = (
    "Pick up the red cube and place it at the conveyor start while avoiding "
    "obstacles."
)


@dataclass
class SmolVLAConfig:
    rlds_root: str
    output_dir: str
    vjepa2_model_id: str
    base_checkpoint: str
    epochs: int
    batch_size: int
    max_episodes: int
    max_samples: int
    clip_frames: int
    image_size: int
    embed_dim: int
    state_dim: int
    action_chunk_size: int
    learning_rate: float
    action_representation: str
    local_files_only: bool
    instruction: str
    text_hash_buckets: int
    text_token_count: int


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rlds-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/smolvla_xarm6"))
    parser.add_argument("--base-checkpoint", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--vjepa2-model-id", default=None)
    parser.add_argument("--dagger-root", type=Path, action="append", default=[])
    parser.add_argument("--include-success-dagger", action="store_true")
    parser.add_argument("--dagger-repeat", type=int, default=4)
    parser.add_argument("--max-episodes", type=int, default=240)
    parser.add_argument("--max-dagger-episodes", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--embed-dim", type=int, default=512)
    parser.add_argument("--action-chunk-size", type=int, default=4)
    parser.add_argument(
        "--action-representation",
        choices=["absolute", "delta", "tcp_delta", "tcp_delta_posture"],
        default="delta",
    )
    parser.add_argument("--target-lookahead-steps", type=int, default=4)
    parser.add_argument("--delta-arm-target-clip", type=float, default=0.02)
    parser.add_argument("--delta-tcp-target-clip", type=float, default=0.04)
    parser.add_argument("--delta-gripper-target-clip", type=float, default=0.15)
    parser.add_argument("--key-phase-repeat", type=int, default=4)
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    parser.add_argument("--text-hash-buckets", type=int, default=512)
    parser.add_argument("--text-token-count", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def instruction_to_hash_tokens(text, buckets=512, token_count=16):
    words = [
        item.strip().lower()
        for item in str(text).replace(",", " ").replace(".", " ").split()
        if item.strip()
    ]
    if not words:
        words = ["instruction"]
    tokens = []
    for word in words[:token_count]:
        digest = hashlib.md5(word.encode("utf-8")).hexdigest()
        tokens.append(int(digest[:8], 16) % int(buckets))
    while len(tokens) < int(token_count):
        tokens.append(0)
    return torch.tensor(tokens[: int(token_count)], dtype=torch.long)


class SmolVLAPolicy(nn.Module):
    def __init__(
        self,
        vjepa2_model_id,
        embed_dim,
        num_phases,
        state_dim,
        action_chunk_size=4,
        text_hash_buckets=512,
        text_token_count=16,
        local_files_only=False,
    ):
        super().__init__()
        from transformers import AutoConfig, VJEPA2Model

        config = AutoConfig.from_pretrained(
            vjepa2_model_id,
            local_files_only=local_files_only,
        )
        self.vjepa2 = VJEPA2Model.from_pretrained(
            vjepa2_model_id,
            local_files_only=local_files_only,
        )
        for param in self.vjepa2.parameters():
            param.requires_grad = False

        hidden = int(getattr(config, "hidden_size", 1024))
        self.action_chunk_size = int(action_chunk_size)
        self.text_token_count = int(text_token_count)
        self.visual_encoder = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.state_encoder = nn.Sequential(
            nn.LayerNorm(state_dim),
            nn.Linear(state_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.text_embedding = nn.Embedding(int(text_hash_buckets), embed_dim)
        self.text_encoder = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=8,
            dim_feedforward=embed_dim * 4,
            dropout=0.05,
            batch_first=True,
            activation="gelu",
        )
        self.fusion = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.post_fusion = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
        )
        self.phase_head = nn.Linear(embed_dim, num_phases)
        self.action_head = nn.Linear(embed_dim, 14)
        self.chunk_head = nn.Linear(embed_dim, self.action_chunk_size * 7)

    def forward(self, frames, state, instruction_tokens):
        self.vjepa2.eval()
        with torch.no_grad():
            output = self.vjepa2(pixel_values_videos=frames, skip_predictor=True)
        pooled = output.last_hidden_state.mean(dim=1)
        visual_token = self.visual_encoder(pooled)
        state_token = self.state_encoder(state)
        text_tokens = self.text_embedding(instruction_tokens)
        text_token = self.text_encoder(text_tokens.mean(dim=1))
        fused = self.fusion(
            torch.stack([visual_token, state_token, text_token], dim=1)
        ).mean(dim=1)
        latent = self.post_fusion(fused)
        return (
            self.phase_head(latent),
            self.action_head(latent),
            self.chunk_head(latent).view(
                latent.shape[0],
                self.action_chunk_size,
                7,
            ),
        )


def smolvla_collate(batch, instruction_tokens):
    if isinstance(instruction_tokens, dict):
        token_batch = [
            instruction_tokens.get(item["episode_id"], instruction_tokens["__default__"])
            for item in batch
        ]
        instruction_batch = torch.stack(token_batch)
    else:
        instruction_batch = instruction_tokens.unsqueeze(0).repeat(len(batch), 1)
    return {
        "episode_id": [item["episode_id"] for item in batch],
        "frames": torch.stack([item["frames"] for item in batch]),
        "state": torch.stack([item["state"] for item in batch]),
        "action_target": torch.stack([item["action_target"] for item in batch]),
        "phase_id": torch.stack([item["phase_id"] for item in batch]),
        "instruction_tokens": instruction_batch,
    }


def episode_instruction(episode, fallback):
    metadata = episode.get("metadata", {})
    candidates = [
        metadata.get("task_instruction"),
        metadata.get("language_instruction"),
        metadata.get("task"),
        metadata.get("raw_metadata", {}).get("task_instruction"),
        metadata.get("raw_metadata", {}).get("language_instruction"),
        metadata.get("raw_metadata", {}).get("task"),
        episode.get("language_instruction"),
        episode.get("episode_metadata", {}).get("task_instruction"),
        episode.get("episode_metadata", {}).get("language_instruction"),
        episode.get("episode_metadata", {}).get("task"),
    ]
    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    steps = episode.get("steps", [])
    if steps:
        observation = steps[0].get("observation", {})
        candidate = observation.get("natural_language_instruction")
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return fallback


def build_instruction_tokens_by_episode(episodes, args):
    default_tokens = instruction_to_hash_tokens(
        args.instruction,
        args.text_hash_buckets,
        args.text_token_count,
    )
    tokens = {"__default__": default_tokens}
    instruction_counts = {}
    for episode in episodes:
        instruction = episode_instruction(episode, args.instruction)
        tokens[episode["episode_id"]] = instruction_to_hash_tokens(
            instruction,
            args.text_hash_buckets,
            args.text_token_count,
        )
        instruction_counts[instruction] = instruction_counts.get(instruction, 0) + 1
    return tokens, instruction_counts


def evaluate(model, loader, device, state_mean, state_std):
    model.eval()
    totals = {"loss": 0.0, "action": 0.0, "chunk": 0.0, "phase": 0.0, "accuracy": 0.0}
    count = 0
    with torch.no_grad():
        for batch in loader:
            frames = batch["frames"].to(device)
            state = (batch["state"].to(device) - state_mean) / state_std
            action_target = batch["action_target"].to(device)
            phase_id = batch["phase_id"].to(device)
            instruction_tokens = batch["instruction_tokens"].to(device)
            phase_logits, action_pred, chunk_pred = model(
                frames,
                state,
                instruction_tokens,
            )
            chunk_target = action_target[:, -7:].unsqueeze(1).repeat(
                1,
                chunk_pred.shape[1],
                1,
            )
            action_loss = F.smooth_l1_loss(action_pred, action_target)
            chunk_loss = F.smooth_l1_loss(chunk_pred, chunk_target)
            phase_loss = F.cross_entropy(phase_logits, phase_id)
            loss = action_loss + 0.25 * chunk_loss + 0.25 * phase_loss
            batch_count = phase_id.numel()
            count += batch_count
            totals["loss"] += float(loss.detach().cpu()) * batch_count
            totals["action"] += float(action_loss.detach().cpu()) * batch_count
            totals["chunk"] += float(chunk_loss.detach().cpu()) * batch_count
            totals["phase"] += float(phase_loss.detach().cpu()) * batch_count
            totals["accuracy"] += float(
                (phase_logits.argmax(dim=-1) == phase_id).float().mean().detach().cpu()
            ) * batch_count
    return {key: value / max(1, count) for key, value in totals.items()} | {
        "samples": count
    }


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    root = args.rlds_root.resolve()
    base_checkpoint = torch.load(
        args.base_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    base_config = base_checkpoint["config"]
    vjepa2_model_id = args.vjepa2_model_id or base_config["vjepa2_model_id"]
    phase_to_id = base_checkpoint["phase_to_id"]
    clip_frames = int(base_config["clip_frames"])
    image_size = int(base_config["image_size"])

    episodes = load_episodes(root, args.max_episodes)
    dagger_episodes = load_dagger_episodes(
        args.dagger_root,
        args.max_dagger_episodes,
        args.dagger_repeat,
        args.include_success_dagger,
    )
    episodes.extend(dagger_episodes)
    state_dim = pad_state_features(episodes)
    dataset = DirectCorrectionDataset(
        root,
        episodes,
        phase_to_id,
        clip_frames,
        image_size,
        args.key_phase_repeat,
        args.max_samples,
        args.seed,
        args.action_representation,
        args.delta_arm_target_clip,
        args.delta_tcp_target_clip,
        0.0,
        args.delta_gripper_target_clip,
        args.target_lookahead_steps,
    )
    instruction_tokens, instruction_counts = build_instruction_tokens_by_episode(
        episodes,
        args,
    )
    print(
        "smolvla_ready "
        f"episodes={len(episodes)} samples={len(dataset)} "
        f"state_dim={state_dim} phases={len(phase_to_id)} "
        f"action_representation={args.action_representation} "
        f"chunk={args.action_chunk_size} "
        f"instructions={len(instruction_counts)}"
    )
    for instruction, count in sorted(
        instruction_counts.items(),
        key=lambda item: (-item[1], item[0]),
    )[:12]:
        print(f"instruction_count={count} text={instruction}", flush=True)
    if args.dry_run:
        return

    validation_size = max(1, int(len(dataset) * 0.08))
    train_size = max(1, len(dataset) - validation_size)
    generator = torch.Generator().manual_seed(args.seed)
    train_dataset, validation_dataset = random_split(
        dataset,
        [train_size, validation_size],
        generator=generator,
    )
    collate = lambda batch: smolvla_collate(batch, instruction_tokens)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate,
    )
    state_mean, state_std = compute_state_stats(dataset)
    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    )
    state_mean = state_mean.to(device)
    state_std = state_std.to(device)
    model = SmolVLAPolicy(
        vjepa2_model_id,
        args.embed_dim,
        len(phase_to_id),
        state_dim,
        args.action_chunk_size,
        args.text_hash_buckets,
        args.text_token_count,
        args.local_files_only,
    ).to(device)
    optimizer = torch.optim.AdamW(
        [
            param
            for param in model.parameters()
            if param.requires_grad
        ],
        lr=args.learning_rate,
        weight_decay=1e-4,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = SmolVLAConfig(
        rlds_root=str(root),
        output_dir=str(args.output_dir.resolve()),
        vjepa2_model_id=vjepa2_model_id,
        base_checkpoint=str(args.base_checkpoint.resolve()),
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_episodes=args.max_episodes,
        max_samples=args.max_samples,
        clip_frames=clip_frames,
        image_size=image_size,
        embed_dim=args.embed_dim,
        state_dim=state_dim,
        action_chunk_size=args.action_chunk_size,
        learning_rate=args.learning_rate,
        action_representation=args.action_representation,
        local_files_only=args.local_files_only,
        instruction=args.instruction,
        text_hash_buckets=args.text_hash_buckets,
        text_token_count=args.text_token_count,
    )
    (args.output_dir / "training_config.json").write_text(
        json.dumps(asdict(config), indent=2),
        encoding="utf-8",
    )
    best_loss = None
    metrics_path = args.output_dir / "metrics.jsonl"
    with metrics_path.open("w", encoding="utf-8") as stream:
        for epoch in range(args.epochs):
            model.train()
            totals = {"loss": 0.0, "action": 0.0, "chunk": 0.0, "phase": 0.0}
            for step, batch in enumerate(train_loader):
                frames = batch["frames"].to(device)
                state = (batch["state"].to(device) - state_mean) / state_std
                action_target = batch["action_target"].to(device)
                phase_id = batch["phase_id"].to(device)
                instruction_batch = batch["instruction_tokens"].to(device)
                phase_logits, action_pred, chunk_pred = model(
                    frames,
                    state,
                    instruction_batch,
                )
                chunk_target = action_target[:, -7:].unsqueeze(1).repeat(
                    1,
                    chunk_pred.shape[1],
                    1,
                )
                action_loss = F.smooth_l1_loss(action_pred, action_target)
                chunk_loss = F.smooth_l1_loss(chunk_pred, chunk_target)
                phase_loss = F.cross_entropy(phase_logits, phase_id)
                loss = action_loss + 0.25 * chunk_loss + 0.25 * phase_loss
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                totals["loss"] += float(loss.detach().cpu())
                totals["action"] += float(action_loss.detach().cpu())
                totals["chunk"] += float(chunk_loss.detach().cpu())
                totals["phase"] += float(phase_loss.detach().cpu())
                if step % 10 == 0:
                    print(
                        f"epoch={epoch + 1}/{args.epochs} "
                        f"step={step + 1}/{len(train_loader)} "
                        f"loss={float(loss.detach().cpu()):.4f}",
                        flush=True,
                    )
            denom = max(1, len(train_loader))
            validation = evaluate(
                model,
                validation_loader,
                device,
                state_mean,
                state_std,
            )
            row = {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "epoch": epoch + 1,
                "train_loss": totals["loss"] / denom,
                "train_action": totals["action"] / denom,
                "train_chunk": totals["chunk"] / denom,
                "train_phase": totals["phase"] / denom,
                **{f"val_{key}": value for key, value in validation.items()},
            }
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")
            stream.flush()
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "config": asdict(config),
                "phase_to_id": phase_to_id,
                "state_mean": state_mean.detach().cpu(),
                "state_std": state_std.detach().cpu(),
                "policy_arch": "smolvla",
                "action_representation": args.action_representation,
                "instruction_tokens": instruction_tokens["__default__"],
                "instruction_counts": instruction_counts,
                "metrics": row,
            }
            torch.save(checkpoint, args.output_dir / "latest_smolvla_policy.pt")
            if best_loss is None or row["val_loss"] < best_loss:
                best_loss = row["val_loss"]
                torch.save(checkpoint, args.output_dir / "best_smolvla_policy.pt")
            print(
                f"saved epoch={epoch + 1} val_loss={row['val_loss']:.4f} "
                f"checkpoint={args.output_dir / 'latest_smolvla_policy.pt'}",
                flush=True,
            )


if __name__ == "__main__":
    main()
