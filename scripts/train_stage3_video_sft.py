"""Stage 3 video SFT policy: video clips to phase and robot action targets."""

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
class Stage3Config:
    rlds_root: str
    output_dir: str
    vjepa2_model_id: str
    epochs: int
    batch_size: int
    max_episodes: int
    windows_per_episode: int
    clip_frames: int
    image_size: int
    embed_dim: int
    learning_rate: float
    device: str
    local_files_only: bool


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rlds-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/stage3_video_sft_policy"))
    parser.add_argument("--vjepa2-model-id", default="facebook/vjepa2-vitl-fpc64-256")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--windows-per-episode", type=int, default=1)
    parser.add_argument("--clip-frames", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--embed-dim", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_episodes(root, max_episodes=0):
    info = json.loads((root / "dataset_info.json").read_text(encoding="utf-8"))
    episodes = []
    for shard in info["shards"]:
        with (root / shard["path"]).open("r", encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    episodes.append(json.loads(line))
                    if max_episodes and len(episodes) >= max_episodes:
                        return episodes
    return episodes


class VideoSFTDataset(Dataset):
    def __init__(self, root, episodes, clip_frames, image_size, windows_per_episode=1):
        self.root = root
        self.episodes = episodes
        self.clip_frames = clip_frames
        self.windows_per_episode = max(1, windows_per_episode)
        self.phase_to_id = self._phase_map()
        self.image_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def _phase_map(self):
        phases = sorted(
            {
                step["observation"]["phase"]
                for episode in self.episodes
                for step in episode["steps"]
            }
        )
        return {phase: index for index, phase in enumerate(phases)}

    def __len__(self):
        return len(self.episodes) * self.windows_per_episode

    def __getitem__(self, index):
        episode_index = index // self.windows_per_episode
        window_index = index % self.windows_per_episode
        episode = self.episodes[episode_index]
        steps = episode["steps"]
        if len(steps) <= self.clip_frames:
            selected = list(range(len(steps)))
        else:
            if self.windows_per_episode == 1:
                start = random.randint(0, len(steps) - self.clip_frames)
            else:
                max_start = len(steps) - self.clip_frames
                start = round(max_start * window_index / (self.windows_per_episode - 1))
            selected = list(range(start, start + self.clip_frames))
        while len(selected) < self.clip_frames:
            selected.append(selected[-1])

        frames = []
        actions = []
        phase_ids = []
        for step_index in selected:
            step = steps[step_index]
            frames.append(self.image_transform(Image.open(self.root / step["observation"]["image"]).convert("RGB")))
            actions.append(torch.tensor(step["action"], dtype=torch.float32))
            phase_ids.append(self.phase_to_id[step["observation"]["phase"]])
        action_mean = torch.stack(actions).mean(dim=0)
        action_last = actions[-1]
        return {
            "episode_id": episode["episode_id"],
            "frames": torch.stack(frames),
            "action_target": torch.cat([action_mean, action_last], dim=0),
            "phase_id": torch.tensor(phase_ids[len(phase_ids) // 2], dtype=torch.long),
        }


class Stage3Policy(nn.Module):
    def __init__(self, vjepa2_model_id, embed_dim, num_phases, local_files_only=False):
        super().__init__()
        from transformers import AutoConfig, VJEPA2Model

        config = AutoConfig.from_pretrained(vjepa2_model_id, local_files_only=local_files_only)
        self.vjepa2 = VJEPA2Model.from_pretrained(vjepa2_model_id, local_files_only=local_files_only)
        for param in self.vjepa2.parameters():
            param.requires_grad = False
        hidden = int(getattr(config, "hidden_size", 1024))
        self.temporal = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.phase_head = nn.Linear(embed_dim, num_phases)
        self.action_head = nn.Linear(embed_dim, 14)

    def forward(self, frames):
        self.vjepa2.eval()
        with torch.no_grad():
            output = self.vjepa2(pixel_values_videos=frames, skip_predictor=True)
        pooled = output.last_hidden_state.mean(dim=1)
        latent = self.temporal(pooled)
        return self.phase_head(latent), self.action_head(latent)


def collate(batch):
    return {
        "episode_id": [item["episode_id"] for item in batch],
        "frames": torch.stack([item["frames"] for item in batch]),
        "action_target": torch.stack([item["action_target"] for item in batch]),
        "phase_id": torch.stack([item["phase_id"] for item in batch]),
    }


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    root = args.rlds_root.resolve()
    episodes = load_episodes(root, args.max_episodes)
    dataset = VideoSFTDataset(
        root,
        episodes,
        args.clip_frames,
        args.image_size,
        args.windows_per_episode,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate)
    sample = dataset[0]
    print(
        f"stage3_video_sft_ready episodes={len(episodes)} samples={len(dataset)} "
        f"windows_per_episode={args.windows_per_episode} phases={len(dataset.phase_to_id)} "
        f"frames={tuple(sample['frames'].shape)} action_target={tuple(sample['action_target'].shape)}"
    )
    if args.dry_run:
        print(f"phase_to_id={dataset.phase_to_id}")
        return

    model = Stage3Policy(args.vjepa2_model_id, args.embed_dim, len(dataset.phase_to_id), args.local_files_only).to(device)
    optimizer = torch.optim.AdamW(
        list(model.temporal.parameters()) + list(model.phase_head.parameters()) + list(model.action_head.parameters()),
        lr=args.learning_rate,
        weight_decay=1e-4,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = Stage3Config(
        rlds_root=str(root),
        output_dir=str(args.output_dir.resolve()),
        vjepa2_model_id=args.vjepa2_model_id,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_episodes=args.max_episodes,
        windows_per_episode=args.windows_per_episode,
        clip_frames=args.clip_frames,
        image_size=args.image_size,
        embed_dim=args.embed_dim,
        learning_rate=args.learning_rate,
        device=str(device),
        local_files_only=args.local_files_only,
    )
    (args.output_dir / "training_config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
    (args.output_dir / "phase_to_id.json").write_text(json.dumps(dataset.phase_to_id, indent=2), encoding="utf-8")
    metrics_path = args.output_dir / "metrics.jsonl"
    total_steps = 0
    with metrics_path.open("w", encoding="utf-8") as metrics_stream:
        for epoch in range(args.epochs):
            totals = {"loss": 0.0, "action": 0.0, "phase": 0.0, "accuracy": 0.0}
            for step, batch in enumerate(loader):
                frames = batch["frames"].to(device)
                phase_id = batch["phase_id"].to(device)
                action_target = batch["action_target"].to(device)
                phase_logits, action_pred = model(frames)
                phase_loss = F.cross_entropy(phase_logits, phase_id)
                action_loss = F.smooth_l1_loss(action_pred, action_target)
                loss = action_loss + 0.5 * phase_loss
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_steps += 1
                acc = (phase_logits.argmax(dim=-1) == phase_id).float().mean()
                values = {
                    "loss": float(loss.detach().cpu()),
                    "action": float(action_loss.detach().cpu()),
                    "phase": float(phase_loss.detach().cpu()),
                    "accuracy": float(acc.detach().cpu()),
                }
                for key, value in values.items():
                    totals[key] += value
                if step % 10 == 0:
                    print(f"epoch={epoch + 1}/{args.epochs} step={step + 1}/{len(loader)} loss={values['loss']:.4f} acc={values['accuracy']:.3f}")
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
                    "phase_to_id": dataset.phase_to_id,
                    "epoch": epoch + 1,
                    "metrics": row,
                },
                args.output_dir / "latest_stage3_policy.pt",
            )
            print(f"saved epoch={epoch + 1} checkpoint={args.output_dir / 'latest_stage3_policy.pt'} loss={row['loss']:.4f}")


if __name__ == "__main__":
    main()
