"""Train a lightweight MLLM alignment adapter for the xArm6 RLDS dataset.

This is the phase-3 bootstrap path. It trains video/text/action alignment from
the RLDS JSONL package without requiring Transformers. The model is intentionally
small so the dataset contract, labels, and checkpoints can be validated locally;
the visual encoder can later be replaced by frozen V-JEPA2 features.
"""

import argparse
import json
import math
import random
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")


@dataclass
class TrainConfig:
    rlds_root: str
    annotations: str
    output_dir: str
    epochs: int
    batch_size: int
    max_episodes: int
    clip_frames: int
    image_size: int
    embed_dim: int
    learning_rate: float
    device: str
    seed: int
    vision_backbone: str
    vjepa2_model_id: str
    freeze_vision: bool
    local_files_only: bool


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rlds-root", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/mllm_alignment_tiny"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--clip-frames", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--vision-backbone",
        choices=["tiny", "vjepa2"],
        default="tiny",
        help="Use the local tiny encoder or a frozen Hugging Face V-JEPA2 encoder.",
    )
    parser.add_argument("--vjepa2-model-id", default="facebook/vjepa2-vitl-fpc64-256")
    parser.add_argument("--no-freeze-vision", action="store_true")
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load Hugging Face files from the local cache without network checks.",
    )
    return parser.parse_args()


def tokenize(text):
    return TOKEN_RE.findall(text.lower())


def load_annotations(path):
    annotations = {}
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                row = json.loads(line)
                annotations[row["episode_id"]] = row
    return annotations


def load_episodes(root, max_episodes=0):
    info = json.loads((root / "dataset_info.json").read_text(encoding="utf-8"))
    episodes = []
    for shard in info["shards"]:
        with (root / shard["path"]).open("r", encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                episodes.append(json.loads(line))
                if max_episodes and len(episodes) >= max_episodes:
                    return episodes
    return episodes


def build_vocab(episodes, annotations, min_count=1):
    counts = {}
    for episode in episodes:
        ann = annotations[episode["episode_id"]]
        texts = [
            ann["language_instruction"],
            ann["episode_summary"],
            *ann.get("instruction_variants", []),
        ]
        for step in ann["steps"]:
            texts.append(step["caption"])
            texts.append(step["phase_goal"])
        for text in texts:
            for token in tokenize(text):
                counts[token] = counts.get(token, 0) + 1
    vocab = {"<pad>": 0, "<unk>": 1}
    for token, count in sorted(counts.items()):
        if count >= min_count:
            vocab[token] = len(vocab)
    return vocab


def encode_text(text, vocab, max_tokens=96):
    ids = [vocab.get(token, vocab["<unk>"]) for token in tokenize(text)[:max_tokens]]
    mask = [1] * len(ids)
    while len(ids) < max_tokens:
        ids.append(vocab["<pad>"])
        mask.append(0)
    return torch.tensor(ids, dtype=torch.long), torch.tensor(mask, dtype=torch.float32)


class RLDSEpisodeDataset(Dataset):
    def __init__(self, root, episodes, annotations, vocab, clip_frames, image_size):
        self.root = root
        self.episodes = episodes
        self.annotations = annotations
        self.vocab = vocab
        self.clip_frames = clip_frames
        self.image_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.phase_to_id = self._build_phase_map()

    def _build_phase_map(self):
        phases = sorted(
            {
                step["observation"]["phase"]
                for episode in self.episodes
                for step in episode["steps"]
            }
        )
        return {phase: index for index, phase in enumerate(phases)}

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, index):
        episode = self.episodes[index]
        steps = episode["steps"]
        if len(steps) <= self.clip_frames:
            selected = list(range(len(steps)))
        else:
            start = random.randint(0, len(steps) - self.clip_frames)
            selected = list(range(start, start + self.clip_frames))
        while len(selected) < self.clip_frames:
            selected.append(selected[-1])

        frames = []
        actions = []
        phase_ids = []
        for step_index in selected:
            step = steps[step_index]
            image_path = self.root / step["observation"]["image"]
            image = Image.open(image_path).convert("RGB")
            frames.append(self.image_transform(image))
            actions.append(torch.tensor(step["action"], dtype=torch.float32))
            phase_ids.append(self.phase_to_id[step["observation"]["phase"]])

        ann = self.annotations[episode["episode_id"]]
        center_step = ann["steps"][selected[len(selected) // 2]]
        text = " ".join(
            [
                ann["language_instruction"],
                ann["episode_summary"],
                center_step["caption"],
            ]
        )
        token_ids, token_mask = encode_text(text, self.vocab)
        phase_id = torch.tensor(phase_ids[len(phase_ids) // 2], dtype=torch.long)
        action_mean = torch.stack(actions).mean(dim=0)
        action_last = actions[-1]
        action_target = torch.cat([action_mean, action_last], dim=0)

        return {
            "episode_id": episode["episode_id"],
            "frames": torch.stack(frames),
            "token_ids": token_ids,
            "token_mask": token_mask,
            "action_target": action_target,
            "phase_id": phase_id,
        }


class TinyFrameEncoder(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(128, embed_dim)

    def forward(self, frames):
        batch, time, channels, height, width = frames.shape
        x = frames.reshape(batch * time, channels, height, width)
        x = self.net(x).flatten(1)
        x = self.proj(x)
        return x.reshape(batch, time, -1).mean(dim=1)


class VJEPA2VideoEncoder(nn.Module):
    def __init__(self, model_id, embed_dim, freeze=True, local_files_only=False):
        super().__init__()
        from transformers import AutoConfig, VJEPA2Model

        self.config = AutoConfig.from_pretrained(model_id, local_files_only=local_files_only)
        self.model = VJEPA2Model.from_pretrained(model_id, local_files_only=local_files_only)
        hidden_size = int(getattr(self.config, "hidden_size", 1024))
        self.proj = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, embed_dim),
        )
        if freeze:
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False
        self.freeze = freeze

    def forward(self, frames):
        if self.freeze:
            self.model.eval()
            with torch.no_grad():
                output = self.model(pixel_values_videos=frames, skip_predictor=True)
        else:
            output = self.model(pixel_values_videos=frames, skip_predictor=True)
        tokens = output.last_hidden_state
        return self.proj(tokens.mean(dim=1))


class TextEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, token_ids, token_mask):
        embeds = self.embedding(token_ids)
        masked = embeds * token_mask.unsqueeze(-1)
        denom = token_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        return self.proj(masked.sum(dim=1) / denom)


class AlignmentAdapter(nn.Module):
    def __init__(
        self,
        vocab_size,
        num_phases,
        embed_dim,
        vision_backbone="tiny",
        vjepa2_model_id="facebook/vjepa2-vitl-fpc64-256",
        freeze_vision=True,
        local_files_only=False,
    ):
        super().__init__()
        if vision_backbone == "vjepa2":
            self.video_encoder = VJEPA2VideoEncoder(
                model_id=vjepa2_model_id,
                embed_dim=embed_dim,
                freeze=freeze_vision,
                local_files_only=local_files_only,
            )
        else:
            self.video_encoder = TinyFrameEncoder(embed_dim)
        self.text_encoder = TextEncoder(vocab_size, embed_dim)
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.SiLU(),
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
        )
        self.action_head = nn.Linear(embed_dim, 14)
        self.phase_head = nn.Linear(embed_dim, num_phases)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))

    def forward(self, frames, token_ids, token_mask):
        video = self.video_encoder(frames)
        text = self.text_encoder(token_ids, token_mask)
        fused = self.fusion(torch.cat([video, text], dim=-1))
        return {
            "video": F.normalize(video, dim=-1),
            "text": F.normalize(text, dim=-1),
            "fused": fused,
            "action": self.action_head(fused),
            "phase": self.phase_head(fused),
        }


def contrastive_loss(video, text, logit_scale):
    if video.shape[0] < 2:
        return video.new_tensor(0.0)
    logits = logit_scale.exp().clamp(max=100.0) * video @ text.t()
    labels = torch.arange(video.shape[0], device=video.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


def collate(batch):
    return {
        "episode_id": [item["episode_id"] for item in batch],
        "frames": torch.stack([item["frames"] for item in batch]),
        "token_ids": torch.stack([item["token_ids"] for item in batch]),
        "token_mask": torch.stack([item["token_mask"] for item in batch]),
        "action_target": torch.stack([item["action_target"] for item in batch]),
        "phase_id": torch.stack([item["phase_id"] for item in batch]),
    }


def main():
    args = parse_args()
    if args.vision_backbone == "vjepa2":
        try:
            import transformers  # noqa: F401
        except Exception as exc:
            raise SystemExit(
                "vision-backbone=vjepa2 requires transformers/tokenizers/safetensors. "
                "Install those packages first, then rerun this script."
            ) from exc
        if args.image_size != 256:
            print("V-JEPA2 expects 256x256 input; overriding --image-size to 256.")
            args.image_size = 256

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    root = args.rlds_root.resolve()
    annotations = load_annotations(args.annotations.resolve())
    episodes = load_episodes(root, args.max_episodes)
    missing = [episode["episode_id"] for episode in episodes if episode["episode_id"] not in annotations]
    if missing:
        raise RuntimeError(f"Missing annotations for episodes: {missing[:5]}")

    vocab = build_vocab(episodes, annotations)
    dataset = RLDSEpisodeDataset(
        root=root,
        episodes=episodes,
        annotations=annotations,
        vocab=vocab,
        clip_frames=args.clip_frames,
        image_size=args.image_size,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate,
    )

    sample = dataset[0]
    print(
        "dataset_ready "
        f"episodes={len(dataset)} vocab={len(vocab)} phases={len(dataset.phase_to_id)} "
        f"frames={tuple(sample['frames'].shape)} action_target={tuple(sample['action_target'].shape)}"
    )
    if args.dry_run:
        print(f"sample_episode={sample['episode_id']} phase_map={dataset.phase_to_id}")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = TrainConfig(
        rlds_root=str(root),
        annotations=str(args.annotations.resolve()),
        output_dir=str(args.output_dir.resolve()),
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_episodes=args.max_episodes,
        clip_frames=args.clip_frames,
        image_size=args.image_size,
        embed_dim=args.embed_dim,
        learning_rate=args.learning_rate,
        device=str(device),
        seed=args.seed,
        vision_backbone=args.vision_backbone,
        vjepa2_model_id=args.vjepa2_model_id,
        freeze_vision=not args.no_freeze_vision,
        local_files_only=args.local_files_only,
    )
    (args.output_dir / "training_config.json").write_text(
        json.dumps(asdict(config), indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "vocab.json").write_text(json.dumps(vocab, indent=2), encoding="utf-8")
    (args.output_dir / "phase_to_id.json").write_text(
        json.dumps(dataset.phase_to_id, indent=2),
        encoding="utf-8",
    )

    model = AlignmentAdapter(
        vocab_size=len(vocab),
        num_phases=len(dataset.phase_to_id),
        embed_dim=args.embed_dim,
        vision_backbone=args.vision_backbone,
        vjepa2_model_id=args.vjepa2_model_id,
        freeze_vision=not args.no_freeze_vision,
        local_files_only=args.local_files_only,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    metrics_path = args.output_dir / "metrics.jsonl"
    total_steps = 0

    with metrics_path.open("w", encoding="utf-8") as metrics_stream:
        for epoch in range(args.epochs):
            model.train()
            totals = {"loss": 0.0, "action": 0.0, "phase": 0.0, "contrastive": 0.0}
            for batch_index, batch in enumerate(loader):
                frames = batch["frames"].to(device)
                token_ids = batch["token_ids"].to(device)
                token_mask = batch["token_mask"].to(device)
                action_target = batch["action_target"].to(device)
                phase_id = batch["phase_id"].to(device)

                output = model(frames, token_ids, token_mask)
                action_loss = F.smooth_l1_loss(output["action"], action_target)
                phase_loss = F.cross_entropy(output["phase"], phase_id)
                clip_loss = contrastive_loss(output["video"], output["text"], model.logit_scale)
                loss = action_loss + 0.5 * phase_loss + 0.1 * clip_loss

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                total_steps += 1
                values = {
                    "loss": float(loss.detach().cpu()),
                    "action": float(action_loss.detach().cpu()),
                    "phase": float(phase_loss.detach().cpu()),
                    "contrastive": float(clip_loss.detach().cpu()),
                }
                for key, value in values.items():
                    totals[key] += value
                if batch_index % 10 == 0:
                    print(
                        f"epoch={epoch + 1}/{args.epochs} step={batch_index + 1}/{len(loader)} "
                        f"loss={values['loss']:.4f} action={values['action']:.4f} "
                        f"phase={values['phase']:.4f} contrastive={values['contrastive']:.4f}"
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
                    "vocab": vocab,
                    "phase_to_id": dataset.phase_to_id,
                    "epoch": epoch + 1,
                    "metrics": row,
                },
                args.output_dir / "latest.pt",
            )
            print(
                f"saved epoch={epoch + 1} checkpoint={args.output_dir / 'latest.pt'} "
                f"loss={row['loss']:.4f}"
            )


if __name__ == "__main__":
    main()
