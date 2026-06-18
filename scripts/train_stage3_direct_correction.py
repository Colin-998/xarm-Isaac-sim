"""Fine-tune Stage-3 policy on dense grasp/place windows for direct control."""

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
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms

from train_stage3_video_sft import Stage3Policy, load_episodes


KEY_PHASES = {
    "approach_cube",
    "descend_to_cube",
    "close_gripper",
    "lift_cube",
    "place_cube",
    "open_gripper",
    "retreat_after_release",
}


@dataclass
class DirectCorrectionConfig:
    rlds_root: str
    dagger_roots: list[str]
    base_checkpoint: str
    output_dir: str
    epochs: int
    batch_size: int
    max_episodes: int
    max_dagger_episodes: int
    dagger_repeat: int
    max_samples: int
    clip_frames: int
    image_size: int
    learning_rate: float
    key_phase_repeat: int
    action_loss_weight: float
    phase_loss_weight: float
    train_temporal: bool
    train_phase_head: bool
    device: str
    local_files_only: bool
    state_conditioned: bool


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rlds-root", type=Path, required=True)
    parser.add_argument(
        "--dagger-root",
        type=Path,
        action="append",
        default=[],
        help=(
            "Optional raw episode root containing failed direct-control "
            "episodes. These samples use oracle action labels and executed "
            "policy observations."
        ),
    )
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/stage3_direct_correction"))
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--max-dagger-episodes", type=int, default=0)
    parser.add_argument(
        "--dagger-repeat",
        type=int,
        default=1,
        help="Repeat failed direct-control episodes to increase correction weight.",
    )
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--key-phase-repeat", type=int, default=4)
    parser.add_argument("--action-loss-weight", type=float, default=1.5)
    parser.add_argument("--phase-loss-weight", type=float, default=0.35)
    parser.add_argument(
        "--train-temporal",
        action="store_true",
        help="Also fine-tune the temporal projection. Default keeps representation stable.",
    )
    parser.add_argument(
        "--train-phase-head",
        action="store_true",
        help="Also fine-tune the phase classifier. Default preserves the base phase policy.",
    )
    parser.add_argument(
        "--state-conditioned",
        action="store_true",
        help="Train a visual + robot-state policy for direct control.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def flatten_raw_state(observation):
    gripper = observation["gripper_joint_positions"]
    return (
        observation["arm_joint_positions"]
        + [gripper["drive_joint"]]
        + observation["tcp_position"]
        + observation["cube_position"]
    )


def load_dagger_episodes(roots, max_episodes=0, repeat=1):
    episodes = []
    repeat = max(1, int(repeat))
    for root in roots:
        root = root.resolve()
        for episode_dir in sorted(root.glob("episode_[0-9][0-9][0-9][0-9][0-9]")):
            metadata_file = episode_dir / "metadata.json"
            actions_file = episode_dir / "actions.jsonl"
            if not metadata_file.exists() or not actions_file.exists():
                continue
            metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
            if metadata.get("success", True):
                continue
            steps = []
            with actions_file.open("r", encoding="utf-8") as stream:
                for line in stream:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    action = row["action"]["arm_joint_positions"] + [
                        row["action"]["gripper_joint_position"]
                    ]
                    observation = row["observation"]
                    steps.append(
                        {
                            "observation": {
                                "image": row["image"],
                                "image_abs": str((episode_dir / row["image"]).resolve()),
                                "natural_language_instruction": (
                                    "Recover from direct-control drift, pick up "
                                    "the red cube, and place it at the conveyor start."
                                ),
                                "state": flatten_raw_state(observation),
                                "arm_joint_positions": observation[
                                    "arm_joint_positions"
                                ],
                                "arm_joint_velocities": observation[
                                    "arm_joint_velocities"
                                ],
                                "gripper_joint_positions": observation[
                                    "gripper_joint_positions"
                                ],
                                "tcp_position": observation["tcp_position"],
                                "tcp_rotation_matrix": observation[
                                    "tcp_rotation_matrix"
                                ],
                                "cube_position": observation["cube_position"],
                                "cube_orientation_wxyz": observation[
                                    "cube_orientation_wxyz"
                                ],
                                "obstacles": observation["obstacles"],
                                "phase": row["phase"],
                                "time_seconds": row["time_seconds"],
                            },
                            "action": action,
                            "executed_action": row.get("executed_action"),
                            "reward": 0.0,
                            "discount": 1.0,
                            "is_first": len(steps) == 0,
                            "is_last": False,
                            "is_terminal": False,
                        }
                    )
            if len(steps) < 2:
                continue
            steps[-1]["is_last"] = True
            steps[-1]["is_terminal"] = True
            steps[-1]["discount"] = 0.0
            for repeat_index in range(repeat):
                episodes.append(
                    {
                        "episode_id": (
                            f"{root.name}/{episode_dir.name}/r{repeat_index:02d}"
                        ),
                        "steps": steps,
                        "metadata": {
                            "source": "dagger_failed_direct",
                            "repeat_index": repeat_index,
                            "raw_metadata": metadata,
                        },
                    }
                )
            if max_episodes and len(episodes) >= max_episodes:
                return episodes[:max_episodes]
    return episodes


class DirectCorrectionDataset(Dataset):
    def __init__(
        self,
        root,
        episodes,
        phase_to_id,
        clip_frames,
        image_size,
        key_phase_repeat,
        max_samples,
        seed,
    ):
        self.root = root
        self.episodes = episodes
        self.phase_to_id = phase_to_id
        self.clip_frames = int(clip_frames)
        self.image_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        self.samples = self._build_samples(key_phase_repeat)
        if max_samples and len(self.samples) > max_samples:
            rng = random.Random(seed)
            self.samples = rng.sample(self.samples, max_samples)

    def _build_samples(self, key_phase_repeat):
        samples = []
        repeat = max(1, int(key_phase_repeat))
        for episode_index, episode in enumerate(self.episodes):
            for step_index, step in enumerate(episode["steps"]):
                phase = step["observation"]["phase"]
                if phase not in self.phase_to_id:
                    continue
                copies = repeat if phase in KEY_PHASES else 1
                for _ in range(copies):
                    samples.append((episode_index, step_index))
        return samples

    def __len__(self):
        return len(self.samples)

    def _clip_indices(self, step_index):
        start = step_index - self.clip_frames + 1
        return [max(0, index) for index in range(start, step_index + 1)]

    def __getitem__(self, index):
        episode_index, step_index = self.samples[index]
        episode = self.episodes[episode_index]
        steps = episode["steps"]
        selected = self._clip_indices(step_index)
        frames = []
        action_window = []
        for selected_index in selected:
            step = steps[selected_index]
            image_path = (
                Path(step["observation"]["image_abs"])
                if "image_abs" in step["observation"]
                else self.root / step["observation"]["image"]
            )
            frames.append(
                self.image_transform(
                    Image.open(image_path).convert("RGB")
                )
            )
            action_window.append(torch.tensor(step["action"], dtype=torch.float32))
        target_step = steps[step_index]
        target_action = torch.tensor(target_step["action"], dtype=torch.float32)
        mean_action = torch.stack(action_window).mean(dim=0)
        phase = target_step["observation"]["phase"]
        tcp = torch.tensor(target_step["observation"]["tcp_position"], dtype=torch.float32)
        cube = torch.tensor(target_step["observation"]["cube_position"], dtype=torch.float32)
        tcp_cube_distance = torch.linalg.vector_norm(tcp - cube)
        key_phase = phase in KEY_PHASES
        return {
            "episode_id": episode["episode_id"],
            "frames": torch.stack(frames),
            "state": torch.tensor(target_step["observation"]["state"], dtype=torch.float32),
            "action_target": torch.cat([mean_action, target_action], dim=0),
            "phase_id": torch.tensor(self.phase_to_id[phase], dtype=torch.long),
            "key_phase": torch.tensor(float(key_phase), dtype=torch.float32),
            "tcp_cube_distance": tcp_cube_distance,
        }


def collate(batch):
    return {
        "episode_id": [item["episode_id"] for item in batch],
        "frames": torch.stack([item["frames"] for item in batch]),
        "state": torch.stack([item["state"] for item in batch]),
        "action_target": torch.stack([item["action_target"] for item in batch]),
        "phase_id": torch.stack([item["phase_id"] for item in batch]),
        "key_phase": torch.stack([item["key_phase"] for item in batch]),
        "tcp_cube_distance": torch.stack([item["tcp_cube_distance"] for item in batch]),
    }


class StateConditionedStage3Policy(nn.Module):
    def __init__(
        self,
        vjepa2_model_id,
        embed_dim,
        num_phases,
        state_dim,
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
        self.visual = nn.Sequential(
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
        self.fusion = nn.Sequential(
            nn.LayerNorm(embed_dim * 2),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.phase_head = nn.Linear(embed_dim, num_phases)
        self.action_head = nn.Linear(embed_dim, 14)

    def forward(self, frames, state):
        self.vjepa2.eval()
        with torch.no_grad():
            output = self.vjepa2(pixel_values_videos=frames, skip_predictor=True)
        pooled = output.last_hidden_state.mean(dim=1)
        visual_latent = self.visual(pooled)
        state_latent = self.state_encoder(state)
        latent = self.fusion(torch.cat([visual_latent, state_latent], dim=-1))
        return self.phase_head(latent), self.action_head(latent)


def compute_state_stats(dataset):
    states = []
    for episode_index, step_index in dataset.samples:
        step = dataset.episodes[episode_index]["steps"][step_index]
        states.append(torch.tensor(step["observation"]["state"], dtype=torch.float32))
    state_tensor = torch.stack(states)
    mean = state_tensor.mean(dim=0)
    std = state_tensor.std(dim=0).clamp_min(1e-4)
    return mean, std


def evaluate(
    model,
    loader,
    device,
    action_loss_weight,
    phase_loss_weight,
    state_mean=None,
    state_std=None,
):
    totals = {
        "loss": 0.0,
        "action": 0.0,
        "phase": 0.0,
        "accuracy": 0.0,
        "key_action": 0.0,
        "close_action": 0.0,
    }
    count = 0
    key_count = 0
    close_count = 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            frames = batch["frames"].to(device)
            state = batch["state"].to(device)
            phase_id = batch["phase_id"].to(device)
            action_target = batch["action_target"].to(device)
            key_phase = batch["key_phase"].to(device)
            tcp_cube_distance = batch["tcp_cube_distance"].to(device)
            if getattr(model, "state_conditioned", False):
                state = (state - state_mean.to(device)) / state_std.to(device)
                phase_logits, action_pred = model(frames, state)
            else:
                phase_logits, action_pred = model(frames)
            per_action = F.smooth_l1_loss(
                action_pred,
                action_target,
                reduction="none",
            ).mean(dim=1)
            sample_weights = 1.0 + action_loss_weight * key_phase
            action_loss = (per_action * sample_weights).mean()
            phase_loss = F.cross_entropy(phase_logits, phase_id)
            loss = action_loss + phase_loss_weight * phase_loss
            pred = phase_logits.argmax(dim=-1)
            batch_count = phase_id.numel()
            count += batch_count
            totals["loss"] += float(loss.detach().cpu()) * batch_count
            totals["action"] += float(per_action.mean().detach().cpu()) * batch_count
            totals["phase"] += float(phase_loss.detach().cpu()) * batch_count
            totals["accuracy"] += float((pred == phase_id).float().mean().detach().cpu()) * batch_count
            key_mask = key_phase > 0.5
            if key_mask.any():
                key_count += int(key_mask.sum().item())
                totals["key_action"] += float(per_action[key_mask].sum().detach().cpu())
            close_mask = tcp_cube_distance < 0.08
            if close_mask.any():
                close_count += int(close_mask.sum().item())
                totals["close_action"] += float(per_action[close_mask].sum().detach().cpu())
    return {
        "loss": totals["loss"] / max(1, count),
        "action": totals["action"] / max(1, count),
        "phase": totals["phase"] / max(1, count),
        "accuracy": totals["accuracy"] / max(1, count),
        "key_action": totals["key_action"] / max(1, key_count),
        "close_action": totals["close_action"] / max(1, close_count),
        "samples": count,
        "key_samples": key_count,
        "close_samples": close_count,
    }


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    root = args.rlds_root.resolve()
    base_checkpoint = torch.load(args.base_checkpoint, map_location="cpu", weights_only=False)
    base_config = base_checkpoint["config"]
    phase_to_id = base_checkpoint["phase_to_id"]
    clip_frames = int(base_config["clip_frames"])
    image_size = int(base_config["image_size"])
    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    )
    episodes = load_episodes(root, args.max_episodes)
    dagger_episodes = load_dagger_episodes(
        args.dagger_root,
        args.max_dagger_episodes,
        args.dagger_repeat,
    )
    if dagger_episodes:
        episodes.extend(dagger_episodes)
    dataset = DirectCorrectionDataset(
        root,
        episodes,
        phase_to_id,
        clip_frames,
        image_size,
        args.key_phase_repeat,
        args.max_samples,
        args.seed,
    )
    validation_size = max(1, int(len(dataset) * 0.08))
    train_size = max(1, len(dataset) - validation_size)
    train_dataset, validation_dataset = random_split(
        dataset,
        [train_size, validation_size],
        generator=torch.Generator().manual_seed(args.seed),
    )
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
    print(
        "stage3_direct_correction_ready "
        f"episodes={len(episodes)} dagger_episodes={len(dagger_episodes)} "
        f"dagger_repeat={args.dagger_repeat} "
        f"samples={len(dataset)} "
        f"train={len(train_dataset)} validation={len(validation_dataset)} "
        f"clip_frames={clip_frames} image_size={image_size}"
    )
    if args.dry_run:
        print(f"phase_to_id={phase_to_id}")
        return

    state_mean, state_std = compute_state_stats(dataset)
    if args.state_conditioned:
        model = StateConditionedStage3Policy(
            base_config["vjepa2_model_id"],
            int(base_config["embed_dim"]),
            len(phase_to_id),
            int(state_mean.numel()),
            args.local_files_only or bool(base_config.get("local_files_only", False)),
        ).to(device)
        model.state_conditioned = True
        base_state = base_checkpoint["model_state_dict"]
        visual_state = {
            key.removeprefix("visual."): value
            for key, value in base_state.items()
            if key.startswith("visual.")
        }
        if not visual_state:
            visual_state = {
                key.removeprefix("temporal."): value
                for key, value in base_state.items()
                if key.startswith("temporal.")
            }
        model.visual.load_state_dict(visual_state)
        model.phase_head.load_state_dict(
            {
                key.removeprefix("phase_head."): value
                for key, value in base_state.items()
                if key.startswith("phase_head.")
            }
        )
        model.action_head.load_state_dict(
            {
                key.removeprefix("action_head."): value
                for key, value in base_state.items()
                if key.startswith("action_head.")
            }
        )
    else:
        model = Stage3Policy(
            base_config["vjepa2_model_id"],
            int(base_config["embed_dim"]),
            len(phase_to_id),
            args.local_files_only or bool(base_config.get("local_files_only", False)),
        ).to(device)
        model.load_state_dict(base_checkpoint["model_state_dict"])
        model.state_conditioned = False
        for param in model.temporal.parameters():
            param.requires_grad = bool(args.train_temporal)
        for param in model.phase_head.parameters():
            param.requires_grad = bool(args.train_phase_head)
        for param in model.action_head.parameters():
            param.requires_grad = True
    trainable = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.learning_rate,
        weight_decay=1e-4,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = DirectCorrectionConfig(
        rlds_root=str(root),
        dagger_roots=[str(path.resolve()) for path in args.dagger_root],
        base_checkpoint=str(args.base_checkpoint.resolve()),
        output_dir=str(args.output_dir.resolve()),
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_episodes=args.max_episodes,
        max_dagger_episodes=args.max_dagger_episodes,
        dagger_repeat=args.dagger_repeat,
        max_samples=args.max_samples,
        clip_frames=clip_frames,
        image_size=image_size,
        learning_rate=args.learning_rate,
        key_phase_repeat=args.key_phase_repeat,
        action_loss_weight=args.action_loss_weight,
        phase_loss_weight=args.phase_loss_weight,
        train_temporal=args.train_temporal,
        train_phase_head=args.train_phase_head,
        device=str(device),
        local_files_only=args.local_files_only,
        state_conditioned=args.state_conditioned,
    )
    (args.output_dir / "training_config.json").write_text(
        json.dumps(asdict(config), indent=2),
        encoding="utf-8",
    )
    metrics_path = args.output_dir / "metrics.jsonl"
    total_steps = 0
    with metrics_path.open("w", encoding="utf-8") as metrics_stream:
        for epoch in range(args.epochs):
            model.train()
            for step, batch in enumerate(train_loader):
                frames = batch["frames"].to(device)
                state = batch["state"].to(device)
                phase_id = batch["phase_id"].to(device)
                action_target = batch["action_target"].to(device)
                key_phase = batch["key_phase"].to(device)
                if args.state_conditioned:
                    normalized_state = (
                        state - state_mean.to(device)
                    ) / state_std.to(device)
                    phase_logits, action_pred = model(frames, normalized_state)
                else:
                    phase_logits, action_pred = model(frames)
                per_action = F.smooth_l1_loss(
                    action_pred,
                    action_target,
                    reduction="none",
                ).mean(dim=1)
                sample_weights = 1.0 + args.action_loss_weight * key_phase
                action_loss = (per_action * sample_weights).mean()
                phase_loss = F.cross_entropy(phase_logits, phase_id)
                loss = action_loss + args.phase_loss_weight * phase_loss
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                total_steps += 1
                if step % 25 == 0:
                    accuracy = (phase_logits.argmax(dim=-1) == phase_id).float().mean()
                    print(
                        f"epoch={epoch + 1}/{args.epochs} "
                        f"step={step + 1}/{len(train_loader)} "
                        f"loss={float(loss.detach().cpu()):.4f} "
                        f"action={float(action_loss.detach().cpu()):.4f} "
                        f"acc={float(accuracy.detach().cpu()):.3f}",
                        flush=True,
                    )
            metrics = evaluate(
                model,
                validation_loader,
                device,
                args.action_loss_weight,
                args.phase_loss_weight,
                state_mean,
                state_std,
            )
            row = {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "epoch": epoch + 1,
                "steps": total_steps,
                **metrics,
            }
            metrics_stream.write(json.dumps(row, separators=(",", ":")) + "\n")
            metrics_stream.flush()
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "config": base_config,
                "phase_to_id": phase_to_id,
                "epoch": epoch + 1,
                "metrics": row,
                "direct_correction_config": asdict(config),
                "policy_arch": (
                    "state_conditioned" if args.state_conditioned else "stage3_video"
                ),
                "state_mean": state_mean.tolist(),
                "state_std": state_std.tolist(),
            }
            torch.save(checkpoint, args.output_dir / "latest_stage3_policy.pt")
            print(
                f"saved_direct_correction epoch={epoch + 1} "
                f"checkpoint={args.output_dir / 'latest_stage3_policy.pt'} "
                f"val_action={row['action']:.4f} "
                f"val_close_action={row['close_action']:.4f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
