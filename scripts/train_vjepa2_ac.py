import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2


parser = argparse.ArgumentParser(
    description="Train a compact action-conditioned predictor on frozen V-JEPA2 features."
)
parser.add_argument("--dataset-root", type=Path, required=True)
parser.add_argument("--output-dir", type=Path, default=Path("outputs/vjepa2_ac"))
parser.add_argument("--model", default="facebook/vjepa2-vitl-fpc64-256")
parser.add_argument("--clip-frames", type=int, default=8)
parser.add_argument("--context-frames", type=int, default=4)
parser.add_argument("--batch-size", type=int, default=1)
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--learning-rate", type=float, default=3e-4)
parser.add_argument("--num-workers", type=int, default=0)
parser.add_argument("--device", default="cuda")
parser.add_argument(
    "--dry-run",
    action="store_true",
    help="Validate dataset windows without downloading or loading V-JEPA2.",
)
args = parser.parse_args()


class ConveyorEpisodeDataset(Dataset):
    def __init__(self, root, clip_frames):
        self.root = root.resolve()
        self.clip_frames = clip_frames
        self.transform = v2.Compose(
            [
                v2.Resize((256, 256), antialias=True),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )
        self.windows = []

        for action_file in sorted(self.root.glob("episode_*/actions.jsonl")):
            rows = [
                json.loads(line)
                for line in action_file.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            for start in range(0, len(rows) - clip_frames + 1):
                self.windows.append((action_file.parent, rows[start : start + clip_frames]))

        if not self.windows:
            raise RuntimeError(f"No {clip_frames}-frame windows found below {self.root}")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        episode_dir, rows = self.windows[index]
        frames = [
            self.transform(Image.open(episode_dir / row["image"]).convert("RGB"))
            for row in rows
        ]
        actions = []
        for row in rows:
            action = row["action"]
            actions.append(
                action["arm_joint_positions"] + [action["gripper_joint_position"]]
            )
        return {
            # V-JEPA video convention: channels, time, height, width.
            "video": torch.stack(frames, dim=1),
            "actions": torch.tensor(actions, dtype=torch.float32),
            "phase": rows[-1]["phase"],
        }


def extract_tokens(model_output):
    if hasattr(model_output, "last_hidden_state"):
        return model_output.last_hidden_state
    if isinstance(model_output, dict) and "last_hidden_state" in model_output:
        return model_output["last_hidden_state"]
    if isinstance(model_output, (tuple, list)):
        return model_output[0]
    if torch.is_tensor(model_output):
        return model_output
    raise TypeError(f"Unsupported V-JEPA2 output type: {type(model_output)}")


class ActionConditionedPredictor(nn.Module):
    def __init__(self, latent_dim, action_dim=7, hidden_dim=1024):
        super().__init__()
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, 256),
            nn.GELU(),
            nn.Linear(256, 256),
        )
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim + 256, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, context_latent, action_sequence):
        action_embedding = self.action_encoder(action_sequence).mean(dim=1)
        return self.predictor(torch.cat([context_latent, action_embedding], dim=-1))


dataset = ConveyorEpisodeDataset(args.dataset_root, args.clip_frames)
print(f"Dataset windows: {len(dataset)}")
sample = dataset[0]
print(
    f"Video={tuple(sample['video'].shape)} "
    f"actions={tuple(sample['actions'].shape)} phase={sample['phase']}"
)
if args.dry_run:
    raise SystemExit(0)

if args.context_frames <= 0 or args.context_frames >= args.clip_frames:
    raise ValueError("--context-frames must be inside the clip")
if args.device.startswith("cuda") and not torch.cuda.is_available():
    raise RuntimeError("CUDA was requested but is not available")

from transformers import AutoModel


device = torch.device(args.device)
dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
encoder = AutoModel.from_pretrained(args.model, torch_dtype=dtype).to(device)
encoder.eval()
for parameter in encoder.parameters():
    parameter.requires_grad_(False)

loader = DataLoader(
    dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.num_workers,
    pin_memory=device.type == "cuda",
)

with torch.no_grad():
    probe_video = sample["video"].unsqueeze(0).to(device=device, dtype=dtype)
    probe_tokens = extract_tokens(encoder(pixel_values_videos=probe_video))
    latent_dim = probe_tokens.shape[-1]

predictor = ActionConditionedPredictor(latent_dim=latent_dim).to(device)
optimizer = torch.optim.AdamW(
    predictor.parameters(),
    lr=args.learning_rate,
    weight_decay=0.04,
)
loss_function = nn.SmoothL1Loss()
args.output_dir.mkdir(parents=True, exist_ok=True)

for epoch in range(args.epochs):
    predictor.train()
    running_loss = 0.0
    for batch in loader:
        video = batch["video"].to(device=device, dtype=dtype, non_blocking=True)
        actions = batch["actions"].to(device=device, non_blocking=True)
        context_video = video[:, :, : args.context_frames]
        target_video = video

        with torch.no_grad():
            context_tokens = extract_tokens(
                encoder(pixel_values_videos=context_video)
            )
            target_tokens = extract_tokens(encoder(pixel_values_videos=target_video))
            context_latent = context_tokens.mean(dim=1).float()
            target_latent = target_tokens.mean(dim=1).float()

        predicted_latent = predictor(
            context_latent,
            actions[:, args.context_frames - 1 :],
        )
        loss = loss_function(predicted_latent, target_latent)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), max_norm=1.0)
        optimizer.step()
        running_loss += loss.item()

    epoch_loss = running_loss / max(len(loader), 1)
    print(f"epoch={epoch + 1:03d} loss={epoch_loss:.6f}", flush=True)
    torch.save(
        {
            "epoch": epoch + 1,
            "model_name": args.model,
            "latent_dim": latent_dim,
            "predictor": predictor.state_dict(),
            "optimizer": optimizer.state_dict(),
            "loss": epoch_loss,
            "config": vars(args),
        },
        args.output_dir / "latest.pt",
    )
