"""Train a small learned release-velocity policy for basket throws.

The policy maps release pose and basket target geometry to a 3D release
velocity.  Labels are generated from projectile physics over randomized basket
placements, so Isaac Sim can use a learned model instead of a hard-coded throw
velocity.
"""

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


GRAVITY = 9.81


@dataclass
class BallisticThrowConfig:
    samples: int
    epochs: int
    batch_size: int
    learning_rate: float
    seed: int
    output: str


class BallisticThrowPolicy(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=96):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, features):
        return self.net(features)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/smolvla_multitask_dagger/ballistic_throw_policy.pt"),
    )
    parser.add_argument("--samples", type=int, default=24000)
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def feature_tensor(release_positions, basket_centers):
    delta = basket_centers - release_positions
    horizontal_distance = torch.linalg.norm(delta[:, :2], dim=1, keepdim=True)
    return torch.cat(
        [
            release_positions,
            basket_centers,
            delta,
            horizontal_distance,
        ],
        dim=1,
    )


def distance_based_flight_time(horizontal_distance):
    return torch.clamp(0.22 + 0.10 * horizontal_distance, 0.24, 0.32)


def analytic_release_velocity(release_positions, basket_centers):
    delta = basket_centers - release_positions
    horizontal_distance = torch.linalg.norm(delta[:, :2], dim=1, keepdim=True)
    flight_time = distance_based_flight_time(horizontal_distance)
    velocity_xy = delta[:, :2] / flight_time
    velocity_z = (
        delta[:, 2:3] + 0.5 * GRAVITY * flight_time * flight_time
    ) / flight_time
    return torch.cat([velocity_xy, velocity_z], dim=1)


def make_dataset(samples, device):
    release_positions = torch.empty(samples, 3, device=device)
    release_positions[:, 0] = torch.empty(samples, device=device).uniform_(0.10, 0.26)
    release_positions[:, 1] = torch.empty(samples, device=device).uniform_(-0.58, -0.38)
    release_positions[:, 2] = torch.empty(samples, device=device).uniform_(0.18, 0.32)

    basket_centers = torch.empty(samples, 3, device=device)
    basket_centers[:, 0] = torch.empty(samples, device=device).uniform_(0.42, 0.78)
    basket_centers[:, 1] = torch.empty(samples, device=device).uniform_(-0.82, -0.42)
    basket_centers[:, 2] = torch.empty(samples, device=device).uniform_(0.07, 0.12)

    features = feature_tensor(release_positions, basket_centers)
    targets = analytic_release_velocity(release_positions, basket_centers)
    return features, targets


def load_ballistic_policy(path, device="cpu"):
    checkpoint = torch.load(Path(path), map_location=device, weights_only=False)
    input_mean = checkpoint["input_mean"].to(device)
    input_std = checkpoint["input_std"].to(device)
    output_mean = checkpoint["output_mean"].to(device)
    output_std = checkpoint["output_std"].to(device)
    model = BallisticThrowPolicy(
        int(checkpoint["input_dim"]),
        int(checkpoint["hidden_dim"]),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, input_mean, input_std, output_mean, output_std


def predict_release_velocity(model_bundle, release_position, basket_center):
    model, input_mean, input_std, output_mean, output_std = model_bundle
    device = input_mean.device
    release = torch.as_tensor(release_position, dtype=torch.float32, device=device).view(1, 3)
    target = torch.as_tensor(basket_center, dtype=torch.float32, device=device).view(1, 3)
    features = feature_tensor(release, target)
    with torch.no_grad():
        prediction = model((features - input_mean) / input_std)
    velocity = prediction * output_std + output_mean
    return velocity[0].detach().cpu().numpy()


def main():
    args = parse_args()
    torch.manual_seed(int(args.seed))
    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    )
    features, targets = make_dataset(int(args.samples), device)
    input_mean = features.mean(dim=0, keepdim=True)
    input_std = features.std(dim=0, keepdim=True).clamp_min(1e-6)
    output_mean = targets.mean(dim=0, keepdim=True)
    output_std = targets.std(dim=0, keepdim=True).clamp_min(1e-6)
    features_n = (features - input_mean) / input_std
    targets_n = (targets - output_mean) / output_std

    model = BallisticThrowPolicy(features.shape[1], 96).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.learning_rate))
    indices = torch.arange(features.shape[0], device=device)
    for epoch in range(1, int(args.epochs) + 1):
        permutation = indices[torch.randperm(indices.numel(), device=device)]
        total_loss = 0.0
        batches = 0
        for start in range(0, permutation.numel(), int(args.batch_size)):
            batch_ids = permutation[start : start + int(args.batch_size)]
            prediction = model(features_n[batch_ids])
            loss = F.smooth_l1_loss(prediction, targets_n[batch_ids])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu())
            batches += 1
        if epoch == 1 or epoch % 20 == 0 or epoch == int(args.epochs):
            with torch.no_grad():
                velocity_error = (
                    model(features_n[:2048]) * output_std
                    + output_mean
                    - targets[:2048]
                ).abs().mean(dim=0)
            print(
                f"epoch={epoch} loss={total_loss / max(batches, 1):.6f} "
                f"mean_abs_velocity_error={velocity_error.cpu().tolist()}",
                flush=True,
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    config = BallisticThrowConfig(
        samples=int(args.samples),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        learning_rate=float(args.learning_rate),
        seed=int(args.seed),
        output=str(args.output),
    )
    torch.save(
        {
            "model": model.state_dict(),
            "input_dim": int(features.shape[1]),
            "hidden_dim": 96,
            "input_mean": input_mean.cpu(),
            "input_std": input_std.cpu(),
            "output_mean": output_mean.cpu(),
            "output_std": output_std.cpu(),
            "config": asdict(config),
        },
        args.output,
    )
    print(f"saved_ballistic_throw_policy={args.output}", flush=True)


if __name__ == "__main__":
    main()
