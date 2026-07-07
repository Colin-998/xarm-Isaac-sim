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
    dagger_loss_weight: float
    close_contact_loss_weight: float
    phase_loss_weight: float
    train_temporal: bool
    train_phase_head: bool
    device: str
    local_files_only: bool
    state_conditioned: bool
    state_stats_source: str
    action_representation: str
    reset_action_head: bool
    freeze_policy_backbone: bool
    delta_arm_target_clip: float
    delta_tcp_target_clip: float
    safety_delta_tcp_target_clip: float
    delta_gripper_target_clip: float
    target_lookahead_steps: int
    safety_tail_steps: int
    clearance_margin: float
    clearance_danger_loss_weight: float
    clearance_danger_margin: float
    tcp_clearance_loss_weight: float
    approach_tcp_loss_weight: float
    clearance_away_loss_weight: float
    clearance_away_min_delta: float
    phase_progress_z_offset: float
    phase_progress_hover_height: float
    phase_progress_outward_offset: float
    safety_recovery_lift: float
    safety_recovery_away: float


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
    parser.add_argument(
        "--include-success-dagger",
        action="store_true",
        help=(
            "Also load successful teacher-corrected rollouts from dagger roots "
            "so failed direct-control corrections are balanced by good behavior."
        ),
    )
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--key-phase-repeat", type=int, default=4)
    parser.add_argument("--action-loss-weight", type=float, default=1.5)
    parser.add_argument(
        "--dagger-loss-weight",
        type=float,
        default=1.5,
        help="Extra action-loss weight for failed direct-control DAgger samples.",
    )
    parser.add_argument(
        "--close-contact-loss-weight",
        type=float,
        default=2.0,
        help="Extra action-loss weight near the cube where contact mistakes matter most.",
    )
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
    parser.add_argument(
        "--recompute-state-stats",
        action="store_true",
        help=(
            "Recompute state normalization from the augmented dataset. "
            "By default a state-conditioned fine-tune preserves the base "
            "checkpoint statistics so the inherited state encoder keeps "
            "the same coordinate scale."
        ),
    )
    parser.add_argument(
        "--action-representation",
        choices=["absolute", "delta", "tcp_delta", "tcp_delta_posture"],
        default="absolute",
        help=(
            "Train absolute joint targets or residual actions relative to "
            "the observed arm/gripper state. tcp_delta trains Cartesian "
            "end-effector residuals in the first three action slots. "
            "tcp_delta_posture additionally trains joint2-4 residuals in "
            "slots four to six so the policy can bias the arm links away "
            "from obstacles while tracking the TCP target."
        ),
    )
    parser.add_argument(
        "--reset-action-head",
        action="store_true",
        help=(
            "Zero-initialize the action head before fine-tuning. This is "
            "recommended when changing from absolute to delta actions."
        ),
    )
    parser.add_argument(
        "--freeze-policy-backbone",
        action="store_true",
        help=(
            "Freeze visual, state, fusion, and phase modules so DAgger only "
            "fits the action head on the established representation."
        ),
    )
    parser.add_argument("--delta-arm-target-clip", type=float, default=0.02)
    parser.add_argument("--delta-tcp-target-clip", type=float, default=0.04)
    parser.add_argument(
        "--safety-delta-tcp-target-clip",
        type=float,
        default=0.0,
        help=(
            "Optional larger TCP delta target clip for DAgger safety-tail "
            "samples. A value <= 0 keeps --delta-tcp-target-clip behavior."
        ),
    )
    parser.add_argument("--delta-gripper-target-clip", type=float, default=0.15)
    parser.add_argument(
        "--target-lookahead-steps",
        type=int,
        default=0,
        help=(
            "Train the final action head against a future step in the same "
            "phase. This makes direct brain control learn a direction over "
            "several frames instead of only imitating a tiny local waypoint."
        ),
    )
    parser.add_argument(
        "--safety-tail-steps",
        type=int,
        default=12,
        help=(
            "For failed DAgger episodes, mark the last N captured samples as "
            "near-violation samples so the policy learns to avoid unsafe "
            "approaches before the monitor trips."
        ),
    )
    parser.add_argument(
        "--clearance-margin",
        type=float,
        default=0.004,
        help=(
            "Additional obstacle-clearance margin in meters used to boost "
            "near-threshold DAgger failures."
        ),
    )
    parser.add_argument(
        "--clearance-danger-loss-weight",
        type=float,
        default=0.0,
        help=(
            "Extra action-loss weight for samples whose recorded link/cube "
            "clearance margin is within --clearance-danger-margin. This "
            "turns near-collision frames into stronger DAgger corrections."
        ),
    )
    parser.add_argument(
        "--clearance-danger-margin",
        type=float,
        default=0.020,
        help=(
            "Clearance margin in meters below which a sample is treated as "
            "dangerous for step-level action weighting."
        ),
    )
    parser.add_argument(
        "--tcp-clearance-loss-weight",
        type=float,
        default=0.0,
        help=(
            "Extra loss on tcp_delta target dimensions for DAgger samples "
            "whose link_tcp is near or over obstacle clearance. The runtime "
            "uses the final action window at indices 7:10."
        ),
    )
    parser.add_argument(
        "--approach-tcp-loss-weight",
        type=float,
        default=0.0,
        help=(
            "Extra loss on tcp_delta target dimensions during approach_cube. "
            "This teaches the direct brain policy to follow the local teacher "
            "waypoint earlier instead of only reacting after a clearance stop."
        ),
    )
    parser.add_argument(
        "--clearance-away-loss-weight",
        type=float,
        default=0.0,
        help=(
            "Extra loss that requires tcp_delta actions near obstacles to "
            "project positively along the recorded clearance away_vector. "
            "This teaches the brain to create clearance itself instead of "
            "waiting for the monitor to stop."
        ),
    )
    parser.add_argument(
        "--clearance-away-min-delta",
        type=float,
        default=0.030,
        help=(
            "Minimum raw tcp_delta projection, in meters, required along "
            "away_vector for dangerous obstacle samples when "
            "--clearance-away-loss-weight is enabled."
        ),
    )
    parser.add_argument(
        "--phase-progress-z-offset",
        type=float,
        default=0.010,
        help=(
            "Z offset above the detected cube center used when converting "
            "phase-hold failures into direct cube-centered DAgger targets."
        ),
    )
    parser.add_argument(
        "--phase-progress-hover-height",
        type=float,
        default=0.075,
        help=(
            "Extra hover height used for approach_cube phase-progress "
            "corrections. This matches the vertical grasp habit used in "
            "the Isaac Sim controller."
        ),
    )
    parser.add_argument(
        "--phase-progress-outward-offset",
        type=float,
        default=0.015,
        help=(
            "Outward XY offset from the robot base used for phase-progress "
            "cube-centered corrections."
        ),
    )
    parser.add_argument(
        "--safety-recovery-lift",
        type=float,
        default=0.08,
        help=(
            "For safety-tail DAgger samples, replace the TCP teacher target "
            "with a recovery target this many meters above the observed TCP."
        ),
    )
    parser.add_argument(
        "--safety-recovery-away",
        type=float,
        default=0.04,
        help=(
            "For obstacle safety-tail samples, move the recovery TCP target "
            "this many meters away from the nearest obstacle in XY."
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def flatten_raw_state(observation):
    gripper = observation["gripper_joint_positions"]
    clearance = observation.get("clearance") or {}
    clearance_m = float(clearance.get("clearance_m", 1.0))
    threshold_m = float(clearance.get("threshold_m", 0.012))
    clearance_margin_m = float(
        clearance.get("clearance_margin_m", clearance_m - threshold_m)
    )
    clearance_kind = 1.0 if clearance.get("kind") == "cube" else 0.0
    away_vector = clearance.get("away_vector", [0.0, 0.0, 0.0])
    if len(away_vector) != 3:
        away_vector = [0.0, 0.0, 0.0]
    away_vector = [float(value) for value in away_vector]
    return (
        observation["arm_joint_positions"]
        + [gripper["drive_joint"]]
        + observation["tcp_position"]
        + observation["cube_position"]
        + [
            clearance_m,
            threshold_m,
            clearance_margin_m,
            clearance_kind,
            *away_vector,
        ]
    )


DEFAULT_CLEARANCE_STATE = [1.0, 0.012, 0.988, 0.0, 0.0, 0.0, 0.0]


def pad_state_features(episodes):
    max_dim = 0
    for episode in episodes:
        for step in episode["steps"]:
            state = step["observation"].get("state", [])
            max_dim = max(max_dim, len(state))
    max_dim = max(max_dim, 13 + len(DEFAULT_CLEARANCE_STATE))
    for episode in episodes:
        for step in episode["steps"]:
            state = list(step["observation"].get("state", []))
            if len(state) == 13 and max_dim >= 17:
                state.extend(DEFAULT_CLEARANCE_STATE)
            elif len(state) < max_dim:
                state.extend([0.0] * (max_dim - len(state)))
            step["observation"]["state"] = state
    return max_dim


def safety_recovery_tcp_target(
    observation,
    lift=0.08,
    away=0.04,
):
    tcp = observation.get("tcp_position")
    if tcp is None:
        return None
    target = [float(value) for value in tcp[:3]]
    target[2] += float(lift)

    clearance = observation.get("clearance") or {}
    away_vector = clearance.get("away_vector")
    if away_vector is None:
        state = observation.get("state") or []
        if len(state) >= 20:
            away_vector = state[17:20]
    if away_vector is not None and away > 0.0:
        direction = [float(value) for value in away_vector[:3]]
        norm = max(sum(value * value for value in direction) ** 0.5, 1e-6)
        if norm > 1e-5:
            for axis in range(3):
                target[axis] += direction[axis] / norm * float(away)
            return target

    obstacles = observation.get("obstacles") or []
    if obstacles and away > 0.0:
        tcp_xy = target[:2]
        nearest = None
        nearest_distance_sq = None
        for obstacle in obstacles:
            position = obstacle.get("position")
            if position is None or len(position) < 2:
                continue
            dx = tcp_xy[0] - float(position[0])
            dy = tcp_xy[1] - float(position[1])
            distance_sq = dx * dx + dy * dy
            if nearest_distance_sq is None or distance_sq < nearest_distance_sq:
                nearest = (dx, dy)
                nearest_distance_sq = distance_sq
        if nearest is not None:
            dx, dy = nearest
            norm = max((dx * dx + dy * dy) ** 0.5, 1e-6)
            target[0] += dx / norm * float(away)
            target[1] += dy / norm * float(away)
    return target


def phase_progress_tcp_target(
    observation,
    phase,
    z_offset=0.010,
    hover_height=0.075,
    outward_offset=0.015,
):
    cube_position = observation.get("cube_position")
    if cube_position is None or len(cube_position) < 3:
        return None
    target = [float(value) for value in cube_position[:3]]
    target[2] += float(z_offset)

    outward = float(outward_offset)
    if abs(outward) > 1e-9:
        norm = (target[0] * target[0] + target[1] * target[1]) ** 0.5
        if norm > 1e-6:
            target[0] += target[0] / norm * outward
            target[1] += target[1] / norm * outward

    if phase == "approach_cube":
        target[2] += float(hover_height)
    return target


def load_dagger_episodes(
    roots,
    max_episodes=0,
    repeat=1,
    include_success=False,
    safety_tail_steps=12,
    clearance_margin=0.004,
    safety_recovery_lift=0.08,
    safety_recovery_away=0.04,
    phase_progress_z_offset=0.010,
    phase_progress_hover_height=0.075,
    phase_progress_outward_offset=0.015,
):
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
            episode_success = bool(metadata.get("success", False))
            if episode_success and not include_success:
                continue
            metrics = metadata.get("metrics", {})
            failure_text = str(
                metrics.get("payload_failure") or metadata.get("error") or ""
            )
            violations = metrics.get("link_clearance_violations") or []
            min_obstacle_clearance = metrics.get(
                "minimum_robot_obstacle_clearance_m"
            )
            min_cube_clearance = metrics.get("minimum_robot_cube_clearance_m")
            clearance_threshold = None
            if violations:
                clearance_threshold = violations[-1].get("threshold_m")
            safety_weight = 1.0
            if failure_text:
                safety_weight += 1.5
            if "grasp_readiness" in failure_text:
                safety_weight += 2.0
            if "brain_phase_hold" in failure_text:
                safety_weight += 2.0
            if "phase mismatch" in failure_text:
                safety_weight += 2.5
            if violations:
                safety_weight += 3.0
            if (
                min_obstacle_clearance is not None
                and clearance_threshold is not None
                and float(min_obstacle_clearance)
                < float(clearance_threshold) + float(clearance_margin)
            ):
                safety_weight += 2.5
            if (
                min_cube_clearance is not None
                and float(min_cube_clearance) < 0.008
            ):
                safety_weight += 2.0
            steps = []
            with actions_file.open("r", encoding="utf-8") as stream:
                for line in stream:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    action = row["action"]["arm_joint_positions"] + [
                        row["action"]["gripper_joint_position"]
                    ]
                    action_tcp_position = row["action"].get("tcp_position")
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
                                "clearance": observation.get("clearance", {}),
                                "phase": row["phase"],
                                "time_seconds": row["time_seconds"],
                            },
                            "action": action,
                            "action_tcp_position": action_tcp_position,
                            "executed_action": row.get("executed_action"),
                            "dagger": row.get("dagger", {}),
                            "reward": 0.0,
                            "discount": 1.0,
                            "is_first": len(steps) == 0,
                            "is_last": False,
                            "is_terminal": False,
                        }
                    )
            if len(steps) < 2:
                continue
            if (violations or failure_text) and safety_tail_steps > 0:
                phase_hold_failure = (
                    not violations
                    and "brain_phase_hold_rejected" in failure_text
                )
                tail_violation = (
                    violations[-1]
                    if violations
                    else {
                        "kind": (
                            "phase_progress_failure"
                            if phase_hold_failure
                            else "episode_failure"
                        ),
                        "reason": failure_text,
                    }
                )
                tail_start = max(0, len(steps) - int(safety_tail_steps))
                for tail_index, step in enumerate(steps[tail_start:]):
                    dagger = step.setdefault("dagger", {})
                    if dagger.get("safety_violation") is None:
                        dagger["safety_violation"] = {
                            **tail_violation,
                            "kind": (
                                tail_violation.get("kind")
                                or "near_clearance_violation"
                            ),
                            "near_violation_tail": True,
                            "tail_step_index": tail_index,
                            "tail_total_steps": len(steps) - tail_start,
                        }
                    if phase_hold_failure:
                        progress_tcp = phase_progress_tcp_target(
                            step["observation"],
                            step["observation"].get("phase"),
                            z_offset=phase_progress_z_offset,
                            hover_height=phase_progress_hover_height,
                            outward_offset=phase_progress_outward_offset,
                        )
                        if progress_tcp is not None:
                            step["action_tcp_position"] = progress_tcp
                            step.setdefault("dagger", {})[
                                "phase_progress_tcp_position"
                            ] = progress_tcp
                            step.setdefault("dagger", {})[
                                "phase_progress_source"
                            ] = "cube_centered_hover"
                        else:
                            step.setdefault("dagger", {})[
                                "phase_progress_tcp_position"
                            ] = step.get("action_tcp_position")
                    else:
                        recovery_tcp = safety_recovery_tcp_target(
                            step["observation"],
                            lift=safety_recovery_lift,
                            away=safety_recovery_away,
                        )
                        if recovery_tcp is not None:
                            step["action_tcp_position"] = recovery_tcp
                            step.setdefault("dagger", {})[
                                "safety_recovery_tcp_position"
                            ] = recovery_tcp
                            step["action"][6] = 0.0
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
                            "source": (
                                "dagger_success_direct"
                                if episode_success
                                else "dagger_failed_direct"
                            ),
                            "repeat_index": repeat_index,
                            "safety_weight": safety_weight,
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
        action_representation="absolute",
        delta_arm_target_clip=0.02,
        delta_tcp_target_clip=0.04,
        safety_delta_tcp_target_clip=0.0,
        delta_gripper_target_clip=0.15,
        target_lookahead_steps=0,
        clearance_danger_margin=0.020,
    ):
        self.root = root
        self.episodes = episodes
        self.phase_to_id = phase_to_id
        self.clip_frames = int(clip_frames)
        self.action_representation = action_representation
        self.delta_arm_target_clip = max(float(delta_arm_target_clip), 0.0)
        self.delta_tcp_target_clip = max(float(delta_tcp_target_clip), 0.0)
        self.safety_delta_tcp_target_clip = max(
            float(safety_delta_tcp_target_clip), 0.0
        )
        self.delta_gripper_target_clip = max(
            float(delta_gripper_target_clip), 0.0
        )
        self.target_lookahead_steps = max(int(target_lookahead_steps), 0)
        self.clearance_danger_margin = max(float(clearance_danger_margin), 1e-6)
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

    def _lookahead_index(self, steps, step_index):
        if self.target_lookahead_steps <= 0:
            return step_index
        phase = steps[step_index]["observation"]["phase"]
        target_index = step_index
        stop = min(len(steps), step_index + self.target_lookahead_steps + 1)
        for candidate in range(step_index + 1, stop):
            if steps[candidate]["observation"]["phase"] != phase:
                break
            target_index = candidate
        return target_index

    def _model_action_for(self, steps, source_index, target_index=None):
        source_step = steps[source_index]
        if target_index is None:
            target_index = source_index
        target_step = steps[target_index]
        if self.action_representation in {"tcp_delta", "tcp_delta_posture"}:
            source_observation = source_step["observation"]
            target_observation = target_step["observation"]
            action = torch.zeros(7, dtype=torch.float32)
            source_tcp = torch.tensor(
                source_observation["tcp_position"], dtype=torch.float32
            )
            action_tcp_position = target_step.get("action_tcp_position")
            if action_tcp_position is not None:
                target_tcp = torch.tensor(action_tcp_position, dtype=torch.float32)
            else:
                target_tcp = torch.tensor(
                    target_observation["tcp_position"], dtype=torch.float32
                )
            tcp_clip = self.delta_tcp_target_clip
            if (
                self.safety_delta_tcp_target_clip > 0.0
                and action_tcp_position is not None
                and target_step.get("dagger", {}).get("safety_violation")
            ):
                tcp_clip = max(tcp_clip, self.safety_delta_tcp_target_clip)
            action[:3] = (target_tcp - source_tcp).clamp(-tcp_clip, tcp_clip)
            if self.action_representation == "tcp_delta_posture":
                source_joints = torch.tensor(
                    source_observation["state"][:6],
                    dtype=torch.float32,
                )
                target_joints = torch.tensor(
                    target_step["action"][:6],
                    dtype=torch.float32,
                )
                action[3:6] = (target_joints[1:4] - source_joints[1:4]).clamp(
                    -self.delta_arm_target_clip,
                    self.delta_arm_target_clip,
                )
            source_gripper = torch.tensor(
                source_observation["state"][6], dtype=torch.float32
            )
            target_gripper = torch.tensor(
                target_step["action"][6], dtype=torch.float32
            )
            action[6] = (target_gripper - source_gripper).clamp(
                -self.delta_gripper_target_clip,
                self.delta_gripper_target_clip,
            )
            return action

        action = torch.tensor(target_step["action"], dtype=torch.float32)
        if self.action_representation == "delta":
            observed_joints = torch.tensor(
                source_step["observation"]["state"][:7], dtype=torch.float32
            )
            action = action - observed_joints
            action[:6] = action[:6].clamp(
                -self.delta_arm_target_clip,
                self.delta_arm_target_clip,
            )
            action[6] = action[6].clamp(
                -self.delta_gripper_target_clip,
                self.delta_gripper_target_clip,
            )
        return action

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
            model_action = self._model_action_for(
                steps,
                selected_index,
                self._lookahead_index(steps, selected_index),
            )
            action_window.append(model_action)
        target_index = self._lookahead_index(steps, step_index)
        target_step = steps[target_index]
        target_action = self._model_action_for(steps, step_index, target_index)
        mean_action = torch.stack(action_window).mean(dim=0)
        phase = target_step["observation"]["phase"]
        tcp = torch.tensor(target_step["observation"]["tcp_position"], dtype=torch.float32)
        cube = torch.tensor(target_step["observation"]["cube_position"], dtype=torch.float32)
        tcp_cube_distance = torch.linalg.vector_norm(tcp - cube)
        key_phase = phase in KEY_PHASES
        approach_phase = phase == "approach_cube"
        metadata = episode.get("metadata", {})
        is_dagger = str(metadata.get("source", "")).startswith("dagger_")
        safety_event = bool(
            target_step.get("dagger", {}).get("safety_violation")
        )
        safety_violation = target_step.get("dagger", {}).get("safety_violation") or {}
        violation_link = str(safety_violation.get("link", ""))
        violation_kind = str(safety_violation.get("kind", ""))
        has_clearance_measurement = (
            "clearance_m" in safety_violation
            and "threshold_m" in safety_violation
        )
        clearance_danger = 0.0
        state = target_step["observation"]["state"]
        if len(state) >= 17:
            clearance_margin_m = float(state[15])
            clearance_kind = float(state[16])
            if clearance_kind < 0.5:
                clearance_danger = max(
                    0.0,
                    min(
                        1.0,
                        (self.clearance_danger_margin - clearance_margin_m)
                        / self.clearance_danger_margin,
                    ),
                )
        tcp_clearance_danger = (
            safety_event
            and has_clearance_measurement
            and violation_kind != "cube"
            and not violation_link.startswith("cube")
            and self.action_representation in {"tcp_delta", "tcp_delta_posture"}
        )
        close_contact = (
            tcp_cube_distance < 0.12
            and phase in {"approach_cube", "descend_to_cube", "close_gripper"}
        )
        return {
            "episode_id": episode["episode_id"],
            "frames": torch.stack(frames),
            "state": torch.tensor(target_step["observation"]["state"], dtype=torch.float32),
            "action_target": torch.cat([mean_action, target_action], dim=0),
            "phase_id": torch.tensor(self.phase_to_id[phase], dtype=torch.long),
            "key_phase": torch.tensor(float(key_phase), dtype=torch.float32),
            "approach_phase": torch.tensor(
                float(approach_phase),
                dtype=torch.float32,
            ),
            "dagger_sample": torch.tensor(float(is_dagger), dtype=torch.float32),
            "close_contact": torch.tensor(float(close_contact), dtype=torch.float32),
            "clearance_danger": torch.tensor(
                float(clearance_danger),
                dtype=torch.float32,
            ),
            "tcp_clearance_danger": torch.tensor(
                float(tcp_clearance_danger),
                dtype=torch.float32,
            ),
            "safety_weight": torch.tensor(
                float(metadata.get("safety_weight", 0.0))
                + (8.0 if safety_event else 0.0),
                dtype=torch.float32,
            ),
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
        "approach_phase": torch.stack([item["approach_phase"] for item in batch]),
        "dagger_sample": torch.stack([item["dagger_sample"] for item in batch]),
        "close_contact": torch.stack([item["close_contact"] for item in batch]),
        "clearance_danger": torch.stack([item["clearance_danger"] for item in batch]),
        "tcp_clearance_danger": torch.stack(
            [item["tcp_clearance_danger"] for item in batch]
        ),
        "safety_weight": torch.stack([item["safety_weight"] for item in batch]),
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
    dagger_loss_weight,
    close_contact_loss_weight,
    clearance_danger_loss_weight,
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
        "expert_action_mae": 0.0,
        "expert_final_arm_mae_rad": 0.0,
        "expert_final_gripper_mae": 0.0,
        "approach_action": 0.0,
    }
    count = 0
    key_count = 0
    close_count = 0
    approach_count = 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            frames = batch["frames"].to(device)
            state = batch["state"].to(device)
            phase_id = batch["phase_id"].to(device)
            action_target = batch["action_target"].to(device)
            key_phase = batch["key_phase"].to(device)
            approach_phase = batch["approach_phase"].to(device)
            dagger_sample = batch["dagger_sample"].to(device)
            close_contact = batch["close_contact"].to(device)
            clearance_danger = batch["clearance_danger"].to(device)
            safety_weight = batch["safety_weight"].to(device)
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
            action_abs = torch.abs(action_pred - action_target)
            final_arm_abs = action_abs[:, -7:-1].mean(dim=1)
            final_gripper_abs = action_abs[:, -1]
            sample_weights = (
                1.0
                + action_loss_weight * key_phase
                + dagger_loss_weight * dagger_sample
                + close_contact_loss_weight * close_contact
                + clearance_danger_loss_weight * clearance_danger
                + safety_weight
            )
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
            totals["expert_action_mae"] += float(action_abs.mean(dim=1).sum().detach().cpu())
            totals["expert_final_arm_mae_rad"] += float(final_arm_abs.sum().detach().cpu())
            totals["expert_final_gripper_mae"] += float(final_gripper_abs.sum().detach().cpu())
            key_mask = key_phase > 0.5
            if key_mask.any():
                key_count += int(key_mask.sum().item())
                totals["key_action"] += float(per_action[key_mask].sum().detach().cpu())
            close_mask = tcp_cube_distance < 0.08
            if close_mask.any():
                close_count += int(close_mask.sum().item())
                totals["close_action"] += float(per_action[close_mask].sum().detach().cpu())
            approach_mask = approach_phase > 0.5
            if approach_mask.any():
                approach_count += int(approach_mask.sum().item())
                totals["approach_action"] += float(
                    per_action[approach_mask].sum().detach().cpu()
                )
    return {
        "loss": totals["loss"] / max(1, count),
        "action": totals["action"] / max(1, count),
        "phase": totals["phase"] / max(1, count),
        "accuracy": totals["accuracy"] / max(1, count),
        "key_action": totals["key_action"] / max(1, key_count),
        "close_action": totals["close_action"] / max(1, close_count),
        "approach_action": totals["approach_action"] / max(1, approach_count),
        "expert_action_mae": totals["expert_action_mae"] / max(1, count),
        "expert_final_arm_mae_rad": totals["expert_final_arm_mae_rad"] / max(1, count),
        "expert_final_gripper_mae": totals["expert_final_gripper_mae"] / max(1, count),
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
        args.include_success_dagger,
        args.safety_tail_steps,
        args.clearance_margin,
        args.safety_recovery_lift,
        args.safety_recovery_away,
        args.phase_progress_z_offset,
        args.phase_progress_hover_height,
        args.phase_progress_outward_offset,
    )
    if dagger_episodes:
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
        args.safety_delta_tcp_target_clip,
        args.delta_gripper_target_clip,
        args.target_lookahead_steps,
        args.clearance_danger_margin,
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
        f"clip_frames={clip_frames} image_size={image_size} "
        f"state_dim={state_dim}"
    )
    if args.dry_run:
        print(f"phase_to_id={phase_to_id}")
        return

    sample_episode_index, sample_step_index = dataset.samples[0]
    sample_state_dim = len(
        dataset.episodes[sample_episode_index]["steps"][sample_step_index][
            "observation"
        ]["state"]
    )
    if (
        args.state_conditioned
        and not args.recompute_state_stats
        and "state_mean" in base_checkpoint
        and "state_std" in base_checkpoint
        and len(base_checkpoint["state_mean"]) == sample_state_dim
    ):
        state_mean = torch.tensor(
            base_checkpoint["state_mean"], dtype=torch.float32
        )
        state_std = torch.tensor(
            base_checkpoint["state_std"], dtype=torch.float32
        ).clamp_min(1e-4)
        state_stats_source = "base_checkpoint"
    else:
        state_mean, state_std = compute_state_stats(dataset)
        state_stats_source = "augmented_dataset"
    print(
        f"state_stats_source={state_stats_source} "
        f"state_dim={int(state_mean.numel())}",
        flush=True,
    )
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
        for module_name in ("state_encoder", "fusion"):
            module_state = {
                key.removeprefix(f"{module_name}."): value
                for key, value in base_state.items()
                if key.startswith(f"{module_name}.")
            }
            if not module_state:
                continue
            current_state = getattr(model, module_name).state_dict()
            if all(
                key in current_state and current_state[key].shape == value.shape
                for key, value in module_state.items()
            ):
                getattr(model, module_name).load_state_dict(module_state)
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
    if args.reset_action_head:
        nn.init.zeros_(model.action_head.weight)
        nn.init.zeros_(model.action_head.bias)
    if args.freeze_policy_backbone:
        backbone_names = (
            "visual",
            "state_encoder",
            "fusion",
            "temporal",
            "phase_head",
        )
        for name in backbone_names:
            module = getattr(model, name, None)
            if module is None:
                continue
            for param in module.parameters():
                param.requires_grad = False
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
        dagger_loss_weight=args.dagger_loss_weight,
        close_contact_loss_weight=args.close_contact_loss_weight,
        phase_loss_weight=args.phase_loss_weight,
        train_temporal=args.train_temporal,
        train_phase_head=args.train_phase_head,
        device=str(device),
        local_files_only=args.local_files_only,
        state_conditioned=args.state_conditioned,
        state_stats_source=state_stats_source,
        action_representation=args.action_representation,
        reset_action_head=args.reset_action_head,
        freeze_policy_backbone=args.freeze_policy_backbone,
        delta_arm_target_clip=args.delta_arm_target_clip,
        delta_tcp_target_clip=args.delta_tcp_target_clip,
        safety_delta_tcp_target_clip=args.safety_delta_tcp_target_clip,
        delta_gripper_target_clip=args.delta_gripper_target_clip,
        target_lookahead_steps=args.target_lookahead_steps,
        safety_tail_steps=args.safety_tail_steps,
        clearance_margin=args.clearance_margin,
        clearance_danger_loss_weight=args.clearance_danger_loss_weight,
        clearance_danger_margin=args.clearance_danger_margin,
        tcp_clearance_loss_weight=args.tcp_clearance_loss_weight,
        approach_tcp_loss_weight=args.approach_tcp_loss_weight,
        clearance_away_loss_weight=args.clearance_away_loss_weight,
        clearance_away_min_delta=args.clearance_away_min_delta,
        phase_progress_z_offset=args.phase_progress_z_offset,
        phase_progress_hover_height=args.phase_progress_hover_height,
        phase_progress_outward_offset=args.phase_progress_outward_offset,
        safety_recovery_lift=args.safety_recovery_lift,
        safety_recovery_away=args.safety_recovery_away,
    )
    (args.output_dir / "training_config.json").write_text(
        json.dumps(asdict(config), indent=2),
        encoding="utf-8",
    )
    metrics_path = args.output_dir / "metrics.jsonl"
    total_steps = 0
    best_expert_action = None
    with metrics_path.open("w", encoding="utf-8") as metrics_stream:
        for epoch in range(args.epochs):
            model.train()
            for step, batch in enumerate(train_loader):
                frames = batch["frames"].to(device)
                state = batch["state"].to(device)
                phase_id = batch["phase_id"].to(device)
                action_target = batch["action_target"].to(device)
                key_phase = batch["key_phase"].to(device)
                approach_phase = batch["approach_phase"].to(device)
                dagger_sample = batch["dagger_sample"].to(device)
                close_contact = batch["close_contact"].to(device)
                clearance_danger = batch["clearance_danger"].to(device)
                tcp_clearance_danger = batch["tcp_clearance_danger"].to(device)
                safety_weight = batch["safety_weight"].to(device)
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
                sample_weights = (
                    1.0
                    + args.action_loss_weight * key_phase
                    + args.dagger_loss_weight * dagger_sample
                    + args.close_contact_loss_weight * close_contact
                    + args.clearance_danger_loss_weight * clearance_danger
                    + safety_weight
                )
                action_loss = (per_action * sample_weights).mean()
                if args.tcp_clearance_loss_weight > 0.0:
                    final_tcp_clearance_loss = F.smooth_l1_loss(
                        action_pred[:, 7:10],
                        action_target[:, 7:10],
                        reduction="none",
                    ).mean(dim=1)
                    mean_tcp_clearance_loss = F.smooth_l1_loss(
                        action_pred[:, :3],
                        action_target[:, :3],
                        reduction="none",
                    ).mean(dim=1)
                    tcp_clearance_loss = (
                        final_tcp_clearance_loss
                        + 0.25 * mean_tcp_clearance_loss
                    )
                    action_loss = action_loss + args.tcp_clearance_loss_weight * (
                        tcp_clearance_loss * tcp_clearance_danger
                    ).mean()
                if args.approach_tcp_loss_weight > 0.0:
                    final_approach_tcp_loss = F.smooth_l1_loss(
                        action_pred[:, 7:10],
                        action_target[:, 7:10],
                        reduction="none",
                    ).mean(dim=1)
                    mean_approach_tcp_loss = F.smooth_l1_loss(
                        action_pred[:, :3],
                        action_target[:, :3],
                        reduction="none",
                    ).mean(dim=1)
                    approach_tcp_loss = (
                        final_approach_tcp_loss
                        + 0.25 * mean_approach_tcp_loss
                    )
                    action_loss = action_loss + args.approach_tcp_loss_weight * (
                        approach_tcp_loss * approach_phase
                    ).mean()
                if args.clearance_away_loss_weight > 0.0:
                    away_vector = state[:, 17:20]
                    away_norm = torch.linalg.vector_norm(
                        away_vector,
                        dim=1,
                        keepdim=True,
                    ).clamp_min(1e-6)
                    away_unit = away_vector / away_norm
                    predicted_projection = (
                        action_pred[:, 7:10] * away_unit
                    ).sum(dim=1)
                    target_projection = (
                        action_target[:, 7:10] * away_unit
                    ).sum(dim=1)
                    required_projection = torch.maximum(
                        target_projection,
                        torch.full_like(
                            target_projection,
                            float(args.clearance_away_min_delta),
                        ),
                    )
                    away_valid = (away_norm.squeeze(1) > 1e-5).float()
                    away_deficit = F.relu(
                        required_projection - predicted_projection
                    )
                    action_loss = action_loss + args.clearance_away_loss_weight * (
                        away_deficit.square()
                        * clearance_danger
                        * away_valid
                    ).mean()
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
                args.dagger_loss_weight,
                args.close_contact_loss_weight,
                args.clearance_danger_loss_weight,
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
                "action_representation": args.action_representation,
                "target_lookahead_steps": args.target_lookahead_steps,
                "delta_tcp_target_clip": args.delta_tcp_target_clip,
                "safety_delta_tcp_target_clip": args.safety_delta_tcp_target_clip,
            }
            torch.save(checkpoint, args.output_dir / "latest_stage3_policy.pt")
            if (
                best_expert_action is None
                or row["expert_action_mae"] < best_expert_action
            ):
                best_expert_action = row["expert_action_mae"]
                torch.save(checkpoint, args.output_dir / "best_stage3_policy.pt")
                (args.output_dir / "best_metrics.json").write_text(
                    json.dumps(row, indent=2),
                    encoding="utf-8",
                )
            print(
                f"saved_direct_correction epoch={epoch + 1} "
                f"checkpoint={args.output_dir / 'latest_stage3_policy.pt'} "
                f"val_action={row['action']:.4f} "
                f"val_close_action={row['close_action']:.4f} "
                f"expert_mae={row['expert_action_mae']:.4f} "
                f"expert_arm_rad={row['expert_final_arm_mae_rad']:.4f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
