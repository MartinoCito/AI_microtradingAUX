from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback


class PPOTrainingMetricsCallback(BaseCallback):
    """Persist training action/reward diagnostics without changing the environment."""

    def __init__(self, output_path: str | Path, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.output_path = Path(output_path)
        self.action_counts = np.zeros(3, dtype=np.int64)
        self.episode_rewards: list[float] = []

    def _on_training_start(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text("", encoding="utf-8")

    def _on_step(self) -> bool:
        actions = self.locals.get("actions")
        if actions is not None:
            values = np.asarray(actions).reshape(-1)
            for action in values:
                action_int = int(action)
                if 0 <= action_int <= 2:
                    self.action_counts[action_int] += 1

        infos = self.locals.get("infos") or []
        for info in infos:
            episode = info.get("episode") if isinstance(info, dict) else None
            if episode and "r" in episode:
                self.episode_rewards.append(float(episode["r"]))

        return True

    def _on_training_end(self) -> None:
        total = int(self.action_counts.sum())
        distribution = (
            (self.action_counts / total).tolist() if total else [0.0, 0.0, 0.0]
        )
        payload: dict[str, Any] = {
            "timesteps": int(self.num_timesteps),
            "action_counts": {
                "short": int(self.action_counts[0]),
                "flat": int(self.action_counts[1]),
                "long": int(self.action_counts[2]),
            },
            "action_distribution": {
                "short": distribution[0],
                "flat": distribution[1],
                "long": distribution[2],
            },
            "degenerate": bool(max(distribution, default=0.0) >= 0.99),
            "training_episode_rewards": {
                "count": len(self.episode_rewards),
                "mean": float(np.mean(self.episode_rewards))
                if self.episode_rewards
                else None,
                "last": float(self.episode_rewards[-1])
                if self.episode_rewards
                else None,
            },
        }
        self.output_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


def build_ppo_callbacks(
    *,
    output_dir: str | Path,
    checkpoint_freq: int,
    eval_callback: Any,
) -> CallbackList:
    """Build the checkpoint + diagnostic callback stack used by PPO."""
    output_dir = Path(output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_callback = CheckpointCallback(
        save_freq=max(1, int(checkpoint_freq)),
        save_path=str(checkpoint_dir),
        name_prefix="ppo",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )
    metrics_callback = PPOTrainingMetricsCallback(
        output_path=output_dir / "training_action_metrics.json",
    )
    return CallbackList([metrics_callback, checkpoint_callback, eval_callback])
