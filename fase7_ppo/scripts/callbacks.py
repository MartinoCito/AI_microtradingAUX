from __future__ import annotations

import json
from pathlib import Path
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class PPOTrainingMonitor(BaseCallback):
    """Checkpoint, reward training e distribuzione delle azioni."""

    def __init__(self, output_path, checkpoint_dir, checkpoint_freq=25000, verbose=0):
        super().__init__(verbose)
        self.output_path = Path(output_path)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_freq = int(checkpoint_freq)
        self.action_counts = np.zeros(3, dtype=np.int64)
        self.episode_rewards = []
        self._episode_reward = 0.0
        self._last_checkpoint = 0

    def _on_training_start(self):
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _on_step(self):
        actions = self.locals.get("actions")
        if actions is not None:
            for a in np.asarray(actions).reshape(-1):
                a = int(a)
                if 0 <= a <= 2:
                    self.action_counts[a] += 1

        rewards = self.locals.get("rewards")
        if rewards is not None:
            self._episode_reward += float(np.asarray(rewards).reshape(-1)[0])

        dones = self.locals.get("dones")
        ended = dones is not None and bool(np.asarray(dones).reshape(-1)[0])
        if ended:
            self.episode_rewards.append(self._episode_reward)
            self._episode_reward = 0.0

        row = {"timesteps": int(self.num_timesteps)}
        if ended:
            row["episode_reward"] = float(self.episode_rewards[-1])
        total = int(self.action_counts.sum())
        if total:
            row.update({
                "action_short": float(self.action_counts[0] / total),
                "action_flat": float(self.action_counts[1] / total),
                "action_long": float(self.action_counts[2] / total),
            })
        with self.output_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

        if self.num_timesteps - self._last_checkpoint >= self.checkpoint_freq:
            self.model.save(str(self.checkpoint_dir / f"ppo_{self.num_timesteps}_steps"))
            self._last_checkpoint = self.num_timesteps
        return True

    def _on_training_end(self):
        rewards = np.asarray(self.episode_rewards, dtype=float)
        summary = {
            "episodes_completed": len(rewards),
            "mean_episode_reward": float(rewards.mean()) if len(rewards) else None,
            "median_episode_reward": float(np.median(rewards)) if len(rewards) else None,
            "min_episode_reward": float(rewards.min()) if len(rewards) else None,
            "max_episode_reward": float(rewards.max()) if len(rewards) else None,
            "last_episode_reward": float(rewards[-1]) if len(rewards) else None,
        }
        (self.output_path.parent / "training_reward_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        total = int(self.action_counts.sum())
        action_summary = {
            "total_actions": total,
            "action_counts": {
                "short": int(self.action_counts[0]),
                "flat": int(self.action_counts[1]),
                "long": int(self.action_counts[2]),
            },
            "action_distribution": {
                "short": float(self.action_counts[0]/total) if total else 0.0,
                "flat": float(self.action_counts[1]/total) if total else 0.0,
                "long": float(self.action_counts[2]/total) if total else 0.0,
            },
        }
        (self.output_path.parent / "training_action_distribution.json").write_text(
            json.dumps(action_summary, indent=2), encoding="utf-8"
        )
