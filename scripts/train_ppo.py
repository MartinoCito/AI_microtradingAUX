from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gold_rl.data.features import (  # noqa: E402
    MARKET_FEATURE_COLUMNS,
    apply_normalization,
    create_market_features,
    fit_normalization_stats,
)
from gold_rl.execution import ExecutionCostsConfig  # noqa: E402
from gold_rl.rl.callbacks import build_ppo_callbacks  # noqa: E402
from gold_rl.trading_env import TradingEnv  # noqa: E402

ACTION_NAMES = {0: "short", 1: "flat", 2: "long"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fase 7 - addestramento PPO preliminare"
    )
    parser.add_argument("--config", type=Path, default=ROOT / "config.yaml")
    parser.add_argument("--m15-path", type=Path, default=None)
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--train-fraction", type=float, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--eval-freq", type=int, default=None)
    parser.add_argument("--checkpoint-freq", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def prepare_data(
    config: dict[str, Any],
    m15_path: Path | None = None,
):
    data_cfg = config["data"]
    features_cfg = config["features"]
    path = m15_path or ROOT / data_cfg.get(
        "m15_path",
        "data/processed/xauusd_m15.parquet",
    )
    if not path.exists():
        raise FileNotFoundError(
            f"Manca il prodotto M15 della Fase 2: {path}. "
            "Eseguire prima la pipeline dati con il dataset locale definitivo."
        )

    ohlcv = pd.read_parquet(path)
    if not isinstance(ohlcv.index, pd.DatetimeIndex):
        raise TypeError("Il prodotto M15 deve avere un DatetimeIndex.")
    if not ohlcv.index.is_monotonic_increasing or ohlcv.index.has_duplicates:
        raise ValueError(
            "Il prodotto M15 deve essere ordinato e privo di timestamp duplicati."
        )

    features = create_market_features(
        ohlcv,
        drop_warmup=True,
        strict_windows=True,
    ).loc[:, MARKET_FEATURE_COLUMNS]
    prices = ohlcv.loc[features.index, "Close"].astype("float64")

    if not np.isfinite(features.to_numpy()).all():
        raise ValueError(
            "Le feature M15 contengono valori non finiti dopo il warm-up."
        )
    if not np.isfinite(prices.to_numpy()).all() or (prices <= 0).any():
        raise ValueError(
            "I prezzi M15 contengono valori non finiti o non positivi."
        )

    train_ratio = float(data_cfg["train_ratio"])
    validation_ratio = float(data_cfg["validation_ratio"])
    test_ratio = float(data_cfg["test_ratio"])
    if not np.isclose(train_ratio + validation_ratio + test_ratio, 1.0):
        raise ValueError("train_ratio + validation_ratio + test_ratio deve essere 1.0.")

    n = len(features)
    train_end = int(n * train_ratio)
    validation_end = train_end + int(n * validation_ratio)
    train_features = features.iloc[:train_end].copy()
    validation_features = features.iloc[train_end:validation_end].copy()

    train_prices = prices.loc[train_features.index].copy()
    validation_prices = prices.loc[validation_features.index].copy()

    if features_cfg.get("normalize", True):
        stats = fit_normalization_stats(
            train_features=train_features,
            feature_columns=MARKET_FEATURE_COLUMNS,
        )
        train_features = apply_normalization(train_features, stats)
        validation_features = apply_normalization(validation_features, stats)

    return (
        train_features,
        validation_features,
        train_prices,
        validation_prices,
    )


def build_env_factory(
    features: pd.DataFrame,
    prices: pd.Series,
    environment_cfg: dict[str, Any],
    mode: str,
    seed: int,
):
    costs = ExecutionCostsConfig(
        spread_rate=float(environment_cfg["spread_rate"]),
        commission_rate=float(environment_cfg["commission_rate"]),
        slippage_rate=float(environment_cfg["slippage_rate"]),
        turnover_penalty=float(environment_cfg["turnover_penalty"]),
    )

    def factory():
        env = TradingEnv(
            features=features,
            mid_prices=prices,
            execution_config=costs,
            initial_balance=float(environment_cfg["initial_balance"]),
            mode=mode,
            max_episode_steps=(
                int(environment_cfg["max_episode_steps"])
                if mode == "train"
                else None
            ),
            random_start=(mode == "train"),
            turnover_reward_weight=0.0,
        )
        env.reset(seed=seed)
        return Monitor(env)

    return factory


def evaluate_deterministic(
    model: PPO,
    env: TradingEnv,
) -> dict[str, Any]:
    obs, info = env.reset(seed=0)
    action_counts = np.zeros(3, dtype=np.int64)
    total_reward = 0.0
    steps = 0

    while True:
        action, _ = model.predict(obs, deterministic=True)
        action_int = int(np.asarray(action).reshape(-1)[0])
        action_counts[action_int] += 1
        obs, reward, terminated, truncated, info = env.step(action_int)
        total_reward += float(reward)
        steps += 1
        if terminated or truncated:
            break

    total_actions = int(action_counts.sum())
    distribution = (
        action_counts / total_actions
        if total_actions
        else np.zeros(3, dtype=float)
    )
    return {
        "episode_reward": float(total_reward),
        "final_equity": float(info["equity"]),
        "steps": steps,
        "action_counts": {
            ACTION_NAMES[i]: int(action_counts[i])
            for i in range(3)
        },
        "action_distribution": {
            ACTION_NAMES[i]: float(distribution[i])
            for i in range(3)
        },
        "degenerate": bool(float(distribution.max()) >= 0.99),
        "final_position": int(info["position"]),
        "final_timestamp": str(info["timestamp"]),
    }


def summarize_training_reward(metrics_path: Path) -> dict[str, Any]:
    if not metrics_path.exists():
        return {
            "episodes": 0,
            "mean_reward": None,
            "last_reward": None,
        }
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    rewards = payload.get("training_episode_rewards", {})
    return {
        "episodes": int(rewards.get("count", 0)),
        "mean_reward": rewards.get("mean"),
        "last_reward": rewards.get("last"),
    }


def train_one_seed(
    *,
    seed: int,
    config: dict[str, Any],
    train_features: pd.DataFrame,
    validation_features: pd.DataFrame,
    train_prices: pd.Series,
    validation_prices: pd.Series,
    output_root: Path,
    timesteps: int,
    eval_freq: int,
    checkpoint_freq: int,
    device: str,
) -> dict[str, Any]:
    agent_cfg = config["agent"]
    environment_cfg = config["environment"]

    seed_root = output_root / "models" / "prototype" / f"seed_{seed}"
    log_root = output_root / "logs" / "prototype" / f"seed_{seed}"
    report_root = output_root / "reports" / "ppo_prototype" / f"seed_{seed}"
    seed_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)
    report_root.mkdir(parents=True, exist_ok=True)

    set_random_seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    train_env = DummyVecEnv(
        [
            build_env_factory(
                train_features,
                train_prices,
                environment_cfg,
                "train",
                seed,
            )
        ]
    )
    validation_env = DummyVecEnv(
        [
            build_env_factory(
                validation_features,
                validation_prices,
                environment_cfg,
                "validation",
                seed,
            )
        ]
    )

    eval_callback = EvalCallback(
        validation_env,
        best_model_save_path=str(seed_root / "best"),
        log_path=str(log_root / "eval"),
        eval_freq=max(1, int(eval_freq)),
        n_eval_episodes=1,
        deterministic=bool(agent_cfg.get("deterministic_validation", True)),
        render=False,
    )
    callbacks = build_ppo_callbacks(
        output_dir=log_root,
        checkpoint_freq=checkpoint_freq,
        eval_callback=eval_callback,
    )

    hidden_layers = list(agent_cfg.get("policy_hidden_layers", [128, 128]))
    policy_kwargs = {
        "net_arch": {
            "pi": hidden_layers,
            "vf": hidden_layers,
        }
    }

    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=float(agent_cfg["learning_rate"]),
        n_steps=int(agent_cfg["n_steps"]),
        batch_size=int(agent_cfg["batch_size"]),
        n_epochs=int(agent_cfg["n_epochs"]),
        gamma=float(agent_cfg["gamma"]),
        gae_lambda=float(agent_cfg["gae_lambda"]),
        clip_range=float(agent_cfg["clip_range"]),
        ent_coef=float(agent_cfg["ent_coef"]),
        vf_coef=float(agent_cfg["vf_coef"]),
        max_grad_norm=float(agent_cfg["max_grad_norm"]),
        policy_kwargs=policy_kwargs,
        seed=seed,
        device=device,
        verbose=1,
    )
    model.set_logger(
        configure(
            str(log_root),
            ["stdout", "csv", "tensorboard"],
        )
    )
    model.learn(
        total_timesteps=int(timesteps),
        callback=callbacks,
        progress_bar=False,
    )

    final_model_path = seed_root / "final_model"
    model.save(str(final_model_path))

    raw_validation_env = TradingEnv(
        features=validation_features,
        mid_prices=validation_prices,
        execution_config=ExecutionCostsConfig(
            spread_rate=float(environment_cfg["spread_rate"]),
            commission_rate=float(environment_cfg["commission_rate"]),
            slippage_rate=float(environment_cfg["slippage_rate"]),
            turnover_penalty=float(environment_cfg["turnover_penalty"]),
        ),
        initial_balance=float(environment_cfg["initial_balance"]),
        mode="validation",
        max_episode_steps=None,
        random_start=False,
    )

    validation_result = evaluate_deterministic(
        model,
        raw_validation_env,
    )
    training_result = summarize_training_reward(
        log_root / "training_action_metrics.json"
    )

    summary = {
        "seed": seed,
        "timesteps": int(timesteps),
        "training": training_result,
        "validation": validation_result,
        "training_vs_validation_reward": {
            "training_mean_reward": training_result["mean_reward"],
            "validation_episode_reward": validation_result["episode_reward"],
        },
        "non_degenerate": not validation_result["degenerate"],
        "reproducibility_seed": seed,
        "final_model": str(final_model_path.relative_to(output_root)),
    }
    (report_root / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    train_env.close()
    validation_env.close()
    raw_validation_env.close()
    return summary


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    agent_cfg = config["agent"]

    timesteps = int(
        args.timesteps
        if args.timesteps is not None
        else agent_cfg["total_timesteps"]
    )
    train_fraction = float(
        args.train_fraction
        if args.train_fraction is not None
        else agent_cfg.get("train_fraction", 0.25)
    )
    if not 0 < train_fraction <= 1:
        raise ValueError("train_fraction deve essere > 0 e <= 1.")

    seeds = (
        args.seeds
        if args.seeds is not None
        else [
            int(value)
            for value in agent_cfg.get(
                "seeds",
                [config["project"]["seed"]],
            )
        ]
    )
    eval_freq = int(
        args.eval_freq
        if args.eval_freq is not None
        else agent_cfg.get("eval_freq", 25000)
    )
    checkpoint_freq = int(
        args.checkpoint_freq
        if args.checkpoint_freq is not None
        else agent_cfg.get("checkpoint_freq", 25000)
    )
    device = args.device or str(agent_cfg.get("device", "auto"))

    (
        train_features,
        validation_features,
        train_prices,
        validation_prices,
    ) = prepare_data(config, m15_path=args.m15_path)

    subset_end = max(2, int(len(train_features) * train_fraction))
    train_features = train_features.iloc[:subset_end].copy()
    train_prices = train_prices.loc[train_features.index].copy()

    output_root = ROOT
    summaries = []
    for seed in seeds:
        summaries.append(
            train_one_seed(
                seed=int(seed),
                config=config,
                train_features=train_features,
                validation_features=validation_features,
                train_prices=train_prices,
                validation_prices=validation_prices,
                output_root=output_root,
                timesteps=timesteps,
                eval_freq=eval_freq,
                checkpoint_freq=checkpoint_freq,
                device=device,
            )
        )

    report = {
        "algorithm": "PPO",
        "seeds": [int(seed) for seed in seeds],
        "train_fraction": train_fraction,
        "timesteps": timesteps,
        "validation_not_test": True,
        "runs": summaries,
        "all_validation_runs_non_degenerate": all(
            item["non_degenerate"] for item in summaries
        ),
    }
    report_path = output_root / "reports" / "ppo_prototype" / "summary.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
