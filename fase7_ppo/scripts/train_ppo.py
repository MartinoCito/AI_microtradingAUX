from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gold_rl.data.features import MARKET_FEATURE_COLUMNS
from gold_rl.execution import ExecutionCostsConfig
from gold_rl.trading_env import TradingEnv
from callbacks import PPOTrainingMonitor


def parse_args():
    p = argparse.ArgumentParser(description="Fase 7 - addestramento PPO preliminare.")
    p.add_argument("--train-features", type=Path, required=True)
    p.add_argument("--validation-features", type=Path, required=True)
    p.add_argument("--train-prices", type=Path, required=True)
    p.add_argument("--validation-prices", type=Path, required=True)
    p.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.yaml")
    p.add_argument("--train-fraction", type=float, default=0.25)
    p.add_argument("--timesteps", type=int, default=250_000)
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--eval-freq", type=int, default=25_000)
    p.add_argument("--checkpoint-freq", type=int, default=25_000)
    p.add_argument("--device", default="auto")
    p.add_argument("--output-root", type=Path, default=PROJECT_ROOT)
    return p.parse_args()


def load_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        df = pd.read_parquet(path)
    elif path.suffix.lower() in {".csv", ".txt"}:
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Formato non supportato: {path.suffix}")
    if not isinstance(df.index, pd.DatetimeIndex):
        for c in ("Date", "date", "timestamp", "Timestamp"):
            if c in df.columns:
                df[c] = pd.to_datetime(df[c], errors="raise")
                df = df.set_index(c)
                break
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError(f"{path}: serve DatetimeIndex o colonna timestamp.")
    df = df.sort_index()
    if df.index.has_duplicates:
        raise ValueError(f"{path}: timestamp duplicati.")
    return df


def load_features(path: Path) -> pd.DataFrame:
    df = load_table(path)
    missing = [c for c in MARKET_FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: feature mancanti: {missing}")
    out = df.loc[:, list(MARKET_FEATURE_COLUMNS)].astype("float64")
    if not np.isfinite(out.to_numpy()).all():
        raise ValueError(f"{path}: feature non finite.")
    return out


def load_close(path: Path) -> pd.Series:
    df = load_table(path)
    c = "Close" if "Close" in df.columns else "close" if "close" in df.columns else None
    if c is None:
        raise ValueError(f"{path}: manca Close.")
    out = pd.to_numeric(df[c], errors="raise").astype("float64")
    if not np.isfinite(out.to_numpy()).all() or (out <= 0).any():
        raise ValueError(f"{path}: prezzi non validi.")
    return out.rename("Close")


def align(features, prices, label):
    idx = features.index.intersection(prices.index)
    if len(idx) < 2:
        raise ValueError(f"{label}: meno di due timestamp comuni.")
    features, prices = features.loc[idx], prices.loc[idx]
    if not features.index.equals(prices.index):
        raise RuntimeError(f"{label}: allineamento fallito.")
    return features, prices


def make_env(features, prices, cfg, mode, seed, max_steps, random_start):
    ecfg = cfg["environment"]
    execution = ExecutionCostsConfig(
        spread_rate=float(ecfg["spread_rate"]),
        commission_rate=float(ecfg["commission_rate"]),
        slippage_rate=float(ecfg["slippage_rate"]),
        turnover_penalty=float(ecfg["turnover_penalty"]),
    )
    def factory():
        return Monitor(TradingEnv(
            features=features,
            mid_prices=prices,
            execution_config=execution,
            initial_balance=float(ecfg["initial_balance"]),
            mode=mode,
            max_episode_steps=max_steps,
            random_start=random_start,
            turnover_reward_weight=float(ecfg.get("turnover_reward_weight", 0.0)),
        ))
    env = VecMonitor(DummyVecEnv([factory]))
    env.seed(seed)
    return env


def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_random_seed(seed)


def evaluate(model, env, seed):
    obs = env.reset()
    done = np.array([False])
    reward = 0.0
    actions, positions, equities = [], [], []
    while not bool(done[0]):
        action, _ = model.predict(obs, deterministic=True)
        actions.append(int(np.asarray(action).reshape(-1)[0]))
        obs, rewards, dones, infos = env.step(action)
        reward += float(rewards[0])
        info = infos[0]
        positions.append(int(info.get("position", 0)))
        equities.append(float(info.get("equity", np.nan)))
        done = dones
    counts = np.bincount(np.asarray(actions), minlength=3)
    n = max(1, len(actions))
    dist = {"short": counts[0]/n, "flat": counts[1]/n, "long": counts[2]/n}
    return {
        "seed": seed,
        "steps": len(actions),
        "episode_reward": float(reward),
        "final_equity": float(equities[-1]) if equities else None,
        "action_counts": {"short": int(counts[0]), "flat": int(counts[1]), "long": int(counts[2])},
        "action_distribution": {k: float(v) for k, v in dist.items()},
        "position_distribution": {
            "short": float(np.mean(np.asarray(positions) == -1)) if positions else 0.0,
            "flat": float(np.mean(np.asarray(positions) == 0)) if positions else 0.0,
            "long": float(np.mean(np.asarray(positions) == 1)) if positions else 0.0,
        },
        "degenerate": max(dist.values()) >= 0.99,
    }


def main():
    args = parse_args()
    with args.config.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    trf, trp = align(load_features(args.train_features), load_close(args.train_prices), "train")
    vf, vp = align(load_features(args.validation_features), load_close(args.validation_prices), "validation")

    if not 0 < args.train_fraction <= 1:
        raise ValueError("--train-fraction deve essere >0 e <=1")
    n = max(2, int(len(trf) * args.train_fraction))
    trf, trp = trf.iloc[:n].copy(), trp.iloc[:n].copy()

    model_root = args.output_root / "models/prototype"
    log_root = args.output_root / "logs/prototype"
    report_root = args.output_root / "reports/ppo_prototype"
    model_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)
    report_root.mkdir(parents=True, exist_ok=True)

    agent = cfg.get("agent", {})
    results = []

    for seed in args.seeds:
        seed_everything(seed)
        seed_dir = model_root / f"seed_{seed}"
        seed_log = log_root / f"seed_{seed}"
        eval_dir = seed_dir / "eval"
        checkpoint_dir = seed_dir / "checkpoints"
        for d in (seed_dir, seed_log, eval_dir, checkpoint_dir):
            d.mkdir(parents=True, exist_ok=True)

        train_env = make_env(trf, trp, cfg, "train", seed,
                             int(cfg["environment"].get("max_episode_steps", 5000)), True)
        val_env = make_env(vf, vp, cfg, "validation", seed, None, False)

        monitor = PPOTrainingMonitor(
            seed_dir / "training_metrics.jsonl",
            checkpoint_dir,
            args.checkpoint_freq,
        )
        eval_cb = EvalCallback(
            val_env,
            best_model_save_path=str(eval_dir),
            log_path=str(eval_dir),
            eval_freq=max(args.eval_freq, int(agent.get("n_steps", 2048))),
            deterministic=True,
            verbose=1,
        )

        model = PPO(
            "MlpPolicy",
            train_env,
            learning_rate=float(agent.get("learning_rate", 3e-4)),
            n_steps=int(agent.get("n_steps", 2048)),
            batch_size=int(agent.get("batch_size", 256)),
            n_epochs=int(agent.get("n_epochs", 10)),
            gamma=float(agent.get("gamma", 0.99)),
            gae_lambda=float(agent.get("gae_lambda", 0.95)),
            clip_range=float(agent.get("clip_range", 0.2)),
            ent_coef=float(agent.get("ent_coef", 0.0)),
            vf_coef=float(agent.get("vf_coef", 0.5)),
            max_grad_norm=float(agent.get("max_grad_norm", 0.5)),
            policy_kwargs={"net_arch": {"pi": [128,128], "vf": [128,128]},
                           "activation_fn": torch.nn.Tanh},
            tensorboard_log=str(seed_log / "tensorboard"),
            seed=seed,
            device=args.device,
            verbose=1,
        )
        model.set_logger(configure(str(seed_log), ["stdout", "csv", "tensorboard"]))

        model.learn(
            total_timesteps=args.timesteps,
            callback=[monitor, eval_cb],
            progress_bar=True,
        )
        model.save(str(seed_dir / "final_model"))

        result = evaluate(model, val_env, seed)
        (seed_dir / "validation_result.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
        results.append(result)
        train_env.close()
        val_env.close()

    summary = {
        "train_rows_used": len(trf),
        "validation_rows": len(vf),
        "train_fraction": args.train_fraction,
        "timesteps": args.timesteps,
        "seeds": args.seeds,
        "results": results,
        "degenerate_seed_count": sum(bool(x["degenerate"]) for x in results),
    }
    (report_root / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
