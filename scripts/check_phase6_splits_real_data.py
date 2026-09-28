from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from gymnasium.utils.env_checker import check_env

from gold_rl.data.features import MARKET_FEATURE_COLUMNS, create_market_features
from gold_rl.execution import ExecutionCostsConfig
from gold_rl.trading_env import TradingEnv


TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15


def make_env(features: pd.DataFrame, prices: pd.Series, mode: str) -> TradingEnv:
    costs = ExecutionCostsConfig(
        spread_rate=0.0002,
        commission_rate=0.0001,
        slippage_rate=0.00005,
        turnover_penalty=0.0001,
    )
    return TradingEnv(
        features=features,
        mid_prices=prices,
        execution_config=costs,
        initial_balance=100_000.0,
        mode=mode,
        max_episode_steps=5000 if mode == "train" else None,
        random_start=True if mode == "train" else False,
    )


def split_by_ratio(
    features: pd.DataFrame,
    prices: pd.Series,
) -> tuple[tuple[pd.DataFrame, pd.Series], tuple[pd.DataFrame, pd.Series], tuple[pd.DataFrame, pd.Series]]:
    n = len(features)
    train_end = int(n * TRAIN_RATIO)
    validation_end = train_end + int(n * VALIDATION_RATIO)

    train_features = features.iloc[:train_end].copy()
    validation_features = features.iloc[train_end:validation_end].copy()
    test_features = features.iloc[validation_end:].copy()

    train_prices = prices.loc[train_features.index].copy()
    validation_prices = prices.loc[validation_features.index].copy()
    test_prices = prices.loc[test_features.index].copy()

    return (
        (train_features, train_prices),
        (validation_features, validation_prices),
        (test_features, test_prices),
    )


def check_segment(name: str, env: TradingEnv, full_episode: bool) -> None:
    check_env(env, skip_render_check=True)
    print(f"{name} check_env: OK")

    obs, _ = env.reset(seed=42)
    assert np.isfinite(obs).all()

    steps = 0
    while True:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        assert np.isfinite(obs).all()
        assert np.isfinite(reward)
        assert np.isfinite(info["equity"])

        steps += 1

        if terminated or truncated:
            break

    if full_episode:
        assert steps == len(env.features) - 1

    print(
        f"{name} episode: OK | steps={steps} | "
        f"final_equity={info['equity']:.2f}"
    )


def main() -> None:
    path = ROOT / "data" / "processed" / "xauusd_m15.parquet"
    ohlcv = pd.read_parquet(path)

    if not isinstance(ohlcv.index, pd.DatetimeIndex):
        raise TypeError("xauusd_m15.parquet deve avere un DatetimeIndex.")

    features = create_market_features(
        ohlcv,
        drop_warmup=True,
        strict_windows=True,
    ).loc[:, MARKET_FEATURE_COLUMNS]

    prices = ohlcv.loc[features.index, "Close"].astype("float64")

    assert len(features) == len(prices)
    assert features.index.equals(prices.index)
    assert features.index.is_monotonic_increasing
    assert np.isfinite(features.to_numpy()).all()
    assert np.isfinite(prices.to_numpy()).all()
    assert (prices > 0).all()

    print(f"Dati totali: {len(features)}")
    print(f"Da: {features.index.min()}")
    print(f"A: {features.index.max()}")

    train, validation, test = split_by_ratio(features, prices)

    train_features, train_prices = train
    validation_features, validation_prices = validation
    test_features, test_prices = test

    print(
        f"Split: train={len(train_features)}, "
        f"validation={len(validation_features)}, test={len(test_features)}"
    )
    print(
        f"Train: {train_features.index.min()} -> {train_features.index.max()}"
    )
    print(
        f"Validation: {validation_features.index.min()} -> "
        f"{validation_features.index.max()}"
    )
    print(
        f"Test: {test_features.index.min()} -> {test_features.index.max()}"
    )

    assert len(train_features) + len(validation_features) + len(test_features) == len(features)
    assert train_features.index.max() < validation_features.index.min()
    assert validation_features.index.max() < test_features.index.min()

    train_env = make_env(train_features, train_prices, "train")
    validation_env = make_env(validation_features, validation_prices, "validation")
    test_env = make_env(test_features, test_prices, "test")

    assert train_env.random_start is True
    assert validation_env.random_start is False
    assert test_env.random_start is False

    check_segment("TRAIN", train_env, full_episode=False)
    check_segment("VALIDATION", validation_env, full_episode=True)
    check_segment("TEST", test_env, full_episode=True)

    print("Phase 6 split/integration check: OK")


if __name__ == "__main__":
    main()
