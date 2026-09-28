import numpy as np
import pandas as pd
from pathlib import Path
import sys
from gymnasium.utils.env_checker import check_env

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIRECTORY = PROJECT_ROOT / "src"
REPORTS_DIRECTORY = PROJECT_ROOT / "reports"

# Permette a Python di trovare il package src/gold_rl.
sys.path.insert(0, str(SRC_DIRECTORY))

from gold_rl.data.features import (
    MARKET_FEATURE_COLUMNS,
    create_market_features,
)
from gold_rl.execution import ExecutionCostsConfig
from gold_rl.trading_env import TradingEnv


# 1. Carica M15
ohlcv = pd.read_parquet(
    "data/processed/xauusd_m15.parquet"
)

# 2. Costruisci feature esattamente come Fase 3
features = create_market_features(
    ohlcv,
    drop_warmup=True,
    strict_windows=True,
)

features = features.loc[:, MARKET_FEATURE_COLUMNS]

# 3. Allinea il prezzo alle feature
prices = ohlcv.loc[features.index, "Close"].astype(float)

# 4. Controlli dati
assert len(features) == len(prices)
assert features.index.is_monotonic_increasing
assert np.isfinite(features.to_numpy()).all()
assert np.isfinite(prices.to_numpy()).all()
assert (prices > 0).all()

print("Dati:", len(features))
print("Da:", features.index.min())
print("A:", features.index.max())

# 5. Costi
costs = costs = ExecutionCostsConfig(
    spread_rate=0.0002,
    commission_rate=0.0001,
    slippage_rate=0.00005,
    turnover_penalty=0.0001,
)

# 6. Environment
env = TradingEnv(
    features=features,
    mid_prices=prices,
    execution_config=costs,
    mode="train",
    max_episode_steps=5000,
    random_start=True,
)

# 7. Gymnasium check
check_env(env, skip_render_check=True)

print("check_env: OK")

# 8. Episodio random
obs, info = env.reset(seed=42)

steps = 0

while True:
    assert np.isfinite(obs).all()

    action = env.action_space.sample()

    obs, reward, terminated, truncated, info = env.step(action)

    assert np.isfinite(obs).all()
    assert np.isfinite(reward)
    assert np.isfinite(info["equity"])

    steps += 1

    if terminated or truncated:
        break

print("Random episode: OK")
print("Steps:", steps)
print("Final equity:", info["equity"])