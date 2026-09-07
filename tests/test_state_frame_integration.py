import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIRECTORY = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIRECTORY))

from gold_rl.data.features import create_market_features
from gold_rl.data.leakage_checks import assert_future_append_invariant
from gold_rl.execution import ExecutionCostsConfig
from gold_rl.simulation import simulate_portfolio_over_series


def make_ohlcv_dataframe(rows: int = 260) -> pd.DataFrame:
    index = pd.date_range("2024-05-01", periods=rows, freq="1min")
    base = 1900.0 + np.linspace(0.0, 15.0, rows)
    close = base + 0.6 * np.sin(np.arange(rows) / 9.0)
    open_price = close - 0.1 * np.cos(np.arange(rows) / 7.0)
    high = np.maximum(open_price, close) + 0.2
    low = np.minimum(open_price, close) - 0.2
    volume = 90.0 + (np.arange(rows) % 13)

    return pd.DataFrame(
        {"Open": open_price, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=index,
    )


def make_config() -> ExecutionCostsConfig:
    return ExecutionCostsConfig(
        spread_rate=0.0002,
        commission_rate=0.0001,
        slippage_rate=0.00005,
        turnover_penalty=0.0001,
    )


def build_features_with_real_state(dataframe: pd.DataFrame) -> pd.DataFrame:
    positions = [((i // 20) % 3) - 1 for i in range(len(dataframe))]

    simulated_state = simulate_portfolio_over_series(
        mid_prices=dataframe["Close"],
        target_positions=positions,
        config=make_config(),
        initial_balance=100000.0,
    )

    state_frame = simulated_state[["current_position", "unrealized_return"]]

    return create_market_features(dataframe=dataframe, state_frame=state_frame)


def test_state_frame_reflects_simulated_positions() -> None:
    dataframe = make_ohlcv_dataframe()
    features = build_features_with_real_state(dataframe)

    assert features["current_position"].isin([-1, 0, 1]).all()
    assert not (features["current_position"] == 0).all()


def test_future_append_holds_with_real_state_frame() -> None:
    dataframe = make_ohlcv_dataframe(rows=260)
    history = dataframe.iloc[:180].copy()
    future = dataframe.iloc[180:].copy()

    assert_future_append_invariant(
        feature_builder=lambda frame: build_features_with_real_state(frame),
        history_dataframe=history,
        future_dataframe=future,
    )