import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from baselines import build_target_positions, calculate_metrics, make_execution_config, run_baselines
from gold_rl.execution import ExecutionCostsConfig
from gold_rl.simulation import simulate_portfolio_over_series


def make_data(rows: int = 120) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=rows, freq="min")
    close = 1900.0 + np.cumsum(np.sin(np.arange(rows) / 7.0) * 0.5 + 0.05)
    return pd.DataFrame({"Close": close}, index=index)


def make_config() -> dict:
    return {
        "project": {"seed": 42},
        "environment": {
            "initial_balance": 100000.0,
            "spread_rate": 0.0002,
            "commission_rate": 0.0001,
            "slippage_rate": 0.00005,
            "turnover_penalty": 0.0001,
        },
    }


def test_all_strategies_generate_allowed_positions_and_same_engine() -> None:
    data = make_data()
    positions = build_target_positions(data, seed=42)
    assert tuple(positions) == (
        "always_flat", "always_long", "random", "momentum", "ema_crossover"
    )
    for series in positions.values():
        assert set(series.unique()).issubset({-1, 0, 1})

    config = make_config()
    baseline = run_baselines(data, config)
    assert len(baseline) == 10
    assert set(baseline["strategy"]) == {
        "always_flat", "always_long", "random", "momentum", "ema_crossover"
    }
    assert set(baseline["costs"]) == {"with_costs", "without_costs"}


def test_zero_cost_config_matches_mid_price_execution() -> None:
    config = make_config()["environment"]
    zero = make_execution_config(config, with_costs=False)
    assert zero == ExecutionCostsConfig(0.0, 0.0, 0.0, 0.0)

    data = make_data(20)
    positions = build_target_positions(data)["always_long"].to_numpy()
    result = simulate_portfolio_over_series(
        data["Close"], positions, zero, 100000.0
    )
    assert result["equity"].notna().all()


def test_metrics_are_well_defined() -> None:
    equity = pd.Series([100.0, 101.0, 99.0, 102.0])
    positions = pd.Series([0, 1, 1, 0])
    metrics = calculate_metrics(equity, positions, 100.0)
    assert np.isclose(metrics["total_return"], 0.02)
    assert metrics["max_drawdown"] < 0.0
    assert metrics["trades"] == 2
