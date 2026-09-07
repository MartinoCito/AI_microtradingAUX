import sys
from pathlib import Path

import math

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIRECTORY = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIRECTORY))

from gold_rl.execution import (
    ExecutionCostsConfig,
    execute_order,
    get_execution_price,
)


def make_config() -> ExecutionCostsConfig:
    return ExecutionCostsConfig(
        spread_rate=0.0002,
        commission_rate=0.0001,
        slippage_rate=0.00005,
        turnover_penalty=0.0001,
    )


def test_buy_execution_price_is_worse_than_mid() -> None:
    price = get_execution_price(
        mid_price=100.0,
        trade_size=1,
        spread_rate=0.0002,
        slippage_rate=0.00005,
    )
    assert math.isclose(price, 100.015, rel_tol=1e-12)


def test_sell_execution_price_is_worse_than_mid() -> None:
    price = get_execution_price(
        mid_price=100.0,
        trade_size=-1,
        spread_rate=0.0002,
        slippage_rate=0.00005,
    )
    assert math.isclose(price, 99.985, rel_tol=1e-12)


def test_no_trade_keeps_mid_price() -> None:
    price = get_execution_price(
        mid_price=100.0,
        trade_size=0,
        spread_rate=0.0002,
        slippage_rate=0.00005,
    )
    assert math.isclose(price, 100.0, rel_tol=1e-12)


def test_turnover_for_flip_is_two() -> None:
    result = execute_order(
        timestamp="2024-01-01 10:00",
        mid_price=100.0,
        previous_position=1,
        target_position=-1,
        config=make_config(),
    )
    assert math.isclose(result.turnover, 2.0, rel_tol=1e-12)


def test_total_cost_is_sum_of_components() -> None:
    result = execute_order(
        timestamp="2024-01-01 10:00",
        mid_price=100.0,
        previous_position=0,
        target_position=1,
        config=make_config(),
    )
    expected = (
        result.spread_cost
        + result.slippage_cost
        + result.commission_cost
        + result.turnover_cost
    )
    assert math.isclose(result.total_cost, expected, rel_tol=1e-12)