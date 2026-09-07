import sys
from pathlib import Path

import math

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIRECTORY = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIRECTORY))

from gold_rl.execution import ExecutionCostsConfig, execute_order
from gold_rl.portfolio import initialize_portfolio, update_portfolio


def make_config() -> ExecutionCostsConfig:
    return ExecutionCostsConfig(
        spread_rate=0.0002,
        commission_rate=0.0001,
        slippage_rate=0.00005,
        turnover_penalty=0.0001,
    )


def test_manual_long_roundtrip_matches_expected_equity() -> None:
    config = make_config()
    state = initialize_portfolio(initial_balance=100000.0)

    open_trade = execute_order(
        timestamp="2024-01-01 10:00",
        mid_price=100.0,
        previous_position=0,
        target_position=1,
        config=config,
    )
    state = update_portfolio(
        previous_state=state,
        execution=open_trade,
        mid_price=100.0,
    )

    close_trade = execute_order(
        timestamp="2024-01-01 10:01",
        mid_price=101.0,
        previous_position=1,
        target_position=0,
        config=config,
    )
    state = update_portfolio(
        previous_state=state,
        execution=close_trade,
        mid_price=101.0,
    )

    open_cost = 100.0 * (
        0.0002 / 2.0 + 0.00005 + 0.0001 + 0.0001
    )
    close_cost = 101.0 * (
        0.0002 / 2.0 + 0.00005 + 0.0001 + 0.0001
    )

    open_exec = 100.0 * (1.0 + 0.0002 / 2.0 + 0.00005)
    close_exec = 101.0 * (1.0 - 0.0002 / 2.0 - 0.00005)

    expected_equity = (
        100000.0
        - open_exec - open_cost
        + close_exec - close_cost
    )

    assert state.position == 0
    assert state.entry_price is None
    assert math.isclose(state.equity, expected_equity, rel_tol=1e-12)


def test_manual_short_roundtrip_matches_expected_equity() -> None:
    config = make_config()
    state = initialize_portfolio(initial_balance=100000.0)

    open_trade = execute_order(
        timestamp="2024-01-01 10:00",
        mid_price=100.0,
        previous_position=0,
        target_position=-1,
        config=config,
    )
    state = update_portfolio(
        previous_state=state,
        execution=open_trade,
        mid_price=100.0,
    )

    close_trade = execute_order(
        timestamp="2024-01-01 10:01",
        mid_price=99.0,
        previous_position=-1,
        target_position=0,
        config=config,
    )
    state = update_portfolio(
        previous_state=state,
        execution=close_trade,
        mid_price=99.0,
    )

    open_cost = 100.0 * (
        0.0002 / 2.0 + 0.00005 + 0.0001 + 0.0001
    )
    close_cost = 99.0 * (
        0.0002 / 2.0 + 0.00005 + 0.0001 + 0.0001
    )

    open_exec = 100.0 * (1.0 - 0.0002 / 2.0 - 0.00005)
    close_exec = 99.0 * (1.0 + 0.0002 / 2.0 + 0.00005)

    expected_equity = (
        100000.0
        + open_exec - open_cost
        - close_exec - close_cost
    )

    assert state.position == 0
    assert state.entry_price is None
    assert math.isclose(state.equity, expected_equity, rel_tol=1e-12)


def test_hold_long_updates_unrealized_pnl() -> None:
    config = make_config()
    state = initialize_portfolio(initial_balance=100000.0)

    open_trade = execute_order(
        timestamp="2024-01-01 10:00",
        mid_price=100.0,
        previous_position=0,
        target_position=1,
        config=config,
    )
    state = update_portfolio(
        previous_state=state,
        execution=open_trade,
        mid_price=100.0,
    )

    hold_trade = execute_order(
        timestamp="2024-01-01 10:01",
        mid_price=102.0,
        previous_position=1,
        target_position=1,
        config=config,
    )
    state = update_portfolio(
        previous_state=state,
        execution=hold_trade,
        mid_price=102.0,
    )

    assert state.position == 1
    assert state.unrealized_pnl > 0.0