from __future__ import annotations

from dataclasses import dataclass

from gold_rl.execution import ExecutionResult


@dataclass(frozen=True)
class PortfolioState:
    cash: float
    equity: float
    position: int
    entry_price: float | None
    realized_pnl: float
    unrealized_pnl: float
    cumulative_costs: float
    cumulative_turnover: float


def initialize_portfolio(
    initial_balance: float,
) -> PortfolioState:
    return PortfolioState(
        cash=float(initial_balance),
        equity=float(initial_balance),
        position=0,
        entry_price=None,
        realized_pnl=0.0,
        unrealized_pnl=0.0,
        cumulative_costs=0.0,
        cumulative_turnover=0.0,
    )


def calculate_unrealized_pnl(
    position: int,
    entry_price: float | None,
    mid_price: float,
) -> float:
    if position == 0 or entry_price is None:
        return 0.0

    if position > 0:
        return float((mid_price - entry_price) * position)

    return float((entry_price - mid_price) * abs(position))


def mark_to_market_value(
    position: int,
    mid_price: float,
) -> float:
    return float(position * mid_price)


def update_portfolio(
    previous_state: PortfolioState,
    execution: ExecutionResult,
    mid_price: float,
) -> PortfolioState:
    previous_position = previous_state.position
    target_position = execution.target_position
    trade_size = execution.trade_size

    cash = previous_state.cash
    realized_pnl = previous_state.realized_pnl
    cumulative_costs = (
        previous_state.cumulative_costs + execution.total_cost
    )
    cumulative_turnover = (
        previous_state.cumulative_turnover + execution.turnover
    )

    previous_entry_price = previous_state.entry_price
    new_entry_price = previous_entry_price

    if trade_size != 0:
        cash -= execution.execution_price * trade_size
        cash -= execution.total_cost

    if previous_position == 0:
        if target_position != 0:
            new_entry_price = execution.execution_price

    elif previous_position > 0:
        if target_position == 0:
            realized_pnl += (
                execution.execution_price - previous_entry_price
            ) * previous_position
            new_entry_price = None

        elif target_position > 0:
            if target_position != previous_position:
                new_entry_price = execution.execution_price

        else:
            realized_pnl += (
                execution.execution_price - previous_entry_price
            ) * previous_position
            new_entry_price = execution.execution_price

    elif previous_position < 0:
        if target_position == 0:
            realized_pnl += (
                previous_entry_price - execution.execution_price
            ) * abs(previous_position)
            new_entry_price = None

        elif target_position < 0:
            if target_position != previous_position:
                new_entry_price = execution.execution_price

        else:
            realized_pnl += (
                previous_entry_price - execution.execution_price
            ) * abs(previous_position)
            new_entry_price = execution.execution_price

    unrealized_pnl = calculate_unrealized_pnl(
        position=target_position,
        entry_price=new_entry_price,
        mid_price=mid_price,
    )

    equity = cash + mark_to_market_value(
        position=target_position,
        mid_price=mid_price,
    )

    return PortfolioState(
        cash=float(cash),
        equity=float(equity),
        position=int(target_position),
        entry_price=new_entry_price,
        realized_pnl=float(realized_pnl),
        unrealized_pnl=float(unrealized_pnl),
        cumulative_costs=float(cumulative_costs),
        cumulative_turnover=float(cumulative_turnover),
    )