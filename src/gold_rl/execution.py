from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExecutionCostsConfig:
    spread_rate: float
    commission_rate: float
    slippage_rate: float
    turnover_penalty: float


@dataclass(frozen=True)
class ExecutionResult:
    timestamp: object
    mid_price: float
    execution_price: float
    previous_position: int
    target_position: int
    trade_size: int
    turnover: float
    spread_cost: float
    slippage_cost: float
    commission_cost: float
    turnover_cost: float
    total_cost: float


def validate_position_transition(
    previous_position: int,
    target_position: int,
    allowed_positions: tuple[int, ...] = (-1, 0, 1),
) -> None:
    if previous_position not in allowed_positions:
        raise ValueError(
            f"Posizione precedente non valida: {previous_position}"
        )

    if target_position not in allowed_positions:
        raise ValueError(
            f"Posizione target non valida: {target_position}"
        )


def get_trade_size(
    previous_position: int,
    target_position: int,
) -> int:
    return target_position - previous_position


def get_turnover(
    previous_position: int,
    target_position: int,
) -> float:
    return float(abs(target_position - previous_position))


def get_execution_price(
    mid_price: float,
    trade_size: int,
    spread_rate: float,
    slippage_rate: float,
) -> float:
    if mid_price <= 0:
        raise ValueError("mid_price deve essere positivo.")

    if trade_size == 0:
        return float(mid_price)

    half_spread = spread_rate / 2.0

    if trade_size > 0:
        return float(
            mid_price * (1.0 + half_spread + slippage_rate)
        )

    return float(
        mid_price * (1.0 - half_spread - slippage_rate)
    )


def calculate_transaction_costs(
    mid_price: float,
    turnover: float,
    config: ExecutionCostsConfig,
) -> tuple[float, float, float, float, float]:
    notional = mid_price * turnover

    spread_cost = notional * (config.spread_rate / 2.0)
    slippage_cost = notional * config.slippage_rate
    commission_cost = notional * config.commission_rate
    turnover_cost = notional * config.turnover_penalty

    total_cost = (
        spread_cost
        + slippage_cost
        + commission_cost
        + turnover_cost
    )

    return (
        float(spread_cost),
        float(slippage_cost),
        float(commission_cost),
        float(turnover_cost),
        float(total_cost),
    )


def execute_order(
    timestamp: object,
    mid_price: float,
    previous_position: int,
    target_position: int,
    config: ExecutionCostsConfig,
) -> ExecutionResult:
    validate_position_transition(
        previous_position=previous_position,
        target_position=target_position,
    )

    trade_size = get_trade_size(
        previous_position=previous_position,
        target_position=target_position,
    )

    turnover = get_turnover(
        previous_position=previous_position,
        target_position=target_position,
    )

    execution_price = get_execution_price(
        mid_price=mid_price,
        trade_size=trade_size,
        spread_rate=config.spread_rate,
        slippage_rate=config.slippage_rate,
    )

    (
        spread_cost,
        slippage_cost,
        commission_cost,
        turnover_cost,
        total_cost,
    ) = calculate_transaction_costs(
        mid_price=mid_price,
        turnover=turnover,
        config=config,
    )

    return ExecutionResult(
        timestamp=timestamp,
        mid_price=float(mid_price),
        execution_price=execution_price,
        previous_position=previous_position,
        target_position=target_position,
        trade_size=trade_size,
        turnover=turnover,
        spread_cost=spread_cost,
        slippage_cost=slippage_cost,
        commission_cost=commission_cost,
        turnover_cost=turnover_cost,
        total_cost=total_cost,
    )