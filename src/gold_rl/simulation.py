from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from gold_rl.execution import ExecutionCostsConfig, execute_order
from gold_rl.portfolio import PortfolioState, initialize_portfolio, update_portfolio


def simulate_portfolio_over_series(
    mid_prices: pd.Series,
    target_positions: Sequence[int],
    config: ExecutionCostsConfig,
    initial_balance: float,
) -> pd.DataFrame:
    """
    Simula l'Execution Engine barra per barra su una sequenza di decisioni
    di posizione già note (es. da una policy fissa o da un backtest storico).

    Ogni riga dipende esclusivamente dal prezzo corrente e dallo stato del
    portafoglio calcolato sulle barre precedenti: nessuna informazione
    futura entra nel calcolo, quindi il risultato è causale per costruzione
    e compatibile con il criterio future-append della Fase 3.
    """
    if len(mid_prices) != len(target_positions):
        raise ValueError(
            "mid_prices e target_positions devono avere la stessa lunghezza."
        )

    state: PortfolioState = initialize_portfolio(initial_balance=initial_balance)

    positions: list[int] = []
    unrealized_returns: list[float] = []
    equities: list[float] = []

    for timestamp, mid_price, target_position in zip(
        mid_prices.index, mid_prices.to_numpy(), target_positions
    ):
        execution = execute_order(
            timestamp=timestamp,
            mid_price=float(mid_price),
            previous_position=state.position,
            target_position=int(target_position),
            config=config,
        )

        state = update_portfolio(
            previous_state=state,
            execution=execution,
            mid_price=float(mid_price),
        )

        positions.append(state.position)
        equities.append(state.equity)
        unrealized_returns.append(
            state.unrealized_pnl / state.equity if state.equity != 0.0 else 0.0
        )

    return pd.DataFrame(
        {
            "current_position": positions,
            "unrealized_return": unrealized_returns,
            "equity": equities,
        },
        index=mid_prices.index,
    )