"""Chronological walk-forward folds and cost-aware OOS accounting."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from gold_rl.data.features import MARKET_FEATURE_COLUMNS, apply_normalization, fit_normalization_stats
from gold_rl.execution import ExecutionCostsConfig, execute_order
from gold_rl.portfolio import initialize_portfolio, update_portfolio


@dataclass(frozen=True)
class Fold:
    number: int
    train_start: pd.Timestamp
    validation_start: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    def metadata(self):
        return {k: str(v) if isinstance(v, pd.Timestamp) else v for k, v in asdict(self).items()}


def make_folds(index, *, n_folds, train_years, validation_months, test_months, holdout_start):
    if not isinstance(index, pd.DatetimeIndex) or not index.is_monotonic_increasing or index.has_duplicates:
        raise ValueError('Indice temporale non valido.')
    if not 3 <= n_folds <= 5 or min(train_years, validation_months, test_months) <= 0:
        raise ValueError('Usare 3-5 fold e durate positive.')
    holdout_start = pd.Timestamp(holdout_start)
    folds = []
    for number in range(1, n_folds + 1):
        # Compute each edge from the same origin to avoid month-end drift.
        end = holdout_start - pd.DateOffset(months=(n_folds - number) * test_months)
        start = holdout_start - pd.DateOffset(months=(n_folds - number + 1) * test_months)
        validation_start = start - pd.DateOffset(months=validation_months)
        train_start = validation_start - pd.DateOffset(years=train_years)
        fold = Fold(number, train_start, validation_start, start, end)
        if train_start < index[0]:
            raise ValueError('Storia insufficiente per la durata richiesta.')
        for left, right in [(train_start, validation_start), (validation_start, start), (start, end)]:
            if ((index >= left) & (index < right)).sum() < 2:
                raise ValueError('Finestra vuota o troppo corta.')
        folds.append(fold)
    return folds


def prepare_fold(features, prices, fold, *, normalize=True):
    train = features.loc[(features.index >= fold.train_start) & (features.index < fold.validation_start)].copy()
    validation = features.loc[(features.index >= fold.validation_start) & (features.index < fold.test_start)].copy()
    test = features.loc[(features.index >= fold.test_start) & (features.index < fold.test_end)].copy()
    if min(len(train), len(validation), len(test)) < 2:
        raise ValueError('Finestra insufficiente.')
    # The preceding observation is context only: only future execution rows are scored.
    validation = pd.concat([train.iloc[[-1]], validation])
    test = pd.concat([features.loc[features.index < fold.test_start].iloc[[-1]], test])
    stats_payload = None
    if normalize:
        stats = fit_normalization_stats(train, feature_columns=MARKET_FEATURE_COLUMNS)
        stats_payload = {'means': stats.means.to_dict(), 'stds': stats.stds.to_dict(),
                         'normalized_columns': list(stats.normalized_columns),
                         'excluded_columns': list(stats.excluded_columns)}
        train, validation, test = [apply_normalization(x, stats) for x in [train, validation, test]]
    return {'train': train, 'validation': validation, 'test': test,
            'train_prices': prices.loc[train.index], 'validation_prices': prices.loc[validation.index],
            'test_prices': prices.loc[test.index], 'normalization': stats_payload}


def rollout(features, prices, costs, initial_balance, *, model=None, targets=None):
    """Observe row t, execute at t+1, liquidate at the last scored close.

    Exactly one of model and targets is provided. Targets are already indexed
    by execution time (the existing baseline interface), including a context row.
    The first row is unscored context. All methods share the Phase 4 engine.
    """
    if (model is None) == (targets is None):
        raise ValueError('Specificare model oppure targets.')
    if len(features) < 2 or not features.index.equals(prices.index):
        raise ValueError('Feature e prezzi non allineati.')
    if targets is not None and not targets.index.equals(features.index):
        raise ValueError('Target non allineati.')
    if not np.isfinite(initial_balance) or initial_balance <= 0:
        raise ValueError('Capitale non positivo.')
    market = features.loc[:, MARKET_FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    price_values = prices.to_numpy(dtype=float)
    state = initialize_portfolio(initial_balance)
    records = []
    for i in range(1, len(features)):
        previous_equity = state.equity
        if model is not None:
            observation = np.concatenate([market[i-1], np.asarray([state.position, state.unrealized_pnl / state.equity], dtype=np.float32)])
            action, _ = model.predict(observation, deterministic=True)
            target = int(np.asarray(action).item()) - 1
        else:
            target = int(targets.iloc[i])
        before_costs, before_turnover = state.cumulative_costs, state.cumulative_turnover
        order = execute_order(features.index[i], float(price_values[i]), state.position, target, costs)
        state = update_portfolio(state, order, float(price_values[i]))
        orders = int(order.trade_size != 0)
        liquidation_cost = 0.0
        if i == len(features) - 1 and state.position:
            close = execute_order(features.index[i], float(price_values[i]), state.position, 0, costs)
            # Report all equity drag, including execution-price impact, not only explicit charges.
            pre_close_equity = state.equity
            state = update_portfolio(state, close, float(price_values[i]))
            liquidation_cost = pre_close_equity - state.equity
            orders += 1
        if state.equity <= 0 or not np.isfinite(state.equity):
            raise FloatingPointError('Equity non positiva: esperimento interrotto.')
        records.append({'timestamp': features.index[i], 'equity': state.equity,
                        'net_return': state.equity / previous_equity - 1,
                        'requested_position': target, 'position': state.position,
                        'explicit_cost': state.cumulative_costs - before_costs,
                        'turnover': state.cumulative_turnover - before_turnover,
                        'orders': orders, 'liquidation_equity_cost': liquidation_cost})
    return pd.DataFrame(records)


def curve_metrics(curve, initial_balance):
    equity = curve['equity'].to_numpy(dtype=float)
    values = np.r_[initial_balance, equity]
    returns = values[1:] / values[:-1] - 1
    std = float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0
    counts = curve['requested_position'].value_counts()
    return {'steps': len(curve), 'initial_equity': float(initial_balance), 'final_equity': float(equity[-1]),
            'pnl': float(equity[-1] - initial_balance), 'total_return': float(equity[-1] / initial_balance - 1),
            'max_drawdown': float(np.min(values / np.maximum.accumulate(values) - 1)),
            'reward_sum': float(returns.sum()), 'mean_return_per_bar': float(returns.mean()),
            'volatility_per_bar': std, 'sharpe_per_bar': float(returns.mean() / std) if std > 0 else 0.0,
            'explicit_costs': float(curve['explicit_cost'].sum()), 'orders': int(curve['orders'].sum()),
            'turnover': float(curve['turnover'].sum()),
            'short_share': float(counts.get(-1, 0) / len(curve)), 'flat_share': float(counts.get(0, 0) / len(curve)),
            'long_share': float(counts.get(1, 0) / len(curve))}


def stitch_curves(curves, initial_balance):
    """Concatenate already sequentially funded windows; never rebase equity."""
    result = pd.concat(curves, ignore_index=True)
    times = pd.DatetimeIndex(result['timestamp'])
    if times.has_duplicates or not times.is_monotonic_increasing:
        raise ValueError('Finestre OOS sovrapposte o non ordinate.')
    expected = np.r_[initial_balance, result['equity'].to_numpy()[:-1]]
    calculated = result['equity'].to_numpy() / expected - 1
    if not np.allclose(calculated, result['net_return'], atol=1e-12, rtol=0):
        raise ValueError('Reset di capitale o rendimento al confine dei fold.')
    return result


def fold_dependence(pnls, *, max_positive_share=.5, min_positive_fraction=.5):
    values = np.asarray(pnls, dtype=float)
    if len(values) < 3 or not np.isfinite(values).all():
        raise ValueError('Servono almeno tre risultati finiti.')
    total = float(values.sum())
    positive = values[values > 0]
    share = float(positive.max() / positive.sum()) if len(positive) else None
    leave_out = (total - values).tolist()
    enough = int((values > 0).sum()) >= int(np.ceil(len(values) * min_positive_fraction))
    passed = bool(total > 0 and enough and share is not None and share <= max_positive_share and min(leave_out) > 0)
    return {'total_pnl': total, 'positive_folds': int((values > 0).sum()),
            'max_positive_pnl_share': share, 'pnl_excluding_each_fold_attribution': leave_out,
            'pnl_excluding_best_fold_attribution': float(total - values.max()),
            'completion_passed': passed,
            'status': 'robust_positive_oos' if passed else ('non_positive_oos' if total <= 0 else 'fold_dependence_not_passed'),
            'criteria': {'max_positive_pnl_share': max_positive_share, 'min_positive_fold_fraction': min_positive_fraction,
                         'positive_total_and_all_leave_one_out_attributions_required': True},
            'note': 'Leave-one-out is PnL attribution on recorded trades, not a counterfactual policy rerun.'}
