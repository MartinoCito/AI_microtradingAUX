from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from gold_rl.data.features import MARKET_FEATURE_COLUMNS
from gold_rl.execution import ExecutionCostsConfig, execute_order
from gold_rl.portfolio import PortfolioState, initialize_portfolio, update_portfolio


ACTION_TO_POSITION = {0: -1, 1: 0, 2: 1}
POSITION_TO_ACTION = {-1: 0, 0: 1, 1: 2}
DEFAULT_MARKET_FEATURES = tuple(MARKET_FEATURE_COLUMNS)


@dataclass(frozen=True)
class EpisodeBounds:
    start: int
    end: int


class TradingEnv(gym.Env):
    """Causal M15 trading environment using the Phase 4 execution engine."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        features: pd.DataFrame,
        mid_prices: pd.Series,
        execution_config: ExecutionCostsConfig,
        initial_balance: float = 100_000.0,
        mode: str = "train",
        max_episode_steps: int | None = 5000,
        market_feature_columns: Sequence[str] = DEFAULT_MARKET_FEATURES,
        random_start: bool | None = None,
        turnover_reward_weight: float = 0.0,
    ) -> None:
        super().__init__()

        if mode not in {"train", "validation", "test"}:
            raise ValueError("mode deve essere 'train', 'validation' o 'test'.")
        if len(features) != len(mid_prices) or len(features) < 2:
            raise ValueError("features e mid_prices devono avere stessa lunghezza e almeno 2 righe.")
        if not isinstance(features.index, pd.DatetimeIndex):
            raise TypeError("features deve avere un DatetimeIndex.")
        if not features.index.is_monotonic_increasing:
            raise ValueError("features deve essere ordinato cronologicamente.")
        if not mid_prices.index.equals(features.index):
            raise ValueError("features e mid_prices devono avere lo stesso indice.")

        columns = tuple(market_feature_columns)
        missing = [column for column in columns if column not in features.columns]
        if missing:
            raise ValueError(f"Feature M15 mancanti: {missing}")

        market = features.loc[:, columns].astype("float64")
        prices = mid_prices.astype("float64")
        if not np.isfinite(market.to_numpy()).all():
            raise ValueError("Le feature M15 contengono NaN o valori infiniti.")
        if not np.isfinite(prices.to_numpy()).all() or (prices <= 0).any():
            raise ValueError("mid_prices deve contenere prezzi positivi e finiti.")
        if max_episode_steps is not None and max_episode_steps < 1:
            raise ValueError("max_episode_steps deve essere >= 1 oppure None.")
        if turnover_reward_weight < 0:
            raise ValueError("turnover_reward_weight deve essere >= 0.")
        if random_start and mode != "train":
            raise ValueError("random_start=True è consentito solo in training.")

        self.mode = mode
        self.features = market
        self.mid_prices = prices
        self.market_feature_columns = columns
        self.execution_config = execution_config
        self.initial_balance = float(initial_balance)
        self.max_episode_steps = max_episode_steps
        self.turnover_reward_weight = float(turnover_reward_weight)
        self.random_start = (mode == "train") if random_start is None else bool(random_start)

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=np.full(len(columns) + 2, -np.inf, dtype=np.float32),
            high=np.full(len(columns) + 2, np.inf, dtype=np.float32),
            dtype=np.float32,
        )

        self._episode: EpisodeBounds | None = None
        self._index: int | None = None
        self._steps = 0
        self._portfolio: PortfolioState | None = None
        self._terminated = False
        self._episode_time_limited = False

    @staticmethod
    def action_to_position(action: int) -> int:
        action = int(action)
        if action not in ACTION_TO_POSITION:
            raise ValueError(f"Azione non valida: {action}")
        return ACTION_TO_POSITION[action]

    def _sample_episode(self) -> EpisodeBounds:
        data_end = len(self.features) - 1
        if self.random_start:
            if self.max_episode_steps is None:
                start = int(self.np_random.integers(0, data_end))
                return EpisodeBounds(start=start, end=data_end)
            latest_start = data_end - self.max_episode_steps
            if latest_start < 0:
                raise ValueError("Dataset troppo corto per max_episode_steps.")
            start = int(self.np_random.integers(0, latest_start + 1))
            return EpisodeBounds(start=start, end=start + self.max_episode_steps)

        start = 0
        end = data_end
        if self.max_episode_steps is not None:
            end = min(start + self.max_episode_steps, data_end)
        return EpisodeBounds(start=start, end=end)

    def _get_observation(self) -> np.ndarray:
        if self._index is None or self._portfolio is None:
            raise RuntimeError("Environment non inizializzato: chiamare reset().")

        market = self.features.iloc[self._index].to_numpy(dtype=np.float32)
        equity = float(self._portfolio.equity)
        unrealized_return = float(self._portfolio.unrealized_pnl / equity) if equity else 0.0
        observation = np.concatenate(
            [market, np.asarray([self._portfolio.position, unrealized_return], dtype=np.float32)]
        ).astype(np.float32)

        if not np.isfinite(observation).all():
            raise FloatingPointError("Observation non-finita.")
        return observation

    def _get_info(self, execution=None, previous_equity: float | None = None, reward: float | None = None) -> dict:
        if self._index is None or self._portfolio is None:
            return {}
        info = {
            "timestamp": self.features.index[self._index],
            "step": int(self._steps),
            "position": int(self._portfolio.position),
            "equity": float(self._portfolio.equity),
            "cash": float(self._portfolio.cash),
            "unrealized_pnl": float(self._portfolio.unrealized_pnl),
            "cumulative_costs": float(self._portfolio.cumulative_costs),
            "cumulative_turnover": float(self._portfolio.cumulative_turnover),
        }
        if execution is not None:
            info.update({
                "trade_size": int(execution.trade_size),
                "turnover": float(execution.turnover),
                "total_cost": float(execution.total_cost),
                "turnover_cost": float(execution.turnover_cost),
            })
        if previous_equity is not None:
            info["net_return"] = float(self._portfolio.equity / previous_equity - 1.0)
        if reward is not None:
            info["reward"] = float(reward)
        return info

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if options:
            unsupported = set(options) - {"start_index", "end_index"}
            if unsupported:
                raise ValueError(f"Opzioni non supportate: {sorted(unsupported)}")

        episode = self._sample_episode()
        if options and "start_index" in options:
            start = int(options["start_index"])
            end = int(options.get("end_index", episode.end))
            if start < 0 or start >= len(self.features) - 1 or end <= start or end >= len(self.features):
                raise ValueError("start_index/end_index non validi.")
            episode = EpisodeBounds(start=start, end=end)

        data_end = len(self.features) - 1
        self._episode = episode
        self._episode_time_limited = episode.end < data_end
        self._index = episode.start
        self._steps = 0
        self._portfolio = initialize_portfolio(self.initial_balance)
        self._terminated = False

        return self._get_observation(), self._get_info()

    def step(self, action: int):
        if self._episode is None or self._index is None or self._portfolio is None:
            raise RuntimeError("reset() deve essere chiamato prima di step().")
        if self._terminated:
            raise RuntimeError("L'episodio è terminato: chiamare reset().")

        target_position = self.action_to_position(action)
        next_index = self._index + 1
        if next_index > self._episode.end:
            raise RuntimeError("Tentativo di accesso oltre la fine dell'episodio.")

        previous_equity = float(self._portfolio.equity)
        execution = execute_order(
            timestamp=self.features.index[next_index],
            mid_price=float(self.mid_prices.iloc[next_index]),
            previous_position=int(self._portfolio.position),
            target_position=target_position,
            config=self.execution_config,
        )
        self._portfolio = update_portfolio(
            previous_state=self._portfolio,
            execution=execution,
            mid_price=float(self.mid_prices.iloc[next_index]),
        )

        if previous_equity <= 0.0 or not np.isfinite(previous_equity):
            raise FloatingPointError("Equity precedente non valida.")

        net_return = float(self._portfolio.equity / previous_equity - 1.0)
        reward = float(net_return - self.turnover_reward_weight * execution.turnover)
        if not np.isfinite(reward) or not np.isfinite(self._portfolio.equity):
            raise FloatingPointError("Reward/equity non finita.")

        self._index = next_index
        self._steps += 1

        reached_end = self._index >= self._episode.end
        reached_data_end = self._index >= len(self.features) - 1
        terminated = bool(reached_end and reached_data_end)
        truncated = bool(reached_end and self._episode_time_limited and not terminated)
        self._terminated = terminated or truncated

        observation = self._get_observation()
        info = self._get_info(execution=execution, previous_equity=previous_equity, reward=reward)
        return observation, reward, terminated, truncated, info

    def render(self):
        return None

    def close(self):
        return None
