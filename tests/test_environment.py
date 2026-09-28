import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium.utils.env_checker import check_env

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from gold_rl.data.features import MARKET_FEATURE_COLUMNS
from gold_rl.execution import ExecutionCostsConfig
from gold_rl.trading_env import TradingEnv


def make_dataset(rows: int = 80):
    index = pd.date_range("2025-01-01", periods=rows, freq="15min")
    t = np.arange(rows, dtype=float)
    features = pd.DataFrame(index=index)

    for i, column in enumerate(MARKET_FEATURE_COLUMNS):
        if column in {"hour_sin", "hour_cos"}:
            values = np.sin(t / 7.0 + i)
        else:
            values = 0.01 * np.sin(t / (3.0 + i)) + 0.001 * i
        features[column] = values

    prices = pd.Series(
        1900.0 + np.cumsum(0.2 + 0.05 * np.sin(t / 5.0)),
        index=index,
        name="Close",
    )
    return features, prices


def make_env(mode="train", max_episode_steps=20, random_start=None):
    features, prices = make_dataset()
    costs = ExecutionCostsConfig(
        spread_rate=0.0002,
        commission_rate=0.0001,
        slippage_rate=0.00005,
        turnover_penalty=0.0001,
    )
    return TradingEnv(
        features=features,
        mid_prices=prices,
        execution_config=costs,
        initial_balance=100_000.0,
        mode=mode,
        max_episode_steps=max_episode_steps,
        random_start=random_start,
    )


def test_action_and_observation_spaces():
    env = make_env()
    assert env.action_space == gym.spaces.Discrete(3)
    assert env.observation_space.shape == (len(MARKET_FEATURE_COLUMNS) + 2,)

    obs, _ = env.reset(seed=42)
    assert obs.shape == env.observation_space.shape
    assert env.observation_space.contains(obs)


def test_action_mapping():
    assert TradingEnv.action_to_position(0) == -1
    assert TradingEnv.action_to_position(1) == 0
    assert TradingEnv.action_to_position(2) == 1


def test_random_training_episodes_are_seedable():
    env = make_env(mode="train", max_episode_steps=10)
    _, info_1 = env.reset(seed=123)
    start_1 = info_1["timestamp"]

    _, info_2 = env.reset(seed=123)
    start_2 = info_2["timestamp"]

    assert start_1 == start_2


def test_validation_and_test_are_chronological():
    for mode in ("validation", "test"):
        env = make_env(mode=mode, max_episode_steps=None)
        _, info = env.reset(seed=999)
        assert info["step"] == 0
        assert info["timestamp"] == env.features.index[0]

        previous_timestamp = info["timestamp"]
        for _ in range(5):
            _, _, terminated, truncated, info = env.step(1)
            assert info["timestamp"] > previous_timestamp
            previous_timestamp = info["timestamp"]
            assert not truncated
            if terminated:
                break


def test_step_executes_at_next_bar_without_out_of_index():
    env = make_env(max_episode_steps=3)
    obs, _ = env.reset(seed=7)

    for expected_step in range(1, 4):
        obs, reward, terminated, truncated, info = env.step(2)
        assert np.isfinite(obs).all()
        assert np.isfinite(reward)
        assert info["step"] == expected_step
        if expected_step < 3:
            assert not terminated
            assert not truncated
        else:
            assert terminated

    try:
        env.step(2)
    except RuntimeError:
        pass
    else:
        raise AssertionError("step() dopo la fine dell'episodio deve fallire.")


def test_random_agent_completes_whole_episode():
    env = make_env(max_episode_steps=25)
    obs, _ = env.reset(seed=42)
    rewards = []

    while True:
        assert np.isfinite(obs).all()
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        assert np.isfinite(obs).all()
        assert np.isfinite(reward)
        assert np.isfinite(info["equity"])
        if terminated or truncated:
            break

    assert len(rewards) == 25


def test_gymnasium_check_env():
    env = make_env(max_episode_steps=10, random_start=False)
    check_env(env, skip_render_check=True)
