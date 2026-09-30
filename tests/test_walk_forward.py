import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts'), str(ROOT)]
from gold_rl.data.features import MARKET_FEATURE_COLUMNS
from gold_rl.execution import ExecutionCostsConfig
from gold_rl.rl.walk_forward import make_folds, prepare_fold, rollout, curve_metrics, stitch_curves, fold_dependence
from gold_rl.trading_env import TradingEnv
from train_walk_forward import train_fold
from train_ppo import load_config, policy_digest


def frames():
    index = pd.date_range('2010-01-01', '2020-01-01', freq='D')
    x = np.arange(len(index))
    f = pd.DataFrame({name: np.sin(x / (i+3)) + x * .0001 for i, name in enumerate(MARKET_FEATURE_COLUMNS)}, index=index)
    p = pd.Series(100 + x * .03, index=index)
    return f, p


def folds_for(f):
    return make_folds(f.index, n_folds=3, train_years=2, validation_months=6, test_months=6, holdout_start='2019-08-31')


def test_fold_boundaries_and_holdout_are_disjoint():
    f, p = frames()
    folds = folds_for(f)
    all_times = []
    for i, fold in enumerate(folds):
        d = prepare_fold(f, p, fold)
        assert d['train'].index.max() < d['validation'].index[1]
        assert d['validation'].index.max() < d['test'].index[1]
        assert d['test'].index.max() < pd.Timestamp('2019-08-31')
        assert d['validation'].index[0] == d['train'].index[-1]
        assert d['test'].index[0] == d['validation'].index[-1]
        if i:
            assert folds[i-1].test_end == fold.test_start
        all_times.extend(d['test'].index[1:])
    assert len(set(all_times)) == len(all_times)


def test_scaler_train_and_validation_do_not_see_test_changes():
    f, p = frames()
    fold = folds_for(f)[0]
    before = prepare_fold(f, p, fold)
    changed = f.copy()
    changed.loc[changed.index >= fold.test_start] += 100
    after = prepare_fold(changed, p, fold)
    assert before['normalization'] == after['normalization']
    pd.testing.assert_frame_equal(before['train'], after['train'])
    pd.testing.assert_frame_equal(before['validation'], after['validation'])


class LongPolicy:
    def __init__(self):
        self.observations = []

    def predict(self, obs, deterministic=True):
        self.observations.append(obs.copy())
        return np.array(2), None


def test_rollout_matches_environment_then_liquidates_with_costs():
    f, p = frames(); f, p = f.iloc[:8], p.iloc[:8]
    costs = ExecutionCostsConfig(.001, .001, .001, .001)
    model = LongPolicy()
    curve = rollout(f, p, costs, 1000, model=model)
    env = TradingEnv(f, p, costs, initial_balance=1000, mode='validation', random_start=False, max_episode_steps=None)
    obs, _ = env.reset(seed=0)
    expected = []
    for i in range(len(f)-1):
        np.testing.assert_array_equal(obs, model.observations[i])
        obs, _, _, _, info = env.step(2)
        expected.append(info['equity'])
    np.testing.assert_allclose(curve['equity'].iloc[:-1], expected[:-1], rtol=0, atol=1e-12)
    assert curve['position'].iloc[-1] == 0
    assert curve['liquidation_equity_cost'].iloc[-1] > 0
    assert curve['equity'].iloc[-1] < expected[-1]
    assert curve['timestamp'].tolist() == f.index[1:].tolist()


def test_stitch_carries_cash_and_rejects_resets():
    f, p = frames(); f, p = f.iloc[:8], p.iloc[:8]
    zero = ExecutionCostsConfig(0, 0, 0, 0)
    first = rollout(f.iloc[:4], p.iloc[:4], zero, 1000, model=LongPolicy())
    second = rollout(f.iloc[3:], p.iloc[3:], zero, first['equity'].iloc[-1], model=LongPolicy())
    joined = stitch_curves([first, second], 1000)
    assert joined['equity'].iloc[-1] == second['equity'].iloc[-1]
    reset = rollout(f.iloc[3:], p.iloc[3:], zero, 1000, model=LongPolicy())
    with pytest.raises(ValueError, match='Reset'):
        stitch_curves([first, reset], 1000)
    with pytest.raises(ValueError, match='sovrapposte'):
        stitch_curves([first, first], 1000)


def test_drawdown_includes_first_execution_cost():
    f, p = frames(); f, p = f.iloc[:3], p.iloc[:3]
    curve = rollout(f, p, ExecutionCostsConfig(.01, .01, .01, .01), 1000, model=LongPolicy())
    metrics = curve_metrics(curve, 1000)
    assert metrics['max_drawdown'] < 0
    assert metrics['orders'] == 2
    assert metrics['turnover'] == 2


def test_profit_attribution_does_not_accept_single_lucky_fold_or_losses():
    assert fold_dependence([10, 8, 9, 7])['completion_passed']
    assert not fold_dependence([100, 1, 1, -10])['completion_passed']
    assert not fold_dependence([-3, -2, -1])['completion_passed']
    assert fold_dependence([-3, -2, -1])['status'] == 'non_positive_oos'


def test_selected_checkpoint_is_validation_maximum_and_loads(tmp_path):
    f, p = frames(); data = prepare_fold(f, p, folds_for(f)[0])
    config = load_config(ROOT / 'config.yaml')
    config['agent'].update(n_steps=16, batch_size=8, n_epochs=1, policy_hidden_layers=[16])
    config['environment']['max_episode_steps'] = 16
    config['walk_forward'].update(total_timesteps=32, eval_freq=16, checkpoint_freq=16)
    result = train_fold(number=1, config=config, train=data['train'], train_prices=data['train_prices'],
        validation=data['validation'], validation_prices=data['validation_prices'],
        model_dir=tmp_path / 'model', log_dir=tmp_path / 'log')
    scores = pd.read_csv(tmp_path / 'model/validation_scores.csv')
    assert scores.iloc[-1]['stage'] == 'final'
    assert result['validation_return'] == pytest.approx(scores['total_return'].max())
    restored = PPO.load(result['model_path'], device='cpu')
    assert policy_digest(restored) == result['policy_sha256']
    assert result['test_used'] is False
    assert (tmp_path / 'log/checkpoints/ppo_16_steps.zip').exists()

def test_baselines_execute_the_previous_bar_signal_without_double_shift():
    from baselines import build_target_positions
    f, _ = frames(); f = f.iloc[:6]
    prices = pd.Series([100., 101., 99., 105., 104., 103.], index=f.index)
    targets = build_target_positions(prices.to_frame('Close'))['momentum']
    curve = rollout(f, prices, ExecutionCostsConfig(0, 0, 0, 0), 1000, targets=targets)
    assert curve['requested_position'].tolist() == [0, 1, -1, 1, -1]
    assert curve['timestamp'].tolist() == f.index[1:].tolist()
