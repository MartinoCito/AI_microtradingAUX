import copy
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
import train_ppo as training
from gold_rl.rl.persistence import PortablePPO, load_ppo
from gold_rl.rl.callbacks import PPOTrainingMetricsCallback


def dataset(rows=500):
    x = np.arange(rows)
    close = 1900 + x * .05 + np.sin(x / 7)
    return pd.DataFrame({'Open': close - .1 * np.cos(x / 3), 'High': close + .4 + .1 * np.sin(x / 4), 'Low': close - .3 - .1 * np.cos(x / 5),
                         'Close': close, 'Volume': 100 + x % 17},
                        index=pd.date_range('2020-01-01', periods=rows, freq='15min'))


def test_subset_normalization_does_not_see_unused_future(tmp_path):
    config = training.load_config(ROOT / 'config.yaml')
    path = tmp_path / 'm15.parquet'
    original = dataset()
    original.to_parquet(path)
    before = training.prepare_data(config, path, return_metadata=True)
    changed = original.copy()
    changed.iloc[200:, :4] *= 2
    changed.to_parquet(path)
    after = training.prepare_data(config, path, return_metadata=True)
    pd.testing.assert_frame_equal(before[0], after[0])
    assert before[4]['normalization'] == after[4]['normalization']
    means = before[0].drop(columns=['hour_sin', 'hour_cos']).mean()
    np.testing.assert_allclose(means, 0, atol=1e-10)
    assert before[0].index[-1] < before[1].index[0]


def test_portable_models_load_with_native_ppo_and_preserve_optimizer(tmp_path):
    torch.set_num_threads(1)
    model = PortablePPO('MlpPolicy', 'CartPole-v1', seed=42, n_steps=16, batch_size=8, n_epochs=1)
    model.learn(32)
    path = tmp_path / 'portable.zip'
    model.save(path)
    restored = PPO.load(path)
    assert training.policy_digest(model) == training.policy_digest(restored)
    assert restored.num_timesteps == 32
    for key, value in model.policy.optimizer.state_dict()['state'].items():
        for field, tensor in value.items():
            torch.testing.assert_close(tensor, restored.policy.optimizer.state_dict()['state'][key][field], rtol=0, atol=0)
    old = io.BytesIO()
    PPO.save(model, old)
    old.seek(0)
    compatible = load_ppo(old)
    assert training.policy_digest(model) == training.policy_digest(compatible)
    model.get_env().close()


def test_threshold_is_used_by_callback(tmp_path):
    callback = PPOTrainingMetricsCallback(tmp_path / 'metrics.json', degeneracy_threshold=.8)
    callback.action_counts = np.array([85, 10, 5])
    callback._on_training_end()
    assert json.loads(callback.output_path.read_text())['degenerate']


def test_repeat_checks_weights_and_does_not_accept_one_seed():
    item = {'seed': 42, 'label': 'seed_42', 'policy_sha256': 'a', 'training': {},
            'validation': {'action_distribution': {'short': .3, 'flat': .3, 'long': .4}},
            'comparison': {'validation': {'mean_window_return': 0}}, 'actual_timesteps': 32,
            'non_degenerate': True}
    repeat = copy.deepcopy(item)
    repeat['label'] += '_repeat'
    criteria = {'max_mean_window_return_range': .1, 'max_action_share_range': .2}
    result = training.aggregate_results([item, repeat], criteria)
    assert result['repeat_verified'] and not result['completion_passed']
    runs = [item]
    for seed in [43, 44]:
        other = copy.deepcopy(item)
        other.update(seed=seed, label=f'seed_{seed}')
        runs.append(other)
    assert training.aggregate_results(runs + [repeat], criteria)['completion_passed']
    repeat['policy_sha256'] = 'different'
    assert not training.aggregate_results(runs + [repeat], criteria)['completion_passed']


def test_training_repeat_and_equal_window_protocol(tmp_path):
    config = training.load_config(ROOT / 'config.yaml')
    config['agent'].update(n_steps=16, batch_size=8, n_epochs=1, comparison_steps=8, policy_hidden_layers=[16])
    config['environment']['max_episode_steps'] = 16
    path = tmp_path / 'data.parquet'
    dataset().to_parquet(path)
    train, validation, tp, vp = training.prepare_data(config, path)
    results = []
    for label in ['seed_42', 'seed_42_repeat']:
        results.append(training.train_one_seed(seed=42, label=label, config=config,
            train_features=train, validation_features=validation, train_prices=tp, validation_prices=vp,
            output_root=tmp_path, run_id='test', timesteps=32, eval_freq=16, checkpoint_freq=16, device='cpu'))
    assert training.aggregate_results(results, config['agent']['acceptance'])['repeat_verified']
    for result in results:
        for split in ['train', 'validation']:
            windows = result['comparison'][split]['windows']
            assert all(w['steps'] == 8 for w in windows)
            expected = sum(w['episode_reward'] for w in windows) / (8 * len(windows))
            assert result['comparison'][split]['reward_per_step'] == expected
        assert result['native_reload_verified']
    checkpoint = tmp_path / 'logs/prototype/test/seed_42/checkpoints/ppo_16_steps.zip'
    assert PPO.load(checkpoint).num_timesteps == 16
    with pytest.raises(FileExistsError):
        training.train_one_seed(seed=42, label='seed_42', config=config,
            train_features=train, validation_features=validation, train_prices=tp, validation_prices=vp,
            output_root=tmp_path, run_id='test', timesteps=32, eval_freq=16, checkpoint_freq=16, device='cpu')
