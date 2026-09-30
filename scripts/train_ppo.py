from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from gold_rl.data.features import (MARKET_FEATURE_COLUMNS, apply_normalization,
                                   create_market_features, fit_normalization_stats)
from gold_rl.execution import ExecutionCostsConfig
from gold_rl.rl.callbacks import build_ppo_callbacks
from gold_rl.rl.persistence import PortablePPO
from gold_rl.trading_env import TradingEnv

ACTION_NAMES = {0: 'short', 1: 'flat', 2: 'long'}


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def parse_args():
    p = argparse.ArgumentParser(description='Fase 7: PPO preliminare riproducibile')
    p.add_argument('--config', type=Path, default=ROOT / 'config.yaml')
    p.add_argument('--m15-path', type=Path)
    for name in ['timesteps', 'eval-freq', 'checkpoint-freq']:
        p.add_argument('--' + name, type=int)
    p.add_argument('--train-fraction', type=float)
    p.add_argument('--seeds', type=int, nargs='+')
    p.add_argument('--device')
    p.add_argument('--run-id', default=None)
    p.add_argument('--workers', type=int, default=1)
    p.add_argument('--skip-repeat', action='store_true', help='Diagnostic only: completion cannot pass without a repeat')
    return p.parse_args()


def load_config(path: Path):
    return yaml.safe_load(path.read_text(encoding='utf-8'))


def prepare_data(config, m15_path=None, train_fraction=None, *, return_metadata=False):
    data_cfg = config['data']
    fraction = float(train_fraction if train_fraction is not None else config['agent'].get('train_fraction', .25))
    if not 0 < fraction <= 1:
        raise ValueError('train_fraction deve essere > 0 e <= 1.')
    ratios = [float(data_cfg[k]) for k in ['train_ratio', 'validation_ratio', 'test_ratio']]
    if any(v <= 0 for v in ratios) or not np.isclose(sum(ratios), 1):
        raise ValueError('Gli split devono essere positivi e sommare a 1.')
    path = Path(m15_path or ROOT / data_cfg['m15_path']).resolve()
    ohlcv = pd.read_parquet(path)
    if not isinstance(ohlcv.index, pd.DatetimeIndex):
        raise TypeError('M15 deve avere DatetimeIndex.')
    if not ohlcv.index.is_monotonic_increasing or ohlcv.index.has_duplicates:
        raise ValueError('M15 deve essere ordinato e senza duplicati.')
    features = create_market_features(ohlcv, drop_warmup=True, strict_windows=True).loc[:, MARKET_FEATURE_COLUMNS]
    prices = ohlcv.loc[features.index, 'Close'].astype('float64')
    if not np.isfinite(features.to_numpy()).all() or not np.isfinite(prices).all() or (prices <= 0).any():
        raise ValueError('Feature o prezzi non validi.')
    train_end = int(len(features) * ratios[0])
    validation_end = train_end + int(len(features) * ratios[1])
    subset_end = int(train_end * fraction)
    train = features.iloc[:subset_end].copy()
    validation = features.iloc[train_end:validation_end].copy()
    if min(len(train), len(validation)) < 2:
        raise ValueError('Split troppo corti.')
    def bounds(frame):
        return {'rows': len(frame), 'start': str(frame.index[0]), 'end': str(frame.index[-1])}
    metadata = {'m15_path': str(path), 'm15_sha256': sha256(path), 'feature_columns': list(MARKET_FEATURE_COLUMNS),
                'train_fraction': fraction, 'train': bounds(train), 'validation': bounds(validation),
                'full_train_rows': train_end, 'test_used': False, 'normalization': None}
    # Fit only on the actual subset, before any validation data is transformed.
    if config['features'].get('normalize', True):
        stats = fit_normalization_stats(train, feature_columns=MARKET_FEATURE_COLUMNS)
        metadata['normalization'] = {'means': stats.means.to_dict(), 'stds': stats.stds.to_dict(),
                                     'normalized_columns': list(stats.normalized_columns),
                                     'excluded_columns': list(stats.excluded_columns)}
        train = apply_normalization(train, stats)
        validation = apply_normalization(validation, stats)
    result = (train, validation, prices.loc[train.index].copy(), prices.loc[validation.index].copy())
    return (*result, metadata) if return_metadata else result


def build_env_factory(features, prices, environment_cfg, mode, seed):
    costs = ExecutionCostsConfig(**{k: float(environment_cfg[k]) for k in
        ['spread_rate', 'commission_rate', 'slippage_rate', 'turnover_penalty']})
    def factory():
        env = TradingEnv(features=features, mid_prices=prices, execution_config=costs,
                         initial_balance=float(environment_cfg['initial_balance']), mode=mode,
                         max_episode_steps=int(environment_cfg['max_episode_steps']) if mode == 'train' else None,
                         random_start=(mode == 'train'), turnover_reward_weight=0.0)
        env.reset(seed=seed)
        return Monitor(env)
    return factory


def evaluate_deterministic(model, env, degeneracy_threshold=.99):
    obs, initial = env.reset(seed=0)
    initial_equity = float(initial['equity'])
    counts = np.zeros(3, dtype=np.int64)
    total_reward = 0.0
    while True:
        action, _ = model.predict(obs, deterministic=True)
        action = int(np.asarray(action).reshape(-1)[0])
        counts[action] += 1
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        if terminated or truncated:
            break
    steps = int(counts.sum())
    distribution = counts / steps
    return {'episode_reward': total_reward, 'reward_per_step': total_reward / steps,
            'final_equity': float(info['equity']), 'net_return': float(info['equity'] / initial_equity - 1),
            'steps': steps, 'action_counts': {ACTION_NAMES[i]: int(counts[i]) for i in range(3)},
            'action_distribution': {ACTION_NAMES[i]: float(distribution[i]) for i in range(3)},
            'degenerate': bool(distribution.max() >= degeneracy_threshold),
            'final_position': int(info['position']), 'final_timestamp': str(info['timestamp'])}


def evaluate_windows(model, features, prices, config):
    length = int(config['agent']['comparison_steps'])
    threshold = float(config['agent']['degeneracy_threshold'])
    results = []
    # Share a boundary observation, not a return; reset portfolio at every window.
    for start in range(0, len(features) - length, length):
        frame = features.iloc[start:start + length + 1]
        env = build_env_factory(frame, prices.loc[frame.index], config['environment'], 'validation', 0)()
        result = evaluate_deterministic(model, env, threshold)
        result['start_timestamp'] = str(frame.index[0])
        results.append(result)
        env.close()
    if not results:
        raise ValueError('Nessuna finestra completa per il confronto.')
    counts = {a: sum(r['action_counts'][a] for r in results) for a in ACTION_NAMES.values()}
    steps = sum(counts.values())
    return {'window_steps': length, 'windows': results, 'window_count': len(results),
            'unused_tail_steps': (len(features) - 1) % length,
            'reward_per_step': sum(r['episode_reward'] for r in results) / steps,
            'mean_window_return': float(np.mean([r['net_return'] for r in results])),
            'std_window_return': float(np.std([r['net_return'] for r in results], ddof=0)),
            'action_distribution': {a: n / steps for a, n in counts.items()}}


def summarize_training_reward(path):
    payload = json.loads(Path(path).read_text(encoding='utf-8'))
    rewards = payload['training_episode_rewards']
    return {'episodes': rewards['count'], 'mean_reward': rewards['mean'], 'last_reward': rewards['last']}


def policy_digest(model):
    h = hashlib.sha256()
    for name, tensor in sorted(model.policy.state_dict().items()):
        h.update(name.encode())
        h.update(tensor.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def train_one_seed(*, seed, label, config, train_features, validation_features,
                   train_prices, validation_prices, output_root, run_id, timesteps,
                   eval_freq, checkpoint_freq, device):
    agent = config['agent']
    torch.set_num_threads(int(agent['torch_threads']))
    torch.use_deterministic_algorithms(True)
    set_random_seed(seed)
    model_dir = output_root / 'models/prototype' / run_id / label
    log_dir = output_root / 'logs/prototype' / run_id / label
    report_dir = output_root / 'reports/ppo_prototype' / run_id / label
    for folder in [model_dir, log_dir, report_dir]:
        folder.mkdir(parents=True, exist_ok=False)
    train_env = DummyVecEnv([build_env_factory(train_features, train_prices, config['environment'], 'train', seed)])
    validation_env = DummyVecEnv([build_env_factory(validation_features, validation_prices, config['environment'], 'validation', seed)])
    threshold = float(agent['degeneracy_threshold'])
    evaluation = EvalCallback(validation_env, best_model_save_path=str(model_dir / 'best'),
                              log_path=str(log_dir / 'eval'), eval_freq=eval_freq,
                              n_eval_episodes=1, deterministic=True, verbose=0)
    callbacks = build_ppo_callbacks(output_dir=log_dir, checkpoint_freq=checkpoint_freq,
                                    eval_callback=evaluation, degeneracy_threshold=threshold)
    model = PortablePPO('MlpPolicy', train_env, seed=seed, device=device, verbose=0,
        policy_kwargs={'net_arch': {'pi': agent['policy_hidden_layers'], 'vf': agent['policy_hidden_layers']}},
        **{k: agent[k] for k in ['learning_rate', 'n_steps', 'batch_size', 'n_epochs', 'gamma',
                                 'gae_lambda', 'clip_range', 'ent_coef', 'vf_coef', 'max_grad_norm']})
    model.set_logger(configure(str(log_dir), ['csv', 'tensorboard']))
    try:
        model.learn(total_timesteps=timesteps, callback=callbacks)
        final_path = model_dir / 'final_model.zip'
        model.save(final_path)
        # Exercise the ordinary public loader, including optimizer tensors.
        reloaded = PPO.load(final_path, device=device)
        if policy_digest(reloaded) != policy_digest(model):
            raise AssertionError('I pesi cambiano dopo il salvataggio.')
        env = build_env_factory(validation_features, validation_prices, config['environment'], 'validation', seed)()
        validation = evaluate_deterministic(reloaded, env, threshold)
        env.close()
        train_comparison = evaluate_windows(reloaded, train_features, train_prices, config)
        validation_comparison = evaluate_windows(reloaded, validation_features, validation_prices, config)
        best_path = model_dir / 'best/best_model.zip'
        best = None
        if best_path.exists():
            best_model = PPO.load(best_path, device=device)
            env = build_env_factory(validation_features, validation_prices, config['environment'], 'validation', seed)()
            best = evaluate_deterministic(best_model, env, threshold)
            best['model_timesteps'] = int(best_model.num_timesteps)
            env.close()
        summary = {'seed': seed, 'label': label, 'nominal_timesteps': timesteps,
            'actual_timesteps': int(model.num_timesteps), 'training': summarize_training_reward(log_dir / 'training_action_metrics.json'),
            'validation': validation, 'best_validation': best,
            'comparison': {'protocol': 'deterministic, reset portfolio, equal-length non-overlapping return windows',
                           'train': train_comparison, 'validation': validation_comparison},
            'non_degenerate': not validation['degenerate'], 'policy_sha256': policy_digest(model),
            'native_reload_verified': True, 'final_model': str(final_path.relative_to(output_root)),
            'final_model_sha256': sha256(final_path)}
        write_json(report_dir / 'summary.json', summary)
        print(f'Completed {label}: {model.num_timesteps} steps', flush=True)
        return summary
    finally:
        train_env.close()
        validation_env.close()


def aggregate_results(runs, criteria):
    primary = [r for r in runs if not r['label'].endswith('_repeat')]
    repeated = next((r for r in runs if r['label'].endswith('_repeat')), None)
    original = next((r for r in primary if repeated and r['seed'] == repeated['seed']), None)
    repeat_ok = bool(original and repeated and all(original[key] == repeated[key] for key in
                      ['policy_sha256', 'training', 'validation', 'comparison', 'actual_timesteps']))
    returns = [r['comparison']['validation']['mean_window_return'] for r in primary]
    return_range = float(np.ptp(returns))
    action_range = max(float(np.ptp([r['validation']['action_distribution'][a] for r in primary])) for a in ACTION_NAMES.values())
    enough_seeds = len({r['seed'] for r in primary}) >= 3
    variability_ok = bool(enough_seeds and return_range <= criteria['max_mean_window_return_range']
                          and action_range <= criteria['max_action_share_range'])
    non_degenerate = all(r['non_degenerate'] for r in primary)
    return {'repeat_verified': repeat_ok, 'distinct_seeds': len(primary),
            'mean_window_return_range': return_range, 'max_action_share_range': action_range,
            'variability_acceptable': variability_ok, 'all_validation_runs_non_degenerate': non_degenerate,
            'completion_passed': bool(repeat_ok and variability_ok and non_degenerate),
            'criteria': criteria, 'note': 'Engineering tolerances for the prototype, not evidence of profitability.'}


def main():
    args = parse_args()
    config = load_config(args.config)
    agent = config['agent']
    for name in ['timesteps', 'train_fraction', 'seeds', 'eval_freq', 'checkpoint_freq', 'device']:
        value = getattr(args, name)
        if value is not None:
            agent['total_timesteps' if name == 'timesteps' else name] = value
    timesteps = int(agent['total_timesteps'])
    if min(timesteps, int(agent['eval_freq']), int(agent['checkpoint_freq']), args.workers,
           int(agent['comparison_steps']), int(agent['torch_threads'])) <= 0:
        raise ValueError('Budget, frequenze, finestre e worker devono essere positivi.')
    if not 1/3 < float(agent['degeneracy_threshold']) <= 1:
        raise ValueError('Soglia di degenerazione non valida.')
    if len(set(agent['seeds'])) != len(agent['seeds']) or not agent['seeds']:
        raise ValueError('Usare seed distinti; la ripetizione viene gestita separatamente.')
    run_id = args.run_id or datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    if not run_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in run_id):
        raise ValueError('run-id deve contenere solo lettere, numeri, trattini e underscore.')
    for parent in ['models/prototype', 'logs/prototype', 'reports/ppo_prototype']:
        if (ROOT / parent / run_id).exists():
            raise FileExistsError(f'Esperimento già presente: {run_id}')
    train, validation, tp, vp, data = prepare_data(config, args.m15_path, return_metadata=True)
    if len(train) <= int(config['environment']['max_episode_steps']):
        raise ValueError('Sottoinsieme troppo corto per un episodio di training.')
    if min(len(train), len(validation)) <= int(agent['comparison_steps']):
        raise ValueError('Split troppo corti per il confronto.')
    report_dir = ROOT / 'reports/ppo_prototype' / run_id
    model_dir = ROOT / 'models/prototype' / run_id
    source_files = [ROOT / 'scripts/train_ppo.py', ROOT / 'src/gold_rl/rl/callbacks.py',
                    ROOT / 'src/gold_rl/rl/persistence.py', ROOT / 'src/gold_rl/trading_env.py',
                    ROOT / 'src/gold_rl/data/features.py', ROOT / 'src/gold_rl/execution.py', ROOT / 'src/gold_rl/portfolio.py']
    manifest = {'run_id': run_id, 'created_utc': datetime.now(timezone.utc).isoformat(),
        'config': config, 'data': data, 'python': platform.python_version(), 'platform': platform.platform(),
        'versions': {name: importlib.metadata.version(name) for name in ['numpy', 'pandas', 'torch', 'stable-baselines3', 'gymnasium', 'pyarrow']},
        'source_sha256': {str(p.relative_to(ROOT)): sha256(p) for p in source_files},
        'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'git_status': subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True),
        'device': agent['device'], 'deterministic_algorithms': True, 'repeat_requested': not args.skip_repeat}
    write_json(report_dir / 'manifest.json', manifest)
    write_json(model_dir / 'manifest.json', manifest)
    jobs = []
    labels = [(int(s), f'seed_{s}') for s in agent['seeds']]
    if not args.skip_repeat:
        labels.append((int(agent['seeds'][0]), f"seed_{agent['seeds'][0]}_repeat"))
    for seed, label in labels:
        jobs.append(dict(seed=seed, label=label, config=config, train_features=train, validation_features=validation,
                         train_prices=tp, validation_prices=vp, output_root=ROOT, run_id=run_id, timesteps=timesteps,
                         eval_freq=int(agent['eval_freq']), checkpoint_freq=int(agent['checkpoint_freq']), device=agent['device']))
    if args.workers == 1:
        runs = [train_one_seed(**job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs))) as pool:
            futures = [pool.submit(train_one_seed, **job) for job in jobs]
            runs = [f.result() for f in futures]
    report = {'algorithm': 'PPO', 'run_id': run_id, 'runs': runs,
              'assessment': aggregate_results(runs, agent['acceptance'])}
    write_json(report_dir / 'summary.json', report)
    print(json.dumps(report['assessment'], indent=2), flush=True)
    print(f'Report: {report_dir / "summary.json"}', flush=True)


if __name__ == '__main__':
    main()
