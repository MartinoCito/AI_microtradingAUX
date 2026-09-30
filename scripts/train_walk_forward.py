from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from baselines import build_target_positions, STRATEGIES
from train_ppo import build_env_factory, load_config, policy_digest, sha256, write_json
from gold_rl.data.features import create_market_features, MARKET_FEATURE_COLUMNS
from gold_rl.execution import ExecutionCostsConfig
from gold_rl.rl.callbacks import build_ppo_callbacks
from gold_rl.rl.persistence import PortablePPO
from gold_rl.rl.walk_forward import (make_folds, prepare_fold, rollout, curve_metrics,
                                     stitch_curves, fold_dependence)


def costs_from_config(config):
    return ExecutionCostsConfig(**{k: float(config['environment'][k]) for k in
        ['spread_rate', 'commission_rate', 'slippage_rate', 'turnover_penalty']})


class ValidationSelection(BaseCallback):
    """Only training and validation are accessible here; no OOS test inputs."""
    def __init__(self, features, prices, config, folder):
        super().__init__()
        self.features, self.prices = features, prices
        self.config, self.folder = config, Path(folder)
        self.best_score = -np.inf
        self.records = []

    def evaluate(self, stage="periodic"):
        initial = float(self.config['environment']['initial_balance'])
        curve = rollout(self.features, self.prices, costs_from_config(self.config), initial, model=self.model)
        metrics = curve_metrics(curve, initial)
        selected = metrics['total_return'] > self.best_score
        self.records.append({'timesteps': int(self.num_timesteps), 'stage': stage, **metrics, 'new_best': bool(selected)})
        if selected:
            self.best_score = metrics['total_return']
            self.model.save(self.folder / 'best_model.zip')
            write_json(self.folder / 'selection.json', {'selected_timesteps': int(self.num_timesteps),
                'validation_return': self.best_score, 'selected_stage': stage, 'policy_sha256': policy_digest(self.model),
                'selection_metric': 'validation total_return after terminal liquidation', 'test_used': False})
        pd.DataFrame(self.records).to_csv(self.folder / 'validation_scores.csv', index=False)
        self.logger.record('eval/validation_return', metrics['total_return'])
        print(f'{self.folder.name}: step {self.num_timesteps}, validation {metrics["total_return"]:.4%}', flush=True)

    def _on_step(self):
        if self.num_timesteps % int(self.config['walk_forward']['eval_freq']) == 0:
            self.evaluate()
        return True

    def _on_training_end(self):
        # A periodic callback at the same timestep precedes the final PPO update.
        self.evaluate(stage='final')


def train_fold(*, number, config, train, train_prices, validation, validation_prices, model_dir, log_dir):
    model_dir, log_dir = Path(model_dir), Path(log_dir)
    model_dir.mkdir(parents=True, exist_ok=False)
    log_dir.mkdir(parents=True, exist_ok=False)
    agent, wf = config['agent'], config['walk_forward']
    seed = int(wf['seed'])
    torch.set_num_threads(int(agent['torch_threads']))
    torch.use_deterministic_algorithms(True)
    set_random_seed(seed)
    env = DummyVecEnv([build_env_factory(train, train_prices, config['environment'], 'train', seed)])
    selection = ValidationSelection(validation, validation_prices, config, model_dir)
    callbacks = build_ppo_callbacks(output_dir=log_dir, checkpoint_freq=int(wf['checkpoint_freq']),
                                    eval_callback=selection, degeneracy_threshold=float(agent['degeneracy_threshold']))
    model = PortablePPO('MlpPolicy', env, device='cpu', seed=seed, verbose=0,
        policy_kwargs={'net_arch': {'pi': agent['policy_hidden_layers'], 'vf': agent['policy_hidden_layers']}},
        **{k: agent[k] for k in ['learning_rate', 'n_steps', 'batch_size', 'n_epochs', 'gamma',
                                 'gae_lambda', 'clip_range', 'ent_coef', 'vf_coef', 'max_grad_norm']})
    model.set_logger(configure(str(log_dir), ['csv', 'tensorboard']))
    try:
        model.learn(total_timesteps=int(wf['total_timesteps']), callback=callbacks)
        model.save(model_dir / 'final_model.zip')
        selected = json.loads((model_dir / 'selection.json').read_text(encoding='utf-8'))
        restored = PPO.load(model_dir / 'best_model.zip', device='cpu')
        if policy_digest(restored) != selected['policy_sha256']:
            raise AssertionError('Checkpoint selezionato non corrispondente.')
        result = {'fold': number, 'seed': seed, 'actual_timesteps': int(model.num_timesteps), **selected,
                  'model_path': str(model_dir / 'best_model.zip'), 'native_reload_verified': True}
        write_json(model_dir / 'training_summary.json', result)
        return result
    finally:
        env.close()


def main():
    p = argparse.ArgumentParser(description='Fase 8: walk-forward PPO e baseline, capitale OOS continuo')
    p.add_argument('--config', type=Path, default=ROOT / 'config.yaml')
    p.add_argument('--run-id', default=None)
    p.add_argument('--folds', type=int)
    p.add_argument('--timesteps', type=int)
    p.add_argument('--workers', type=int, default=1)
    p.add_argument('--plot-python', default=sys.executable, help='Interprete con pandas/matplotlib installati')
    args = p.parse_args()
    config = load_config(args.config)
    wf = config['walk_forward']
    if args.folds is not None:
        wf['n_folds'] = args.folds
    if args.timesteps is not None:
        wf['total_timesteps'] = args.timesteps
    if min(args.workers, int(wf['total_timesteps']), int(wf['eval_freq']), int(wf['checkpoint_freq'])) < 1:
        raise ValueError('Worker, budget e frequenze devono essere positivi.')
    run_id = args.run_id or datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    if not run_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in run_id):
        raise ValueError('run-id non valido.')
    report_dir = ROOT / 'reports/walk_forward' / run_id
    model_dir = ROOT / 'models/walk_forward' / run_id
    log_dir = ROOT / 'logs/walk_forward' / run_id
    if any(x.exists() for x in [report_dir, model_dir, log_dir]):
        raise FileExistsError('Esperimento già esistente; usare un nuovo run-id.')
    source = ROOT / config['data']['m15_path']
    ohlcv = pd.read_parquet(source)
    features = create_market_features(ohlcv, drop_warmup=True, strict_windows=True)[MARKET_FEATURE_COLUMNS]
    if features.index.has_duplicates or not np.isfinite(features.to_numpy()).all():
        raise ValueError('Feature non valide.')
    prices = ohlcv.loc[features.index, 'Close'].astype(float)
    ratios = [float(config['data'][k]) for k in ['train_ratio', 'validation_ratio', 'test_ratio']]
    if any(x <= 0 for x in ratios) or not np.isclose(sum(ratios), 1):
        raise ValueError('Split train/validation/test non validi.')
    if not np.isfinite(prices).all() or (prices <= 0).any():
        raise ValueError('Prezzi non validi.')
    holdout_index = int(len(features) * config['data']['train_ratio']) + int(len(features) * config['data']['validation_ratio'])
    holdout_start = features.index[holdout_index]
    folds = make_folds(features.index, n_folds=int(wf['n_folds']), train_years=int(wf['train_years']),
                       validation_months=int(wf['validation_months']), test_months=int(wf['test_months']),
                       holdout_start=holdout_start)
    prepared = [prepare_fold(features, prices, f, normalize=config['features'].get('normalize', True)) for f in folds]
    if any(len(data['train']) <= config['environment']['max_episode_steps'] for data in prepared):
        raise ValueError('Training troppo corto per gli episodi.')
    report_dir.mkdir(parents=True)
    model_dir.mkdir(parents=True)
    source_paths = [Path(__file__), ROOT / 'scripts/train_ppo.py', ROOT / 'baselines.py',
                    *sorted((ROOT / 'src/gold_rl').rglob('*.py'))]
    phase7_path = ROOT / wf['phase7_report'] if wf.get('phase7_report') else None
    phase7 = json.loads(phase7_path.read_text(encoding='utf-8'))['assessment'] if phase7_path and phase7_path.exists() else None
    manifest = {'run_id': run_id, 'created_utc': datetime.now(timezone.utc).isoformat(), 'config': config,
        'm15_sha256': sha256(source), 'holdout_start': str(holdout_start), 'holdout_used': False,
        'folds': [{**f.metadata(), 'train_rows': len(d['train']), 'validation_rows': len(d['validation'])-1,
                   'test_rows': len(d['test'])-1, 'normalization': d['normalization']} for f, d in zip(folds, prepared)],
        'baseline_parameters': {'momentum_lag': 1, 'ema_fast': 12, 'ema_slow': 26, 'random_seed': int(wf['seed'])},
        'capital_protocol': 'Sequential capital, fixed units, liquidate at each fold end with engine costs; no equity rebasing.',
        'research_limit': 'Retrospective folds; some periods have already been inspected in Phase 7. Not a pristine final holdout.',
        'selection': 'Maximum validation compounded return after liquidation; earliest candidate wins ties.',
        'phase7_assessment': phase7,
        'source_sha256': {str(path.relative_to(ROOT)): sha256(path) for path in source_paths},
        'versions': {name: importlib.metadata.version(name) for name in ['numpy', 'pandas', 'torch', 'stable-baselines3', 'gymnasium']},
        'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()}
    write_json(report_dir / 'manifest.json', manifest)
    write_json(model_dir / 'manifest.json', manifest)
    with zipfile.ZipFile(report_dir / 'source_snapshot.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in source_paths:
            archive.write(path, str(path.relative_to(ROOT)))
    jobs = [dict(number=f.number, config=config, train=d['train'], train_prices=d['train_prices'],
                 validation=d['validation'], validation_prices=d['validation_prices'],
                 model_dir=model_dir / f'fold_{f.number}', log_dir=log_dir / f'fold_{f.number}') for f, d in zip(folds, prepared)]
    if args.workers == 1:
        trained = [train_fold(**job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs))) as pool:
            pending = [pool.submit(train_fold, **job) for job in jobs]
            trained = [task.result() for task in pending]
    # Baseline signals are causal and fixed in advance; no test-based tuning.
    baseline_targets = build_target_positions(ohlcv.loc[ohlcv.index < holdout_start], seed=int(wf['seed']))
    initial = float(config['environment']['initial_balance'])
    strategies = ('ppo', *STRATEGIES)
    balances = {name: initial for name in strategies}
    curves = {name: [] for name in strategies}
    metric_rows = []
    torch.set_num_threads(int(config['agent']['torch_threads']))
    for fold, data, result in zip(folds, prepared, trained):
        model = PPO.load(result['model_path'], device='cpu')
        for strategy in strategies:
            start_balance = balances[strategy]
            params = {'model': model} if strategy == 'ppo' else {'targets': baseline_targets[strategy].loc[data['test'].index]}
            curve = rollout(data['test'], data['test_prices'], costs_from_config(config), start_balance, **params)
            curve['fold'], curve['strategy'], curve['run_id'] = fold.number, strategy, run_id
            metrics = curve_metrics(curve, start_balance)
            metric_rows.append({'run_id': run_id, 'scope': 'fold', 'fold': fold.number, 'strategy': strategy,
                                'start': str(curve['timestamp'].iloc[0]), 'end': str(curve['timestamp'].iloc[-1]),
                                'selected_timesteps': result['selected_timesteps'] if strategy == 'ppo' else None,
                                **metrics})
            balances[strategy] = float(curve['equity'].iloc[-1])
            curves[strategy].append(curve)
        print(f'OOS fold {fold.number} complete: PPO equity {balances["ppo"]:.2f}', flush=True)
    equity_frames = []
    assessments = {}
    for strategy in strategies:
        combined = stitch_curves(curves[strategy], initial)
        if combined['timestamp'].max() >= holdout_start:
            raise AssertionError('Accesso al holdout finale.')
        equity_frames.append(combined)
        metric_rows.append({'run_id': run_id, 'scope': 'aggregate', 'fold': 0, 'strategy': strategy,
                            'start': str(combined['timestamp'].iloc[0]), 'end': str(combined['timestamp'].iloc[-1]),
                            **curve_metrics(combined, initial)})
        pnl = [row['pnl'] for row in metric_rows if row['scope'] == 'fold' and row['strategy'] == strategy]
        assessments[strategy] = fold_dependence(pnl, max_positive_share=float(wf['max_positive_pnl_share']),
                                               min_positive_fraction=float(wf['min_positive_fold_fraction']))
    metrics = pd.DataFrame(metric_rows)
    equity = pd.concat(equity_frames).sort_values(['timestamp', 'strategy']).reset_index(drop=True)
    metrics.to_csv(report_dir / 'walk_forward_metrics.csv', index=False, float_format='%.17g')
    equity.to_csv(report_dir / 'walk_forward_equity.csv', index=False, float_format='%.17g')
    write_json(report_dir / 'summary.json', {'run_id': run_id, 'models': trained, 'assessment': assessments,
                                           'completion_passed': assessments['ppo']['completion_passed'],
                                           'holdout_used': False, 'oos_starting_capital': initial})
    subprocess.run([args.plot_python, str(ROOT / 'scripts/plot_walk_forward.py'), '--report-dir', str(report_dir)], check=True)
    # Publish only a complete run; the per-run originals remain available.
    for name in ['walk_forward_metrics.csv', 'walk_forward_equity.csv']:
        shutil.copy2(report_dir / name, ROOT / 'reports' / name)
    (ROOT / 'reports/figures').mkdir(parents=True, exist_ok=True)
    shutil.copy2(report_dir / 'walk_forward.png', ROOT / 'reports/figures/walk_forward.png')
    write_json(ROOT / 'reports/walk_forward_latest.json', {'run_id': run_id, 'report_dir': str(report_dir)})
    print(json.dumps(assessments['ppo'], indent=2), flush=True)
    print(f'Reports saved: {report_dir}', flush=True)


if __name__ == '__main__':
    main()
