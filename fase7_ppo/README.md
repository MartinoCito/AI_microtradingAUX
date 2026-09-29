# Fase 7 — Addestramento PPO preliminare

Questa cartella aggiunge il prototipo della Fase 7 alla pipeline esistente.

## Obiettivi

- addestramento PPO su un sottoinsieme cronologico del train set;
- più seed;
- monitoraggio delle metriche PPO tramite logger Stable-Baselines3;
- checkpoint periodici;
- valutazione validation separata;
- confronto reward training/validation;
- distribuzione delle azioni short/flat/long;
- rilevazione diagnostica di policy degenerata.

## Dataset locale

I raw dati presenti nel repository sono temporanei e non vengono assunti come dataset definitivo.

Lo script richiede esplicitamente quattro file locali:

- feature M15 train;
- feature M15 validation;
- prezzi M15 train;
- prezzi M15 validation.

Le feature devono contenere le 13 market feature della Fase 3 già usate da `TradingEnv`.

## Esecuzione locale

Dalla root del repository:

```bash
pip install -r requirements.txt
pip install -r fase7_ppo/requirements-ppo.txt
```

Esempio:

```bash
python fase7_ppo/scripts/train_ppo.py \
  --train-features data/processed/xauusd_m15_train_features.parquet \
  --validation-features data/processed/xauusd_m15_validation_features.parquet \
  --train-prices data/processed/xauusd_m15_train.parquet \
  --validation-prices data/processed/xauusd_m15_validation.parquet \
  --train-fraction 0.25 \
  --timesteps 250000 \
  --seeds 42 43 44 \
  --eval-freq 25000 \
  --checkpoint-freq 25000
```

I percorsi sopra sono esempi: usare i file prodotti dalla pipeline locale definitiva.

## Output

```
models/prototype/
  seed_42/
  seed_43/
  seed_44/

logs/prototype/
  seed_42/
  seed_43/
  seed_44/

reports/ppo_prototype/
  summary.json
```

## Metriche

Stable-Baselines3 salva nel logger CSV/TensorBoard:

- policy gradient loss;
- value loss;
- entropy loss;
- explained variance;
- episode reward medio.

Il callback salva inoltre reward episodici, distribuzione delle azioni e checkpoint.

## Policy degenerata

La valutazione marca come diagnostica una policy che produce almeno il 99% della stessa azione.

Questo non è un giudizio automatico sulla strategia: una concentrazione estrema richiede analisi di reward, costi, dati, seed e dinamica del training.

## Test set

Il test set non viene utilizzato in questa fase. È riservato alla successiva valutazione finale dopo che la pipeline PPO preliminare sarà verificata.
