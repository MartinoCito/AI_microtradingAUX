# AI_microtradingAUX

**Fase 1 – Analisi e validazione del CSV** nel progetto `AI_microtradingAUX`, con particolare attenzione alla struttura dei file, alle scelte progettuali e al flusso operativo dei vari script.

## Obiettivi della Fase 1

La Fase 1 ha come obiettivo la preparazione e la validazione del dataset M1 (un minuto) per XAU/USD a partire da un file CSV grezzo con schema:

- `Date;Open;High;Low;Close;Volume`

Le attività richieste sono:

- Caricare il file con separatore `;`.
- Convertire la colonna `Date` in timestamp e usarla come indice temporale.
- Ordinare cronologicamente il dataset.
- Verificare ed eventualmente gestire i timestamp duplicati.
- Trovare gli intervalli mancanti rispetto alla frequenza attesa (1 minuto).
- Controllare la coerenza delle barre OHLC (Open, High, Low, Close).
- Analizzare il volume (zero/negative volume, media, distribuzione).
- Identificare weekend e chiusure prolungate del mercato.
- Documentare l’analisi del fuso orario del broker **senza assumerlo automaticamente**.

Il criterio di completamento è:

- Nessun timestamp duplicato nel dataset preparato.
- Nessuna barra OHLC “impossibile” (incoerente) accettata nel dataset validato.

## Struttura dei file rilevanti

### 1. File di dati grezzi

- `data/raw/XAU_1m_data.csv`

Contiene il dataset M1 originale con colonne `Date`, `Open`, `High`, `Low`, `Close`, `Volume`, separate da `;`. Questo file è la sorgente principale utilizzata da tutti gli script di caricamento e validazione dati.

### 2. Configurazione del caricamento dati

- `config.yaml`

La sezione `data` in `config.yaml` definisce i parametri chiave per la Fase 1:

- `raw_path`: percorso del CSV grezzo (es. `data/raw/XAU_1m_data.csv`).
- `separator`: separatore usato nel CSV, impostato a `";"`.
- `timestamp_column`: nome della colonna temporale, `Date`.
- `timestamp_format`: formato della data, es. `%Y.%m.%d %H:%M`.
- `duplicate_policy`: strategia per i duplicati (`keep_last`, `keep_first` o `raise`).
- `expected_frequency`: frequenza attesa delle barre, `1min`.

Questi parametri vengono riutilizzati dagli script `inspect_data.py` e `build_data_quality.py`, in modo da centralizzare la configurazione.

### 3. Modulo di caricamento e validazione

- `src/gold_rl/data/loader.py`

Questo modulo contiene la logica principale di caricamento, pulizia e validazione del dataset OHLCV. Le funzioni chiave sono:

#### 3.1 `load_market_data`

Responsabilità:

- Caricare il CSV dal percorso indicato (`file_path`) usando il separatore configurato.
- Validare la presenza delle colonne richieste (`Date`, `Open`, `High`, `Low`, `Close`, `Volume`).
- Rimuovere righe completamente vuote e righe con residui HTML (`<br>` ecc.).
- Convertire la colonna temporale `Date` in `DatetimeIndex` usando il formato configurato.
- Convertire le colonne numeriche (`Open`, `High`, `Low`, `Close`, `Volume`) in tipo numerico, segnalando eventuali valori non validi.
- Ordinare il dataset cronologicamente.
- Gestire i timestamp duplicati secondo la `duplicate_policy` (`keep_last` di default).
- Impostare `Date` come indice temporale e ridurre il consumo di memoria con tipi `float32`.
- Validare la coerenza delle barre OHLCV tramite `validate_ohlcv`.

Questa funzione garantisce che il dataset risultante sia:

- Ordinato cronologicamente.
- Privo di duplicati sull’indice temporale.
- Privo di valori non numerici nelle colonne OHLCV.
- Privo di barre OHLC impossibili.

#### 3.2 `validate_required_columns`

Controlla che nel dataset siano presenti tutte le colonne richieste (timestamp, Open, High, Low, Close, Volume). In caso contrario solleva un’eccezione con un messaggio dettagliato, indicando quali colonne mancano e quali sono state trovate.

#### 3.3 `remove_empty_and_html_rows`

- Rimuove righe completamente vuote.
- Rimuove righe che contengono residui HTML nella colonna temporale (es. `<br>`), tipici di CSV copiati da pagine web.
- Logga a console quante righe sono state rimosse per questo motivo.

#### 3.4 `handle_duplicates`

Gestisce i timestamp duplicati in base alla `duplicate_policy`:

- `keep_first`: mantiene la prima occorrenza per ciascun timestamp.
- `keep_last` (default): mantiene l’ultima occorrenza per ciascun timestamp.
- `raise`: solleva un’eccezione se vengono trovati duplicati.

In tutti i casi viene prodotto un messaggio a console con il conteggio delle righe coinvolte nei duplicati. Al termine, l’indice temporale del dataset è garantito privo di duplicati.

#### 3.5 `validate_ohlcv`

Effettua controlli approfonditi sulla coerenza delle barre:

- Il dataset non deve essere vuoto.
- L’indice deve essere un `DatetimeIndex` monotono crescente e senza duplicati.
- I prezzi `Open`, `High`, `Low`, `Close` devono essere strettamente positivi.
- Il volume `Volume` non deve essere negativo.
- Deve valere:
  - `High >= max(Open, Close, Low)`.
  - `Low <= min(Open, Close, High)`.

Se una di queste condizioni fallisce, viene sollevata un’eccezione con esempi delle righe problematiche. Questo assicura che nessuna barra OHLC impossibile entri nel dataset preparato.

#### 3.6 `get_dataset_summary`

Calcola un riepilogo diagnostico globale del dataset:

- Numero di righe (`rows`) e lista delle colonne (`columns`).
- Timestamp iniziale (`start`) e finale (`end`).
- Durata in giorni (`duration_days`).
- Numero di timestamp duplicati residui (che deve essere 0 dopo `handle_duplicates`).
- Numero totale di valori mancanti (`missing_values`).
- Numero di barre con volume zero (`zero_volume_bars`) e volume negativo (`negative_volume_bars`).
- Numero di gap rilevati rispetto alla frequenza attesa (`detected_gaps`).
- Numero stimato di barre mancanti (`estimated_missing_bars`).
- Gap più grande (`largest_gap`) e intervallo mediano (`median_interval`).
- Memoria utilizzata dal dataset (`memory_mb`).
- Minimum e maximum del prezzo di chiusura (`minimum_close`, `maximum_close`).
- Volume medio (`average_volume`).

Questi valori vengono poi usati nei report JSON e Markdown.

#### 3.7 `get_gaps_table`

Costruisce una tabella dei gap temporali rispetto all’intervallo atteso (es. 1 minuto):

- `gap_start`: timestamp dell’ultima barra prima del gap.
- `gap_end`: timestamp della prima barra dopo il gap.
- `gap_duration`: durata del gap (differenza tra `gap_end` e `gap_start`).
- `missing_bars`: numero stimato di barre mancanti all’interno del gap.

Questa tabella viene esportata in `reports/gaps.csv`.

#### 3.8 `analyze_calendar`

Analizza il calendario del dataset per identificare pattern giornalieri e chiusure prolungate:

- `bars_per_day_of_week`: numero di barre per ciascun giorno della settimana (0=lunedì, …, 6=domenica).
- `weekend_bars`: numero totale di barre in sabato e domenica.
- `weeks_with_data`: numero di settimane che contengono almeno una barra.
- `average_trading_days_per_week`: numero medio di giorni di trading per settimana.
- `long_closures`: lista di chiusure prolungate (gap di durata ≥ soglia, tipicamente 2 giorni), con:
  - `start`: timestamp iniziale della chiusura.
  - `end`: timestamp finale.
  - `duration`: durata complessiva.
  - `estimated_missing_bars`: barre mancanti stimate.
- `timezone`: lasciato a `None` per indicare che non è stata assegnata una timezone.
- `timezone_note`: testo esplicito che spiega che il fuso orario del broker deve essere determinato da metadati esterni (documentazione, label EET/EEST, impostazioni MT4/MT5) e non assunto automaticamente.

Questa funzione supporta i requisiti “identificare weekend e chiusure” e “stabilire il fuso orario del broker” senza forzare una timezone.

#### 3.9 `build_data_quality_report`

Combina `get_dataset_summary` e `analyze_calendar` per produrre un report strutturato pronto per essere serializzato in `reports/data_quality.json`. Il dizionario risultante include:

- Sezione `schema`:
  - `columns`: lista delle colonne del dataset.
  - `separator`: separatore del CSV (`;`).
  - `timestamp_column`: nome della colonna temporale (`Date`).
  - `timestamp_format`: formato della data (`%Y.%m.%d %H:%M`).
  - `timezone`: valore `None`, per indicare che la timezone non è stata assegnata automaticamente.
- Sezione `integrity`:
  - `rows`, `duplicate_timestamps`, `missing_values`.
  - `detected_gaps`, `estimated_missing_bars`.
  - `largest_gap`, `median_interval`.
- Sezione `volume`:
  - `zero_volume_bars`, `negative_volume_bars`, `average_volume`.
- Sezione `price`:
  - `minimum_close`, `maximum_close`.
- Sezione `calendar`:
  - output di `analyze_calendar`.
- Sezione `notes`:
  - `timezone_note` e nota sull’ipotesi di frequenza (`frequency_assumption`).

Questa struttura fornisce una vista completa della qualità e della struttura temporale del dataset.

### 4. Script di ispezione dati

- `scripts/inspect_data.py`

Scopo principale:

- Caricare il dataset M1 tramite `load_market_data` utilizzando i parametri di `config.yaml`.
- Calcolare il riepilogo con `get_dataset_summary`.
- Stampare a console il riepilogo.
- Salvare un report in `reports/dataset_summary.json`.

Questo script è utile per una prima analisi generale dei dati ed è complementare a `build_data_quality.py`.

### 5. Script di generazione dei report di qualità

- `scripts/build_data_quality.py`

Questo script automatizza l’intera pipeline della Fase 1, generando tre output principali:

- `reports/data_quality.json`: report strutturato di qualità e calendario.
- `reports/gaps.csv`: tabella dei gap temporali.
- `reports/data_summary.md`: riepilogo human‑readable.

Passi principali:

1. Caricamento della configurazione da `config.yaml`.
2. Calcolo del percorso completo del CSV (`raw_path`).
3. Caricamento e validazione del dataset tramite `load_market_data`.
4. Calcolo del riepilogo con `get_dataset_summary`.
5. Costruzione della tabella dei gap con `get_gaps_table`.
6. Costruzione del report di qualità con `build_data_quality_report`.
7. Serializzazione in:
   - `data_quality.json` (JSON formattato, con `default=str` per gestire i timestamp).
   - `gaps.csv` (CSV senza indice).
   - `data_summary.md` (Markdown con sezioni Overview, Integrity, Volume, Prices, Calendar, Timezone).

Questo script è il riferimento operativo per la Fase 1: eseguendolo dalla root del progetto si rigenerano tutti i report coerenti con lo stato corrente del CSV e della configurazione.

### 6. Report generati

- `reports/dataset_summary.json`

  - Contiene il riepilogo prodotto da `inspect_data.py` con valori di sintesi su righe, memorizzazione, gap e volume.

- `reports/data_quality.json`

  - Include tutte le sezioni prodotte da `build_data_quality_report` (schema, integrità, volume, prezzi, calendario, note).
  - È il file principale per documentare “cosa c’è dentro” il dataset e come si comporta nel tempo.

- `reports/gaps.csv`

  - Elenca in forma tabellare tutti i gap temporali identificati rispetto alla frequenza attesa.
  - Ogni riga permette di risalire rapidamente a periodi problematici o a chiusure di mercato.

- `reports/data_summary.md`

  - Fornisce un riepilogo leggibile, adatto ad essere incluso come sezione di una relazione di progetto.
  - Riassume i principali indicatori quantitativi (righe, durata, volume, gap) e qualitativi (weekend, chiusure, note sul fuso orario).

### 7. Test

- `tests/test_data_validation.py`

Sebbene semplice, questo file è il punto di partenza per introdurre test automatici sulla Fase 1. Può essere esteso per:

- Verificare che `load_market_data` sollevi errori in caso di CSV con colonne mancanti o valori non numerici.
- Testare che `validate_ohlcv` rifiuti barre con OHLC incoerenti.
- Controllare che `build_data_quality_report` produca chiavi attese in `data_quality.json`.

## Scelte progettuali chiave

### Separatore e formato data

- Il separatore è `;`, coerente con il pattern originario del CSV.
- Il formato `Date` è `YYYY.MM.DD HH:MM` (`%Y.%m.%d %H:%M`), compatibile con molti esport di piattaforme di trading.

### Gestione dei duplicati

- Strategia predefinita: `keep_last`, che mantiene l’ultima barra osservata per un certo timestamp.
- Opzione `raise` utilizzabile per pipeline più rigide, dove ogni duplicato è considerato un errore bloccante.

### Gap temporali e minuti “vuoti”

- I gap vengono rilevati rispetto a una frequenza attesa (`1min`).
- I minuti senza transazioni in un dataset M1 reale sono trattati come gap, ma la nota in `data_quality.json` chiarisce che ciò è spesso legittimo su dati FX.

### Coerenza OHLC

- Vengono rifiutate barre con prezzi nulli/negativi o con High/Low incongrui rispetto a Open/Close.
- Questo rende il dataset più affidabile per qualsiasi fase successiva (feature engineering, training RL, backtest).

### Fuso orario del broker

- Nessuna timezone viene applicata automaticamente ai timestamp.
- Il campo `timezone` in `data_quality.json` è impostato a `null`.
- `timezone_note` sottolinea che il fuso orario va determinato in base alla provenienza del file (broker, MT4/MT5, label EET/EEST) e non inferito dal codice.
- Questo approccio è allineato alle indicazioni del repository di riferimento (Uso di `Europe/Helsinki` solo per dataset esplicitamente etichettati come EET/EEST).

## Conclusione

La Fase 1 del progetto `AI_microtradingAUX` è stata implementata in modo da garantire:

- Dataset M1 pulito, ordinato cronologicamente e privo di duplicati temporali.
- Nessuna barra OHLC impossibile nel dataset preparato.
- Documentazione completa di gap, volume, struttura temporale e comportamento del calendario (inclusi weekend e chiusure prolungate).
- Flessibilità nel trattamento del fuso orario, che viene analizzato ma non imposto senza informazioni esterne.

## Fase 7 — Addestramento PPO preliminare

La preparazione parte dal prodotto M15 della Fase 2, costruisce feature causali e seleziona il sottoinsieme cronologico **prima** di stimare la normalizzazione. Lo scaler non utilizza né il resto del training né validation/test.

```bash
python scripts/train_ppo.py --run-id nuovo_esperimento --workers 4
```

Il comando esegue i tre seed configurati e ripete da zero il primo seed con lo stesso budget. CPU e un thread PyTorch sono i default riproducibili. `--workers 1` esegue i lavori in sequenza. Sono disponibili `--timesteps`, `--train-fraction`, `--m15-path`, `--seeds`, `--eval-freq`, `--checkpoint-freq` e `--device`. `--skip-repeat` è solo diagnostico: non certifica il completamento.

Ogni esperimento ha una directory identificata dal `run-id`, generato automaticamente se omesso; un identificativo già esistente viene rifiutato per evitare sovrascritture e log misti:

- `models/prototype/<run_id>/<seed>/`: modelli finali e best; il manifest con scaler è nella directory superiore dell'esperimento.
- `logs/prototype/<run_id>/<seed>/`: metriche CSV/TensorBoard, azioni, checkpoint e valutazioni periodiche.
- `reports/ppo_prototype/<run_id>/`: manifest, riepilogo e risultati dettagliati.

Il manifest salva configurazione effettiva, hash dei dati M15 e dei sorgenti, commit/stato Git, versioni, confini temporali e statistiche di normalizzazione. Per utilizzare un modello conservare il manifest insieme ai pesi. I risultati precedenti senza run-id restano separati.

`scripts/train_ppo.py` gestisce l'esperimento; `src/gold_rl/rl/callbacks.py` registra azioni e checkpoint. `src/gold_rl/rl/persistence.py` salva archivi leggibili dal normale `stable_baselines3.PPO.load`, evitando il problema di lettura dei tensori annidati osservato su Windows. `load_ppo` nello stesso modulo legge anche gli archivi precedenti senza modificarli.

### Monitoraggio e confronto

Sono registrati policy gradient loss, value loss, entropy loss, reward degli episodi e distribuzione short/flat/long. Entropy loss è il negativo dell'entropia. Il budget effettivo può superare quello nominale per completare un rollout.

La stessa policy finale viene ricaricata e valutata deterministicamente su training e validation in finestre di 5.000 passi con reset del portafoglio e stessi costi. Il report mostra reward per passo, rendimento composto medio, dispersione e coda esclusa. Queste misure sostituiscono il confronto improprio tra reward totali su durate diverse. Sono inoltre valutati sull'intera validation i modelli finali e best; il test resta escluso.

Il criterio di completamento richiede almeno tre seed, nessuna azione dominante oltre la soglia configurata (99%), identità di pesi/risultati nella ripetizione e variabilità entro `agent.acceptance`. Le tolleranze sono fissate prima dell'esecuzione: intervallo dei rendimenti medi su finestre al massimo 10 punti percentuali e delle quote di azione al massimo 20 punti percentuali. Sono criteri preliminari di stabilità, non di redditività; se falliscono il report dichiara la fase non conclusa.

### Notebook e verifiche

`pipeline/phase_7_ppo_preliminary_training.ipynb` legge i risultati tramite `RUN_ID`, mostra curve PPO, confronto omogeneo, azioni deterministiche, modelli best ed esito della ripetizione. Non contiene tabelle di risultati inserite manualmente.

Le dipendenze del notebook sono opzionali rispetto al training:

```bash
pip install -r requirements-notebook.txt
python -m pytest -q
```

Per TensorBoard: `tensorboard --logdir logs/prototype/<run_id>`.

## Fase 8 — Walk-forward validation

```bash
python scripts/train_walk_forward.py --run-id nuovo_walk_forward --workers 4
```

Il default crea **4 fold mobili**: 5 anni di training, 6 mesi di validation e 6 mesi successivi di test OOS. Le finestre di test sono consecutive e non sovrapposte e terminano prima del holdout finale, che resta escluso. La frontiera del holdout coincide con lo split M15 della Fase 7. `--folds` ammette 3–5 fold, `--timesteps` cambia il budget per modello; i parametri temporali sono in `walk_forward` nel file di configurazione.

Ogni modello parte da zero con lo stesso seed predefinito 42 e budget nominale di 250.000 passi. Lo scaler è stimato solo sul training di quel fold. Il modello scelto massimizza il rendimento composto della validation dopo i costi di liquidazione; in caso di parità si conserva il primo candidato. Viene valutato anche il modello dopo l'ultimo aggiornamento PPO. Le finestre OOS non sono passate al callback di selezione.

La fase produce:

- `reports/walk_forward_metrics.csv`: metriche per fold e aggregate per PPO e cinque baseline.
- `reports/walk_forward_equity.csv`: equity e rendimenti per timestamp, fold e strategia, in formato lungo.
- `reports/figures/walk_forward.png`: equity OOS e contributi al PnL dei fold.
- `reports/walk_forward/<run_id>/`: copia dei report, manifest, snapshot dei sorgenti e criterio di completamento.
- `models/walk_forward/<run_id>/fold_<n>/`: modello finale, checkpoint selezionato e punteggi di validation.
- `logs/walk_forward/<run_id>/fold_<n>/`: metriche PPO, checkpoint periodici e azioni di training.

Gli artefatti di un esperimento esistente non vengono sovrascritti. I tre report principali sono aggiornati solo dopo il completamento del nuovo esperimento e del grafico. `--plot-python` permette di indicare un altro interprete con pandas e matplotlib; normalmente non serve se sono installate le dipendenze di `requirements.txt`.

### Causalità, baseline e continuità dell'equity

Si confrontano PPO, always_flat, always_long, random, momentum a un passo ed EMA 12/26. I segnali delle baseline sono quelli della Fase 5, ricalcolati sui prezzi M15 con storia passata, senza tuning sul test. Per tutte le strategie la decisione usa la barra precedente e viene eseguita sul close successivo con il medesimo motore di execution/portfolio. L'osservazione immediatamente precedente a ogni finestra serve solo da contesto e non compare tra le righe OOS valutate.

Le posizioni vengono liquidate all'ultimo close del fold, pagando i costi del motore. Il saldo finale finanzia il fold seguente. Non si riportano le equity a 100.000 e non si moltiplicano curve costruite con capitali indipendenti: il portafoglio usa quantità fisse di una unità e lo stato osservato dalla policy dipende dal capitale effettivo. La concatenazione controlla timestamp univoci e rendimenti coerenti anche ai confini. Lo Sharpe riportato è per barra, senza riutilizzare impropriamente l'annualizzazione M1 della Fase 5.

### Dipendenza dai fold e limiti

Il criterio preliminare richiede PnL OOS totale positivo, almeno metà dei fold positivi, nessun fold con più del 50% della somma dei PnL positivi e contributo totale ancora positivo togliendo ciascun fold. Le soglie sono registrate prima del training. L'esclusione di un fold è un'analisi di attribuzione sui trade osservati, non una nuova simulazione controfattuale della policy con un capitale diverso. Se PPO perde complessivamente, il report dichiara `non_positive_oos`: non viene presentato come un risultato positivo robusto.

Questo studio non risolve né nasconde la variabilità multi-seed emersa in Fase 7: usa un seed fisso per isolare la variabilità temporale. Inoltre alcuni periodi sono già stati osservati nella ricerca precedente. Si tratta quindi di una verifica walk-forward retrospettiva, distinta dal holdout finale escluso da questa fase. Le finestre già trascorse possono entrare nella storia di training dei fold successivi, come avverrebbe riaddestrando nel tempo.

Il disegno riprende l'idea di finestre OOS concatenate descritta nel [repository di riferimento](https://github.com/ZiadFrancis/Reinforcement_Trading_Part_2/blob/main/README.md), adattata al motore e all'azione discreta di questo progetto. Non ne copia il sistema di bracket/stop-loss/take-profit.
