# Aachen Building Heat Demand Surrogate Model

LSTM-based surrogate model for predicting annual daily heat demand of residential buildings in Aachen.


---

## Project overview

Physical simulations of building heat demand (EDpyFlow + OpenModelica + AixLib + TEASER) are computationally expensive. This project trains a surrogate model that replicates the simulation's input/output interface at a fraction of the cost.

**Surrogate I/O — mirrors the physical simulation exactly:**

| | Physical simulation | This surrogate |
|---|---|---|
| Weather input | Full TRY year (365 days × 10 features) | Same |
| Building input | 6 static parameters | Same |
| Output | 365 daily heat demand values [kWh/day] | Same |

The weather data is the **Testreferenzjahr (TRY)** for Aachen, a synthetic reference climate year published by DWD representing typical local conditions.

---

## Dataset

- **3600 buildings** sampled via Latin Hypercube Sampling (LHS), covering 4 typologies × 3 refurbishment levels × 300 samples
- **3546 valid simulations** (54 unrefurbished MFH buildings from 1919–1948 failed to converge due to stiff solver dynamics)
- Simulations run at **6-hour resolution**; daily heat demand obtained by trapezoidal integration of instantaneous heating power (`multizone.PHeater[1]`)

---

## Model architecture

Dual-branch LSTM:
- **Branch 1 (sequential):** LSTM (hidden=64, layers=2, dropout=0.25) processes the last 14 days of weather
- **Branch 2 (static):** 6 building features concatenated to LSTM output
- **Head:** MLP (64→64→1) predicts heat demand per unit area; multiplied by floor area → kWh/day

---

## Results

| Metric | Value | ASHRAE threshold |
|---|---|---|
| CV(RMSE) | 22.61% | 30% ✅ |
| NMBE | +0.13% | ±10% ✅ |
| Test set | 2346 buildings, 790,602 days | |

Hyperparameter sensitivity analysis (varying training size, lookback, hidden size) shows CV stabilises around 22%, indicating the residual error reflects the inherent stochasticity of daily heat demand rather than model capacity.

---

## Repository structure

```
├── train_aachen_daily.py   # Main training script + surrogate inference functions
├── train_tune.py           # Hyperparameter sensitivity analysis
├── preprocess_6h_v2.py     # Raw simulation data processing (6h block deduplication)
├── aggregate_daily.py      # Trapezoidal integration → daily heat demand
├── README.md
```

**Training outputs** (not tracked in git — too large):
```
D:/outputs_daily/
├── model.pt        # Trained model weights
├── scalers.pkl     # Fitted StandardScalers
├── metrics.txt     # CV(RMSE), NMBE
└── predictions.png # Surrogate vs simulation plot
```

---

## Usage

**Train the surrogate:**
```bash
python train_aachen_daily.py
```

**Hyperparameter sweep:**
```bash
python train_tune.py --ntrain 1500 --lookback 21 --hidden 96
```

**Run surrogate inference** (after training):
```python
import pickle, torch, numpy as np
from train_aachen_daily import HeatDemandLSTM, predict_annual, SEQ_COLS, STAT_COLS

scalers = pickle.load(open("D:/outputs_daily/scalers.pkl", "rb"))
model   = HeatDemandLSTM(len(SEQ_COLS), len(STAT_COLS))
model.load_state_dict(torch.load("D:/outputs_daily/model.pt"))

# weather_365 : np.ndarray (365, 10) — full TRY year
# building    : np.ndarray (6,)      — static building features
heat_demand = predict_annual(model, scalers["seq"], scalers["static"], scalers["y"],
                             weather_365, building)
# returns (365,) kWh/day
```


