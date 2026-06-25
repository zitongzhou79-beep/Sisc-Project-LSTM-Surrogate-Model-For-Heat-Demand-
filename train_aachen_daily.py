
import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_CSV = "D:/surrogate_project/aachen_dataset_daily.csv"
OUT_DIR  = "D:/outputs"

# Hyperparameters
N_TRAIN, N_VAL, N_TEST = 800, 400, None   # N_TEST=None → use all remaining buildings
LOOKBACK    = 14    # days of weather history per prediction step
WARMUP_DAYS = 14    # skip first N days of each building (no full lookback available)
HIDDEN      = 64
LAYERS      = 2
DROPOUT     = 0.25
LR          = 1e-3
BATCH       = 256
EPOCHS      = 120
PATIENCE    = 15    # early stopping patience (based on validation loss)
SEED        = 42

# Feature columns
SEQ_COLS = [
    "T_out", "sol_global", "sol_diffuse", "sol_direct",
    "rel_hum", "wind_speed", "cloud_opaque", "T_dew",
    "doy_sin", "doy_cos",                         # seasonal encoding
]
STAT_COLS = [
    "construction_year", "net_leased_area", "num_floors",
    "floor_height", "is_MFH", "refurb_level",
]
TARGET = "daily_heat_per_area"   # W/m²; multiplied by area → kWh/day at inference

# Reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(OUT_DIR, exist_ok=True)
print("Device:", DEVICE)



# 1. Data utilities

def make_windows(df: pd.DataFrame):
    """
    Convert a single building's time-series DataFrame into sliding-window arrays.

    Returns (Xs, Xt, Y, A) where:
      Xs : (N, LOOKBACK, len(SEQ_COLS))  — weather sequences
      Xt : (N, len(STAT_COLS))           — static features (repeated per window)
      Y  : (N,)                          — target heat per area [W/m²]
      A  : (N,)                          — net_leased_area [m²] for kWh conversion
    """
    df = df.sort_values("day").iloc[WARMUP_DAYS:].reset_index(drop=True)
    if len(df) <= LOOKBACK:
        empty = lambda s: np.empty(s, np.float32)
        return (empty((0, LOOKBACK, len(SEQ_COLS))), empty((0, len(STAT_COLS))),
                empty((0,)), empty((0,)))

    W = df[SEQ_COLS].to_numpy(np.float32)
    S = df[STAT_COLS].to_numpy(np.float32)
    y = df[TARGET].to_numpy(np.float32)
    a = df["net_leased_area"].to_numpy(np.float32)

    Xs, Xt, Y, A = [], [], [], []
    for t in range(LOOKBACK, len(df)):
        Xs.append(W[t - LOOKBACK:t])
        Xt.append(S[t])
        Y.append(y[t])
        A.append(a[t])

    return (np.asarray(Xs, np.float32), np.asarray(Xt, np.float32),
            np.asarray(Y, np.float32),  np.asarray(A, np.float32))


class WindowDataset(Dataset):
    def __init__(self, Xs, Xt, Y):
        self.Xs = torch.tensor(Xs)
        self.Xt = torch.tensor(Xt)
        self.Y  = torch.tensor(Y)

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, i):
        return self.Xs[i], self.Xt[i], self.Y[i]



# 2. Model


class HeatDemandLSTM(nn.Module):
    """
    Dual-branch LSTM for daily heat demand prediction.

    Branch 1: LSTM processes the weather sequence (last LOOKBACK days).
    Branch 2: Static building features concatenated to the LSTM output.
    Head:     Two-layer MLP → scalar heat demand per unit area.
    """
    def __init__(self, n_seq: int, n_stat: int):
        super().__init__()
        self.lstm = nn.LSTM(n_seq, HIDDEN, LAYERS, batch_first=True, dropout=DROPOUT)
        self.head = nn.Sequential(
            nn.Linear(HIDDEN + n_stat, 64),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(64, 1),
        )

    def forward(self, xs: torch.Tensor, xt: torch.Tensor) -> torch.Tensor:
        o, _ = self.lstm(xs)
        return self.head(torch.cat([o[:, -1, :], xt], dim=1)).squeeze(-1)



# 3. Surrogate inference interface
#    Mirrors the physical simulation I/O exactly.


def predict_annual(
    model:   HeatDemandLSTM,
    ss:      StandardScaler,
    st:      StandardScaler,
    sy:      StandardScaler,
    weather_365:        np.ndarray,
    building_features:  np.ndarray,
) -> np.ndarray:
    """
    Surrogate inference for one building — matches the simulation I/O contract.

    Physical simulation interface:
      IN  : full TRY year weather (365 days) + building parameters
      OUT : full year daily heat demand (365 days, kWh/day)

    Parameters
    ----------
    model             : trained HeatDemandLSTM
    ss, st, sy        : fitted StandardScalers (weather, static, target)
    weather_365       : np.ndarray (365, 10)  — one full TRY year
    building_features : np.ndarray (6,)       — static building parameters

    Returns
    -------
    heat_demand : np.ndarray (365,)
        Daily heat demand in kWh/day. Days 0 … WARMUP_DAYS+LOOKBACK-1
        have no lookback coverage and are returned as NaN.
    """
    weather_365       = np.asarray(weather_365,       dtype=np.float32)
    building_features = np.asarray(building_features, dtype=np.float32)

    assert weather_365.shape       == (365, len(SEQ_COLS)),  \
        f"Expected (365, {len(SEQ_COLS)}), got {weather_365.shape}"
    assert building_features.shape == (len(STAT_COLS),), \
        f"Expected ({len(STAT_COLS)},), got {building_features.shape}"

    area  = float(building_features[1])          # net_leased_area → kWh conversion
    start = WARMUP_DAYS + LOOKBACK               # first predictable day (= day 28)

    W_sc  = ss.transform(weather_365)            # (365, 10) scaled
    xt_sc = st.transform(building_features.reshape(1, -1)).astype(np.float32)  # (1, 6)

    # build all sliding windows for days [start … 364]
    idx = np.arange(start, 365)
    Xs  = np.stack([W_sc[t - LOOKBACK:t] for t in idx]).astype(np.float32)    # (N, 14, 10)
    Xt  = np.repeat(xt_sc, len(idx), axis=0)                                  # (N, 6)

    model.eval()
    with torch.no_grad():
        pr = model(
            torch.tensor(Xs).to(DEVICE),
            torch.tensor(Xt).to(DEVICE),
        ).cpu().numpy()

    pred_per_area = np.clip(sy.inverse_transform(pr.reshape(-1, 1)).ravel(), 0, None)
    pred_kwh      = pred_per_area * area

    result         = np.full(365, np.nan, dtype=np.float32)
    result[start:] = pred_kwh
    return result


def predict_annual_batch(
    model:                    HeatDemandLSTM,
    ss:                       StandardScaler,
    st:                       StandardScaler,
    sy:                       StandardScaler,
    weather_365:              np.ndarray,
    building_features_matrix: np.ndarray,
    chunk: int = 512,
) -> np.ndarray:
    """
    Batch surrogate inference for N buildings sharing one TRY weather year.

    Parameters
    ----------
    weather_365              : np.ndarray (365, 10)
    building_features_matrix : np.ndarray (N, 6)
    chunk                    : buildings per GPU batch (reduce if OOM)

    Returns
    -------
    heat_demand : np.ndarray (N, 365)  — kWh/day, NaN prefix as in predict_annual
    """
    weather_365              = np.asarray(weather_365,              dtype=np.float32)
    building_features_matrix = np.asarray(building_features_matrix, dtype=np.float32)

    N     = building_features_matrix.shape[0]
    start = WARMUP_DAYS + LOOKBACK
    idx   = np.arange(start, 365)
    n_pred = len(idx)

    W_sc   = ss.transform(weather_365)                          # (365, 10)
    Xt_all = st.transform(building_features_matrix).astype(np.float32)  # (N, 6)
    Xs_base = np.stack([W_sc[t - LOOKBACK:t] for t in idx]).astype(np.float32)  # (n_pred, 14, 10)

    results = np.full((N, 365), np.nan, dtype=np.float32)
    model.eval()

    for b0 in range(0, N, chunk):
        b1   = min(b0 + chunk, N)
        csz  = b1 - b0
        areas = building_features_matrix[b0:b1, 1]             # (csz,)

        Xs = np.repeat(Xs_base[np.newaxis], csz, axis=0) \
               .reshape(csz * n_pred, LOOKBACK, len(SEQ_COLS))  # (csz*n_pred, 14, 10)
        Xt = np.repeat(Xt_all[b0:b1], n_pred, axis=0)          # (csz*n_pred, 6)

        with torch.no_grad():
            pr = model(
                torch.tensor(Xs).to(DEVICE),
                torch.tensor(Xt).to(DEVICE),
            ).cpu().numpy().reshape(csz, n_pred)

        pred_per_area = np.clip(
            sy.inverse_transform(pr.reshape(-1, 1)).ravel(), 0, None
        ).reshape(csz, n_pred)

        results[b0:b1, start:] = pred_per_area * areas[:, np.newaxis]

    return results



# 4. Training


# Load and split data
raw = pd.read_csv(DATA_CSV)
ids = np.array(sorted(raw.building_id.unique()))
print(f"{len(ids)} buildings loaded")

rng = np.random.default_rng(SEED)
rng.shuffle(ids)
TR   = list(ids[:N_TRAIN])
rest = list(ids[N_TRAIN:]);  rng.shuffle(rest)
VA   = list(rest[:N_VAL])
rest2 = rest[N_VAL:]
TE   = list(rest2) if N_TEST is None else list(rest2[:N_TEST])
print(f"Split  →  train {len(TR)}  /  val {len(VA)}  /  test {len(TE)}")

by  = {b: g for b, g in raw.groupby("building_id")}
MIN = WARMUP_DAYS + LOOKBACK + 1
TR  = [b for b in TR if len(by[b]) >= MIN]
VA  = [b for b in VA if len(by[b]) >= MIN]
TE  = [b for b in TE if len(by[b]) >= MIN]

# Fit scalers on training set only
trw = [make_windows(by[b]) for b in TR]
ss, st, sy = StandardScaler(), StandardScaler(), StandardScaler()
ss.fit(np.concatenate([w[0].reshape(-1, len(SEQ_COLS)) for w in trw]))
st.fit(np.concatenate([w[1] for w in trw]))
sy.fit(np.concatenate([w[2] for w in trw]).reshape(-1, 1))

def apply_scalers(windows):
    Xs, Xt, Y, A = [], [], [], []
    for xs, xt, y, a in windows:
        if len(y) == 0:
            continue
        n, L, F = xs.shape
        Xs.append(ss.transform(xs.reshape(-1, F)).reshape(n, L, F))
        Xt.append(st.transform(xt))
        Y.append(sy.transform(y.reshape(-1, 1)).ravel())
        A.append(a)
    return (np.concatenate(Xs).astype(np.float32),
            np.concatenate(Xt).astype(np.float32),
            np.concatenate(Y).astype(np.float32),
            np.concatenate(A).astype(np.float32))

Xs_tr, Xt_tr, Y_tr, _    = apply_scalers(trw)
Xs_va, Xt_va, Y_va, _    = apply_scalers([make_windows(by[b]) for b in VA])
Xs_te, Xt_te, Y_te, A_te = apply_scalers([make_windows(by[b]) for b in TE])
print(f"Windows  →  train {len(Y_tr):,}  /  val {len(Y_va):,}  /  test {len(Y_te):,}")

# DataLoaders
train_loader = DataLoader(WindowDataset(Xs_tr, Xt_tr, Y_tr), batch_size=BATCH, shuffle=True)
val_loader   = DataLoader(WindowDataset(Xs_va, Xt_va, Y_va), batch_size=BATCH)

# Model, optimiser, scheduler
model = HeatDemandLSTM(len(SEQ_COLS), len(STAT_COLS)).to(DEVICE)
opt   = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
sch   = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=5)
loss_fn = nn.MSELoss()

# Training loop (early stopping on validation loss)
best_val_loss = np.inf
best_state    = None
wait          = 0

for ep in range(1, EPOCHS + 1):
    # train
    model.train()
    tl, tn = 0.0, 0
    for xs, xt, y in train_loader:
        xs, xt, y = xs.to(DEVICE), xt.to(DEVICE), y.to(DEVICE)
        opt.zero_grad()
        loss = loss_fn(model(xs, xt), y)
        loss.backward()
        opt.step()
        tl += loss.item() * len(y);  tn += len(y)
    tl /= tn

    # validate
    model.eval()
    vl, vn = 0.0, 0
    with torch.no_grad():
        for xs, xt, y in val_loader:
            xs, xt, y = xs.to(DEVICE), xt.to(DEVICE), y.to(DEVICE)
            vl += loss_fn(model(xs, xt), y).item() * len(y);  vn += len(y)
    vl /= vn
    sch.step(vl)

    marker = ""
    if vl < best_val_loss - 1e-6:
        best_val_loss = vl
        best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        wait          = 0
        marker        = "  ← best"
    else:
        wait += 1

    print(f"ep {ep:3d}  train {tl:.4f}  val {vl:.4f}{marker}")
    if wait >= PATIENCE:
        print("Early stopping.")
        break

# restore best checkpoint (selected on validation loss)
model.load_state_dict(best_state)
model.eval()

# 5. Test evaluation  (test set touched exactly once)
pr = []
with torch.no_grad():
    for i in range(0, len(Xs_te), 4096):
        pr.append(model(
            torch.tensor(Xs_te[i:i+4096]).to(DEVICE),
            torch.tensor(Xt_te[i:i+4096]).to(DEVICE),
        ).cpu().numpy())
pr   = np.concatenate(pr)
pred = np.clip(sy.inverse_transform(pr.reshape(-1, 1)).ravel(), 0, None) * A_te
true = sy.inverse_transform(Y_te.reshape(-1, 1)).ravel() * A_te

rmse = np.sqrt(np.mean((pred - true) ** 2))
cv   = 100 * rmse / true.mean()
nmbe = 100 * (pred - true).mean() / true.mean()

summ = (
    f"=== Aachen DAILY surrogate — test set ===\n"
    f"Buildings : {len(TE)}\n"
    f"Days      : {len(Y_te):,}\n"
    f"RMSE      : {rmse:.2f} kWh/day\n"
    f"CV(RMSE)  : {cv:.2f}%  (ASHRAE threshold 30%)\n"
    f"NMBE      : {nmbe:+.2f}%  (ASHRAE threshold ±10%)\n"
)
print("\n" + summ)

# 6. Save artefacts

torch.save(model.state_dict(), f"{OUT_DIR}/model.pt")
pickle.dump({"seq": ss, "static": st, "y": sy}, open(f"{OUT_DIR}/scalers.pkl", "wb"))
open(f"{OUT_DIR}/metrics.txt", "w").write(summ)

#  Surrogate validation plot for one test building
b0  = TE[0]
df0 = by[b0].sort_values("day")

weather_365 = df0[SEQ_COLS].to_numpy(dtype=np.float32)           # (365, 10)
bfeat       = df0[STAT_COLS].iloc[0].to_numpy(dtype=np.float32)  # (6,)
true_annual = df0["daily_heat_per_area"].to_numpy(np.float32) * float(bfeat[1])  # kWh/day

# call the surrogate with full-year I/O (the simulation-equivalent interface)
pred_annual = predict_annual(model, ss, st, sy, weather_365, bfeat)

start = WARMUP_DAYS + LOOKBACK
days  = np.arange(365)
plt.figure(figsize=(12, 4))
plt.plot(days[start:], true_annual[start:],  label="Simulation", lw=1.4)
plt.plot(days[start:], pred_annual[start:],  label="Surrogate",  lw=1.1, alpha=0.8)
plt.ylabel("Daily heat demand [kWh/day]")
plt.xlabel("Day of year")
plt.title(f"Surrogate vs. simulation — test building {b0}")
plt.legend(); plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/predictions.png", dpi=130)

print(f"Artefacts saved to {OUT_DIR}")
print("\nSurrogate I/O contract:")
print(f"  INPUT  : weather (365, {len(SEQ_COLS)})  +  building ({len(STAT_COLS)},)")
print(f"  OUTPUT : heat demand (365,) kWh/day")
print("  (mirrors EDpyFlow / OpenModelica simulation interface)")
