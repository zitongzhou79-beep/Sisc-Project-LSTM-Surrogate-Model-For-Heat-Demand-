
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
N_TRAIN, N_VAL, N_TEST = 1200, 400, None  # N_TEST=None → use all remaining buildings
LOOKBACK    = 14    # days of weather history before the target day
SEQ_LEN     = LOOKBACK + 1   # window length: past LOOKBACK days + the current day
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

# Static features split into numeric (standardised) and one-hot (NOT standardised).
# building_type is one-hot encoded across all four typologies: AB, MFH, SFH, TH.
STAT_NUM_COLS = [
    "construction_year", "net_leased_area", "num_floors",
    "floor_height", "refurb_level",
]
BUILDING_TYPES = ["AB", "MFH", "SFH", "TH"]                 # fixed order
TYPE_OHE_COLS  = [f"type_{t}" for t in BUILDING_TYPES]
STAT_COLS      = STAT_NUM_COLS + TYPE_OHE_COLS              # full static feature order

# Position of net_leased_area within STAT_COLS (used for kWh conversion at inference)
AREA_IDX = STAT_COLS.index("net_leased_area")

TARGET = "daily_heat_per_area"   # W/m²; multiplied by area → kWh/day at inference

# Reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(OUT_DIR, exist_ok=True)
print("Device:", DEVICE)



# 1. Data utilities

def add_type_onehot(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add the four one-hot type columns (type_AB, type_MFH, type_SFH, type_TH)
    to a copy of df, derived from the 'building_type' column. Ensures all four
    columns always exist and are in the fixed BUILDING_TYPES order.
    """
    df = df.copy()
    for t, col in zip(BUILDING_TYPES, TYPE_OHE_COLS):
        df[col] = (df["building_type"] == t).astype(np.float32)
    return df


def make_windows(df: pd.DataFrame):
    """
    Convert a single building's time-series DataFrame into sliding-window arrays.

    Returns (Xs, Xt, Y, A) where:
      Xs : (N, SEQ_LEN, len(SEQ_COLS))  — weather sequences (incl. current day)
      Xt : (N, len(STAT_COLS))          — static features (repeated per window)
      Y  : (N,)                         — target heat per area [W/m²]
      A  : (N,)                         — net_leased_area [m²] for kWh conversion
    """
    df = df.sort_values("day").reset_index(drop=True)
    if len(df) <= LOOKBACK:
        empty = lambda s: np.empty(s, np.float32)
        return (empty((0, SEQ_LEN, len(SEQ_COLS))), empty((0, len(STAT_COLS))),
                empty((0,)), empty((0,)))

    W = df[SEQ_COLS].to_numpy(np.float32)
    S = df[STAT_COLS].to_numpy(np.float32)
    y = df[TARGET].to_numpy(np.float32)
    a = df["net_leased_area"].to_numpy(np.float32)

    # circular padding: prepend the last LOOKBACK days to the front
    W_pad = np.concatenate([W[-LOOKBACK:], W], axis=0)   # (len(df)+LOOKBACK, F)

    Xs, Xt, Y, A = [], [], [], []
    for t in range(len(df)):                 # predict every day t = 0 … len-1
        # window = padded[t … t+LOOKBACK] inclusive → past LOOKBACK days + current day
        Xs.append(W_pad[t : t + SEQ_LEN])
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

    Branch 1: LSTM processes the weather sequence (past LOOKBACK days + current day).
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



# 3. Static-feature scaling (numeric only; one-hot passed through unchanged)


def scale_static(st: StandardScaler, Xt: np.ndarray) -> np.ndarray:
    """
    Standardise only the numeric static columns; leave the one-hot type columns
    untouched. `st` is fitted on the numeric block alone.

    Xt : (N, len(STAT_COLS)) with numeric columns first, then one-hot columns.
    """
    Xt = np.asarray(Xt, dtype=np.float32)
    n_num = len(STAT_NUM_COLS)
    num   = st.transform(Xt[:, :n_num]).astype(np.float32)
    ohe   = Xt[:, n_num:]
    return np.concatenate([num, ohe], axis=1).astype(np.float32)



# 4. Surrogate inference interface
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
    ss, st, sy        : fitted StandardScalers (weather, numeric-static, target)
    weather_365       : np.ndarray (365, 10)  — one full TRY year
    building_features : np.ndarray (len(STAT_COLS),) — numeric features then 4 one-hot

    Returns
    -------
    heat_demand : np.ndarray (365,)
        Daily heat demand in kWh/day for every day 0 … 364 (no NaN prefix; the
        calendar is wrapped so the first two weeks are predictable too).
    """
    weather_365       = np.asarray(weather_365,       dtype=np.float32)
    building_features = np.asarray(building_features, dtype=np.float32)

    assert weather_365.shape       == (365, len(SEQ_COLS)),  \
        f"Expected (365, {len(SEQ_COLS)}), got {weather_365.shape}"
    assert building_features.shape == (len(STAT_COLS),), \
        f"Expected ({len(STAT_COLS)},), got {building_features.shape}"

    area  = float(building_features[AREA_IDX])   # net_leased_area → kWh conversion

    W_sc  = ss.transform(weather_365)            # (365, 10) scaled
    xt_sc = scale_static(st, building_features.reshape(1, -1))                 # (1, n_stat)

    # circular padding, identical to make_windows
    W_pad = np.concatenate([W_sc[-LOOKBACK:], W_sc], axis=0)                   # (365+LB, 10)
    idx   = np.arange(365)
    Xs    = np.stack([W_pad[t : t + SEQ_LEN] for t in idx]).astype(np.float32) # (365, 15, 10)
    Xt    = np.repeat(xt_sc, len(idx), axis=0)                                 # (365, n_stat)

    model.eval()
    with torch.no_grad():
        pr = model(
            torch.tensor(Xs).to(DEVICE),
            torch.tensor(Xt).to(DEVICE),
        ).cpu().numpy()

    pred_per_area = np.clip(sy.inverse_transform(pr.reshape(-1, 1)).ravel(), 0, None)
    return (pred_per_area * area).astype(np.float32)


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
    building_features_matrix : np.ndarray (N, len(STAT_COLS))
    chunk                    : buildings per GPU batch (reduce if OOM)

    Returns
    -------
    heat_demand : np.ndarray (N, 365)  — kWh/day for every day (no NaN prefix)
    """
    weather_365              = np.asarray(weather_365,              dtype=np.float32)
    building_features_matrix = np.asarray(building_features_matrix, dtype=np.float32)

    N      = building_features_matrix.shape[0]
    idx    = np.arange(365)
    n_pred = len(idx)

    W_sc    = ss.transform(weather_365)                                        # (365, 10)
    Xt_all  = scale_static(st, building_features_matrix)                       # (N, n_stat)
    W_pad   = np.concatenate([W_sc[-LOOKBACK:], W_sc], axis=0)                 # (365+LB, 10)
    Xs_base = np.stack([W_pad[t : t + SEQ_LEN] for t in idx]).astype(np.float32)  # (365,15,10)

    results = np.full((N, 365), np.nan, dtype=np.float32)
    model.eval()

    for b0 in range(0, N, chunk):
        b1   = min(b0 + chunk, N)
        csz  = b1 - b0
        areas = building_features_matrix[b0:b1, AREA_IDX]      # (csz,)

        Xs = np.repeat(Xs_base[np.newaxis], csz, axis=0) \
               .reshape(csz * n_pred, SEQ_LEN, len(SEQ_COLS))  # (csz*n_pred, 15, 10)
        Xt = np.repeat(Xt_all[b0:b1], n_pred, axis=0)          # (csz*n_pred, n_stat)

        with torch.no_grad():
            pr = model(
                torch.tensor(Xs).to(DEVICE),
                torch.tensor(Xt).to(DEVICE),
            ).cpu().numpy().reshape(csz, n_pred)

        pred_per_area = np.clip(
            sy.inverse_transform(pr.reshape(-1, 1)).ravel(), 0, None
        ).reshape(csz, n_pred)

        results[b0:b1, :] = pred_per_area * areas[:, np.newaxis]

    return results



# 5. Training


# Load and split data
raw = pd.read_csv(DATA_CSV)
raw = add_type_onehot(raw)                       # adds type_AB / type_MFH / type_SFH / type_TH
ids = np.array(sorted(raw.building_id.unique()))
print(f"{len(ids)} buildings loaded")
print("Type counts:", raw.drop_duplicates('building_id')['building_type'].value_counts().to_dict())

rng = np.random.default_rng(SEED)
rng.shuffle(ids)
TR   = list(ids[:N_TRAIN])
rest = list(ids[N_TRAIN:]);  rng.shuffle(rest)
VA   = list(rest[:N_VAL])
rest2 = rest[N_VAL:]
TE   = list(rest2) if N_TEST is None else list(rest2[:N_TEST])
print(f"Split  →  train {len(TR)}  /  val {len(VA)}  /  test {len(TE)}")

by  = {b: g for b, g in raw.groupby("building_id")}
MIN = LOOKBACK + 1                               # warm-up removed; only need a full window
TR  = [b for b in TR if len(by[b]) >= MIN]
VA  = [b for b in VA if len(by[b]) >= MIN]
TE  = [b for b in TE if len(by[b]) >= MIN]

# Fit scalers on training set only
trw = [make_windows(by[b]) for b in TR]
ss, st, sy = StandardScaler(), StandardScaler(), StandardScaler()
ss.fit(np.concatenate([w[0].reshape(-1, len(SEQ_COLS)) for w in trw]))
# static scaler is fitted on the NUMERIC block only (first len(STAT_NUM_COLS) cols)
st.fit(np.concatenate([w[1][:, :len(STAT_NUM_COLS)] for w in trw]))
sy.fit(np.concatenate([w[2] for w in trw]).reshape(-1, 1))

def apply_scalers(windows):
    Xs, Xt, Y, A = [], [], [], []
    for xs, xt, y, a in windows:
        if len(y) == 0:
            continue
        n, L, F = xs.shape
        Xs.append(ss.transform(xs.reshape(-1, F)).reshape(n, L, F))
        Xt.append(scale_static(st, xt))          # numeric scaled, one-hot passthrough
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

# 6. Test evaluation  (test set touched exactly once)
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

# --- Per-type breakdown -------------------------------------------------
# Recover each test window's building type from the one-hot block in Xt_te
# (last len(BUILDING_TYPES) columns, passed through un-standardised as 0/1).
n_num     = len(STAT_NUM_COLS)
ohe_block = Xt_te[:, n_num:]                       # (n_windows, 4) one-hot
type_idx  = ohe_block.argmax(axis=1)              # window → type index

per_type_lines = ["\n--- Per building type ---"]
per_type_lines.append(f"{'Type':<6}{'Bldgs':>7}{'Days':>12}{'CV(RMSE)':>11}{'NMBE':>9}")
# building-count per type (over test buildings, not windows)
te_type_count = {t: 0 for t in BUILDING_TYPES}
for b in TE:
    te_type_count[str(by[b]['building_type'].iloc[0])] += 1

for k, t in enumerate(BUILDING_TYPES):
    mask = (type_idx == k)
    n_days = int(mask.sum())
    if n_days == 0:
        per_type_lines.append(f"{t:<6}{te_type_count[t]:>7}{0:>12}{'n/a':>11}{'n/a':>9}")
        continue
    p_t = pred[mask]; y_t = true[mask]
    rmse_t = np.sqrt(np.mean((p_t - y_t) ** 2))
    cv_t   = 100 * rmse_t / y_t.mean()
    nmbe_t = 100 * (p_t - y_t).mean() / y_t.mean()
    per_type_lines.append(
        f"{t:<6}{te_type_count[t]:>7}{n_days:>12,}{cv_t:>10.2f}%{nmbe_t:>+8.2f}%"
    )

summ += "\n".join(per_type_lines) + "\n"
print("\n" + summ)

# 7. Save artefacts

torch.save(model.state_dict(), f"{OUT_DIR}/model.pt")
pickle.dump({"seq": ss, "static": st, "y": sy}, open(f"{OUT_DIR}/scalers.pkl", "wb"))
open(f"{OUT_DIR}/metrics.txt", "w").write(summ)

#  Surrogate validation plots — one representative TEST building per building type.
#  Picks, for each typology (AB, MFH, SFH, TH), the first test building of that
#  type and plots full-year Simulation vs. Surrogate. Saves four individual PNGs
#  plus a 2x2 summary figure.

def building_type_of(b):
    """Return the building_type string for a building id from its rows."""
    return str(by[b]["building_type"].iloc[0])

# map each type → first available test building of that type
type_to_building = {}
for b in TE:
    t = building_type_of(b)
    if t in BUILDING_TYPES and t not in type_to_building:
        type_to_building[t] = b
    if len(type_to_building) == len(BUILDING_TYPES):
        break

days = np.arange(365)

def surrogate_curves(b):
    """Return (true_annual, pred_annual) in kWh/day for one building."""
    dfb   = by[b].sort_values("day")
    w365  = dfb[SEQ_COLS].to_numpy(dtype=np.float32)             # (365, 10)
    bfeat = dfb[STAT_COLS].iloc[0].to_numpy(dtype=np.float32)    # (len(STAT_COLS),)
    true_a = dfb["daily_heat_per_area"].to_numpy(np.float32) * float(bfeat[AREA_IDX])
    pred_a = predict_annual(model, ss, st, sy, w365, bfeat)
    return true_a, pred_a

# 2x2 summary figure + individual figures
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
for ax, t in zip(axes.ravel(), BUILDING_TYPES):
    b = type_to_building.get(t)
    if b is None:
        ax.set_title(f"{t}: no test building")
        ax.axis("off")
        continue

    true_a, pred_a = surrogate_curves(b)

    # summary panel
    ax.plot(days, true_a, label="Simulation", lw=1.4)
    ax.plot(days, pred_a, label="Surrogate",  lw=1.1, alpha=0.8)
    ax.set_title(f"{t} — building {b}")
    ax.set_xlabel("Day of year")
    ax.set_ylabel("Heat demand [kWh/day]")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # individual figure
    plt.figure(figsize=(12, 4))
    plt.plot(days, true_a, label="Simulation", lw=1.4)
    plt.plot(days, pred_a, label="Surrogate",  lw=1.1, alpha=0.8)
    plt.ylabel("Daily heat demand [kWh/day]")
    plt.xlabel("Day of year")
    plt.title(f"Surrogate vs. simulation — {t} (test building {b})")
    plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/predictions_{t}.png", dpi=130)
    plt.close()

fig.suptitle("Surrogate vs. simulation by building type (test set)", y=1.02)
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/predictions.png", dpi=130, bbox_inches="tight")
plt.close(fig)
print("Saved per-type plots:",
      ", ".join(f"predictions_{t}.png" for t in BUILDING_TYPES),
      "and predictions.png (2x2 summary)")

print(f"Artefacts saved to {OUT_DIR}")
print("\nSurrogate I/O contract:")
print(f"  INPUT  : weather (365, {len(SEQ_COLS)})  +  building ({len(STAT_COLS)},)")
print(f"  OUTPUT : heat demand (365,) kWh/day")
print("  (mirrors EDpyFlow / OpenModelica simulation interface)")
