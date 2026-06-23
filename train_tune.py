"""
train_tune.py
=============
Hyperparameter sensitivity analysis for the Aachen heat demand surrogate.

Trains the same dual-branch LSTM as train_aachen_daily.py with configurable
hyperparameters supplied via command-line arguments.

Hyperparameter selection rule
------------------------------
All decisions (early stopping, model checkpointing) are made using the
VALIDATION set. The test set is evaluated once at the end and is never
used for selection.

Usage examples
--------------
    # baseline
    python train_tune.py

    # larger model, more training data
    python train_tune.py --ntrain 1500 --lookback 21 --hidden 96

    # custom seed for reproducibility check
    python train_tune.py --seed 0
"""

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

# ── CLI ───────────────────────────────────────────────────────────────────────
ap = argparse.ArgumentParser(description="Hyperparameter sweep for heat demand LSTM")
ap.add_argument("--ntrain",   type=int, default=800,  help="Training buildings")
ap.add_argument("--nval",     type=int, default=400,  help="Validation buildings")
ap.add_argument("--lookback", type=int, default=14,   help="Sliding-window length (days)")
ap.add_argument("--hidden",   type=int, default=64,   help="LSTM hidden size")
ap.add_argument("--layers",   type=int, default=2,    help="LSTM layers")
ap.add_argument("--epochs",   type=int, default=120,  help="Max training epochs")
ap.add_argument("--seed",     type=int, default=42,   help="Random seed")
a = ap.parse_args()

# ── Fixed constants ───────────────────────────────────────────────────────────
DATA_CSV    = "D:/aachen_dataset_daily.csv"
WARMUP      = 14
DROPOUT     = 0.25
LR          = 1e-3
BATCH       = 256
PATIENCE    = 15

SEQ_COLS  = ["T_out","sol_global","sol_diffuse","sol_direct","rel_hum",
             "wind_speed","cloud_opaque","T_dew","doy_sin","doy_cos"]
STAT_COLS = ["construction_year","net_leased_area","num_floors",
             "floor_height","is_MFH","refurb_level"]
TARGET    = "daily_heat_per_area"

torch.manual_seed(a.seed);  np.random.seed(a.seed)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device {DEV} | ntrain={a.ntrain} lookback={a.lookback} "
      f"hidden={a.hidden} layers={a.layers} seed={a.seed}")


# ── Data ──────────────────────────────────────────────────────────────────────
def make_windows(df, lb):
    df = df.sort_values("day").iloc[WARMUP:].reset_index(drop=True)
    if len(df) <= lb:
        return None
    W = df[SEQ_COLS].to_numpy(np.float32)
    S = df[STAT_COLS].to_numpy(np.float32)
    y = df[TARGET].to_numpy(np.float32)
    ar = df["net_leased_area"].to_numpy(np.float32)
    Xs, Xt, Y, A = [], [], [], []
    for t in range(lb, len(df)):
        Xs.append(W[t-lb:t]);  Xt.append(S[t]);  Y.append(y[t]);  A.append(ar[t])
    return (np.asarray(Xs,np.float32), np.asarray(Xt,np.float32),
            np.asarray(Y,np.float32),  np.asarray(A,np.float32))


class DS(Dataset):
    def __init__(self, Xs, Xt, Y):
        self.Xs=torch.tensor(Xs); self.Xt=torch.tensor(Xt); self.Y=torch.tensor(Y)
    def __len__(self): return len(self.Y)
    def __getitem__(self, i): return self.Xs[i], self.Xt[i], self.Y[i]


# ── Model ─────────────────────────────────────────────────────────────────────
class HeatDemandLSTM(nn.Module):
    def __init__(self, ns, nt, h, l):
        super().__init__()
        self.lstm = nn.LSTM(ns, h, l, batch_first=True, dropout=DROPOUT)
        self.head = nn.Sequential(
            nn.Linear(h+nt, 64), nn.ReLU(), nn.Dropout(DROPOUT), nn.Linear(64, 1)
        )
    def forward(self, xs, xt):
        o, _ = self.lstm(xs)
        return self.head(torch.cat([o[:,-1,:], xt], dim=1)).squeeze(-1)


# ── Load & split ──────────────────────────────────────────────────────────────
raw = pd.read_csv(DATA_CSV)
ids = np.array(sorted(raw.building_id.unique()))
rng = np.random.default_rng(a.seed);  rng.shuffle(ids)
TR  = list(ids[:a.ntrain]);  rest = list(ids[a.ntrain:]);  rng.shuffle(rest)
VA  = list(rest[:a.nval]);   TE   = list(rest[a.nval:])
by  = {b: g for b, g in raw.groupby("building_id")}
MIN = WARMUP + a.lookback + 1
TR  = [b for b in TR if len(by[b]) >= MIN]
VA  = [b for b in VA if len(by[b]) >= MIN]
TE  = [b for b in TE if len(by[b]) >= MIN]
print(f"Split  →  train {len(TR)}  /  val {len(VA)}  /  test {len(TE)}")

# ── Scalers (fit on train only) ───────────────────────────────────────────────
trw = [make_windows(by[b], a.lookback) for b in TR]
trw = [w for w in trw if w]
ss, st, sy = StandardScaler(), StandardScaler(), StandardScaler()
ss.fit(np.concatenate([w[0].reshape(-1, len(SEQ_COLS)) for w in trw]))
st.fit(np.concatenate([w[1] for w in trw]))
sy.fit(np.concatenate([w[2] for w in trw]).reshape(-1, 1))

def scale(bids):
    Xs, Xt, Y, A = [], [], [], []
    for b in bids:
        w = make_windows(by[b], a.lookback)
        if not w or len(w[2]) == 0: continue
        xs, xt, y, ar = w
        n, L, F = xs.shape
        Xs.append(ss.transform(xs.reshape(-1,F)).reshape(n,L,F))
        Xt.append(st.transform(xt))
        Y.append(sy.transform(y.reshape(-1,1)).ravel())
        A.append(ar)
    return (np.concatenate(Xs).astype(np.float32), np.concatenate(Xt).astype(np.float32),
            np.concatenate(Y).astype(np.float32),  np.concatenate(A).astype(np.float32))

Xtr, Ttr, Ytr, _   = scale(TR)
Xva, Tva, Yva, _   = scale(VA)
Xte, Tte, Yte, Ate = scale(TE)
print(f"Windows  →  train {len(Ytr):,}  /  val {len(Yva):,}  /  test {len(Yte):,}")

# ── Train ─────────────────────────────────────────────────────────────────────
trl = DataLoader(DS(Xtr,Ttr,Ytr), batch_size=BATCH, shuffle=True)
vl  = DataLoader(DS(Xva,Tva,Yva), batch_size=BATCH)
m   = HeatDemandLSTM(len(SEQ_COLS), len(STAT_COLS), a.hidden, a.layers).to(DEV)
op  = torch.optim.Adam(m.parameters(), lr=LR, weight_decay=1e-5)
sch = torch.optim.lr_scheduler.ReduceLROnPlateau(op, factor=0.5, patience=5)
lf  = nn.MSELoss()

best, bs, wait = np.inf, None, 0
for ep in range(1, a.epochs + 1):
    m.train()
    for xs, xt, y in trl:
        xs,xt,y = xs.to(DEV),xt.to(DEV),y.to(DEV)
        op.zero_grad();  lf(m(xs,xt),y).backward();  op.step()

    m.eval();  v = 0;  n = 0
    with torch.no_grad():
        for xs,xt,y in vl:
            xs,xt,y = xs.to(DEV),xt.to(DEV),y.to(DEV)
            v += lf(m(xs,xt),y).item()*len(y);  n += len(y)
    v /= n;  sch.step(v)

    marker = ""
    if v < best - 1e-6:
        best, bs, wait = v, {k: val.cpu().clone() for k,val in m.state_dict().items()}, 0
        marker = "  ← best val"
    else:
        wait += 1

    if ep % 10 == 0 or wait >= PATIENCE:
        print(f"ep {ep:3d}  val {v:.4f}{marker}")
    if wait >= PATIENCE:
        print("Early stopping.");  break

# ── Test (once, after hyperparameter selection via val) ───────────────────────
m.load_state_dict(bs);  m.eval()
pr = []
with torch.no_grad():
    for i in range(0, len(Xte), 4096):
        pr.append(m(torch.tensor(Xte[i:i+4096]).to(DEV),
                    torch.tensor(Tte[i:i+4096]).to(DEV)).cpu().numpy())
pr   = np.concatenate(pr)
pred = np.clip(sy.inverse_transform(pr.reshape(-1,1)).ravel(), 0, None) * Ate
true = sy.inverse_transform(Yte.reshape(-1,1)).ravel() * Ate

rmse = np.sqrt(np.mean((pred-true)**2))
cv   = 100 * rmse / true.mean()
nmbe = 100 * (pred-true).mean() / true.mean()

print(f"\n>>> ntrain={a.ntrain} lookback={a.lookback} hidden={a.hidden} "
      f"seed={a.seed}")
print(f"    CV(RMSE)={cv:.2f}%   NMBE={nmbe:+.2f}%   RMSE={rmse:.2f} kWh/day")
print(f"    test: {len(TE)} buildings, {len(Yte):,} days")
