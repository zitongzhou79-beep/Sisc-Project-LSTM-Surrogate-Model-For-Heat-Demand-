import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

# CLI
ap = argparse.ArgumentParser(description="Hyperparameter sweep for heat demand LSTM")
ap.add_argument("--ntrain",   type=int, default=800,  help="Training buildings")
ap.add_argument("--nval",     type=int, default=400,  help="Validation buildings")
ap.add_argument("--lookback", type=int, default=14,   help="Days of weather history before target day")
ap.add_argument("--hidden",   type=int, default=64,   help="LSTM hidden size")
ap.add_argument("--layers",   type=int, default=2,    help="LSTM layers")
ap.add_argument("--epochs",   type=int, default=120,  help="Max training epochs")
ap.add_argument("--seed",     type=int, default=42,   help="Random seed")
a = ap.parse_args()

# window length = past `lookback` days + current day
SEQ_LEN = a.lookback + 1

# Fixed constants
DATA_CSV    = "D:/surrogate_project/aachen_dataset_daily.csv"
DROPOUT     = 0.25
LR          = 1e-3
BATCH       = 256
PATIENCE    = 15

SEQ_COLS  = ["T_out","sol_global","sol_diffuse","sol_direct","rel_hum",
             "wind_speed","cloud_opaque","T_dew","doy_sin","doy_cos"]

# static features: numeric (standardised) + one-hot building type (NOT standardised)
STAT_NUM_COLS = ["construction_year","net_leased_area","num_floors",
                 "floor_height","refurb_level"]
BUILDING_TYPES = ["AB", "MFH", "SFH", "TH"]
TYPE_OHE_COLS  = [f"type_{t}" for t in BUILDING_TYPES]
STAT_COLS      = STAT_NUM_COLS + TYPE_OHE_COLS
AREA_IDX       = STAT_COLS.index("net_leased_area")

TARGET    = "daily_heat_per_area"

torch.manual_seed(a.seed)
np.random.seed(a.seed)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device {DEV} | ntrain={a.ntrain} lookback={a.lookback} "
      f"hidden={a.hidden} layers={a.layers} seed={a.seed}")


# Data
def add_type_onehot(df):
    df = df.copy()
    for t, col in zip(BUILDING_TYPES, TYPE_OHE_COLS):
        df[col] = (df["building_type"] == t).astype(np.float32)
    return df


def make_windows(df, lb):
    # circular padding → no warm-up dropped; every day predictable; window includes current day
    df = df.sort_values("day").reset_index(drop=True)
    if len(df) <= lb:
        return None
    seq_len = lb + 1
    W = df[SEQ_COLS].to_numpy(np.float32)
    S = df[STAT_COLS].to_numpy(np.float32)
    y = df[TARGET].to_numpy(np.float32)
    ar = df["net_leased_area"].to_numpy(np.float32)
    W_pad = np.concatenate([W[-lb:], W], axis=0)
    Xs, Xt, Y, A = [], [], [], []
    for t in range(len(df)):
        Xs.append(W_pad[t:t+seq_len])
        Xt.append(S[t])
        Y.append(y[t])
        A.append(ar[t])
    return (np.asarray(Xs,np.float32), np.asarray(Xt,np.float32),
            np.asarray(Y,np.float32),  np.asarray(A,np.float32))


def scale_static(st, Xt):
    Xt = np.asarray(Xt, np.float32)
    n_num = len(STAT_NUM_COLS)
    num = st.transform(Xt[:, :n_num]).astype(np.float32)
    ohe = Xt[:, n_num:]
    return np.concatenate([num, ohe], axis=1).astype(np.float32)


class HeatDemandDataset(Dataset):
    def __init__(self, Xs, Xt, Y):
        self.Xs=torch.tensor(Xs)
        self.Xt=torch.tensor(Xt)
        self.Y=torch.tensor(Y)
    def __len__(self): return len(self.Y)
    def __getitem__(self, i):
        return self.Xs[i], self.Xt[i], self.Y[i]


# Model
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


# Load & split
raw = pd.read_csv(DATA_CSV)
raw = add_type_onehot(raw)
ids = np.array(sorted(raw.building_id.unique()))
rng = np.random.default_rng(a.seed);  rng.shuffle(ids)
TR  = list(ids[:a.ntrain]);  rest = list(ids[a.ntrain:]);  rng.shuffle(rest)
VA  = list(rest[:a.nval]);   TE   = list(rest[a.nval:])
by  = {b: g for b, g in raw.groupby("building_id")}
MIN = a.lookback + 1
TR  = [b for b in TR if len(by[b]) >= MIN]
VA  = [b for b in VA if len(by[b]) >= MIN]
TE  = [b for b in TE if len(by[b]) >= MIN]
print(f"Split  →  train {len(TR)}  /  val {len(VA)}  /  test {len(TE)}")

# Scalers
trw = [make_windows(by[b], a.lookback) for b in TR]
trw = [w for w in trw if w is not None]
ss, st, sy = StandardScaler(), StandardScaler(), StandardScaler()
ss.fit(np.concatenate([w[0].reshape(-1, len(SEQ_COLS)) for w in trw]))
st.fit(np.concatenate([w[1][:, :len(STAT_NUM_COLS)] for w in trw]))   # numeric block only
sy.fit(np.concatenate([w[2] for w in trw]).reshape(-1, 1))

def scale(bids):
    Xs, Xt, Y, A = [], [], [], []
    for b in bids:
        w = make_windows(by[b], a.lookback)
        if w is None or len(w[2]) == 0: continue
        xs, xt, y, ar = w
        n, L, F = xs.shape
        Xs.append(ss.transform(xs.reshape(-1,F)).reshape(n,L,F))
        Xt.append(scale_static(st, xt))
        Y.append(sy.transform(y.reshape(-1,1)).ravel())
        A.append(ar)
    return (np.concatenate(Xs).astype(np.float32), np.concatenate(Xt).astype(np.float32),
            np.concatenate(Y).astype(np.float32),  np.concatenate(A).astype(np.float32))

Xtr, Ttr, Ytr, _   = scale(TR)
Xva, Tva, Yva, _   = scale(VA)
Xte, Tte, Yte, Ate = scale(TE)
print(f"Windows  →  train {len(Ytr):,}  /  val {len(Yva):,}  /  test {len(Yte):,}")

# Train
trl = DataLoader(HeatDemandDataset(Xtr,Ttr,Ytr), batch_size=BATCH, shuffle=True)
vl  = DataLoader(HeatDemandDataset(Xva,Tva,Yva), batch_size=BATCH)
m   = HeatDemandLSTM(len(SEQ_COLS), len(STAT_COLS), a.hidden, a.layers).to(DEV)
op  = torch.optim.Adam(m.parameters(), lr=LR, weight_decay=1e-5)
sch = torch.optim.lr_scheduler.ReduceLROnPlateau(op, factor=0.5, patience=5)
lf  = nn.MSELoss()

best, bs, wait = np.inf, None, 0
for ep in range(1, a.epochs + 1):
    m.train()
    for xs, xt, y in trl:
        xs,xt,y = xs.to(DEV),xt.to(DEV),y.to(DEV)
        op.zero_grad()
        lf(m(xs,xt),y).backward()
        op.step()

    m.eval();  v = 0;  n = 0
    with torch.no_grad():
        for xs,xt,y in vl:
            xs,xt,y = xs.to(DEV),xt.to(DEV),y.to(DEV)
            v += lf(m(xs,xt),y).item()*len(y)
            n += len(y)
    v /= n
    sch.step(v)

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

#Test
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
