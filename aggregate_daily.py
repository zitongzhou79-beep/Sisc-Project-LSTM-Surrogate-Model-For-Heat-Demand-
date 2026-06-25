import pandas as pd
import numpy as np
import os
import sys

IN  = "/mnt/d/aachen_dataset_6h_raw.csv"
OUT = "/mnt/d/aachen_dataset_daily.csv"

WEATHER = ["T_out","sol_global","sol_diffuse","sol_direct",
           "rel_hum","wind_speed","cloud_opaque","T_dew"]
STATIC  = ["construction_year","net_leased_area","num_floors",
           "floor_height","building_type","is_MFH","refurb_level","volume"]

print(f"loading 6-hourly  dataset: {IN}")
df = pd.read_csv(IN)
print(f"  loade: {len(df):,} rows, {df.building_id.nunique()} buildings")

df = df.sort_values(["building_id","hour"]).reset_index(drop=True)
df["day"]    = df["hour"] // 24
df["time_s"] = df["hour"] * 3600.0


parts = []
for bid, gb in df.groupby("building_id"):
    h = gb["hour"].to_numpy()
    t = gb["time_s"].to_numpy()
    p = gb["P_heater"].to_numpy()
    if len(t) < 2: continue
    seg = (p[:-1]+p[1:])/2.0 * (t[1:]-t[:-1]) / 3_600_000.0  # kWh per segment
    sday = h[:-1] // 24
    e = pd.DataFrame({"building_id":bid,"day":sday,"kwh":seg})
    parts.append(e.groupby(["building_id","day"],as_index=False)["kwh"].sum())
energy = pd.concat(parts, ignore_index=True).rename(columns={"kwh":"daily_heat_kWh"})

# static features and aggregate weather
agg = {c:"mean" for c in WEATHER if c in df.columns}
agg.update({c:"first" for c in STATIC if c in df.columns})
meta = df.groupby(["building_id","day"], as_index=False).agg(agg)

g = energy.merge(meta, on=["building_id","day"], how="left")
g["daily_heat_per_area"] = g["daily_heat_kWh"] / g["net_leased_area"]
doy = (g["day"] % 365).to_numpy()
g["doy_sin"] = np.sin(2*np.pi*doy/365)
g["doy_cos"] = np.cos(2*np.pi*doy/365)
g = g.sort_values(["building_id","day"]).reset_index(drop=True)
g.to_csv(OUT, index=False)

per = g.groupby("building_id").size()
print(f"  daily: {len(g):,} rows, {g.building_id.nunique()} buildings -> {OUT}")
print(f"  Average days pero building {per.mean():.0f} days")
print(f"  daily_heat_kWh: {g.daily_heat_kWh.min():.1f} ~ {g.daily_heat_kWh.max():.1f}, MEAN {g.daily_heat_kWh.mean():.1f}")
b0 = g.building_id.iloc[0]
print(f"  Sanity Check:Building{b0} annnual total heat = {g[g.building_id==b0].daily_heat_kWh.sum():.1f} kWh/year")
