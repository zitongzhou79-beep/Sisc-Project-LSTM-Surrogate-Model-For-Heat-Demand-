import pandas as pd, numpy as np, os, sys

IN  = "/mnt/d/aachen_dataset_6h_raw.csv"
OUT = "/mnt/d/aachen_dataset_daily.csv"

WEATHER = ["T_out","sol_global","sol_diffuse","sol_direct",
           "rel_hum","wind_speed","cloud_opaque","T_dew"]
STATIC  = ["construction_year","net_leased_area","num_floors",
           "floor_height","building_type","is_MFH","refurb_level","volume"]

print(f"读取6h数据集: {IN}")
df = pd.read_csv(IN)
print(f"  原始: {len(df):,} 行, {df.building_id.nunique()} 栋")

df = df.sort_values(["building_id","hour"]).reset_index(drop=True)
df["day"]    = df["hour"] // 24
df["time_s"] = df["hour"] * 3600.0

# 每日热量: 分段梯形积分(功率W -> 能量kWh), 段归属到起点所在天
parts = []
for bid, gb in df.groupby("building_id"):
    h = gb["hour"].to_numpy(); t = gb["time_s"].to_numpy(); p = gb["P_heater"].to_numpy()
    if len(t) < 2: continue
    seg = (p[:-1]+p[1:])/2.0 * (t[1:]-t[:-1]) / 3_600_000.0  # kWh per segment
    sday = h[:-1] // 24
    e = pd.DataFrame({"building_id":bid,"day":sday,"kwh":seg})
    parts.append(e.groupby(["building_id","day"],as_index=False)["kwh"].sum())
energy = pd.concat(parts, ignore_index=True).rename(columns={"kwh":"daily_heat_kWh"})

# 天气日均 + 建筑特征
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
print(f"\n完成 (每日热需求 kWh, 梯形积分):")
print(f"  daily: {len(g):,} 行, {g.building_id.nunique()} 栋 -> {OUT}")
print(f"  每栋平均 {per.mean():.0f} 天 (应约365)")
print(f"  daily_heat_kWh: {g.daily_heat_kWh.min():.1f} ~ {g.daily_heat_kWh.max():.1f}, 均值 {g.daily_heat_kWh.mean():.1f}")
b0 = g.building_id.iloc[0]
print(f"  验证 楼{b0} 年度总热需求(每日之和) = {g[g.building_id==b0].daily_heat_kWh.sum():.1f} kWh/年")
