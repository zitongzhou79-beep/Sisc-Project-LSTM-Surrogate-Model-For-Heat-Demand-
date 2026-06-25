import os
import re
import sys
import glob
import pandas as pd

RESULTS_DIR = "/mnt/d/aachen_6h_rescsv"
SAMPLES     = os.path.expanduser("~/EDpyFlow/runs/aachen_300_6h/samples.csv")
OUT_PATH    = "/mnt/d/aachen_dataset_6h_raw.csv"
MIN_BLOCKS  = 1400
BLOCK_SEC   = 21600

WEATHER = {
    "weaDat.weaBus.TDryBul":"T_out","weaDat.weaBus.HGloHor":"sol_global",
    "weaDat.weaBus.HDifHor":"sol_diffuse","weaDat.weaBus.HDirNor":"sol_direct",
    "weaDat.weaBus.relHum":"rel_hum","weaDat.weaBus.winSpe":"wind_speed",
    "weaDat.weaBus.nOpa":"cloud_opaque","weaDat.weaBus.TDewPoi":"T_dew",
}
TARGET="multizone.PHeater[1]"
USECOLS=["time",TARGET]+list(WEATHER)
REFURB={"standard":0,"retrofit":1,"adv_retrofit":2}

chars=pd.read_csv(SAMPLES)
files=sorted(glob.glob(os.path.join(RESULTS_DIR,"*residential_*_res.csv")))
frames = []
ok, skip = 0, 0
for f in files:
    m=re.search(r"residential_(\d+)_res\.csv$",os.path.basename(f))
    if not m:
        continue
    bid=int(m.group(1))
    try:
        df=pd.read_csv(f,usecols=USECOLS)
    except ValueError:
        continue

    df["block"]=(df["time"]//BLOCK_SEC).astype(int)
    df=df.drop_duplicates("block",keep="last").sort_values("block").reset_index(drop=True)
    if len(df)<MIN_BLOCKS:
        print(f"  skipping incomplete building {bid}: {len(df)} blocks")
        skip+=1
        continue
    df["hour"]=df["block"]*6
    df=df.rename(columns={TARGET:"P_heater",**WEATHER})
    df["T_out"]-=273.15
    df["T_dew"]-=273.15
    r=chars[chars.id==bid]
    if r.empty:
        continue
    r=r.iloc[0]
    df["building_id"]=bid; df["construction_year"]=r.construction_year
    df["net_leased_area"]=r.net_leased_area; df["num_floors"]=r.num_floors
    df["floor_height"]=r.floor_height; df["building_type"]=r.building_type
    df["is_MFH"]=int(r.building_type=="MFH")
    df["refurb_level"]=REFURB.get(str(r.refurbishment_status),0)
    df["volume"]=r.net_leased_area*r.floor_height
    df["P_per_area"]=df["P_heater"]/r.net_leased_area
    frames.append(df.drop(columns=["time","block"]))
    ok+=1
    if ok%100==0:
        print(f"  processed {ok} buildings...")
data=pd.concat(frames,ignore_index=True)
data.to_csv(OUT_PATH,index=False)
per=data.groupby("building_id").size()

