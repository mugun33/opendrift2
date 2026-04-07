from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any
import tempfile
import os

import numpy as np
import xarray as xr

from opendrift.models.oceandrift import OceanDrift
from opendrift.readers import reader_netCDF_CF_generic

app = FastAPI(title="OpenDrift API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # sonra kendi domaininle sınırla
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class DriftRequest(BaseModel):
    lat: float
    lon: float
    particles: int = 20
    date: str  # örn: "20100911"

DATA_DIR = Path("./data")

UO_FILE = DATA_DIR / "BLK_1d_20100911_20100917_grid_surface_uo.nc"
VO_FILE = DATA_DIR / "BLK_1d_20100911_20100917_grid_surface_vo.nc"


def build_combined_current_file() -> str:
    if not UO_FILE.exists():
        raise FileNotFoundError(f"Missing file: {UO_FILE}")
    if not VO_FILE.exists():
        raise FileNotFoundError(f"Missing file: {VO_FILE}")

    ds_u = xr.open_dataset(UO_FILE)
    ds_v = xr.open_dataset(VO_FILE)

    if "uo" not in ds_u:
        raise ValueError("Variable 'uo' not found in U file")
    if "vo" not in ds_v:
        raise ValueError("Variable 'vo' not found in V file")

    # Tek bir dataset oluştur
    ds = xr.Dataset()

    # Koordinatlar
    ds = ds.assign_coords({
        "time": ds_u["time_counter"],
        "y": ds_u["y"],
        "x": ds_u["x"]
    })

    # 2D lat/lon koordinatlarını ekle
    ds["nav_lat"] = ds_u["nav_lat"]
    ds["nav_lon"] = ds_u["nav_lon"]

    ds["nav_lat"].attrs = ds_u["nav_lat"].attrs
    ds["nav_lon"].attrs = ds_u["nav_lon"].attrs

    # OpenDrift'in anlayacağı isimlerle değişkenleri ekle
    # Boyutları: (time_counter, depthu, y, x) -> yüzey olduğu için depth zaten 1
    u = ds_u["uo"].squeeze(drop=True)
    v = ds_v["vo"].squeeze(drop=True)

    # Zaman boyutunu "time" olarak yeniden adlandır
    rename_map_u = {}
    rename_map_v = {}

    if "time_counter" in u.dims:
        rename_map_u["time_counter"] = "time"
    if "time_counter" in v.dims:
        rename_map_v["time_counter"] = "time"

    u = u.rename(rename_map_u)
    v = v.rename(rename_map_v)

    ds["sea_water_x_velocity"] = u
    ds["sea_water_y_velocity"] = v

    ds["sea_water_x_velocity"].attrs["standard_name"] = "sea_water_x_velocity"
    ds["sea_water_y_velocity"].attrs["standard_name"] = "sea_water_y_velocity"
    ds["sea_water_x_velocity"].attrs["units"] = "m/s"
    ds["sea_water_y_velocity"].attrs["units"] = "m/s"
    ds["sea_water_x_velocity"].attrs["coordinates"] = "nav_lat nav_lon"
    ds["sea_water_y_velocity"].attrs["coordinates"] = "nav_lat nav_lon"

    # time coordinate attrs
    ds["time"].attrs = ds_u["time_counter"].attrs
    ds["time"].attrs["standard_name"] = "time"

    # Global attrs
    ds.attrs["Conventions"] = "CF-1.6"
    ds.attrs["title"] = "Combined surface currents for OpenDrift"

    tmp = tempfile.NamedTemporaryFile(suffix=".nc", delete=False)
    tmp_name = tmp.name
    tmp.close()

    ds.to_netcdf(tmp_name)

    ds_u.close()
    ds_v.close()
    ds.close()

    return tmp_name


def build_start_time(date_str: str, reader_start_time):
    # Wix'ten gelen 20100911 -> 2010-09-11 00:00:00
    requested = datetime.strptime(date_str, "%Y%m%d")

    # Eğer reader başlangıç zamanı aynı haftadaysa o tarihi kullan
    # Saat bilgisi verilmediği için 00:00 başlatıyoruz
    if reader_start_time is not None:
        return requested

    return requested


@app.get("/")
def root():
    return {"status": "ok", "message": "OpenDrift API is running"}


@app.post("/run_drift")
def run_drift(req: DriftRequest):
    temp_file = None
    try:
        temp_file = build_combined_current_file()

        reader = reader_netCDF_CF_generic.Reader(temp_file)

        o = OceanDrift(loglevel=20)
        o.add_reader(reader)

        # Kıyıya vurursa bir önceki konumda kalsın
        o.set_config("general:coastline_action", "previous")

        # Dikey süreçleri kapat, yüzey drift istiyoruz
        o.set_config("drift:vertical_mixing", False)
        o.set_config("drift:vertical_advection", False)

        # Basit yatay adveksiyon
        o.set_config("drift:advection_scheme", "runge-kutta4")

        start_time = build_start_time(req.date, reader.start_time)

        o.seed_elements(
            lon=req.lon,
            lat=req.lat,
            number=min(req.particles, 20),  # şimdilik 20 ile sınırla
            radius=0,
            time=start_time
        )

        # 24 saat sürüklet, 1 saatte bir çıktı al
        o.run(
            duration=timedelta(hours=24),
            time_step=900,
            time_step_output=3600
        )

        lons = o.result.lon.values
        lats = o.result.lat.values
        times = o.result.time.values

        particles: List[Dict[str, Any]] = []

        # shape genelde [time, particle]
        n_times = lons.shape[0]
        n_particles = lons.shape[1]

        for p in range(n_particles):
            track = []
            for t in range(n_times):
                lon_val = lons[t, p]
                lat_val = lats[t, p]

                if np.ma.is_masked(lon_val) or np.ma.is_masked(lat_val):
                    continue

                lon_val = float(lon_val)
                lat_val = float(lat_val)

                if np.isnan(lon_val) or np.isnan(lat_val):
                    continue

                track.append({
                    "lat": lat_val,
                    "lon": lon_val,
                    "time": str(times[t])
                })

            particles.append({
                "id": p + 1,
                "track": track
            })

        return {
            "ok": True,
            "input": {
                "lat": req.lat,
                "lon": req.lon,
                "particles": req.particles,
                "date": req.date,
                "reader_start_time": str(reader.start_time),
                "reader_end_time": str(reader.end_time)
            },
            "particles": particles
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenDrift failed: {str(e)}")

    finally:
        if temp_file and os.path.exists(temp_file):
            os.remove(temp_file)
