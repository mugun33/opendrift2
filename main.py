from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
from pathlib import Path
from datetime import datetime, timedelta
import tempfile

import numpy as np
import xarray as xr

from opendrift.models.oceandrift import OceanDrift
from opendrift.readers import reader_netCDF_CF_generic

app = FastAPI(title="OpenDrift API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # sonra kendi domaininle sınırla
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class DriftRequest(BaseModel):
    lat: float
    lon: float
    particles: int = 20
    date: str   # örn: "20100911"

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

    # Burada temel varsayım:
    # hız değişkenleri uo ve vo
    if "uo" not in ds_u:
        raise ValueError("Variable 'uo' not found in uo file")
    if "vo" not in ds_v:
        raise ValueError("Variable 'vo' not found in vo file")

    # Tek dataset içinde birleştir
    ds = xr.Dataset()

    # Ortak koordinatları kopyala
    for coord_name in ds_u.coords:
        ds = ds.assign_coords({coord_name: ds_u.coords[coord_name]})

    # Değişkenleri OpenDrift'in beklediği isimlere çevir
    ds["x_sea_water_velocity"] = ds_u["uo"]
    ds["y_sea_water_velocity"] = ds_v["vo"]

    # Attribute eklemek faydalı olabilir
    ds["x_sea_water_velocity"].attrs["standard_name"] = "x_sea_water_velocity"
    ds["y_sea_water_velocity"].attrs["standard_name"] = "y_sea_water_velocity"

    # Zaman koordinatının adı farklıysa burada düzeltmek gerekebilir.
    # Örnek olarak zaman "time" ise sorun yok.

    tmp = tempfile.NamedTemporaryFile(suffix=".nc", delete=False)
    ds.to_netcdf(tmp.name)
    tmp.close()

    ds_u.close()
    ds_v.close()
    ds.close()

    return tmp.name

@app.get("/")
def root():
    return {"status": "ok", "message": "OpenDrift API is running"}

@app.post("/run_drift")
def run_drift(req: DriftRequest):
    try:
        combined_file = build_combined_current_file()

        reader = reader_netCDF_CF_generic.Reader(combined_file)

        o = OceanDrift(loglevel=20)
        o.add_reader(reader)

        o.set_config("general:coastline_action", "previous")
        o.set_config("drift:advection_scheme", "runge-kutta4")

        # Reader'ın başlangıç zamanını kullan
        start_time = reader.start_time
        if start_time is None:
            start_time = datetime.strptime(req.date, "%Y%m%d")

        o.seed_elements(
            lon=req.lon,
            lat=req.lat,
            number=req.particles,
            radius=0,
            time=start_time
        )

        o.run(
            duration=timedelta(hours=24),
            time_step=900,
            time_step_output=3600
        )

        lons = o.result.lon.values
        lats = o.result.lat.values
        times = o.result.time.values

        particles: List[Dict[str, Any]] = []
        n_particles = lons.shape[1]

        for p in range(n_particles):
            track = []
            for t in range(lons.shape[0]):
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
                "date": req.date
            },
            "particles": particles
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
