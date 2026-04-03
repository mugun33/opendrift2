from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any

app = FastAPI(title="OpenDrift API", version="1.0")

# Wix alan adından istek alabilmek için CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # ilk test için; sonra kendi domaininle sınırla
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class DriftRequest(BaseModel):
    lat: float
    lon: float
    particles: int = 500
    date: str | None = None

@app.get("/")
def root():
    return {"status": "ok", "message": "OpenDrift API is running"}

@app.post("/run_drift")
def run_drift(req: DriftRequest):
    lat = req.lat
    lon = req.lon
    n = req.particles

    # Şimdilik sahte test çıktısı
    particles: List[Dict[str, Any]] = []

    for i in range(min(n, 20)):  # ilk testte 500 yerine 20 fake track dönelim
        track = [
            {"lat": lat, "lon": lon, "time": "2026-04-03T00:00:00Z"},
            {"lat": lat + 0.01 * (i / 20), "lon": lon + 0.02, "time": "2026-04-03T01:00:00Z"},
            {"lat": lat + 0.02 * (i / 20), "lon": lon + 0.03, "time": "2026-04-03T02:00:00Z"},
        ]
        particles.append({
            "id": i + 1,
            "track": track
        })

    return {
        "ok": True,
        "input": {
            "lat": lat,
            "lon": lon,
            "particles": n,
            "date": req.date
        },
        "particles": particles
    }
