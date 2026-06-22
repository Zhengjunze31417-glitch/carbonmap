"""
Flask backend for the CarbonMap field prediction app.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import math
import re
import threading
import time as _time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import urllib.request
from flask import Flask, Response, jsonify, request, send_from_directory

from predictor import predict_field, preview_tiles, analyze_year

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

app = Flask(__name__, static_folder="static", static_url_path="")

# ── Wayback config cache (TTL = 1 h) ─────────────────────────────────────────
_wayback_cfg: dict | None = None
_wayback_cfg_ts: float = 0
_WAYBACK_CFG_URL = "https://s3-us-west-2.amazonaws.com/config.maptiles.arcgis.com/waybackconfig.json"


def _get_wayback_config() -> dict:
    global _wayback_cfg, _wayback_cfg_ts
    if _wayback_cfg is None or _time.time() - _wayback_cfg_ts > 3600:
        import json as _j
        req = urllib.request.Request(_WAYBACK_CFG_URL, headers={"User-Agent": "CarbonMap/1.0"})
        with urllib.request.urlopen(req, timeout=15) as r:
            _wayback_cfg = _j.loads(r.read())
        _wayback_cfg_ts = _time.time()
    return _wayback_cfg


def _config_to_releases(config: dict) -> list[dict]:
    """Return [{id, date, tile_url}, …] sorted newest-first for every release."""
    out = []
    for release_id, info in config.items():
        title = info.get("itemTitle", "")
        m = re.search(r"Wayback (\d{4}-\d{2}-\d{2})", title)
        if not m:
            continue
        tile_url = (
            info.get("itemURL", "")
            .replace("{level}", "{z}")
            .replace("{row}", "{y}")
            .replace("{col}", "{x}")
        )
        out.append({"id": int(release_id), "date": m.group(1), "tile_url": tile_url})
    out.sort(key=lambda r: r["date"], reverse=True)
    return out


# HTTP opener that does NOT follow 301 redirects
class _NoRedirect(urllib.request.HTTPErrorProcessor):
    def http_response(self, request, response):
        return response
    https_response = http_response

_no_redirect_opener = urllib.request.build_opener(_NoRedirect)


def _tile_has_change(release_id: int, zoom: int, ty: int, tx: int) -> bool:
    """HTTP 200 = this release introduced new imagery; HTTP 301 = redirects to older source."""
    url = (
        f"https://wayback.maptiles.arcgis.com/arcgis/rest/services/World_Imagery/"
        f"WMTS/1.0.0/default028mm/MapServer/tile/{release_id}/{zoom}/{ty}/{tx}"
    )
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "CarbonMap/1.0"})
        with _no_redirect_opener.open(req, timeout=8) as r:
            return r.getcode() == 200
    except Exception:
        return False

# In-memory job store: job_id → {status, progress, result, error}
_jobs: dict[str, dict] = {}
_lock = threading.Lock()


def _set(job_id: str, **kwargs) -> None:
    with _lock:
        _jobs[job_id].update(kwargs)


@app.get("/")
def index():
    return send_from_directory("static", "index.html")


@app.post("/api/analyze")
def analyze():
    body = request.get_json(silent=True) or {}
    polygon = body.get("polygon")
    if not polygon or len(polygon) < 3:
        return jsonify({"error": "polygon must have at least 3 [lng,lat] pairs"}), 400

    date = body.get("date", "")
    logging.info("ANALYZE  date=%r  polygon_pts=%d", date, len(polygon))

    job_id = str(uuid.uuid4())
    with _lock:
        _jobs[job_id] = {"status": "running", "progress": "Queued…", "result": None, "error": None}

    def run():
        try:
            result = predict_field(
                polygon,
                progress=lambda msg: _set(job_id, progress=msg),
                date=date,
            )
            _set(job_id, status="done", result=result)
        except Exception as exc:
            logging.exception("predict_field failed")
            _set(job_id, status="error", error=str(exc))

    threading.Thread(target=run, daemon=True).start()
    return jsonify({"job_id": job_id}), 202


@app.post("/api/analyze_year")
def analyze_year_endpoint():
    body = request.get_json(silent=True) or {}
    polygon = body.get("polygon")
    year = int(body.get("year", 2025))
    if not polygon or len(polygon) < 3:
        return jsonify({"error": "polygon must have at least 3 [lng,lat] pairs"}), 400
    logging.info("ANALYZE_YEAR  year=%d  polygon_pts=%d", year, len(polygon))
    job_id = str(uuid.uuid4())
    with _lock:
        _jobs[job_id] = {"status": "running", "progress": "Queued…", "result": None, "error": None}
    def run():
        try:
            result = analyze_year(polygon, year, progress=lambda msg: _set(job_id, progress=msg))
            _set(job_id, status="done", result=result)
        except Exception as exc:
            logging.exception("analyze_year failed")
            _set(job_id, status="error", error=str(exc))
    threading.Thread(target=run, daemon=True).start()
    return jsonify({"job_id": job_id}), 202


@app.post("/api/preview")
def preview():
    body = request.get_json(silent=True) or {}
    bounds = body.get("bounds")          # [west, south, east, north]
    soc_date = body.get("soc_date", "")
    if not bounds or len(bounds) != 4:
        return jsonify({"error": "bounds must be [west, south, east, north]"}), 400
    try:
        tile_url = preview_tiles(bounds, target_date=soc_date)
        return jsonify({"tile_url": tile_url})
    except Exception as exc:
        logging.exception("preview_tiles failed")
        return jsonify({"error": str(exc)}), 500


@app.get("/api/wayback/releases")
def wayback_releases():
    """Return Wayback releases that have genuinely new imagery for the given location.

    Uses the redirect trick: a tile URL for release R returns HTTP 200 when R introduced
    new imagery for that tile, or HTTP 301 (→ older release) when R reuses older data.
    All 194 releases are checked in parallel with 30 workers (~5s total).
    """
    lat = request.args.get("lat", type=float)
    lng = request.args.get("lng", type=float)
    if lat is None or lng is None:
        return jsonify({"error": "lat and lng required"}), 400
    zoom = 9  # fixed: each tile covers ~40km², accurate enough for change detection
    try:
        n = 2 ** zoom
        tx = int((lng + 180) / 360 * n)
        lat_r = math.radians(lat)
        ty = int((1 - math.log(math.tan(lat_r) + 1 / math.cos(lat_r)) / math.pi) / 2 * n)
        tx = max(0, min(n - 1, tx))
        ty = max(0, min(n - 1, ty))

        all_releases = _config_to_releases(_get_wayback_config())
        changed: list[dict] = []
        with ThreadPoolExecutor(max_workers=30) as pool:
            futures = {pool.submit(_tile_has_change, r["id"], zoom, ty, tx): r for r in all_releases}
            for fut in as_completed(futures):
                if fut.result():
                    changed.append(futures[fut])

        changed.sort(key=lambda r: r["date"], reverse=True)
        return jsonify({"releases": changed})
    except Exception as exc:
        logging.exception("wayback_releases failed")
        return jsonify({"error": str(exc)}), 500


@app.get("/api/wayback/<int:year>/<int:month>")
@app.get("/api/wayback/<int:year>")
def wayback(year: int, month: int = 7):
    """Find the closest Esri Wayback release to the requested year/month."""
    from datetime import datetime
    try:
        releases = _config_to_releases(_get_wayback_config())
        capped_year = min(year, datetime.now().year)
        target_ts = datetime(capped_year, max(1, min(12, month)), 1).timestamp()
        best, best_delta = None, float("inf")
        for r in releases:
            ts = datetime.strptime(r["date"], "%Y-%m-%d").timestamp()
            delta = abs(ts - target_ts)
            if delta < best_delta:
                best_delta, best = delta, r
        if not best:
            return jsonify({"error": "No release found"}), 404
        return jsonify({"tile_url": best["tile_url"], "date": best["date"]})
    except Exception as exc:
        logging.exception("wayback lookup failed")
        return jsonify({"error": str(exc)}), 500


_FEEDBACK_FILE = Path(__file__).parent / "feedback.jsonl"
_SOIL_FILE = Path(__file__).parent / "soil_contributions.csv"
_SOIL_REQUIRED = {"lat", "lng"}
_SOIL_MEASURE  = {"soc_g_kg", "agc_mgc_ha"}
_SOIL_ALL_COLS = ["lat", "lng", "soc_g_kg", "agc_mgc_ha", "date", "depth_cm", "notes",
                  "contributor_name", "contributor_email", "contributor_institution", "submitted_at"]
_TEMPLATE_CSV  = (
    "lat,lng,soc_g_kg,agc_mgc_ha,date,depth_cm,notes\n"
    "41.868,-93.097,28.5,,2024-05-15,30,Iowa corn field after harvest\n"
    "49.512,34.187,,12.3,2024-03-20,,Ukraine winter wheat\n"
)

@app.post("/api/feedback")
def feedback():
    body = request.get_json(silent=True) or {}
    if not body.get("comment", "").strip():
        return jsonify({"error": "comment required"}), 400
    entry = {
        "ts": body.get("timestamp", ""),
        "comment": body["comment"].strip(),
        "polygon_pts": body.get("polygon_pts"),
        "center": body.get("center_lng_lat"),
        "zoom": body.get("map_zoom"),
    }
    with _lock:
        with open(_FEEDBACK_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    logging.info("FEEDBACK  %s", entry)
    return jsonify({"ok": True}), 201


@app.get("/api/soil_template")
def soil_template():
    return Response(
        _TEMPLATE_CSV,
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=soil_data_template.csv"},
    )


@app.post("/api/upload_soil")
def upload_soil():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    f = request.files["file"]
    if not f.filename.lower().endswith(".csv"):
        return jsonify({"error": "Only CSV files are accepted"}), 400
    try:
        content = f.read().decode("utf-8-sig")
        reader = csv.DictReader(io.StringIO(content))
        cols = set(reader.fieldnames or [])
        if not _SOIL_REQUIRED.issubset(cols):
            return jsonify({"error": "CSV must contain columns: lat, lng"}), 400
        if not _SOIL_MEASURE.intersection(cols):
            return jsonify({"error": "CSV must contain at least one of: soc_g_kg, agc_mgc_ha"}), 400

        contributor = {
            "contributor_name":        request.form.get("name", "").strip() or None,
            "contributor_email":       request.form.get("email", "").strip() or None,
            "contributor_institution": request.form.get("institution", "").strip() or None,
        }
        accepted, skipped = [], []
        now = _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime())
        for i, row in enumerate(reader, start=2):
            try:
                lat = float(row["lat"])
                lng = float(row["lng"])
                if not (-90 <= lat <= 90 and -180 <= lng <= 180):
                    skipped.append(f"row {i}: lat/lng out of range")
                    continue
                soc = row.get("soc_g_kg", "").strip()
                agc = row.get("agc_mgc_ha", "").strip()
                if not soc and not agc:
                    skipped.append(f"row {i}: no measurement value")
                    continue
                accepted.append({
                    "lat": lat, "lng": lng,
                    "soc_g_kg": soc or None,
                    "agc_mgc_ha": agc or None,
                    "date": row.get("date", "").strip() or None,
                    "depth_cm": row.get("depth_cm", "").strip() or None,
                    "notes": row.get("notes", "").strip() or None,
                    **contributor,
                    "submitted_at": now,
                })
            except (ValueError, KeyError) as exc:
                skipped.append(f"row {i}: {exc}")

        if not accepted:
            return jsonify({"error": "No valid rows found", "details": skipped[:5]}), 400

        file_exists = _SOIL_FILE.exists()
        with _lock:
            with open(_SOIL_FILE, "a", newline="", encoding="utf-8") as out:
                writer = csv.DictWriter(out, fieldnames=_SOIL_ALL_COLS)
                if not file_exists:
                    writer.writeheader()
                writer.writerows(accepted)

        logging.info("SOIL_UPLOAD  accepted=%d  skipped=%d", len(accepted), len(skipped))
        return jsonify({"accepted": len(accepted), "skipped": len(skipped)}), 201
    except Exception as exc:
        logging.exception("upload_soil failed")
        return jsonify({"error": str(exc)}), 500


@app.get("/api/status/<job_id>")
def status(job_id: str):
    with _lock:
        job = _jobs.get(job_id)
    if job is None:
        return jsonify({"error": "job not found"}), 404
    return jsonify(job)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False, use_reloader=True, reloader_type="stat")
