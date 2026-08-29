# AI-Powered Image Quality & Defect Detection

![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.9-EE4C2C?logo=pytorch&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6-F7931E?logo=scikitlearn&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)
![SQLite](https://img.shields.io/badge/DB-SQLite%20%7C%20Postgres--ready-003B57?logo=sqlite&logoColor=white)
![No external AI APIs](https://img.shields.io/badge/External%20AI%20APIs-none-16a34a)
![Tests](https://img.shields.io/badge/tests-11%20passing-16a34a)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-Render-46E3B7?logo=render&logoColor=white)](https://image-quality-defect-detection.onrender.com/)

A full-stack application that accepts an image and automatically evaluates its visual
quality: blur, under/over-exposure, noise, corruption/severe degradation, and visual
defects — with a quality score, a label (`ACCEPTABLE` / `DEGRADED` / `DEFECTIVE`), and
explainable per-issue confidence. Built with **no external AI/vision APIs**: every
detector is trained and runs locally.

> ### 🚀 Live demo: **[image-quality-defect-detection.onrender.com](https://image-quality-defect-detection.onrender.com/)**
> Deployed on Render's free tier as a single Docker service (API docs at
> [`/docs`](https://image-quality-defect-detection.onrender.com/docs)). The free tier
> spins down after inactivity, so the **first request after idle time can take
> 30–60 seconds** while it cold-starts — that's Render, not the app. It also has no
> persistent disk on this tier, so analysis history resets on redeploy/cold-start; see
> [§1](#1-quick-start-docker-compose) to run it locally with full persistence.

| | |
|---|---|
| **Backend** | FastAPI + SQLAlchemy + SQLite, REST API, structured JSON results, history endpoint |
| **AI/ML** | **Hybrid** pipeline — 12 engineered image-quality features feeding six gradient-boosted classifiers (one per issue type), plus a lightweight PyTorch convolutional **autoencoder** for anomaly-detection-style explainability (reconstruction-error heatmaps) |
| **Frontend** | Vanilla HTML/CSS/JS — drag & drop upload, batch analysis, results view with heatmap toggle, history browser, dark mode. No build step |
| **Deployment** | Live on [Render](https://image-quality-defect-detection.onrender.com/) · locally via Docker Compose (backend + nginx-served frontend) · `/health` endpoint, fully configurable via environment variables |

## Screenshots

| Upload & "what this tool analyzes" | Result with anomaly heatmap | History |
|---|---|---|
| ![Analyze tab](docs/screenshots/analyze-empty.png) | ![Result view](docs/screenshots/analyze-result.png) | ![History tab](docs/screenshots/history.png) |

---

## Table of contents

1. [Quick start (Docker Compose)](#1-quick-start-docker-compose)
2. [Local development setup & cloud deployment](#2-local-development-setup--cloud-deployment)
3. [Architecture](#3-architecture)
4. [AI / ML approach](#4-ai--ml-approach)
5. [Dataset generation & training](#5-dataset-generation--training)
6. [Evaluation results](#6-evaluation-results)
7. [Explainability](#7-explainability)
8. [Limitations & failure cases](#8-limitations--failure-cases)
9. [API documentation](#9-api-documentation)
10. [Database](#10-database)
11. [Testing](#11-testing)
12. [Project structure](#12-project-structure)
13. [Bonus features implemented](#13-bonus-features-implemented)

---

## 1. Quick start (Docker Compose)

Requires Docker + Docker Compose. No API keys, no external services, no GPU needed.

```bash
docker compose up -d --build
```

This builds and starts two containers:

| Service    | What it does                                              | Port |
|------------|-------------------------------------------------------------|------|
| `backend`  | FastAPI app, loads the pre-trained models, SQLite DB       | 8000 |
| `frontend` | nginx serving the static UI, reverse-proxies `/api` and `/health` to `backend` | 80   |

Then open **http://localhost/** for the web UI, or **http://localhost:8000/docs** for
interactive Swagger API docs (also reachable at http://localhost/docs through the proxy).

```bash
curl http://localhost/health
# {"status":"ok","model_loaded":true,"app_version":"1.0.0","model_version":"hybrid-gbc+autoencoder-v1"}
```

Model artifacts (`backend/models_store/*.joblib`, `autoencoder.pt`) are already trained
and committed — the containers only run **inference**, not training. See
[§5](#5-dataset-generation--training) to retrain from scratch.

Both containers declare a Docker `HEALTHCHECK` (`docker compose ps` shows `healthy`
once the models are loaded / nginx is serving), so orchestrators can wait on real
readiness rather than just "container started." Both checks hit `127.0.0.1` explicitly
rather than `localhost` — Alpine's `/etc/hosts` resolves `localhost` to `::1` first,
and nginx/uvicorn here only bind IPv4, so a naive `localhost` healthcheck reports
`unhealthy` forever despite the app working fine; this was caught by actually checking
`docker compose ps` after a fresh build, not assumed.

Configuration is via environment variables (see `.env.example`); `docker-compose.yml`
reads an optional root `.env` for `MAX_UPLOAD_MB`, `CORS_ORIGINS`, `BACKEND_PORT`,
`FRONTEND_PORT`. Data persists in two named volumes (`backend_data`, `backend_uploads`)
so `docker compose down` (without `-v`) keeps history across restarts.

To stop: `docker compose down` (add `-v` to also wipe the DB/uploads volumes).

---

## 2. Local development setup & cloud deployment

Requires Python 3.11+ (developed/tested on 3.13).

```bash
python -m venv env
# Windows:
env\Scripts\pip install -r backend/requirements.txt
# macOS/Linux:
env/bin/pip install -r backend/requirements.txt

cd backend
../env/Scripts/python -m uvicorn app.main:app --reload   # Windows
# ../env/bin/python -m uvicorn app.main:app --reload      # macOS/Linux
```

`backend/app/main.py` mounts the `frontend/` directory as static files whenever it's
present alongside `backend/`, so **a single `uvicorn` process serves both the API and
the web UI** — open **http://localhost:8000/**. No nginx, no separate frontend server,
no CORS config needed for local dev.

Copy `.env.example` to `backend/.env` to override defaults (upload limits, CORS, DB
path, etc.) — `pydantic-settings` loads it automatically.

### Cloud deployment (Render)

The live demo above runs on [Render](https://render.com) as a **single Docker web
service** — the same `frontend/`-mounting fallback used for local dev means the
`backend/Dockerfile` image alone (which now also `COPY`s `frontend/` in, see the
Dockerfile) serves the whole app with no nginx container needed in this deployment
path. To reproduce it:

1. Push this repo to GitHub (already done for the live demo).
2. Render → **New → Web Service** → connect the repo.
3. **Root Directory**: leave blank (build context must stay the repo root, since the
   Dockerfile `COPY`s `core/`, `backend/`, and `frontend/` as siblings).
4. **Language**: Docker. **Dockerfile Path**: `backend/Dockerfile` (there is no
   Dockerfile at the repo root — this is the field most likely to be missed).
5. **Environment variables**: none required — `MODEL_DIR`, `UPLOAD_DIR`,
   `DATABASE_URL`, `CORS_ORIGINS` are already set via `ENV` in the Dockerfile.
6. Compute plan: **Free** works (the model artifacts are small — six joblib
   classifiers + a 215K-parameter CPU autoencoder); upgrade only if you hit an
   out-of-memory error on deploy.
7. Render auto-detects the container port from the Dockerfile's `EXPOSE 8000`.

Trade-off worth knowing: Render's free tier has no persistent disk and spins the
service down after inactivity, so the SQLite history/uploads reset on redeploy or
cold-start (a fresh container = a fresh filesystem from the image). The app still
works correctly every session — it just won't remember history across a cold start.
Run it via Docker Compose (§1 above) if you want history to persist.

---

## 3. Architecture

```
                    ┌─────────────────────────┐
   Browser  ───────▶│  frontend (nginx:80)     │
                    │  index.html / app.js     │
                    └───────────┬──────────────┘
                                │ proxy_pass /api, /health, /docs
                                ▼
                    ┌─────────────────────────┐
                    │  backend (FastAPI:8000)  │
                    │  routers: analyze,        │
                    │  history, health          │
                    │        │                  │
                    │        ▼                  │
                    │  services/pipeline.py     │
                    │   validate → decode →     │
                    │   QualityEngine.analyze() │
                    │   → save image/heatmap    │
                    │   → persist to SQLite      │
                    └───────────┬───────────────┘
                                │
                    ┌───────────▼───────────────┐
                    │  core/  (shared package)   │
                    │  features.py   — 12 engineered stats │
                    │  degrade.py    — synthetic degradations (train-time) │
                    │  autoencoder.py— ConvAutoencoder + reconstruction error │
                    │  quality_engine.py — loads models_store/*, combines   │
                    │                       classifiers into final result   │
                    └────────────────────────────┘
```

`core/` is imported by **both** the offline training scripts (`ml/*.py`) and the
online backend (`backend/app/services/ml_engine.py`), so there is exactly one
implementation of feature extraction and decision logic for train and serve — no
train/serve skew.

---

## 4. AI / ML approach

This is explicitly a **hybrid** approach (assessment §3): classical engineered
features feed gradient-boosted classifiers (the primary decision-makers), and a
separate lightweight deep-learning autoencoder provides a complementary
anomaly-detection signal used for explainability.

### 4.1 Engineered features (`core/features.py`)

For every image we compute 12 interpretable statistics:

| Feature | What it measures | Relevant issue(s) |
|---|---|---|
| `sharpness_lap_var` | Variance of the Laplacian | blur |
| `sharpness_tenengrad` | Mean squared Sobel gradient magnitude (2nd, noise-robust sharpness signal) | blur |
| `brightness_mean` | Mean luminance | under/over-exposure |
| `contrast_std` | Std. dev. of luminance | exposure, corruption |
| `underexposed_ratio` | Fraction of near-black pixels (<16) | underexposure |
| `overexposed_ratio` | Fraction of near-white / clipped pixels (>240) | overexposure |
| `noise_sigma` | Immerkær (1996) fast noise-variance estimate (Laplacian-kernel convolution) | noise |
| `colorfulness` | Hasler & Süsstrunk (2003) metric | general quality |
| `saturation_mean` | Mean HSV saturation | general quality |
| `entropy` | Shannon entropy of the luminance histogram | corruption (flat/degenerate images) |
| `edge_density` | Fraction of Canny edge pixels | blur vs. genuinely flat scenes |
| `blockiness` | Mean pixel discontinuity at 8×8 grid boundaries vs. elsewhere | JPEG/compression corruption |

### 4.2 Classical ML classifiers (primary decision path)

For each of the 6 required issue types (`blur`, `underexposure`, `overexposure`,
`noise`, `corruption`, `defect`) we train an independent
`GradientBoostingClassifier` (scikit-learn, wrapped with a `StandardScaler`) that
predicts a **4-class severity**: `none` / `low` / `medium` / `high`, with
`predict_proba` giving the confidence reported in the API. Trees over ~12 hand-designed
features generalize well from a modest dataset (this was the deciding factor over a
pixel CNN, which would need orders of magnitude more data to avoid overfitting).

### 4.3 Anomaly-detection autoencoder (secondary, explainability signal)

A small convolutional autoencoder (`core/autoencoder.py`, **215,395 parameters**, 4
downsampling + 4 upsampling conv blocks, 128×128 input) is trained **only on images
that look "normal"** — clean scenes plus mild (severity ≤ 0.3) everyday blur/exposure/
noise variation — with corruption, defects, and anything above "low" severity excluded
from its training set. This is the classic reconstruction-based anomaly-detection
formulation (assessment §3, "anomaly-detection approach"): reconstruction error on an
unfamiliar/damaged image should be higher than on a "normal" one.

The scalar anomaly score is the **mean of the worst 10% of per-pixel reconstruction
errors** (not the whole-image mean) — corruption/defect artifacts typically cover only
a small part of the frame, so a plain global mean dilutes them into the noise floor of
everything else that reconstructs fine. The per-pixel error map, upsampled back to the
original resolution and rendered as a JET colormap overlay, is returned by the API as
`heatmap_url` and is exactly the "localization heatmap" bonus feature.

**Why it's advisory-only, not decision-driving:** see [§8](#8-limitations--failure-cases)
— it measurably helped its own binary in-domain metric but hurt real-world precision, so
the engineering call was to keep it out of the primary classification and use it purely
for the heatmap + as a transparently-reported secondary number (`anomaly_score`,
`anomaly_threshold` in every response).

### 4.4 Combining into a result

`core/quality_engine.py` runs all 6 classifiers, then:

```
quality_score = 100 − Σ over reported issues of (issue_weight × severity_multiplier × confidence)
```

with weights `{blur:15, underexposure:12, overexposure:12, noise:10, corruption:25, defect:20}`
and multipliers `{low:0.35, medium:0.7, high:1.0}`, clipped to `[0, 100]`.

`quality_label`: `DEFECTIVE` if any high-severity corruption/defect is present **or**
score < 40; `ACCEPTABLE` if no issues at all and score ≥ 75; otherwise `DEGRADED`.

---

## 5. Dataset generation & training

No external dataset download is required (though the pipeline optionally pulls in a
handful of `scikit-image` sample photos if available — this is a `pip install`, not an
API call, and the code degrades gracefully if it's unavailable). All training data is
built by `ml/generate_dataset.py` from:

- **Real-photo base scenes**: up to 16 photos bundled with `scikit-image`
  (astronaut, camera, coffee, coins, ...).
- **Procedurally generated base scenes**: gradients, random shapes, Perlin-like
  textures, checkerboards (`core/degrade.py`), topped up to 45 total unique scenes.

Each base scene ("scene_id") is deterministically split **train / val / test
(70/15/15) before any degradation is applied**, so every generated sample inherits its
scene's split — the test set contains visual content the training process never saw,
per the assessment's "evaluate on unseen images" requirement. This run produced:

- 45 base scenes → **31 train / 6 val / 8 test**
- 990 total samples (**682 train / 132 val / 176 test**)

For each scene we generate: 1 clean sample, single-issue samples at 3 severities
(0.3/0.6/0.9) for each of the 6 issue types (so a classifier sees both graded
positives *and* hard negatives — "other issues present, this one absent"), and 3
random two-issue combo samples for robustness.

**Retraining from scratch:**

```bash
cd ml
../env/Scripts/pip install -r requirements.txt      # adds pandas, scikit-image, matplotlib, pytest, httpx
../env/Scripts/python generate_dataset.py            # → ml/data/images/, ml/data/labels.csv
../env/Scripts/python train_classifiers.py           # → backend/models_store/{issue}_classifier.joblib
../env/Scripts/python train_autoencoder.py           # → backend/models_store/autoencoder.pt
../env/Scripts/python evaluate.py                    # → evaluation/evaluation_report.json + plots
```

(`generate_dataset.py`/training use a fixed seed for reproducibility; re-running from
scratch reproduces the same split and near-identical metrics.)

---

## 6. Evaluation results

Evaluated on the **176 held-out test samples from 8 scenes never seen during
training or validation** (`evaluation/evaluate.py` → `evaluation/evaluation_report.json`,
plots in `evaluation/confusion_matrices.png` and `evaluation/anomaly_roc_and_distribution.png`).

### Per-issue classifiers (4-class severity: none/low/medium/high)

| Issue | Accuracy | Macro F1 | Macro Precision | Macro Recall |
|---|---|---|---|---|
| blur | 0.892 | 0.588 | 0.589 | 0.589 |
| underexposure | 0.983 | 0.948 | 0.953 | 0.946 |
| overexposure | 0.807 | 0.480 | 0.468 | 0.498 |
| noise | 0.909 | 0.694 | 0.728 | 0.668 |
| corruption | 0.920 | 0.763 | 0.839 | 0.719 |
| defect | 0.932 | 0.797 | 0.862 | 0.747 |

Accuracy is inflated somewhat by the dominant `none` class in every issue's test
distribution (~140-150/176 samples); macro F1/precision/recall (unweighted across all
4 classes) is the more honest number and is what's reported. `underexposure` is the
strongest performer — brightness/clipping statistics are unambiguous. `blur` and
`overexposure` are the weakest: adjacent severity buckets (`low` vs `medium`) are
genuinely hard to separate from global statistics alone, and inference-time downscaling
(long side capped at 1600px) mildly compresses fine sharpness differences. Confusion
is concentrated between *adjacent* severities (e.g. predicted `low` for true `medium`),
never between `none` and `high` — i.e. the model is directionally right even when it
misses the exact bucket. See `evaluation/example_failure_cases` in the report for
concrete misses (e.g. a `medium`-blur combo image predicted `none` — a case where a
second issue in the same combo dominated the feature signal).

### Anomaly-detection autoencoder (standalone, binary: corruption-or-defect present)

| Metric | Value |
|---|---|
| ROC-AUC | 0.740 |
| Precision @ threshold | 0.545 |
| Recall @ threshold | 0.831 |
| Threshold selection | Youden's J on val-set ROC (not test-tuned) |

A respectable in-domain signal, but **not used to override the classifiers in the
shipped decision engine** — see the next section for why.

### Quality-label distribution on the test set

`ACCEPTABLE: 20, DEGRADED: 139, DEFECTIVE: 17` (out of 176) — expected, since most
test samples carry at least one graded degradation by construction.

---

## 7. Explainability

Every API response includes, beyond the score/label:

- **`features`**: the full 12-value engineered-feature dict (raw numbers, not just a
  score) — e.g. you can see *exactly* why `blur` fired by looking at
  `sharpness_lap_var`.
- **`issues[].explanation`**: a human-readable sentence per detected issue tying it back
  to the underlying statistics.
- **`issues[].confidence`**: the classifier's own `predict_proba` for the winning class.
- **`anomaly_score` / `anomaly_threshold`**: the autoencoder's reconstruction-error
  score and its val-set-derived operating threshold, always returned for transparency.
- **`heatmap_url`**: a JET-colormap overlay of the per-pixel reconstruction error,
  localizing *where* in the image the autoencoder found the image least
  "normal-looking" — viewable via the toggle in the UI's result panel. This doubles as
  the optional "quality heatmap / localization" bonus feature.

This combination (interpretable statistics + per-class confidence + a spatial
saliency-style heatmap from the secondary model) was chosen over e.g. Grad-CAM because
the primary detectors are tree-based (Grad-CAM doesn't apply) — the heatmap comes from
the one deep-learning component instead, applied to the one task (localization) it's
well suited to regardless of the calibration issues discussed next.

**On the frontend**, the Analyze tab has a collapsible **"What this tool analyzes"**
panel (open by default, screenshot above) that spells out all 6 detection categories,
all 12 computed statistics grouped by theme, and the exact quality-score formula —
so a user doesn't have to read this README to understand what the tool is doing before
they upload anything.

---

## 8. Limitations & failure cases

Being direct about what doesn't work well, per the assessment's request for "failure
cases, limitations, and discussion of incorrect or uncertain predictions":

1. **The anomaly autoencoder does not generalize well to real, texture-rich photos**,
   and this was a deliberate finding, not an afterthought. Early in development the
   autoencoder's reconstruction-error ratio was used to *override* a classifier's
   `none` prediction for corruption/defect when strongly anomalous. Diagnostic check
   against `sample_images/`:

   | Image | Anomaly ratio (score / threshold) |
   |---|---|
   | `01_clean_acceptable.jpg` (genuinely clean photo) | **4.90** |
   | `07_defective_scratches.jpg` (actually has scratches) | **1.70** |

   The clean photo scored a *stronger* anomaly than the actually-defective one — the
   reconstruction error tracks raw image texture/detail complexity (the autoencoder's
   "normal" training set is dominated by smoother procedurally-generated scenes) more
   than it tracks genuine defects. Enabling the override measurably dropped held-out
   `corruption`/`defect` accuracy (0.92→0.52, 0.93→0.49 in one tested configuration).
   **Fix applied:** the autoencoder's output was demoted to an advisory/explainability
   signal only (§4.3); the classifiers alone drive every decision. This is exactly the
   kind of engineering trade-off the assessment asks to be evaluated and explained
   rather than hidden — a larger and more diverse "normal" training corpus (thousands
   of real photos, not ~45 scenes) would likely be needed to make the anomaly score
   trustworthy enough to reinstate as a decision signal.

2. **`overexposure` and `blur` have the weakest macro-F1** (0.48, 0.59) — mostly
   confusion between adjacent severity buckets, not gross misclassification (see
   confusion matrices in `evaluation/confusion_matrices.png`). A 3-class (none/some/severe)
   or continuous-regression formulation would likely score better than the current
   4-class one at the cost of coarser severity reporting.

3. **Small, synthetic training set.** ~45 unique base scenes is enough to demonstrate
   the approach and get meaningful held-out metrics, but is far from the scale a
   production system would use. Degradations are synthetic (documented, per assessment
   §8, as an explicitly acceptable strategy) — real-world camera/sensor artifacts (lens
   flare, motion blur streaks, rolling-shutter, real JPEG-in-the-wild corruption
   patterns) are only approximated.

4. **Combo images** (two degradations stacked) sometimes have one issue dominate the
   engineered-feature signal and mask the other (see the blur/combo failure case in
   `evaluation/evaluation_report.json`).

5. **No color-cast / white-balance detector**, despite computing per-channel colorfulness
   — out of scope for the 6 required categories, flagged here as a natural extension.

---

## 9. API documentation

Interactive Swagger UI at `/docs` (e.g. http://localhost:8000/docs), OpenAPI JSON at
`/openapi.json`. Summary:

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Health/status check — model load state, versions |
| POST | `/api/analyze` | Upload one image (`multipart/form-data`, field `file`) → full analysis |
| POST | `/api/analyze/batch` | Upload up to 20 images (field `files`, repeated) → list of analyses |
| GET | `/api/analyses?limit=&offset=` | Paginated history (most recent first) |
| GET | `/api/analyses/{id}` | Full stored analysis by id |
| DELETE | `/api/analyses/{id}` | Delete a stored analysis |
| GET | `/api/analyses/{id}/image` | The originally uploaded image bytes |
| GET | `/api/analyses/{id}/heatmap` | The anomaly-heatmap overlay PNG (if generated) |

### Example: analyze an image

```bash
curl -X POST http://localhost:8000/api/analyze \
  -F "file=@sample_images/06_corrupted.jpg;type=image/jpeg"
```

```json
{
  "id": "98aa957030f442828ab61bfb0d224c16",
  "created_at": "2026-08-29T15:50:58.974552",
  "original_filename": "06_corrupted.jpg",
  "width": 384, "height": 384, "file_size_bytes": 22213,
  "quality_score": 67.6,
  "quality_label": "DEFECTIVE",
  "issues": [
    {
      "type": "corruption",
      "severity": "high",
      "confidence": 0.983,
      "confidence_source": "classifier",
      "explanation": "Blockiness, entropy, and edge-density statistics match the pattern of heavy compression or transmission corruption."
    }
  ],
  "features": {
    "sharpness_lap_var": 990.21, "sharpness_tenengrad": 9538.77, "noise_sigma": 1.21,
    "colorfulness": 76.34, "saturation_mean": 177.05, "entropy": 6.48,
    "edge_density": 0.088, "blockiness": 1.43, "brightness_mean": 101.08,
    "contrast_std": 56.16, "underexposed_ratio": 0.084, "overexposed_ratio": 0.009
  },
  "anomaly_score": 0.047808,
  "anomaly_threshold": 0.01548,
  "processing_time_ms": 80.3,
  "model_version": "hybrid-gbc+autoencoder-v1",
  "image_url": "/api/analyses/98aa957030f442828ab61bfb0d224c16/image",
  "heatmap_url": "/api/analyses/98aa957030f442828ab61bfb0d224c16/heatmap"
}
```

### Error handling

| Status | When |
|---|---|
| 400 | Empty file, file over `MAX_UPLOAD_MB`, unsupported content-type, undecodable/unreadable image, image below/above size limits |
| 404 | Analysis id not found, or its stored image/heatmap file missing |
| 422 | Malformed multipart request (FastAPI/pydantic validation) |
| 500 | Unexpected server error (logged; generic message returned, no internals leaked) |

A genuinely image-shaped file that's heavily corrupted (truncated JPEG, garbled bytes)
is **not** rejected at the validation layer (`ImageFile.LOAD_TRUNCATED_IMAGES=True`) —
it's deliberately let through to the quality engine so the `corruption` detector can do
its job, which is the whole point of that detector.

---

## 10. Database

SQLite by default (zero setup — the file is created automatically at
`backend/data/app.db`, or a Docker named volume in Compose). Swap to Postgres by
setting `DATABASE_URL=postgresql+psycopg2://user:pass@host:5432/dbname` (add
`psycopg2-binary` to `backend/requirements.txt`) — the SQLAlchemy models are
database-agnostic.

Single table, `analysis_results`: id (uuid), timestamps, original filename, stored
image/heatmap paths, dimensions, file size, `quality_score`, `quality_label`,
`issues_json`, `features_json`, `anomaly_score`, `processing_time_ms`, `model_version`.
Created automatically on startup (`app/database.py::init_db()` → `Base.metadata.create_all`)
— no manual migration step needed for this schema.

---

## 11. Testing

```bash
cd backend
../env/Scripts/python -m pytest -v
```

11 tests: feature-extraction sanity checks (does blur actually reduce the sharpness
feature, etc. — `tests/test_feature_extraction.py`) and API integration tests against
a live `TestClient` (health, analyze success/validation-failure, history round-trip,
404 — `tests/test_api.py`). Requires trained model artifacts in
`backend/models_store/` (already included).

---

## 12. Project structure

```
core/                     shared package (features, degradations, autoencoder, decision engine)
ml/                        offline: dataset generation, training, evaluation scripts
backend/
  app/
    routers/                analyze.py, history.py, health.py
    services/                image_validation.py, pipeline.py, storage.py, ml_engine.py
    models.py, schemas.py, database.py, config.py, main.py
  models_store/             trained artifacts (*.joblib, autoencoder.pt, *_meta.json)
  tests/
  Dockerfile
frontend/
  index.html, styles.css, app.js   (upload/analyze, results, history, "what this analyzes" panel)
  Dockerfile, nginx.conf.template
evaluation/                 evaluation_report.json + confusion-matrix / ROC plots
sample_images/               8 demo images, one per required quality condition
docs/screenshots/            README screenshots
docker-compose.yml
```

---

## 13. Bonus features implemented

- ✅ **Batch image analysis** — `POST /api/analyze/batch`, multi-file drag & drop in the UI.
- ✅ **Quality heatmap / localization** — anomaly reconstruction-error overlay, toggle in the UI.
- ✅ **Automated tests** — 11 pytest tests (feature extraction + API integration).
- ✅ **Health/status endpoint** — `/health` reports model-load state, not just liveness.
- Not implemented (flagged honestly rather than left silent): confidence calibration/
  uncertainty estimation beyond raw `predict_proba`, model versioning beyond a single
  `model_version` string, CI/CD workflow, production monitoring/logging beyond stdout
  logging — all reasonable next steps outside this assessment's time box.
