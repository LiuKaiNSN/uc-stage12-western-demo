# Pre-endoscopic UC Triage — Western Stage 1 & 2 (Interactive Demo)

Public research prototype for the multistage UC prediction study (Western objective features).

## Live app

Deploy on [Streamlit Community Cloud](https://share.streamlit.io) from this repository.

**Main file:** `streamlit_app.py`

After deployment, your public URL will look like:

`https://uc-stage12-western-demo.streamlit.app`

*(Update this README with the exact URL once Streamlit assigns it.)*

## What this tool does

| Stage | Task | Model |
|-------|------|--------|
| **Stage 1** | Pre-endoscopic organic disease triage: IE–FDIBS spectrum vs organic colorectal disease (UC/CD/IC/CRC) | LightGBM |
| **Stage 2** | UC-specific discrimination: UC vs CD/IC/CRC | CatBoost |

- **Inputs:** symptoms, signs, laboratory tests (pre-endoscopic only).
- **Thresholds:** from DCA (Stage 1 ≈ 0.34; Stage 2 ≈ 0.51).
- **W2 framework:** stages externally validated separately; **not** a probability cascade.
- **Privacy:** user inputs are **not stored**.

## Disclaimer

Research, peer review, and academic demonstration **only**. Not for commercial or standalone clinical diagnosis. Endoscopy and histopathology retain priority.

仅供科研、同行评议与学术演示；不得用于商业或独立临床诊断。

## Local run

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Rebuild from full project (maintainers)

```bash
python F:/KeTi/Project/Script_web/deploy/build_stage12_western_cloud.py
# Then restore streamlit_app.py, requirements.txt, README.md from repo templates if needed
```

## Model versions

- Stage 1: `3_run_20260707_142952` (LightGBM)
- Stage 2: `4_run_20260707_030540` (CatBoost)

## Paper 2 Hub (Baseline Model | TCM Integrated Model)

**Separate Streamlit Cloud app** — does **not** replace the Paper 1 entry above.

| Item | Value |
|------|--------|
| Main file | **`streamlit_app_hub.py`** |
| Suggested Cloud app name | `uc-stage12-western-tcm-hub` |
| Local preview | `run_hub_local.bat` (port **8511**) |

Hub sidebar modules:

1. **Baseline Model (Stage 1–2)** — same locked LightGBM / CatBoost as Paper 1  
2. **TCM Integrated Model (Stage 1–2)** — RF models with TCM encodings  

Paper 1 URL / `streamlit_app.py` are **intentionally unchanged**.

Local Hub:

```bash
streamlit run streamlit_app_hub.py --server.port 8511
```

Rebuild TCM assets into this repo (maintainers):

```bash
python F:/KeTi/Project/Script_web/deploy/build_stage12_hub_assets.py
```

## Thesis Chapter 4 — Full system (Stage 1–3 × Baseline | TCM)

**Third Streamlit Cloud app** — does **not** replace Paper 1 or Paper 2 SCI URLs.

| Item | Value |
|------|--------|
| Main file | **`streamlit_app_full.py`** |
| Suggested Cloud app name | `uc-stage123-full-assist` |
| Local preview | `run_full_local.bat` (port **8512**) |

Sidebar:

1. **Baseline Model** or **TCM Integrated Model**  
2. **Stage 1 / Stage 2 / Stage 3** (Montreal E1–E3)

Thesis-locked Stage 3:

- Baseline: XGBoost `1_run_20260610_211647` (E3 thr ≈ 0.35)  
- TCM Integrated: CatBoost `4_run_20260611_192747` (E3 thr ≈ 0.29)

Local:

```bash
streamlit run streamlit_app_full.py --server.port 8512
```

Rebuild Stage 3 assets (maintainers):

```bash
python F:/KeTi/Project/Script_web/deploy/build_stage3_full_assets.py
```

## Analysis code (journals)

| Paper | Folder | Scope |
|-------|--------|--------|
| Paper 1 | [`Code/`](Code/) | Baseline Model Stage 1–2 training, external validation, DCA, figures |
| Paper 2 | [`Code_TCM/`](Code_TCM/) | TCM Integrated Model Stage 1–2 + Baseline vs TCM comparison scripts |

Patient-level CSVs are **not** included. See each folder’s README for run directories and environment.

## License

MIT — see [LICENSE](LICENSE).
