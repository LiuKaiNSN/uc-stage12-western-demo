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

## License

MIT — see [LICENSE](LICENSE).
