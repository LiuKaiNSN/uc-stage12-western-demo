# Code_TCM — Paper 2 analysis scripts (Stage 1–2)

Analysis code for the multicenter **two-stage pre-endoscopic** study comparing:

- **Baseline Model** — objective (Western) features only  
- **TCM Integrated Model** — Baseline features plus TCM encodings  

Same clinical endpoints as Paper 1:

- **Stage 1:** endoscopy-risk organic disease (UC / CD / IC / CRC) versus IE / FDIBS  
- **Stage 2:** UC versus other organic diagnoses (CD / IC / CRC)

Paper 1 analysis scripts live in [`../Code/`](../Code/). This folder is the Paper 2 counterpart.

OSF registration: `https://doi.org/10.17605/OSF.IO/78EUC`

## Environment

- Python **3.11**
- Main packages: `pandas`, `numpy`, `scikit-learn`, `scipy`, `xgboost`, `lightgbm`, `catboost`, `torch`, `shap`, `matplotlib`

Example:

```bash
pip install pandas numpy scikit-learn scipy xgboost lightgbm catboost torch shap matplotlib
```

## Local project layout (development machine)

Scripts were developed under `F:\KeTi\Project\` with the following layout. **Patient-level CSVs are not uploaded** to this repository.

```text
PROJECT/                          # F:\KeTi\Project
  TCM/
    data/                         # ATrain-Stage*.csv, ATest-Stage*.csv  (NOT uploaded)
    output/
      Stage1/
        6_run_20260708_052605/    # final Stage 1 RF (TCM Integrated Model)
        dca_batch/6_run_20260708_052605/overall/
      Stage2/
        6_run_20260707_111716/    # final Stage 2 RF (TCM Integrated Model)
        dca_batch/6_run_20260707_111716/overall/
      workflow_batch/tcm/06_6_best_cross/workflow_stage12.json
  outputs/                        # Baseline Model runs (same as Paper 1 Code/)
    Stage1/3_run_20260707_142952/
    Stage2/4_run_20260707_030540/
  Script/                         # source .py → mirrored here as Train / External… / Model…
  Figure/Stage12_TCM/            # publication figure outputs
```

To re-run locally, place scripts under `PROJECT/Script/` (or adjust `PROJECT` path constants inside each entry script).

## Final models used in the manuscript

### TCM Integrated Model

| Stage | Algorithm | Run directory | Approx. DCA threshold |
|-------|-----------|---------------|------------------------|
| Stage 1 | Random Forest | `TCM/output/Stage1/6_run_20260708_052605` | ≈ 0.43 |
| Stage 2 | Random Forest | `TCM/output/Stage2/6_run_20260707_111716` | ≈ 0.53 |

Workflow operating points: `TCM/output/workflow_batch/tcm/06_6_best_cross/` (`pt1≈0.43`, `pt2≈0.53`).

### Baseline Model (comparator; same locks as Paper 1)

| Stage | Algorithm | Run directory |
|-------|-----------|---------------|
| Stage 1 | LightGBM | `outputs/Stage1/3_run_20260707_142952` |
| Stage 2 | CatBoost | `outputs/Stage2/4_run_20260707_030540` |

---

### 1) Training (`Train/`)

Same trainer suite as Paper 1; TCM runs use `TCM/data/ATrain-Stage*.csv`.

| File | Role |
|------|------|
| `Try.py` | Training launcher / batch entry |
| `XGBoost.py` / `LR.py` / `LightGBM.py` / `CatBoost.py` / `SVM.py` / `RF.py` / `MLP.py` | Classical / GBM trainers |
| `DCNV2.py` / `FTTransformer.py` / `TabTransformer.py` | Deep-tabular trainers |
| `_deep_tabular_pipeline.py` | Shared deep-tabular utilities |

### 2) External validation, DCA, workflow (`External validation and DCA/`)

| File | Role |
|------|------|
| `external_validate_stage.py` | Generic binary external validation |
| `external_validate_stage1.py` | Stage 1 external validation |
| `external_validate_stage1_dl.py` / `external_validate_stage_dl.py` | Deep-model external validation |
| `dca_binary_external.py` / `dca_batch.py` / `_dca_common.py` | DCA |
| `workflow_stage12.py` | Sequential Stage 1→2 operating-point workflow |
| `dca_overlay_western_tcm.py` | Baseline vs TCM Integrated DCA overlay |

### 3) Publication tables / figures / interpretability (`Model interpretability/`)

| File | Role |
|------|------|
| `run_stage12_tcm_publication_figures.py` | Batch launcher (data + Figures 2–6) |
| `export_internal_cv_tcm_stage12.py` | Nested-CV summary (TCM Integrated) |
| `export_stage12_tcm_tables.py` | External validation tables |
| `plot_figure2_tcm_model_comparison.py` | Model comparison figures |
| `plot_figure3_tcm_external_validation.py` | External ROC + DCA panels |
| `plot_figure6_tcm_workflow.py` | Workflow figure |
| `interpret_binary_best_model_Stage12_tcm.py` | SHAP / PDP entry (TCM paths) |
| `interpret_binary_best_model_Stage12.py` | Shared interpret pipeline (called by TCM wrapper) |
| `export_paper2_*.py` / `enrich_paper2_*.py` | Paper 2 SI / comparison exports |
| `compare_stage2_tcm_vs_western.py` | Paired Baseline vs TCM Integrated comparisons |
| `_*.py` | Shared figure / CV / SHAP helpers |

---

## Typical run order (high level)

1. Train candidates on `TCM/data/` → `TCM/output/Stage1|Stage2/<run>/`  
2. External validation + DCA → `external_validate_*.py`, `dca_*.py`  
3. Workflow JSON → `workflow_stage12.py` (best cross: `06_6_best_cross`)  
4. Publication batch → `run_stage12_tcm_publication_figures.py`  
5. Interpretability → `interpret_binary_best_model_Stage12_tcm.py --stage-label stage1|stage2`

Example one-shot publication batch (development machine):

```bash
cd F:\KeTi\Project
python Script\run_stage12_tcm_publication_figures.py
```

