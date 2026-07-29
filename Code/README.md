Code accompanying the multicenter two-stage pre-endoscopic machine-learning study:

- **Stage 1:** endoscopy-risk organic disease (UC / CD / IC / CRC) versus IE / FDIBS  
- **Stage 2:** UC versus other organic diagnoses (CD / IC / CRC)

OSF registration: `https://doi.org/10.17605/OSF.IO/78EUC`

## Environment

- Python **3.11**
- Main packages: `pandas`, `numpy`, `scikit-learn`, `scipy`, `xgboost`, `lightgbm`, `catboost`, `torch`, `shap`, `matplotlib`

Example:

```bash
pip install pandas numpy scikit-learn scipy xgboost lightgbm catboost torch shap matplotlib
```

## Local project layout (development machine)

Scripts were developed under `F:\KeTi\Project\` with the following layout:

```text
PROJECT/                          # F:\KeTi\Project
  Data/                           # ATrain-Stage*.csv, ATest-Stage*.csv  (NOT uploaded)
  Script/                         # source .py  → copy into this GitHub folder as scripts/
  outputs/
    Stage1/
      3_run_20260707_142952/      # final Stage 1 LightGBM run (manuscript)
      dca_batch/3_run_20260707_142952/overall/
    Stage2/
      4_run_20260707_030540/      # final Stage 2 CatBoost run (manuscript)
      dca_batch/4_run_20260707_030540/overall/
    workflow_batch/western/03_4_best_cross/workflow_stage12.json
  Figure/Stage12/                 # publication figure outputs
```


## Final models used in the manuscript

| Stage | Algorithm | Run directory |
|-------|-----------|---------------|
| Stage 1 | LightGBM | `outputs/Stage1/3_run_20260707_142952` |
| Stage 2 | CatBoost | `outputs/Stage2/4_run_20260707_030540` |


### 1) Training (`scripts/training/` or flat)

| File | Role |
|------|------|
| `Try.py` | Training launcher / batch entry |
| `XGBoost.py` | XGBoost trainer |
| `LR.py` | Logistic regression trainer |
| `LightGBM.py` | LightGBM trainer |
| `CatBoost.py` | CatBoost trainer |
| `SVM.py` | SVM trainer |
| `RF.py` | Random forest trainer |
| `MLP.py` | MLP trainer |
| `DCNV2.py` | Deep & Cross Network v2 trainer |
| `FTTransformer.py` | FT-Transformer trainer |
| `TabTransformer.py` | TabTransformer trainer |
| `_deep_tabular_pipeline.py` | Shared deep-tabular utilities |

### 2) External validation, DCA, workflow (`scripts/validation/`)

| File | Role |
|------|------|
| `external_validate_stage.py` | Generic binary external validation |
| `external_validate_stage1.py` | Stage 1 external validation |
| `external_validate_stage1_dl.py` | Stage 1 deep-model external validation |
| `external_validate_stage_dl.py` | Deep-model external validation helpers |
| `dca_binary_external.py` | Binary DCA on external cohorts |
| `dca_batch.py` | Batch DCA |
| `_dca_common.py` | Shared DCA utilities |
| `workflow_stage12.py` | Sequential Stage 1→2 operating-point workflow |

### 3) Publication tables / figures / interpretability (`scripts/publication/`)

| File | Role |
|------|------|
| `run_stage12_publication_figures.py` | Batch launcher for Stage12 publication outputs |
| `export_internal_cv_western_stage12.py` | Nested-CV summary (Western Stage 1–2) |
| `_internal_cv_export.py` | Nested-CV export helpers |
| `export_stage12_tables.py` | External validation tables |
| `plot_figure2_model_comparison.py` | Model comparison figures |
| `plot_figure3_external_validation.py` | External ROC + DCA panels |
| `plot_figure4_dca.py` | DCA-related plots |
| `plot_figure6_workflow.py` | Workflow figure |
| `interpret_binary_best_model_Stage12.py` | SHAP / PDP for final models |
| `repair_pct_pdp_stage1.py` | Stage 1 PDP repair helper |
| `_shap_composite_plot.py` | SHAP composite plotting |
| `_figure_common.py` | Shared figure helpers |
| `_pub_plot_style.py` | Publication plot style |
| `_interpret_feature_labels.py` | Feature label mapping |
| `_param_aggregate_utils.py` | Parameter aggregation utilities |

---

## Typical run order (high level)

1. Train candidates (via `Try.py` / individual trainers) → `outputs/Stage1|Stage2/<run>/`  
2. External validation + DCA → `external_validate_*.py`, `dca_*.py`  
3. Workflow JSON → `workflow_stage12.py`  
4. Publication batch → `run_stage12_publication_figures.py`  
5. Interpretability → `interpret_binary_best_model_Stage12.py`

---

## Methods notes (for readers)

- Nested cross-validation: **5 outer × 3 inner** folds; primary metric = mean outer-fold AUC.  
- Missing data: iterative imputation (`IterativeImputer`) with a random-forest regressor; fit on training folds only.  
- Feature processing within folds: ReliefF → RFE → Spearman redundancy filter → consensus selection.  
- Hyperparameters: inner-loop randomized search.
