# Streamlit Cloud 部署指南 / Deployment Guide

部署包已生成在：

**`F:\KeTi\Project\deploy\uc-stage12-western-demo`**

本机未检测到 Git，需你手动推送到 GitHub，再在 Streamlit Cloud 一键发布。

---

## Step 1 — 上传到 GitHub

### 方式 A：Git 命令行（推荐，**必须 Git LFS**）

模型文件约 **480 MB**（两个 `.joblib` 各 >100 MB），**必须使用 Git LFS**，否则 GitHub 会拒绝 push。

在 **Git Bash** 中：

```bash
# 1) 安装 Git LFS（一次性）
# 下载: https://git-lfs.com/
git lfs install

cd F:/KeTi/Project/deploy/uc-stage12-western-demo

# 2) 已含 .gitattributes 跟踪 *.joblib
git lfs track "*.joblib"

git init
git remote add origin https://github.com/LiuKaiNSN/uc-stage12-western-demo.git
git pull origin main --allow-unrelated-histories

git add .
git commit -m "Add Streamlit Cloud demo for Western Stage 1+2 pre-endoscopic triage"
git push -u origin main
```

若远程已有 Initial commit（README/LICENSE），第一次 push 前执行 `git pull origin main --allow-unrelated-histories`，解决冲突后保留本部署包中的 `README.md` 与 `streamlit_app.py`。

### 方式 B：GitHub 网页上传

1. 打开 https://github.com/LiuKaiNSN/uc-stage12-western-demo  
2. **Add file → Upload files**  
3. 将 `deploy\uc-stage12-western-demo` 下全部文件夹拖入（含 `src/`、`assets/`、`stage12_western/`、`streamlit_app.py`、`requirements.txt`）  
4. Commit  

> `assets/models/*.joblib` 合计约 **480 MB**，**不能**用普通网页上传；请用 **Git + Git LFS**（见方式 A）。

---

## Step 2 — Streamlit Community Cloud

1. 登录 https://share.streamlit.io （GitHub 授权）  
2. **Create app → From existing repo**  
3. 填写：

| 字段 | 值 |
|------|-----|
| Repository | `LiuKaiNSN/uc-stage12-western-demo` |
| Branch | `main` |
| Main file path | **`streamlit_app.py`** |
| App URL (optional) | `uc-stage12-western-demo` |

4. **Deploy**  
5. 等待 3–8 分钟（首次需安装 lightgbm/catboost）  
6. 成功后得到链接，例如：  
   **`https://uc-stage12-western-demo.streamlit.app`**

---

## Step 3 — 验证清单

- [ ] 首页无 “Startup validation failed”  
- [ ] Stage 1 / Stage 2 两个 Tab 均可输入并出概率  
- [ ] 右侧显示外验 AUC 等指标  
- [ ] SHAP 蜂群图 + rank01–05 PDP 能显示  
- [ ] 底部 enlarged SHAP correlation 图能显示  

本地可先测：

```powershell
cd F:\KeTi\Project\deploy\uc-stage12-western-demo
F:\KeTi\Project\.venv\Scripts\streamlit.exe run streamlit_app.py --server.port 8510
```

---

## Step 4 — 写入论文

**Data / Software availability (English):**

> An interactive research prototype implementing the locked Stage 1 (LightGBM) and Stage 2 (CatBoost) models is publicly available at **[YOUR STREAMLIT URL]** (accessed [DATE]). The tool accepts pre-endoscopic symptoms, signs, and laboratory inputs only; user data are not stored. Intended for research and peer review; not for standalone clinical diagnosis.

**中文（方法或补材）：**

> Stage 1 与 Stage 2 西医模型的交互式推理演示平台公开于 **[URL]**，供审稿专家及见刊后读者使用；输入不落盘，仅用于科研演示。

---

## 维护：从主项目重新打包

```powershell
python F:\KeTi\Project\Script_web\deploy\build_stage12_western_cloud.py
```

然后重新复制/恢复根目录的 `streamlit_app.py`、`requirements.txt`、`README.md`、`DEPLOY.md`（build 脚本会清空输出目录）。

---

## 模型与阈值（与论文一致）

| Stage | Model | Run | DCA threshold | External n |
|-------|-------|-----|-----------------|------------|
| 1 | LightGBM | 3_run_20260707_142952 | 0.34 | 597 |
| 2 | CatBoost | 4_run_20260707_030540 | 0.51 | 459 |

---

## 常见问题

**Q: 部署失败 ModuleNotFoundError**  
A: 确认仓库根目录有 `requirements.txt` 且含 `lightgbm`、`catboost`。

**Q: 图片不显示**  
A: 确认 `assets/figures/stage1|stage2/` 已上传且含 `external_shap_*.jpg` 与 `rank01_*.jpg` 等。

**Q: 想暂时去掉 SHAP/PDP 减小体积**  
A: 在 `western_stage12.yaml` 中删除 `figures.stage_publication` 整块即可（需重新 commit）。

---

## Paper 2 — Hub（第二篇：Baseline | TCM Integrated）

**红线：不要改动第一个 Cloud App 的 Main file（保持 `streamlit_app.py`）。**

### 本地预览 Hub

```powershell
cd F:\KeTi\Project\deploy\uc-stage12-western-demo
.\run_hub_local.bat
# 或
F:\KeTi\Project\.venv\Scripts\streamlit.exe run streamlit_app_hub.py --server.port 8511
```

检查：侧边栏可切换 **Baseline Model** / **TCM Integrated Model**；各含 Stage1/2、SHAP/PDP。

### 推送增量（同仓库）

```bash
cd F:/KeTi/Project/deploy/uc-stage12-western-demo
git lfs install
git add streamlit_app_hub.py run_hub_local.bat
git add stage12_baseline stage12_tcm
git add assets/models/tcm_stage1 assets/models/tcm_stage2
git add assets/figures/tcm_stage1 assets/figures/tcm_stage2
git add README.md DEPLOY.md
# 确认未误改 Paper 1：
git diff -- streamlit_app.py
git commit -m "Add Paper 2 hub: Baseline Model + TCM Integrated Model (Stage 1-2)"
git push origin main
```

新增 TCM 两个 `.joblib` 合计约 **500 MB**，必须走 **Git LFS**。

### 新建第二个 Streamlit Cloud App

1. https://share.streamlit.io → **New app**（不要编辑第一个 App）  
2. Repository: `LiuKaiNSN/uc-stage12-western-demo`  
3. Branch: `main`  
4. **Main file path: `streamlit_app_hub.py`**  
5. App URL (optional): `uc-stage12-western-tcm-hub`  
6. Deploy → 得到第二篇论文用的新链接  

### Hub 模型版本

| Module | Stage1 | Stage2 |
|--------|--------|--------|
| Baseline Model | LightGBM `3_run_20260707_142952`, thr 0.34 | CatBoost `4_run_20260707_030540`, thr 0.51 |
| TCM Integrated | RF `6_run_20260708_052605`, thr 0.43 | RF `6_run_20260707_111716`, thr 0.53 |

### 第二篇 Availability（英文草稿）

> An interactive research hub comparing the Baseline Model and the TCM Integrated Model for pre-endoscopic Stage 1–2 triage is available at **[HUB URL]**. The Paper 1 Baseline-only demonstration remains at **[PAPER1 URL]**. User inputs are not stored.
