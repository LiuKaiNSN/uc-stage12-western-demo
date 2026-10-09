@echo off
REM Thesis Ch.4 full App local preview (port 8512). Does NOT start Paper1/Hub.
cd /d F:\KeTi\Project\deploy\uc-stage12-western-demo
start "" http://localhost:8512
F:\KeTi\Project\.venv\Scripts\streamlit.exe run streamlit_app_full.py --server.port 8512
pause
