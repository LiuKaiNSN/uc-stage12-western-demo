@echo off
REM Paper 2 Hub local preview (port 8511). Does NOT start Paper 1 streamlit_app.py.
cd /d F:\KeTi\Project\deploy\uc-stage12-western-demo
start "" http://localhost:8511
F:\KeTi\Project\.venv\Scripts\streamlit.exe run streamlit_app_hub.py --server.port 8511
pause
