@echo off
REM Quick local test for cloud bundle (port 8510)
cd /d F:\KeTi\Project\deploy\uc-stage12-western-demo
start "" http://localhost:8510
F:\KeTi\Project\.venv\Scripts\streamlit.exe run streamlit_app.py --server.port 8510
pause
