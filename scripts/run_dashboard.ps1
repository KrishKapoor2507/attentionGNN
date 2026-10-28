# Start the dashboard on THIS computer only (not reachable from the network):
#   powershell -ExecutionPolicy Bypass -File .\scripts\run_dashboard.ps1
Set-Location (Split-Path $PSScriptRoot -Parent)
.\.venv\Scripts\python.exe -m streamlit run app\streamlit_app.py --server.address localhost --server.port 8501 --server.headless false
