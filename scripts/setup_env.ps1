# One-command environment setup on Windows (run from the project folder in PowerShell):
#   powershell -ExecutionPolicy Bypass -File .\scripts\setup_env.ps1
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
if (-not (Test-Path .venv)) { py -3.11 -m venv .venv }
.\.venv\Scripts\python.exe -m pip install --upgrade pip
# requirements-docs.txt includes requirements-dev.txt and requirements.txt (Windows: also report/slide tooling)
.\.venv\Scripts\python.exe -m pip install -r requirements-docs.txt
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe scripts\download_data.py
Write-Host "`nSETUP OK. Next: .\.venv\Scripts\python.exe -m attngnn.cli smoke"
