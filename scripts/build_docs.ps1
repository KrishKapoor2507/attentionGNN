# Rebuild every figure, table, the report (DOCX/PDF/MD) and the slides from saved runs:
#   powershell -ExecutionPolicy Bypass -File .\scripts\build_docs.ps1
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
$py = ".\.venv\Scripts\python.exe"
& $py scripts\build_results.py
& $py scripts\make_diagrams.py
& $py scripts\build_report.py
& $py scripts\build_pptx.py
Write-Host "`nDOCS OK. See the reports\ folder."
