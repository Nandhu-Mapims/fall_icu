# Purpose: Start backend quickly for local development without reinstalling dependencies every run.
param(
    [string]$Python = "python"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# Always run relative paths from backend directory.
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

if (!(Test-Path ".\.venv")) {
    & $Python -m venv .venv
}

if (!(Test-Path ".\.venv\Scripts\python.exe")) {
    throw "Virtual environment is incomplete. Recreate backend\.venv."
}

$HasUvicorn = & ".\.venv\Scripts\python.exe" -c "import uvicorn" 2>$null
if ($LASTEXITCODE -ne 0) {
    & ".\.venv\Scripts\python.exe" -m pip install --upgrade pip
    & ".\.venv\Scripts\python.exe" -m pip install -r ".\requirements.txt"
}

if (!(Test-Path ".\models\yolov8n-pose.pt")) {
    Write-Host "Warning: model missing at backend\models\yolov8n-pose.pt"
}

& ".\.venv\Scripts\python.exe" -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
