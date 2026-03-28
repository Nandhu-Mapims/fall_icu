# Purpose: Create/activate virtual environment and run FastAPI with CUDA-ready dependencies.
param(
    [string]$Python = "python"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if (!(Test-Path ".venv")) {
    & $Python -m venv .venv
}

& ".\.venv\Scripts\python.exe" -m pip install --upgrade pip
& ".\.venv\Scripts\python.exe" -m pip install -r ".\requirements.txt"

if (!(Test-Path ".\models\yolov8n-pose.pt")) {
    Write-Host "Place yolov8n-pose.pt in backend\models before production use."
}

& ".\.venv\Scripts\python.exe" -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
