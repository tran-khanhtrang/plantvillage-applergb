# --- Config ---
$ErrorActionPreference = "Stop"
$VENV = ".venv_gpu"

Write-Host "==> Creating venv at $VENV" -ForegroundColor Cyan
python -m venv $VENV
. .\$VENV\Scripts\Activate.ps1

Write-Host "==> Upgrading pip" -ForegroundColor Cyan
python -m pip install --upgrade pip

Write-Host "==> Installing PyTorch (CUDA 12.1) for Windows" -ForegroundColor Cyan
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

Write-Host "==> Installing extra requirements" -ForegroundColor Cyan
pip install -r requirements.txt

# Quick GPU test
Write-Host "==> Verifying GPU availability..." -ForegroundColor Cyan
$testOut = python .\test_gpu.py 2>&1
New-Item -ItemType Directory -Path ".\logs" -ErrorAction SilentlyContinue | Out-Null
$testOut | Out-File ".\logs\test_gpu.txt" -Encoding UTF8

Write-Host $testOut -ForegroundColor Green
Write-Host "==> Done. Activate with:  . .\$VENV\Scripts\Activate.ps1" -ForegroundColor Green
