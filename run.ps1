# Convenience script to start the Company RAG service in dev mode (Windows / PowerShell).
$ErrorActionPreference = "Stop"

Set-Location -Path $PSScriptRoot

if (-not (Test-Path ".venv")) {
    Write-Host "==> Creating virtualenv (.venv)"
    if (Get-Command py -ErrorAction SilentlyContinue) {
        py -3 -m venv .venv
    } else {
        python -m venv .venv
    }
}

# Activate venv
$activate = Join-Path ".venv" "Scripts\Activate.ps1"
. $activate

Write-Host "==> Installing dependencies"
python -m pip install --upgrade pip | Out-Null
pip install -q -r requirements.txt

if (-not (Test-Path ".env")) {
    Write-Host "==> Creating .env from .env.example (fill it in!)"
    Copy-Item ".env.example" ".env"
}

$envHost = if ($env:HOST) { $env:HOST } else { "0.0.0.0" }
$envPort = if ($env:PORT) { $env:PORT } else { "8000" }

Write-Host "==> Starting FastAPI on http://localhost:$envPort"
uvicorn app.main:app --host $envHost --port $envPort --reload
