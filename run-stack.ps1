#!/usr/bin/env powershell
# Script para levantar todo el stack en Windows (API + Streamlit)

Write-Host "╔════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║ Churn Prediction Dashboard - Stack Startup               ║" -ForegroundColor Cyan
Write-Host "╚════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan

$projectRoot = "D:\HDD\Data Science\Proyectos"
Set-Location $projectRoot

# Verificar que estamos en el lugar correcto
if (-not (Test-Path "api/main.py") -or -not (Test-Path "streamlit_app.py")) {
    Write-Host "Error: No se encontraron los archivos esperados en $projectRoot" -ForegroundColor Red
    exit 1
}

Write-Host "`n[1/3] Iniciando API en puerto 8000..." -ForegroundColor Yellow
Start-Process -FilePath "C:/Users/gasty/AppData/Local/Microsoft/WindowsApps/python3.12.exe" `
    -ArgumentList "-m", "uvicorn", "api.main:app", "--host", "127.0.0.1", "--port", "8000", "--reload" `
    -NoNewWindow -PassThru | Out-Null

Start-Sleep -Seconds 3

Write-Host "[✓] API iniciada (http://127.0.0.1:8000)" -ForegroundColor Green

Write-Host "`n[2/3] Iniciando Streamlit en puerto 8501..." -ForegroundColor Yellow
Set-Location $projectRoot
Start-Process -FilePath "C:/Users/gasty/AppData/Local/Microsoft/WindowsApps/python3.12.exe" `
    -ArgumentList "-m", "streamlit", "run", "streamlit_app.py", "--server.port", "8501", "--server.address", "127.0.0.1" `
    -NoNewWindow -PassThru | Out-Null

Start-Sleep -Seconds 5

Write-Host "[✓] Streamlit iniciado (http://127.0.0.1:8501)" -ForegroundColor Green

Write-Host "`n╔════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║ Stack Running!                                            ║" -ForegroundColor Cyan
Write-Host "╠════════════════════════════════════════════════════════════╣" -ForegroundColor Cyan
Write-Host "║                                                            ║" -ForegroundColor Cyan
Write-Host "║  Streamlit : http://127.0.0.1:8501                      ║" -ForegroundColor Cyan
Write-Host "║  API       : http://127.0.0.1:8000                      ║" -ForegroundColor Cyan
Write-Host "║                                                            ║" -ForegroundColor Cyan
Write-Host "║  Presiona cualquier tecla para parar todo...             ║" -ForegroundColor Cyan
Write-Host "║                                                            ║" -ForegroundColor Cyan
Write-Host "╚════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan

$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")

Write-Host "`nParando servicios..." -ForegroundColor Yellow
Stop-Process -Name "python*" -Force -ErrorAction SilentlyContinue
Write-Host "[✓] Servicios detenidos" -ForegroundColor Green
