# 演示启动脚本：一键起后端 + 前端，并等待就绪
# 用法：右键「使用 PowerShell 运行」，或  powershell -ExecutionPolicy Bypass -File demo\run_demo.ps1
param([switch]$NoFrontend)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$py = Join-Path $root ".venv\Scripts\python.exe"

if (-not (Test-Path $py)) { throw "找不到虚拟环境 python：$py" }

Write-Host "[1/4] 启动后端 8000 ..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList @(
    "-NoExit", "-Command",
    "`$host.UI.RawUI.WindowTitle='RAG-backend'; Set-Location '$root\backend\src'; & '$py' main.py"
)

Write-Host "[2/4] 等待 /health（首次要加载两个模型，最长约 2 分钟）..." -ForegroundColor Cyan
$ok = $false
for ($i = 0; $i -lt 90; $i++) {
    Start-Sleep -Seconds 2
    try {
        $r = Invoke-RestMethod -Uri "http://127.0.0.1:8000/health" -TimeoutSec 3
        if ($r.status -eq "ok" -or $r) { $ok = $true; break }
    } catch { }
}
if (-not $ok) { throw "后端 90 秒内未就绪，去看后端窗口的报错" }
Write-Host "      后端就绪：$($r | ConvertTo-Json -Compress)" -ForegroundColor Green

if (-not $NoFrontend) {
    Write-Host "[3/4] 启动前端 8501 ..." -ForegroundColor Cyan
    Start-Process powershell -ArgumentList @(
        "-NoExit", "-Command",
        "`$host.UI.RawUI.WindowTitle='RAG-frontend'; Set-Location '$root\frontend'; & '$py' -m streamlit run app.py --server.port 8501"
    )
    Start-Sleep -Seconds 3
    Start-Process "http://127.0.0.1:8501"
} else {
    Write-Host "[3/4] 跳过前端（-NoFrontend）" -ForegroundColor Yellow
}

Write-Host "[4/4] 演示样例在 demo\samples：" -ForegroundColor Cyan
Get-ChildItem (Join-Path $PSScriptRoot "samples") | ForEach-Object { Write-Host "      - $($_.Name)" }
Write-Host ""
Write-Host "演示顺序（3 分钟）：" -ForegroundColor Green
Write-Host "  1) 传 01_陷阱合同.pdf，问「押金不退怎么办」→ 看高风险清单和引用原文"
Write-Host "  2) 传 02_合规合同.pdf，问同一句 → 看不误报"
Write-Host "  3) 传 03_扫描件.pdf → 看 OCR 降级后仍能出结论"
Write-Host "  断网时用预先录屏，接口文档在 http://127.0.0.1:8000/docs"
