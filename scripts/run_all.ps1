# Chạy toàn bộ pipeline trên Windows (bản PowerShell của run_all.sh), theo đúng thứ tự trong doc/12_runbook.md
#   powershell -ExecutionPolicy Bypass -File scripts\run_all.ps1          # chạy đầy đủ
#   powershell -ExecutionPolicy Bypass -File scripts\run_all.ps1 -Quick   # 2 epoch, 1 seed, chỉ để kiểm tra pipeline
param([switch]$Quick)
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

function Run {
    & python @args
    if ($LASTEXITCODE -ne 0) { throw "Lệnh lỗi: python $($args -join ' ')" }
}

$extra = @(); $seeds = @(); $evalSeeds = @()
if ($Quick) {
    $extra = @("--set", "train.epochs=2")
    $seeds = @("--seeds", "0")
    $evalSeeds = @("--set", "train.seeds=[0]")
    Write-Host ">> QUICK: 2 epoch, seed 0 (kết quả không có ý nghĩa khoa học)"
}

Run -m pytest -q tests
Run -m src.eda
Run -m src.preprocess
Run -m src.baselines
Run -m src.train --model all @seeds @extra
Run -m src.evaluate @evalSeeds
Run -m src.attention --seed 0
Run -m src.training_report
Write-Host ">> Xong. Xem outputs\results_report.md, outputs\attention\ và notebooks\report.ipynb"
