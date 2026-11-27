# Ablation D21: bỏ nhóm nhiệt độ (T1..T9, T_out, Tdewpoint) khỏi input, chạy lại toàn bộ pipeline
# vào thư mục RIÊNG (data/processed_no_temp, outputs_no_temp) nên KHÔNG ghi đè kết quả chính.
#
# Chạy từ thư mục energy_ts:
#   powershell -ExecutionPolicy Bypass -File scripts\run_ablation_no_temp.ps1
# Chạy nhanh để kiểm tra (2 epoch, 1 seed):
#   powershell -ExecutionPolicy Bypass -File scripts\run_ablation_no_temp.ps1 -Quick
param([switch]$Quick)
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

function Run {
    & python @args
    if ($LASTEXITCODE -ne 0) { throw "Lệnh lỗi: python $($args -join ' ')" }
}

$ex = "data.exclude_features=[T1,T2,T3,T4,T5,T6,T7,T8,T9,T_out,Tdewpoint]"
$pp = "paths.processed_dir=data/processed_no_temp"
$po = "paths.output_dir=outputs_no_temp"
$common = @("--set", $ex, "--set", $pp, "--set", $po)

$extra = @()
$seedArgs = @()
$evalSeeds = @()
if ($Quick) {
    $extra = @("--set", "train.epochs=2")
    $seedArgs = @("--seeds", "0")
    $evalSeeds = @("--set", "train.seeds=[0]")
    Write-Host ">> QUICK: 2 epoch, seed 0 (kết quả không có ý nghĩa)"
}

Run -m src.preprocess @common
Run -m src.baselines @common
Run -m src.train --model all @common @extra @seedArgs
Run -m src.evaluate @common @evalSeeds
Run -m src.training_report @common
Run -m src.compare_ablation --abl-out outputs_no_temp @evalSeeds
Write-Host ">> Xong. Xem outputs_no_temp\ablation_comparison.md"
