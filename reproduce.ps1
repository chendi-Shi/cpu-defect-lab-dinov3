$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$taskPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (!(Test-Path $taskPython)) { throw 'Run setup.ps1 first.' }
function Invoke-Lab([string[]]$Arguments) {
    & $taskPython @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Step failed: $Arguments" }
}
Invoke-Lab @('download_data.py', '--category', 'bottle')
Invoke-Lab @('download_data.py', '--category', 'screw')
Invoke-Lab @('lab.py', '--method', 'global')
Invoke-Lab @('lab.py', '--method', 'local', '--dims', '64', '--bank-size', '2000')
Invoke-Lab @('report.py')
Invoke-Lab @('lab.py', '--method', 'global_shared', '--dims', '64', '--bank-size', '2000')
Invoke-Lab @('analyze_errors.py')
Invoke-Lab @('round2_report.py')
Invoke-Lab @('validation_study.py')
Invoke-Lab @('plot_results.py')
Invoke-Lab @('run_final_experiments.py')
Invoke-Lab @('verify_delivery.py')
Invoke-Lab @('finalize.py')
Write-Host 'Reproduction complete. Open reports/FINAL.md or run start_demo.cmd.'
