param([string]$Python = 'python')
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
& $Python -m venv .venv
if ($LASTEXITCODE -ne 0) { throw 'Unable to create environment. Use Python 3.10 to 3.12.' }
$taskPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
& $taskPython -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
if ($LASTEXITCODE -ne 0) { throw 'PyTorch installation failed' }
& $taskPython -m pip install numpy Pillow matplotlib
if ($LASTEXITCODE -ne 0) { throw 'Remaining dependency installation failed' }
Write-Host 'Environment ready. See README.md for reproduction steps.'
