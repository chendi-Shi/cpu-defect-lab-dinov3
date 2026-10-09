$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$taskPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (!(Test-Path $taskPython)) { throw 'Create .venv and install requirements.txt first; see README.md.' }
& $taskPython lab.py --method global
if ($LASTEXITCODE -ne 0) { throw 'Global experiment failed' }
& $taskPython lab.py --method local --dims 64 --bank-size 2000
if ($LASTEXITCODE -ne 0) { throw 'Local experiment failed' }
& $taskPython report.py
if ($LASTEXITCODE -ne 0) { throw 'Report generation failed' }
