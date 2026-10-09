# Reproduce baseline history and the fixed DINOSaur CPU study.
# Interrupted runs can reuse verified downloads and feature caches.
# Browser GUI acceptance is a separate manual step and is not run here.
$ErrorActionPreference = 'Stop'
$taskPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (!(Test-Path -LiteralPath $taskPython -PathType Leaf)) {
    throw 'Project environment is missing. Run setup.ps1 first.'
}
if ($PSVersionTable.PSEdition -eq 'Core') {
    $taskPowerShell = Join-Path $PSHOME 'pwsh.exe'
} else {
    $taskPowerShell = Join-Path $PSHOME 'powershell.exe'
}

function Invoke-Frontier {
    param([Parameter(Mandatory = $true)][string[]]$CommandArguments)
    & $taskPython @CommandArguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python step failed (exit $LASTEXITCODE): $($CommandArguments -join ' ')"
    }
}

Push-Location $PSScriptRoot
try {
    Write-Host 'Reproducing the complete ResNet18 baseline history first...'
    & $taskPowerShell -NoLogo -NoProfile -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot 'reproduce.ps1')
    if ($LASTEXITCODE -ne 0) {
        throw "Baseline reproduction failed (exit $LASTEXITCODE)."
    }

    # Keep the CPU torch/torchvision versions supplied by setup.ps1.
    Invoke-Frontier -CommandArguments @('-m', 'pip', 'install', '--no-deps', '-r', 'requirements-frontier.txt')
    # Small runtime dependencies omitted by --no-deps; no torch packages here.
    Invoke-Frontier -CommandArguments @('-m', 'pip', 'install', 'filelock', 'fsspec', 'packaging', 'PyYAML', 'requests', 'tqdm', 'typing-extensions')
    Invoke-Frontier -CommandArguments @('frontier_setup.py')

    foreach ($taskCategory in @('bottle', 'screw', 'hazelnut', 'metal_nut')) {
        Invoke-Frontier -CommandArguments @('download_data.py', '--category', $taskCategory)
    }
    foreach ($taskCategory in @('hazelnut', 'metal_nut')) {
        $taskThreads = if ($taskCategory -eq 'hazelnut') { '1' } else { '4' }
        Invoke-Frontier -CommandArguments @('lab.py', '--category', $taskCategory, '--method', 'local', '--dims', '64', '--bank-size', '2000', '--threads', $taskThreads)
    }

    Invoke-Frontier -CommandArguments @('frontier.py', '--continual')
    Invoke-Frontier -CommandArguments @('frontier_report.py')
    Invoke-Frontier -CommandArguments @('verify_frontier.py')
    Invoke-Frontier -CommandArguments @('-c', 'from finalize import source_archive; print(source_archive())')
    Write-Host 'Reproduction complete. Review reports/frontier and the refreshed delivery archive.'
    Write-Host 'Browser GUI acceptance was not run automatically. Start start_demo.cmd and check the page separately.'
} finally {
    Pop-Location
}
