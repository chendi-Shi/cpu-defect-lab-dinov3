# Run after the initial project reproduction; formal settings are frozen.
$ErrorActionPreference = 'Stop'
$screwPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (!(Test-Path -LiteralPath $screwPython -PathType Leaf)) {
    throw 'Project environment is missing. Run setup.ps1 and reproduce_frontier.ps1 first.'
}
function Invoke-ScrewRefinement {
    param([Parameter(Mandatory = $true)][string[]]$CommandArguments)
    & $screwPython @CommandArguments
    if ($LASTEXITCODE -ne 0) { throw "Screw experiment step failed: $($CommandArguments -join ' ')" }
}
Push-Location $PSScriptRoot
try {
    if (!(Test-Path -LiteralPath 'outputs\frontier\screw\seed-42\spatial_kcenter_r3\config.json')) {
        throw 'Original fixed screw experiment is missing. Run reproduce_frontier.ps1 first.'
    }
    Invoke-ScrewRefinement -CommandArguments @('screw_refine.py', '--stage', 'all')
    Invoke-ScrewRefinement -CommandArguments @('release_screw_refinement.py')
    Invoke-ScrewRefinement -CommandArguments @('verify_screw_refinement.py')
    Invoke-ScrewRefinement -CommandArguments @('screw_refine_report.py')
    Invoke-ScrewRefinement -CommandArguments @('-c', 'from finalize import source_archive; print(source_archive())')
    Write-Host 'Screw refinement complete. See reports/SCREW_REFINEMENT.md and start_demo.cmd.'
    Write-Host 'Actual browser acceptance is recorded separately; this script does not perform GUI checks.'
} finally { Pop-Location }
