[CmdletBinding()]
param(
    [string]$OutputPath,
    [switch]$IncludeDatabase
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

# Keep this script ASCII-only so Windows PowerShell 5 does not decode the
# Chinese catalog path through the active legacy code page.
$catalogDirectory = "A$([char]0x80A1)$([char]0x6570)$([char]0x636E)"
$catalogFile = "$([char]0x80A1)$([char]0x7968)$([char]0x5217)$([char]0x8868).csv"
$catalogPath = "$catalogDirectory/$catalogFile"

if (-not (Test-Path "frontend/dist/index.html")) {
    throw "frontend/dist is missing. Run the frontend production build first."
}

if (-not $OutputPath) {
    $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $OutputPath = Join-Path $projectRoot "tmp/kronos-release-$stamp.tar.gz"
}

$output = [System.IO.Path]::GetFullPath($OutputPath)
$outputDirectory = Split-Path -Parent $output
New-Item -ItemType Directory -Force -Path $outputDirectory | Out-Null

$releasePaths = @(
    ".gitignore",
    "README.md",
    "requirements.txt",
    "src",
    "scripts",
    "third_party/Kronos",
    "config",
    "deploy",
    "frontend/dist",
    $catalogPath
)

if ($IncludeDatabase) {
    $releasePaths += @(
        "data/kronos.db",
        "data/raw/a_share_download_meta.json"
    )
}

$missing = $releasePaths | Where-Object { -not (Test-Path $_) }
if ($missing) {
    throw "Release inputs are missing: $($missing -join ', ')"
}

$tarArguments = @(
    "-czf", $output,
    "--exclude=__pycache__",
    "--exclude=*.pyc",
    "--exclude=.pytest_cache"
) + $releasePaths

& tar @tarArguments
if ($LASTEXITCODE -ne 0) {
    throw "tar failed with exit code $LASTEXITCODE"
}

$archive = Get-Item $output
Write-Host ("Release archive: {0}" -f $archive.FullName)
Write-Host ("Size: {0:N2} MiB" -f ($archive.Length / 1MB))
Write-Host ("Database included: {0}" -f [bool]$IncludeDatabase)
