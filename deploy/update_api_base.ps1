# =============================================================================
# update_api_base.ps1
#
# Run this AFTER your EC2 instance is running with a known public IP.
# It updates the API_BASE in dashboard.html, then re-deploys to Firebase.
#
# Usage:
#   .\deploy\update_api_base.ps1 -EC2IP "13.235.XXX.XXX"
# =============================================================================
param(
    [Parameter(Mandatory=$true)]
    [string]$EC2IP
)

$frontendDir = Join-Path $PSScriptRoot "..\frontend"
$dashboardFile = Join-Path $frontendDir "dashboard.html"

if (-not (Test-Path $dashboardFile)) {
    Write-Error "Could not find $dashboardFile"
    exit 1
}

$newBase = "http://${EC2IP}:8000"
Write-Host "==> Updating API_BASE to: $newBase"

$content = Get-Content $dashboardFile -Raw
$updated = $content -replace 'const API_BASE\s*=\s*"[^"]*"', "const API_BASE = `"$newBase`""

if ($content -eq $updated) {
    Write-Warning "No change made — API_BASE pattern not found or already set to $newBase"
} else {
    Set-Content -Path $dashboardFile -Value $updated -NoNewline
    Write-Host "    dashboard.html updated."
}

Write-Host ""
Write-Host "==> Deploying to Firebase Hosting..."
cmd /c "firebase deploy --only hosting"

Write-Host ""
Write-Host "======================================================"
Write-Host "  Done! Frontend now points to: $newBase"
Write-Host "  Live site: https://traff2ic-detector.web.app"
Write-Host "======================================================"
