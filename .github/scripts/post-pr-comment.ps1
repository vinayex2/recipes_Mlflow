# Azure DevOps PR comment poster
# This script posts the contents of a summary file to the current PR thread using the Azure DevOps REST API.

param(
    [string]$SummaryFile = 'pr_review_summary.md'
)

if (-not $env:SYSTEM_PULLREQUEST_PULLREQUESTID) {
    Write-Host 'Skipping PR comment: not running in a PR context.'
    exit 0
}

if (-not (Test-Path $SummaryFile)) {
    Write-Error "Summary file not found: $SummaryFile"
    exit 1
}

$body = Get-Content -Raw -Path $SummaryFile
$comment = [ordered]@{
    comments = @(
        [ordered]@{
            parentCommentId = 0
            content = $body
            commentType = 1
        }
    )
    status = 'active'
}

$collectionUri = $env:SYSTEM_COLLECTIONURI.TrimEnd('/')
$project = $env:SYSTEM_TEAMPROJECT
$repoId = $env:BUILD_REPOSITORY_ID
$prId = $env:SYSTEM_PULLREQUEST_PULLREQUESTID
$uri = "$collectionUri/$project/_apis/git/repositories/$repoId/pullRequests/$prId/threads?api-version=7.1-preview.1"

Write-Host "Posting PR comment to $uri"
try {
    Invoke-RestMethod -Uri $uri -Method Post -Headers @{ Authorization = "Bearer $env:SYSTEM_ACCESSTOKEN" } -ContentType 'application/json' -Body ($comment | ConvertTo-Json -Depth 10)
    Write-Host '✅ PR comment posted successfully.'
} catch {
    Write-Error "Failed to post PR comment: $_"
    exit 1
}
