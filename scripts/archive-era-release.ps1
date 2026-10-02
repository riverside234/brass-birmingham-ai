param([string]$BuildDirectory='artifacts/economy-v3/package-1.4.20-v4')
$ErrorActionPreference='Stop'
$root=Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot 'patch-install-common.ps1')
$build=Join-Path $root $BuildDirectory
$candidate=Read-BrassJson (Join-Path $build 'build-manifest.json')
$validation=Read-BrassJson (Join-Path $build 'validation.json')
$record=Read-BrassJson (Join-Path $root 'artifacts/patch-installation.json')
Assert-BrassCandidate $candidate $validation $build
if($record.version -ne $candidate.version -or $record.status -ne 'installed'){throw 'Qualified release must be installed before archiving'}
$destination=Join-Path $root ('artifacts/patch-releases/'+$candidate.version)
if(Test-Path -LiteralPath $destination){throw 'Release archive already exists; preserve it'}
function Copy-Archive([string]$source,[string]$relative){
 $target=Join-Path $destination $relative
 New-Item -ItemType Directory -Path (Split-Path -Parent $target) -Force|Out-Null
 Copy-Item -LiteralPath $source -Destination $target
 if((Get-FileHash -LiteralPath $source).Hash -ne (Get-FileHash -LiteralPath $target).Hash){throw 'Archive copy differs'}
}
foreach($file in $candidate.files){
 $installed=Join-Path $root ('BrassBirminghamUnity_Data/Managed/'+$file.name)
 if((Get-FileHash -LiteralPath $installed).Hash -ne $file.sha256 -or @($record.files|Where-Object {$_.name -eq $file.name -and $_.sha256 -eq $file.sha256}).Count -ne 1){throw 'Installed fingerprint differs'}
 Copy-Archive $installed ('Managed/'+$file.name)
}
foreach($name in 'build-manifest','validation','local-ai-qualification'){
 Copy-Archive (Join-Path $build ($name+'.json')) ($name+'.json')
}
foreach($entry in $validation.localAi.evidence){Copy-Archive (Join-Path $build ($entry.name+'.json')) ($entry.name+'.json')}
foreach($pin in $candidate.sourcePins){
 $source=Join-Path $build ('source/'+$pin.path)
 if((Get-FileHash -LiteralPath $source).Hash -ne $pin.sha256){throw 'Compiled snapshot changed'}
 Copy-Archive $source $pin.path
}
Copy-Archive (Join-Path $root 'artifacts/patch-installation.json') 'patch-installation.json'
Copy-Archive (Join-Path $root $validation.smoke.report) 'strength/bank-receipt.json'
foreach($name in 'build-era-ai.ps1','install-combined-patch.ps1','patch-install-common.ps1','stage-archived-patch.ps1','archive-era-release.ps1'){
 Copy-Archive (Join-Path $PSScriptRoot $name) ('scripts/'+$name)
}
foreach($file in Get-ChildItem -LiteralPath (Join-Path $root 'tests/MaskedISMCTS') -File|Where-Object Extension -in '.cs','.ps1','.py'){
 Copy-Archive $file.FullName ('tests/MaskedISMCTS/'+$file.Name)
}
foreach($file in Get-ChildItem -LiteralPath (Join-Path $root 'tests/BrassLocalAI') -File|Where-Object Extension -in '.cs','.ps1'){
 Copy-Archive $file.FullName ('tests/BrassLocalAI/'+$file.Name)
}
[ordered]@{passed=$true;version=$candidate.version;verifiedUtc=[DateTime]::UtcNow.ToString('o');testedFiles=$candidate.files;
 scope='Post-install exact-byte verification against the natively qualified package; preserves the prior recovery backup. No user saves modified.'}|ConvertTo-Json -Depth 8|Set-Content -LiteralPath (Join-Path $destination 'installed-verification.json') -Encoding UTF8
$inventory=@(Get-ChildItem -LiteralPath $destination -Recurse -File|Sort-Object FullName|ForEach-Object{
 [ordered]@{path=[IO.Path]::GetRelativePath($destination,$_.FullName).Replace('\','/');sha256=(Get-FileHash -LiteralPath $_.FullName).Hash}
})
[ordered]@{version=$candidate.version;createdUtc=[DateTime]::UtcNow.ToString('o');files=$inventory;priorArchive=$candidate.baseVersion;priorBackup=$record.previousReleaseBackup}|ConvertTo-Json -Depth 8|Set-Content -LiteralPath (Join-Path $destination 'archive-manifest.json') -Encoding UTF8
& (Join-Path $PSScriptRoot 'stage-archived-patch.ps1') -Version $candidate.version -VerifyOnly
Write-Output ('Archived qualified '+$candidate.version+': '+$inventory.Count+' files')


