param([string]$BuildDirectory='artifacts/game-finishing-margin/build')
$ErrorActionPreference='Stop'
$root=Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot 'patch-install-common.ps1')
$build=Join-Path $root $BuildDirectory
$manifest=Read-BrassJson (Join-Path $build 'build-manifest.json');$validation=Read-BrassJson (Join-Path $build 'validation.json')
$installed=Read-BrassJson (Join-Path $root 'artifacts/patch-installation.json')
Assert-BrassCandidate $manifest $validation $build
if($installed.status -ne 'installed' -or $installed.version -ne $manifest.version){throw 'Qualified candidate must be installed before archiving.'}
$archive=Join-Path $root ('artifacts/patch-releases/'+$manifest.version)
if(Test-Path -LiteralPath $archive){throw 'Preserve the existing immutable release archive.'}
function Copy-FinishingArchive([string]$source,[string]$relative){
 $target=Join-Path $archive $relative;New-Item -ItemType Directory -Path (Split-Path -Parent $target) -Force|Out-Null
 Copy-Item -LiteralPath $source -Destination $target
 if((Get-FileHash -LiteralPath $source).Hash -ne (Get-FileHash -LiteralPath $target).Hash){throw 'Archive copy differs.'}
}
foreach($file in $manifest.files){
 $path=Join-Path $root ('BrassBirminghamUnity_Data/Managed/'+$file.name)
 if((Get-FileHash -LiteralPath $path).Hash -ne $file.sha256 -or @($installed.files|Where-Object {$_.name -eq $file.name -and $_.sha256 -eq $file.sha256}).Count -ne 1){throw 'Installed fingerprint differs.'}
 Copy-FinishingArchive $path ('Managed/'+$file.name)
}
foreach($name in 'build-manifest.json','validation.json','local-ai-qualification.json','local-ai-native.json','era-native.log','deadline-boundaries.json','preserved-runtime.json','packaging-checks.json'){
 Copy-FinishingArchive (Join-Path $build $name) $name
}
foreach($pin in $manifest.sourcePins){
 $path=Join-Path $build ('runtime/'+$pin.path)
 if((Get-FileHash -LiteralPath $path).Hash -ne $pin.sha256){throw 'Compiled source snapshot changed.'}
 Copy-FinishingArchive $path $pin.path
}
foreach($name in 'baseline-native/report.json','baseline-native/native.log','candidate/report.json','candidate/native.log','comparison.json','deadline-fixtures.json','training-scope-before.json','training-scope-verified.json'){
 Copy-FinishingArchive (Join-Path $root ('artifacts/game-finishing-margin/'+$name)) ('timing/'+$name)
}
foreach($name in 'build-game-finishing-margin.ps1','archive-game-finishing-margin-release.ps1','patch-install-common.ps1','install-combined-patch.ps1','stage-archived-patch.ps1'){
 Copy-FinishingArchive (Join-Path $PSScriptRoot $name) ('scripts/'+$name)
}
foreach($name in 'FinishingMarginChecks.cs','FinishingMarginNativeTests.cs','FinishingMarginPreservedRuntimeChecks.cs','run-finishing-margin.ps1','qualify-finishing-margin.ps1'){
 Copy-FinishingArchive (Join-Path $root ('tests/BrassLocalAI/'+$name)) ('tests/BrassLocalAI/'+$name)
}
Copy-FinishingArchive (Join-Path $root 'docs/GAME_AI_FINISHING_MARGIN_1.4.28.md') 'docs/GAME_AI_FINISHING_MARGIN_1.4.28.md'
Copy-FinishingArchive (Join-Path $root 'artifacts/patch-installation.json') 'patch-installation.json'
[ordered]@{passed=$true;version=$manifest.version;verifiedUtc=[DateTime]::UtcNow.ToString('o');testedFiles=$manifest.files;previousReleaseBackup=$installed.previousReleaseBackup;scope='Installed DLLs exactly match the natively qualified game-only candidate. Training and user saves unchanged.'}|ConvertTo-Json -Depth 8|Set-Content -LiteralPath (Join-Path $archive 'installed-verification.json') -Encoding UTF8
$files=@(Get-ChildItem -LiteralPath $archive -Recurse -File|Sort-Object FullName|ForEach-Object{[ordered]@{path=[IO.Path]::GetRelativePath($archive,$_.FullName).Replace('\','/');sha256=(Get-FileHash -LiteralPath $_.FullName).Hash}})
[ordered]@{version=$manifest.version;createdUtc=[DateTime]::UtcNow.ToString('o');files=$files;priorArchive=$manifest.baseVersion;priorBackup=$installed.previousReleaseBackup}|ConvertTo-Json -Depth 8|Set-Content -LiteralPath (Join-Path $archive 'archive-manifest.json') -Encoding UTF8
& (Join-Path $PSScriptRoot 'stage-archived-patch.ps1') -Version $manifest.version -VerifyOnly
Write-Output ('Archived '+$manifest.version+': '+$files.Count+' verified files.')
