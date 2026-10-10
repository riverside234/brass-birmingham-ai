param([string]$BuildDirectory='artifacts/statistics-geography-refinement-build')
$ErrorActionPreference='Stop'
$historyRoot=Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot 'patch-install-common.ps1')
$historyBuild=Join-Path $historyRoot $BuildDirectory
$candidate=Read-BrassJson (Join-Path $historyBuild 'build-manifest.json')
$destination=Join-Path $historyRoot ('artifacts/patch-releases/'+$candidate.version)
if(Test-Path -LiteralPath $destination){throw 'Release archive already exists; refusing to replace it.'}
$validation=Read-BrassJson (Join-Path $historyBuild 'validation.json')
$receipt=Read-BrassJson (Join-Path $historyRoot 'artifacts/patch-installation.json')
$tracker=Read-BrassJson (Join-Path $historyBuild 'statistics-installed.json')
$packaging=Read-BrassJson (Join-Path $historyBuild 'packaging-results.json')
Assert-BrassCandidate $candidate $validation $historyBuild
if($candidate.version -ne '1.4.30' -or $receipt.version -ne $candidate.version -or $receipt.status -ne 'installed' -or -not $tracker.passed -or -not $packaging.passed){throw 'Installed Geography and packaging qualification required.'}
function Copy-HistoryArchive([string]$source,[string]$relative){
 $target=[IO.Path]::GetFullPath((Join-Path $destination $relative))
 if(-not $target.StartsWith([IO.Path]::GetFullPath($destination)+[IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase)){throw 'Archive path escapes its directory.'}
 New-Item -ItemType Directory -Path (Split-Path -Parent $target) -Force|Out-Null
 Copy-Item -LiteralPath $source -Destination $target
 if((Get-FileHash -LiteralPath $source).Hash -ne (Get-FileHash -LiteralPath $target).Hash){throw 'Archive copy differs.'}
}
foreach($file in $candidate.files){
 $source=Join-Path $historyRoot ('BrassBirminghamUnity_Data/Managed/'+$file.name)
 if((Get-FileHash -LiteralPath $source).Hash -ne $file.sha256){throw 'Installed candidate differs.'}
 foreach($report in $tracker,$packaging){if(@($report.testedFiles|Where-Object {$_.name -eq $file.name -and $_.sha256 -eq $file.sha256}).Count -ne 1){throw 'Installed/packaging evidence belongs to another candidate.'}}
 Copy-HistoryArchive $source ('Managed/'+$file.name)
}
foreach($name in 'build-manifest','validation','local-ai-qualification','preserved-runtime','visual-checks','statistics-native','statistics-installed','history-native','packaging-results'){
 Copy-HistoryArchive (Join-Path $historyBuild ($name+'.json')) ($name+'.json')
}
Copy-HistoryArchive (Join-Path $historyRoot 'artifacts/patch-installation.json') 'patch-installation.json'
foreach($pin in $candidate.sourcePins){
 $source=Join-Path $historyBuild ('runtime/'+$pin.path)
 if((Get-FileHash -LiteralPath $source).Hash -ne $pin.sha256){throw 'Compiled source snapshot changed.'}
 Copy-HistoryArchive $source $pin.path
}
$baseline=Join-Path $historyRoot ('artifacts/patch-releases/'+$candidate.baseVersion)
foreach($name in 'archive-manifest.json','build-manifest.json','validation.json','patch-installation.json'){Copy-HistoryArchive (Join-Path $baseline $name) ('baseline/'+$candidate.baseVersion+'/'+$name)}
foreach($name in 'Assembly-CSharp.dll','BrassPatch.dll','BrassLegacyAI.dll'){Copy-HistoryArchive (Join-Path $baseline ('Managed/'+$name)) ('baseline/'+$candidate.baseVersion+'/Managed/'+$name)}
foreach($name in 'build-statistics-geography-refinement.ps1','archive-statistics-geography-refinement-release.ps1','install-combined-patch.ps1','patch-install-common.ps1','stage-archived-patch.ps1'){
 Copy-HistoryArchive (Join-Path $historyRoot ('scripts/'+$name)) ('scripts/'+$name)
}
foreach($folder in 'tests/BrassPatch/InformationUI','tests/BrassPatch/Statistics','tests/BrassPatch/History'){
 foreach($file in Get-ChildItem -LiteralPath (Join-Path $historyRoot $folder) -File|Where-Object Extension -in '.cs','.ps1'){Copy-HistoryArchive $file.FullName ($folder+'/'+$file.Name)}
}
Copy-HistoryArchive (Join-Path $historyRoot 'docs/STATISTICS_GEOGRAPHY_1.4.30.md') 'docs/STATISTICS_GEOGRAPHY_1.4.30.md'
Copy-HistoryArchive (Join-Path $historyRoot 'rules-audit/STATISTICS_GEOGRAPHY_MAP_REVISION_PLAN.md') 'rules-audit/STATISTICS_GEOGRAPHY_MAP_REVISION_PLAN.md'
$screens=Join-Path $historyRoot 'artifacts/card-tracker-player/artifacts/patch-build'
foreach($entry in $validation.visualChecks.cases){
 $source=Join-Path $screens $entry.name
 if((Get-FileHash -LiteralPath $source).Hash -ne $entry.sha256){throw 'Reviewed screenshot changed.'}
 Copy-HistoryArchive $source ('screenshots/'+$entry.name)
}
Copy-HistoryArchive (Join-Path $historyBuild 'statistics-native.log') 'logs/statistics-native.log'
Copy-HistoryArchive (Join-Path $historyBuild 'statistics-installed.log') 'logs/statistics-installed.log'
Copy-HistoryArchive (Join-Path $historyBuild 'history-native.log') 'logs/history-native.log'
$inventory=@(Get-ChildItem -LiteralPath $destination -Recurse -File|Sort-Object FullName|ForEach-Object{[ordered]@{path=[IO.Path]::GetRelativePath($destination,$_.FullName).Replace('\','/');sha256=(Get-FileHash -LiteralPath $_.FullName).Hash}})
[ordered]@{version=$candidate.version;createdUtc=[DateTime]::UtcNow.ToString('o');priorArchive=$candidate.baseVersion;priorBackup=$receipt.previousReleaseBackup;files=$inventory}|ConvertTo-Json -Depth 8|Set-Content -LiteralPath (Join-Path $destination 'archive-manifest.json') -Encoding utf8
& (Join-Path $PSScriptRoot 'stage-archived-patch.ps1') -Version $candidate.version -VerifyOnly
Write-Output ('Archived and verified Statistics geography '+$candidate.version+': '+$inventory.Count+' files.')
