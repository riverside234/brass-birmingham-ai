$ErrorActionPreference='Stop'
$reviewRoot=Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot 'patch-install-common.ps1')
$reviewBuild=Join-Path $reviewRoot 'artifacts/statistics-review-build'
$reviewCandidate=Read-BrassJson (Join-Path $reviewBuild 'build-manifest.json')
$reviewDestination=Join-Path $reviewRoot ('artifacts/patch-releases/'+$reviewCandidate.version)
if(Test-Path -LiteralPath $reviewDestination){throw 'Release archive exists; refusing to replace it.'}
$reviewValidation=Read-BrassJson (Join-Path $reviewBuild 'validation.json')
$reviewRecord=Read-BrassJson (Join-Path $reviewRoot 'artifacts/patch-installation.json')
$reviewInstalled=Read-BrassJson (Join-Path $reviewBuild 'statistics-installed.json')
$reviewMenu=Read-BrassJson (Join-Path $reviewBuild 'menu-installed.json')
$reviewPackaging=Read-BrassJson (Join-Path $reviewBuild 'packaging-results.json')
Assert-BrassCandidate $reviewCandidate $reviewValidation $reviewBuild
if($reviewCandidate.version -ne '1.4.17' -or $reviewRecord.version -ne $reviewCandidate.version -or $reviewRecord.status -ne 'installed' -or -not $reviewInstalled.passed -or -not $reviewMenu.passed -or -not $reviewPackaging.passed){throw 'Installed statistics, menu and packaging qualification are required.'}
function Copy-ReviewArchive([string]$source,[string]$relative){
 $target=[IO.Path]::GetFullPath((Join-Path $reviewDestination $relative))
 if(-not $target.StartsWith([IO.Path]::GetFullPath($reviewDestination)+[IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase)){throw 'Archive path escapes its directory.'}
 New-Item -ItemType Directory -Path (Split-Path -Parent $target) -Force|Out-Null
 Copy-Item -LiteralPath $source -Destination $target
 if((Get-FileHash -LiteralPath $source).Hash -ne (Get-FileHash -LiteralPath $target).Hash){throw 'Archive copy differs.'}
}
foreach($file in $reviewCandidate.files){
 $source=Join-Path $reviewRoot ('BrassBirminghamUnity_Data/Managed/'+$file.name)
 if((Get-FileHash -LiteralPath $source).Hash -ne $file.sha256){throw 'Installed candidate differs.'}
 foreach($report in $reviewInstalled,$reviewMenu,$reviewPackaging){if(@($report.testedFiles|Where-Object {$_.name -eq $file.name -and $_.sha256 -eq $file.sha256}).Count -ne 1){throw 'Installed or packaging evidence differs.'}}
 Copy-ReviewArchive $source ('Managed/'+$file.name)
}
foreach($name in 'build-manifest','validation','local-ai-qualification','preserved-runtime','statistics-native','tracker-native','presentation-native','link-native','menu-native','statistics-installed','menu-installed','packaging-results'){
 Copy-ReviewArchive (Join-Path $reviewBuild ($name+'.json')) ($name+'.json')
}
Copy-ReviewArchive (Join-Path $reviewRoot 'artifacts/patch-installation.json') 'patch-installation.json'
foreach($pin in $reviewCandidate.sourcePins){$source=Join-Path $reviewBuild ('runtime/'+$pin.path);if((Get-FileHash -LiteralPath $source).Hash -ne $pin.sha256){throw 'Compiled source snapshot changed.'};Copy-ReviewArchive $source $pin.path}
foreach($version in '1.4.13','1.4.16'){
 $baseline=Join-Path $reviewRoot ('artifacts/patch-releases/'+$version)
 foreach($name in 'archive-manifest.json','build-manifest.json','validation.json'){Copy-ReviewArchive (Join-Path $baseline $name) ('baseline/'+$version+'/'+$name)}
 foreach($file in Get-ChildItem -LiteralPath (Join-Path $baseline 'Managed') -File|Where-Object Name -in 'Assembly-CSharp.dll','BrassPatch.dll','BrassLegacyAI.dll'){Copy-ReviewArchive $file.FullName ('baseline/'+$version+'/Managed/'+$file.Name)}
}
foreach($name in 'build-statistics-review.ps1','archive-statistics-review-release.ps1','install-combined-patch.ps1','install-patch.ps1','uninstall-patch.ps1','patch-install-common.ps1','stage-archived-patch.ps1'){Copy-ReviewArchive (Join-Path $reviewRoot ('scripts/'+$name)) ('scripts/'+$name)}
foreach($folder in 'tests/BrassPatch/Statistics','tests/BrassPatch/CardTracker','tests/BrassPatch/Presentation','tests/BrassPatch/CityLocator'){
 $source=Join-Path $reviewRoot $folder
 foreach($file in Get-ChildItem -LiteralPath $source -Recurse -File|Where-Object Extension -in '.cs','.ps1'){Copy-ReviewArchive $file.FullName ($folder+'/'+[IO.Path]::GetRelativePath($source,$file.FullName))}
}
foreach($name in 'LocalAiUnityTests.cs','AiExecutionResourceTests.cs'){Copy-ReviewArchive (Join-Path $reviewRoot ('tests/BrassLocalAI/'+$name)) ('tests/BrassLocalAI/'+$name)}
Copy-ReviewArchive (Join-Path $reviewRoot 'docs/END_GAME_STATISTICS_1.4.17.md') 'docs/END_GAME_STATISTICS_1.4.17.md'
Copy-ReviewArchive (Join-Path $reviewRoot 'rules-audit/END_GAME_STATISTICS_IMPROVEMENT_PLAN.md') 'rules-audit/END_GAME_STATISTICS_IMPROVEMENT_PLAN.md'
$screens=Join-Path $reviewRoot 'artifacts/card-tracker-player/artifacts/patch-build'
foreach($file in Get-ChildItem -LiteralPath $screens -File|Where-Object {$_.Name -like 'statistics-*.png' -or $_.Name -like '*-card-tracker-native.log'}){Copy-ReviewArchive $file.FullName ('screenshots/'+$file.Name)}
$reviewInventory=@(Get-ChildItem -LiteralPath $reviewDestination -Recurse -File|Sort-Object FullName|ForEach-Object{[ordered]@{path=[IO.Path]::GetRelativePath($reviewDestination,$_.FullName).Replace('\','/');sha256=(Get-FileHash -LiteralPath $_.FullName).Hash}})
[ordered]@{version=$reviewCandidate.version;createdUtc=[DateTime]::UtcNow.ToString('o');priorArchive='1.4.16';helperSourceArchive='1.4.13';priorBackup=$reviewRecord.previousReleaseBackup;files=$reviewInventory}|ConvertTo-Json -Depth 8|Set-Content -LiteralPath (Join-Path $reviewDestination 'archive-manifest.json') -Encoding utf8
& (Join-Path $PSScriptRoot 'stage-archived-patch.ps1') -Version $reviewCandidate.version -VerifyOnly
Write-Output ('Archived and verified statistics '+$reviewCandidate.version+': '+$reviewInventory.Count+' files.')
