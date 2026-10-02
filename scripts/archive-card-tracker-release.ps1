$ErrorActionPreference='Stop'
$trackerRoot=Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot 'patch-install-common.ps1')
$trackerBuild=Join-Path $trackerRoot 'artifacts/card-tracker-build'
$trackerCandidate=Read-BrassJson (Join-Path $trackerBuild 'build-manifest.json')
$trackerDestination=Join-Path $trackerRoot ('artifacts/patch-releases/'+$trackerCandidate.version)
if(Test-Path -LiteralPath $trackerDestination){throw 'Release archive exists; refusing to replace it.'}
$trackerValidation=Read-BrassJson (Join-Path $trackerBuild 'validation.json')
$trackerRecord=Read-BrassJson (Join-Path $trackerRoot 'artifacts/patch-installation.json')
$trackerInstalled=Read-BrassJson (Join-Path $trackerBuild 'tracker-installed.json')
$trackerMenu=Read-BrassJson (Join-Path $trackerBuild 'menu-installed.json')
$trackerPackaging=Read-BrassJson (Join-Path $trackerBuild 'packaging-results.json')
Assert-BrassCandidate $trackerCandidate $trackerValidation $trackerBuild
if($trackerCandidate.version -ne '1.4.13' -or $trackerRecord.version -ne $trackerCandidate.version -or $trackerRecord.status -ne 'installed' -or -not $trackerInstalled.passed -or -not $trackerMenu.passed -or -not $trackerPackaging.passed){throw 'Installed tracker, menu and packaging qualification are required.'}
function Copy-TrackerArchive([string]$source,[string]$relative){
 $target=Join-Path $trackerDestination $relative
 New-Item -ItemType Directory -Path (Split-Path -Parent $target) -Force|Out-Null
 Copy-Item -LiteralPath $source -Destination $target
 if((Get-FileHash -LiteralPath $source).Hash -ne (Get-FileHash -LiteralPath $target).Hash){throw 'Archive copy differs.'}
}
foreach($file in $trackerCandidate.files){
 $source=Join-Path $trackerRoot ('BrassBirminghamUnity_Data/Managed/'+$file.name)
 if((Get-FileHash -LiteralPath $source).Hash -ne $file.sha256){throw 'Installed candidate differs.'}
 foreach($report in $trackerInstalled,$trackerMenu,$trackerPackaging){if(@($report.testedFiles|Where-Object {$_.name -eq $file.name -and $_.sha256 -eq $file.sha256}).Count -ne 1){throw 'Installed/packaging evidence differs.'}}
 Copy-TrackerArchive $source ('Managed/'+$file.name)
}
foreach($name in 'build-manifest','validation','local-ai-qualification','preserved-runtime','tracker-native','menu-native','tracker-installed','menu-installed','packaging-results'){
 Copy-TrackerArchive (Join-Path $trackerBuild ($name+'.json')) ($name+'.json')
}
Copy-TrackerArchive (Join-Path $trackerBuild 'Managed/Assembly-CSharp.dll.manifest.json') 'Assembly-CSharp.dll.manifest.json'
Copy-TrackerArchive (Join-Path $trackerRoot 'artifacts/patch-installation.json') 'patch-installation.json'
foreach($pin in $trackerCandidate.sourcePins){
 $source=Join-Path $trackerBuild ('runtime/'+$pin.path)
 if((Get-FileHash -LiteralPath $source).Hash -ne $pin.sha256){throw 'Compiled source snapshot changed.'}
 Copy-TrackerArchive $source $pin.path
}
$trackerBase=Join-Path $trackerBuild ('baseline/'+$trackerCandidate.baseVersion)
if(-not(Test-Path -LiteralPath $trackerBase)){$trackerBase=Join-Path $trackerRoot ('artifacts/patch-releases/'+$trackerCandidate.baseVersion)}
foreach($file in Get-ChildItem -LiteralPath $trackerBase -Recurse -File){Copy-TrackerArchive $file.FullName ('baseline/'+$trackerCandidate.baseVersion+'/'+[IO.Path]::GetRelativePath($trackerBase,$file.FullName))}
foreach($name in 'build-card-tracker.ps1','archive-card-tracker-release.ps1','install-combined-patch.ps1','install-patch.ps1','uninstall-patch.ps1','patch-install-common.ps1','stage-archived-patch.ps1'){
 Copy-TrackerArchive (Join-Path $trackerRoot ('scripts/'+$name)) ('scripts/'+$name)
}
Copy-TrackerArchive (Join-Path $trackerRoot 'src/BrassPatch.Weaver/Program.cs') 'src/BrassPatch.Weaver/Program.cs'
foreach($folder in 'tests/BrassPatch/CardTracker','tests/BrassPatch/Presentation','tests/BrassPatch/CityLocator'){
 $source=Join-Path $trackerRoot $folder
 foreach($file in Get-ChildItem -LiteralPath $source -Recurse -File|Where-Object Extension -in '.cs','.ps1'){Copy-TrackerArchive $file.FullName ($folder+'/'+[IO.Path]::GetRelativePath($source,$file.FullName))}
}
foreach($name in 'LocalAiUnityTests.cs','AiExecutionResourceTests.cs'){Copy-TrackerArchive (Join-Path $trackerRoot ('tests/BrassLocalAI/'+$name)) ('tests/BrassLocalAI/'+$name)}
Copy-TrackerArchive (Join-Path $trackerRoot 'artifacts/patch-releases/1.4.10/tests/BrassPatch/Statistics/StatisticsNativeTests.cs') 'tests/BrassPatch/Statistics/StatisticsNativeTests.cs'
Copy-TrackerArchive (Join-Path $trackerRoot 'docs/CARD_TRACKER_APPEARANCE_1.4.13.md') 'docs/CARD_TRACKER_APPEARANCE_1.4.13.md'
Copy-TrackerArchive (Join-Path $trackerRoot 'rules-audit/CARD_TRACKER_UI_PLAN.md') 'rules-audit/CARD_TRACKER_UI_PLAN.md'
Copy-TrackerArchive (Join-Path $trackerRoot 'rules-audit/card-tracker-deck-evidence.json') 'rules-audit/card-tracker-deck-evidence.json'
$trackerScreens=Join-Path $trackerRoot 'artifacts/card-tracker-player/artifacts/patch-build'
foreach($file in Get-ChildItem -LiteralPath $trackerScreens -File|Where-Object {$_.Name -like 'tracker-*.png' -or $_.Name -like '*card-tracker-native.log'}){Copy-TrackerArchive $file.FullName ('screenshots/'+$file.Name)}
$trackerInventory=@(Get-ChildItem -LiteralPath $trackerDestination -Recurse -File|Sort-Object FullName|ForEach-Object{[ordered]@{path=[IO.Path]::GetRelativePath($trackerDestination,$_.FullName).Replace('\','/');sha256=(Get-FileHash -LiteralPath $_.FullName).Hash}})
[ordered]@{version=$trackerCandidate.version;createdUtc=[DateTime]::UtcNow.ToString('o');priorArchive=$trackerCandidate.baseVersion;priorBackup=$trackerRecord.previousReleaseBackup;files=$trackerInventory}|ConvertTo-Json -Depth 8|Set-Content -LiteralPath (Join-Path $trackerDestination 'archive-manifest.json') -Encoding utf8
& (Join-Path $PSScriptRoot 'stage-archived-patch.ps1') -Version $trackerCandidate.version -VerifyOnly
Write-Output ('Archived and verified Card Tracker '+$trackerCandidate.version+': '+$trackerInventory.Count+' files.')
