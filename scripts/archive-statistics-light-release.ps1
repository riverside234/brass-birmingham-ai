param([string]$BuildDirectory='artifacts/statistics-light-build')
$ErrorActionPreference='Stop'
$statRoot=Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot 'patch-install-common.ps1')
$statBuild=Join-Path $statRoot $BuildDirectory
$statCandidate=Read-BrassJson (Join-Path $statBuild 'build-manifest.json')
$statDestination=Join-Path $statRoot ('artifacts/patch-releases/'+$statCandidate.version)
if(Test-Path -LiteralPath $statDestination){throw 'Release archive exists; refusing to replace it.'}
$statValidation=Read-BrassJson (Join-Path $statBuild 'validation.json')
$statRecord=Read-BrassJson (Join-Path $statRoot 'artifacts/patch-installation.json')
$statInstalled=@('tracker','statistics','menu')|ForEach-Object {Read-BrassJson (Join-Path $statBuild ($_+'-installed.json'))}
$statPackaging=Read-BrassJson (Join-Path $statBuild 'packaging-results.json')
Assert-BrassCandidate $statCandidate $statValidation $statBuild
if($statCandidate.version -ne '1.4.21' -or $statRecord.version -ne $statCandidate.version -or $statRecord.status -ne 'installed' -or @($statInstalled|Where-Object {-not $_.passed}).Count -or -not $statPackaging.passed){throw 'Installed Statistics, Card Tracker, menu and packaging qualification are required.'}
function Copy-StatArchive([string]$source,[string]$relative){
 $target=[IO.Path]::GetFullPath((Join-Path $statDestination $relative))
 if(-not $target.StartsWith([IO.Path]::GetFullPath($statDestination)+[IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase)){throw 'Archive path escapes its directory.'}
 New-Item -ItemType Directory -Path (Split-Path -Parent $target) -Force|Out-Null
 Copy-Item -LiteralPath $source -Destination $target
 if((Get-FileHash -LiteralPath $source).Hash -ne (Get-FileHash -LiteralPath $target).Hash){throw 'Archive copy differs.'}
}
foreach($file in $statCandidate.files){
 $source=Join-Path $statRoot ('BrassBirminghamUnity_Data/Managed/'+$file.name)
 if((Get-FileHash -LiteralPath $source).Hash -ne $file.sha256){throw 'Installed candidate differs.'}
 foreach($report in @($statInstalled)+@($statPackaging)){if(@($report.testedFiles|Where-Object {$_.name -eq $file.name -and $_.sha256 -eq $file.sha256}).Count -ne 1){throw 'Installed or packaging evidence differs.'}}
 Copy-StatArchive $source ('Managed/'+$file.name)
}
foreach($name in 'build-manifest','validation','local-ai-qualification','preserved-runtime','visual-checks','statistics-native','tracker-native','presentation-native','link-native','menu-native','tracker-installed','statistics-installed','menu-installed','packaging-results'){
 Copy-StatArchive (Join-Path $statBuild ($name+'.json')) ($name+'.json')
}
Copy-StatArchive (Join-Path $statRoot 'artifacts/patch-installation.json') 'patch-installation.json'
foreach($pin in $statCandidate.sourcePins){$source=Join-Path $statBuild ('runtime/'+$pin.path);if((Get-FileHash -LiteralPath $source).Hash -ne $pin.sha256){throw 'Compiled source snapshot changed.'};Copy-StatArchive $source $pin.path}
foreach($version in @($statCandidate.helperSourceBaseVersion,$statCandidate.baseVersion)|Select-Object -Unique){
 $statBaseline=Join-Path $statRoot ('artifacts/patch-releases/'+$version)
 foreach($name in 'archive-manifest.json','build-manifest.json','validation.json'){Copy-StatArchive (Join-Path $statBaseline $name) ('baseline/'+$version+'/'+$name)}
 foreach($name in 'Assembly-CSharp.dll','BrassPatch.dll','BrassLegacyAI.dll'){Copy-StatArchive (Join-Path $statBaseline ('Managed/'+$name)) ('baseline/'+$version+'/Managed/'+$name)}
}
foreach($name in 'build-statistics-light.ps1','archive-statistics-light-release.ps1','install-combined-patch.ps1','install-patch.ps1','uninstall-patch.ps1','patch-install-common.ps1','stage-archived-patch.ps1'){Copy-StatArchive (Join-Path $statRoot ('scripts/'+$name)) ('scripts/'+$name)}
foreach($folder in 'tests/BrassPatch/InformationUI','tests/BrassPatch/Statistics','tests/BrassPatch/CardTracker','tests/BrassPatch/Presentation','tests/BrassPatch/CityLocator'){
 $source=Join-Path $statRoot $folder
 foreach($file in Get-ChildItem -LiteralPath $source -Recurse -File|Where-Object Extension -in '.cs','.ps1'){Copy-StatArchive $file.FullName ($folder+'/'+[IO.Path]::GetRelativePath($source,$file.FullName))}
}
foreach($name in 'LocalAiUnityTests.cs','AiExecutionResourceTests.cs'){Copy-StatArchive (Join-Path $statRoot ('tests/BrassLocalAI/'+$name)) ('tests/BrassLocalAI/'+$name)}
foreach($path in 'docs/STATISTICS_LIGHT_THEME_1.4.21.md','rules-audit/END_GAME_STATISTICS_LIGHT_THEME_REVISION_PLAN.md','rules-audit/END_GAME_STATISTICS_PLOT_DATA_AUDIT.md'){Copy-StatArchive (Join-Path $statRoot $path) $path}
$statAudit=Join-Path $statRoot 'artifacts/statistics-audit/2026-10-02-plan-r3'
foreach($file in Get-ChildItem -LiteralPath $statAudit -File|Where-Object Extension -in '.json','.md','.cs','.ps1'){Copy-StatArchive $file.FullName ('planning-audit/'+$file.Name)}
$statScreens=Join-Path $statRoot 'artifacts/card-tracker-player/artifacts/patch-build'
foreach($file in Get-ChildItem -LiteralPath $statScreens -File|Where-Object {$_.Name -match '^(statistics|tracker)(-results-entry|-button-placement|-depleted-dots|-card-list|-card-detail)?-1920x(1080|1200).*\.png$' -or $_.Name -match '^statistics-(light|special)-1920x(1080|1200).*\.png$' -or $_.Name -like 'statistics-detail-*.png' -or $_.Name -like '*-card-tracker-native.log'}){Copy-StatArchive $file.FullName ('screenshots/'+$file.Name)}
foreach($name in 'tracker','statistics','menu'){Copy-StatArchive (Join-Path $statRoot ('artifacts/patch-build/'+$name+'-card-tracker-native.log')) ('installed-logs/'+$name+'.log')}
$statInventory=@(Get-ChildItem -LiteralPath $statDestination -Recurse -File|Sort-Object FullName|ForEach-Object{[ordered]@{path=[IO.Path]::GetRelativePath($statDestination,$_.FullName).Replace('\','/');sha256=(Get-FileHash -LiteralPath $_.FullName).Hash}})
[ordered]@{version=$statCandidate.version;createdUtc=[DateTime]::UtcNow.ToString('o');priorArchive=$statCandidate.baseVersion;priorBackup=$statRecord.previousReleaseBackup;files=$statInventory}|ConvertTo-Json -Depth 8|Set-Content -LiteralPath (Join-Path $statDestination 'archive-manifest.json') -Encoding utf8
& (Join-Path $PSScriptRoot 'stage-archived-patch.ps1') -Version $statCandidate.version -VerifyOnly
Write-Output ('Archived and verified Statistics light theme '+$statCandidate.version+': '+$statInventory.Count+' files.')
