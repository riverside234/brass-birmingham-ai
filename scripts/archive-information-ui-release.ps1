$ErrorActionPreference='Stop'
$uiRoot=Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot 'patch-install-common.ps1')
$uiBuild=Join-Path $uiRoot 'artifacts/information-ui-build'
$uiCandidate=Read-BrassJson (Join-Path $uiBuild 'build-manifest.json')
$uiDestination=Join-Path $uiRoot ('artifacts/patch-releases/'+$uiCandidate.version)
if(Test-Path -LiteralPath $uiDestination){throw 'Release archive exists; refusing to replace it.'}
$uiValidation=Read-BrassJson (Join-Path $uiBuild 'validation.json')
$uiRecord=Read-BrassJson (Join-Path $uiRoot 'artifacts/patch-installation.json')
$uiInstalled=@('tracker','statistics','menu')|ForEach-Object {Read-BrassJson (Join-Path $uiBuild ($_+'-installed.json'))}
$uiPackaging=Read-BrassJson (Join-Path $uiBuild 'packaging-results.json')
Assert-BrassCandidate $uiCandidate $uiValidation $uiBuild
if($uiCandidate.version -ne '1.4.19' -or $uiRecord.version -ne $uiCandidate.version -or $uiRecord.status -ne 'installed' -or @($uiInstalled|Where-Object {-not $_.passed}).Count -or -not $uiPackaging.passed){throw 'Installed UI, menu and packaging qualification are required.'}
function Copy-UiArchive([string]$source,[string]$relative){
 $target=[IO.Path]::GetFullPath((Join-Path $uiDestination $relative))
 if(-not $target.StartsWith([IO.Path]::GetFullPath($uiDestination)+[IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase)){throw 'Archive path escapes its directory.'}
 New-Item -ItemType Directory -Path (Split-Path -Parent $target) -Force|Out-Null
 Copy-Item -LiteralPath $source -Destination $target
 if((Get-FileHash -LiteralPath $source).Hash -ne (Get-FileHash -LiteralPath $target).Hash){throw 'Archive copy differs.'}
}
foreach($file in $uiCandidate.files){
 $source=Join-Path $uiRoot ('BrassBirminghamUnity_Data/Managed/'+$file.name)
 if((Get-FileHash -LiteralPath $source).Hash -ne $file.sha256){throw 'Installed candidate differs.'}
 foreach($report in @($uiInstalled)+@($uiPackaging)){if(@($report.testedFiles|Where-Object {$_.name -eq $file.name -and $_.sha256 -eq $file.sha256}).Count -ne 1){throw 'Installed or packaging evidence differs.'}}
 Copy-UiArchive $source ('Managed/'+$file.name)
}
foreach($name in 'build-manifest','validation','local-ai-qualification','preserved-runtime','visual-checks','statistics-native','tracker-native','presentation-native','link-native','menu-native','tracker-installed','statistics-installed','menu-installed','packaging-results'){
 Copy-UiArchive (Join-Path $uiBuild ($name+'.json')) ($name+'.json')
}
Copy-UiArchive (Join-Path $uiRoot 'artifacts/patch-installation.json') 'patch-installation.json'
foreach($pin in $uiCandidate.sourcePins){$source=Join-Path $uiBuild ('runtime/'+$pin.path);if((Get-FileHash -LiteralPath $source).Hash -ne $pin.sha256){throw 'Compiled source snapshot changed.'};Copy-UiArchive $source $pin.path}
foreach($version in @($uiCandidate.helperSourceBaseVersion,$uiCandidate.baseVersion)|Select-Object -Unique){
 $uiBaseline=Join-Path $uiRoot ('artifacts/patch-releases/'+$version)
 foreach($name in 'archive-manifest.json','build-manifest.json','validation.json'){Copy-UiArchive (Join-Path $uiBaseline $name) ('baseline/'+$version+'/'+$name)}
 foreach($file in Get-ChildItem -LiteralPath (Join-Path $uiBaseline 'Managed') -File|Where-Object Name -in 'Assembly-CSharp.dll','BrassPatch.dll','BrassLegacyAI.dll'){Copy-UiArchive $file.FullName ('baseline/'+$version+'/Managed/'+$file.Name)}
}
foreach($name in 'build-information-ui.ps1','archive-information-ui-release.ps1','install-combined-patch.ps1','install-patch.ps1','uninstall-patch.ps1','patch-install-common.ps1','stage-archived-patch.ps1'){Copy-UiArchive (Join-Path $uiRoot ('scripts/'+$name)) ('scripts/'+$name)}
foreach($folder in 'tests/BrassPatch/InformationUI','tests/BrassPatch/Statistics','tests/BrassPatch/CardTracker','tests/BrassPatch/Presentation','tests/BrassPatch/CityLocator'){
 $source=Join-Path $uiRoot $folder
 foreach($file in Get-ChildItem -LiteralPath $source -Recurse -File|Where-Object Extension -in '.cs','.ps1'){Copy-UiArchive $file.FullName ($folder+'/'+[IO.Path]::GetRelativePath($source,$file.FullName))}
}
foreach($name in 'LocalAiUnityTests.cs','AiExecutionResourceTests.cs'){Copy-UiArchive (Join-Path $uiRoot ('tests/BrassLocalAI/'+$name)) ('tests/BrassLocalAI/'+$name)}
Copy-UiArchive (Join-Path $uiRoot 'docs/INFORMATION_PANELS_1.4.19.md') 'docs/INFORMATION_PANELS_1.4.19.md'
$uiScreens=Join-Path $uiRoot 'artifacts/card-tracker-player/artifacts/patch-build'
foreach($file in Get-ChildItem -LiteralPath $uiScreens -File|Where-Object {$_.Name -match '^(statistics|tracker)(-results-entry|-button-placement|-depleted-dots|-card-list|-card-detail)?-1920x(1080|1200).*\.png$' -or $_.Name -like 'statistics-detail-*.png' -or $_.Name -like '*-card-tracker-native.log'}){Copy-UiArchive $file.FullName ('screenshots/'+$file.Name)}
foreach($name in 'tracker','statistics','menu'){Copy-UiArchive (Join-Path $uiRoot ('artifacts/patch-build/'+$name+'-card-tracker-native.log')) ('installed-logs/'+$name+'.log')}
$uiInventory=@(Get-ChildItem -LiteralPath $uiDestination -Recurse -File|Sort-Object FullName|ForEach-Object{[ordered]@{path=[IO.Path]::GetRelativePath($uiDestination,$_.FullName).Replace('\','/');sha256=(Get-FileHash -LiteralPath $_.FullName).Hash}})
[ordered]@{version=$uiCandidate.version;createdUtc=[DateTime]::UtcNow.ToString('o');priorArchive=$uiCandidate.baseVersion;priorBackup=$uiRecord.previousReleaseBackup;files=$uiInventory}|ConvertTo-Json -Depth 8|Set-Content -LiteralPath (Join-Path $uiDestination 'archive-manifest.json') -Encoding utf8
& (Join-Path $PSScriptRoot 'stage-archived-patch.ps1') -Version $uiCandidate.version -VerifyOnly
Write-Output ('Archived and verified information UI '+$uiCandidate.version+': '+$uiInventory.Count+' files.')
