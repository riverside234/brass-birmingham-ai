param([string]$BuildDirectory='artifacts/ai-selection-ui-build')
$ErrorActionPreference='Stop'
$aiUiRoot=Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot 'patch-install-common.ps1')
$aiUiBuild=Join-Path $aiUiRoot $BuildDirectory
$aiUiCandidate=Read-BrassJson (Join-Path $aiUiBuild 'build-manifest.json')
$aiUiDestination=Join-Path $aiUiRoot ('artifacts/patch-releases/'+$aiUiCandidate.version)
if(Test-Path -LiteralPath $aiUiDestination){throw 'Release archive already exists; refusing to replace it.'}
$aiUiValidation=Read-BrassJson (Join-Path $aiUiBuild 'validation.json')
$aiUiRecord=Read-BrassJson (Join-Path $aiUiRoot 'artifacts/patch-installation.json')
$aiUiInstalled=Read-BrassJson (Join-Path $aiUiBuild 'menu-installed.json')
$aiUiPackaging=Read-BrassJson (Join-Path $aiUiBuild 'packaging-results.json')
Assert-BrassCandidate $aiUiCandidate $aiUiValidation $aiUiBuild
if($aiUiCandidate.version -ne '1.4.22' -or $aiUiRecord.version -ne $aiUiCandidate.version -or $aiUiRecord.status -ne 'installed' -or -not $aiUiInstalled.passed -or -not $aiUiPackaging.passed){throw 'Installed menu and packaging qualification are required.'}
function Copy-AiUiArchive([string]$source,[string]$relative){
 $target=[IO.Path]::GetFullPath((Join-Path $aiUiDestination $relative))
 if(-not $target.StartsWith([IO.Path]::GetFullPath($aiUiDestination)+[IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase)){throw 'Archive path escapes its directory.'}
 New-Item -ItemType Directory -Path (Split-Path -Parent $target) -Force|Out-Null
 Copy-Item -LiteralPath $source -Destination $target
 if((Get-FileHash -LiteralPath $source).Hash -ne (Get-FileHash -LiteralPath $target).Hash){throw 'Archive copy differs.'}
}
foreach($file in $aiUiCandidate.files){
 $source=Join-Path $aiUiRoot ('BrassBirminghamUnity_Data/Managed/'+$file.name)
 if((Get-FileHash -LiteralPath $source).Hash -ne $file.sha256){throw 'Installed candidate differs.'}
 foreach($report in $aiUiInstalled,$aiUiPackaging){if(@($report.testedFiles|Where-Object {$_.name -eq $file.name -and $_.sha256 -eq $file.sha256}).Count -ne 1){throw 'Installed/packaging evidence belongs to another candidate.'}}
 Copy-AiUiArchive $source ('Managed/'+$file.name)
}
foreach($name in 'build-manifest','validation','local-ai-qualification','preserved-runtime','visual-checks','menu-native','statistics-native','tracker-native','menu-installed','packaging-results'){
 Copy-AiUiArchive (Join-Path $aiUiBuild ($name+'.json')) ($name+'.json')
}
Copy-AiUiArchive (Join-Path $aiUiRoot 'artifacts/patch-installation.json') 'patch-installation.json'
foreach($pin in $aiUiCandidate.sourcePins){
 $source=Join-Path $aiUiBuild ('runtime/'+$pin.path)
 if((Get-FileHash -LiteralPath $source).Hash -ne $pin.sha256){throw 'Compiled source snapshot changed.'}
 Copy-AiUiArchive $source $pin.path
}
$aiUiBaseline=Join-Path $aiUiRoot ('artifacts/patch-releases/'+$aiUiCandidate.baseVersion)
foreach($name in 'archive-manifest.json','build-manifest.json','validation.json'){Copy-AiUiArchive (Join-Path $aiUiBaseline $name) ('baseline/'+$aiUiCandidate.baseVersion+'/'+$name)}
foreach($name in 'Assembly-CSharp.dll','BrassPatch.dll','BrassLegacyAI.dll'){Copy-AiUiArchive (Join-Path $aiUiBaseline ('Managed/'+$name)) ('baseline/'+$aiUiCandidate.baseVersion+'/Managed/'+$name)}
foreach($name in 'build-ai-selection-ui.ps1','archive-ai-selection-ui-release.ps1','install-combined-patch.ps1','patch-install-common.ps1','stage-archived-patch.ps1'){
 Copy-AiUiArchive (Join-Path $aiUiRoot ('scripts/'+$name)) ('scripts/'+$name)
}
foreach($name in 'LocalAiUnityTests.cs','AiExecutionResourceTests.cs','AiSelectionPreservedRuntimeChecks.cs','qualify-selection-ui-release.ps1'){
 Copy-AiUiArchive (Join-Path $aiUiRoot ('tests/BrassLocalAI/'+$name)) ('tests/BrassLocalAI/'+$name)
}
foreach($folder in 'tests/BrassPatch/InformationUI','tests/BrassPatch/Statistics','tests/BrassPatch/CardTracker'){
 $source=Join-Path $aiUiRoot $folder
 foreach($file in Get-ChildItem -LiteralPath $source -File|Where-Object Extension -in '.cs','.ps1'){
  Copy-AiUiArchive $file.FullName ($folder+'/'+$file.Name)
 }
}
foreach($path in 'docs/AI_SELECTION_PANEL_1.4.22.md','rules-audit/AI_SELECTION_PANEL_V4_GAMMA_PLAN.md'){Copy-AiUiArchive (Join-Path $aiUiRoot $path) $path}
$aiUiScreens=Join-Path $aiUiRoot 'artifacts/card-tracker-player/artifacts/patch-build'
foreach($name in $aiUiValidation.menu.images){Copy-AiUiArchive $name ('screenshots/'+[IO.Path]::GetFileName($name))}
foreach($name in 'statistics-1920x1200-0-larger.png','statistics-1920x1080-0-larger.png','tracker-1920x1200-overlay.png','tracker-1920x1080-overlay.png'){
 $source=Join-Path $aiUiScreens $name;if(Test-Path -LiteralPath $source){Copy-AiUiArchive $source ('screenshots/'+$name)}
}
foreach($name in 'menu','statistics','tracker'){Copy-AiUiArchive (Join-Path $aiUiScreens ($name+'-card-tracker-native.log')) ('logs/'+$name+'-native.log')}
Copy-AiUiArchive (Join-Path $aiUiRoot 'artifacts/patch-build/menu-card-tracker-native.log') 'logs/menu-installed.log'
$aiUiInventory=@(Get-ChildItem -LiteralPath $aiUiDestination -Recurse -File|Sort-Object FullName|ForEach-Object{[ordered]@{path=[IO.Path]::GetRelativePath($aiUiDestination,$_.FullName).Replace('\','/');sha256=(Get-FileHash -LiteralPath $_.FullName).Hash}})
[ordered]@{version=$aiUiCandidate.version;createdUtc=[DateTime]::UtcNow.ToString('o');priorArchive=$aiUiCandidate.baseVersion;priorBackup=$aiUiRecord.previousReleaseBackup;files=$aiUiInventory}|ConvertTo-Json -Depth 8|Set-Content -LiteralPath (Join-Path $aiUiDestination 'archive-manifest.json') -Encoding utf8
& (Join-Path $PSScriptRoot 'stage-archived-patch.ps1') -Version $aiUiCandidate.version -VerifyOnly
Write-Output ('Archived and verified AI selection UI '+$aiUiCandidate.version+': '+$aiUiInventory.Count+' files.')
