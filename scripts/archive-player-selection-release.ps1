param([string]$BuildDirectory='artifacts/player-selection-build')
$ErrorActionPreference='Stop'
$historyRoot=Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot 'patch-install-common.ps1')
$historyBuild=Join-Path $historyRoot $BuildDirectory
$candidate=Read-BrassJson (Join-Path $historyBuild 'build-manifest.json')
$destination=Join-Path $historyRoot ('artifacts/patch-releases/'+$candidate.version)
if(Test-Path -LiteralPath $destination){throw 'Release archive already exists; refusing to replace it.'}
$validation=Read-BrassJson (Join-Path $historyBuild 'validation.json')
$receipt=Read-BrassJson (Join-Path $historyRoot 'artifacts/patch-installation.json')
$menu=Read-BrassJson (Join-Path $historyBuild 'menu-installed.json')
$packaging=Read-BrassJson (Join-Path $historyBuild 'packaging-results.json')
Assert-BrassCandidate $candidate $validation $historyBuild
if($candidate.version -ne '1.4.24' -or $receipt.version -ne $candidate.version -or $receipt.status -ne 'installed' -or -not $menu.passed -or -not $packaging.passed){throw 'Installed menu and packaging qualification required.'}
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
 foreach($report in $menu,$packaging){if(@($report.testedFiles|Where-Object {$_.name -eq $file.name -and $_.sha256 -eq $file.sha256}).Count -ne 1){throw 'Installed/packaging evidence belongs to another candidate.'}}
 Copy-HistoryArchive $source ('Managed/'+$file.name)
}
foreach($name in 'build-manifest','validation','local-ai-qualification','preserved-runtime','visual-checks','history-native','menu-native','statistics-native','tracker-native','menu-installed','packaging-results'){
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
foreach($name in 'build-player-selection.ps1','archive-player-selection-release.ps1','install-combined-patch.ps1','patch-install-common.ps1','stage-archived-patch.ps1'){
 Copy-HistoryArchive (Join-Path $historyRoot ('scripts/'+$name)) ('scripts/'+$name)
}
foreach($folder in 'tests/BrassPatch/History','tests/BrassPatch/InformationUI','tests/BrassPatch/Statistics','tests/BrassPatch/CardTracker','tests/BrassPatch/Presentation'){
 foreach($file in Get-ChildItem -LiteralPath (Join-Path $historyRoot $folder) -File|Where-Object Extension -in '.cs','.ps1'){Copy-HistoryArchive $file.FullName ($folder+'/'+$file.Name)}
}
foreach($name in 'LocalAiUnityTests.cs','AiExecutionResourceTests.cs','PlayerSelectionPreservedRuntimeChecks.cs','qualify-player-selection-release.ps1'){Copy-HistoryArchive (Join-Path $historyRoot ('tests/BrassLocalAI/'+$name)) ('tests/BrassLocalAI/'+$name)}
Copy-HistoryArchive (Join-Path $historyRoot 'tests/BrassPatch/CityLocator/UnityCityLocatorRegression.cs') 'tests/BrassPatch/CityLocator/UnityCityLocatorRegression.cs'
foreach($path in 'docs/PLAYER_SELECTION_1.4.24.md','rules-audit/PLAYER_SELECTION_PANEL_REVISION_PLAN.md'){Copy-HistoryArchive (Join-Path $historyRoot $path) $path}
$screens=Join-Path $historyRoot 'artifacts/card-tracker-player/artifacts/patch-build'
foreach($image in $validation.menu.images){Copy-HistoryArchive $image ('screenshots/'+[IO.Path]::GetFileName($image))}
foreach($name in 'statistics-1920x1200-0-larger.png','tracker-1920x1200-overlay.png'){
 $source=Join-Path $screens $name;if(Test-Path -LiteralPath $source){Copy-HistoryArchive $source ('screenshots/'+$name)}
}
foreach($name in 'history','menu','statistics','tracker'){Copy-HistoryArchive (Join-Path $screens ($name+'-card-tracker-native.log')) ('logs/'+$name+'-native.log')}
Copy-HistoryArchive (Join-Path $historyRoot 'artifacts/patch-build/menu-card-tracker-native.log') 'logs/menu-installed.log'
$inventory=@(Get-ChildItem -LiteralPath $destination -Recurse -File|Sort-Object FullName|ForEach-Object{[ordered]@{path=[IO.Path]::GetRelativePath($destination,$_.FullName).Replace('\','/');sha256=(Get-FileHash -LiteralPath $_.FullName).Hash}})
[ordered]@{version=$candidate.version;createdUtc=[DateTime]::UtcNow.ToString('o');priorArchive=$candidate.baseVersion;priorBackup=$receipt.previousReleaseBackup;files=$inventory}|ConvertTo-Json -Depth 8|Set-Content -LiteralPath (Join-Path $destination 'archive-manifest.json') -Encoding utf8
& (Join-Path $PSScriptRoot 'stage-archived-patch.ps1') -Version $candidate.version -VerifyOnly
Write-Output ('Archived and verified player selection '+$candidate.version+': '+$inventory.Count+' files.')
