param([string]$BuildDirectory='artifacts/game-history-build')
$ErrorActionPreference='Stop'
$historyRoot=Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot 'patch-install-common.ps1')
$historyBuild=Join-Path $historyRoot $BuildDirectory
$candidate=Read-BrassJson (Join-Path $historyBuild 'build-manifest.json')
$destination=Join-Path $historyRoot ('artifacts/patch-releases/'+$candidate.version)
if(Test-Path -LiteralPath $destination){throw 'Release archive already exists; refusing to replace it.'}
$validation=Read-BrassJson (Join-Path $historyBuild 'validation.json')
$receipt=Read-BrassJson (Join-Path $historyRoot 'artifacts/patch-installation.json')
$installed=Read-BrassJson (Join-Path $historyBuild 'history-installed.json')
$menu=Read-BrassJson (Join-Path $historyBuild 'menu-installed.json')
$statistics=Read-BrassJson (Join-Path $historyBuild 'statistics-installed.json')
$packaging=Read-BrassJson (Join-Path $historyBuild 'packaging-results.json')
Assert-BrassCandidate $candidate $validation $historyBuild
if($candidate.version -ne '1.4.23' -or $receipt.version -ne $candidate.version -or $receipt.status -ne 'installed' -or -not $installed.passed -or -not $menu.passed -or -not $statistics.passed -or -not $packaging.passed){throw 'Installed history/menu/statistics and packaging qualification required.'}
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
 foreach($report in $installed,$menu,$statistics,$packaging){if(@($report.testedFiles|Where-Object {$_.name -eq $file.name -and $_.sha256 -eq $file.sha256}).Count -ne 1){throw 'Installed/packaging evidence belongs to another candidate.'}}
 Copy-HistoryArchive $source ('Managed/'+$file.name)
}
foreach($name in 'build-manifest','validation','local-ai-qualification','preserved-runtime','visual-checks','history-native','menu-native','statistics-native','tracker-native','presentation-native','link-native','history-installed','menu-installed','statistics-installed','packaging-results','statistics-before-plot-fixes'){
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
foreach($name in 'build-game-history.ps1','archive-game-history-release.ps1','install-combined-patch.ps1','patch-install-common.ps1','stage-archived-patch.ps1'){
 Copy-HistoryArchive (Join-Path $historyRoot ('scripts/'+$name)) ('scripts/'+$name)
}
foreach($folder in 'tests/BrassPatch/History','tests/BrassPatch/InformationUI','tests/BrassPatch/Statistics','tests/BrassPatch/CardTracker','tests/BrassPatch/Presentation'){
 foreach($file in Get-ChildItem -LiteralPath (Join-Path $historyRoot $folder) -File|Where-Object Extension -in '.cs','.ps1'){Copy-HistoryArchive $file.FullName ($folder+'/'+$file.Name)}
}
foreach($name in 'LocalAiUnityTests.cs','AiExecutionResourceTests.cs'){Copy-HistoryArchive (Join-Path $historyRoot ('tests/BrassLocalAI/'+$name)) ('tests/BrassLocalAI/'+$name)}
Copy-HistoryArchive (Join-Path $historyRoot 'tests/BrassPatch/CityLocator/UnityCityLocatorRegression.cs') 'tests/BrassPatch/CityLocator/UnityCityLocatorRegression.cs'
foreach($path in 'docs/GAME_HISTORY_1.4.23.md','rules-audit/GAME_HISTORY_STATISTICS_PLAN.md','rules-audit/STATISTICS_PANEL_REVIEW_1.4.23.md'){Copy-HistoryArchive (Join-Path $historyRoot $path) $path}
$screens=Join-Path $historyRoot 'artifacts/card-tracker-player/artifacts/patch-build'
foreach($image in $validation.history.images){Copy-HistoryArchive $image ('screenshots/'+[IO.Path]::GetFileName($image))}
foreach($image in Get-ChildItem -LiteralPath $screens -Filter 'live-plot-1200-*.png'){Copy-HistoryArchive $image.FullName ('screenshots/'+$image.Name)}
Copy-HistoryArchive (Join-Path $screens 'live-literal-names-1200.png') 'screenshots/live-literal-names-1200.png'
foreach($name in 'statistics-map-before-label-fix.png','statistics-seats-before-key-fix.png'){Copy-HistoryArchive (Join-Path $historyBuild $name) ('screenshots/'+$name)}
foreach($name in 'statistics-1920x1200-0-larger.png','statistics-1920x1080-0-larger.png','tracker-1920x1200-overlay.png','tracker-1920x1080-overlay.png'){
 $source=Join-Path $screens $name;if(Test-Path -LiteralPath $source){Copy-HistoryArchive $source ('screenshots/'+$name)}
}
foreach($name in 'history','menu','statistics','tracker','presentation','link'){Copy-HistoryArchive (Join-Path $screens ($name+'-card-tracker-native.log')) ('logs/'+$name+'-native.log')}
foreach($name in 'history','menu','statistics'){Copy-HistoryArchive (Join-Path $historyRoot ('artifacts/patch-build/'+$name+'-card-tracker-native.log')) ('logs/'+$name+'-installed.log')}
$inventory=@(Get-ChildItem -LiteralPath $destination -Recurse -File|Sort-Object FullName|ForEach-Object{[ordered]@{path=[IO.Path]::GetRelativePath($destination,$_.FullName).Replace('\','/');sha256=(Get-FileHash -LiteralPath $_.FullName).Hash}})
[ordered]@{version=$candidate.version;createdUtc=[DateTime]::UtcNow.ToString('o');priorArchive=$candidate.baseVersion;priorBackup=$receipt.previousReleaseBackup;files=$inventory}|ConvertTo-Json -Depth 8|Set-Content -LiteralPath (Join-Path $destination 'archive-manifest.json') -Encoding utf8
& (Join-Path $PSScriptRoot 'stage-archived-patch.ps1') -Version $candidate.version -VerifyOnly
Write-Output ('Archived and verified Game History '+$candidate.version+': '+$inventory.Count+' files.')
