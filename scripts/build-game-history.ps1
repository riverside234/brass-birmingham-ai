param([string]$BaseVersion='1.4.22',[string]$Version='1.4.23')
$ErrorActionPreference='Stop'
$historyRoot=Split-Path -Parent $PSScriptRoot
$historyOut=Join-Path $historyRoot 'artifacts/game-history-build'
$historySources=Join-Path $historyOut 'runtime'
$historyBase=Join-Path $historyRoot ('artifacts/patch-releases/'+$BaseVersion)
$historyManifest=Get-Content -LiteralPath (Join-Path $historyBase 'build-manifest.json') -Raw|ConvertFrom-Json
if($historyManifest.version -ne $BaseVersion -or -not (Get-Content -LiteralPath (Join-Path $historyBase 'validation.json') -Raw|ConvertFrom-Json).passed){throw 'Qualified accepted source baseline required.'}
foreach($historyPin in (Get-Content -LiteralPath (Join-Path $historyBase 'archive-manifest.json') -Raw|ConvertFrom-Json).files){
 $historyPath=[IO.Path]::GetFullPath((Join-Path $historyBase $historyPin.path))
 if(-not $historyPath.StartsWith([IO.Path]::GetFullPath($historyBase)+[IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase) -or (Get-FileHash -LiteralPath $historyPath).Hash -ne $historyPin.sha256){throw 'Accepted archive changed.'}
}
New-Item -ItemType Directory -Path $historyOut,$historySources,(Join-Path $historyOut 'Managed') -Force|Out-Null
foreach($historyPin in $historyManifest.sourcePins){
 if($historyPin.path.Contains('..') -or $historyPin.path -notmatch '^(src|configs|docs)/'){throw 'Unexpected source pin.'}
 $historyTarget=Join-Path $historySources $historyPin.path
 New-Item -ItemType Directory -Path (Split-Path -Parent $historyTarget) -Force|Out-Null
 Copy-Item -LiteralPath (Join-Path $historyBase $historyPin.path) -Destination $historyTarget -Force
}
$historyScope=@('src/BrassPatch/State/OfflineMode.cs','src/BrassPatch/Statistics/StatisticsEntryPoints.cs','src/BrassPatch/Statistics/StatisticsPanel.cs','src/BrassPatch/Statistics/StatisticsViewHost.cs')
$historyScope+=@('src/BrassPatch/Statistics/StatisticsPlotViews.cs','src/BrassPatch/Statistics/StatisticsWidgets.cs','src/BrassPatch/Statistics/StatisticsPlotSemantics.cs')
$historyScope+=@(Get-ChildItem -LiteralPath (Join-Path $historyRoot 'src/BrassPatch/History') -Filter '*.cs'|ForEach-Object{'src/BrassPatch/History/'+$_.Name})
foreach($historyPath in $historyScope){
 $historyTarget=Join-Path $historySources $historyPath
 New-Item -ItemType Directory -Path (Split-Path -Parent $historyTarget) -Force|Out-Null
 Copy-Item -LiteralPath (Join-Path $historyRoot $historyPath) -Destination $historyTarget -Force
}
$historyCompiler=Join-Path $historyRoot 'artifacts/patch-tools/microsoft.net.compilers.toolset.4.12.0/tasks/net472/csc.exe'
$historyStock=Join-Path $historyRoot 'artifacts/patch-baseline/Assembly-CSharp.dll'
if((Get-FileHash -LiteralPath $historyStock).Hash -ne $historyManifest.stockSha256){throw 'Stock fingerprint differs.'}
$historyManaged=Join-Path $historyRoot 'BrassBirminghamUnity_Data/Managed'
$historyReferences=@(Get-ChildItem -LiteralPath $historyManaged -Filter '*.dll'|Where-Object {$_.Name -notin 'Assembly-CSharp.dll','BrassPatch.dll','BrassLegacyAI.dll' -and $_.Name -notmatch 'Tests|Regression'}|ForEach-Object{'/reference:"'+$_.FullName+'"'})
$historyInputs=@(Get-ChildItem -LiteralPath (Join-Path $historySources 'src/BrassPatch') -Recurse -Filter '*.cs'|ForEach-Object{'"'+$_.FullName+'"'})
@('/nologo','/target:library','/langversion:7.3','/nostdlib+','/deterministic+','/optimize+',('/out:"'+(Join-Path $historyOut 'Managed/BrassPatch.dll')+'"'))+$historyReferences+@('/reference:"'+$historyStock+'"')+$historyInputs|Set-Content -LiteralPath (Join-Path $historyOut 'helper.rsp') -Encoding utf8
& $historyCompiler ('@'+(Join-Path $historyOut 'helper.rsp'))
if($LASTEXITCODE -ne 0){throw 'History helper compilation failed.'}
$historyCecil=Join-Path $historyRoot 'artifacts/patch-tools/mono.cecil.0.11.6/lib/net40/Mono.Cecil.dll'
$historyHook=Join-Path $historyOut 'HistoryMenuHook.exe'
& $historyCompiler /nologo /target:exe ('/out:'+$historyHook) ('/reference:'+$historyCecil) (Join-Path $historyRoot 'tests/BrassPatch/History/HistoryMenuHook.cs')
if($LASTEXITCODE -ne 0){throw 'History menu-hook compilation failed.'}
Copy-Item -LiteralPath $historyCecil -Destination (Join-Path $historyOut 'Mono.Cecil.dll') -Force
& $historyHook (Join-Path $historyBase 'Managed/Assembly-CSharp.dll') (Join-Path $historyOut 'Managed/BrassPatch.dll') (Join-Path $historyOut 'Managed/Assembly-CSharp.dll') $historyManaged
if($LASTEXITCODE -ne 0){throw 'History menu weaving failed.'}
Copy-Item -LiteralPath (Join-Path $historyBase 'Managed/BrassLegacyAI.dll') -Destination (Join-Path $historyOut 'Managed/BrassLegacyAI.dll') -Force
$historyPins=@(Get-ChildItem -LiteralPath $historySources -Recurse -File|Where-Object Extension -in '.cs','.json','.md'|Sort-Object FullName|ForEach-Object{[ordered]@{path=[IO.Path]::GetRelativePath($historySources,$_.FullName).Replace('\','/');sha256=(Get-FileHash -LiteralPath $_.FullName).Hash}})
[ordered]@{version=$Version;stockSha256=$historyManifest.stockSha256;baseVersion=$BaseVersion;baseFiles=$historyManifest.files;sourceScope=$historyScope;sourcePins=$historyPins;gammaEnabled=$false;searchWorkers=2;historyLimit=10;reportSchema=1;gameHooks=@('Frameworks.View.Menu.MainMenuView::PrepareReferences');files=@(foreach($historyName in 'Assembly-CSharp.dll','BrassPatch.dll','BrassLegacyAI.dll'){[ordered]@{name=$historyName;sha256=(Get-FileHash -LiteralPath (Join-Path $historyOut ('Managed/'+$historyName))).Hash}})}|ConvertTo-Json -Depth 7|Set-Content -LiteralPath (Join-Path $historyOut 'build-manifest.json') -Encoding utf8
Write-Output ('Built '+$Version+' Game History from accepted '+$BaseVersion+'; AI binary preserved.')
