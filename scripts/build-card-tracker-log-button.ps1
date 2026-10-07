param([string]$BaseVersion='1.4.26',[string]$Version='1.4.27')
$ErrorActionPreference='Stop'
$aiUiRoot=Split-Path -Parent $PSScriptRoot
$aiUiOut=Join-Path $aiUiRoot 'artifacts/card-tracker-log-button-build'
$aiUiSources=Join-Path $aiUiOut 'runtime'
$aiUiBase=Join-Path $aiUiRoot ('artifacts/patch-releases/'+$BaseVersion)
function Read-AiUiJson([string]$path){Get-Content -LiteralPath $path -Raw|ConvertFrom-Json}
$aiUiManifest=Read-AiUiJson (Join-Path $aiUiBase 'build-manifest.json')
if($aiUiManifest.version -ne $BaseVersion -or -not (Read-AiUiJson (Join-Path $aiUiBase 'validation.json')).passed){throw 'Qualified source baseline required.'}
foreach($pin in (Read-AiUiJson (Join-Path $aiUiBase 'archive-manifest.json')).files){
 $path=[IO.Path]::GetFullPath((Join-Path $aiUiBase $pin.path))
 if(-not $path.StartsWith([IO.Path]::GetFullPath($aiUiBase)+[IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase) -or (Get-FileHash -LiteralPath $path).Hash -ne $pin.sha256){throw 'Accepted archive changed.'}
}
New-Item -ItemType Directory -Path $aiUiOut,$aiUiSources,(Join-Path $aiUiOut 'Managed') -Force|Out-Null
foreach($pin in $aiUiManifest.sourcePins){
 if($pin.path.Contains('..') -or $pin.path -notmatch '^(src|configs|docs)/'){throw 'Unexpected pinned path.'}
 $source=Join-Path $aiUiBase $pin.path;$target=Join-Path $aiUiSources $pin.path
 if((Get-FileHash -LiteralPath $source).Hash -ne $pin.sha256){throw 'Source snapshot changed.'}
 New-Item -ItemType Directory -Path (Split-Path -Parent $target) -Force|Out-Null
 Copy-Item -LiteralPath $source -Destination $target -Force
}
$aiUiScope=@('src/BrassPatch/CardTracker/CardTrackerOverlay.cs')
foreach($path in $aiUiScope){
 $target=Join-Path $aiUiSources $path;New-Item -ItemType Directory -Path (Split-Path -Parent $target) -Force|Out-Null
 Copy-Item -LiteralPath (Join-Path $aiUiRoot $path) -Destination $target -Force
}
$aiUiCompiler=Join-Path $aiUiRoot 'artifacts/patch-tools/microsoft.net.compilers.toolset.4.12.0/tasks/net472/csc.exe'
$aiUiStock=Join-Path $aiUiRoot 'artifacts/patch-baseline/Assembly-CSharp.dll'
if((Get-FileHash -LiteralPath $aiUiStock).Hash -ne $aiUiManifest.stockSha256){throw 'Stock fingerprint changed.'}
$aiUiManaged=Join-Path $aiUiRoot 'BrassBirminghamUnity_Data/Managed'
$aiUiReferences=@(Get-ChildItem -LiteralPath $aiUiManaged -Filter '*.dll'|Where-Object {$_.Name -notin 'Assembly-CSharp.dll','BrassPatch.dll','BrassLegacyAI.dll' -and $_.Name -notmatch 'Tests|Regression'}|ForEach-Object{'/reference:"'+$_.FullName+'"'})
$aiUiInputs=@(Get-ChildItem -LiteralPath (Join-Path $aiUiSources 'src/BrassPatch') -Recurse -Filter '*.cs'|ForEach-Object{'"'+$_.FullName+'"'})
@('/nologo','/target:library','/langversion:7.3','/nostdlib+','/deterministic+','/optimize+',('/out:"'+(Join-Path $aiUiOut 'Managed/BrassPatch.dll')+'"'))+$aiUiReferences+@('/reference:"'+$aiUiStock+'"')+$aiUiInputs|Set-Content -LiteralPath (Join-Path $aiUiOut 'helper.rsp') -Encoding utf8
& $aiUiCompiler ('@'+(Join-Path $aiUiOut 'helper.rsp'))
if($LASTEXITCODE -ne 0){throw 'Card Tracker button compilation failed.'}
foreach($name in 'Assembly-CSharp.dll','BrassLegacyAI.dll'){
 Copy-Item -LiteralPath (Join-Path $aiUiBase ('Managed/'+$name)) -Destination (Join-Path $aiUiOut ('Managed/'+$name)) -Force
}
$aiUiPins=@(Get-ChildItem -LiteralPath $aiUiSources -Recurse -File|Where-Object Extension -in '.cs','.json','.md'|Sort-Object FullName|ForEach-Object{[ordered]@{path=[IO.Path]::GetRelativePath($aiUiSources,$_.FullName).Replace('\','/');sha256=(Get-FileHash -LiteralPath $_.FullName).Hash}})
[ordered]@{version=$Version;stockSha256=$aiUiManifest.stockSha256;baseVersion=$BaseVersion;baseFiles=$aiUiManifest.files;sourceScope=$aiUiScope;sourcePins=$aiUiPins;gammaEnabled=$false;searchWorkers=2;playerSelectionPages=@('Human','V4','Gamma');draftSelection=$true;trackerTheme='light';trackerDotColor='red';trackerButtonAnchor='below-game-log';trackerButtonSize=@(152,36);trackerButtonOpacity=.76;trackerYieldsToGameLog=$true;historyLimit=10;reportSchema=1;efficiencyViews=@('Returns','Conversion','Transition & networks');livePresentationSuspension=$true;overviewMedals=$true;files=@(foreach($name in 'Assembly-CSharp.dll','BrassPatch.dll','BrassLegacyAI.dll'){[ordered]@{name=$name;sha256=(Get-FileHash -LiteralPath (Join-Path $aiUiOut ('Managed/'+$name))).Hash}})}|ConvertTo-Json -Depth 7|Set-Content -LiteralPath (Join-Path $aiUiOut 'build-manifest.json') -Encoding utf8
Write-Output ('Built '+$Version+' Card Tracker button from accepted '+$BaseVersion+'; game assembly and frozen profiles preserved.')
