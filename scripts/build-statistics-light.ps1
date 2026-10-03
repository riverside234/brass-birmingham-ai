param([string]$BaseVersion='1.4.20',[string]$Version='1.4.21')
$ErrorActionPreference='Stop'
$statRoot=Split-Path -Parent $PSScriptRoot
$statOut=Join-Path $statRoot 'artifacts/statistics-light-build'
$statSources=Join-Path $statOut 'runtime'
$statBase=Join-Path $statRoot ('artifacts/patch-releases/'+$BaseVersion)
$statHelper=Join-Path $statRoot 'artifacts/patch-releases/1.4.19'
function Read-StatJson([string]$path){Get-Content -LiteralPath $path -Raw|ConvertFrom-Json}
foreach($archive in @($statHelper,$statBase)){
 foreach($pin in (Read-StatJson (Join-Path $archive 'archive-manifest.json')).files){
  $path=[IO.Path]::GetFullPath((Join-Path $archive $pin.path))
  if(-not $path.StartsWith([IO.Path]::GetFullPath($archive)+[IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase) -or (Get-FileHash -LiteralPath $path).Hash -ne $pin.sha256){throw 'Accepted archive changed.'}
 }
}
$statManifest=Read-StatJson (Join-Path $statBase 'build-manifest.json')
if($statManifest.version -ne $BaseVersion -or -not (Read-StatJson (Join-Path $statBase 'validation.json')).passed){throw 'Qualified current baseline required.'}
if((Get-FileHash -LiteralPath (Join-Path $statHelper 'Managed/BrassPatch.dll')).Hash -ne ($statManifest.files|Where-Object name -eq 'BrassPatch.dll').sha256){throw 'Helper sources do not match the current accepted helper.'}
New-Item -ItemType Directory -Path $statOut,$statSources,(Join-Path $statOut 'Managed') -Force|Out-Null
foreach($archive in @($statHelper,$statBase)){
 foreach($pin in (Read-StatJson (Join-Path $archive 'build-manifest.json')).sourcePins){
  if($pin.path.Contains('..') -or $pin.path -notmatch '^(src|configs|docs)/'){throw 'Unexpected pinned path.'}
  $source=Join-Path $archive $pin.path;$target=Join-Path $statSources $pin.path
  if((Get-FileHash -LiteralPath $source).Hash -ne $pin.sha256){throw 'Source snapshot changed.'}
  New-Item -ItemType Directory -Path (Split-Path -Parent $target) -Force|Out-Null
  Copy-Item -LiteralPath $source -Destination $target -Force
 }
}
$statRetired='src/BrassPatch/Statistics/StatisticsInsights.cs'
$statRetiredTarget=[IO.Path]::GetFullPath((Join-Path $statSources $statRetired))
if(-not $statRetiredTarget.StartsWith([IO.Path]::GetFullPath($statSources)+[IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase)){throw 'Retired source escapes staged workspace.'}
if(Test-Path -LiteralPath $statRetiredTarget){Remove-Item -LiteralPath $statRetiredTarget}
$statScope=@(Get-ChildItem -LiteralPath (Join-Path $statRoot 'src/BrassPatch/Statistics') -File|Where-Object Extension -in '.cs','.json'|ForEach-Object{[IO.Path]::GetRelativePath($statRoot,$_.FullName).Replace('\','/')})
foreach($path in $statScope){Copy-Item -LiteralPath (Join-Path $statRoot $path) -Destination (Join-Path $statSources $path) -Force}
$statCompiler=Join-Path $statRoot 'artifacts/patch-tools/microsoft.net.compilers.toolset.4.12.0/tasks/net472/csc.exe'
$statStock=Join-Path $statRoot 'artifacts/patch-baseline/Assembly-CSharp.dll'
if((Get-FileHash -LiteralPath $statStock).Hash -ne $statManifest.stockSha256){throw 'Stock fingerprint changed.'}
$statReferences=@(Get-ChildItem -LiteralPath (Join-Path $statRoot 'BrassBirminghamUnity_Data/Managed') -Filter '*.dll'|Where-Object {$_.Name -notin 'Assembly-CSharp.dll','BrassPatch.dll','BrassLegacyAI.dll' -and $_.Name -notmatch 'Tests|Regression'}|ForEach-Object{'/reference:"'+$_.FullName+'"'})
$statReferences+='/reference:"'+$statStock+'"'
$statInputs=@(Get-ChildItem -LiteralPath (Join-Path $statSources 'src/BrassPatch') -Recurse -Filter '*.cs'|ForEach-Object{'"'+$_.FullName+'"'})
@('/nologo','/target:library','/langversion:7.3','/nostdlib+','/deterministic+','/optimize+',('/out:"'+(Join-Path $statOut 'Managed/BrassPatch.dll')+'"'))+$statReferences+$statInputs|Set-Content -LiteralPath (Join-Path $statOut 'helper.rsp') -Encoding utf8
& $statCompiler ('@'+(Join-Path $statOut 'helper.rsp'))
if($LASTEXITCODE -ne 0){throw 'Statistics helper compilation failed.'}
foreach($name in 'Assembly-CSharp.dll','BrassLegacyAI.dll'){Copy-Item -LiteralPath (Join-Path $statBase ('Managed/'+$name)) -Destination (Join-Path $statOut ('Managed/'+$name)) -Force}
$statPins=@(Get-ChildItem -LiteralPath $statSources -Recurse -File|Where-Object Extension -in '.cs','.json','.md'|ForEach-Object{[ordered]@{path=[IO.Path]::GetRelativePath($statSources,$_.FullName).Replace('\','/');sha256=(Get-FileHash -LiteralPath $_.FullName).Hash}})
[ordered]@{version=$Version;stockSha256=$statManifest.stockSha256;baseVersion=$BaseVersion;helperSourceBaseVersion='1.4.19';baseFiles=$statManifest.files;sourceScope=$statScope;retiredSources=@($statRetired);sourcePins=$statPins;files=@(Get-ChildItem -LiteralPath (Join-Path $statOut 'Managed') -File|Where-Object Name -in 'Assembly-CSharp.dll','BrassPatch.dll','BrassLegacyAI.dll'|ForEach-Object{[ordered]@{name=$_.Name;sha256=(Get-FileHash -LiteralPath $_.FullName).Hash}})}|ConvertTo-Json -Depth 7|Set-Content -LiteralPath (Join-Path $statOut 'build-manifest.json') -Encoding utf8
Write-Output ('Built '+$Version+' from '+$BaseVersion+'; current game and AI binaries preserved.')
