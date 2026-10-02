param([string]$BaseVersion='1.4.18',[string]$Version='1.4.19')
$ErrorActionPreference='Stop'
$uiRoot=Split-Path -Parent $PSScriptRoot
$uiOut=Join-Path $uiRoot 'artifacts/information-ui-build'
$uiSources=Join-Path $uiOut 'runtime'
$uiArchive=Join-Path $uiRoot ('artifacts/patch-releases/'+$BaseVersion)
$uiHelperArchive=Join-Path $uiRoot 'artifacts/patch-releases/1.4.17'
foreach($archive in @($uiHelperArchive,$uiArchive)|Select-Object -Unique){
 $inventory=Get-Content -LiteralPath (Join-Path $archive 'archive-manifest.json') -Raw|ConvertFrom-Json
 foreach($pin in $inventory.files){
  $source=[IO.Path]::GetFullPath((Join-Path $archive $pin.path))
  if(-not $source.StartsWith([IO.Path]::GetFullPath($archive)+[IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase)){throw 'Archive path escapes its directory.'}
  if((Get-FileHash -LiteralPath $source).Hash -ne $pin.sha256){throw 'Accepted archive changed.'}
 }
}
$uiBase=Get-Content -LiteralPath (Join-Path $uiArchive 'build-manifest.json') -Raw|ConvertFrom-Json
$uiValidation=Get-Content -LiteralPath (Join-Path $uiArchive 'validation.json') -Raw|ConvertFrom-Json
if($uiBase.version -ne $BaseVersion -or -not $uiValidation.passed){throw 'Qualified baseline required.'}
if((Get-FileHash -LiteralPath (Join-Path $uiHelperArchive 'Managed/BrassPatch.dll')).Hash -ne ($uiBase.files|Where-Object name -eq 'BrassPatch.dll').sha256){throw 'Helper source baseline must match the accepted helper binary.'}
New-Item -ItemType Directory -Path $uiOut,$uiSources,(Join-Path $uiOut 'Managed') -Force|Out-Null
foreach($archive in @($uiHelperArchive,$uiArchive)|Select-Object -Unique){
$sourceManifest=Get-Content -LiteralPath (Join-Path $archive 'build-manifest.json') -Raw|ConvertFrom-Json
foreach($pin in $sourceManifest.sourcePins){
 if(($pin.path -notlike 'src/*' -and $pin.path -notlike 'configs/*' -and $pin.path -notlike 'docs/*') -or $pin.path.Contains('..')){throw 'Unexpected source path.'}
 $source=Join-Path $archive $pin.path
 if((Get-FileHash -LiteralPath $source).Hash -ne $pin.sha256){throw 'Pinned source mismatch.'}
 $target=Join-Path $uiSources $pin.path
 New-Item -ItemType Directory -Path (Split-Path -Parent $target) -Force|Out-Null
 Copy-Item -LiteralPath $source -Destination $target -Force
}
}
$uiOverlay=@('src/BrassPatch/CardTracker/CardTrackerOverlay.cs','src/BrassPatch/Statistics/StatisticsEntryPoints.cs','src/BrassPatch/Statistics/StatisticsEntrySurface.cs','src/BrassPatch/Statistics/StatisticsPanel.cs')
foreach($path in $uiOverlay){Copy-Item -LiteralPath (Join-Path $uiRoot $path) -Destination (Join-Path $uiSources $path) -Force}
$uiCompiler=Join-Path $uiRoot 'artifacts/patch-tools/microsoft.net.compilers.toolset.4.12.0/tasks/net472/csc.exe'
$uiManaged=Join-Path $uiRoot 'BrassBirminghamUnity_Data/Managed'
$uiStock=Join-Path $uiRoot 'artifacts/patch-baseline/Assembly-CSharp.dll'
if((Get-FileHash -LiteralPath $uiStock).Hash -ne $uiBase.stockSha256){throw 'Stock fingerprint mismatch.'}
$uiReferences=@(Get-ChildItem -LiteralPath $uiManaged -Filter '*.dll'|Where-Object {$_.Name -notin 'Assembly-CSharp.dll','BrassPatch.dll','BrassLegacyAI.dll' -and $_.Name -notmatch 'Tests|Regression'}|ForEach-Object {'/reference:"'+$_.FullName+'"'})
$uiReferences+='/reference:"'+$uiStock+'"'
$uiInput=@(Get-ChildItem -LiteralPath (Join-Path $uiSources 'src/BrassPatch') -Recurse -Filter '*.cs'|ForEach-Object {'"'+$_.FullName+'"'})
@('/nologo','/target:library','/langversion:7.3','/nostdlib+','/deterministic+','/optimize+',('/out:"'+(Join-Path $uiOut 'Managed/BrassPatch.dll')+'"'))+$uiReferences+$uiInput|Set-Content -LiteralPath (Join-Path $uiOut 'helper.rsp') -Encoding utf8
& $uiCompiler ('@'+(Join-Path $uiOut 'helper.rsp'))
if($LASTEXITCODE -ne 0){throw 'UI helper compilation failed.'}
foreach($name in 'Assembly-CSharp.dll','BrassLegacyAI.dll'){Copy-Item -LiteralPath (Join-Path $uiArchive ('Managed/'+$name)) -Destination (Join-Path $uiOut ('Managed/'+$name)) -Force}
$uiPins=@(Get-ChildItem -LiteralPath $uiSources -Recurse -File|Where-Object Extension -in '.cs','.json','.md'|ForEach-Object {[ordered]@{path=[IO.Path]::GetRelativePath($uiSources,$_.FullName).Replace('\','/');sha256=(Get-FileHash -LiteralPath $_.FullName).Hash}})
[ordered]@{version=$Version;stockSha256=$uiBase.stockSha256;baseVersion=$BaseVersion;helperSourceBaseVersion='1.4.17';baseFiles=$uiBase.files;sourceScope=$uiOverlay;sourcePins=$uiPins;files=@(Get-ChildItem -LiteralPath (Join-Path $uiOut 'Managed') -File|Where-Object Name -in 'Assembly-CSharp.dll','BrassPatch.dll','BrassLegacyAI.dll'|ForEach-Object {[ordered]@{name=$_.Name;sha256=(Get-FileHash -LiteralPath $_.FullName).Hash}})}|ConvertTo-Json -Depth 7|Set-Content -LiteralPath (Join-Path $uiOut 'build-manifest.json') -Encoding utf8
Write-Output ('Built '+$Version+' from '+$BaseVersion+'; accepted game and AI DLLs retained.')
