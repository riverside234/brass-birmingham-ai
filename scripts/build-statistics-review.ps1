param([string]$BaseVersion='1.4.16',[string]$Version='1.4.17')
$ErrorActionPreference='Stop'
$reviewRoot=Split-Path -Parent $PSScriptRoot
$reviewOut=Join-Path $reviewRoot 'artifacts/statistics-review-build'
$reviewSources=Join-Path $reviewOut 'runtime'
$reviewArchive=Join-Path $reviewRoot ('artifacts/patch-releases/'+$BaseVersion)
$reviewHelperArchive=Join-Path $reviewRoot 'artifacts/patch-releases/1.4.13'
function VerifiedArchive([string]$path){
 $inventory=Get-Content -LiteralPath (Join-Path $path 'archive-manifest.json') -Raw|ConvertFrom-Json
 foreach($pin in $inventory.files){
  $source=[IO.Path]::GetFullPath((Join-Path $path $pin.path))
  if(-not $source.StartsWith([IO.Path]::GetFullPath($path)+[IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase)){throw 'Archive path escapes its directory.'}
  if((Get-FileHash -LiteralPath $source).Hash -ne $pin.sha256){throw ('Archived file changed: '+$pin.path)}
 }
 $manifest=Get-Content -LiteralPath (Join-Path $path 'build-manifest.json') -Raw|ConvertFrom-Json
 $validation=Get-Content -LiteralPath (Join-Path $path 'validation.json') -Raw|ConvertFrom-Json
 if(-not $validation.passed){throw 'Unqualified baseline.'}
 return $manifest
}
$reviewBase=VerifiedArchive $reviewArchive
$reviewHelperBase=VerifiedArchive $reviewHelperArchive
if($reviewBase.version -ne $BaseVersion){throw 'Baseline version mismatch.'}
New-Item -ItemType Directory -Path $reviewOut,$reviewSources,(Join-Path $reviewOut 'Managed') -Force|Out-Null
# 1.4.16 retains the 1.4.13 helper byte for byte but archives AI sources only.
$baseHelper=@($reviewBase.files|Where-Object name -eq 'BrassPatch.dll')[0]
$helperHelper=@($reviewHelperBase.files|Where-Object name -eq 'BrassPatch.dll')[0]
if($baseHelper.sha256 -ne $helperHelper.sha256){throw 'Helper provenance changed; review source baseline.'}
foreach($baseline in @(@{path=$reviewHelperArchive;manifest=$reviewHelperBase},@{path=$reviewArchive;manifest=$reviewBase})){
 foreach($pin in $baseline.manifest.sourcePins){
  if(($pin.path -notlike 'src/*' -and $pin.path -notlike 'configs/*' -and $pin.path -notlike 'docs/*') -or $pin.path.Contains('..')){throw ('Unexpected source path: '+$pin.path)}
  $source=Join-Path $baseline.path $pin.path
  if((Get-FileHash -LiteralPath $source).Hash -ne $pin.sha256){throw 'Pinned source mismatch.'}
  $target=Join-Path $reviewSources $pin.path
  New-Item -ItemType Directory -Path (Split-Path -Parent $target) -Force|Out-Null
  Copy-Item -LiteralPath $source -Destination $target -Force
 }
}
Get-ChildItem -LiteralPath (Join-Path $reviewRoot 'src/BrassPatch/Statistics') -File|ForEach-Object {
 Copy-Item -LiteralPath $_.FullName -Destination (Join-Path $reviewSources 'src/BrassPatch/Statistics') -Force
}
$reviewCompiler=Join-Path $reviewRoot 'artifacts/patch-tools/microsoft.net.compilers.toolset.4.12.0/tasks/net472/csc.exe'
$reviewManaged=Join-Path $reviewRoot 'BrassBirminghamUnity_Data/Managed'
$reviewStock=Join-Path $reviewRoot 'artifacts/patch-baseline/Assembly-CSharp.dll'
if((Get-FileHash -LiteralPath $reviewStock).Hash -ne $reviewBase.stockSha256){throw 'Stock fingerprint mismatch.'}
$reviewReferences=@(Get-ChildItem -LiteralPath $reviewManaged -Filter '*.dll'|Where-Object {$_.Name -notin 'Assembly-CSharp.dll','BrassPatch.dll','BrassLegacyAI.dll' -and $_.Name -notmatch 'Tests|Regression'}|ForEach-Object {'/reference:"'+$_.FullName+'"'})
$reviewReferences+='/reference:"'+$reviewStock+'"'
$reviewInput=@(Get-ChildItem -LiteralPath (Join-Path $reviewSources 'src/BrassPatch') -Recurse -Filter '*.cs'|ForEach-Object {'"'+$_.FullName+'"'})
$reviewHelper=Join-Path $reviewOut 'Managed/BrassPatch.dll'
@('/nologo','/target:library','/langversion:7.3','/nostdlib+','/deterministic+','/optimize+',('/out:"'+$reviewHelper+'"'))+$reviewReferences+$reviewInput|Set-Content -LiteralPath (Join-Path $reviewOut 'helper.rsp') -Encoding utf8
& $reviewCompiler ('@'+(Join-Path $reviewOut 'helper.rsp'))
if($LASTEXITCODE -ne 0){throw 'Statistics helper compilation failed.'}
foreach($name in 'Assembly-CSharp.dll','BrassLegacyAI.dll'){
 Copy-Item -LiteralPath (Join-Path $reviewArchive ('Managed/'+$name)) -Destination (Join-Path $reviewOut ('Managed/'+$name)) -Force
}
$reviewPins=@(Get-ChildItem -LiteralPath $reviewSources -Recurse -File|Where-Object Extension -in '.cs','.json','.md'|ForEach-Object {
 [ordered]@{path=[IO.Path]::GetRelativePath($reviewSources,$_.FullName).Replace('\','/');sha256=(Get-FileHash -LiteralPath $_.FullName).Hash}
})
$reviewManifest=[ordered]@{version=$Version;stockSha256=$reviewBase.stockSha256;baseVersion=$BaseVersion;helperSourceBaseVersion='1.4.13';baseFiles=$reviewBase.files;sourcePins=$reviewPins;files=@(Get-ChildItem -LiteralPath (Join-Path $reviewOut 'Managed') -File|Where-Object Name -in 'Assembly-CSharp.dll','BrassPatch.dll','BrassLegacyAI.dll'|ForEach-Object {[ordered]@{name=$_.Name;sha256=(Get-FileHash -LiteralPath $_.FullName).Hash}})}
$reviewManifest|ConvertTo-Json -Depth 7|Set-Content -LiteralPath (Join-Path $reviewOut 'build-manifest.json') -Encoding utf8
Write-Output ('Built '+$Version+' from '+$BaseVersion+'. Game and AI DLLs retained exactly; statistics helper rebuilt in dedicated staging.')
