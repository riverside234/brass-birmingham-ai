param([string]$BaseVersion='1.4.27',[string]$Version='1.4.28',[string]$BuildDirectory='artifacts/game-finishing-margin/build')
$ErrorActionPreference='Stop'
$root=Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot 'patch-install-common.ps1')
$archive=Join-Path $root ('artifacts/patch-releases/'+$BaseVersion)
$base=Read-BrassJson (Join-Path $archive 'build-manifest.json')
$inventory=Read-BrassJson (Join-Path $archive 'archive-manifest.json')
if($base.version -ne $BaseVersion -or -not (Read-BrassJson (Join-Path $archive 'validation.json')).passed){throw 'Qualified installed baseline required.'}
$prefix=[IO.Path]::GetFullPath($archive)+[IO.Path]::DirectorySeparatorChar
foreach($entry in $inventory.files){$path=[IO.Path]::GetFullPath((Join-Path $archive $entry.path));if(-not $path.StartsWith($prefix,[StringComparison]::OrdinalIgnoreCase) -or (Get-FileHash -LiteralPath $path).Hash -ne $entry.sha256){throw 'Baseline archive changed.'}}
$build=Join-Path $root $BuildDirectory;$sources=Join-Path $build 'runtime';$managed=Join-Path $build 'Managed'
if(Test-Path -LiteralPath (Join-Path $build 'validation.json')){throw 'Preserve qualified output; choose a new build directory.'}
New-Item -ItemType Directory -Path $sources,$managed -Force|Out-Null
foreach($pin in $base.sourcePins){
 if($pin.path.Contains('..') -or $pin.path -notmatch '^(src|configs|docs)/'){throw 'Unexpected snapshot path.'}
 $source=Join-Path $archive $pin.path;$target=Join-Path $sources $pin.path
 if((Get-FileHash -LiteralPath $source).Hash -ne $pin.sha256){throw 'Baseline source changed.'}
 New-Item -ItemType Directory -Path (Split-Path -Parent $target) -Force|Out-Null
 Copy-Item -LiteralPath $source -Destination $target -Force
}
$scope=@('src/BrassLocalAI/Policy/Coordinator.cs','src/BrassLocalAI/Policy/TurnFinishingBudget.cs')
foreach($path in $scope){$target=Join-Path $sources $path;New-Item -ItemType Directory -Path (Split-Path -Parent $target) -Force|Out-Null;Copy-Item -LiteralPath (Join-Path $root $path) -Destination $target -Force}
Get-ChildItem -LiteralPath (Join-Path $root 'BrassBirminghamUnity_Data/Managed') -Filter '*.dll'|Where-Object Name -notmatch 'Tests|Regression'|Copy-Item -Destination $managed -Force
foreach($name in 'Assembly-CSharp.dll','BrassPatch.dll'){Copy-Item -LiteralPath (Join-Path $archive ('Managed/'+$name)) -Destination $managed -Force}
$inputs=@(Get-ChildItem -LiteralPath (Join-Path $sources 'src/BrassLocalAI') -Recurse -Filter '*.cs')+@(Get-ChildItem -LiteralPath (Join-Path $sources 'src/BrassMaskedSearch') -Filter '*.cs')
$resources=@()
foreach($style in 'balanced','rail-tempo','cotton','goods-beer','supplier','pottery-flex'){
 foreach($suffix in 'v2','v3','v3-masked','v3-masked-strength','v3-masked-era','v3-masked-industry','v3-masked-economy'){
  $resources+='/resource:"'+(Join-Path $sources ('configs/legacy-ai/'+$style+'-'+$suffix+'.json'))+'",BrassLegacyAI.Profiles.'+$style+'-'+$suffix+'.json'
 }
 $resources+='/resource:"'+(Join-Path $sources ('configs/legacy-ai/archive/1.4.0/'+$style+'-v3.json'))+'",BrassLegacyAI.Profiles.'+$style+'-v3-1.4.0.json'
}
$rsp=Join-Path $build 'game-ai.rsp'
@('/nologo','/target:library','/langversion:7.3','/nostdlib+','/deterministic+','/optimize+',('/out:"'+(Join-Path $managed 'BrassLegacyAI.dll')+'"'))+
 @(Get-ChildItem -LiteralPath $managed -Filter '*.dll'|Where-Object {$_.Name -ne 'BrassLegacyAI.dll' -and $_.Name -notmatch 'Tests|Regression'}|ForEach-Object{'/reference:"'+$_.FullName+'"'})+
 @($inputs|ForEach-Object{'"'+$_.FullName+'"'})+$resources|Set-Content -LiteralPath $rsp -Encoding UTF8
& (Join-Path $root 'artifacts/patch-tools/microsoft.net.compilers.toolset.4.12.0/tasks/net472/csc.exe') ('@'+$rsp)
if($LASTEXITCODE -ne 0){throw 'Game-only finishing-margin compilation failed.'}
$pins=@(Get-ChildItem -LiteralPath $sources -Recurse -File|Sort-Object FullName|ForEach-Object{[ordered]@{path=[IO.Path]::GetRelativePath($sources,$_.FullName).Replace('\','/');sha256=(Get-FileHash -LiteralPath $_.FullName).Hash}})
$files=@('Assembly-CSharp.dll','BrassPatch.dll','BrassLegacyAI.dll'|ForEach-Object{[ordered]@{name=$_;sha256=(Get-FileHash -LiteralPath (Join-Path $managed $_)).Hash}})
[ordered]@{version=$Version;baseVersion=$BaseVersion;stockSha256=$base.stockSha256;baseFiles=$base.files;files=$files;sourcePins=$pins;sourceScope=$scope;searchWorkers=2;turnOptionsMs=@(2000,5000,12000);strategyRuntime='brass-local-ai/1.4.20';baselineValidationSha256=(Get-FileHash -LiteralPath (Join-Path $archive 'validation.json')).Hash;scope='Game coordinator finishing margin only. Shared Masked ISMCTS, strategies, profiles, UI, rules and training remain unchanged.'}|ConvertTo-Json -Depth 9|Set-Content -LiteralPath (Join-Path $build 'build-manifest.json') -Encoding UTF8
Write-Output ('Built '+$Version+' over '+$BaseVersion+' in '+$build)
