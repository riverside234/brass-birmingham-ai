param([string]$BaseVersion='1.4.33',[string]$Version='1.4.34',[string]$BuildDirectory='artifacts/network-input-build',[string]$BaselineHookInventory)
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
$scope=@('src/BrassPatch/NetworkInput/NetworkInputFixes.cs','src/BrassPatch/NetworkInput/manifest.json')
foreach($path in $scope){$target=Join-Path $sources $path;New-Item -ItemType Directory -Path (Split-Path -Parent $target) -Force|Out-Null;Copy-Item -LiteralPath (Join-Path $root $path) -Destination $target -Force}
Get-ChildItem -LiteralPath (Join-Path $root 'BrassBirminghamUnity_Data/Managed') -Filter '*.dll'|Where-Object Name -notmatch 'Tests|Regression'|Copy-Item -Destination $managed -Force
Copy-Item -LiteralPath (Join-Path $archive 'Managed/BrassLegacyAI.dll') -Destination $managed -Force
$stock=Join-Path $root 'artifacts/patch-baseline/Assembly-CSharp.dll'
if((Get-FileHash -LiteralPath $stock).Hash -ne $base.stockSha256){throw 'Stock fingerprint changed.'}
$compiler=Join-Path $root 'artifacts/patch-tools/microsoft.net.compilers.toolset.4.12.0/tasks/net472/csc.exe'
$rsp=Join-Path $build 'helper.rsp'
@('/nologo','/target:library','/langversion:7.3','/nostdlib+','/deterministic+','/optimize+',('/out:"'+(Join-Path $managed 'BrassPatch.dll')+'"'))+
 @(Get-ChildItem -LiteralPath $managed -Filter '*.dll'|Where-Object {$_.Name -notin 'Assembly-CSharp.dll','BrassPatch.dll','BrassLegacyAI.dll' -and $_.Name -notmatch 'Tests|Regression'}|ForEach-Object{'/reference:"'+$_.FullName+'"'})+
 @('/reference:"'+$stock+'"')+@(Get-ChildItem -LiteralPath (Join-Path $sources 'src/BrassPatch') -Recurse -Filter '*.cs'|ForEach-Object{'"'+$_.FullName+'"'})|Set-Content -LiteralPath $rsp -Encoding utf8
& $compiler ('@'+$rsp)
if($LASTEXITCODE -ne 0){throw 'Network guard compilation failed.'}
# Retain the matching baseline hook inventory. This is metadata; the
# overlay weaver independently checks every changed existing method against
# the two new hooks, and qualification compares every prior method body.
$baselineDll=Join-Path $build 'baseline/Assembly-CSharp.dll'
New-Item -ItemType Directory -Path (Split-Path -Parent $baselineDll) -Force|Out-Null
Copy-Item -LiteralPath (Join-Path $archive 'Managed/Assembly-CSharp.dll') -Destination $baselineDll -Force
$hookInventory=if($BaselineHookInventory){(Resolve-Path -LiteralPath $BaselineHookInventory).Path}else{Join-Path $root 'artifacts/patch-build/Managed/Assembly-CSharp.dll.manifest.json'}
$oldHooks=Get-Content -LiteralPath $hookInventory -Raw|ConvertFrom-Json
$inventoryDll=$hookInventory.Substring(0,$hookInventory.Length-'.manifest.json'.Length)
if(-not $oldHooks.methods.Count -or (Get-FileHash -LiteralPath $inventoryDll).Hash -ne ($base.files|Where-Object name -eq 'Assembly-CSharp.dll').sha256){throw 'Matching baseline hook inventory required.'}
$oldHooks|ConvertTo-Json -Depth 8|Set-Content -LiteralPath ($baselineDll+'.manifest.json') -Encoding utf8
& (Join-Path $root 'artifacts/patch-build/BrassPatch.Weaver.exe') --overlay $baselineDll (Join-Path $managed 'BrassPatch.dll') (Join-Path $sources 'src/BrassPatch/NetworkInput') (Join-Path $managed 'Assembly-CSharp.dll') ($base.files|Where-Object name -eq 'Assembly-CSharp.dll').sha256
if($LASTEXITCODE -ne 0){throw 'Network hook weaving failed.'}
$pins=@(Get-ChildItem -LiteralPath $sources -Recurse -File|Sort-Object FullName|ForEach-Object{[ordered]@{path=[IO.Path]::GetRelativePath($sources,$_.FullName).Replace('\','/');sha256=(Get-FileHash -LiteralPath $_.FullName).Hash}})
$files=@('Assembly-CSharp.dll','BrassPatch.dll','BrassLegacyAI.dll'|ForEach-Object{[ordered]@{name=$_;sha256=(Get-FileHash -LiteralPath (Join-Path $managed $_)).Hash}})
$manifest=$base
$fields=[ordered]@{version=$Version;baseVersion=$BaseVersion;baseFiles=$base.files;files=$files;sourcePins=$pins;sourceScope=$scope;networkInputGuard=$true;baselineValidationSha256=(Get-FileHash -LiteralPath (Join-Path $archive 'validation.json')).Hash;scope='Reject stale Network UI callbacks and validate action ownership before placement mutation. All prior gameplay, two-worker AI, four search allowances and UI preserved.'}
foreach($key in $fields.Keys){$manifest|Add-Member -NotePropertyName $key -NotePropertyValue $fields[$key] -Force}
$manifest|ConvertTo-Json -Depth 10|Set-Content -LiteralPath (Join-Path $build 'build-manifest.json') -Encoding utf8
Write-Output ('Built '+$Version+' over '+$BaseVersion+' in '+$build)
