param([string]$Root=(Split-Path -Parent $PSScriptRoot),[string]$BuildDirectory='artifacts/economy-v3/package-1.4.20-v4',[switch]$RetryFailedBuild)
$ErrorActionPreference='Stop'
. (Join-Path $PSScriptRoot 'patch-install-common.ps1')
$root=(Resolve-Path -LiteralPath $Root).Path
$selectedBank='bank-3dc65253a610191d66a5'
$selected=Read-BrassJson (Join-Path $root ('training/opponents/bank/'+$selectedBank+'/manifest.json'))
foreach($path in $selected.payload.runtime_source_paths|Where-Object {$_ -like 'src/BrassMaskedSearch/*.cs' -or $_ -like 'src/BrassLegacyAI/*.cs'}){
 if((Get-FileHash -LiteralPath (Join-Path $root $path)).Hash -ne $selected.payload.source_files.PSObject.Properties[$path].Value){throw ('V4 must use the selected previous strategy: '+$path)}
}
if(Test-Path -LiteralPath (Join-Path $root 'src/BrassMaskedSearch/NetworkAndHand.cs')){throw 'Rejected network/Scout source must not enter v4'}
$record=Read-BrassJson (Join-Path $root 'artifacts/patch-installation.json')
if(-not $record -or $record.status -ne 'installed'){throw 'A qualified installed UI baseline is required'}
$archive=Join-Path $root ('artifacts/patch-releases/'+$record.version)
$inventory=Read-BrassJson (Join-Path $archive 'archive-manifest.json')
$prior=Read-BrassJson (Join-Path $archive 'build-manifest.json')
$validation=Read-BrassJson (Join-Path $archive 'validation.json')
if(-not $inventory -or -not $validation.passed){throw 'Installed release archive must be qualified'}
foreach($file in $inventory.files){if((Get-FileHash -LiteralPath (Join-Path $archive $file.path)).Hash -ne $file.sha256){throw 'UI baseline archive changed'}}
foreach($file in $record.files){
 if(@($prior.files|Where-Object {$_.name -eq $file.name -and $_.sha256 -eq $file.sha256}).Count -ne 1 -or
   (Get-FileHash -LiteralPath (Join-Path $root ('BrassBirminghamUnity_Data/Managed/'+$file.name))).Hash -ne $file.sha256){throw 'Installed UI differs from its qualified archive'}
}
$build=Join-Path $root $BuildDirectory
if((Test-Path -LiteralPath $build) -and (-not $RetryFailedBuild -or (Test-Path -LiteralPath (Join-Path $build 'build-manifest.json')))){throw 'Use a new build directory; preserve earlier evidence'}
$managed=Join-Path $build 'Managed';New-Item -ItemType Directory -Path $managed -Force|Out-Null
Get-ChildItem -LiteralPath (Join-Path $root 'BrassBirminghamUnity_Data/Managed') -Filter '*.dll'|Copy-Item -Destination $managed
$compiler=Join-Path $root 'artifacts/patch-tools/microsoft.net.compilers.toolset.4.12.0/tasks/net472/csc.exe'
$sources=@(Get-ChildItem -LiteralPath (Join-Path $root 'src/BrassLocalAI') -Filter '*.cs' -Recurse)
$sources+=@(Get-ChildItem -LiteralPath (Join-Path $root 'src/BrassMaskedSearch') -Filter '*.cs')
$profiles=@();$resources=@()
foreach($style in 'balanced','rail-tempo','cotton','goods-beer','supplier','pottery-flex'){
 foreach($suffix in 'v2','v3','v3-masked','v3-masked-strength','v3-masked-era','v3-masked-industry','v3-masked-economy'){
  $profile=Join-Path $root ('configs/legacy-ai/'+$style+'-'+$suffix+'.json');$profiles+=Get-Item -LiteralPath $profile
  $resources+='/resource:"'+$profile+'",BrassLegacyAI.Profiles.'+$style+'-'+$suffix+'.json'
 }
 $profile=Join-Path $root ('configs/legacy-ai/archive/1.4.0/'+$style+'-v3.json');$profiles+=Get-Item -LiteralPath $profile
 $resources+='/resource:"'+$profile+'",BrassLegacyAI.Profiles.'+$style+'-v3-1.4.0.json'
}
$pins=@($sources+$profiles|Sort-Object FullName|ForEach-Object{
 [ordered]@{path=[IO.Path]::GetRelativePath($root,$_.FullName).Replace('\','/');sha256=(Get-FileHash -LiteralPath $_.FullName).Hash}
})
$references=@(Get-ChildItem -LiteralPath $managed -Filter '*.dll'|Where-Object {$_.Name -ne 'BrassLegacyAI.dll' -and $_.Name -notmatch 'Tests|Regression'}|ForEach-Object{'/reference:"'+$_.FullName+'"'})
$rsp=Join-Path $build 'era-ai.rsp'
@('/nologo','/target:library','/langversion:7.3','/nostdlib+','/deterministic+','/optimize+',('/out:"'+(Join-Path $managed 'BrassLegacyAI.dll')+'"'))+$references+@($sources|ForEach-Object{'"'+$_.FullName+'"'})+$resources|Set-Content -LiteralPath $rsp -Encoding UTF8
& $compiler ('@'+$rsp)
if($LASTEXITCODE -ne 0){throw 'AI helper compilation failed'}
foreach($pin in $pins){
 $snapshot=Join-Path $build ('source/'+$pin.path);New-Item -ItemType Directory -Path (Split-Path -Parent $snapshot) -Force|Out-Null
 Copy-Item -LiteralPath (Join-Path $root $pin.path) -Destination $snapshot
 if((Get-FileHash -LiteralPath $snapshot).Hash -ne $pin.sha256){throw 'AI source changed during compilation'}
}
$files=@(foreach($name in 'Assembly-CSharp.dll','BrassPatch.dll','BrassLegacyAI.dll'){
 [ordered]@{name=$name;sha256=(Get-FileHash -LiteralPath (Join-Path $managed $name)).Hash}
})
[ordered]@{version='1.4.20';generation='v4';selectionBasis='user-directed-masked-v4/1.0.0';selectedBank=$selectedBank;searchWorkers=2;strategyRuntime='brass-local-ai/1.4.20';baseVersion=$record.version;baseFiles=$record.files;stockSha256=$BrassStockSha256;files=$files;sourcePins=$pins;
 baselineArchiveSha256=(Get-FileHash -LiteralPath (Join-Path $archive 'archive-manifest.json')).Hash;
 baselineValidationSha256=(Get-FileHash -LiteralPath (Join-Path $archive 'validation.json')).Hash;
 scope='User-selected previous advanced strategy as v4, including productive intent progress and opening-biased candidates. AI helper only; UI/rules retained; two game roots, existing 2/5/12 clocks, four lab roots. No further strength tournament.'}|ConvertTo-Json -Depth 12|Set-Content -LiteralPath (Join-Path $build 'build-manifest.json') -Encoding UTF8
Write-Output ('Built AI-only 1.4.20 over qualified '+$record.version+' at '+$build)


