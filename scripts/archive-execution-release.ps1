$ErrorActionPreference='Stop'
$root=Split-Path -Parent $PSScriptRoot
$build=Join-Path $root 'artifacts/patch-build'
$destination=Join-Path $root 'artifacts/patch-releases/1.4.7'
if(Test-Path -LiteralPath $destination){throw 'Release archive already exists; refusing to replace it'}
$candidate=Get-Content -LiteralPath (Join-Path $build 'build-manifest.json') -Raw|ConvertFrom-Json
$record=Get-Content -LiteralPath (Join-Path $root 'artifacts/patch-installation.json') -Raw|ConvertFrom-Json
$qualification=Get-Content -LiteralPath (Join-Path $build 'local-ai-qualification.json') -Raw|ConvertFrom-Json
if($candidate.version -ne '1.4.7' -or $record.version -ne $candidate.version -or $record.status -ne 'installed' -or -not $qualification.passed){throw 'Qualified installed 1.4.7 required'}
foreach($pin in @($qualification.sourcePins)+@($qualification.testPins)){if((Get-FileHash -LiteralPath (Join-Path $root $pin.path)).Hash -ne $pin.sha256){throw 'Qualified source changed'}}
foreach($name in 'local-ai-installed','local-ai-h5-installed','installation-verification','packaging-results'){
 $result=Get-Content -LiteralPath (Join-Path $build ($name+'.json')) -Raw|ConvertFrom-Json
 if(-not $result.passed){throw ('Installed/recovery check missing: '+$name)}
 if($name -ne 'local-ai-h5-installed'){
  $fingerprints=if($name -eq 'local-ai-installed'){$result.testedFiles}else{$result.files}
  foreach($f in $candidate.files){if(@($fingerprints|Where-Object {$_.name -eq $f.name -and $_.sha256 -eq $f.sha256}).Count -ne 1){throw 'Installed/recovery evidence differs from release'}}
 }
}
New-Item -ItemType Directory -Path $destination|Out-Null
function Copy-Archive([string]$source,[string]$relative){
 $target=Join-Path $destination $relative
 New-Item -ItemType Directory -Path (Split-Path -Parent $target) -Force|Out-Null
 Copy-Item -LiteralPath $source -Destination $target
 if((Get-FileHash -LiteralPath $source).Hash -ne (Get-FileHash -LiteralPath $target).Hash){throw 'Archive copy differs'}
}
foreach($f in $candidate.files){
 $installed=Join-Path $root ('BrassBirminghamUnity_Data/Managed/'+$f.name)
 if((Get-FileHash -LiteralPath $installed).Hash -ne $f.sha256){throw 'Installed candidate changed'}
 Copy-Archive $installed $f.name
}
Copy-Archive (Join-Path $build 'Managed/Assembly-CSharp.dll.manifest.json') 'Assembly-CSharp.dll.manifest.json'
foreach($name in 'build-manifest','validation','local-ai-qualification','local-ai-execution','local-ai-runtime-0','local-ai-native','local-ai-resource-baseline','local-ai-gameplay-live','local-ai-v2-save-compatibility','local-ai-v3-save-compatibility','local-ai-1.4.3-save-compatibility','local-ai-installed','local-ai-h5','local-ai-h5-installed','packaging-results','installation-verification'){
 Copy-Archive (Join-Path $build ($name+'.json')) ($name+'.json')
}
Copy-Archive (Join-Path $root 'artifacts/patch-installation.json') 'patch-installation.json'
Copy-Archive (Join-Path $root 'docs/AI_EXECUTION_RESOURCES_1.4.7.md') 'AI_EXECUTION_RESOURCES_1.4.7.md'
foreach($name in 'BrassPatch','BrassLocalAI'){
 $sourceRoot=Join-Path $root ('src/'+$name)
 foreach($file in Get-ChildItem -LiteralPath $sourceRoot -Recurse -File){Copy-Archive $file.FullName ($name+'/'+[IO.Path]::GetRelativePath($sourceRoot,$file.FullName))}
}
foreach($folder in 'scripts','tests/BrassPatch','tests/BrassLocalAI'){
 $sourceRoot=Join-Path $root $folder
 foreach($file in Get-ChildItem -LiteralPath $sourceRoot -Recurse -File|Where-Object Extension -in '.cs','.ps1'){
  Copy-Archive $file.FullName ($folder+'/'+[IO.Path]::GetRelativePath($sourceRoot,$file.FullName))
 }
}
$files=@(Get-ChildItem -LiteralPath $destination -Recurse -File|Sort-Object FullName|ForEach-Object{
 [ordered]@{path=[IO.Path]::GetRelativePath($destination,$_.FullName).Replace('\','/');sha256=(Get-FileHash -LiteralPath $_.FullName).Hash}
})
[ordered]@{version='1.4.7';createdUtc=[DateTime]::UtcNow.ToString('o');files=$files}|ConvertTo-Json -Depth 6|Set-Content -LiteralPath (Join-Path $destination 'archive-manifest.json') -Encoding UTF8
foreach($file in $files){if((Get-FileHash -LiteralPath (Join-Path $destination $file.path)).Hash -ne $file.sha256){throw 'Archive inventory verification failed'}}
Write-Output ('Archived and verified 1.4.7: '+$files.Count+' fingerprinted files.')
