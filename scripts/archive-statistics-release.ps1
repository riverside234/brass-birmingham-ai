$ErrorActionPreference='Stop'
$root=Split-Path -Parent $PSScriptRoot
$build=Join-Path $root 'artifacts/patch-build'
$destination=Join-Path $root 'artifacts/patch-releases/1.4.8'
if(Test-Path -LiteralPath $destination){throw 'Release archive already exists; refusing to replace it.'}
$candidate=Get-Content -LiteralPath (Join-Path $build 'build-manifest.json') -Raw|ConvertFrom-Json
$record=Get-Content -LiteralPath (Join-Path $root 'artifacts/patch-installation.json') -Raw|ConvertFrom-Json
$installed=Get-Content -LiteralPath (Join-Path $build 'statistics-installed.json') -Raw|ConvertFrom-Json
if($candidate.version -ne '1.4.8' -or $record.version -ne $candidate.version -or $record.status -ne 'installed' -or -not $installed.passed){throw 'Installed, natively qualified 1.4.8 is required.'}
function Copy-Archive([string]$source,[string]$relative){
 $target=Join-Path $destination $relative
 New-Item -ItemType Directory -Path (Split-Path -Parent $target) -Force|Out-Null
 Copy-Item -LiteralPath $source -Destination $target
 if((Get-FileHash -LiteralPath $source).Hash -ne (Get-FileHash -LiteralPath $target).Hash){throw 'Archive copy differs.'}
}
foreach($file in $candidate.files){
 $path=Join-Path $root ('BrassBirminghamUnity_Data/Managed/'+$file.name)
 if((Get-FileHash -LiteralPath $path).Hash -ne $file.sha256){throw 'Installed candidate differs.'}
 if(@($installed.testedFiles|Where-Object {$_.name -eq $file.name -and $_.sha256 -eq $file.sha256}).Count -ne 1){throw 'Installed evidence differs.'}
 Copy-Archive $path $file.name
}
foreach($name in 'build-manifest','validation','local-ai-qualification','statistics-native','statistics-installed','presentation-lifecycle','link-replay','unity-isolated-result','local-ai-runtime-0','local-ai-menu','local-ai-installed','packaging-results'){
 Copy-Archive (Join-Path $build ($name+'.json')) ($name+'.json')
}
Copy-Archive (Join-Path $root 'artifacts/patch-installation.json') 'patch-installation.json'
Copy-Archive (Join-Path $build 'Managed/Assembly-CSharp.dll.manifest.json') 'Assembly-CSharp.dll.manifest.json'
Copy-Archive (Join-Path $root 'docs/END_GAME_STATISTICS_1.4.8.md') 'END_GAME_STATISTICS_1.4.8.md'
Copy-Archive (Join-Path $root 'rules-audit/END_GAME_STATISTICS_UI_PLAN.md') 'END_GAME_STATISTICS_UI_PLAN.md'
foreach($name in 'BrassPatch','BrassLocalAI'){
 $source=Join-Path $root ('src/'+$name)
 foreach($file in Get-ChildItem -LiteralPath $source -Recurse -File){Copy-Archive $file.FullName ($name+'/'+[IO.Path]::GetRelativePath($source,$file.FullName))}
}
foreach($folder in 'scripts','tests/BrassPatch','tests/BrassLocalAI'){
 $source=Join-Path $root $folder
 foreach($file in Get-ChildItem -LiteralPath $source -Recurse -File|Where-Object Extension -in '.cs','.ps1'){Copy-Archive $file.FullName ($folder+'/'+[IO.Path]::GetRelativePath($source,$file.FullName))}
}
$screens=Join-Path $root 'artifacts/city-locator-player/artifacts/patch-build'
foreach($file in Get-ChildItem -LiteralPath $screens -Filter 'statistics-*.png'){Copy-Archive $file.FullName ('screenshots/'+$file.Name)}
Copy-Archive (Join-Path $screens 'statistics-example.json') 'statistics-example.json'
$inventory=@(Get-ChildItem -LiteralPath $destination -Recurse -File|Sort-Object FullName|ForEach-Object{
 [ordered]@{path=[IO.Path]::GetRelativePath($destination,$_.FullName).Replace('\','/');sha256=(Get-FileHash -LiteralPath $_.FullName).Hash}
})
[ordered]@{version='1.4.8';createdUtc=[DateTime]::UtcNow.ToString('o');files=$inventory}|ConvertTo-Json -Depth 8|Set-Content -LiteralPath (Join-Path $destination 'archive-manifest.json') -Encoding UTF8
foreach($file in $inventory){if((Get-FileHash -LiteralPath (Join-Path $destination $file.path)).Hash -ne $file.sha256){throw 'Archive inventory differs.'}}
Write-Output ('Archived and verified 1.4.8: '+$inventory.Count+' files.')
