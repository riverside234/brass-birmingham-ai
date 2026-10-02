$ErrorActionPreference='Stop'
$root=Split-Path -Parent $PSScriptRoot
$build=Join-Path $root 'artifacts/patch-build'
$destination=Join-Path $root 'artifacts/patch-releases/1.4.10'
if(Test-Path -LiteralPath $destination){throw 'Release archive exists; refusing to replace it.'}
$candidate=Get-Content -LiteralPath (Join-Path $build 'build-manifest.json') -Raw|ConvertFrom-Json
$record=Get-Content -LiteralPath (Join-Path $root 'artifacts/patch-installation.json') -Raw|ConvertFrom-Json
$installed=Get-Content -LiteralPath (Join-Path $build 'statistics-installed.json') -Raw|ConvertFrom-Json
$smoke=Get-Content -LiteralPath (Join-Path $build 'local-ai-installed.json') -Raw|ConvertFrom-Json
if($candidate.version -ne '1.4.10' -or $record.version -ne $candidate.version -or $record.status -ne 'installed' -or -not $installed.passed -or -not $smoke.passed){throw 'Installed, natively qualified 1.4.10 is required.'}
function Copy-Archive([string]$source,[string]$relative){
 $target=Join-Path $destination $relative
 New-Item -ItemType Directory -Path (Split-Path -Parent $target) -Force|Out-Null
 Copy-Item -LiteralPath $source -Destination $target
 if((Get-FileHash -LiteralPath $source).Hash -ne (Get-FileHash -LiteralPath $target).Hash){throw 'Archive copy differs.'}
}
foreach($file in $candidate.files){
 $path=Join-Path $root ('BrassBirminghamUnity_Data/Managed/'+$file.name)
 if((Get-FileHash -LiteralPath $path).Hash -ne $file.sha256){throw 'Installed candidate differs.'}
 foreach($report in $installed,$smoke){if(@($report.testedFiles|Where-Object {$_.name -eq $file.name -and $_.sha256 -eq $file.sha256}).Count -ne 1){throw 'Installed evidence differs.'}}
 Copy-Archive $path ('Managed/'+$file.name)
}
foreach($name in 'build-manifest','validation','local-ai-qualification','statistics-native','statistics-installed','presentation-lifecycle','link-replay','local-ai-menu','local-ai-installed','packaging-results'){
 Copy-Archive (Join-Path $build ($name+'.json')) ($name+'.json')
}
Copy-Archive (Join-Path $root 'artifacts/patch-installation.json') 'patch-installation.json'
Copy-Archive (Join-Path $build 'Managed/Assembly-CSharp.dll.manifest.json') 'Assembly-CSharp.dll.manifest.json'
Copy-Archive (Join-Path $root 'docs/STATISTICS_READABILITY_1.4.10.md') 'docs/STATISTICS_READABILITY_1.4.10.md'
Copy-Archive (Join-Path $root 'rules-audit/END_GAME_STATISTICS_UI_PLAN.md') 'docs/END_GAME_STATISTICS_UI_PLAN.md'
foreach($folder in 'src/BrassPatch','src/BrassLocalAI','src/BrassMaskedSearch','configs/legacy-ai'){
 $source=Join-Path $root $folder
 foreach($file in Get-ChildItem -LiteralPath $source -Recurse -File){Copy-Archive $file.FullName ($folder+'/'+[IO.Path]::GetRelativePath($source,$file.FullName))}
}
foreach($folder in 'scripts','tests/BrassPatch','tests/BrassLocalAI','tests/MaskedISMCTS'){
 $source=Join-Path $root $folder
 foreach($file in Get-ChildItem -LiteralPath $source -Recurse -File|Where-Object Extension -in '.cs','.ps1'){Copy-Archive $file.FullName ($folder+'/'+[IO.Path]::GetRelativePath($source,$file.FullName))}
}
$screens=Join-Path $root 'artifacts/city-locator-player/artifacts/patch-build'
foreach($file in Get-ChildItem -LiteralPath $screens -Filter 'statistics-*.png'){Copy-Archive $file.FullName ('screenshots/'+$file.Name)}
Copy-Archive (Join-Path $screens 'statistics-example.json') 'statistics-example.json'
$inventory=@(Get-ChildItem -LiteralPath $destination -Recurse -File|Sort-Object FullName|ForEach-Object{
 [ordered]@{path=[IO.Path]::GetRelativePath($destination,$_.FullName).Replace('\','/');sha256=(Get-FileHash -LiteralPath $_.FullName).Hash}
})
[ordered]@{version='1.4.10';createdUtc=[DateTime]::UtcNow.ToString('o');priorArchive='1.4.9';priorBackup=$record.previousReleaseBackup;files=$inventory}|ConvertTo-Json -Depth 8|Set-Content -LiteralPath (Join-Path $destination 'archive-manifest.json') -Encoding UTF8
& (Join-Path $PSScriptRoot 'stage-archived-patch.ps1') -Version 1.4.10 -VerifyOnly
Write-Output ('Archived and verified 1.4.10: '+$inventory.Count+' files.')
