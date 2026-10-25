param([string]$BuildDirectory='artifacts/network-input-build')
$ErrorActionPreference='Stop'
$historyRoot=Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot 'patch-install-common.ps1')
$historyBuild=Join-Path $historyRoot $BuildDirectory
$candidate=Read-BrassJson (Join-Path $historyBuild 'build-manifest.json')
$destination=Join-Path $historyRoot ('artifacts/patch-releases/'+$candidate.version)
if(Test-Path -LiteralPath $destination){throw 'Release archive already exists; refusing to replace it.'}
$validation=Read-BrassJson (Join-Path $historyBuild 'validation.json')
$receipt=Read-BrassJson (Join-Path $historyRoot 'artifacts/patch-installation.json')
$tracker=Read-BrassJson (Join-Path $historyBuild 'menu-installed.json')
$packaging=Read-BrassJson (Join-Path $historyBuild 'packaging-results.json')
Assert-BrassCandidate $candidate $validation $historyBuild
if($candidate.version -ne '1.4.34' -or $receipt.version -ne $candidate.version -or $receipt.status -ne 'installed' -or -not $tracker.passed -or -not $packaging.passed){throw 'Installed Network/menu and packaging qualification required.'}
function Copy-HistoryArchive([string]$source,[string]$relative){
 $target=[IO.Path]::GetFullPath((Join-Path $destination $relative))
 if(-not $target.StartsWith([IO.Path]::GetFullPath($destination)+[IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase)){throw 'Archive path escapes its directory.'}
 New-Item -ItemType Directory -Path (Split-Path -Parent $target) -Force|Out-Null
 Copy-Item -LiteralPath $source -Destination $target
 if((Get-FileHash -LiteralPath $source).Hash -ne (Get-FileHash -LiteralPath $target).Hash){throw 'Archive copy differs.'}
}
foreach($file in $candidate.files){
 $source=Join-Path $historyRoot ('BrassBirminghamUnity_Data/Managed/'+$file.name)
 if((Get-FileHash -LiteralPath $source).Hash -ne $file.sha256){throw 'Installed candidate differs.'}
 foreach($report in $tracker,$packaging){if(@($report.testedFiles|Where-Object {$_.name -eq $file.name -and $_.sha256 -eq $file.sha256}).Count -ne 1){throw 'Installed/packaging evidence belongs to another candidate.'}}
 Copy-HistoryArchive $source ('Managed/'+$file.name)
}
foreach($name in 'build-manifest','validation','local-ai-qualification','preserved-runtime','baseline-reproduction','link-native','link-installed','presentation-native','menu-native','menu-installed','packaging-results'){
 Copy-HistoryArchive (Join-Path $historyBuild ($name+'.json')) ($name+'.json')
}
Copy-HistoryArchive (Join-Path $historyRoot 'artifacts/patch-installation.json') 'patch-installation.json'
$link=Read-BrassJson (Join-Path $historyBuild 'link-installed.json')
if(-not $link.passed){throw 'Installed Network regression required.'}
foreach($file in $candidate.files){if(@($link.testedFiles|Where-Object {$_.name -eq $file.name -and $_.sha256 -eq $file.sha256}).Count -ne 1){throw 'Installed Network evidence fingerprint differs.'}}
Copy-HistoryArchive (Join-Path $historyBuild 'Managed/Assembly-CSharp.dll.manifest.json') 'Managed/Assembly-CSharp.dll.manifest.json'
Copy-HistoryArchive (Join-Path $historyRoot 'artifacts/link-action-bug/evidence/player-before-tests.log') 'evidence/player-before-tests.log'
foreach($pin in $candidate.sourcePins){
 $source=Join-Path $historyBuild ('runtime/'+$pin.path)
 if((Get-FileHash -LiteralPath $source).Hash -ne $pin.sha256){throw 'Compiled source snapshot changed.'}
 Copy-HistoryArchive $source $pin.path
}
$baseline=Join-Path $historyRoot ('artifacts/patch-releases/'+$candidate.baseVersion)
foreach($name in 'archive-manifest.json','build-manifest.json','validation.json','patch-installation.json'){Copy-HistoryArchive (Join-Path $baseline $name) ('baseline/'+$candidate.baseVersion+'/'+$name)}
foreach($name in 'Assembly-CSharp.dll','BrassPatch.dll','BrassLegacyAI.dll'){Copy-HistoryArchive (Join-Path $baseline ('Managed/'+$name)) ('baseline/'+$candidate.baseVersion+'/Managed/'+$name)}
Copy-HistoryArchive (Join-Path $historyBuild 'baseline/Assembly-CSharp.dll.manifest.json') ('baseline/'+$candidate.baseVersion+'/Managed/Assembly-CSharp.dll.manifest.json')
foreach($name in 'build-network-input-fix.ps1','archive-network-input-release.ps1','install-combined-patch.ps1','install-patch.ps1','patch-install-common.ps1','stage-archived-patch.ps1'){
 Copy-HistoryArchive (Join-Path $historyRoot ('scripts/'+$name)) ('scripts/'+$name)
}
foreach($folder in 'tests/BrassPatch/InformationUI','tests/BrassLocalAI','tests/BrassPatch/Presentation','tests/BrassPatch/CityLocator'){
 foreach($file in Get-ChildItem -LiteralPath (Join-Path $historyRoot $folder) -File|Where-Object Extension -in '.cs','.ps1'){Copy-HistoryArchive $file.FullName ($folder+'/'+$file.Name)}
}
Copy-HistoryArchive (Join-Path $historyRoot 'docs/NETWORK_INPUT_FIX_1.4.34.md') 'docs/NETWORK_INPUT_FIX_1.4.34.md'
$screens=Join-Path $baseline 'screenshots'
foreach($entry in $validation.visualChecks.cases){
 $source=Join-Path $screens $entry.name
 if((Get-FileHash -LiteralPath $source).Hash -ne $entry.sha256){throw 'Reviewed screenshot changed.'}
 Copy-HistoryArchive $source ('screenshots/'+$entry.name)
}
foreach($name in 'link-native','link-installed','presentation-native'){
 Copy-HistoryArchive (Join-Path $historyBuild ($name+'.log')) ('logs/'+$name+'.log')
}
Copy-HistoryArchive (Join-Path $historyBuild 'menu-native.log') 'logs/menu-native.log'
Copy-HistoryArchive (Join-Path $historyBuild 'menu-installed.log') 'logs/menu-installed.log'
$inventory=@(Get-ChildItem -LiteralPath $destination -Recurse -File|Sort-Object FullName|ForEach-Object{[ordered]@{path=[IO.Path]::GetRelativePath($destination,$_.FullName).Replace('\','/');sha256=(Get-FileHash -LiteralPath $_.FullName).Hash}})
[ordered]@{version=$candidate.version;createdUtc=[DateTime]::UtcNow.ToString('o');priorArchive=$candidate.baseVersion;priorBackup=$receipt.previousReleaseBackup;files=$inventory}|ConvertTo-Json -Depth 8|Set-Content -LiteralPath (Join-Path $destination 'archive-manifest.json') -Encoding utf8
& (Join-Path $PSScriptRoot 'stage-archived-patch.ps1') -Version $candidate.version -VerifyOnly
Write-Output ('Archived and verified Network input fix '+$candidate.version+': '+$inventory.Count+' files.')
