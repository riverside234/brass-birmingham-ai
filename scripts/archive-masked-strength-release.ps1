$ErrorActionPreference='Stop'
$root=Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot 'patch-install-common.ps1')
$build=Join-Path $root 'artifacts/patch-build'
$destination=Join-Path $root 'artifacts/patch-releases/1.4.11'
if(Test-Path -LiteralPath $destination){throw 'Release archive already exists; refusing to replace it.'}
$candidate=Read-BrassJson (Join-Path $build 'build-manifest.json')
$validation=Read-BrassJson (Join-Path $build 'validation.json')
$record=Read-BrassJson (Join-Path $root 'artifacts/patch-installation.json')
$installed=Read-BrassJson (Join-Path $build 'local-ai-installed.json')
Assert-BrassCandidate $candidate $validation $build
if($candidate.version -ne '1.4.11' -or $record.version -ne '1.4.11' -or $record.status -ne 'installed' -or -not $installed.passed -or $installed.checks -lt 259 -or -not $installed.resourceTests.passed){throw 'Installed, natively qualified 1.4.11 is required.'}
$sourceSnapshots=@{}
foreach($pin in $candidate.sourcePins){
 $source=Join-Path $root $pin.path
 if((Get-FileHash -LiteralPath $source).Hash -ne $pin.sha256){
  $source=Join-Path $root ('artifacts/patch-releases/1.4.10/'+$pin.path)
  if(-not $pin.path.StartsWith('src/BrassPatch/') -or -not(Test-Path -LiteralPath $source) -or (Get-FileHash -LiteralPath $source).Hash -ne $pin.sha256){throw ('Compiled source snapshot is unavailable: '+$pin.path)}
 }
 $sourceSnapshots[$pin.path]=$source
}
& (Join-Path $PSScriptRoot 'stage-archived-patch.ps1') -Version 1.4.10 -VerifyOnly
$prior=Read-BrassJson (Join-Path $root 'artifacts/patch-releases/1.4.10/build-manifest.json')
foreach($file in $prior.files){if((Get-FileHash -LiteralPath (Join-Path $record.previousReleaseBackup $file.name)).Hash -ne $file.sha256){throw 'Coherent previous-release backup differs.'}}
function Copy-Archive([string]$source,[string]$relative){
 $target=Join-Path $destination $relative
 New-Item -ItemType Directory -Path (Split-Path -Parent $target) -Force|Out-Null
 Copy-Item -LiteralPath $source -Destination $target
 if((Get-FileHash -LiteralPath $source).Hash -ne (Get-FileHash -LiteralPath $target).Hash){throw 'Archive copy differs.'}
}
foreach($file in $candidate.files){
 $path=Join-Path $root ('BrassBirminghamUnity_Data/Managed/'+$file.name)
 if((Get-FileHash -LiteralPath $path).Hash -ne $file.sha256){throw 'Installed candidate differs.'}
 foreach($evidence in @($record.files,$installed.testedFiles)){
  if(@($evidence|Where-Object {$_.name -eq $file.name -and $_.sha256 -eq $file.sha256}).Count -ne 1){throw 'Installed evidence differs.'}
 }
 Copy-Archive $path ('Managed/'+$file.name)
}
foreach($name in 'build-manifest','validation','local-ai-qualification','masked-runtime','local-ai-execution','local-ai-native','local-ai-installed','local-ai-conversion-0','local-ai-credit-0','local-ai-cash-beer-0','local-ai-v2-save-compatibility','local-ai-v3-save-compatibility','local-ai-1.4.3-save-compatibility'){
 Copy-Archive (Join-Path $build ($name+'.json')) ($name+'.json')
}
Copy-Archive (Join-Path $root 'artifacts/patch-installation.json') 'patch-installation.json'
Copy-Archive (Join-Path $build 'Managed/Assembly-CSharp.dll.manifest.json') 'Assembly-CSharp.dll.manifest.json'
foreach($pin in $candidate.sourcePins){Copy-Archive $sourceSnapshots[$pin.path] $pin.path}
foreach($folder in 'src/BrassLegacyAI','tests/MaskedISMCTS'){
 $source=Join-Path $root $folder
 foreach($file in Get-ChildItem -LiteralPath $source -Recurse -File|Where-Object {$_.Extension -in '.cs','.py','.json','.md','.ps1' -or $_.Name -like 'LICENSE*'}){Copy-Archive $file.FullName ($folder+'/'+[IO.Path]::GetRelativePath($source,$file.FullName))}
}
foreach($folder in 'src/BrassLocalAI','src/BrassMaskedSearch'){
 foreach($file in Get-ChildItem -LiteralPath (Join-Path $root $folder) -Recurse -File|Where-Object {$_.Extension -eq '.md' -or $_.Name -like 'LICENSE*'}){Copy-Archive $file.FullName ($folder+'/'+[IO.Path]::GetRelativePath((Join-Path $root $folder),$file.FullName))}
}
foreach($file in Get-ChildItem -LiteralPath (Join-Path $root 'configs/legacy-ai') -Filter '*-v3-masked*.json'){
 Copy-Archive $file.FullName ('configs/legacy-ai/'+$file.Name)
}
foreach($name in 'MASKED_ISMCTS_STRENGTH_1.4.11.md','MASKED_ISMCTS_STRENGTH_EXIT_CHECKS.md','AI_EXECUTION_RESOURCES_1.4.7.md','STATISTICS_READABILITY_1.4.10.md','TRAINING_RECOVERY_LAB.md'){
 Copy-Archive (Join-Path $root ('docs/'+$name)) ('docs/'+$name)
}
foreach($name in 'build-local-ai.ps1','build-patch.ps1','install-patch.ps1','install-combined-patch.ps1','patch-install-common.ps1','test-installed-local-ai.ps1','stage-archived-patch.ps1','archive-masked-strength-release.ps1','uninstall-patch.ps1'){
 Copy-Archive (Join-Path $PSScriptRoot $name) ('scripts/'+$name)
}
foreach($file in Get-ChildItem -LiteralPath (Join-Path $root 'tests/BrassLocalAI') -File|Where-Object Extension -in '.cs','.ps1'){
 Copy-Archive $file.FullName ('tests/BrassLocalAI/'+$file.Name)
}
Copy-Archive (Join-Path $root 'configs/contracts/masked-ismcts/manifest.json') 'configs/contracts/masked-ismcts/manifest.json'
Copy-Archive (Join-Path $root 'configs/references/masked-ismcts.lock.json') 'configs/references/masked-ismcts.lock.json'
Copy-Archive (Join-Path $root 'artifacts/masked-strength/release-1.4.11/exit-summary.json') 'exit-summary.json'
foreach($name in 'screen','decisions','laptop-clocks'){Copy-Archive (Join-Path $root ('artifacts/masked-strength/release-1.4.11/'+$name+'.json')) ('strength/'+$name+'.json')}
Copy-Archive (Join-Path $root 'training/opponents/evaluations/masked-strength-1.1.0/report.json') 'strength/bank-receipt.json'
$inventory=@(Get-ChildItem -LiteralPath $destination -Recurse -File|Sort-Object FullName|ForEach-Object{
 [ordered]@{path=[IO.Path]::GetRelativePath($destination,$_.FullName).Replace('\','/');sha256=(Get-FileHash -LiteralPath $_.FullName).Hash}
})
[ordered]@{version='1.4.11';createdUtc=[DateTime]::UtcNow.ToString('o');files=$inventory;priorArchive='1.4.10';priorBackup=$record.previousReleaseBackup}|ConvertTo-Json -Depth 8|Set-Content -LiteralPath (Join-Path $destination 'archive-manifest.json') -Encoding UTF8
& (Join-Path $PSScriptRoot 'stage-archived-patch.ps1') -Version 1.4.11 -VerifyOnly
Write-Output ('Archived and verified 1.4.11: '+$inventory.Count+' files.')
