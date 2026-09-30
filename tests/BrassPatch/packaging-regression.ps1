param([ValidateSet('1.0.0','1.1.0','1.2.0','1.3.0','1.3.1','1.3.2','1.4.0','1.4.1','1.4.2','1.4.3','1.4.4','1.4.5','1.4.6')][string]$PreviousRelease='1.3.1')
$ErrorActionPreference='Stop'
Set-StrictMode -Version Latest
$root=Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$build=Join-Path $root 'artifacts/patch-build'
[ordered]@{passed=$false;status='running';previousRelease=$PreviousRelease;startedUtc=[DateTime]::UtcNow.ToString('o')}|ConvertTo-Json|Set-Content -LiteralPath (Join-Path $build 'packaging-results.json') -Encoding UTF8
$previous=Join-Path $root ('artifacts/patch-releases/'+$PreviousRelease)
$fixture=Join-Path $root ('artifacts/packaging-upgrade-tests/'+[Guid]::NewGuid().ToString('N'))
$game=Join-Path $fixture 'game'
$managed=Join-Path $game 'BrassBirminghamUnity_Data/Managed'
$scripts=Join-Path $fixture 'scripts'
$candidate=Join-Path $fixture 'artifacts/patch-build'
$baseline=Join-Path $fixture 'artifacts/patch-baseline'
$saves=Join-Path $fixture 'artificial-saves'
New-Item -ItemType Directory -Path $managed,$scripts,(Join-Path $candidate 'Managed'),$baseline,$saves -Force | Out-Null
foreach($name in @('install-patch.ps1','install-combined-patch.ps1','uninstall-patch.ps1','patch-install-common.ps1')){Copy-Item -LiteralPath (Join-Path $root ('scripts/'+$name)) -Destination $scripts}
foreach($name in @('Assembly-CSharp.dll','BrassPatch.dll')){
 Copy-Item -LiteralPath (Join-Path $previous $name) -Destination $managed
 Copy-Item -LiteralPath (Join-Path $build ('Managed/'+$name)) -Destination (Join-Path $candidate 'Managed')
}
if(Test-Path -LiteralPath (Join-Path $build 'Managed/BrassLegacyAI.dll')){Copy-Item -LiteralPath (Join-Path $build 'Managed/BrassLegacyAI.dll') -Destination (Join-Path $candidate 'Managed')}
if(Test-Path -LiteralPath (Join-Path $previous 'BrassLegacyAI.dll')){Copy-Item -LiteralPath (Join-Path $previous 'BrassLegacyAI.dll') -Destination $managed}
Copy-Item -LiteralPath (Join-Path $root 'artifacts/patch-baseline/Assembly-CSharp.dll') -Destination $baseline
foreach($name in @('build-manifest.json','validation.json')){Copy-Item -LiteralPath (Join-Path $build $name) -Destination $candidate}
$recordPath=Join-Path $fixture 'artifacts/patch-installation.json'
$record=Get-Content -LiteralPath (Join-Path $previous 'patch-installation.json') -Raw | ConvertFrom-Json
$record.gameRoot=$game;$record.stockBackup=Join-Path $baseline 'Assembly-CSharp.dll';$record.saveDataPaths=@($saves)
$record|ConvertTo-Json -Depth 12|Set-Content -LiteralPath $recordPath -Encoding UTF8
$manifest=Get-Content -LiteralPath (Join-Path $candidate 'build-manifest.json') -Raw | ConvertFrom-Json
$checks=0
function Restore-Prior {
 foreach($file in $record.files){Copy-Item -LiteralPath (Join-Path $previous $file.name) -Destination (Join-Path $managed $file.name) -Force}
 $installedAi=Join-Path $managed 'BrassLegacyAI.dll'
 if('BrassLegacyAI.dll' -notin @($record.files|ForEach-Object name) -and (Test-Path -LiteralPath $installedAi)){Remove-Item -LiteralPath $installedAi}
}
function Run-Package([string]$name,[bool]$success){
 $output=& pwsh -NoProfile -File (Join-Path $scripts $name) -GameRoot $game 2>&1
 if(($LASTEXITCODE -eq 0) -ne $success){throw ($name+' unexpected result: '+($output -join [Environment]::NewLine))}
 $script:checks++
}
function Assert-Candidate {
 foreach($file in $manifest.files){if((Get-FileHash -LiteralPath (Join-Path $managed $file.name)).Hash -ne $file.sha256){throw 'Fixture install hash mismatch'};$script:checks++}
 $current=Get-Content -LiteralPath $recordPath -Raw|ConvertFrom-Json
 if($current.version -ne $manifest.version -or $current.onlineEnabled -or $current.status -ne 'installed'){throw 'Fixture installation record incorrect'};$script:checks++
 if(@($current.saveDataPaths).Count -ne 1 -or $current.saveDataPaths[0] -ne $saves){throw 'Configured save directory was not retained'};$script:checks++
}
# Real script upgrade from the prior release, entirely within the fixture tree.
Run-Package 'install-patch.ps1' $true
Assert-Candidate
# A changed candidate cannot be published over the installed, validated pair.
$candidateHelper=Join-Path $candidate 'Managed/BrassPatch.dll'
$stream=[IO.File]::OpenWrite($candidateHelper);try{$stream.WriteByte(0)}finally{$stream.Dispose()}
Run-Package 'install-patch.ps1' $false
Assert-Candidate
Copy-Item -LiteralPath (Join-Path $build 'Managed/BrassPatch.dll') -Destination $candidateHelper -Force
if($manifest.version -in @('1.3.0','1.3.1','1.3.2','1.4.0','1.4.1','1.4.2','1.4.3','1.4.4','1.4.5','1.4.6','1.4.7')){
 $candidateAi=Join-Path $candidate 'Managed/BrassLegacyAI.dll'
 $stream=[IO.File]::OpenWrite($candidateAi);try{$stream.WriteByte(0)}finally{$stream.Dispose()}
 Run-Package 'install-patch.ps1' $false
 Assert-Candidate
 Copy-Item -LiteralPath (Join-Path $build 'Managed/BrassLegacyAI.dll') -Destination $candidateAi -Force
 # New AI saves must never resume under the old runtime with silently different bots.
 $aiSave=Join-Path $saves 'metadata.storage'
 '{"brasspatch_ai":{"format":"brass-local-ai-save/1.0.0"}}'|Set-Content -LiteralPath $aiSave
 Run-Package 'uninstall-patch.ps1' $false
 Assert-Candidate
 Remove-Item -LiteralPath $aiSave
}
# Pending recovery must block restoring an engine unable to read the journal.
$journal=Join-Path $saves 'brasspatch-save-v1.pending'
'fixture journal'|Set-Content -LiteralPath $journal
Run-Package 'uninstall-patch.ps1' $false
Assert-Candidate
Remove-Item -LiteralPath $journal
Run-Package 'uninstall-patch.ps1' $true
if((Get-FileHash -LiteralPath (Join-Path $managed 'Assembly-CSharp.dll')).Hash -ne $manifest.stockSha256 -or (Test-Path -LiteralPath (Join-Path $managed 'BrassPatch.dll')) -or (Test-Path -LiteralPath (Join-Path $managed 'BrassLegacyAI.dll'))){throw 'Stock restoration was not exact'};$checks++
Run-Package 'install-patch.ps1' $true
Assert-Candidate
# Simulate one DLL already replaced during an interrupted upgrade. The durable
# intent identifies the old and candidate files, so rerunning installation heals it.
$intent=[ordered]@{operation='install';gameRoot=$game;priorFiles=$record.files;candidateFiles=$manifest.files;saveDataPaths=@($saves)}
$intent|ConvertTo-Json -Depth 10|Set-Content -LiteralPath (Join-Path $fixture 'artifacts/patch-installation.pending.json') -Encoding UTF8
Copy-Item -LiteralPath (Join-Path $previous 'Assembly-CSharp.dll') -Destination (Join-Path $managed 'Assembly-CSharp.dll') -Force
Run-Package 'install-patch.ps1' $true
Assert-Candidate
if(Test-Path -LiteralPath (Join-Path $fixture 'artifacts/patch-installation.pending.json')){throw 'Completed upgrade retained pending intent'};$checks++
if($manifest.version -in @('1.3.0','1.3.1','1.3.2','1.4.0','1.4.1','1.4.2','1.4.3','1.4.4','1.4.5','1.4.6','1.4.7')){
 # Simulate process termination after each dependency/assembly publication boundary.
 $order=@('BrassPatch.dll','BrassLegacyAI.dll','Assembly-CSharp.dll')
 foreach($boundary in 0..3){
  Restore-Prior
  $record|ConvertTo-Json -Depth 12|Set-Content -LiteralPath $recordPath -Encoding UTF8
  $intent|ConvertTo-Json -Depth 10|Set-Content -LiteralPath (Join-Path $fixture 'artifacts/patch-installation.pending.json') -Encoding UTF8
  for($i=0;$i -lt $boundary;$i++){Copy-Item -LiteralPath (Join-Path $candidate ('Managed/'+$order[$i])) -Destination (Join-Path $managed $order[$i]) -Force}
  Run-Package 'install-patch.ps1' $true
  Assert-Candidate
 }
 # Fail the final replacement by locking that destination. Already published
 # helpers must roll back to the coherent prior package, leaving the old engine intact.
 Restore-Prior
 $record|ConvertTo-Json -Depth 12|Set-Content -LiteralPath $recordPath -Encoding UTF8
 $locked=[IO.File]::Open((Join-Path $managed 'Assembly-CSharp.dll'),[IO.FileMode]::Open,[IO.FileAccess]::Read,[IO.FileShare]::Read)
 try{Run-Package 'install-patch.ps1' $false}finally{$locked.Dispose()}
 foreach($file in $record.files){if((Get-FileHash -LiteralPath (Join-Path $managed $file.name)).Hash -ne $file.sha256){throw 'Failed installation did not recover the prior release'};$checks++}
 # If rollback could not republish the locked, unchanged engine, its journal must
 # remain recoverable. A retry after unlocking must finish the candidate cleanly.
 Run-Package 'install-patch.ps1' $true
 Assert-Candidate
}
[ordered]@{passed=$true;assertions=$checks;fixture=$fixture;previousRelease=$PreviousRelease;candidateRelease=$manifest.version;files=$manifest.files}|ConvertTo-Json -Depth 8|Tee-Object -FilePath (Join-Path $build 'packaging-results.json')
