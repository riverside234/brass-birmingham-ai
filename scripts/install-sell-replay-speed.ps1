param([ValidateSet('Install','Rollback','Recover')][string]$Action='Install',
 [string]$GameRoot=(Split-Path -Parent $PSScriptRoot),[string]$Backup,[switch]$TestInterrupt)
$ErrorActionPreference='Stop'
. (Join-Path $PSScriptRoot 'patch-install-common.ps1')
$repo=Split-Path -Parent $PSScriptRoot
$game=[IO.Path]::GetFullPath($GameRoot).TrimEnd('\')
$build=Join-Path $repo 'artifacts/sell-replay-speed/20261101/release-1.4.49'
$record=Join-Path $game 'artifacts/patch-installation.json'
$intent=Join-Path $game 'artifacts/sell-replay-speed-install.pending'
$names=@('Assembly-CSharp.dll','BrassPatch.dll','BrassLegacyAI.dll')
function Verify($pins,[string]$directory,[string]$member='name'){
 foreach($p in $pins){if((Get-FileHash -LiteralPath (Join-Path $directory $p.$member)).Hash -ne $p.sha256){throw ('Runtime changed: '+$p.$member)}}
}
function SafeBackup([string]$path){
 $resolved=[IO.Path]::GetFullPath($path)
 $boundary=[IO.Path]::GetFullPath((Join-Path $game 'artifacts/patch-recovery'))+'\'
 if(-not $resolved.StartsWith($boundary,[StringComparison]::OrdinalIgnoreCase)){throw 'Recovery escaped the game recovery directory.'}
 return $resolved
}
function Restore([string]$directory){
 $directory=SafeBackup $directory
 $r=Read-BrassJson (Join-Path $directory 'recovery.json')
 if(-not $r -or $r.gameRoot -ne $game -or $r.operation -ne 'sell-replay-speed-1.4.49' -or
  (Get-FileHash -LiteralPath (Join-Path $directory 'patch-installation.json')).Hash -ne $r.receiptSha256){throw 'Recovery identity changed.'}
 foreach($f in $r.files){
  if($f.name -notin @('Assembly-CSharp.dll','BrassPatch.dll')){throw 'Unexpected recovery member.'}
  if((Get-FileHash -LiteralPath (Join-Path $directory $f.name)).Hash -ne $f.priorSha256){throw 'Recovery backup changed.'}
  $current=(Get-FileHash -LiteralPath (Join-Path $game ('BrassBirminghamUnity_Data/Managed/'+$f.name))).Hash
  if($current -ne $f.priorSha256 -and $current -ne $f.candidateSha256){throw 'Unrelated runtime change; recovery stopped.'}
 }
 foreach($f in $r.files){Copy-BrassAtomic (Join-Path $directory $f.name) (Join-Path $game ('BrassBirminghamUnity_Data/Managed/'+$f.name))}
 Copy-BrassAtomic (Join-Path $directory 'patch-installation.json') $record
 if(Test-Path -LiteralPath $intent){Remove-Item -LiteralPath $intent}
}
if(Get-Process -Name BrassBirminghamUnity,brass-ai-local -ErrorAction SilentlyContinue){throw 'Close the game before replacing its animation patch.'}
$prior=Read-BrassJson $record
if(-not $prior -or $prior.gameRoot -ne $game){throw 'Matching installation receipt required.'}
Assert-BrassNoPendingSaves (Get-BrassSaveDataPaths @($prior))
if($Action -eq 'Recover'){
 $pending=Read-BrassJson $intent
 if(-not $pending -or $pending.gameRoot -ne $game){throw 'No matching pending operation.'}
 Restore $pending.backup;Write-Output 'Recovered exact prior runtime and receipt.';return
}
if(Test-Path -LiteralPath $intent){throw 'Recover the interrupted animation installation first.'}
if($Action -eq 'Rollback'){
 if($prior.version -ne '1.4.49-rust-local' -or -not $prior.sellReplaySpeedRecovery){throw 'No matching animation rollback receipt.'}
 Restore $prior.sellReplaySpeedRecovery;Write-Output 'Restored exact 1.4.48 runtime and receipt.';return
}
$m=Read-BrassJson (Join-Path $build 'build-manifest.json');$q=Read-BrassJson (Join-Path $build 'qualification.json')
if(-not $q.passed -or $q.format -ne 'brass-sell-replay-speed/1.0.0' -or $m.version -ne '1.4.49-rust-local' -or
 $m.baseVersion -ne '1.4.48-rust-local' -or $prior.version -ne $m.baseVersion -or
 $q.build_manifest_sha256 -ne (Get-FileHash -LiteralPath (Join-Path $build 'build-manifest.json')).Hash){throw 'Qualified replay-speed candidate and exact installed base required.'}
if($game -eq $repo -and (Get-FileHash -LiteralPath $record).Hash -ne $m.installed_baseline_receipt_sha256){throw 'Installed base receipt changed.'}
if((@($m.files.name|Sort-Object)-join '|') -ne (@($names|Sort-Object)-join '|')){throw 'Unexpected managed package.'}
Verify $m.files (Join-Path $build 'Managed')
Verify $m.baseFiles (Join-Path $game 'BrassBirminghamUnity_Data/Managed')
Verify $m.nativeFiles (Join-Path $game 'BrassNative') 'path'
if(@(Get-ChildItem -LiteralPath (Join-Path $game 'BrassNative') -Recurse -File).Count -ne @($m.nativeFiles).Count){throw 'Native package inventory changed.'}
if(($m.files|Where-Object name -eq 'BrassLegacyAI.dll').sha256 -ne ($m.baseFiles|Where-Object name -eq 'BrassLegacyAI.dll').sha256){throw 'Animation patch cannot change the AI.'}
foreach($p in $m.sourceScope){if((Get-FileHash -LiteralPath (Join-Path $repo $p)).Hash -ne ($m.sourcePins|Where-Object path -eq $p).sha256){throw 'Animation source changed after qualification.'}}
$backup=SafeBackup (Join-Path $game ('artifacts/patch-recovery/sell-speed-'+[Guid]::NewGuid().ToString('N')))
[void](New-Item -ItemType Directory -Path $backup)
$files=@()
foreach($name in @('Assembly-CSharp.dll','BrassPatch.dll')){
 Copy-Item -LiteralPath (Join-Path $game ('BrassBirminghamUnity_Data/Managed/'+$name)) -Destination (Join-Path $backup $name)
 $files+=@{name=$name;priorSha256=($m.baseFiles|Where-Object name -eq $name).sha256;candidateSha256=($m.files|Where-Object name -eq $name).sha256}
}
Copy-Item -LiteralPath $record -Destination (Join-Path $backup 'patch-installation.json')
Write-BrassJsonAtomic (Join-Path $backup 'recovery.json') @{gameRoot=$game;operation='sell-replay-speed-1.4.49';files=$files;receiptSha256=(Get-FileHash -LiteralPath $record).Hash}
Write-BrassJsonAtomic $intent @{gameRoot=$game;backup=$backup;operation='sell-replay-speed-1.4.49'}
try{
 foreach($f in $files){Copy-BrassAtomic (Join-Path $build ('Managed/'+$f.name)) (Join-Path $game ('BrassBirminghamUnity_Data/Managed/'+$f.name))}
 if($TestInterrupt){throw 'TEST INTERRUPT: replay-speed files published'}
 Verify $m.files (Join-Path $game 'BrassBirminghamUnity_Data/Managed')
 foreach($entry in @{version=$m.version;files=$m.files;installedUtc=[DateTime]::UtcNow.ToString('o');sellReplaySpeedRecovery=$backup;
  sellReplaySpeedQualificationSha256=(Get-FileHash -LiteralPath (Join-Path $build 'qualification.json')).Hash;
  sellReplaySpeedCandidateSha256=(Get-FileHash -LiteralPath (Join-Path $build 'build-manifest.json')).Hash;sellReplaySpeedInstalledSmoke='pending'}.GetEnumerator()){
  $prior|Add-Member -NotePropertyName $entry.Key -NotePropertyValue $entry.Value -Force
 }
 Write-BrassJsonAtomic $record $prior;Remove-Item -LiteralPath $intent
}catch{
 $failure=$_;if(-not $TestInterrupt){try{Restore $backup}catch{Write-Warning ('Recovery pending: '+$_.Exception.Message)}};throw $failure
}
Write-Output ('Installed 1.4.49: sell replay double-click speed fix. Recovery backup: '+$backup)
