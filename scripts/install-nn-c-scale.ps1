param(
 [ValidateSet('Install','Rollback','Recover')][string]$Action='Install',
 [string]$BuildDirectory='artifacts/nn-c-scale/20261109/release-1.4.54',
 [string]$GameRoot=(Split-Path -Parent $PSScriptRoot),
 [switch]$TestInterrupt
)
$ErrorActionPreference='Stop'
. (Join-Path $PSScriptRoot 'patch-install-common.ps1')
$repo=Split-Path -Parent $PSScriptRoot
$game=(Resolve-Path -LiteralPath $GameRoot).Path
$record=Join-Path $game 'artifacts/patch-installation.json'
$intent=Join-Path $game 'artifacts/nn-c-scale.pending.json'
$baseVersion='1.4.53-rust-local';$version='1.4.54-rust-local'
function Inside([string]$Path,[string]$Parent){
 $p=[IO.Path]::GetFullPath($Path)
 if(-not $p.StartsWith([IO.Path]::GetFullPath($Parent).TrimEnd('\')+'\',[StringComparison]::OrdinalIgnoreCase)){throw 'NN c_scale path escapes boundary.'}
 return $p
}
function Verify($Entries,[string]$Directory,[switch]$Exact){
 foreach($entry in $Entries){$p=Inside (Join-Path $Directory $entry.path) $Directory
  if(-not(Test-Path -LiteralPath $p -PathType Leaf) -or (Get-FileHash -LiteralPath $p).Hash -ne $entry.sha256){throw ('Changed file: '+$entry.path)}}
 if($Exact -and @(Get-ChildItem -LiteralPath $Directory -Recurse -File).Count -ne @($Entries).Count){throw 'Unexpected native package member.'}
}
function Restore([string]$Backup){
 $Backup=Inside $Backup (Join-Path $game 'artifacts/patch-recovery')
 $r=Read-BrassJson (Join-Path $Backup 'recovery.json')
 if(-not $r -or $r.gameRoot -ne $game -or $r.operation -ne 'nn-c-scale-1.4.54'){throw 'Recovery identity mismatch.'}
 if((Get-FileHash -LiteralPath (Join-Path $Backup 'patch-installation.json')).Hash -ne $r.receiptSha256){throw 'Recovery receipt changed.'}
 foreach($f in $r.files){
  $target=Inside (Join-Path $game ('BrassNative/'+$f.path)) $game
  if((Get-FileHash -LiteralPath (Inside (Join-Path $Backup $f.path) $Backup)).Hash -ne $f.priorSha256){throw 'Recovery file changed.'}
  $hash=(Get-FileHash -LiteralPath $target).Hash
  if($hash -ne $f.candidateSha256 -and $hash -ne $f.priorSha256){throw 'Unrelated runtime change; recovery stopped.'}
 }
 foreach($f in $r.files){Copy-BrassAtomic (Join-Path $Backup $f.path) (Join-Path $game ('BrassNative/'+$f.path))}
 Verify $r.priorNative (Join-Path $game 'BrassNative') -Exact
 Copy-BrassAtomic (Join-Path $Backup 'patch-installation.json') $record
}
if(Get-Process -Name BrassBirminghamUnity,brass-ai-local -ErrorAction SilentlyContinue){throw 'Close the game and let the native worker retire before patching.'}
$prior=Read-BrassJson $record
if(-not $prior -or $prior.gameRoot -ne $game){throw 'Matching installation receipt required.'}
Assert-BrassNoPendingSaves (Get-BrassSaveDataPaths @($prior))
if($TestInterrupt){
 [void](Inside $game (Join-Path $repo 'artifacts/nn-c-scale/install-tests'))
 if($Action -ne 'Install'){throw 'Interruption injection is only for disposable installation tests.'}
}
if($Action -eq 'Recover'){
 $pending=Read-BrassJson $intent
 if(-not $pending -or $pending.gameRoot -ne $game){throw 'No matching interrupted NN c_scale installation.'}
 Restore $pending.backup;Remove-Item -LiteralPath $intent;Write-Output 'Recovered exact prior policies and receipt.';return
}
if(Test-Path -LiteralPath $intent){throw 'Recover the interrupted NN c_scale installation first.'}
if($Action -eq 'Rollback'){
 if(-not $prior.nnCScaleRecovery -or $prior.version -ne $version){throw 'No NN c_scale rollback receipt.'}
 Restore $prior.nnCScaleRecovery;Write-Output 'Restored exact previous c_scale 0.1 policies and receipt.';return
}
$build=Inside ([IO.Path]::GetFullPath((Join-Path $repo $BuildDirectory))) (Join-Path $repo 'artifacts')
$m=Read-BrassJson (Join-Path $build 'build-manifest.json')
$q=Read-BrassJson (Join-Path $build 'qualification.json')
if(-not $q.passed -or $q.format -ne 'brass-nn-c-scale-release/1.0.0' -or $q.build_manifest_sha256 -ne (Get-FileHash -LiteralPath (Join-Path $build 'build-manifest.json')).Hash -or
 $m.version -ne $version -or $m.baseVersion -ne $baseVersion -or $prior.version -ne $baseVersion -or -not $prior.localReleaseQualified -or
 $prior.stockSha256 -ne $m.stockSha256 -or $m.search_method -ne 'search_v1' -or $m.v205_enabled -or $m.c2 -ne 1){throw 'Qualified c_scale candidate and installed 1.4.53 base required.'}
if($game -eq $repo -and (Get-FileHash -LiteralPath $record).Hash -ne $m.installed_baseline_receipt_sha256){throw 'Installed base receipt changed.'}
Verify $q.files (Join-Path $build 'BrassNative') -Exact
Verify $m.baseNative (Join-Path $game 'BrassNative') -Exact
foreach($p in $m.nativeFiles){
 $old=@($m.baseNative|Where-Object path -eq $p.path)
 if($old.Count -ne 1){throw 'Native inventory differs.'}
 if($old[0].sha256 -ne $p.sha256 -and $p.path -notin $m.changedNative){throw ('Unexpected native change: '+$p.path)}
}
$changed=@($m.nativeFiles|Where-Object {$n=$_;$o=@($m.baseNative|Where-Object path -eq $n.path)[0];$o.sha256 -ne $n.sha256})
if(@($changed).Count -ne 21){throw 'Expected only the 18 policies and 3 build lists to change.'}
$backup=Inside (Join-Path $game ('artifacts/patch-recovery/nn-c-scale-'+[Guid]::NewGuid().ToString('N'))) $game
[void](New-Item -ItemType Directory -Path $backup)
$files=@()
foreach($c in $changed){
 $target=Inside (Join-Path $game ('BrassNative/'+$c.path)) $game
 $copy=Inside (Join-Path $backup $c.path) $backup
 [void](New-Item -ItemType Directory -Path (Split-Path -Parent $copy) -Force)
 Copy-Item -LiteralPath $target -Destination $copy
 $files+=@{path=$c.path;priorSha256=(Get-FileHash -LiteralPath $target).Hash;candidateSha256=$c.sha256}
}
Copy-Item -LiteralPath $record -Destination (Join-Path $backup 'patch-installation.json')
Write-BrassJsonAtomic (Join-Path $backup 'recovery.json') ([ordered]@{gameRoot=$game;operation='nn-c-scale-1.4.54';files=$files;priorNative=$m.baseNative;receiptSha256=(Get-FileHash -LiteralPath $record).Hash})
Write-BrassJsonAtomic $intent ([ordered]@{gameRoot=$game;backup=$backup;operation='nn-c-scale-1.4.54'})
try{
 foreach($f in $files){Copy-BrassAtomic (Join-Path $build ('BrassNative/'+$f.path)) (Join-Path $game ('BrassNative/'+$f.path))}
 if($TestInterrupt){throw 'TEST INTERRUPT: NN c_scale files published'}
 Verify $m.nativeFiles (Join-Path $game 'BrassNative') -Exact
 foreach($item in @{version=$m.version;installedUtc=[DateTime]::UtcNow.ToString('o');nnCScale=$m.c2;nnCScaleRecovery=$backup;
  nnCScaleQualificationSha256=(Get-FileHash -LiteralPath (Join-Path $build 'qualification.json')).Hash;nnCScaleInstalledSmoke='pending'}.GetEnumerator()){
  $prior|Add-Member -NotePropertyName $item.Key -NotePropertyValue $item.Value -Force
 }
 Write-BrassJsonAtomic $record $prior;Remove-Item -LiteralPath $intent
}catch{
 $failure=$_
 if(-not $TestInterrupt){try{Restore $backup;Remove-Item -LiteralPath $intent}catch{Write-Warning ('Recovery pending: '+$_.Exception.Message)}}
 throw $failure
}
Write-Output ('Installed 1.4.54: NN search c_scale 1.0 in all policies. Rollback: '+$backup)
