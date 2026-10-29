param(
 [ValidateSet('Install','Rollback','Recover')][string]$Action='Install',
 [string]$BuildDirectory='artifacts/nn-search-choice/candidate',
 [string]$GameRoot=(Split-Path -Parent $PSScriptRoot),
 [switch]$TestInterrupt
)
$ErrorActionPreference='Stop'
. (Join-Path $PSScriptRoot 'patch-install-common.ps1')
$repo=Split-Path -Parent $PSScriptRoot
$game=(Resolve-Path -LiteralPath $GameRoot).Path
$record=Join-Path $game 'artifacts/patch-installation.json'
$intent=Join-Path $game 'artifacts/nn-search-choice.pending.json'
$changes=@('BrassBirminghamUnity_Data/Managed/BrassLegacyAI.dll','BrassNative/alpha-g890/build.json','BrassNative/alpha-g890/policy-k8.json')
function Inside([string]$Path,[string]$Parent){
 $p=[IO.Path]::GetFullPath($Path)
 if(-not $p.StartsWith([IO.Path]::GetFullPath($Parent).TrimEnd('\')+'\',[StringComparison]::OrdinalIgnoreCase)){throw 'NN selector path escapes boundary.'}
 return $p
}
function Verify($Entries,[string]$Directory,[switch]$Exact){
 foreach($entry in $Entries){$p=Inside (Join-Path $Directory $entry.path) $Directory
  if(-not(Test-Path -LiteralPath $p -PathType Leaf) -or (Get-FileHash -LiteralPath $p).Hash -ne $entry.sha256){throw ('Changed file: '+$entry.path)}}
 if($Exact -and @(Get-ChildItem -LiteralPath $Directory -Recurse -File).Count -ne @($Entries).Count){throw 'Unexpected native package member.'}
}
function ManagedPins($Files){@($Files|ForEach-Object{@{path=$_.name;sha256=$_.sha256}})}
function Restore([string]$Backup){
 $Backup=Inside $Backup (Join-Path $game 'artifacts/patch-recovery')
 $r=Read-BrassJson (Join-Path $Backup 'recovery.json')
 if(-not $r -or $r.gameRoot -ne $game -or (@($r.files.path|Sort-Object)-join '|') -ne (@($changes|Sort-Object)-join '|')){throw 'Recovery identity mismatch.'}
 if((Get-FileHash -LiteralPath (Join-Path $Backup 'patch-installation.json')).Hash -ne $r.receiptSha256){throw 'Recovery receipt changed.'}
 foreach($f in $r.files){
  $target=Inside (Join-Path $game $f.path) $game
  if($f.existed -and (Get-FileHash -LiteralPath (Inside (Join-Path $Backup $f.path) $Backup)).Hash -ne $f.priorSha256){throw 'Recovery file changed.'}
  if(Test-Path -LiteralPath $target){
   $hash=(Get-FileHash -LiteralPath $target).Hash
   if($hash -ne $f.candidateSha256 -and (-not $f.existed -or $hash -ne $f.priorSha256)){throw 'Unrelated runtime change; recovery stopped.'}
  }elseif($f.existed){throw 'Existing runtime file is missing; recovery stopped.'}
 }
 foreach($f in $r.files){
  $target=Inside (Join-Path $game $f.path) $game
  if($f.existed){Copy-BrassAtomic (Join-Path $Backup $f.path) $target}
  elseif(Test-Path -LiteralPath $target){Remove-Item -LiteralPath $target}
 }
 $old=Read-BrassJson (Join-Path $Backup 'patch-installation.json')
 Verify (ManagedPins $old.files) (Join-Path $game 'BrassBirminghamUnity_Data/Managed')
 Verify $r.priorNative (Join-Path $game 'BrassNative') -Exact
 Copy-BrassAtomic (Join-Path $Backup 'patch-installation.json') $record
}
if(Get-Process -Name BrassBirminghamUnity,brass-ai-local -ErrorAction SilentlyContinue){throw 'Close the game and let the native worker retire before patching.'}
$prior=Read-BrassJson $record
if(-not $prior -or $prior.gameRoot -ne $game){throw 'Matching installation receipt required.'}
Assert-BrassNoPendingSaves @($prior.saveDataPaths)
if($TestInterrupt){
 [void](Inside $game (Join-Path $repo 'artifacts/nn-search-choice/upgrade-test'))
 if($Action -ne 'Install'){throw 'Interruption injection is only for disposable installation tests.'}
}
if($Action -eq 'Recover'){
 $pending=Read-BrassJson $intent
 if(-not $pending -or $pending.gameRoot -ne $game){throw 'No matching interrupted NN selector installation.'}
 Restore $pending.backup;Remove-Item -LiteralPath $intent;return
}
if(Test-Path -LiteralPath $intent){throw 'Recover the interrupted NN selector installation first.'}
if($Action -eq 'Rollback'){
 if(-not $prior.nnSearchChoiceRecovery -or $prior.version -ne '1.4.40-rust-local'){throw 'No NN selector rollback receipt.'}
 Restore $prior.nnSearchChoiceRecovery;Write-Output 'Restored exact previous runtime, policy and receipt.';return
}
$build=Inside ([IO.Path]::GetFullPath((Join-Path $repo $BuildDirectory))) (Join-Path $repo 'artifacts')
$m=Read-BrassJson (Join-Path $build 'build-manifest.json')
$q=Read-BrassJson (Join-Path $build 'qualification.json')
if(-not $q.passed -or $q.format -ne 'brass-nn-search-choice/1.0.0' -or $q.build_manifest_sha256 -ne (Get-FileHash -LiteralPath (Join-Path $build 'build-manifest.json')).Hash -or
 $m.version -ne '1.4.40-rust-local' -or $m.baseVersion -ne $prior.version -or -not $prior.localReleaseQualified -or
 $prior.stockSha256 -ne $m.stockSha256 -or $m.search_method -ne 'search_v2_focus_only' -or $m.v205_enabled){throw 'Qualified cache-only NN selector candidate and installed 1.4.39 base required.'}
Verify $q.files $build
Verify (ManagedPins $m.baseFiles) (Join-Path $game 'BrassBirminghamUnity_Data/Managed')
Verify (ManagedPins $prior.files) (Join-Path $game 'BrassBirminghamUnity_Data/Managed')
Verify $m.baseNative (Join-Path $game 'BrassNative') -Exact
Verify (ManagedPins $m.files) (Join-Path $build 'Managed')
Verify $m.nativeFiles (Join-Path $build 'BrassNative') -Exact
if((@($m.files.name|Sort-Object)-join '|') -ne 'Assembly-CSharp.dll|BrassLegacyAI.dll|BrassPatch.dll'){throw 'Unexpected managed file list.'}
foreach($name in 'Assembly-CSharp.dll','BrassPatch.dll'){
 if(($m.files|Where-Object name -eq $name).sha256 -ne ($prior.files|Where-Object name -eq $name).sha256){throw 'Unrelated managed binary change.'}
}
foreach($p in $m.nativeFiles){
 $old=@($m.baseNative|Where-Object path -eq $p.path)
 if(($old.Count -ne 1 -or $old[0].sha256 -ne $p.sha256) -and $p.path -notin @('alpha-g890/build.json','alpha-g890/policy-k8.json')){throw 'Unrelated native binary/model/policy change.'}
}
if(@($m.nativeFiles).Count -ne @($m.baseNative).Count+1){throw 'Expected only the additional K8 policy.'}
$backup=Inside (Join-Path $game ('artifacts/patch-recovery/nn-search-choice-'+[Guid]::NewGuid().ToString('N'))) $game
[void](New-Item -ItemType Directory -Path $backup)
$files=@()
foreach($path in $changes){
 $target=Inside (Join-Path $game $path) $game
 $source=if($path -like 'BrassBirminghamUnity_Data/Managed/*'){Join-Path $build ('Managed/'+(Split-Path -Leaf $path))}else{Join-Path $build $path}
 $exists=Test-Path -LiteralPath $target
 $oldHash=$null
 if($exists){
  $oldHash=(Get-FileHash -LiteralPath $target).Hash
  $copy=Inside (Join-Path $backup $path) $backup
  [void](New-Item -ItemType Directory -Path (Split-Path -Parent $copy) -Force)
  Copy-Item -LiteralPath $target -Destination $copy
 }
 $files+=@{path=$path;existed=$exists;priorSha256=$oldHash;candidateSha256=(Get-FileHash -LiteralPath $source).Hash;source=$source}
}
Copy-Item -LiteralPath $record -Destination (Join-Path $backup 'patch-installation.json')
Write-BrassJsonAtomic (Join-Path $backup 'recovery.json') ([ordered]@{gameRoot=$game;files=$files;priorNative=$m.baseNative;receiptSha256=(Get-FileHash -LiteralPath $record).Hash})
Write-BrassJsonAtomic $intent ([ordered]@{gameRoot=$game;backup=$backup;operation='nn-search-choice'})
try{
 foreach($f in $files){Copy-BrassAtomic $f.source (Join-Path $game $f.path)}
 if($TestInterrupt){throw 'TEST INTERRUPT: NN selector files published'}
 Verify (ManagedPins $m.files) (Join-Path $game 'BrassBirminghamUnity_Data/Managed')
 Verify $m.nativeFiles (Join-Path $game 'BrassNative') -Exact
 foreach($item in @{version=$m.version;files=$m.files;installedUtc=[DateTime]::UtcNow.ToString('o');nativeBuildSha256=$m.native_build_sha256;
  nnSearchKChoices=@(8,16);nnSearchKDefault=16;nnSearchChoiceRecovery=$backup;nnSearchChoiceQualificationSha256=(Get-FileHash -LiteralPath (Join-Path $build 'qualification.json')).Hash;
  nnSearchChoiceInstalledSmoke='pending';searchV2InstalledSmoke='pending/nn-search-choice'}.GetEnumerator()){
  $prior|Add-Member -NotePropertyName $item.Key -NotePropertyValue $item.Value -Force
 }
 Write-BrassJsonAtomic $record $prior;Remove-Item -LiteralPath $intent
}catch{
 $failure=$_
 if(-not $TestInterrupt){try{Restore $backup;Remove-Item -LiteralPath $intent}catch{Write-Warning ('Recovery pending: '+$_.Exception.Message)}}
 throw $failure
}
Write-Output ('Installed 1.4.40: NN search K8/K16 selector. Rollback: '+$backup)
