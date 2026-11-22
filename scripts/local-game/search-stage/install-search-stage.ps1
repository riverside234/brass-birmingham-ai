param(
 [ValidateSet('Install','Rollback','Recover')][string]$Action='Install',
 [string]$GameRoot=(Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot)))
)
# 1.4.63: Beta and Beta-Plus with the search_v1 startup estimate by era stage; Alpha Test and IL retired.
# Native files only. The retired packages move into the recovery backup, so Rollback and Recover restore them.
$ErrorActionPreference='Stop'
. (Join-Path $PSScriptRoot '../../patch-install-common.ps1')
$repo=Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
$study=Join-Path $repo 'artifacts/search-stage'
$build=Join-Path $study 'release-1.4.63'
$game=(Resolve-Path -LiteralPath $GameRoot).Path
$record=Join-Path $game 'artifacts/patch-installation.json'
$intent=Join-Path $game 'artifacts/search-stage.pending.json'
$backupRoot=Join-Path $game 'artifacts/patch-recovery'
$baseVersion='1.4.62-rust-local'
$version='1.4.63-rust-local'
$managed='BrassBirminghamUnity_Data/Managed'
$retired=@('alpha-g890','il-v11-2')
$changedPaths=@()
foreach($model in 'beta-g1290','beta2-g1190'){
 foreach($name in 'brass-ai-local.exe','native-source.zip','build.json','policy.json','policy-k8.json','policy-k32.json','policy-t15.json','policy-k8-t15.json','policy-k32-t15.json'){$changedPaths+="BrassNative/$model/$name"}
}

function Inside([string]$Path,[string]$Parent){
 $full=[IO.Path]::GetFullPath($Path)
 $boundary=[IO.Path]::GetFullPath($Parent).TrimEnd('\')+'\'
 if(-not $full.StartsWith($boundary,[StringComparison]::OrdinalIgnoreCase)){throw 'Search-stage path escaped its directory.'}
 for($node=$full;$node.Length -ge $boundary.Length;$node=[IO.Path]::GetDirectoryName($node)){
  if((Test-Path -LiteralPath $node) -and ([IO.File]::GetAttributes($node) -band [IO.FileAttributes]::ReparsePoint)){throw 'Linked installation path refused.'}
 }
 return $full
}
function Hash([string]$Path){(Get-FileHash -LiteralPath $Path).Hash}
function RuntimePins($Files,$Native){
 foreach($f in $Files){@{path="$managed/$($f.name)";sha256=$f.sha256}}
 foreach($f in $Native){@{path="BrassNative/$($f.path)";sha256=$f.sha256}}
}
function Verify($Pins,[string]$Root){
 foreach($pin in $Pins){if((Hash (Inside (Join-Path $Root $pin.path) $Root)) -ne $pin.sha256){throw ('File changed: '+$pin.path)}}
}
function VerifyNativeExact($Native){
 # Exactly these package files under BrassNative, nothing more.
 $root=Join-Path $game 'BrassNative'
 Verify @(foreach($f in $Native){@{path=$f.path;sha256=$f.sha256}}) $root
 $count=@(Get-ChildItem -LiteralPath $root -Recurse -File).Count
 if($count -ne @($Native).Count){throw 'Unexpected native package file.'}
}
function CopyInto([string]$Source,[string]$Destination){
 [void](New-Item -ItemType Directory -Path (Split-Path -Parent $Destination) -Force)
 Copy-BrassAtomic $Source $Destination
}
function Restore([string]$Backup){
 $backup=Inside $Backup $backupRoot
 $recovery=Read-BrassJson (Join-Path $backup 'recovery.json')
 $savedReceipt=Join-Path $backup 'patch-installation.json'
 if(-not $recovery -or $recovery.gameRoot -ne $game -or $recovery.version -ne $version -or (Hash $savedReceipt) -ne $recovery.receiptSha256){throw 'Recovery identity changed.'}
 foreach($f in @($recovery.files)+@($recovery.removed)){
  if((Hash (Inside (Join-Path $backup $f.path) $backup)) -ne $f.old){throw 'Recovery file changed.'}
 }
 foreach($f in $recovery.files){
  if($f.path -notin $changedPaths){throw 'Recovery scope changed.'}
  $live=Inside (Join-Path $game $f.path) $game
  if((Hash $live) -notin @($f.old,$f.new)){throw ('Unrelated runtime change: '+$f.path)}
 }
 foreach($f in $recovery.removed){
  if(($f.path -split '/')[1] -notin $retired){throw 'Recovery scope changed.'}
  $live=Inside (Join-Path $game $f.path) $game
  if((Test-Path -LiteralPath $live) -and (Hash $live) -ne $f.old){throw ('Unrelated runtime change: '+$f.path)}
 }
 foreach($f in @($recovery.files)+@($recovery.removed)){CopyInto (Join-Path $backup $f.path) (Join-Path $game $f.path)}
 $saved=Read-BrassJson $savedReceipt
 Verify (RuntimePins $saved.files @()) $game
 VerifyNativeExact $recovery.native
 Copy-BrassAtomic $savedReceipt $record
 return $saved.version
}

if($game -ne $repo){[void](Inside $game $study)}
if(Get-Process -Name BrassBirminghamUnity,brass-ai-local -ErrorAction SilentlyContinue){throw 'Close the game and let the native worker retire before patching.'}
$prior=Read-BrassJson $record
if(-not $prior -or $prior.gameRoot -ne $game){throw 'Matching installation receipt required.'}
Assert-BrassNoPendingSaves (Get-BrassSaveDataPaths @($prior))
if($Action -eq 'Recover'){
 $pending=Read-BrassJson $intent
 if(-not $pending -or $pending.gameRoot -ne $game){throw 'No matching interrupted search-stage installation.'}
 $restored=Restore $pending.backup
 Remove-Item -LiteralPath $intent
 Write-Output "Recovered $restored."
 return
}
if(Test-Path -LiteralPath $intent){throw 'Recover the pending search-stage installation first.'}
if($Action -eq 'Rollback'){
 if($prior.version -ne $version -or -not $prior.searchStage -or -not $prior.searchStage.recovery){throw 'No matching search-stage rollback receipt.'}
 $restored=Restore $prior.searchStage.recovery
 Write-Output "Restored $restored (Alpha Test and IL packages restored)."
 return
}

$m=Read-BrassJson (Join-Path $build 'build-manifest.json')
$q=Read-BrassJson (Join-Path $build 'qualification.json')
if(-not $q.passed -or $q.format -ne 'brass-search-stage-release/1.0.0' -or $q.manifest_sha256 -ne (Hash (Join-Path $build 'build-manifest.json')) -or
 $prior.version -ne $baseVersion -or $m.baseVersion -ne $prior.version -or $m.version -ne $version -or $prior.stockSha256 -ne $m.stockSha256){throw 'Qualified search-stage candidate and exact installed base required.'}
if($game -eq $repo -and (Hash $record) -ne $m.installedBaselineReceiptSha256){throw 'Installed baseline receipt changed.'}
Verify $m.sourcePins $repo
Verify (RuntimePins $m.files @()) $game
VerifyNativeExact $m.baseNative
$old=@{}
foreach($pin in $m.baseNative){$old["BrassNative/$($pin.path)"]=$pin.sha256}
$candidate=@(foreach($pin in $m.nativeFiles){@{path="BrassNative/$($pin.path)";sha256=$pin.sha256}})
foreach($pin in $candidate){if((Hash (Inside (Join-Path $build $pin.path) $build)) -ne $pin.sha256){throw ('Candidate changed: '+$pin.path)}}
$changes=@(foreach($pin in $candidate){
 if(-not $old.ContainsKey($pin.path)){throw ('Unexpected runtime member: '+$pin.path)}
 if($old[$pin.path] -ne $pin.sha256){@{path=$pin.path;old=$old[$pin.path];new=$pin.sha256}}
})
if((($changes.path|Sort-Object) -join '|') -ne (($changedPaths|Sort-Object) -join '|')){throw 'Unexpected search-stage patch scope.'}
$kept=@{};foreach($pin in $candidate){$kept[$pin.path]=$true}
$removed=@(foreach($pin in $m.baseNative){$path="BrassNative/$($pin.path)";if(-not $kept.ContainsKey($path)){@{path=$path;old=$pin.sha256}}})
foreach($f in $removed){if(($f.path -split '/')[1] -notin $retired){throw ('Unexpected removal: '+$f.path)}}
if(@($removed).Count -ne @($m.removedNative).Count){throw 'Unexpected search-stage removal scope.'}
$backup=Inside (Join-Path $backupRoot ('search-stage-'+[Guid]::NewGuid().ToString('N'))) $game
foreach($f in @($changes)+@($removed)){
 $saved=Inside (Join-Path $backup $f.path) $backup
 [void](New-Item -ItemType Directory -Path (Split-Path -Parent $saved) -Force)
 Copy-Item -LiteralPath (Join-Path $game $f.path) -Destination $saved
}
Copy-Item -LiteralPath $record -Destination (Join-Path $backup 'patch-installation.json')
Write-BrassJsonAtomic (Join-Path $backup 'recovery.json') @{gameRoot=$game;version=$version;files=$changes;removed=$removed;native=$m.baseNative;receiptSha256=(Hash $record)}
Write-BrassJsonAtomic $intent @{gameRoot=$game;backup=$backup;version=$version}
try{
 foreach($f in $changes){Copy-BrassAtomic (Join-Path $build $f.path) (Join-Path $game $f.path)}
 foreach($f in $removed){Remove-Item -LiteralPath (Inside (Join-Path $game $f.path) $game)}
 foreach($model in $retired){
  $directory=Inside (Join-Path $game "BrassNative/$model") $game
  if(Test-Path -LiteralPath $directory){
   if(@(Get-ChildItem -LiteralPath $directory -Recurse -File).Count -ne 0){throw ('Retired package still has files: '+$model)}
   Remove-Item -LiteralPath $directory -Recurse
  }
 }
 VerifyNativeExact $m.nativeFiles
 $stage=@{installedUtc=[DateTime]::UtcNow.ToString('o');recovery=$backup;startupNsByStage=$m.startupNsByStage;startupNs=$m.startupNs;
  previousStartupNs=$m.previousStartupNs;retiredModels=$retired;calibrationSha256=$m.calibrationSha256;
  qualificationSha256=(Hash (Join-Path $build 'qualification.json'));nativeFiles=$changes;removedNative=$removed;status='installed'}
 foreach($entry in @{version=$version;installedUtc=$stage.installedUtc;searchStage=$stage}.GetEnumerator()){
  $prior|Add-Member -NotePropertyName $entry.Key -NotePropertyValue $entry.Value -Force
 }
 Write-BrassJsonAtomic $record $prior
 Remove-Item -LiteralPath $intent
}catch{
 $failure=$_
 try{[void](Restore $backup);Remove-Item -LiteralPath $intent}catch{Write-Warning ('Recovery pending: '+$_.Exception.Message)}
 throw $failure
}
Write-Output "Installed $version (Alpha Test and IL retired). Recovery backup: $backup"
