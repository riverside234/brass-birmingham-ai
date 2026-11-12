param(
 [ValidateSet('Install','Rollback','Recover')][string]$Action='Install',
 [string]$GameRoot=(Split-Path -Parent $PSScriptRoot)
)
$ErrorActionPreference='Stop'
. (Join-Path $PSScriptRoot 'patch-install-common.ps1')
$repo=Split-Path -Parent $PSScriptRoot
$study=Join-Path $repo 'artifacts/nn-opening-temperature'
$build=Join-Path $study 'release-1.4.58'
$game=(Resolve-Path -LiteralPath $GameRoot).Path
$record=Join-Path $game 'artifacts/patch-installation.json'
$intent=Join-Path $game 'artifacts/nn-opening-temperature.pending.json'
$backupRoot=Join-Path $game 'artifacts/patch-recovery'
$openingVersion='1.4.57-rust-local'
$version='1.4.58-rust-local'
$managed='BrassBirminghamUnity_Data/Managed'
$labelPaths=@("$managed/BrassLegacyAI.dll")
$changedPaths=$labelPaths
foreach($model in 'alpha-g890','beta-g1290','beta2-g1190','il-v11-2'){
 foreach($name in 'brass-ai-local.exe','native-source.zip','build.json'){$changedPaths+="BrassNative/$model/$name"}
}

function Inside([string]$Path,[string]$Parent){
 $full=[IO.Path]::GetFullPath($Path)
 $boundary=[IO.Path]::GetFullPath($Parent).TrimEnd('\')+'\'
 if(-not $full.StartsWith($boundary,[StringComparison]::OrdinalIgnoreCase)){throw 'Opening-temperature path escaped its directory.'}
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
function Restore([string]$Backup){
 $backup=Inside $Backup $backupRoot
 $recovery=Read-BrassJson (Join-Path $backup 'recovery.json')
 $savedReceipt=Join-Path $backup 'patch-installation.json'
 if(-not $recovery -or $recovery.gameRoot -ne $game -or $recovery.version -ne $version -or (Hash $savedReceipt) -ne $recovery.receiptSha256){throw 'Recovery identity changed.'}
 foreach($f in $recovery.files){
  if($f.path -notin $changedPaths -or (Hash (Inside (Join-Path $backup $f.path) $backup)) -ne $f.old){throw 'Recovery file changed.'}
  if((Hash (Inside (Join-Path $game $f.path) $game)) -notin @($f.old,$f.new)){throw ('Unrelated runtime change: '+$f.path)}
 }
 foreach($f in $recovery.files){Copy-BrassAtomic (Join-Path $backup $f.path) (Join-Path $game $f.path)}
 $saved=Read-BrassJson $savedReceipt
 Verify (RuntimePins $saved.files $recovery.native) $game
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
 if(-not $pending -or $pending.gameRoot -ne $game){throw 'No matching interrupted opening-temperature installation.'}
 $restored=Restore $pending.backup
 Remove-Item -LiteralPath $intent
 Write-Output "Recovered $restored."
 return
}
if(Test-Path -LiteralPath $intent){throw 'Recover the pending opening-temperature installation first.'}
if($Action -eq 'Rollback'){
 if($prior.version -ne $version -or -not $prior.nnOpeningTemperatureRecovery){throw 'No matching opening-temperature rollback receipt.'}
 $restored=Restore $prior.nnOpeningTemperatureRecovery
 Write-Output "Restored $restored."
 return
}

$m=Read-BrassJson (Join-Path $build 'build-manifest.json')
$q=Read-BrassJson (Join-Path $build 'qualification.json')
if(-not $q.passed -or $q.format -ne 'brass-nn-opening-temperature-release/1.0.0' -or $q.manifest_sha256 -ne (Hash (Join-Path $build 'build-manifest.json')) -or
 $prior.version -notin @('1.4.56-rust-local',$openingVersion) -or $m.baseVersion -ne $prior.version -or $m.version -ne $version -or $prior.stockSha256 -ne $m.stockSha256){throw 'Qualified opening-temperature candidate and exact installed base required.'}
if($game -eq $repo -and (Hash $record) -ne $m.installedBaselineReceiptSha256){throw 'Installed baseline receipt changed.'}
Verify $m.sourcePins $repo
Verify (RuntimePins $m.baseFiles $m.baseNative) $game
$old=@{}
foreach($pin in (RuntimePins $m.baseFiles $m.baseNative)){$old[$pin.path]=$pin.sha256}
$candidate=RuntimePins $m.files $m.nativeFiles
foreach($pin in $candidate){
 $source=Inside (Join-Path $build ($pin.path -replace '^BrassBirminghamUnity_Data/','')) $build
 if((Hash $source) -ne $pin.sha256){throw ('Candidate changed: '+$pin.path)}
}
$changes=@(foreach($pin in $candidate){
 if(-not $old.ContainsKey($pin.path)){throw ('Unexpected runtime member: '+$pin.path)}
 if($old[$pin.path] -ne $pin.sha256){@{path=$pin.path;old=$old[$pin.path];new=$pin.sha256}}
})
$expectedPaths=if($prior.version -eq $openingVersion){$labelPaths}else{$changedPaths}
if((($changes.path|Sort-Object) -join '|') -ne (($expectedPaths|Sort-Object) -join '|')){throw 'Unexpected opening-temperature patch scope.'}
$backup=Inside (Join-Path $backupRoot ('nn-opening-temperature-'+[Guid]::NewGuid().ToString('N'))) $game
foreach($f in $changes){
 $saved=Inside (Join-Path $backup $f.path) $backup
 [void](New-Item -ItemType Directory -Path (Split-Path -Parent $saved) -Force)
 Copy-Item -LiteralPath (Join-Path $game $f.path) -Destination $saved
}
Copy-Item -LiteralPath $record -Destination (Join-Path $backup 'patch-installation.json')
Write-BrassJsonAtomic (Join-Path $backup 'recovery.json') @{gameRoot=$game;version=$version;files=$changes;native=$m.baseNative;receiptSha256=(Hash $record)}
Write-BrassJsonAtomic $intent @{gameRoot=$game;backup=$backup;version=$version}
try{
 foreach($f in $changes){Copy-BrassAtomic (Join-Path $build ($f.path -replace '^BrassBirminghamUnity_Data/','')) (Join-Path $game $f.path)}
 Verify $candidate $game
 foreach($entry in @{version=$version;files=$m.files;installedUtc=[DateTime]::UtcNow.ToString('o');nnOpeningTemperatureRecovery=$backup;
  nnOpeningTemperatureChoices=$m.openingTemperatureChoices;nnOpeningMovesPerSeat=$m.openingMovesPerSeat;nnTemperatureDefault=$m.regularTemperature;
  nnOpeningTemperatureQualificationSha256=(Hash (Join-Path $build 'qualification.json'));nnOpeningTemperatureNativeSourceSha256=$m.nativeSourceSha256}.GetEnumerator()){
  $prior|Add-Member -NotePropertyName $entry.Key -NotePropertyValue $entry.Value -Force
 }
 Write-BrassJsonAtomic $record $prior
 Remove-Item -LiteralPath $intent
}catch{
 $failure=$_
 try{[void](Restore $backup);Remove-Item -LiteralPath $intent}catch{Write-Warning ('Recovery pending: '+$_.Exception.Message)}
 throw $failure
}
Write-Output "Installed $version. Recovery backup: $backup"
