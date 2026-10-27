param(
 [ValidateSet('Install','Rollback','Recover')][string]$Action='Install',
 [string]$BuildDirectory='artifacts/rust-backend/20261026-l13-qualified',
 [string]$GameRoot=(Split-Path -Parent $PSScriptRoot)
)
$ErrorActionPreference='Stop'
. (Join-Path $PSScriptRoot 'patch-install-common.ps1')
$alphaRepo=Split-Path -Parent $PSScriptRoot
$alphaGame=(Resolve-Path -LiteralPath $GameRoot).Path
$alphaManaged=Join-Path $alphaGame 'BrassBirminghamUnity_Data/Managed'
$alphaRecord=Join-Path $alphaGame 'artifacts/patch-installation.json'
$alphaIntent=Join-Path $alphaGame 'artifacts/rust-alpha-installation.pending.json'
$alphaNative=Join-Path $alphaGame 'BrassNative'
$alphaPrevious=Read-BrassJson $alphaRecord
if(Get-Process -Name BrassBirminghamUnity -ErrorAction SilentlyContinue){throw 'Close the game before changing its local runtime.'}
if(Get-Process -Name brass-ai-local -ErrorAction SilentlyContinue){throw 'A native worker is still retiring; retry after it exits.'}
if(-not $alphaPrevious -or $alphaPrevious.gameRoot -ne $alphaGame){throw 'A matching installation receipt is required.'}
Assert-BrassNoPendingSaves @($alphaPrevious.saveDataPaths)
function Assert-AlphaInside([string]$Path,[string]$Parent){
 $resolved=[IO.Path]::GetFullPath($Path)
 if(-not $resolved.StartsWith([IO.Path]::GetFullPath($Parent).TrimEnd('\')+'\',[StringComparison]::OrdinalIgnoreCase)){throw 'Recovery path escapes its game directory.'}
 return $resolved
}
function Assert-AlphaFiles($Entries,[string]$Directory){
 foreach($entry in $Entries){
  $path=Assert-AlphaInside (Join-Path $Directory $entry.name) $Directory
  if((Get-FileHash -LiteralPath $path).Hash -ne $entry.sha256){throw ('File changed: '+$entry.name)}
 }
}
function Restore-AlphaBackup([string]$Backup){
 $recovery=Read-BrassJson (Join-Path $Backup 'recovery.json')
 Assert-AlphaFiles $recovery.files (Join-Path $Backup 'Managed')
 if((Get-FileHash -LiteralPath (Join-Path $Backup 'patch-installation.json')).Hash -ne $recovery.recordSha256){throw 'Recovery record changed.'}
 foreach($file in $recovery.files){Copy-BrassAtomic (Join-Path $Backup ('Managed/'+$file.name)) (Join-Path $alphaManaged $file.name)}
 Assert-AlphaFiles $recovery.files $alphaManaged
 Copy-BrassAtomic (Join-Path $Backup 'patch-installation.json') $alphaRecord
 if(Test-Path -LiteralPath $alphaNative){
  $from=Assert-AlphaInside $alphaNative $alphaGame
  $to=Assert-AlphaInside (Join-Path $Backup ('retired-native-'+[Guid]::NewGuid().ToString('N'))) $alphaGame
  Move-Item -LiteralPath $from -Destination $to
 }
}
if($Action -eq 'Recover'){
 $intent=Read-BrassJson $alphaIntent
 if(-not $intent -or $intent.gameRoot -ne $alphaGame){throw 'No matching interrupted Alpha Test installation.'}
 foreach($file in $intent.priorFiles){
  $hash=(Get-FileHash -LiteralPath (Join-Path $alphaManaged $file.name)).Hash
  if($hash -notin @(Get-BrassHashes @($intent) $file.name)){throw 'A recovery target has unrecognized changes.'}
 }
 $backup=Assert-AlphaInside $intent.backup (Join-Path $alphaGame 'artifacts/patch-recovery')
 Restore-AlphaBackup $backup;Remove-Item -LiteralPath $alphaIntent
 Write-Output 'Recovered the interrupted Alpha Test installation to its exact prior files.'
 return
}
if($Action -eq 'Rollback'){
 if($alphaPrevious.version -ne '1.4.35-alpha-test' -or -not $alphaPrevious.nativeAlphaRecovery){throw 'No installed Alpha Test recovery receipt.'}
 Assert-AlphaFiles $alphaPrevious.files $alphaManaged
 $backup=Assert-AlphaInside $alphaPrevious.nativeAlphaRecovery (Join-Path $alphaGame 'artifacts/patch-recovery')
 Restore-AlphaBackup $backup
 Write-Output 'Restored the exact prior installation. NN saves remain untouched; use Alpha Test to resume them.'
 return
}
$alphaBuild=[IO.Path]::GetFullPath((Join-Path $alphaRepo $BuildDirectory))
$q=Read-BrassJson (Join-Path $alphaBuild 'qualification.json')
$audit=Read-BrassJson (Join-Path $alphaBuild 'final-verification.json')
$manifest=Read-BrassJson (Join-Path $alphaBuild 'build-manifest.json')
if(-not $q.passed -or $q.format -ne 'brass-local-l13-qualification/1.0.0' -or $q.scope -ne 'local-alpha-test-integration' -or
   -not $audit.passed -or $audit.qualification_sha256 -ne (Get-FileHash -LiteralPath (Join-Path $alphaBuild 'qualification.json')).Hash -or
   $manifest.version -ne '1.4.35-alpha-test' -or $manifest.local_release_qualified -ne $false){throw 'Sealed L13 Alpha Test integration qualification is required.'}
foreach($pin in $q.files){
 $path=Assert-AlphaInside (Join-Path $alphaBuild $pin.path) $alphaBuild
 if((Get-FileHash -LiteralPath $path).Hash -ne $pin.sha256){throw ('Qualified evidence changed: '+$pin.path)}
}
if((Get-FileHash -LiteralPath (Join-Path $alphaBuild 'build-manifest.json')).Hash -ne $q.build_manifest_sha256){throw 'Build manifest changed.'}
if($alphaPrevious.version -ne $manifest.baseVersion -or $alphaPrevious.stockSha256 -ne $manifest.stockSha256){throw 'Alpha Test requires the qualified 1.4.34 installation.'}
Assert-AlphaFiles $alphaPrevious.files $alphaManaged
Assert-AlphaFiles $manifest.files (Join-Path $alphaBuild 'Managed')
if(Test-Path -LiteralPath $alphaNative){throw 'An existing native package was retained. Resolve it before installing a different package.'}
if(Test-Path -LiteralPath $alphaIntent){throw 'An unfinished installation intent was retained; recover it before retrying.'}
$alphaBackup=Join-Path $alphaGame ('artifacts/patch-recovery/alpha-'+[Guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path (Join-Path $alphaBackup 'Managed') -Force|Out-Null
foreach($file in $alphaPrevious.files){Copy-Item -LiteralPath (Join-Path $alphaManaged $file.name) -Destination (Join-Path $alphaBackup ('Managed/'+$file.name))}
Copy-Item -LiteralPath $alphaRecord -Destination (Join-Path $alphaBackup 'patch-installation.json')
Write-BrassJsonAtomic (Join-Path $alphaBackup 'recovery.json') ([ordered]@{files=$alphaPrevious.files;recordSha256=(Get-FileHash -LiteralPath $alphaRecord).Hash})
Write-BrassJsonAtomic $alphaIntent ([ordered]@{operation='install-alpha-test';gameRoot=$alphaGame;backup=$alphaBackup;priorFiles=$alphaPrevious.files;candidateFiles=$manifest.files})
try{
 Copy-Item -LiteralPath (Join-Path $alphaBuild 'BrassNative') -Destination $alphaGame -Recurse
 foreach($pin in $q.files|Where-Object {$_.path.StartsWith('BrassNative/')} ){
  $path=Assert-AlphaInside (Join-Path $alphaGame $pin.path) $alphaGame
  if((Get-FileHash -LiteralPath $path).Hash -ne $pin.sha256){throw 'Installed native package hash mismatch.'}
 }
 foreach($file in $manifest.files){Copy-BrassAtomic (Join-Path $alphaBuild ('Managed/'+$file.name)) (Join-Path $alphaManaged $file.name)}
 Assert-AlphaFiles $manifest.files $alphaManaged
 $record=[ordered]@{version=$manifest.version;status='installed';installedUtc=[DateTime]::UtcNow.ToString('o');gameRoot=$alphaGame;
  stockBackup=$alphaPrevious.stockBackup;stockSha256=$alphaPrevious.stockSha256;files=$manifest.files;onlineEnabled=$false;
  saveDataPaths=$alphaPrevious.saveDataPaths;nativeAlphaRecovery=$alphaBackup;nativeBuildSha256=$manifest.native_build_sha256;
  qualificationSha256=(Get-FileHash -LiteralPath (Join-Path $alphaBuild 'qualification.json')).Hash;qualificationScope=$q.scope;localReleaseQualified=$false}
 Write-BrassJsonAtomic $alphaRecord $record
 Remove-Item -LiteralPath $alphaIntent
}catch{
 $failure=$_
 try{Restore-AlphaBackup $alphaBackup;Remove-Item -LiteralPath $alphaIntent}catch{Write-Warning ('Recovery remains in '+$alphaBackup+': '+$_.Exception.Message)}
 throw $failure
}
Write-Output 'Installed opt-in Alpha Test (L13 integration). L14 performance/quality and L15 full-game qualification remain pending.'
