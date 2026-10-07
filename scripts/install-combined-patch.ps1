param([string]$GameRoot=(Split-Path -Parent $PSScriptRoot),[string[]]$SaveDataPaths=@(),[string]$BuildDirectory)
$ErrorActionPreference='Stop'
. (Join-Path $PSScriptRoot 'patch-install-common.ps1')
$repo=Split-Path -Parent $PSScriptRoot
$game=(Resolve-Path -LiteralPath $GameRoot).Path
$managed=Join-Path $game 'BrassBirminghamUnity_Data/Managed'
$build=if($BuildDirectory){(Resolve-Path -LiteralPath $BuildDirectory).Path}else{Join-Path $repo 'artifacts/patch-build'}
$recordPath=Join-Path $repo 'artifacts/patch-installation.json'
$intentPath=Join-Path $repo 'artifacts/patch-installation.pending.json'
$manifest=Read-BrassJson (Join-Path $build 'build-manifest.json')
$validation=Read-BrassJson (Join-Path $build 'validation.json')
Assert-BrassCandidate $manifest $validation $build
if(Get-Process -Name BrassBirminghamUnity -ErrorAction SilentlyContinue){throw 'Close Brass: Birmingham before installing. No files were changed.'}
$original=Join-Path $repo 'artifacts/patch-baseline/Assembly-CSharp.dll'
if(-not(Test-Path -LiteralPath $original) -or (Get-FileHash -LiteralPath $original).Hash -ne $BrassStockSha256){throw 'Original backup is missing or changed; no installation was performed.'}
$previous=Read-BrassJson $recordPath
$pending=Read-BrassJson $intentPath
if($manifest.version -in @('1.4.12','1.4.13','1.4.15','1.4.16','1.4.17','1.4.18','1.4.19','1.4.20','1.4.21','1.4.22','1.4.23','1.4.24','1.4.25','1.4.26','1.4.27','1.4.28','1.4.29')){
 if(-not $previous -or $previous.status -ne 'installed' -or $previous.version -notin @($manifest.baseVersion,$manifest.version)){throw 'This revision requires its qualified baseline or the same installed release; rebase before installing.'}
 $expectedPrior=if($previous.version -eq $manifest.version){$manifest.files}else{$manifest.baseFiles}
 foreach($file in $expectedPrior){if(@($previous.files|Where-Object {$_.name -eq $file.name -and $_.sha256 -eq $file.sha256}).Count -ne 1){throw 'Installed baseline record changed; rebase before installing.'}}
}
foreach($item in @($previous,$pending)){if($item -and $item.gameRoot -ne $game){throw 'Installation record belongs to a different game directory.'}}
$paths=@(Get-BrassSaveDataPaths @($previous,$pending) $SaveDataPaths)
$known=@($previous,$pending,$manifest)
$names=@('BrassPatch.dll','BrassLegacyAI.dll','Assembly-CSharp.dll')
$priorFiles=@()
foreach($name in $names){
 $path=Join-Path $managed $name
 if(Test-Path -LiteralPath $path){
  $hash=(Get-FileHash -LiteralPath $path).Hash
  $allowed=@(Get-BrassHashes $known $name)
  if($name -eq 'Assembly-CSharp.dll'){$allowed+=@($BrassStockSha256)}
  if($hash -notin $allowed){throw ('Unrecognized installed file; refusing to overwrite '+$name)}
  $priorFiles+=([ordered]@{name=$name;sha256=$hash})
 }elseif($name -eq 'Assembly-CSharp.dll'){throw 'The installed game assembly is missing.'}
}
$coherentPrior=(@($priorFiles|Where-Object name -eq 'Assembly-CSharp.dll')[0].sha256 -eq $BrassStockSha256)
if($previous -and $previous.status -eq 'installed'){
 $coherentPrior=$true
 foreach($file in $previous.files){if($file.sha256 -notin @($priorFiles|Where-Object name -eq $file.name|ForEach-Object sha256)){$coherentPrior=$false}}
}
$backup=Join-Path $repo ('artifacts/patch-recovery/'+[Guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $backup -Force|Out-Null
foreach($file in $priorFiles){
 Copy-Item -LiteralPath (Join-Path $managed $file.name) -Destination (Join-Path $backup $file.name)
 if((Get-FileHash -LiteralPath (Join-Path $backup $file.name)).Hash -ne $file.sha256){throw 'Prior release backup failed verification.'}
}
$intent=[ordered]@{operation='install';createdUtc=[DateTime]::UtcNow.ToString('o');gameRoot=$game;stockSha256=$BrassStockSha256;
 priorFiles=$priorFiles;candidateFiles=$manifest.files;backupDirectory=$backup;coherentPrior=$coherentPrior;priorRecord=$previous;saveDataPaths=$paths}
Write-BrassJsonAtomic $intentPath $intent
try{
 # Publish dependencies first; the game is closed throughout the transaction.
 foreach($name in $names){Copy-BrassAtomic (Join-Path $build ('Managed/'+$name)) (Join-Path $managed $name)}
 foreach($file in $manifest.files){if((Get-FileHash -LiteralPath (Join-Path $managed $file.name)).Hash -ne $file.sha256){throw 'Installed hash mismatch.'}}
 $record=[ordered]@{version=$manifest.version;status='installed';installedUtc=[DateTime]::UtcNow.ToString('o');gameRoot=$game;
 stockBackup=(Join-Path $repo 'artifacts/patch-baseline/Assembly-CSharp.dll');stockSha256=$BrassStockSha256;files=$manifest.files;onlineEnabled=$false;saveDataPaths=$paths;previousReleaseBackup=$backup}
 Write-BrassJsonAtomic $recordPath $record
 Remove-Item -LiteralPath $intentPath
}catch{
 $installError=$_
 if($coherentPrior){
  try{
   foreach($file in $priorFiles){if((Get-FileHash -LiteralPath (Join-Path $backup $file.name)).Hash -ne $file.sha256){throw 'Recovery backup changed.'}}
   foreach($name in $names){if($name -in @($priorFiles|ForEach-Object name)){Copy-BrassAtomic (Join-Path $backup $name) (Join-Path $managed $name)}}
   foreach($name in $names|Where-Object {$_ -notin @($priorFiles|ForEach-Object name)}){
    $path=Join-Path $managed $name
    if(Test-Path -LiteralPath $path){if((Get-FileHash -LiteralPath $path).Hash -notin @(Get-BrassHashes @($intent) $name)){throw 'Changed recovery target retained.'};Remove-Item -LiteralPath $path}
   }
   if($previous){Write-BrassJsonAtomic $recordPath $previous}
   Remove-Item -LiteralPath $intentPath
  }catch{Write-Warning ('Recovery is incomplete; the prior binaries and transaction record were retained. '+$_.Exception.Message)}
 }else{Write-Warning 'The installation was already incomplete. Its recovery record was retained; rerun installation after resolving the file error.'}
 throw $installError
}
Write-Output ("Installed combined patch " + $manifest.version + ". Online play remains disabled.")
