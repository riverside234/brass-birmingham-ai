param([string]$GameRoot=(Split-Path -Parent $PSScriptRoot),[string[]]$SaveDataPaths=@())
$ErrorActionPreference='Stop'
. (Join-Path $PSScriptRoot 'patch-install-common.ps1')
$repo=Split-Path -Parent $PSScriptRoot
$recordPath=Join-Path $repo 'artifacts/patch-installation.json'
$intentPath=Join-Path $repo 'artifacts/patch-installation.pending.json'
$record=Read-BrassJson $recordPath
$pending=Read-BrassJson $intentPath
$game=(Resolve-Path -LiteralPath $GameRoot).Path
foreach($item in @($record,$pending)){if($item -and $item.gameRoot -ne $game){throw 'Installation record belongs to a different game directory.'}}
if(Get-Process -Name BrassBirminghamUnity -ErrorAction SilentlyContinue){throw 'Close Brass: Birmingham before restoring stock files.'}
$paths=@(Get-BrassSaveDataPaths @($record,$pending) $SaveDataPaths)
Assert-BrassNoPendingSaves $paths
Assert-BrassNoNewAiSaves $paths
$original=Join-Path $repo 'artifacts/patch-baseline/Assembly-CSharp.dll'
if((Get-FileHash -LiteralPath $original).Hash -ne $BrassStockSha256){throw 'Original backup hash mismatch.'}
# A matching current candidate is also recoverable if an old installation record was lost.
$manifest=Read-BrassJson (Join-Path $repo 'artifacts/patch-build/build-manifest.json')
$validation=Read-BrassJson (Join-Path $repo 'artifacts/patch-build/validation.json')
$known=@($record,$pending)
try{Assert-BrassCandidate $manifest $validation (Join-Path $repo 'artifacts/patch-build');$known+=@($manifest)}
catch{Write-Verbose 'Current candidate cannot provide additional recovery fingerprints.'}
$managed=Join-Path $game 'BrassBirminghamUnity_Data/Managed'
$assembly=Join-Path $managed 'Assembly-CSharp.dll'
$current=(Get-FileHash -LiteralPath $assembly).Hash
if($current -notin (@($BrassStockSha256)+@(Get-BrassHashes $known 'Assembly-CSharp.dll'))){throw 'Game files changed since installation; refusing to overwrite them.'}
$helper=Join-Path $managed 'BrassPatch.dll'
if((Test-Path -LiteralPath $helper) -and (Get-FileHash -LiteralPath $helper).Hash -notin @(Get-BrassHashes $known 'BrassPatch.dll')){throw 'Helper changed since installation; refusing to remove it.'}
$aiHelper=Join-Path $managed 'BrassLegacyAI.dll'
if((Test-Path -LiteralPath $aiHelper) -and (Get-FileHash -LiteralPath $aiHelper).Hash -notin @(Get-BrassHashes $known 'BrassLegacyAI.dll')){throw 'AI helper changed since installation; refusing to remove it.'}
$allFiles=@()
foreach($name in @('Assembly-CSharp.dll','BrassPatch.dll','BrassLegacyAI.dll')){foreach($hash in @(Get-BrassHashes $known $name)){$allFiles+=([ordered]@{name=$name;sha256=$hash})}}
$intent=[ordered]@{operation='restore';createdUtc=[DateTime]::UtcNow.ToString('o');gameRoot=$game;stockBackup=$original;stockSha256=$BrassStockSha256;priorFiles=$allFiles;saveDataPaths=$paths}
Write-BrassJsonAtomic $intentPath $intent
if($current -ne $BrassStockSha256){Copy-BrassAtomic $original $assembly}
if((Get-FileHash -LiteralPath $assembly).Hash -ne $BrassStockSha256){throw 'Restore hash verification failed.'}
if(Test-Path -LiteralPath $helper){Remove-Item -LiteralPath $helper}
if(Test-Path -LiteralPath $aiHelper){Remove-Item -LiteralPath $aiHelper}
$version='1.0.0'
if($record -and $record.version){$version=$record.version}
$restored=[ordered]@{version=$version;status='restored';restoredUtc=[DateTime]::UtcNow.ToString('o');gameRoot=$game;stockBackup=$original;stockSha256=$BrassStockSha256;files=$allFiles;onlineEnabled=$true;saveDataPaths=$paths}
Write-BrassJsonAtomic $recordPath $restored
Remove-Item -LiteralPath $intentPath
Write-Output 'Original game restored. Saves were not changed. Games created by the patch still require reinstalling it to continue.'
