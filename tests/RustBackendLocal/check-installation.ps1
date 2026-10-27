param([string]$BuildDirectory='artifacts/rust-backend/20261026-l13-qualified')
$ErrorActionPreference='Stop'
$installTestRoot=Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
. (Join-Path $installTestRoot 'scripts/patch-install-common.ps1')
$installTestGame=Join-Path $installTestRoot 'artifacts/card-tracker-player'
$installTestOut=Join-Path $installTestRoot 'artifacts/rust-backend/20261026-l13-deployment'
New-Item -ItemType Directory -Path $installTestOut -Force|Out-Null
if(Get-Process -Name BrassBirminghamUnity -ErrorAction SilentlyContinue){throw 'The isolated player must be closed.'}
$testNative=[IO.Path]::GetFullPath((Join-Path $installTestGame 'BrassNative'))
$retained=[IO.Path]::GetFullPath((Join-Path $installTestGame ('artifacts/native-test-retained-'+[Guid]::NewGuid().ToString('N'))))
$expected=[IO.Path]::GetFullPath($installTestGame)+'\'
if(-not $testNative.StartsWith($expected) -or -not $retained.StartsWith($expected)){throw 'Isolated retention paths escaped test player.'}
if(Test-Path -LiteralPath $testNative){Move-Item -LiteralPath $testNative -Destination $retained}
$prior=Read-BrassJson (Join-Path $installTestRoot 'artifacts/patch-installation.json')
if($prior.version -ne '1.4.34'){throw 'Run this isolated installation check before changing the real baseline.'}
$prior.gameRoot=$installTestGame;$prior.saveDataPaths=@()
$testRecord=Join-Path $installTestGame 'artifacts/patch-installation.json'
Write-BrassJsonAtomic $testRecord $prior
$originalHash=(Get-FileHash -LiteralPath $testRecord).Hash
foreach($file in $prior.files){Copy-BrassAtomic (Join-Path $installTestRoot ('artifacts/patch-releases/1.4.34/Managed/'+$file.name)) (Join-Path $installTestGame ('BrassBirminghamUnity_Data/Managed/'+$file.name))}
$installer=Join-Path $installTestRoot 'scripts/install-rust-alpha.ps1'
& $installer -BuildDirectory $BuildDirectory -GameRoot $installTestGame
$installed=Read-BrassJson $testRecord
if($installed.version -ne '1.4.35-alpha-test' -or $installed.localReleaseQualified -ne $false){throw 'Alpha Test scope lost.'}
& $installer -Action Rollback -GameRoot $installTestGame
if((Get-FileHash -LiteralPath $testRecord).Hash -ne $originalHash -or (Test-Path -LiteralPath $testNative)){throw 'Rollback did not restore the exact prior record/package state.'}
foreach($file in $prior.files){if((Get-FileHash -LiteralPath (Join-Path $installTestGame ('BrassBirminghamUnity_Data/Managed/'+$file.name))).Hash -ne $file.sha256){throw 'Rollback hash mismatch.'}}
# Simulate interruption after helper publication but before recording success.
& $installer -BuildDirectory $BuildDirectory -GameRoot $installTestGame
$installed=Read-BrassJson $testRecord
$backup=$installed.nativeAlphaRecovery
Write-BrassJsonAtomic (Join-Path $installTestGame 'artifacts/rust-alpha-installation.pending.json') ([ordered]@{gameRoot=$installTestGame;backup=$backup;priorFiles=$prior.files;candidateFiles=$installed.files})
Copy-BrassAtomic (Join-Path $backup 'patch-installation.json') $testRecord
& $installer -Action Recover -GameRoot $installTestGame
if((Get-FileHash -LiteralPath $testRecord).Hash -ne $originalHash -or (Test-Path -LiteralPath $testNative)){throw 'Interrupted installation recovery failed.'}
foreach($file in $prior.files){if((Get-FileHash -LiteralPath (Join-Path $installTestGame ('BrassBirminghamUnity_Data/Managed/'+$file.name))).Hash -ne $file.sha256){throw 'Recovery hash mismatch.'}}
Write-BrassJsonAtomic (Join-Path $installTestOut 'installation-tests.json') ([ordered]@{passed=$true;scope='isolated-player';install=$true;exactRollback=$true;interruptedInstallRecovery=$true;installerSha256=(Get-FileHash -LiteralPath $installer).Hash;qualificationSha256=$installed.qualificationSha256;priorRecordSha256=$originalHash;files=$prior.files})
Write-Output 'Isolated install, exact rollback and interrupted-install recovery passed.'
