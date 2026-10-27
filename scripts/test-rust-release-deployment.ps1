param([string]$StudyDirectory='artifacts/rust-backend/20261026-l15')
$ErrorActionPreference='Stop'
$releaseRoot=Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot 'patch-install-common.ps1')
$releaseStudy=[IO.Path]::GetFullPath((Join-Path $releaseRoot $StudyDirectory))
if(-not $releaseStudy.StartsWith((Join-Path $releaseRoot 'artifacts/rust-backend')+'\',[StringComparison]::OrdinalIgnoreCase)){throw 'Study path escaped artifacts.'}
$releaseOut=Join-Path $releaseStudy 'deployment'
if(Test-Path -LiteralPath $releaseOut){throw 'Preserve existing deployment evidence.'}
$releaseGame=Join-Path $releaseOut 'game'
$releaseManaged=Join-Path $releaseGame 'BrassBirminghamUnity_Data/Managed'
[void](New-Item -ItemType Directory -Path $releaseManaged)
[void](New-Item -ItemType Directory -Path (Join-Path $releaseGame 'artifacts'))
$releaseBase=Join-Path $releaseRoot 'artifacts/patch-releases/1.4.34'
$releasePrior=Read-BrassJson (Join-Path $releaseBase 'patch-installation.json')
$releasePrior.gameRoot=$releaseGame;$releasePrior.saveDataPaths=@()
$releaseRecord=Join-Path $releaseGame 'artifacts/patch-installation.json'
Write-BrassJsonAtomic $releaseRecord $releasePrior
$releasePriorHash=(Get-FileHash -LiteralPath $releaseRecord).Hash
foreach($file in $releasePrior.files){Copy-Item -LiteralPath (Join-Path $releaseBase ('Managed/'+$file.name)) -Destination (Join-Path $releaseManaged $file.name)}
$releaseInstaller=Join-Path $PSScriptRoot 'install-rust-alpha.ps1'
# L13 admission remains strict: an L15 smoke candidate cannot masquerade as an
# accepted release. Verify rejection before touching any candidate files.
$rejected=$false
try{& $releaseInstaller -BuildDirectory ($StudyDirectory+'/candidate') -GameRoot $releaseGame}catch{$rejected=$true}
if(-not $rejected -or (Get-FileHash -LiteralPath $releaseRecord).Hash -ne $releasePriorHash){throw 'Unqualified release was not rejected without mutation.'}
& $releaseInstaller -BuildDirectory 'artifacts/rust-backend/20261026-l13-qualified' -GameRoot $releaseGame
$releaseInstalled=Read-BrassJson $releaseRecord
$releaseCandidate=Read-BrassJson (Join-Path $releaseStudy 'candidate/build-manifest.json')
# Authored rollback fixture, not an admission/install receipt: put the current
# candidate bytes in the disposable target, retain the real pre-Alpha backup,
# and exercise the same whole-installation rollback path against those bytes.
foreach($file in $releaseCandidate.files){Copy-BrassAtomic (Join-Path $releaseStudy ('candidate/Managed/'+$file.name)) (Join-Path $releaseManaged $file.name)}
Copy-Item -LiteralPath (Join-Path $releaseStudy 'candidate/BrassNative') -Destination $releaseGame -Recurse -Force
$releaseInstalled.files=$releaseCandidate.files
$releaseInstalled.qualificationScope='l15-unqualified-rollback-fixture'
$releaseInstalled.qualificationSha256=$null
$releaseInstalled.nativeBuildSha256=$releaseCandidate.native_build_sha256
Write-BrassJsonAtomic $releaseRecord $releaseInstalled
$saveSentinel=Join-Path $releaseGame 'nn-save-untouched.json'
[IO.File]::WriteAllText($saveSentinel,'{"family":"nn","fixture":true}')
$saveHash=(Get-FileHash -LiteralPath $saveSentinel).Hash
& $releaseInstaller -Action Rollback -GameRoot $releaseGame
if((Get-FileHash -LiteralPath $releaseRecord).Hash -ne $releasePriorHash -or (Test-Path -LiteralPath (Join-Path $releaseGame 'BrassNative'))){throw 'Exact rollback failed.'}
foreach($file in $releasePrior.files){if((Get-FileHash -LiteralPath (Join-Path $releaseManaged $file.name)).Hash -ne $file.sha256){throw 'Prior binary not restored.'}}
if((Get-FileHash -LiteralPath $saveSentinel).Hash -ne $saveHash){throw 'Rollback changed NN saves.'}
# Interrupted publication recovery uses the original install intent contract.
& $releaseInstaller -BuildDirectory 'artifacts/rust-backend/20261026-l13-qualified' -GameRoot $releaseGame
$releaseInstalled=Read-BrassJson $releaseRecord
Write-BrassJsonAtomic (Join-Path $releaseGame 'artifacts/rust-alpha-installation.pending.json') ([ordered]@{
 gameRoot=$releaseGame;backup=$releaseInstalled.nativeAlphaRecovery;priorFiles=$releasePrior.files;candidateFiles=$releaseInstalled.files})
Copy-BrassAtomic (Join-Path $releaseInstalled.nativeAlphaRecovery 'patch-installation.json') $releaseRecord
& $releaseInstaller -Action Recover -GameRoot $releaseGame
if((Get-FileHash -LiteralPath $releaseRecord).Hash -ne $releasePriorHash -or (Test-Path -LiteralPath (Join-Path $releaseGame 'BrassNative'))){throw 'Interrupted publication recovery failed.'}
Write-BrassJsonAtomic (Join-Path $releaseOut 'receipt.json') ([ordered]@{
 passed=$true;scope='disposable deployment fixture';candidateInstalled=$false;unqualifiedAdmissionRejected=$true;
 exactCandidateToV4Rollback=$true;interruptedL13PublicationRecovery=$true;nnSaveUntouched=$true;
 installerSha256=(Get-FileHash -LiteralPath $releaseInstaller).Hash;candidateManifestSha256=(Get-FileHash -LiteralPath (Join-Path $releaseStudy 'candidate/build-manifest.json')).Hash;
 priorReceiptSha256=$releasePriorHash})
Write-Output 'Unqualified admission rejected; candidate rollback, save preservation and interrupted publication recovery passed in disposable target.'
