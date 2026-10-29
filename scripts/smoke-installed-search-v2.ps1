param([string]$EvidenceDirectory='artifacts/search-v2/20261028/installed-smoke', [ValidateSet(8,16)][int]$ExpectedK=8)
$ErrorActionPreference='Stop'
. (Join-Path $PSScriptRoot 'patch-install-common.ps1')
$root=Split-Path -Parent $PSScriptRoot
$out=[IO.Path]::GetFullPath((Join-Path $root $EvidenceDirectory))
if(-not $out.StartsWith((Join-Path $root 'artifacts/search-v2')+'\',[StringComparison]::OrdinalIgnoreCase)){throw 'Smoke output escapes artifacts.'}
if(Test-Path -LiteralPath $out){throw 'Preserve existing smoke evidence.'}
if(Get-Process -Name BrassBirminghamUnity,brass-ai-local -ErrorAction SilentlyContinue){throw 'Game or native worker is running.'}
[void](New-Item -ItemType Directory -Path $out)
$recordPath=Join-Path $root 'artifacts/patch-installation.json'
$installed=Read-BrassJson $recordPath
if($installed.searchMethod -ne 'search_v2_focus_only' -or $installed.v205Enabled){throw 'Cache-only installation required.'}
$pins=@($installed.files|ForEach-Object{@{path=('BrassBirminghamUnity_Data/Managed/'+$_.name);sha256=$_.sha256}})
$pins+=@(Get-ChildItem -LiteralPath (Join-Path $root 'BrassNative') -Recurse -File|ForEach-Object{@{path=$_.FullName.Substring($root.Length+1);sha256=(Get-FileHash -LiteralPath $_.FullName).Hash}})
function VerifyRuntime {
 foreach($pin in $pins){if((Get-FileHash -LiteralPath (Join-Path $root $pin.path)).Hash -ne $pin.sha256){throw ('Installed runtime changed: '+$pin.path)}}
 $p=Read-BrassJson (Join-Path $root 'BrassNative/alpha-g890/policy.json')
 if($p.method -ne 'search_v2_focus_only' -or $p.k_by_allowance_ms.'4000' -ne $ExpectedK){throw 'Method/K mismatch.'}
}
VerifyRuntime
$source=Join-Path $root 'artifacts/search-v2/20261028/v205-unity-cache-normal/evidence'
$testBuild=Read-BrassJson (Join-Path $source 'test-build.json')
$testDll=Join-Path $source 'LocalAiUnityTests.dll'
if((Get-FileHash -LiteralPath $testDll).Hash -ne $testBuild.testDllSha256){throw 'Smoke harness pin changed.'}
$testRoot=Join-Path $root 'artifacts/patch-build'
$targets=@('Managed/LocalAiUnityTests.dll','local-ai-native.json','local-ai-cold-menu.png')
$backedUp=@()
foreach($name in $targets){
 $path=Join-Path $testRoot $name
 if(Test-Path -LiteralPath $path){$backup=Join-Path $out ('prior-test-artifacts/'+$name);[void](New-Item -ItemType Directory -Path (Split-Path -Parent $backup) -Force);Copy-Item -LiteralPath $path -Destination $backup;$backedUp+=$name}
}
$variables=@('BRASS_SEARCH_V2_RESULT','BRASS_SEARCH_V2_EVENTS','BRASS_SEARCH_V2_MS','BRASS_SEARCH_V2_SCENARIO','BRASS_L14_TRACE')
$oldEnv=@{};foreach($name in $variables){$oldEnv[$name]=[Environment]::GetEnvironmentVariable($name,'Process')}
try{
 Copy-BrassAtomic $testDll (Join-Path $testRoot 'Managed/LocalAiUnityTests.dll')
 Write-BrassJsonAtomic (Join-Path $testRoot 'local-ai-native.json') @{passed=$false;status='running-installed-smoke'}
 [Environment]::SetEnvironmentVariable('BRASS_SEARCH_V2_RESULT',(Join-Path $out 'result.json'),'Process')
 [Environment]::SetEnvironmentVariable('BRASS_SEARCH_V2_EVENTS',(Join-Path $root 'artifacts/rust-backend/20261026-l14-study-v2/trusted-live-fixtures/live-92781-CanalEra-2.json'),'Process')
 [Environment]::SetEnvironmentVariable('BRASS_SEARCH_V2_MS','4000','Process')
 [Environment]::SetEnvironmentVariable('BRASS_SEARCH_V2_SCENARIO','normal','Process')
 [Environment]::SetEnvironmentVariable('BRASS_L14_TRACE',$null,'Process')
 $arguments=@('-batchmode','-screen-width','1920','-screen-height','1080','-screen-fullscreen','0','-logFile',('"'+(Join-Path $out 'unity.log')+'"'),
  '--brass-local-ai-test','--brass-native-local-test','--brass-native-matched-test','--brass-search-v2-test')
 $process=Start-Process -FilePath (Join-Path $root 'BrassBirminghamUnity.exe') -ArgumentList $arguments -WorkingDirectory $root -WindowStyle Hidden -PassThru
 $timer=[Diagnostics.Stopwatch]::StartNew()
 while(-not $process.WaitForExit(1000)){if($timer.Elapsed.TotalSeconds -gt 60){$process.Kill();$process.WaitForExit();throw 'Installed smoke watchdog.'}}
 Copy-Item -LiteralPath (Join-Path $testRoot 'local-ai-native.json') -Destination (Join-Path $out 'unity.json')
 Copy-Item -LiteralPath (Join-Path $testRoot 'local-ai-cold-menu.png') -Destination (Join-Path $out 'menu.png')
 $unity=Read-BrassJson (Join-Path $out 'unity.json');$result=Read-BrassJson (Join-Path $out 'result.json')
 if($process.ExitCode -ne 0 -or -not $unity.passed -or -not $result.passed -or -not $result.focused -or $result.requests -ne 1){throw 'Installed Unity smoke failed.'}
 foreach($reply in $result.replies){
  if($reply.search_revision.focus.method -ne 'search_v2_focus_only'){throw 'Wrong live search method.'}
  if(@($reply.search_revision.focused_owners).Count -ne 2){throw 'Expected two independent focused owners.'}
  if($reply.requested_k -ne $ExpectedK){throw 'Unexpected requested K.'}
  foreach($lane in $reply.search_revision.focused_owners){if($lane.epoch -ne 2 -or $lane.effective_k -lt 1 -or $lane.effective_k -gt $ExpectedK){throw 'Unexpected focused owner configuration.'}}
 }
 if($result.commits.Count -ne 2 -or $result.commits[-1].remaining -ne 0 -or $result.delivery_ms -ge 4000){throw 'Incomplete/late actual turn.'}
 VerifyRuntime
 Write-BrassJsonAtomic (Join-Path $out 'verification.json') ([ordered]@{passed=$true;gameRoot=$root;executable=(Join-Path $root 'BrassBirminghamUnity.exe');
  scope=('installed-player/4s-K'+$ExpectedK+'/one-complete-turn/isolated-save');requestedK=$ExpectedK;searchMethod='search_v2_focus_only';v205Enabled=$false;checks=$unity.checks;
  actualActions=$result.commits.Count;requests=$result.requests;deliveryMs=$result.delivery_ms;testedFiles=$pins;
  testDllSha256=$testBuild.testDllSha256;candidateSha256=$installed.searchV2CandidateSha256;
  resultSha256=(Get-FileHash -LiteralPath (Join-Path $out 'result.json')).Hash;unitySha256=(Get-FileHash -LiteralPath (Join-Path $out 'unity.json')).Hash})
 $installed|Add-Member -NotePropertyName searchV2InstalledSmoke -NotePropertyValue ('passed/4s-K'+$ExpectedK) -Force
 $installed|Add-Member -NotePropertyName searchV2InstalledSmokePath -NotePropertyValue (Join-Path $out 'verification.json') -Force
 $installed|Add-Member -NotePropertyName searchV2InstalledSmokeSha256 -NotePropertyValue ((Get-FileHash -LiteralPath (Join-Path $out 'verification.json')).Hash) -Force
 Write-BrassJsonAtomic $recordPath $installed
 $result|Select-Object passed,focused,requests,delivery_ms,commits|ConvertTo-Json -Depth 5
}finally{
 foreach($name in $variables){[Environment]::SetEnvironmentVariable($name,$oldEnv[$name],'Process')}
 foreach($name in $targets){
  $path=Join-Path $testRoot $name
  if($name -in $backedUp){Copy-BrassAtomic (Join-Path $out ('prior-test-artifacts/'+$name)) $path}
  elseif(Test-Path -LiteralPath $path){Remove-Item -LiteralPath $path}
 }
}
