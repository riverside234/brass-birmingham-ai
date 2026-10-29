param([ValidateSet(16,32)][int]$SearchK=16)
$ErrorActionPreference='Stop'
. (Join-Path $PSScriptRoot 'patch-install-common.ps1')
$root=Split-Path -Parent $PSScriptRoot
$study=Join-Path $root 'artifacts/host-delivery/20261029'
$out=Join-Path $study ('installed-k'+$SearchK)
if(Test-Path -LiteralPath $out){throw 'Preserve installed smoke evidence.'}
if(Get-Process -Name BrassBirminghamUnity,brass-ai-local -ErrorAction SilentlyContinue){throw 'Game or worker is running.'}
$record=Join-Path $root 'artifacts/patch-installation.json'
$installed=Read-BrassJson $record
$build=Join-Path $study 'release-1.4.44'
$m=Read-BrassJson (Join-Path $build 'build-manifest.json')
$q=Read-BrassJson (Join-Path $build 'qualification.json')
if($installed.version -ne '1.4.44-rust-local' -or -not $q.passed -or $installed.hostDeliveryQualificationSha256 -ne (Get-FileHash -LiteralPath (Join-Path $build 'qualification.json')).Hash){throw 'Qualified host-delivery installation required.'}
$pins=@($m.files|ForEach-Object{@{path=('BrassBirminghamUnity_Data/Managed/'+$_.name);sha256=$_.sha256}})
$pins+=@($m.nativeFiles|ForEach-Object{@{path=('BrassNative/'+$_.path);sha256=$_.sha256}})
function VerifyRuntime {
 foreach($pin in $pins){if((Get-FileHash -LiteralPath (Join-Path $root $pin.path)).Hash -ne $pin.sha256){throw ('Installed file changed: '+$pin.path)}}
 if(@(Get-ChildItem -LiteralPath (Join-Path $root 'BrassNative') -Recurse -File).Count -ne @($m.nativeFiles).Count){throw 'Native inventory changed.'}
}
VerifyRuntime
$source=Join-Path $study ('reserve08-canal-k'+$SearchK)
$test=Read-BrassJson (Join-Path $source 'test-build.json')
$dll=Join-Path $source 'LocalAiUnityTests.dll'
if($test.candidateManifestSha256 -ne $q.build_manifest_sha256 -or (Get-FileHash -LiteralPath $dll).Hash -ne $test.testDllSha256){throw 'Test harness identity changed.'}
[void](New-Item -ItemType Directory -Path $out)
$testRoot=Join-Path $root 'artifacts/patch-build'
$targets=@('Managed/LocalAiUnityTests.dll','local-ai-native.json','local-ai-cold-menu.png')
$retained=@()
foreach($name in $targets){
 $path=Join-Path $testRoot $name
 if(Test-Path -LiteralPath $path){
  $saved=Join-Path $out ('prior-test-artifacts/'+$name)
  [void](New-Item -ItemType Directory -Path (Split-Path -Parent $saved) -Force)
  Copy-Item -LiteralPath $path -Destination $saved;$retained+=$name
 }
}
$variables=@('BRASS_SEARCH_V2_RESULT','BRASS_SEARCH_V2_EVENTS','BRASS_SEARCH_V2_MS','BRASS_SEARCH_V2_SCENARIO','BRASS_SEARCH_K','BRASS_NN_TEMPERATURE','BRASS_L14_TRACE')
$old=@{};foreach($name in $variables){$old[$name]=[Environment]::GetEnvironmentVariable($name,'Process')}
try{
 Copy-BrassAtomic $dll (Join-Path $testRoot 'Managed/LocalAiUnityTests.dll')
 Write-BrassJsonAtomic (Join-Path $testRoot 'local-ai-native.json') @{passed=$false;status='running-installed-host-delivery-smoke'}
 foreach($entry in @{
  BRASS_SEARCH_V2_RESULT=(Join-Path $out 'result.json');
  BRASS_SEARCH_V2_EVENTS=(Join-Path $root 'artifacts/rust-backend/20261026-l14-study-v2/trusted-live-fixtures/live-92781-CanalEra-2.json');
  BRASS_SEARCH_V2_MS='4000';BRASS_SEARCH_V2_SCENARIO='normal';BRASS_SEARCH_K=([string]$SearchK);BRASS_NN_TEMPERATURE='1.0';BRASS_L14_TRACE=$null
 }.GetEnumerator()){[Environment]::SetEnvironmentVariable($entry.Key,$entry.Value,'Process')}
 $arguments=@('-batchmode','-screen-width','1920','-screen-height','1080','-screen-fullscreen','0','-logFile',('"'+(Join-Path $out 'unity.log')+'"'),
  '--brass-local-ai-test','--brass-native-local-test','--brass-native-matched-test','--brass-search-v2-test')
 $process=Start-Process -FilePath (Join-Path $root 'BrassBirminghamUnity.exe') -ArgumentList $arguments -WorkingDirectory $root -WindowStyle Hidden -PassThru
 $timer=[Diagnostics.Stopwatch]::StartNew()
 while(-not $process.WaitForExit(1000)){if($timer.Elapsed.TotalSeconds -gt 60){$process.Kill();$process.WaitForExit();throw 'Installed smoke watchdog.'}}
 Copy-Item -LiteralPath (Join-Path $testRoot 'local-ai-native.json') -Destination (Join-Path $out 'unity.json')
 Copy-Item -LiteralPath (Join-Path $testRoot 'local-ai-cold-menu.png') -Destination (Join-Path $out 'menu.png')
 $unity=Read-BrassJson (Join-Path $out 'unity.json');$result=Read-BrassJson (Join-Path $out 'result.json')
 if($process.ExitCode -ne 0 -or -not $unity.passed -or -not $result.passed -or $result.search_k -ne $SearchK -or $result.search_method -ne 'search_v1' -or $result.focused -or
  $result.readiness.k32_host_delivery_reserve_ns -ne 800000000 -or $result.commits.Count -ne 2 -or $result.commits[-1].remaining -ne 0 -or $result.delivery_ms -ge 4000){throw 'Installed host-delivery turn failed or exceeded four seconds.'}
 foreach($reply in $result.replies){if($reply.requested_k -ne $SearchK -or $reply.sessions -ne 2 -or $reply.threads_per_session -ne 1 -or $reply.policy_temperature -ne 1.0 -or
  $reply.response_ready_ns -ge $reply.deadline_ns -or $null -ne $reply.search_revision){throw 'Installed search configuration/timing differs.'}}
 if(-not $result.replies.Count){throw 'No actual native reply.'}
 VerifyRuntime
 Write-BrassJsonAtomic (Join-Path $out 'verification.json') @{passed=$true;version=$installed.version;scope='installed-game/4s/two-actions/isolated-save';
  searchK=$SearchK;temperature=1.0;checks=$unity.checks;deliveryMs=$result.delivery_ms;k32MinimumDeliveryReserveMs=800;
  qualificationSha256=$installed.hostDeliveryQualificationSha256;testDllSha256=$test.testDllSha256;testedFiles=$pins;resultSha256=(Get-FileHash -LiteralPath (Join-Path $out 'result.json')).Hash}
 $installed|Add-Member -NotePropertyName ('hostDeliveryInstalledK'+$SearchK) -NotePropertyValue (Join-Path $out 'verification.json') -Force
 $installed|Add-Member -NotePropertyName ('hostDeliveryInstalledK'+$SearchK+'Sha256') -NotePropertyValue ((Get-FileHash -LiteralPath (Join-Path $out 'verification.json')).Hash) -Force
 Write-BrassJsonAtomic $record $installed
 $result|Select-Object passed,search_k,requests,delivery_ms|ConvertTo-Json
}finally{
 foreach($name in $variables){[Environment]::SetEnvironmentVariable($name,$old[$name],'Process')}
 foreach($name in $targets){
  $path=Join-Path $testRoot $name
  if($name -in $retained){Copy-BrassAtomic (Join-Path $out ('prior-test-artifacts/'+$name)) $path}
  elseif(Test-Path -LiteralPath $path){Remove-Item -LiteralPath $path}
 }
}
