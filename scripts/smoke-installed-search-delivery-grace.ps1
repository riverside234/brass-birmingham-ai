param([ValidateSet(8,16,32)][int]$SearchK=16,[ValidateSet(4000,8000,10000,12000)][int]$Allowance=4000,[switch]$Late)
$ErrorActionPreference='Stop'
. (Join-Path $PSScriptRoot 'patch-install-common.ps1')
$root=Split-Path -Parent $PSScriptRoot
$study=Join-Path $root 'artifacts/search-delivery-grace/20261029'
$out=Join-Path $study ('installed-'+$Allowance+'-k'+$SearchK)
if(Test-Path -LiteralPath $out){throw 'Preserve installed smoke evidence.'}
if(Get-Process -Name BrassBirminghamUnity,brass-ai-local -ErrorAction SilentlyContinue){throw 'Game or worker is running.'}
$record=Join-Path $root 'artifacts/patch-installation.json';$installed=Read-BrassJson $record
$build=Join-Path $study 'release-1.4.45';$m=Read-BrassJson (Join-Path $build 'build-manifest.json');$q=Read-BrassJson (Join-Path $build 'qualification.json')
if($installed.version -ne '1.4.45-rust-local' -or -not $q.passed -or $installed.searchDeliveryGraceQualificationSha256 -ne (Get-FileHash -LiteralPath (Join-Path $build 'qualification.json')).Hash){throw 'Qualified search-delivery-grace installation required.'}
$pins=@($m.files|ForEach-Object{@{path=('BrassBirminghamUnity_Data/Managed/'+$_.name);sha256=$_.sha256}})
$pins+=@($m.nativeFiles|ForEach-Object{@{path=('BrassNative/'+$_.path);sha256=$_.sha256}})
function VerifyRuntime {
 foreach($pin in $pins){if((Get-FileHash -LiteralPath (Join-Path $root $pin.path)).Hash -ne $pin.sha256){throw ('Installed runtime changed: '+$pin.path)}}
 if(@(Get-ChildItem -LiteralPath (Join-Path $root 'BrassNative') -Recurse -File).Count -ne @($m.nativeFiles).Count){throw 'Unexpected native member.'}
}
VerifyRuntime
$source=Join-Path $study ('late-'+$Allowance+'-k'+$SearchK);$test=Read-BrassJson (Join-Path $source 'test-build.json');$dll=Join-Path $source 'LocalAiUnityTests.dll'
if($test.candidateManifestSha256 -ne $q.build_manifest_sha256 -or (Get-FileHash -LiteralPath $dll).Hash -ne $test.testDllSha256){throw 'Test harness changed.'}
[void](New-Item -ItemType Directory -Path $out)
$testRoot=Join-Path $root 'artifacts/patch-build';$targets=@('Managed/LocalAiUnityTests.dll','local-ai-native.json','local-ai-cold-menu.png');$retained=@()
foreach($name in $targets){
 $path=Join-Path $testRoot $name
 if(Test-Path -LiteralPath $path){$saved=Join-Path $out ('prior-test-artifacts/'+$name);[void](New-Item -ItemType Directory -Path (Split-Path -Parent $saved) -Force);Copy-Item -LiteralPath $path -Destination $saved;$retained+=$name}
}
$variables=@('BRASS_SEARCH_V2_RESULT','BRASS_SEARCH_V2_EVENTS','BRASS_SEARCH_V2_MS','BRASS_SEARCH_V2_SCENARIO','BRASS_SEARCH_K','BRASS_NN_TEMPERATURE','BRASS_L14_TRACE','BRASS_SEARCH_HOLD_UNTIL_MS')
$old=@{};foreach($name in $variables){$old[$name]=[Environment]::GetEnvironmentVariable($name,'Process')}
$reserve=1000
try{
 Copy-BrassAtomic $dll (Join-Path $testRoot 'Managed/LocalAiUnityTests.dll')
 Write-BrassJsonAtomic (Join-Path $testRoot 'local-ai-native.json') @{passed=$false;status='running-installed-search-delivery-grace-smoke'}
 foreach($entry in @{
  BRASS_SEARCH_V2_RESULT=(Join-Path $out 'result.json');BRASS_SEARCH_V2_EVENTS=(Join-Path $root $(if($Allowance -eq 10000){'artifacts/rust-backend/20261026-l14-study-v2/trusted-live-fixtures/live-92782-RailEra-7.json'}else{'artifacts/rust-backend/20261026-l14-study-v2/trusted-live-fixtures/live-92781-RailEra-2.json'}));
  BRASS_SEARCH_V2_MS=([string]$Allowance);BRASS_SEARCH_V2_SCENARIO='normal';BRASS_SEARCH_K=([string]$SearchK);BRASS_NN_TEMPERATURE='1.0';BRASS_L14_TRACE=$null;
  BRASS_SEARCH_HOLD_UNTIL_MS=([string]$(if($Late){$Allowance}else{0}))
 }.GetEnumerator()){[Environment]::SetEnvironmentVariable($entry.Key,$entry.Value,'Process')}
 $arguments=@('-batchmode','-screen-width','1920','-screen-height','1080','-screen-fullscreen','0','-logFile',('"'+(Join-Path $out 'unity.log')+'"'),
  '--brass-local-ai-test','--brass-native-local-test','--brass-native-matched-test','--brass-search-v2-test')
 $process=Start-Process -FilePath (Join-Path $root 'BrassBirminghamUnity.exe') -ArgumentList $arguments -WorkingDirectory $root -WindowStyle Hidden -PassThru
 $timer=[Diagnostics.Stopwatch]::StartNew()
 while(-not $process.WaitForExit(1000)){if($timer.Elapsed.TotalSeconds -gt 60){$process.Kill();$process.WaitForExit();throw 'Installed search-delivery-grace smoke watchdog.'}}
 Copy-Item -LiteralPath (Join-Path $testRoot 'local-ai-native.json') -Destination (Join-Path $out 'unity.json')
 $unity=Read-BrassJson (Join-Path $out 'unity.json');$r=Read-BrassJson (Join-Path $out 'result.json')
 if($process.ExitCode -ne 0 -or -not $unity.passed -or -not $r.passed -or $r.search_k -ne $SearchK -or $r.allowance_ms -ne $Allowance -or
  $r.search_method -ne 'search_v1' -or $r.focused -or $r.commits.Count -ne 2 -or $r.commits[-1].remaining -ne 0 -or $r.delivery_ms -ge ($Allowance+1000) -or
  $r.readiness.host_delivery_reserve_policy -ne 'search-plus-delivery/1.0.0' -or $r.readiness.host_delivery_reserve_ns_by_k."$SearchK" -ne $reserve*1000000){throw 'Installed search-delivery-grace turn failed or late.'}
 if($Late -and ($r.late_delivery_resumed_ms -lt $Allowance -or ($Allowance+1000)-$r.turn_clock_ms -lt 100)){throw 'Insufficient measured late-delivery headroom.'}
 foreach($reply in $r.replies){if($reply.requested_k -ne $SearchK -or $reply.sessions -ne 2 -or $reply.threads_per_session -ne 1 -or $reply.response_ready_ns -ge $reply.deadline_ns){throw 'Installed settings/deadline mismatch.'}}
 VerifyRuntime
 Write-BrassJsonAtomic (Join-Path $out 'verification.json') @{passed=$true;version=$installed.version;searchK=$SearchK;allowanceMs=$Allowance;deliveryGraceMs=$reserve;totalAllowanceMs=($Allowance+1000);
  lateProbe=[bool]$Late;checks=$unity.checks;deliveryMs=$r.delivery_ms;turnClockMs=$r.turn_clock_ms;headroomMs=(($Allowance+1000)-$r.turn_clock_ms);
  tailMs=$r.delivery_tail_after_resume_ms;actions=2;qualificationSha256=$installed.searchDeliveryGraceQualificationSha256;testedFiles=$pins;
  resultSha256=(Get-FileHash -LiteralPath (Join-Path $out 'result.json')).Hash}
 $key='searchDeliveryGraceInstalled'+$Allowance+'K'+$SearchK
 $installed|Add-Member -NotePropertyName $key -NotePropertyValue (Join-Path $out 'verification.json') -Force
 $installed|Add-Member -NotePropertyName ($key+'Sha256') -NotePropertyValue ((Get-FileHash -LiteralPath (Join-Path $out 'verification.json')).Hash) -Force
 Write-BrassJsonAtomic $record $installed
 $r|Select-Object passed,search_k,allowance_ms,delivery_ms,turn_clock_ms,delivery_tail_after_resume_ms|ConvertTo-Json
}finally{
 foreach($name in $variables){[Environment]::SetEnvironmentVariable($name,$old[$name],'Process')}
 foreach($name in $targets){$path=Join-Path $testRoot $name;if($name -in $retained){Copy-BrassAtomic (Join-Path $out ('prior-test-artifacts/'+$name)) $path}elseif(Test-Path -LiteralPath $path){Remove-Item -LiteralPath $path}}
}
