param(
 [ValidateSet(8,16)][int]$SearchK=16,
 [ValidateSet(1.0,1.5)][double]$Temperature=1.0,
 [string]$EvidenceDirectory,
 [string]$HarnessDirectory='artifacts/search-v1-default/live-k16'
)
$ErrorActionPreference='Stop'
. (Join-Path $PSScriptRoot 'patch-install-common.ps1')
$root=Split-Path -Parent $PSScriptRoot
if(-not $EvidenceDirectory){$EvidenceDirectory='artifacts/search-v1-default/installed-k'+$SearchK}
$out=[IO.Path]::GetFullPath((Join-Path $root $EvidenceDirectory))
if(-not $out.StartsWith((Join-Path $root 'artifacts/search-v1-default')+'\',[StringComparison]::OrdinalIgnoreCase)){throw 'Smoke output escapes artifacts.'}
if(Test-Path -LiteralPath $out){throw 'Preserve existing smoke evidence.'}
if(Get-Process -Name BrassBirminghamUnity,brass-ai-local -ErrorAction SilentlyContinue){throw 'Game or native worker is running.'}
$record=Join-Path $root 'artifacts/patch-installation.json'
$installed=Read-BrassJson $record
if($installed.version -ne '1.4.42-rust-local' -or $installed.searchMethod -ne 'search_v1' -or $installed.v205Enabled){throw 'Installed search_v1 default release required.'}
$candidate=Join-Path $root 'artifacts/search-v1-default/candidate'
$m=Read-BrassJson (Join-Path $candidate 'build-manifest.json')
$q=Read-BrassJson (Join-Path $candidate 'qualification.json')
if(-not $q.passed -or (Get-FileHash -LiteralPath (Join-Path $candidate 'qualification.json')).Hash -ne $installed.searchV1DefaultQualificationSha256){throw 'Installed qualification identity changed.'}
$pins=@($m.files|ForEach-Object{@{path=('BrassBirminghamUnity_Data/Managed/'+$_.name);sha256=$_.sha256}})
$pins+=@($m.nativeFiles|ForEach-Object{@{path=('BrassNative/'+$_.path);sha256=$_.sha256}})
function VerifyRuntime {
 foreach($p in $pins){if((Get-FileHash -LiteralPath (Join-Path $root $p.path)).Hash -ne $p.sha256){throw ('Installed runtime changed: '+$p.path)}}
 if(@(Get-ChildItem -LiteralPath (Join-Path $root 'BrassNative') -Recurse -File).Count -ne @($m.nativeFiles).Count){throw 'Unexpected native file.'}
}
VerifyRuntime
$source=Join-Path $root $HarnessDirectory
$test=Read-BrassJson (Join-Path $source 'test-build.json')
$dll=Join-Path $source 'LocalAiUnityTests.dll'
if((Get-FileHash -LiteralPath $dll).Hash -ne $test.testDllSha256 -or $test.candidateManifestSha256 -ne $q.build_manifest_sha256){throw 'Test harness identity changed.'}
[void](New-Item -ItemType Directory -Path $out)
$testRoot=Join-Path $root 'artifacts/patch-build'
$targets=@('Managed/LocalAiUnityTests.dll','local-ai-native.json','local-ai-cold-menu.png')
$retained=@()
foreach($name in $targets){
 $path=Join-Path $testRoot $name
 if(Test-Path -LiteralPath $path){
  $backup=Join-Path $out ('prior-test-artifacts/'+$name)
  [void](New-Item -ItemType Directory -Path (Split-Path -Parent $backup) -Force)
  Copy-Item -LiteralPath $path -Destination $backup;$retained+=$name
 }
}
$variables=@('BRASS_SEARCH_V2_RESULT','BRASS_SEARCH_V2_EVENTS','BRASS_SEARCH_V2_MS','BRASS_SEARCH_V2_SCENARIO','BRASS_SEARCH_K','BRASS_NN_TEMPERATURE','BRASS_L14_TRACE')
$oldEnv=@{};foreach($name in $variables){$oldEnv[$name]=[Environment]::GetEnvironmentVariable($name,'Process')}
try{
 Copy-BrassAtomic $dll (Join-Path $testRoot 'Managed/LocalAiUnityTests.dll')
 Write-BrassJsonAtomic (Join-Path $testRoot 'local-ai-native.json') @{passed=$false;status='running-installed-NN-selector-smoke'}
 [Environment]::SetEnvironmentVariable('BRASS_SEARCH_V2_RESULT',(Join-Path $out 'result.json'),'Process')
 [Environment]::SetEnvironmentVariable('BRASS_SEARCH_V2_EVENTS',(Join-Path $root 'artifacts/rust-backend/20261026-l14-study-v2/trusted-live-fixtures/live-92781-CanalEra-2.json'),'Process')
 [Environment]::SetEnvironmentVariable('BRASS_SEARCH_V2_MS','4000','Process')
 [Environment]::SetEnvironmentVariable('BRASS_SEARCH_V2_SCENARIO','normal','Process')
 [Environment]::SetEnvironmentVariable('BRASS_SEARCH_K',([string]$SearchK),'Process')
 [Environment]::SetEnvironmentVariable('BRASS_NN_TEMPERATURE',$Temperature.ToString('0.0',[Globalization.CultureInfo]::InvariantCulture),'Process')
 [Environment]::SetEnvironmentVariable('BRASS_L14_TRACE',$null,'Process')
 $arguments=@('-batchmode','-screen-width','1920','-screen-height','1080','-screen-fullscreen','0','-logFile',('"'+(Join-Path $out 'unity.log')+'"'),
  '--brass-local-ai-test','--brass-native-local-test','--brass-native-matched-test','--brass-search-v2-test')
 $process=Start-Process -FilePath (Join-Path $root 'BrassBirminghamUnity.exe') -ArgumentList $arguments -WorkingDirectory $root -WindowStyle Hidden -PassThru
 $timer=[Diagnostics.Stopwatch]::StartNew()
 while(-not $process.WaitForExit(1000)){if($timer.Elapsed.TotalSeconds -gt 60){$process.Kill();$process.WaitForExit();throw 'Installed search_v1 default smoke watchdog.'}}
 Copy-Item -LiteralPath (Join-Path $testRoot 'local-ai-native.json') -Destination (Join-Path $out 'unity.json')
 Copy-Item -LiteralPath (Join-Path $testRoot 'local-ai-cold-menu.png') -Destination (Join-Path $out 'menu.png')
 $unity=Read-BrassJson (Join-Path $out 'unity.json');$result=Read-BrassJson (Join-Path $out 'result.json')
 if($process.ExitCode -ne 0 -or -not $unity.passed -or -not $result.passed -or $result.search_k -ne $SearchK -or $result.search_method -ne 'search_v1' -or $result.focused){throw 'Installed search_v1 default Unity smoke failed.'}
 foreach($reply in $result.replies){
  if($reply.requested_k -ne $SearchK -or $reply.policy_temperature -ne $Temperature -or $reply.temperature_scope -ne 'all_nn_policy_evaluations' -or $null -ne $reply.search_revision -or $reply.sessions -ne 2 -or $reply.threads_per_session -ne 1){throw 'Installed search settings differ from selection.'}
 }
 if(-not $result.replies.Count -or $result.commits.Count -ne 2 -or $result.commits[-1].remaining -ne 0 -or $result.delivery_ms -ge 4000){throw 'Incomplete/late actual turn.'}
 VerifyRuntime
 Write-BrassJsonAtomic (Join-Path $out 'verification.json') ([ordered]@{
  passed=$true;version=$installed.version;scope=('installed-player/4s-K'+$SearchK+'/one-complete-turn/isolated-save');
  gameRoot=$root;checks=$unity.checks;searchK=$SearchK;temperature=$Temperature;actions=$result.commits.Count;deliveryMs=$result.delivery_ms;
  searchMethod='search_v1';v205Enabled=$false;testedFiles=$pins;testDllSha256=$test.testDllSha256;
  qualificationSha256=$installed.searchV1DefaultQualificationSha256;resultSha256=(Get-FileHash -LiteralPath (Join-Path $out 'result.json')).Hash
 })
 $key='searchV1DefaultInstalledK'+$SearchK
 $installed|Add-Member -NotePropertyName $key -NotePropertyValue (Join-Path $out 'verification.json') -Force
 $installed|Add-Member -NotePropertyName ($key+'Sha256') -NotePropertyValue ((Get-FileHash -LiteralPath (Join-Path $out 'verification.json')).Hash) -Force
 $installed|Add-Member -NotePropertyName searchV1DefaultInstalledSmoke -NotePropertyValue 'passed/4s/one-complete-turn' -Force
 Write-BrassJsonAtomic $record $installed
 $result|Select-Object passed,search_k,temperature,requests,delivery_ms|ConvertTo-Json
}finally{
 foreach($name in $variables){[Environment]::SetEnvironmentVariable($name,$oldEnv[$name],'Process')}
 foreach($name in $targets){
  $path=Join-Path $testRoot $name
  if($name -in $retained){Copy-BrassAtomic (Join-Path $out ('prior-test-artifacts/'+$name)) $path}
  elseif(Test-Path -LiteralPath $path){Remove-Item -LiteralPath $path}
 }
}
