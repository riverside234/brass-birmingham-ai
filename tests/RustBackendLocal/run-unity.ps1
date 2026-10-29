param([string]$BuildDirectory='artifacts/rust-backend/l13-dev',[switch]$Quick,[switch]$Matched,[ValidateRange(30,480)][int]$MaxSeconds=480,
 [ValidateSet('nn-human','nn-ai','v4-human','v4-two','v4-three','failures','nn-live','v4-live')][string]$ReleaseCase,
 [string]$EvidenceDirectory,[switch]$TurnHistory,[ValidateSet('write','read')][string]$SetupPreferences,[switch]$SearchV2)
$ErrorActionPreference='Stop'
$localRoot=Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$localBuild=Join-Path $localRoot $BuildDirectory
$localOut=if($EvidenceDirectory){Join-Path $localRoot $EvidenceDirectory}else{$localBuild}
if($EvidenceDirectory){if(Test-Path -LiteralPath $localOut){throw 'Evidence directory exists.'};[void](New-Item -ItemType Directory -Path $localOut)}
$localPlayer=Join-Path $localRoot 'artifacts/card-tracker-player'
$localManaged=Join-Path $localPlayer 'BrassBirminghamUnity_Data/Managed'
$localTests=Join-Path $localPlayer 'artifacts/patch-build'
if(Get-Process -Name BrassBirminghamUnity -ErrorAction SilentlyContinue){throw 'Another game/test player is running.'}
$localManifest=Get-Content -LiteralPath (Join-Path $localBuild 'build-manifest.json') -Raw|ConvertFrom-Json
foreach($file in $localManifest.files){
 $source=Join-Path $localBuild ('Managed/'+$file.name)
 if((Get-FileHash -LiteralPath $source).Hash -ne $file.sha256){throw 'Candidate changed.'}
 Copy-Item -LiteralPath $source -Destination (Join-Path $localManaged $file.name) -Force
}
$localNative=[IO.Path]::GetFullPath((Join-Path $localPlayer 'BrassNative'))
$localRetained=[IO.Path]::GetFullPath((Join-Path $localPlayer ('artifacts/native-retained-'+[Guid]::NewGuid().ToString('N'))))
$localBoundary=[IO.Path]::GetFullPath($localPlayer)+'\'
if(-not $localNative.StartsWith($localBoundary,[StringComparison]::OrdinalIgnoreCase) -or -not $localRetained.StartsWith($localBoundary,[StringComparison]::OrdinalIgnoreCase)){throw 'Test package retention escaped isolated player.'}
if(Test-Path -LiteralPath $localNative){Move-Item -LiteralPath $localNative -Destination $localRetained}
Copy-Item -LiteralPath (Join-Path $localBuild 'BrassNative') -Destination $localPlayer -Recurse
$localDll=Join-Path $localTests 'Managed/LocalAiUnityTests.dll'
$localRsp=Join-Path $localOut 'native-tests.rsp'
$localSources=@('tests/BrassLocalAI/LocalAiUnityTests.cs','tests/BrassLocalAI/AiExecutionResourceTests.cs','tests/RustBackendLocal/NativeLocalUnityTests.cs','tests/RustBackendLocal/NativeMatchedUnityTests.cs','tests/RustBackendLocal/NativeValidationLifecycleTests.cs','tests/RustBackendLocal/NativeReleaseUnityTests.cs','tests/RustBackendLocal/ReleaseSegmentUnityTests.cs')
if($TurnHistory){$localSources+=@('tests/BrassPatch/TurnHistory/TurnHistoryRegression.cs','tests/BrassPatch/TurnHistory/TurnHistoryUnityTests.cs')}
if($SetupPreferences){$localSources+=@('tests/BrassLocalAI/SetupPreferencesUnityTests.cs')}
if($SearchV2){$localSources+=@('tests/RustBackendLocal/SearchV2UnityTests.cs')}
$localCompiled=@();$localSourcePins=@()
foreach($source in $localSources){
 $target=Join-Path $localOut ('test-source/'+$source)
 [void](New-Item -ItemType Directory -Path (Split-Path -Parent $target) -Force)
 Copy-Item -LiteralPath (Join-Path $localRoot $source) -Destination $target
 $localCompiled+=$target;$localSourcePins+=@{path=$source;sha256=(Get-FileHash -LiteralPath $target).Hash.ToLowerInvariant()}
}
@('/nologo','/target:library','/langversion:7.3','/nostdlib+','/deterministic+',('/out:"'+$localDll+'"'))+
 @(Get-ChildItem -LiteralPath $localManaged -Filter '*.dll'|Where-Object Name -notmatch 'Tests|Regression'|ForEach-Object{'/reference:"'+$_.FullName+'"'})+
 @($localCompiled|ForEach-Object{'"'+$_+'"'})|Set-Content -LiteralPath $localRsp -Encoding utf8
& (Join-Path $localRoot 'artifacts/patch-tools/microsoft.net.compilers.toolset.4.12.0/tasks/net472/csc.exe') ('@'+$localRsp)
if($LASTEXITCODE -ne 0){throw 'Native test compilation failed.'}
Copy-Item -LiteralPath $localDll -Destination (Join-Path $localOut 'LocalAiUnityTests.dll') -Force
@{sources=$localSourcePins;testDllSha256=(Get-FileHash -LiteralPath $localDll).Hash.ToLowerInvariant();candidateManifestSha256=(Get-FileHash -LiteralPath (Join-Path $localBuild 'build-manifest.json')).Hash.ToLowerInvariant()}|ConvertTo-Json -Depth 5|Set-Content -LiteralPath (Join-Path $localOut 'test-build.json')
$localResult=Join-Path $localTests 'local-ai-native.json'
# Sidecars belong to this invocation; stale prior evidence must never pass it.
foreach($suffix in '.release.json','.permitted.jsonl','.native-turns.json','.failure.json','.turn-history.json'){
 if(Test-Path -LiteralPath ($localResult+$suffix)){Move-Item -LiteralPath ($localResult+$suffix) -Destination ($localResult+$suffix+'.retained-'+[Guid]::NewGuid().ToString('N'))}
}
@{passed=$false;status='running'}|ConvertTo-Json|Set-Content -LiteralPath $localResult
$localLog=Join-Path $localOut 'unity.log'
$localArguments=@('-batchmode','-screen-width','1920','-screen-height','1080','-screen-fullscreen','0','-logFile',('"'+$localLog+'"'),'--brass-local-ai-test','--brass-native-local-test')
if($Quick){$localArguments+='--brass-native-short-test'}
if($Matched){$localArguments+='--brass-native-matched-test'}
if($SearchV2){$localArguments+=@('--brass-native-matched-test','--brass-search-v2-test')}
if($TurnHistory){$localArguments+='--brass-turn-history-test'}
if($SetupPreferences){$localArguments+=('--brass-setup-preferences='+$SetupPreferences)}
if($ReleaseCase){$localArguments+=('--brass-release-case='+$ReleaseCase)}
$localProcess=Start-Process -FilePath (Join-Path $localPlayer 'BrassBirminghamUnity.exe') -ArgumentList $localArguments -WorkingDirectory $localPlayer -WindowStyle Hidden -PassThru
$localTimer=[Diagnostics.Stopwatch]::StartNew()
while(-not $localProcess.WaitForExit(1000)){if($localTimer.Elapsed.TotalSeconds -gt $MaxSeconds){$localProcess.Kill();throw 'Native test watchdog.'}}
Copy-Item -LiteralPath $localResult -Destination (Join-Path $localOut 'unity.json') -Force
$localEvidence=Get-Content -LiteralPath $localResult -Raw|ConvertFrom-Json
foreach($suffix in '.native-turns.json','.release.json','.permitted.jsonl','.failure.json','.turn-history.json'){
 if(Test-Path -LiteralPath ($localResult+$suffix)){Copy-Item -LiteralPath ($localResult+$suffix) -Destination (Join-Path $localOut ('unity'+$suffix))}
}
if(Test-Path -LiteralPath ($localResult+'.native-turns.json')){Copy-Item -LiteralPath ($localResult+'.native-turns.json') -Destination (Join-Path $localOut 'unity-turns.json') -Force}
$localEvidence|Select-Object passed,checks,error|ConvertTo-Json -Depth 4
if(-not $localEvidence.passed){throw 'Native Unity qualification failed.'}
