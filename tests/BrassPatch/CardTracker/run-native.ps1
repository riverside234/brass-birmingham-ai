param([switch]$Installed,[ValidateSet('tracker','statistics','presentation','link','menu')][string]$Suite='tracker')
$ErrorActionPreference='Stop'
$trackerRoot=Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
$trackerBuild=Join-Path $trackerRoot 'artifacts/card-tracker-build'
$trackerPlayer=if($Installed){$trackerRoot}else{Join-Path $trackerRoot 'artifacts/card-tracker-player'}
$trackerTestBuild=Join-Path $trackerPlayer 'artifacts/patch-build'
$trackerManaged=Join-Path $trackerPlayer 'BrassBirminghamUnity_Data/Managed'
$trackerManifest=Get-Content -LiteralPath (Join-Path $trackerBuild 'build-manifest.json') -Raw|ConvertFrom-Json
if(Get-Process -Name BrassBirminghamUnity -ErrorAction SilentlyContinue){throw 'Another game/test player is running.'}
foreach($file in $trackerManifest.files){
 $candidate=Join-Path $trackerBuild ('Managed/'+$file.name)
 if((Get-FileHash -LiteralPath $candidate).Hash -ne $file.sha256){throw 'Candidate changed.'}
 if($Installed){if((Get-FileHash -LiteralPath (Join-Path $trackerManaged $file.name)).Hash -ne $file.sha256){throw 'Installed candidate differs.'}}
 else{Copy-Item -LiteralPath $candidate -Destination $trackerManaged -Force}
}
New-Item -ItemType Directory -Path (Join-Path $trackerTestBuild 'Managed') -Force|Out-Null
$trackerSource=switch($Suite){
 'tracker'{Join-Path $PSScriptRoot 'CardTrackerNativeTests.cs'}
 'statistics'{Join-Path $trackerRoot 'artifacts/patch-releases/1.4.10/tests/BrassPatch/Statistics/StatisticsNativeTests.cs'}
 'presentation'{Join-Path $trackerRoot 'tests/BrassPatch/Presentation/PresentationLifecycleTests.cs'}
 'link'{Join-Path $trackerRoot 'tests/BrassPatch/Presentation/LinkReplayTests.cs'}
 'menu'{Join-Path $trackerRoot 'tests/BrassLocalAI/LocalAiUnityTests.cs'}
}
$trackerDll=Join-Path $trackerTestBuild 'Managed/LocalAiUnityTests.dll'
$trackerResult=Join-Path $trackerTestBuild 'local-ai-native.json'
$trackerSavedDll=if($Installed -and (Test-Path -LiteralPath $trackerDll)){[IO.File]::ReadAllBytes($trackerDll)}else{$null}
$trackerSavedResult=if($Installed -and (Test-Path -LiteralPath $trackerResult)){[IO.File]::ReadAllBytes($trackerResult)}else{$null}
try{
$trackerRsp=Join-Path $trackerBuild ($Suite+'-native.rsp')
$trackerSources=@('"'+$trackerSource+'"')
if($Suite -in 'presentation','link'){$trackerSources+='"'+(Join-Path $trackerRoot 'tests/BrassPatch/CityLocator/UnityCityLocatorRegression.cs')+'"'}
if($Suite -eq 'menu'){$trackerSources+='"'+(Join-Path $trackerRoot 'tests/BrassLocalAI/AiExecutionResourceTests.cs')+'"'}
@('/nologo','/target:library','/langversion:7.3','/nostdlib+','/deterministic+',('/out:"'+$trackerDll+'"'))+@(Get-ChildItem -LiteralPath $trackerManaged -Filter '*.dll'|Where-Object {$_.Name -notmatch 'Tests|Regression'}|ForEach-Object {'/reference:"'+$_.FullName+'"'})+$trackerSources|Set-Content -LiteralPath $trackerRsp -Encoding utf8
& (Join-Path $trackerRoot 'artifacts/patch-tools/microsoft.net.compilers.toolset.4.12.0/tasks/net472/csc.exe') ('@'+$trackerRsp)
if($LASTEXITCODE -ne 0){throw 'Native tests compilation failed.'}
$trackerLog=Join-Path $trackerTestBuild ($Suite+'-card-tracker-native.log')
@{passed=$false;status='running'}|ConvertTo-Json|Set-Content -LiteralPath $trackerResult
$trackerArguments=@('-batchmode','-screen-width','1920','-screen-height','1080','-screen-fullscreen','0','-logFile',('"'+$trackerLog+'"'),'--brass-local-ai-test')
if($Suite -eq 'menu'){$trackerArguments+='--brass-local-ai-smoke'}
$trackerProcess=Start-Process -FilePath (Join-Path $trackerPlayer 'BrassBirminghamUnity.exe') -ArgumentList $trackerArguments -WorkingDirectory $trackerPlayer -WindowStyle Hidden -PassThru
$trackerTimer=[Diagnostics.Stopwatch]::StartNew()
while(-not $trackerProcess.WaitForExit(1000)){if($trackerTimer.Elapsed.TotalMinutes -gt 5){$trackerProcess.Kill();throw 'Native tests timed out.'}}
$trackerEvidence=Get-Content -LiteralPath $trackerResult -Raw|ConvertFrom-Json
foreach($file in $trackerManifest.files){if((Get-FileHash -LiteralPath (Join-Path $trackerManaged $file.name)).Hash -ne $file.sha256){throw 'Runtime changed.'}}
$trackerEvidence|Add-Member -NotePropertyName testedFiles -NotePropertyValue $trackerManifest.files -Force
$trackerEvidence|ConvertTo-Json -Depth 18|Set-Content -LiteralPath (Join-Path $trackerBuild ($Suite+$(if($Installed){'-installed'}else{'-native'})+'.json')) -Encoding utf8
$trackerEvidence|Select-Object passed,checks,failures,error|ConvertTo-Json
if(-not $trackerEvidence.passed){throw ($Suite+' native qualification failed.')}
}finally{
 if($Installed){
  if($null -ne $trackerSavedDll){[IO.File]::WriteAllBytes($trackerDll,$trackerSavedDll)}elseif(Test-Path -LiteralPath $trackerDll){Remove-Item -LiteralPath $trackerDll}
  if($null -ne $trackerSavedResult){[IO.File]::WriteAllBytes($trackerResult,$trackerSavedResult)}elseif(Test-Path -LiteralPath $trackerResult){Remove-Item -LiteralPath $trackerResult}
 }
}
