param([switch]$ExistingSuites,[switch]$MenuOnly)
$ErrorActionPreference='Stop'
$root=Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$candidate=Join-Path $root 'artifacts/patch-build/Managed'
$isolated=Join-Path $root 'artifacts/city-locator-player'
if($ExistingSuites){
 # The legacy Mono file APIs impose MAX_PATH on the older persistence fixtures.
 # A workspace-local junction shortens paths without moving or sharing saves.
 $short=Join-Path $root 'artifacts/u'
 if(-not(Test-Path -LiteralPath $short)){New-Item -ItemType Junction -Path $short -Target $isolated|Out-Null}
 if((Get-Item -LiteralPath $short).Target -ne $isolated){throw 'Unexpected isolated-player junction target'}
 $isolated=$short
}
$data=Join-Path $isolated 'BrassBirminghamUnity_Data'
$testBuild=Join-Path $isolated 'artifacts/patch-build'
if(Get-Process BrassBirminghamUnity -ErrorAction SilentlyContinue){throw 'Another game/test player is already running.'}
if(-not(Test-Path -LiteralPath (Join-Path $isolated 'BrassBirminghamUnity.exe'))){throw 'Prepare the isolated player with tests/BrassPatch/CityLocator/run-unity.ps1 first.'}
$compiler=Join-Path $root 'artifacts/patch-tools/microsoft.net.compilers.toolset.4.12.0/tasks/net472/csc.exe'
if(-not $ExistingSuites){
 $rsp=Join-Path $testBuild 'local-ai-native.rsp'
 @('/nologo','/target:library','/langversion:7.3','/nostdlib+','/deterministic+',('/out:"'+(Join-Path $candidate 'LocalAiUnityTests.dll')+'"'))+@(Get-ChildItem -LiteralPath $candidate -Filter '*.dll'|Where-Object { $_.Name -notmatch 'Tests|Regression' }|ForEach-Object{'/reference:"'+$_.FullName+'"'})+@('"'+(Join-Path $PSScriptRoot 'LocalAiUnityTests.cs')+'"')|Set-Content -LiteralPath $rsp -Encoding UTF8
 & $compiler ('@'+$rsp)
 if($LASTEXITCODE -ne 0){throw 'Local AI native test compilation failed'}
}
Get-ChildItem -LiteralPath $candidate -File|Where-Object Extension -in '.dll','.exe'|Copy-Item -Destination (Join-Path $data 'Managed') -Force
Get-ChildItem -LiteralPath $candidate -File|Where-Object Extension -in '.dll','.exe'|Copy-Item -Destination (Join-Path $testBuild 'Managed') -Force
$name=if($ExistingSuites){'unity-selftest-result.json'}else{'local-ai-native.json'}
$published=Join-Path $root ('artifacts/patch-build/'+$(if($ExistingSuites){'unity-isolated-result.json'}elseif($MenuOnly){'local-ai-menu.json'}else{$name}))
@{passed=$false;status='running'}|ConvertTo-Json|Set-Content -LiteralPath $published -Encoding UTF8
$result=Join-Path $testBuild $name
if(Test-Path -LiteralPath $result){Remove-Item -LiteralPath $result}
$flag=if($ExistingSuites){'--brass-patch-selftest'}else{'--brass-local-ai-test'}
$arguments=@('-batchmode','-logFile',('"'+(Join-Path $testBuild ($name+'.log'))+'"'),$flag)
if($MenuOnly){$arguments+='--brass-local-ai-smoke'}
$process=Start-Process -FilePath (Join-Path $isolated 'BrassBirminghamUnity.exe') -ArgumentList $arguments -WorkingDirectory $isolated -WindowStyle Hidden -PassThru
Write-Output "Isolated Unity tests started (PID $($process.Id)); result: $result"
$timer=[Diagnostics.Stopwatch]::StartNew()
while(-not $process.HasExited){if($timer.Elapsed.TotalMinutes -gt 32){$process.Kill();throw 'Isolated Unity tests timed out'};Start-Sleep -Milliseconds 500;$process.Refresh()}
if(-not(Test-Path -LiteralPath $result)){throw 'Unity did not produce a test result'}
$evidence=Get-Content -LiteralPath $result -Raw|ConvertFrom-Json
if(-not $evidence.passed){throw ('Unity checks failed: '+$evidence.error)}
$manifest=Get-Content -LiteralPath (Join-Path $root 'artifacts/patch-build/build-manifest.json') -Raw|ConvertFrom-Json
foreach($file in $manifest.files){if((Get-FileHash -LiteralPath (Join-Path $data ('Managed/'+$file.name))).Hash -ne $file.sha256){throw 'Isolated files do not match current candidate'}}
$evidence|Add-Member -NotePropertyName testedFiles -NotePropertyValue $manifest.files -Force
$evidence|ConvertTo-Json -Depth 12|Set-Content -LiteralPath $published -Encoding UTF8
Write-Output 'Unity checks passed for the fingerprinted candidate.'
