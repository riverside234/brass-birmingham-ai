param([switch]$Render)
$ErrorActionPreference='Stop'
$root=Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
$candidate=Join-Path $root 'artifacts/patch-build/Managed'
$isolated=Join-Path $root 'artifacts/city-locator-player'
$data=Join-Path $isolated 'BrassBirminghamUnity_Data'
$testBuild=Join-Path $isolated 'artifacts/patch-build'
if(Get-Process BrassBirminghamUnity -ErrorAction SilentlyContinue){throw 'Close the game before running the isolated Unity tests.'}
New-Item -ItemType Directory -Path $isolated,$data,(Join-Path $data 'Managed'),(Join-Path $testBuild 'Managed') -Force|Out-Null
# Only immutable engine/assets are shared. All patched/test binaries are private copies.
foreach($file in Get-ChildItem -LiteralPath $root -File|Where-Object Name -in @('BrassBirminghamUnity.exe','UnityPlayer.dll','UnityCrashHandler64.exe')){
 $target=Join-Path $isolated $file.Name
 if(-not(Test-Path -LiteralPath $target)){New-Item -ItemType HardLink -Path $target -Target $file.FullName|Out-Null}
}
foreach($file in Get-ChildItem -LiteralPath (Join-Path $root 'BrassBirminghamUnity_Data') -File -Recurse|Where-Object {$_.FullName -notlike '*\Managed\*'}){
 $relative=$file.FullName.Substring((Join-Path $root 'BrassBirminghamUnity_Data').Length+1)
 $target=Join-Path $data $relative
 if(-not(Test-Path -LiteralPath $target)){New-Item -ItemType Directory -Path (Split-Path -Parent $target) -Force|Out-Null;New-Item -ItemType HardLink -Path $target -Target $file.FullName|Out-Null}
}
Copy-Item -LiteralPath (Join-Path $root 'MonoBleedingEdge') -Destination $isolated -Recurse -Force
$compiler=Join-Path $root 'artifacts/patch-tools/microsoft.net.compilers.toolset.4.12.0/tasks/net472/csc.exe'
$rsp=Join-Path $testBuild 'city-tests.rsp'
@('/nologo','/target:library','/langversion:7.3','/nostdlib+','/deterministic+',('/out:"'+(Join-Path $candidate 'UnityCityLocatorRegression.dll')+'"'))+@(Get-ChildItem -LiteralPath $candidate -Filter '*.dll'|Where-Object Name -notin @('UnityCityLocatorRegression.dll')|ForEach-Object{'/reference:"'+$_.FullName+'"'})+@('"'+(Join-Path $PSScriptRoot 'UnityCityLocatorRegression.cs')+'"')|Set-Content -LiteralPath $rsp -Encoding UTF8
& $compiler ('@'+$rsp)
if($LASTEXITCODE -ne 0){throw 'City locator test compilation failed'}
Get-ChildItem -LiteralPath $candidate -Filter '*.dll' | Copy-Item -Destination (Join-Path $data 'Managed') -Force
Copy-Item -LiteralPath (Join-Path $candidate 'UnityCityLocatorRegression.dll') -Destination (Join-Path $testBuild 'Managed') -Force
$result=Join-Path $testBuild $(if($Render){'city-render-result.json'}else{'unity-selftest-result.json'})
if(Test-Path -LiteralPath $result){Remove-Item -LiteralPath $result}
$arguments=@('-batchmode','-logFile',('"'+(Join-Path $testBuild 'city-unity.log')+'"'))
if($Render){$arguments+='--brass-city-locator-render'}else{$arguments+=@('-nographics','--brass-city-locator-test')}
$process=Start-Process -FilePath (Join-Path $isolated 'BrassBirminghamUnity.exe') -ArgumentList $arguments -WorkingDirectory $isolated -WindowStyle Hidden -PassThru
Write-Output "Isolated city locator tests started (PID $($process.Id)); result: $result"
$timer=[Diagnostics.Stopwatch]::StartNew()
while(-not $process.HasExited){
 if($timer.Elapsed.TotalMinutes -gt 4){$process.Kill();throw 'Isolated city locator tests timed out'}
 Start-Sleep -Milliseconds 500
 $process.Refresh()
}
if(-not(Test-Path -LiteralPath $result)){throw 'Unity did not produce a city locator test result'}
$evidence=Get-Content -LiteralPath $result -Raw|ConvertFrom-Json
if(-not $evidence.passed){throw ('City locator checks failed: '+$evidence.error)}
$manifest=Get-Content -LiteralPath (Join-Path $root 'artifacts/patch-build/build-manifest.json') -Raw|ConvertFrom-Json
foreach($file in $manifest.files){if((Get-FileHash -LiteralPath (Join-Path $data ('Managed/'+$file.name))).Hash -ne $file.sha256){throw 'Isolated files do not match the current candidate'}}
$evidence|Add-Member -NotePropertyName testedFiles -NotePropertyValue $manifest.files -Force
$evidence|ConvertTo-Json -Depth 8|Set-Content -LiteralPath (Join-Path $root ('artifacts/patch-build/'+$(if($Render){'city-render-result.json'}else{'city-native-result.json'}))) -Encoding UTF8
Write-Output 'City locator Unity checks passed for the fingerprinted candidate.'
