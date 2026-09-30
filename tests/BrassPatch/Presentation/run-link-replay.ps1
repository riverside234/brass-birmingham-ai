param([switch]$Baseline,[switch]$Installed)
$ErrorActionPreference='Stop'
$root=Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
$build=Join-Path $root 'artifacts/patch-build'
$isolated=Join-Path $root 'artifacts/city-locator-player'
$testBuild=Join-Path $isolated 'artifacts/patch-build'
$data=Join-Path $isolated 'BrassBirminghamUnity_Data/Managed'
if(Get-Process BrassBirminghamUnity -ErrorAction SilentlyContinue){throw 'Game/test player is running'}
if($Installed -and $Baseline){throw 'Installed baseline checks are unsupported'}
if($Installed){$isolated=$root;$testBuild=$build;$data=Join-Path $root 'BrassBirminghamUnity_Data/Managed'}
$package=if($Baseline){Join-Path $root 'artifacts/patch-releases/1.4.5'}else{$build}
$manifest=Get-Content -LiteralPath (Join-Path $package 'build-manifest.json') -Raw|ConvertFrom-Json
foreach($f in $manifest.files){
 $source=Join-Path $package $(if($Baseline){$f.name}else{'Managed/'+$f.name})
 if((Get-FileHash -LiteralPath $source).Hash -ne $f.sha256){throw 'Package changed'}
 if($Installed){
  if((Get-FileHash -LiteralPath (Join-Path $data $f.name)).Hash -ne $f.sha256){throw 'Installed runtime differs from candidate'}
 }else{
  Copy-Item -LiteralPath $source -Destination $data -Force
  Copy-Item -LiteralPath $source -Destination (Join-Path $testBuild 'Managed') -Force
 }
}
$dll=Join-Path $testBuild 'Managed/LocalAiUnityTests.dll'
$rsp=Join-Path $testBuild 'link-replay.rsp'
@('/nologo','/target:library','/langversion:7.3','/nostdlib+','/deterministic+',('/out:"'+$dll+'"'))+@(Get-ChildItem -LiteralPath $data -Filter '*.dll'|Where-Object {$_.Name -notmatch 'Tests|Regression'}|ForEach-Object {'/reference:"'+$_.FullName+'"'})+@('"'+(Join-Path $PSScriptRoot 'LinkReplayTests.cs')+'"','"'+(Join-Path $root 'tests/BrassPatch/CityLocator/UnityCityLocatorRegression.cs')+'"')|Set-Content -LiteralPath $rsp -Encoding UTF8
& (Join-Path $root 'artifacts/patch-tools/microsoft.net.compilers.toolset.4.12.0/tasks/net472/csc.exe') ('@'+$rsp)
if($LASTEXITCODE -ne 0){throw 'Link replay tests compilation failed'}
if(-not $Installed){Copy-Item -LiteralPath $dll -Destination $data -Force}
$result=Join-Path $testBuild 'local-ai-native.json'
if($Installed){$savedResult=if(Test-Path -LiteralPath $result){[IO.File]::ReadAllBytes($result)}else{$null}}
try{
@{passed=$false;status='running'}|ConvertTo-Json|Set-Content -LiteralPath $result
$arguments=@('-batchmode','-logFile',('"'+(Join-Path $testBuild 'link-replay.log')+'"'),'--brass-local-ai-test')
if($Baseline){$arguments+='--brass-link-replay-baseline'}
$process=Start-Process -FilePath (Join-Path $isolated 'BrassBirminghamUnity.exe') -ArgumentList $arguments -WorkingDirectory $isolated -WindowStyle Hidden -PassThru
$timer=[Diagnostics.Stopwatch]::StartNew()
while(-not $process.HasExited){if($timer.Elapsed.TotalSeconds -gt 120){$process.Kill();throw 'Link replay tests timed out'};Start-Sleep -Milliseconds 250;$process.Refresh()}
$evidence=Get-Content -LiteralPath $result -Raw|ConvertFrom-Json
foreach($f in $manifest.files){if((Get-FileHash -LiteralPath (Join-Path $data $f.name)).Hash -ne $f.sha256){throw 'Runtime changed'}}
$evidence|Add-Member -NotePropertyName testedFiles -NotePropertyValue $manifest.files -Force
$out=Join-Path $build $(if($Installed){'link-replay-installed.json'}elseif($Baseline){'link-replay-baseline.json'}else{'link-replay.json'})
if($Installed){$evidence|Add-Member -NotePropertyName installedLocation -NotePropertyValue $root -Force}
$evidence|ConvertTo-Json -Depth 14|Set-Content -LiteralPath $out -Encoding UTF8
$evidence|Select-Object passed,checks,failures,error|ConvertTo-Json
if(-not $evidence.passed -and -not $Baseline){throw 'Link replay regressions failed'}
}finally{if($Installed){if($null -ne $savedResult){[IO.File]::WriteAllBytes($result,$savedResult)}elseif(Test-Path -LiteralPath $result){Remove-Item -LiteralPath $result}}}
