param([switch]$Installed)
$ErrorActionPreference='Stop'
$root=Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
$build=Join-Path $root 'artifacts/patch-build'
$player=if($Installed){$root}else{Join-Path $root 'artifacts/city-locator-player'}
$testBuild=Join-Path $player 'artifacts/patch-build'
$data=Join-Path $player 'BrassBirminghamUnity_Data/Managed'
$manifest=Get-Content -LiteralPath (Join-Path $build 'build-manifest.json') -Raw|ConvertFrom-Json
if(Get-Process BrassBirminghamUnity -ErrorAction SilentlyContinue){throw 'Another game/test player is running.'}
foreach($file in $manifest.files){
 $source=Join-Path $build ('Managed/'+$file.name)
 if((Get-FileHash -LiteralPath $source).Hash -ne $file.sha256){throw 'Candidate changed'}
 if($Installed){if((Get-FileHash -LiteralPath (Join-Path $data $file.name)).Hash -ne $file.sha256){throw 'Installed candidate differs'}}
 else{Copy-Item -LiteralPath $source -Destination $data -Force}
}
$dll=Join-Path $testBuild 'Managed/LocalAiUnityTests.dll'
$rsp=Join-Path $testBuild 'statistics-native.rsp'
@('/nologo','/target:library','/langversion:7.3','/nostdlib+','/deterministic+',('/out:"'+$dll+'"'))+@(Get-ChildItem -LiteralPath $data -Filter '*.dll'|Where-Object {$_.Name -notmatch 'Tests|Regression'}|ForEach-Object {'/reference:"'+$_.FullName+'"'})+@('"'+(Join-Path $PSScriptRoot 'StatisticsNativeTests.cs')+'"')|Set-Content -LiteralPath $rsp -Encoding UTF8
& (Join-Path $root 'artifacts/patch-tools/microsoft.net.compilers.toolset.4.12.0/tasks/net472/csc.exe') ('@'+$rsp)
if($LASTEXITCODE -ne 0){throw 'Statistics tests compilation failed'}
if(-not $Installed){Copy-Item -LiteralPath $dll -Destination $data -Force}
$result=Join-Path $testBuild 'local-ai-native.json'
$prior=if(Test-Path -LiteralPath $result){[IO.File]::ReadAllBytes($result)}else{$null}
try {
 @{passed=$false;status='running'}|ConvertTo-Json|Set-Content -LiteralPath $result
 $log=Join-Path $testBuild 'statistics-native.log'
 # A graphical batch player renders real canvases without a visible window.
 $process=Start-Process -FilePath (Join-Path $player 'BrassBirminghamUnity.exe') -ArgumentList @('-batchmode','-screen-width','1280','-screen-height','720','-screen-fullscreen','0','-logFile',('"'+$log+'"'),'--brass-local-ai-test') -WorkingDirectory $player -WindowStyle Hidden -PassThru
 if(-not $process.WaitForExit(240000)){$process.Kill();throw 'Statistics tests timed out'}
 $evidence=Get-Content -LiteralPath $result -Raw|ConvertFrom-Json
 foreach($file in $manifest.files){if((Get-FileHash -LiteralPath (Join-Path $data $file.name)).Hash -ne $file.sha256){throw 'Runtime changed'}}
 $evidence|Add-Member -NotePropertyName testedFiles -NotePropertyValue $manifest.files -Force
 $out=Join-Path $build $(if($Installed){'statistics-installed.json'}else{'statistics-native.json'})
 $evidence|ConvertTo-Json -Depth 16|Set-Content -LiteralPath $out -Encoding UTF8
 $evidence|Select-Object passed,checks,failures,matches,pages,error|ConvertTo-Json
 if(-not $evidence.passed){throw 'Statistics qualification failed'}
} finally {if($null -ne $prior){[IO.File]::WriteAllBytes($result,$prior)}}
