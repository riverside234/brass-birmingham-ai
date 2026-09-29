param([switch]$Baseline)
$ErrorActionPreference='Stop'
$root=Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
$build=Join-Path $root 'artifacts/patch-build'
$managed=Join-Path $build 'Managed'
$isolated=Join-Path $root 'artifacts/city-locator-player'
$testBuild=Join-Path $isolated 'artifacts/patch-build'
$data=Join-Path $isolated 'BrassBirminghamUnity_Data/Managed'
if(Get-Process BrassBirminghamUnity -ErrorAction SilentlyContinue){throw 'Game/test player is running'}
Get-ChildItem -LiteralPath $managed -Filter '*.dll'|Copy-Item -Destination $data -Force
Get-ChildItem -LiteralPath $managed -Filter '*.dll'|Copy-Item -Destination (Join-Path $testBuild 'Managed') -Force
$manifest=Get-Content -LiteralPath (Join-Path $build 'build-manifest.json') -Raw|ConvertFrom-Json
if($Baseline){
 $archive=Join-Path $root 'artifacts/patch-releases/1.4.1'
 $manifest=Get-Content -LiteralPath (Join-Path $archive 'build-manifest.json') -Raw|ConvertFrom-Json
 foreach($f in $manifest.files){Copy-Item -LiteralPath (Join-Path $archive $f.name) -Destination $data -Force;Copy-Item -LiteralPath (Join-Path $archive $f.name) -Destination (Join-Path $testBuild 'Managed') -Force}
}
$dll=Join-Path $testBuild 'Managed/LocalAiUnityTests.dll'
$rsp=Join-Path $testBuild 'presentation.rsp'
@('/nologo','/target:library','/langversion:7.3','/nostdlib+','/deterministic+',('/out:"'+$dll+'"'))+@(Get-ChildItem -LiteralPath $data -Filter '*.dll'|Where-Object {$_.Name -notmatch 'Tests|Regression'}|ForEach-Object {'/reference:"'+$_.FullName+'"'})+@('"'+(Join-Path $PSScriptRoot 'PresentationLifecycleTests.cs')+'"','"'+(Join-Path $root 'tests/BrassPatch/CityLocator/UnityCityLocatorRegression.cs')+'"')|Set-Content -LiteralPath $rsp -Encoding UTF8
& (Join-Path $root 'artifacts/patch-tools/microsoft.net.compilers.toolset.4.12.0/tasks/net472/csc.exe') ('@'+$rsp)
if($LASTEXITCODE -ne 0){throw 'Presentation tests compilation failed'}
Copy-Item -LiteralPath $dll -Destination $data -Force
$result=Join-Path $testBuild 'local-ai-native.json'
@{passed=$false;status='running'}|ConvertTo-Json|Set-Content -LiteralPath $result
$process=Start-Process -FilePath (Join-Path $isolated 'BrassBirminghamUnity.exe') -ArgumentList @('-batchmode','-logFile',('"'+(Join-Path $testBuild 'presentation.log')+'"'),'--brass-local-ai-test') -WorkingDirectory $isolated -WindowStyle Hidden -PassThru
if(-not $process.WaitForExit(120000)){Stop-Process -Id $process.Id;throw 'Presentation tests timed out'}
$evidence=Get-Content -LiteralPath $result -Raw|ConvertFrom-Json
foreach($f in $manifest.files){if((Get-FileHash -LiteralPath (Join-Path $data $f.name)).Hash -ne $f.sha256){throw 'Runtime changed'}}
$evidence|Add-Member -NotePropertyName testedFiles -NotePropertyValue $manifest.files -Force
$out=Join-Path $build $(if($Baseline){'presentation-baseline.json'}else{'presentation-lifecycle.json'})
$evidence|ConvertTo-Json -Depth 14|Set-Content -LiteralPath $out -Encoding UTF8
$evidence|Select-Object passed,checks,failures,error|ConvertTo-Json
if(-not $evidence.passed -and -not $Baseline){throw 'Presentation lifecycle regressions failed'}
