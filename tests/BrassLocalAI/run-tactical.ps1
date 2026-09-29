$ErrorActionPreference='Stop'
$root=Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$build=Join-Path $root 'artifacts/patch-build'
$managed=Join-Path $build 'Managed'
$runtime=Join-Path $root 'artifacts/local-ai-tactical-runtime'
$output=Join-Path $build 'local-ai-tactical.json'
$manifest=Get-Content -LiteralPath (Join-Path $build 'build-manifest.json') -Raw|ConvertFrom-Json

# Pin the exact candidate before copying, so an old runtime cannot be certified.
foreach($file in $manifest.files){
 if((Get-FileHash -LiteralPath (Join-Path $managed $file.name)).Hash -ne $file.sha256){throw 'Candidate fingerprint differs before tactical tests'}
}
@{passed=$false;status='running';testedFiles=$manifest.files}|ConvertTo-Json -Depth 12|Set-Content -LiteralPath $output -Encoding UTF8
New-Item -ItemType Directory -Path $runtime -Force|Out-Null
Get-ChildItem -LiteralPath $managed -Filter '*.dll'|Copy-Item -Destination $runtime -Force
$exe=Join-Path $runtime 'TacticalIntentTests.exe'
$rsp=Join-Path $runtime 'tactical-tests.rsp'
@('/nologo','/target:exe','/langversion:7.3','/nostdlib+','/deterministic+',('/out:"'+$exe+'"'))+@(Get-ChildItem -LiteralPath $runtime -Filter '*.dll'|Where-Object {$_.Name -notmatch 'Tests|Regression'}|ForEach-Object {'/reference:"'+$_.FullName+'"'})+@('"'+(Join-Path $PSScriptRoot 'TacticalIntentTests.cs')+'"')|Set-Content -LiteralPath $rsp -Encoding UTF8
& (Join-Path $root 'artifacts/patch-tools/microsoft.net.compilers.toolset.4.12.0/tasks/net472/csc.exe') ('@'+$rsp)
if($LASTEXITCODE -ne 0){throw 'Tactical intent test compilation failed'}
$runtimeOutput=Join-Path $runtime 'tactical-result.json'
@{passed=$false;status='running'}|ConvertTo-Json|Set-Content -LiteralPath $runtimeOutput -Encoding UTF8
& $exe $runtimeOutput
$testExit=$LASTEXITCODE
if($testExit -ne 0){throw 'Tactical intent test process failed'}
$result=Get-Content -LiteralPath $runtimeOutput -Raw|ConvertFrom-Json
if(-not $result.passed){throw 'Tactical intent tests failed or produced no fresh success'}
foreach($file in $manifest.files){
 if((Get-FileHash -LiteralPath (Join-Path $runtime $file.name)).Hash -ne $file.sha256){throw 'Tactical test runtime changed'}
 if((Get-FileHash -LiteralPath (Join-Path $managed $file.name)).Hash -ne $file.sha256){throw 'Candidate changed during tactical tests'}
}
$result|Add-Member -NotePropertyName testedFiles -NotePropertyValue $manifest.files -Force
$result|ConvertTo-Json -Depth 12|Set-Content -LiteralPath $output -Encoding UTF8
