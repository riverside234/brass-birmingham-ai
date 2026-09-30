$ErrorActionPreference='Stop'
$root=Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$build=Join-Path $root 'artifacts/patch-build'
$managed=Join-Path $build 'Managed'
$manifest=Get-Content -LiteralPath (Join-Path $build 'build-manifest.json') -Raw|ConvertFrom-Json
$output=Join-Path $build 'local-ai-execution.json'
foreach($f in $manifest.files){if((Get-FileHash -LiteralPath (Join-Path $managed $f.name)).Hash -ne $f.sha256){throw 'Execution test candidate changed'}}
@{passed=$false;status='running'}|ConvertTo-Json|Set-Content -LiteralPath $output -Encoding UTF8
$exe=Join-Path $managed 'AiExecutionTests.exe'
$rsp=Join-Path $build 'ai-execution-tests.rsp'
@('/nologo','/target:exe','/langversion:7.3','/nostdlib+','/deterministic+',('/out:"'+$exe+'"'))+@(Get-ChildItem -LiteralPath $managed -Filter '*.dll'|Where-Object {$_.Name -notmatch 'Tests|Regression'}|ForEach-Object {'/reference:"'+$_.FullName+'"'})+@('"'+(Join-Path $PSScriptRoot 'AiExecutionTests.cs')+'"')|Set-Content -LiteralPath $rsp -Encoding UTF8
& (Join-Path $root 'artifacts/patch-tools/microsoft.net.compilers.toolset.4.12.0/tasks/net472/csc.exe') ('@'+$rsp)
if($LASTEXITCODE -ne 0){throw 'Execution test compilation failed'}
& $exe $output
if($LASTEXITCODE -ne 0){throw 'Execution lifecycle tests failed'}
$result=Get-Content -LiteralPath $output -Raw|ConvertFrom-Json
if(-not $result.passed){throw 'Execution lifecycle test report failed'}
foreach($f in $manifest.files){if((Get-FileHash -LiteralPath (Join-Path $managed $f.name)).Hash -ne $f.sha256){throw 'Execution test candidate changed'}}
$result|Add-Member -NotePropertyName testedFiles -NotePropertyValue $manifest.files -Force
$result|ConvertTo-Json -Depth 12|Set-Content -LiteralPath $output -Encoding UTF8
Write-Output ('Execution lifecycle passed: '+$result.checks+' checks.')
