param([string]$ManagedPath=(Join-Path $PSScriptRoot '../../artifacts/patch-build/Managed'))
$ErrorActionPreference='Stop'
$root=Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$ManagedPath=(Resolve-Path -LiteralPath $ManagedPath).Path
$compiler=Join-Path $root 'artifacts/patch-tools/microsoft.net.compilers.toolset.4.12.0/tasks/net472/csc.exe'
$destination=Join-Path $ManagedPath 'CoreRegression.exe'
$response=@('/nologo','/target:exe','/langversion:7.3','/nostdlib+','/deterministic+',('/out:"'+$destination+'"'))
$response+=@(Get-ChildItem -LiteralPath $ManagedPath -Filter '*.dll'|ForEach-Object{'/reference:"'+$_.FullName+'"'})
$response+='"'+(Join-Path $PSScriptRoot 'CoreRegression.cs')+'"'
$rsp=Join-Path $root 'artifacts/patch-build/core-tests.rsp'
$response|Set-Content -LiteralPath $rsp -Encoding UTF8
& $compiler ('@'+$rsp)
if($LASTEXITCODE -ne 0){throw 'Core tests compilation failed'}
& $destination
if($LASTEXITCODE -ne 0){throw 'Core tests failed'}
