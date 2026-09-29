param([string]$ManagedPath, [switch]$SkipBroadMatrix)
$ErrorActionPreference='Stop'
$root=Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
if(-not $ManagedPath){$ManagedPath=Join-Path $root 'artifacts/patch-build/Managed'}
$ManagedPath=(Resolve-Path -LiteralPath $ManagedPath).Path
$compiler=Join-Path $root 'artifacts/patch-tools/microsoft.net.compilers.toolset.4.12.0/tasks/net472/csc.exe'
$dll=Join-Path $PSScriptRoot 'ResourcesRegression.Tests.dll'
$references=@(Get-ChildItem -LiteralPath $ManagedPath -Filter '*.dll'|ForEach-Object{'/reference:"'+$_.FullName+'"'})
$response=@('/nologo','/target:library','/langversion:7.3','/nostdlib+','/deterministic+',('/out:"'+$dll+'"'))+$references+@(('"'+(Join-Path $PSScriptRoot 'ResourcesRegression.cs')+'"'),('"'+(Join-Path $root 'rules-audit/extended-network-matrix.cs')+'"'))
$rsp=Join-Path $PSScriptRoot 'test-compile.rsp'
$response|Set-Content -LiteralPath $rsp -Encoding UTF8
& $compiler ('@'+$rsp)
if($LASTEXITCODE -ne 0){throw 'Regression test compilation failed.'}
[void][Reflection.Assembly]::LoadFrom((Join-Path $ManagedPath 'Assembly-CSharp.dll'))
[void][Reflection.Assembly]::LoadFrom((Join-Path $ManagedPath 'BrassPatch.dll'))
[void][Reflection.Assembly]::LoadFrom($dll)
$targeted=[ResourcesRegression]::Run()
$broad=$null
if(-not $SkipBroadMatrix){$broad=[ExtendedNetworkMatrix]::Run();if($broad.failures.Count -gt 0){$broad|ConvertTo-Json -Depth 15;throw 'Existing network matrix regressed.'}}
$report=[ordered]@{utc=[DateTime]::UtcNow.ToString('o');assemblySha256=(Get-FileHash (Join-Path $ManagedPath 'Assembly-CSharp.dll')).Hash;helperSha256=(Get-FileHash (Join-Path $ManagedPath 'BrassPatch.dll')).Hash;targeted=$targeted;broad=$broad}
$report|ConvertTo-Json -Depth 16|Set-Content -LiteralPath (Join-Path $PSScriptRoot 'results.json') -Encoding UTF8
$report|ConvertTo-Json -Depth 16
