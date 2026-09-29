param([string]$ManagedDirectory = (Join-Path (Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))) 'artifacts/patch-build/Managed'))
$ErrorActionPreference = 'Stop'
$workspace = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
$compiler = Join-Path $workspace 'artifacts/patch-tools/microsoft.net.compilers.toolset.4.12.0/tasks/net472/csc.exe'
$destination = Join-Path $ManagedDirectory 'StateRegression.exe'
$response = @('/nologo', '/target:exe', '/langversion:7.3', '/nostdlib+', '/deterministic+', ('/out:"'+$destination+'"'))
$response += @(Get-ChildItem -LiteralPath $ManagedDirectory -Filter '*.dll' | ForEach-Object { '/reference:"'+$_.FullName+'"' })
$response += '"'+(Join-Path $PSScriptRoot 'StateRegression.cs')+'"'
$responseFile = Join-Path $PSScriptRoot 'compile.rsp'
$response | Set-Content -LiteralPath $responseFile -Encoding UTF8
& $compiler ('@'+$responseFile)
if ($LASTEXITCODE -ne 0) { throw 'State test compilation failed' }
& $destination 2>&1 | Tee-Object -FilePath (Join-Path $PSScriptRoot 'results.txt')
if ($LASTEXITCODE -ne 0) { throw 'State regressions failed' }
