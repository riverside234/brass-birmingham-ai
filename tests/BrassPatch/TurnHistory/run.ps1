param([string]$BuildDirectory='artifacts/city-cancel-fix/candidate',[switch]$Reproduce)
$ErrorActionPreference='Stop'
$root=Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
$build=Join-Path $root $BuildDirectory
$managed=Join-Path $build 'Managed'
$compiler=Join-Path $root 'artifacts/patch-tools/microsoft.net.compilers.toolset.4.12.0/tasks/net472/csc.exe'
$exe=Join-Path $managed 'TurnHistoryRegression.exe'
$rsp=Join-Path $build 'turn-history-tests.rsp'
@('/nologo','/target:exe','/langversion:7.3','/nostdlib+','/deterministic+',('/out:"'+$exe+'"'))+
 @(Get-ChildItem -LiteralPath $managed -Filter '*.dll'|Where-Object Name -notmatch 'Tests|Regression'|ForEach-Object{'/reference:"'+$_.FullName+'"'})+
 ('"'+(Join-Path $PSScriptRoot 'TurnHistoryRegression.cs')+'"')|Set-Content -LiteralPath $rsp -Encoding utf8
& $compiler ('@'+$rsp)
if($LASTEXITCODE -ne 0){throw 'Turn history regression compilation failed.'}
$arguments=@((Join-Path $build 'turn-history-tests.json'))
if($Reproduce){$arguments+='--reproduce'}
& $exe @arguments
if($LASTEXITCODE -ne 0){throw 'Turn history regression failed.'}
