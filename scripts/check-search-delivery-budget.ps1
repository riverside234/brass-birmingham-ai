$ErrorActionPreference='Stop'
$root=Split-Path -Parent $PSScriptRoot
$study=Join-Path $root 'artifacts/search-delivery-grace/20261029'
$managed=Join-Path $study 'release-1.4.45/Managed'
$out=Join-Path $study 'budget-checks'
if(Test-Path -LiteralPath $out){throw 'Preserve budget check evidence.'}
[void](New-Item -ItemType Directory -Path $out)
$source=Join-Path $root 'tests/RustBackendLocal/SearchDeliveryBudgetChecks.cs'
Copy-Item -LiteralPath $source -Destination $out
$exe=Join-Path $out 'SearchDeliveryBudgetChecks.exe';$rsp=Join-Path $out 'compile.rsp'
@('/nologo','/target:exe','/main:SearchDeliveryBudgetChecks','/langversion:7.3','/nostdlib+','/optimize+',('/out:"'+$exe+'"'))+
 @(Get-ChildItem -LiteralPath $managed -Filter '*.dll'|Where-Object Name -notmatch 'Tests|Regression'|ForEach-Object{'/reference:"'+$_.FullName+'"'})+
 @('"'+(Join-Path $out 'SearchDeliveryBudgetChecks.cs')+'"')|Set-Content -LiteralPath $rsp -Encoding utf8
& (Join-Path $root 'artifacts/patch-tools/microsoft.net.compilers.toolset.4.12.0/tasks/net472/csc.exe') ('@'+$rsp)
if($LASTEXITCODE -ne 0){throw 'Budget check compile failed.'}
$hostExe=Join-Path $root 'artifacts/host-delivery/20261029/probe-candidate/allocation-host.exe'
& $hostExe (Join-Path $root 'MonoBleedingEdge/EmbedRuntime/mono-2.0-bdwgc.dll') $managed (Join-Path $root 'MonoBleedingEdge/etc') $exe (Join-Path $out 'result.json') unused
if($LASTEXITCODE -ne 0){throw 'Budget checks failed.'}
Get-Content -LiteralPath (Join-Path $out 'result.json') -Raw|ConvertFrom-Json|Select-Object passed,checks,error|ConvertTo-Json
