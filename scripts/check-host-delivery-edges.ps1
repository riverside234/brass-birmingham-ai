param([ValidateSet('baseline','candidate')][string]$Arm='candidate',[ValidatePattern('^[a-z0-9-]+$')][string]$Label='edge')
$ErrorActionPreference='Stop'
$taskRoot=Split-Path -Parent $PSScriptRoot
$study=Join-Path $taskRoot 'artifacts/host-delivery/20261029'
$managed=Join-Path $study ($Arm+'/Managed')
$output=Join-Path $study ($Label+'-'+$Arm)
if(Test-Path -LiteralPath $output){throw 'Preserve edge evidence.'}
[void](New-Item -ItemType Directory -Path $output)
$sources=@('tests/RustBackendLocal/HostDeliveryEdgeChecks.cs','tests/RustBackendLocal/ValidationBench.cs')
foreach($source in $sources){Copy-Item -LiteralPath (Join-Path $taskRoot $source) -Destination $output}
$exe=Join-Path $output 'HostDeliveryEdgeChecks.exe'
$rsp=Join-Path $output 'edge.rsp'
@('/nologo','/target:exe','/main:HostDeliveryEdgeChecks','/nostdlib+','/langversion:7.3','/optimize+',('/out:"'+$exe+'"'))+
 @(Get-ChildItem -LiteralPath $managed -Filter '*.dll'|Where-Object Name -notmatch 'Tests|Regression'|ForEach-Object{'/reference:"'+$_.FullName+'"'})+
 @($sources|ForEach-Object{'"'+(Join-Path $output (Split-Path $_ -Leaf))+'"'})|Set-Content -LiteralPath $rsp -Encoding utf8
& (Join-Path $taskRoot 'artifacts/patch-tools/microsoft.net.compilers.toolset.4.12.0/tasks/net472/csc.exe') ('@'+$rsp)
if($LASTEXITCODE -ne 0){throw 'Edge compilation failed.'}
& (Join-Path $study ('probe-'+$Arm+'/allocation-host.exe')) (Join-Path $taskRoot 'MonoBleedingEdge/EmbedRuntime/mono-2.0-bdwgc.dll') $managed (Join-Path $taskRoot 'MonoBleedingEdge/etc') $exe (Join-Path $taskRoot 'artifacts/rust-backend/20261026-l14-study-v2/validation-fixtures.json') (Join-Path $output 'result.json') (Join-Path $taskRoot 'artifacts/rust-backend/20261026-l14-study-v2/trusted-live-fixtures/live-92782-CanalEra-7.json')
if($LASTEXITCODE -ne 0){throw 'Host edge checks failed.'}
Get-Content -LiteralPath (Join-Path $output 'result.json') -Raw|ConvertFrom-Json|Select-Object passed,checks,error|ConvertTo-Json
