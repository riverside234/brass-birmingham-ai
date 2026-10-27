param([Parameter(Mandatory=$true)][string]$BuildDirectory,[Parameter(Mandatory=$true)][string]$StudyDirectory,[string]$Label='contract-checks')
$ErrorActionPreference='Stop'
$l14Root=Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$l14Build=Join-Path $l14Root $BuildDirectory
$l14Study=Join-Path $l14Root $StudyDirectory
if($Label -notmatch '^[a-zA-Z0-9][a-zA-Z0-9_-]*$'){throw 'Invalid evidence label.'}
$l14Output=Join-Path $l14Study $Label
if(Test-Path -LiteralPath $l14Output){throw 'Contract evidence already exists; preserve it and choose a new label.'}
[void](New-Item -ItemType Directory -Path $l14Output)
Get-ChildItem -LiteralPath (Join-Path $l14Build 'Managed') -Filter '*.dll' | Where-Object Name -notmatch 'Tests|Regression' | ForEach-Object {
 Copy-Item -LiteralPath $_.FullName -Destination $l14Output
}
$l14Sources=@('tests/RustBackendLocal/ValidationBench.cs','tests/RustBackendLocal/NativeValidationContractChecks.cs','tests/RustBackendLocal/L14ContractChecks.cs','tests/BrassPatch/Liquidation/LiquidationTests.cs')
foreach($source in $l14Sources){Copy-Item -LiteralPath (Join-Path $l14Root $source) -Destination $l14Output}
$l14Exe=Join-Path $l14Output 'L14ContractChecks.exe'
$l14Rsp=Join-Path $l14Output 'checks.rsp'
@('/nologo','/target:exe','/main:L14ContractChecks','/langversion:7.3','/nostdlib+','/deterministic+','/optimize+',('/out:"'+$l14Exe+'"'))+
 @(Get-ChildItem -LiteralPath $l14Output -Filter '*.dll' | ForEach-Object {'/reference:"'+$_.FullName+'"'})+
 @($l14Sources | ForEach-Object {'"'+(Join-Path $l14Output (Split-Path $_ -Leaf))+'"'}) | Set-Content -LiteralPath $l14Rsp -Encoding utf8
& (Join-Path $l14Root 'artifacts/patch-tools/microsoft.net.compilers.toolset.4.12.0/tasks/net472/csc.exe') ('@'+$l14Rsp)
if($LASTEXITCODE -ne 0){throw 'L14 contract test compilation failed.'}
& $l14Exe (Join-Path $l14Study 'validation-fixtures.json') $l14Output
if($LASTEXITCODE -ne 0){throw 'L14 event/identity or checkpoint contract failed.'}
Write-Output 'L14 event/identity and human End checkpoint contracts passed.'
