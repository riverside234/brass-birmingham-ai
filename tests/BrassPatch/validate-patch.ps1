$ErrorActionPreference='Stop'
$root=Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$build=Join-Path $root 'artifacts/patch-build'
$manifest=Get-Content -LiteralPath (Join-Path $build 'build-manifest.json') -Raw|ConvertFrom-Json
$checks=@(
 @('pwsh','run-core.ps1'),
 @('pwsh','State/run.ps1'),
 @('pwsh','AI/run.ps1'),
 @('powershell','Scout/Run-ScoutTests.ps1'),
 @('powershell','upgrade-compatibility.ps1'),
 @('powershell','Resources/run.ps1'),
 @('powershell','Liquidation/Run-LiquidationTests.ps1'),
 @('powershell','Persistence/run.ps1'),
 @('pwsh','run-audit-regressions.ps1','-BuildMatrix'),
 @('pwsh','run-audit-regressions.ps1','-Players','2'),
 @('pwsh','run-audit-regressions.ps1','-Players','3'),
 @('pwsh','run-audit-regressions.ps1','-Players','4')
)
$completed=@()
foreach($check in $checks){
    $command=$check[0]; $path=Join-Path $PSScriptRoot $check[1]
    $arguments=@('-NoProfile','-ExecutionPolicy','Bypass','-File',$path)
    if($check.Count -gt 2){$arguments+=$check[2..($check.Count-1)]}
    & $command @arguments
    if($LASTEXITCODE -ne 0){throw ('Validation failed: '+($check -join ' '))}
    $completed+=($check -join ' ')
}
# Compile the additional test libraries loaded only by the explicit Unity self-test mode.
$compiler=Join-Path $root 'artifacts/patch-tools/microsoft.net.compilers.toolset.4.12.0/tasks/net472/csc.exe'
$managed=Join-Path $build 'Managed'
foreach($pair in @(@('Liquidation/LiquidationTests.cs','LiquidationTests.dll'),@('Liquidation/UnityPresentationRegression.cs','UnityPresentationRegression.dll'),@('Scout/ScoutTests.cs','ScoutTests.dll'),@('Scout/UnityScoutRegression.cs','UnityScoutRegression.dll'),@('CityLocator/UnityCityLocatorRegression.cs','UnityCityLocatorRegression.dll'))){
    $rsp=Join-Path $build ($pair[1]+'.rsp')
    $response=@('/nologo','/target:library','/langversion:7.3','/nostdlib+','/deterministic+',('/out:"'+(Join-Path $managed $pair[1])+'"'))
    $response+=@(Get-ChildItem -LiteralPath $managed -Filter '*.dll'|Where-Object Name -notlike '*Tests*'|Where-Object Name -notin @('UnityPresentationRegression.dll','UnityScoutRegression.dll','UnityCityLocatorRegression.dll')|ForEach-Object{'/reference:"'+$_.FullName+'"'})
    $response+='"'+(Join-Path $PSScriptRoot $pair[0])+'"'
    $response|Set-Content -LiteralPath $rsp -Encoding UTF8
    & $compiler ('@'+$rsp)
    if($LASTEXITCODE -ne 0){throw 'Unity test compilation failed'}
}
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'Resources/ResourcesRegression.Tests.dll') -Destination $managed -Force
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'Persistence/Persistence.Tests.dll') -Destination $managed -Force
foreach($file in $manifest.files){if((Get-FileHash -LiteralPath (Join-Path $managed $file.name)).Hash -ne $file.sha256){throw 'Candidate changed during validation'}}
[ordered]@{passed=$true;validatedUtc=[DateTime]::UtcNow.ToString('o');files=$manifest.files;suites=$completed;unityPlayerValidation='pending after installation'}|ConvertTo-Json -Depth 6|Set-Content -LiteralPath (Join-Path $build 'validation.json') -Encoding UTF8
Write-Output 'All managed regression gates passed for the fingerprinted candidate.'
