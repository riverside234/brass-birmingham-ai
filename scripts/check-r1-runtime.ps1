param([switch]$Unity,[string]$GameRoot=(Split-Path -Parent $PSScriptRoot))
$ErrorActionPreference='Stop'
$repo=Split-Path -Parent $PSScriptRoot
$managed=Join-Path $repo 'artifacts/r1-host/guarded'
$compiler=Join-Path $repo 'artifacts/patch-tools/microsoft.net.compilers.toolset.4.12.0/tasks/net472/csc.exe'
$sources=@('tests/R1/RuntimeChecks.cs','tests/R1/UnityEntry.cs','tests/BrassPatch/CoreRegression.cs','tests/BrassPatch/State/StateRegression.cs','tests/BrassPatch/Resources/ResourcesRegression.cs','rules-audit/extended-network-matrix.cs','tests/BrassPatch/Liquidation/LiquidationTests.cs','tests/BrassPatch/Scout/ScoutTests.cs')
$response=@('/nologo','/target:exe','/main:RuntimeChecks','/langversion:7.3','/nostdlib+','/deterministic+',('/out:"'+(Join-Path $managed 'R1.RuntimeChecks.exe')+'"'),('/reference:"'+(Join-Path $managed 'BrassHost.exe')+'"'))
$response+=@(Get-ChildItem -LiteralPath $managed -Filter '*.dll'|ForEach-Object{'/reference:"'+$_.FullName+'"'})
$response+=@($sources|ForEach-Object{'"'+(Join-Path $repo $_)+'"'})
$rsp=Join-Path $repo 'artifacts/r1-host/runtime-checks.rsp';$response|Set-Content -LiteralPath $rsp -Encoding UTF8
& $compiler ('@'+$rsp)
if($LASTEXITCODE -ne 0){throw 'Runtime checks compilation failed'}
$traces=Join-Path $repo 'artifacts/r1-host/traces.json'
if(-not $Unity){
    & (Join-Path $managed 'R1.RuntimeChecks.exe') $traces (Join-Path $repo 'artifacts/r1-host/dotnet-result.json')
    if($LASTEXITCODE -ne 0){throw 'Runtime checks failed'}
    return
}
$isolated=Join-Path $repo 'artifacts/r1-unity'
New-Item -ItemType Directory -Force -Path $isolated|Out-Null
foreach($name in @('BrassBirminghamUnity.exe','UnityPlayer.dll','MonoBleedingEdge','BrassBirminghamUnity_Data')){
    $destination=Join-Path $isolated $name
    if(-not(Test-Path -LiteralPath $destination)){Copy-Item -LiteralPath (Join-Path $GameRoot $name) -Destination $destination -Recurse}
}
$copyManaged=Join-Path $isolated 'BrassBirminghamUnity_Data/Managed'
. (Join-Path $PSScriptRoot 'sync-research-managed.ps1')
Sync-BrassResearchManaged -ProjectRoot $repo -Source $managed -Destination $copyManaged
Copy-Item -LiteralPath (Join-Path $managed 'BrassHost.exe'),(Join-Path $managed 'R1.RuntimeChecks.exe') -Destination $copyManaged -Force
$cecil=Join-Path $repo 'artifacts/patch-tools/mono.cecil.0.11.6/lib/net40/Mono.Cecil.dll'
$weaver=Join-Path $isolated 'UnityWeaver.exe'
& $compiler /nologo /target:exe /deterministic+ ('/out:'+$weaver) ('/reference:'+$cecil) (Join-Path $repo 'tests/R1/UnityWeaver.cs')
if($LASTEXITCODE -ne 0){throw 'Unity startup weaver compilation failed'}
Copy-Item -LiteralPath $cecil -Destination $isolated -Force
& $weaver (Join-Path $managed 'Assembly-CSharp.dll') (Join-Path $managed 'R1.RuntimeChecks.exe') (Join-Path $copyManaged 'Assembly-CSharp.dll')
if($LASTEXITCODE -ne 0){throw 'Unity startup hook failed'}
Copy-Item -LiteralPath $traces -Destination (Join-Path $isolated 'traces.json') -Force
$output=Join-Path $isolated 'unity-result.json'
if(Test-Path -LiteralPath $output){Remove-Item -LiteralPath $output}
$process=Start-Process -FilePath (Join-Path $isolated 'BrassBirminghamUnity.exe') -ArgumentList @('-batchmode','-nographics','-logFile',('"'+(Join-Path $isolated 'unity.log')+'"')) -WorkingDirectory $isolated -WindowStyle Hidden -PassThru
Write-Output "Isolated Unity validation started (PID $($process.Id)). Result: $output"
$process.Id|Set-Content -LiteralPath (Join-Path $isolated 'process-id.txt')
