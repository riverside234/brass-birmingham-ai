param([switch]$Unity,[switch]$Overlay)
$ErrorActionPreference='Stop'
$repo=Split-Path -Parent $PSScriptRoot
$runtime=Join-Path $repo 'artifacts/legacy-ai/runtime'
if($Overlay){$runtime=Join-Path $repo 'artifacts/legacy-ai/overlay'}
$aiAssembly=if($Overlay){'BrassLegacyAI.dll'}else{'BrassLegacyAI.exe'}
$compiler=Join-Path $repo 'artifacts/patch-tools/microsoft.net.compilers.toolset.4.12.0/tasks/net472/csc.exe'
$rsp=Join-Path $runtime 'bridge-checks.rsp'
@('/nologo','/target:exe','/main:BridgeChecks','/langversion:7.3','/nostdlib+','/deterministic+',('/out:"'+(Join-Path $runtime 'BridgeChecks.exe')+'"'),('/reference:"'+(Join-Path $runtime $aiAssembly)+'"'))+@(Get-ChildItem -LiteralPath $runtime -Filter '*.dll'|Where-Object Name -ne $aiAssembly|ForEach-Object{'/reference:"'+$_.FullName+'"'})+@('tests/legacy-ai/BridgeChecks.cs','tests/legacy-ai/LiveMatchChecks.cs','tests/legacy-ai/UnityEntry.cs'|ForEach-Object{'"'+(Join-Path $repo $_)+'"'})|Set-Content -LiteralPath $rsp -Encoding UTF8
& $compiler ('@'+$rsp)
if($LASTEXITCODE -ne 0){throw 'Bridge checks compilation failed'}
if(-not $Unity){& (Join-Path $runtime 'BridgeChecks.exe') $repo (Join-Path $repo 'artifacts/legacy-ai/bridge-result.json');if($LASTEXITCODE -ne 0){throw 'Bridge checks failed'};return}
$isolated=Join-Path $repo 'artifacts/legacy-ai/unity'
New-Item -ItemType Directory -Force -Path $isolated|Out-Null
foreach($name in @('BrassBirminghamUnity.exe','UnityPlayer.dll','MonoBleedingEdge','BrassBirminghamUnity_Data')){
    $destination=Join-Path $isolated $name
    if(-not(Test-Path -LiteralPath $destination)){Copy-Item -LiteralPath (Join-Path $repo $name) -Destination $destination -Recurse}
}
$copyManaged=Join-Path $isolated 'BrassBirminghamUnity_Data/Managed'
. (Join-Path $PSScriptRoot 'sync-research-managed.ps1')
Sync-BrassResearchManaged -ProjectRoot $repo -Source $runtime -Destination $copyManaged
Copy-Item -LiteralPath (Join-Path $runtime $aiAssembly),(Join-Path $runtime 'BridgeChecks.exe') -Destination $copyManaged -Force
$cecil=Join-Path $repo 'artifacts/patch-tools/mono.cecil.0.11.6/lib/net40/Mono.Cecil.dll'
$weaver=Join-Path $isolated 'UnityWeaver.exe'
& $compiler /nologo /target:exe /deterministic+ ('/out:'+$weaver) ('/reference:'+$cecil) (Join-Path $repo 'tests/R1/UnityWeaver.cs')
if($LASTEXITCODE -ne 0){throw 'Startup weaver compilation failed'}
Copy-Item -LiteralPath $cecil -Destination $isolated -Force
& $weaver (Join-Path $runtime 'Assembly-CSharp.dll') (Join-Path $runtime 'BridgeChecks.exe') (Join-Path $copyManaged 'Assembly-CSharp.dll')
if($LASTEXITCODE -ne 0){throw 'Startup weaving failed'}
New-Item -ItemType Directory -Force -Path (Join-Path $isolated 'configs/contracts'),(Join-Path $isolated 'configs/legacy-ai')|Out-Null
Copy-Item -LiteralPath (Join-Path $repo 'configs/contracts/v0.2.0') -Destination (Join-Path $isolated 'configs/contracts') -Recurse -Force
Get-ChildItem -LiteralPath (Join-Path $repo 'configs/legacy-ai') -Filter '*-v*.json'|Copy-Item -Destination (Join-Path $isolated 'configs/legacy-ai') -Force
$process=Start-Process -FilePath (Join-Path $isolated 'BrassBirminghamUnity.exe') -ArgumentList @('-batchmode','-nographics','-logFile',('"'+(Join-Path $isolated 'unity.log')+'"')) -WorkingDirectory $isolated -WindowStyle Hidden -PassThru
Write-Output "Isolated Unity bridge checks started (PID $($process.Id)); result: $isolated/legacy-unity-result.json"
