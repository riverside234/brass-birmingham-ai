$ErrorActionPreference='Stop'
$repo=Split-Path -Parent $PSScriptRoot
$overlay=Join-Path $repo 'artifacts/legacy-ai/overlay'
$isolated=Join-Path $repo 'artifacts/legacy-ai/unity'
if(Get-Process BrassBirminghamUnity -ErrorAction SilentlyContinue){throw 'Wait for the other game/Unity test process before running overlay UI checks.'}
if(-not(Test-Path -LiteralPath (Join-Path $isolated 'BrassBirminghamUnity.exe'))){throw 'Run check-legacy-bridge.ps1 -Unity -Overlay first to prepare the isolated copy.'}
$compiler=Join-Path $repo 'artifacts/patch-tools/microsoft.net.compilers.toolset.4.12.0/tasks/net472/csc.exe'
$copyManaged=Join-Path $isolated 'BrassBirminghamUnity_Data/Managed'
. (Join-Path $PSScriptRoot 'sync-research-managed.ps1')
Sync-BrassResearchManaged -ProjectRoot $repo -Source $overlay -Destination $copyManaged
$helper=Join-Path $copyManaged 'OverlayChecks.dll'
$rsp=Join-Path $isolated 'overlay-checks.rsp'
@('/nologo','/target:library','/langversion:7.3','/nostdlib+','/deterministic+',('/out:"'+$helper+'"'))+@(Get-ChildItem -LiteralPath $overlay -Filter '*.dll'|ForEach-Object{'/reference:"'+$_.FullName+'"'})+@('"'+(Join-Path $repo 'tests/legacy-ai/OverlayUnityEntry.cs')+'"')|Set-Content -LiteralPath $rsp -Encoding UTF8
& $compiler ('@'+$rsp)
if($LASTEXITCODE -ne 0){throw 'Overlay UI checks compilation failed'}
Copy-Item -LiteralPath (Join-Path $overlay 'BrassLegacyAI.dll') -Destination $copyManaged -Force
& (Join-Path $isolated 'UnityWeaver.exe') (Join-Path $overlay 'Assembly-CSharp.dll') $helper (Join-Path $copyManaged 'Assembly-CSharp.dll')
if($LASTEXITCODE -ne 0){throw 'Overlay UI startup weaving failed'}
$result=Join-Path $isolated 'overlay-ui-result.json'
if(Test-Path -LiteralPath $result){Remove-Item -LiteralPath $result}
$process=Start-Process -FilePath (Join-Path $isolated 'BrassBirminghamUnity.exe') -ArgumentList @('-batchmode','-nographics','-logFile',('"'+(Join-Path $isolated 'overlay-ui.log')+'"')) -WorkingDirectory $isolated -WindowStyle Hidden -PassThru
Write-Output "Overlay UI checks started (PID $($process.Id)); result: $result"
