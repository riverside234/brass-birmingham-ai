$ErrorActionPreference='Stop'
$gameRoot=Split-Path -Parent $PSScriptRoot
$assemblyPath=Join-Path $gameRoot 'BrassBirminghamUnity_Data\Managed\Assembly-CSharp.dll'
$hash=(Get-FileHash -LiteralPath $assemblyPath -Algorithm SHA256).Hash
if($hash -ne '600EBF309670B2726001C16A3375F8BBF3DE76A23C0202BE7CF198BA338981FB'){throw 'Recheck changed assembly.'}
[void][Reflection.Assembly]::LoadFrom($assemblyPath)
Add-Type -Path (Join-Path $PSScriptRoot 'extended-network-matrix.cs') -ReferencedAssemblies @($assemblyPath,'System.Core','System')
$watch=[Diagnostics.Stopwatch]::StartNew()
$result=[ExtendedNetworkMatrix]::Run()
$report=[ordered]@{assemblySha256=$hash;utc=[DateTime]::UtcNow.ToString('o');elapsedSeconds=$watch.Elapsed.TotalSeconds;method='Original DLL queries/actions/usecases, isolated fixtures. Independently written graph/cost oracle. Stub presenter queues replay completion; cancel/undo cases use production BrassGameHistory, BaseRepository, and in-memory storage. No production ReplayUseCase or Unity animations in this suite.';result=$result}
$json=$report|ConvertTo-Json -Depth 20
$json|Set-Content -LiteralPath (Join-Path $PSScriptRoot 'extended-network-matrix-results.json') -Encoding UTF8
$json
