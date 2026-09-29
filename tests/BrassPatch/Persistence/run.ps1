param([switch]$Stock,[string]$ManagedPath)
$ErrorActionPreference='Stop'
$root=Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
if(-not $ManagedPath){$ManagedPath=Join-Path $root 'artifacts/patch-build/Managed'}
$managed=(Resolve-Path -LiteralPath $ManagedPath).Path
$assembly=Join-Path $managed 'Assembly-CSharp.dll'
if($Stock){$assembly=Join-Path $root 'artifacts/patch-baseline/Assembly-CSharp.dll'}
$compiler=Join-Path $root 'artifacts/patch-tools/microsoft.net.compilers.toolset.4.12.0/tasks/net472/csc.exe'
$dll=Join-Path $PSScriptRoot $(if($Stock){'StockProof.Tests.dll'}else{'Persistence.Tests.dll'})
$references=@(Get-ChildItem -LiteralPath $managed -Filter '*.dll'|Where-Object Name -notin @('Assembly-CSharp.dll','BrassPatch.dll')|ForEach-Object{'/reference:"'+$_.FullName+'"'})
$references+='/reference:"'+$assembly+'"'
$response=@('/nologo','/target:library','/langversion:7.3','/nostdlib+','/deterministic+',('/out:"'+$dll+'"'))+$references+('"'+(Join-Path $PSScriptRoot 'PersistenceTests.cs')+'"')
$rsp=Join-Path $PSScriptRoot 'compile.rsp';$response|Set-Content -LiteralPath $rsp -Encoding UTF8
& $compiler ('@'+$rsp);if($LASTEXITCODE -ne 0){throw 'Test compilation failed.'}
# Stock proof explicitly uses the verified unmodified DLL in a fresh process.
if($Stock -and (Get-FileHash -LiteralPath $assembly).Hash -ne '600EBF309670B2726001C16A3375F8BBF3DE76A23C0202BE7CF198BA338981FB'){throw 'Stock fingerprint mismatch.'}
[void][Reflection.Assembly]::LoadFrom((Join-Path $managed 'Newtonsoft.Json.dll'))
[void][Reflection.Assembly]::LoadFrom($assembly)
if(-not $Stock){[void][Reflection.Assembly]::LoadFrom((Join-Path $managed 'BrassPatch.dll'))}
[void][Reflection.Assembly]::LoadFrom($dll)
$fixtures=Join-Path $PSScriptRoot 'isolated-saves'
if($Stock){$result=[PersistenceTests]::Stock($fixtures);$file='stock-proof.json'}else{$result=[PersistenceTests]::Patched($fixtures);$file='results.json'}
$report=[ordered]@{utc=[DateTime]::UtcNow.ToString('o');assemblySha256=(Get-FileHash -LiteralPath $assembly).Hash;result=$result}
$report|ConvertTo-Json -Depth 8|Set-Content -LiteralPath (Join-Path $PSScriptRoot $file) -Encoding UTF8
$report|ConvertTo-Json -Depth 8
