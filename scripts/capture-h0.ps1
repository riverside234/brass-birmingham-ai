param([int]$Seed=4100,[ValidateRange(1,100)][int]$Games=5)
$ErrorActionPreference='Stop'
$repo=Split-Path -Parent $PSScriptRoot
$guarded=Join-Path $repo 'artifacts/r1-host/guarded'
if(-not(Test-Path -LiteralPath (Join-Path $guarded 'runtime.json'))){throw 'Build the R1 host first.'}
$stock=Join-Path $repo 'artifacts/patch-baseline/Assembly-CSharp.dll'
if((Get-FileHash -LiteralPath $stock).Hash -ne '600EBF309670B2726001C16A3375F8BBF3DE76A23C0202BE7CF198BA338981FB'){throw 'Stock fingerprint mismatch'}
$compiler=Join-Path $repo 'artifacts/patch-tools/microsoft.net.compilers.toolset.4.12.0/tasks/net472/csc.exe'
foreach($profile in @('stock-hard-original','stock-hard-guarded-1.1.0')){
    $out=Join-Path $repo ('artifacts/h0/'+$profile)
    New-Item -ItemType Directory -Force -Path $out|Out-Null
    Get-ChildItem -LiteralPath $guarded -Filter '*.dll'|Where-Object Name -ne 'BrassPatch.dll'|Copy-Item -Destination $out -Force
    if($profile -eq 'stock-hard-original'){Copy-Item -LiteralPath $stock -Destination (Join-Path $out 'Assembly-CSharp.dll') -Force}
    else{Copy-Item -LiteralPath (Join-Path $guarded 'BrassPatch.dll') -Destination $out -Force}
    $response=@('/nologo','/target:exe','/langversion:7.3','/nostdlib+','/deterministic+',('/out:"'+(Join-Path $out 'BaselineCapture.exe')+'"'))
    $response+=@(Get-ChildItem -LiteralPath $out -Filter '*.dll'|ForEach-Object{'/reference:"'+$_.FullName+'"'})
    $response+='"'+(Join-Path $repo 'tests/R1/BaselineCapture.cs')+'"'
    $rsp=Join-Path $out 'capture.rsp';$response|Set-Content -LiteralPath $rsp -Encoding UTF8
    & $compiler ('@'+$rsp)
    if($LASTEXITCODE -ne 0){throw 'H0 compilation failed'}
    & (Join-Path $out 'BaselineCapture.exe') $profile (Join-Path $out 'capture.json') $Seed $Games
    if($LASTEXITCODE -ne 0){throw 'H0 capture process failed'}
    & (Join-Path $out 'BaselineCapture.exe') $profile (Join-Path $out 'capture-plain.json') $Seed 1 plain
    if($LASTEXITCODE -ne 0){throw 'H0 uninstrumented control failed'}
}
