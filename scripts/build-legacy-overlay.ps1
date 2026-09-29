param([string]$GameRoot=(Split-Path -Parent $PSScriptRoot))
$ErrorActionPreference='Stop'
$repo=Split-Path -Parent $PSScriptRoot
$managed=Join-Path $GameRoot 'BrassBirminghamUnity_Data/Managed'
$deps=Get-Content -LiteralPath (Join-Path $repo 'configs/rules/runtime-dependencies-1.1.0.json') -Raw|ConvertFrom-Json
foreach($f in $deps.files){if((Get-FileHash -LiteralPath (Join-Path $managed $f.name)).Hash -ne $f.sha256){throw "Rules dependency mismatch: $($f.name)"}}
$out=Join-Path $repo 'artifacts/legacy-ai/overlay'
New-Item -ItemType Directory -Force -Path $out|Out-Null
$compiler=Join-Path $repo 'artifacts/patch-tools/microsoft.net.compilers.toolset.4.12.0/tasks/net472/csc.exe'
$references=@(Get-ChildItem -LiteralPath $managed -Filter '*.dll'|ForEach-Object{'/reference:"'+$_.FullName+'"'})
$source=@(Get-ChildItem -LiteralPath (Join-Path $repo 'src/BrassHost') -Filter '*.cs'|Where-Object Name -ne 'Program.cs')+@(Get-ChildItem -LiteralPath (Join-Path $repo 'src/BrassLegacyAI') -Filter '*.cs' -Recurse|Where-Object Name -notin @('Program.cs','Harness.cs'))
$rsp=Join-Path $out 'bridge.rsp'
@('/nologo','/target:library','/langversion:7.3','/nostdlib+','/deterministic+','/optimize+',('/out:"'+(Join-Path $out 'BrassLegacyAI.dll')+'"'))+$references+@($source|ForEach-Object{'"'+$_.FullName+'"'})|Set-Content -LiteralPath $rsp -Encoding UTF8
& $compiler ('@'+$rsp)
if($LASTEXITCODE -ne 0){throw 'Bridge compilation failed'}
$cecil=Get-Item -LiteralPath (Join-Path $repo 'artifacts/patch-tools/mono.cecil.0.11.6/lib/net40/Mono.Cecil.dll')
Copy-Item -LiteralPath $cecil.FullName -Destination (Join-Path $out 'Mono.Cecil.dll') -Force
$rsp=Join-Path $out 'weaver.rsp'
@('/nologo','/target:exe','/langversion:7.3','/deterministic+',('/out:"'+(Join-Path $out 'LegacyWeaver.exe')+'"'),('/reference:"'+$cecil.FullName+'"'),('"'+(Join-Path $repo 'scripts/LegacyWeaver.cs')+'"'))|Set-Content -LiteralPath $rsp -Encoding UTF8
& $compiler ('@'+$rsp)
if($LASTEXITCODE -ne 0){throw 'Overlay weaver compilation failed'}
& (Join-Path $out 'LegacyWeaver.exe') (Join-Path $managed 'Assembly-CSharp.dll') (Join-Path $out 'BrassLegacyAI.dll') (Join-Path $out 'Assembly-CSharp.dll')
if($LASTEXITCODE -ne 0){throw 'Overlay weaving failed'}
Get-ChildItem -LiteralPath $managed -Filter '*.dll'|Where-Object Name -ne 'Assembly-CSharp.dll'|Copy-Item -Destination $out -Force
[ordered]@{format='legacy-overlay/0.1.0';rules='steam-brass-patch-1.1.0';installed=$false;qualified=$false;files=@('BrassLegacyAI.dll','Assembly-CSharp.dll'|ForEach-Object{[ordered]@{name=$_;sha256=(Get-FileHash -LiteralPath (Join-Path $out $_)).Hash.ToLowerInvariant()}})}|ConvertTo-Json -Depth 6|Set-Content -LiteralPath (Join-Path $out 'manifest.json') -Encoding UTF8
Write-Output "Built optional overlay at $out. Not installed; live qualification is required."
