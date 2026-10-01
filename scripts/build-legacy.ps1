param([string]$GameRoot=(Split-Path -Parent $PSScriptRoot),[string]$ManagedPath)
$ErrorActionPreference='Stop'
$repo=Split-Path -Parent $PSScriptRoot
$managed=if($ManagedPath){$ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($ManagedPath)}else{Join-Path $GameRoot 'BrassBirminghamUnity_Data/Managed'}
$deps=Get-Content -LiteralPath (Join-Path $repo 'configs/rules/runtime-dependencies-1.1.0.json') -Raw|ConvertFrom-Json
foreach($file in $deps.files){if((Get-FileHash -LiteralPath (Join-Path $managed $file.name)).Hash -ne $file.sha256){throw "Dependency changed: $($file.name)"}}
$compiler=Join-Path $repo 'artifacts/patch-tools/microsoft.net.compilers.toolset.4.12.0/tasks/net472/csc.exe'
if(-not(Test-Path -LiteralPath $compiler)){throw 'Run build-host.ps1 first to provision the pinned compiler.'}
$out=Join-Path $repo 'artifacts/legacy-ai/runtime'
New-Item -ItemType Directory -Force -Path $out|Out-Null
Get-ChildItem -LiteralPath $managed -Filter '*.dll'|Copy-Item -Destination $out -Force
$references=@(Get-ChildItem -LiteralPath $managed -Filter '*.dll'|ForEach-Object{'/reference:"'+$_.FullName+'"'})
$files=@(Get-ChildItem -LiteralPath (Join-Path $repo 'src/BrassHost') -Filter '*.cs'|Where-Object Name -ne 'Program.cs')+@(Get-ChildItem -LiteralPath (Join-Path $repo 'src/BrassLegacyAI') -Filter '*.cs')
$files+=@(Get-ChildItem -LiteralPath (Join-Path $repo 'src/BrassMaskedSearch') -Filter '*.cs')
$sources=@($files|Sort-Object FullName|ForEach-Object{'"'+$_.FullName+'"'})
$rsp=Join-Path $out 'build.rsp'
@('/nologo','/target:exe','/main:BrassLegacyAI.Program','/langversion:7.3','/nostdlib+','/deterministic+','/optimize+',('/out:"'+(Join-Path $out 'BrassLegacyAI.exe')+'"'))+$references+$sources|Set-Content -LiteralPath $rsp -Encoding UTF8
& $compiler ('@'+$rsp)
if($LASTEXITCODE -ne 0){throw 'Legacy compilation failed'}
$pins=@($files|Sort-Object FullName|ForEach-Object{[ordered]@{path=$_.FullName.Substring($repo.Length+1).Replace('\','/');sha256=(Get-FileHash -LiteralPath $_.FullName).Hash.ToLowerInvariant()}})
$binaries=@(Get-ChildItem -LiteralPath $out -Filter '*.dll')+@(Get-Item -LiteralPath (Join-Path $out 'BrassLegacyAI.exe'))
[ordered]@{format='legacy-runtime/0.1.0';player_count=4;rules='steam-brass-patch-1.1.0';protocol='0.2.0';sources=$pins;files=@($binaries|Sort-Object Name|ForEach-Object{[ordered]@{name=$_.Name;sha256=(Get-FileHash -LiteralPath $_.FullName).Hash.ToLowerInvariant()}})}|ConvertTo-Json -Depth 8|Set-Content -LiteralPath (Join-Path $out 'runtime.json') -Encoding UTF8
Write-Output "Built $out (installed rules and R1 executable unchanged)."
