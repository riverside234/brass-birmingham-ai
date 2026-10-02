param([string]$RuntimeDirectory='artifacts/legacy-ai/runtime',[string]$Output='artifacts/v4-network/focused-1.json')
$ErrorActionPreference='Stop'
$root=Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$runtime=Join-Path $root $RuntimeDirectory;$out=Join-Path $root $Output
if(Test-Path -LiteralPath $out){throw 'Preserve earlier network evidence; select a new output'}
New-Item -ItemType Directory -Path (Split-Path -Parent $out) -Force|Out-Null
$exe=Join-Path $runtime 'NetworkOptionsChecks.exe';$rsp=Join-Path (Split-Path -Parent $out) 'network-options.rsp'
$worker=Join-Path $runtime 'BrassLegacyAI.exe';$source=Join-Path $root 'tests/MaskedISMCTS/NetworkOptionsChecks.cs'
@('/nologo','/target:exe','/langversion:7.3','/nostdlib+','/deterministic+',('/out:"'+$exe+'"'),('/reference:"'+$worker+'"'))+
@(Get-ChildItem -LiteralPath $runtime -Filter '*.dll'|Where-Object {$_.Name -notmatch 'BrassLegacyAI|Tests|Regression'}|ForEach-Object{'/reference:"'+$_.FullName+'"'})+
@('"'+$source+'"')|Set-Content -LiteralPath $rsp -Encoding UTF8
& (Join-Path $root 'artifacts/patch-tools/microsoft.net.compilers.toolset.4.12.0/tasks/net472/csc.exe') ('@'+$rsp)
if($LASTEXITCODE -ne 0){throw 'Network checks compilation failed'}
& $exe $out $root
if($LASTEXITCODE -ne 0){throw 'Network checks failed'}
$report=Get-Content -LiteralPath $out -Raw|ConvertFrom-Json
$report|Add-Member -NotePropertyName runtimeSha256 -NotePropertyValue (Get-FileHash -LiteralPath $worker).Hash.ToLowerInvariant()
$report|Add-Member -NotePropertyName fixtureSha256 -NotePropertyValue (Get-FileHash -LiteralPath $source).Hash.ToLowerInvariant()
$report|ConvertTo-Json -Depth 12|Set-Content -LiteralPath $out -Encoding UTF8
