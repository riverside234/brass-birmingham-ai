param([switch]$Research)
$ErrorActionPreference='Stop'
$root=Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$build=Join-Path $root 'artifacts/patch-build'
$manifest=Get-Content -LiteralPath (Join-Path $build 'build-manifest.json') -Raw|ConvertFrom-Json
$source=if($Research){Join-Path $root 'artifacts/legacy-ai/runtime'}else{Join-Path $build 'Managed'}
$runtime=Join-Path $root ('artifacts/credit-tests-'+[int][bool]$Research)
New-Item -ItemType Directory -Force -Path $runtime|Out-Null
$output=Join-Path $build ('local-ai-credit-'+[int][bool]$Research+'.json')
@{passed=$false;status='running'}|ConvertTo-Json|Set-Content -LiteralPath $output
Get-ChildItem -LiteralPath $source -Filter '*.dll'|Copy-Item -Destination $runtime -Force
$references=@(Get-ChildItem -LiteralPath $runtime -Filter '*.dll'|Where-Object {$_.Name -notmatch 'Tests|Regression'}|ForEach-Object {'/reference:"'+$_.FullName+'"'})
if($Research){Copy-Item -LiteralPath (Join-Path $source 'BrassLegacyAI.exe') -Destination $runtime -Force;$references+='/reference:"'+(Join-Path $runtime 'BrassLegacyAI.exe')+'"'}
$exe=Join-Path $runtime 'CreditTests.exe';$rsp=Join-Path $runtime 'tests.rsp'
@('/nologo','/target:exe','/langversion:7.3','/nostdlib+','/deterministic+',('/out:"'+$exe+'"'))+$references+@('"'+(Join-Path $PSScriptRoot 'CreditTests.cs')+'"')|Set-Content -LiteralPath $rsp -Encoding UTF8
& (Join-Path $root 'artifacts/patch-tools/microsoft.net.compilers.toolset.4.12.0/tasks/net472/csc.exe') ('@'+$rsp)
if($LASTEXITCODE -ne 0){throw 'Conversion compilation failed'}
$raw=Join-Path $runtime 'result.json'
@{passed=$false;status='running'}|ConvertTo-Json|Set-Content -LiteralPath $raw
& $exe $raw $root
if($LASTEXITCODE -ne 0){throw 'Conversion tests failed'}
$result=Get-Content -LiteralPath $raw -Raw|ConvertFrom-Json
if(-not $result.passed){throw 'Conversion checks incomplete'}
if($Research){$files=(Get-Content -LiteralPath (Join-Path $source 'runtime.json') -Raw|ConvertFrom-Json).files}else{$files=$manifest.files}
foreach($file in $files){foreach($dir in $source,$runtime){if((Get-FileHash -LiteralPath (Join-Path $dir $file.name)).Hash -ne $file.sha256){throw 'Conversion fingerprint mismatch'}}}
$result|Add-Member -NotePropertyName testedFiles -NotePropertyValue $files
$result|ConvertTo-Json -Depth 20|Set-Content -LiteralPath $output -Encoding UTF8
