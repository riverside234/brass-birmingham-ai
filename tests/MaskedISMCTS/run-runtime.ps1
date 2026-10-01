$ErrorActionPreference='Stop'
$root=Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$build=Join-Path $root 'artifacts/patch-build'
$managed=Join-Path $build 'Managed'
$manifest=Get-Content -LiteralPath (Join-Path $build 'build-manifest.json') -Raw|ConvertFrom-Json
$output=Join-Path $build 'masked-runtime.json'
$exe=Join-Path $managed 'MaskedRuntimeTests.exe'
$rsp=Join-Path $build 'masked-runtime.rsp'
@('/nologo','/target:exe','/langversion:7.3','/nostdlib+','/deterministic+',('/out:"'+$exe+'"'))+@(Get-ChildItem -LiteralPath $managed -Filter '*.dll'|Where-Object {$_.Name -notmatch 'Tests|Regression'}|ForEach-Object {'/reference:"'+$_.FullName+'"'})+@('"'+(Join-Path $PSScriptRoot 'RuntimeChecks.cs')+'"')|Set-Content -LiteralPath $rsp -Encoding UTF8
& (Join-Path $root 'artifacts/patch-tools/microsoft.net.compilers.toolset.4.12.0/tasks/net472/csc.exe') ('@'+$rsp)
if($LASTEXITCODE -ne 0){throw 'Masked runtime checks compilation failed'}
& $exe $output
if($LASTEXITCODE -ne 0){throw 'Masked runtime checks failed'}
$result=Get-Content -LiteralPath $output -Raw|ConvertFrom-Json
foreach($file in $manifest.files){if((Get-FileHash -LiteralPath (Join-Path $managed $file.name)).Hash -ne $file.sha256){throw 'Candidate changed during checks'}}
$result|Add-Member -NotePropertyName testedFiles -NotePropertyValue $manifest.files
$result|ConvertTo-Json -Depth 12|Set-Content -LiteralPath $output -Encoding UTF8
Write-Output ('Masked runtime checks passed: '+$result.checks)
