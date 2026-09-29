$ErrorActionPreference='Stop'
$root=Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$runtime=Join-Path $root 'artifacts/local-ai-live-runtime'
$build=Join-Path $root 'artifacts/patch-build'
New-Item -ItemType Directory -Path $runtime -Force|Out-Null
Get-ChildItem -LiteralPath (Join-Path $build 'Managed') -Filter '*.dll'|Copy-Item -Destination $runtime -Force
Copy-Item -LiteralPath (Join-Path $build 'build-manifest.json') -Destination $runtime -Force
$rsp=Join-Path $runtime 'live-tests.rsp'
@('/nologo','/target:exe','/langversion:7.3','/nostdlib+','/deterministic+',('/out:"'+(Join-Path $runtime 'LocalLiveTests.exe')+'"'))+@(Get-ChildItem -LiteralPath $runtime -Filter '*.dll'|Where-Object {$_.Name -notmatch 'Tests|Regression'}|ForEach-Object {'/reference:"'+$_.FullName+'"'})+@('"'+(Join-Path $PSScriptRoot 'LocalLiveTests.cs')+'"')|Set-Content -LiteralPath $rsp -Encoding UTF8
& (Join-Path $root 'artifacts/patch-tools/microsoft.net.compilers.toolset.4.12.0/tasks/net472/csc.exe') ('@'+$rsp)
if($LASTEXITCODE -ne 0){throw 'Live test compilation failed'}
$output=Join-Path $build 'local-ai-live.json'
& (Join-Path $runtime 'LocalLiveTests.exe') $output
if($LASTEXITCODE -ne 0){throw 'Live mixed-game tests failed'}
$result=Get-Content -LiteralPath $output -Raw|ConvertFrom-Json
$manifest=Get-Content -LiteralPath (Join-Path $runtime 'build-manifest.json') -Raw|ConvertFrom-Json
foreach($file in $manifest.files){if((Get-FileHash -LiteralPath (Join-Path $runtime $file.name)).Hash -ne $file.sha256){throw 'Live test runtime changed'}}
$result|Add-Member -NotePropertyName testedFiles -NotePropertyValue $manifest.files -Force
$result|ConvertTo-Json -Depth 12|Set-Content -LiteralPath $output -Encoding UTF8
