param([ValidateSet('regression','matches','live-budget')][string]$Mode='regression')
$ErrorActionPreference='Stop'
$root=Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$build=Join-Path $root 'artifacts/patch-build'
$manifest=Get-Content -LiteralPath (Join-Path $build 'build-manifest.json') -Raw|ConvertFrom-Json
$managed=Join-Path $build 'Managed'
$runtime=Join-Path $root ('artifacts/local-ai-cash-'+$Mode)
$name='local-ai-cash-'+$Mode
$output=Join-Path $build ($name+'.json')
@{passed=$false;status='running';testedFiles=$manifest.files}|ConvertTo-Json -Depth 10|Set-Content -LiteralPath $output
New-Item -ItemType Directory -Path $runtime -Force|Out-Null
foreach($file in $manifest.files){if((Get-FileHash -LiteralPath (Join-Path $managed $file.name)).Hash -ne $file.sha256){throw 'Candidate mismatch'}}
Get-ChildItem -LiteralPath $managed -Filter '*.dll'|Copy-Item -Destination $runtime -Force
$class=if($Mode -eq 'regression'){'CashManagementTests'}else{'CashMatchTests'}
$exe=Join-Path $runtime ($class+'.exe');$rsp=Join-Path $runtime 'tests.rsp'
@('/nologo','/target:exe','/langversion:7.3','/nostdlib+','/deterministic+',('/out:"'+$exe+'"'))+@(Get-ChildItem -LiteralPath $runtime -Filter '*.dll'|Where-Object {$_.Name -notmatch 'Tests|Regression'}|ForEach-Object {'/reference:"'+$_.FullName+'"'})+@('"'+(Join-Path $PSScriptRoot ($class+'.cs'))+'"')|Set-Content -LiteralPath $rsp -Encoding UTF8
& (Join-Path $root 'artifacts/patch-tools/microsoft.net.compilers.toolset.4.12.0/tasks/net472/csc.exe') ('@'+$rsp)
if($LASTEXITCODE -ne 0){throw 'Cash test compilation failed'}
$raw=Join-Path $runtime 'result.json'
@{passed=$false;status='running'}|ConvertTo-Json|Set-Content -LiteralPath $raw
& $exe $raw $Mode
if($LASTEXITCODE -ne 0){throw 'Cash test process failed'}
$result=Get-Content -LiteralPath $raw -Raw|ConvertFrom-Json
if(-not $result.passed){throw 'No fresh passing cash result'}
foreach($file in $manifest.files){foreach($dir in $managed,$runtime){if((Get-FileHash -LiteralPath (Join-Path $dir $file.name)).Hash -ne $file.sha256){throw 'Candidate changed during cash tests'}}}
$result|Add-Member -NotePropertyName testedFiles -NotePropertyValue $manifest.files -Force
$result|ConvertTo-Json -Depth 20|Set-Content -LiteralPath $output -Encoding UTF8
