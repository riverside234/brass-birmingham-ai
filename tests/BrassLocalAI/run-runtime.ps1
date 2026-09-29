param([int]$Players=0,[int]$Work=48,[switch]$SkipBuild,[switch]$GameplayOnly)
$ErrorActionPreference='Stop'
$root=Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$build=Join-Path $root 'artifacts/patch-build'
$managed=Join-Path $build 'Managed'
$output=Join-Path $build ('local-ai-runtime-'+$Players+'.json')
$manifest=Get-Content -LiteralPath (Join-Path $build 'build-manifest.json') -Raw|ConvertFrom-Json
foreach($file in $manifest.files){if((Get-FileHash -LiteralPath (Join-Path $managed $file.name)).Hash -ne $file.sha256){throw 'Candidate fingerprint differs before local runtime tests'}}
@{passed=$false;status='running';testedFiles=$manifest.files}|ConvertTo-Json -Depth 10|Set-Content -LiteralPath $output -Encoding UTF8
$exe=Join-Path $managed 'LocalRuntimeTests.exe'
if(-not $SkipBuild){
 $rsp=Join-Path $build 'local-ai-tests.rsp'
 @('/nologo','/target:exe','/langversion:7.3','/nostdlib+','/deterministic+',('/out:"'+$exe+'"'))+@(Get-ChildItem -LiteralPath $managed -Filter '*.dll'|Where-Object {$_.Name -notmatch 'Tests|Regression'}|ForEach-Object{'/reference:"'+$_.FullName+'"'})+@('"'+(Join-Path $PSScriptRoot 'LocalRuntimeTests.cs')+'"')|Set-Content -LiteralPath $rsp -Encoding UTF8
 & (Join-Path $root 'artifacts/patch-tools/microsoft.net.compilers.toolset.4.12.0/tasks/net472/csc.exe') ('@'+$rsp)
 if($LASTEXITCODE -ne 0){throw 'Local runtime test compilation failed'}
}
if($Players){& $exe $output $Players $Work}elseif($GameplayOnly){& $exe $output 'gameplay-only'}else{& $exe $output}
if($LASTEXITCODE -ne 0){
 @{passed=$false;status='process-failed'}|ConvertTo-Json|Set-Content -LiteralPath $output -Encoding UTF8
 throw 'Local runtime tests failed'
}
$result=Get-Content -LiteralPath $output -Raw|ConvertFrom-Json
if(-not $result.passed){throw 'Local runtime tests produced no fresh success'}
foreach($file in $manifest.files){if((Get-FileHash -LiteralPath (Join-Path $managed $file.name)).Hash -ne $file.sha256){
 @{passed=$false;status='candidate-changed'}|ConvertTo-Json|Set-Content -LiteralPath $output -Encoding UTF8
 throw 'Candidate changed during local runtime tests'
}}
$result|Add-Member -NotePropertyName testedFiles -NotePropertyValue $manifest.files -Force
$result|Add-Member -NotePropertyName saveTestsRun -NotePropertyValue (-not $Players -and -not $GameplayOnly) -Force
$result|ConvertTo-Json -Depth 20|Set-Content -LiteralPath $output -Encoding UTF8
