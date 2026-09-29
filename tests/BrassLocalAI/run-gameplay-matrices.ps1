param([Parameter(Mandatory=$true)][ValidateSet(2,3,4)][int]$Players)
$ErrorActionPreference='Stop'
$root=Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$build=Join-Path $root 'artifacts/patch-build'
$managed=Join-Path $build 'Managed'
$manifest=Get-Content -LiteralPath (Join-Path $build 'build-manifest.json') -Raw|ConvertFrom-Json
$runtime=Join-Path $root ('artifacts/local-ai-gameplay-matrix-'+$manifest.version+'-'+$Players)
New-Item -ItemType Directory -Path $runtime -Force|Out-Null
foreach($file in $manifest.files){if((Get-FileHash -LiteralPath (Join-Path $managed $file.name)).Hash -ne $file.sha256){throw 'Candidate fingerprint mismatch'}}
Get-ChildItem -LiteralPath $managed -Filter '*.dll'|Copy-Item -Destination $runtime -Force
Copy-Item -LiteralPath (Join-Path $build 'build-manifest.json') -Destination $runtime -Force
foreach($kind in @('runtime','identical')){
 $name='local-ai-'+$kind+'-'+$Players+'.json'
 foreach($directory in @($runtime,$build)){
  @{passed=$false;status='running';testedFiles=$manifest.files}|ConvertTo-Json -Depth 10|Set-Content -LiteralPath (Join-Path $directory $name) -Encoding UTF8
 }
}
$exe=Join-Path $runtime 'LocalRuntimeTests.exe'
$rsp=Join-Path $runtime 'matrix-tests.rsp'
@('/nologo','/target:exe','/langversion:7.3','/nostdlib+','/deterministic+',('/out:"'+$exe+'"'))+@(Get-ChildItem -LiteralPath $runtime -Filter '*.dll'|Where-Object {$_.Name -notmatch 'Tests|Regression'}|ForEach-Object {'/reference:"'+$_.FullName+'"'})+@('"'+(Join-Path $PSScriptRoot 'LocalRuntimeTests.cs')+'"')|Set-Content -LiteralPath $rsp -Encoding UTF8
& (Join-Path $root 'artifacts/patch-tools/microsoft.net.compilers.toolset.4.12.0/tasks/net472/csc.exe') ('@'+$rsp)
if($LASTEXITCODE -ne 0){throw 'Game matrix compilation failed'}
foreach($kind in @('runtime','identical')){
 $name='local-ai-'+$kind+'-'+$Players+'.json'
 $resultPath=Join-Path $runtime $name
 $output=Join-Path $build $name
 @{passed=$false;status='running';testedFiles=$manifest.files}|ConvertTo-Json -Depth 10|Set-Content -LiteralPath $output -Encoding UTF8
 if($kind -eq 'identical'){& $exe $resultPath $Players 1 'identical'}else{& $exe $resultPath $Players 1}
 $testExit=$LASTEXITCODE
 if($testExit -ne 0){throw ($kind+' matrix process failed')}
 $result=Get-Content -LiteralPath $resultPath -Raw|ConvertFrom-Json
 if(-not $result.passed){throw ($kind+' matrix failed or produced no fresh success')}
 foreach($file in $manifest.files){
  if((Get-FileHash -LiteralPath (Join-Path $managed $file.name)).Hash -ne $file.sha256 -or (Get-FileHash -LiteralPath (Join-Path $runtime $file.name)).Hash -ne $file.sha256){throw 'Candidate or matrix runtime changed'}
 }
 $result|Add-Member -NotePropertyName testedFiles -NotePropertyValue $manifest.files -Force
 $result|Add-Member -NotePropertyName runtime -NotePropertyValue $runtime -Force
 $result|Add-Member -NotePropertyName workLimit -NotePropertyValue 1 -Force
 $result|Add-Member -NotePropertyName saveTestsRun -NotePropertyValue $false -Force
 $result|ConvertTo-Json -Depth 20|Set-Content -LiteralPath $output -Encoding UTF8
}
