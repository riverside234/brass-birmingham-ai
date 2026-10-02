param([string]$BuildDirectory='artifacts/economy-v3/package-1.4.20-v4')
$ErrorActionPreference='Stop'
$root=Split-Path -Parent (Split-Path -Parent $PSScriptRoot);$build=Join-Path $root $BuildDirectory;$managed=Join-Path $build 'Managed'
$manifest=Get-Content -LiteralPath (Join-Path $build 'build-manifest.json') -Raw|ConvertFrom-Json
$compiler=Join-Path $root 'artifacts/patch-tools/microsoft.net.compilers.toolset.4.12.0/tasks/net472/csc.exe'
function Compile([string]$Directory,[string]$Class,[string]$Source,[string]$Main=''){
 $exe=Join-Path $Directory ($Class+'.exe');$rsp=Join-Path $Directory ($Class+'.rsp')
 $flags=@('/nologo','/target:exe','/langversion:7.3','/nostdlib+','/deterministic+',('/out:"'+$exe+'"'))
 if($Main){$flags+='/main:'+$Main}
 $flags+@(Get-ChildItem -LiteralPath $Directory -Filter '*.dll'|Where-Object {$_.Name -notmatch 'Tests|Regression'}|ForEach-Object{'/reference:"'+$_.FullName+'"'})+@('"'+(Join-Path $root $Source)+'"')|Set-Content -LiteralPath $rsp -Encoding UTF8
 & $compiler ('@'+$rsp);if($LASTEXITCODE -ne 0){throw 'Save test compilation failed'};return $exe
}
$versions=@('1.3.2','1.4.0','1.4.3','1.4.9','1.4.11','1.4.15','1.4.17')
if($manifest.version -eq '1.4.20'){$versions+='1.4.18'}
foreach($version in $versions){
 $fixture=Join-Path $build ('save-fixtures/'+$version);$producer=Join-Path $fixture 'producer';$consumer=Join-Path $fixture 'consumer';$saves=Join-Path $fixture 'saves'
 New-Item -ItemType Directory -Path $producer,$consumer,$saves -Force|Out-Null
 foreach($dir in $producer,$consumer){Get-ChildItem -LiteralPath $managed -Filter '*.dll'|Where-Object Name -notmatch 'Tests|Regression'|Copy-Item -Destination $dir -Force}
 $archive=Join-Path $root ('artifacts/patch-releases/'+$version);$prior=Get-Content -LiteralPath (Join-Path $archive 'build-manifest.json') -Raw|ConvertFrom-Json
 foreach($file in $prior.files){
  $source=Join-Path $archive $file.name;if(-not(Test-Path -LiteralPath $source)){$source=Join-Path $archive ('Managed/'+$file.name)}
  if((Get-FileHash -LiteralPath $source).Hash -ne $file.sha256){throw 'Archive producer changed'}
  Copy-Item -LiteralPath $source -Destination (Join-Path $producer $file.name) -Force
 }
 $index=Join-Path $fixture 'fixtures.json';$output=Join-Path $build ('save-'+$version+'.json')
 if($version -in @('1.3.2','1.4.0','1.4.3')){
  $make=Compile $producer 'V2SaveFixture' 'tests/BrassLocalAI/V2SaveFixture.cs';$check=Compile $consumer 'SaveCompatibilityTests' 'tests/BrassLocalAI/SaveCompatibilityTests.cs'
  & $make $saves $index;if($LASTEXITCODE -ne 0){throw 'Archive game fixture failed'}
  & $check $saves $index $output
 }else{
  $make=Compile $producer 'EraSaveProducer' 'tests/MaskedISMCTS/EraSaveChecks.cs' 'EraSaveProducer'
  $check=Compile $consumer 'EraSaveConsumer' 'tests/MaskedISMCTS/EraSaveChecks.cs' 'EraSaveConsumer'
  & $make $index $saves;if($LASTEXITCODE -ne 0){throw 'Archive metadata fixture failed'}
  & $check $index $output $saves
 }
 if($LASTEXITCODE -ne 0){throw ('Archived save restore failed: '+$version)}
 $report=Get-Content -LiteralPath $output -Raw|ConvertFrom-Json
 if(-not $report.passed){throw ('Saved profile check failed: '+$report.error)}
 $report|Add-Member -NotePropertyName testedFiles -NotePropertyValue $manifest.files -Force
 $report|ConvertTo-Json -Depth 18|Set-Content -LiteralPath $output -Encoding UTF8
 Write-Output ($version+': '+$report.checks+' saved-profile checks passed')
}


