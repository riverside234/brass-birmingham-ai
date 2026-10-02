param([string]$BuildDirectory='artifacts/economy-v3/package-1.4.20-v4',[switch]$Native,[string[]]$Suites=@())
$ErrorActionPreference='Stop'
$root=Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$build=Join-Path $root $BuildDirectory;$managed=Join-Path $build 'Managed'
$manifest=Get-Content -LiteralPath (Join-Path $build 'build-manifest.json') -Raw|ConvertFrom-Json
$compiler=Join-Path $root 'artifacts/patch-tools/microsoft.net.compilers.toolset.4.12.0/tasks/net472/csc.exe'
function Compile-Tests([string]$Name,[string[]]$Sources,[switch]$Library){
 $exe=Join-Path $managed ($Name+$(if($Library){'.dll'}else{'.exe'}));$rsp=Join-Path $build ($Name+'.rsp')
 @('/nologo',$(if($Library){'/target:library'}else{'/target:exe'}),'/langversion:7.3','/nostdlib+','/deterministic+',('/out:"'+$exe+'"'))+
 @(Get-ChildItem -LiteralPath $managed -Filter '*.dll'|Where-Object {$_.Name -notmatch 'Tests|Regression'}|ForEach-Object{'/reference:"'+$_.FullName+'"'})+
 @($Sources|ForEach-Object{'"'+(Join-Path $root $_)+'"'})|Set-Content -LiteralPath $rsp -Encoding UTF8
 & $compiler ('@'+$rsp);if($LASTEXITCODE -ne 0){throw ('Test compilation failed: '+$Name)};return $exe
}
function Publish-Test([string]$Name){
 $path=Join-Path $build ($Name+'.json');$report=Get-Content -LiteralPath $path -Raw|ConvertFrom-Json
 if(-not $report.passed){throw ('Test failed: '+$Name+' '+$report.error)}
 foreach($file in $manifest.files){if((Get-FileHash -LiteralPath (Join-Path $managed $file.name)).Hash -ne $file.sha256){throw 'Candidate changed'}}
 $report|Add-Member -NotePropertyName testedFiles -NotePropertyValue $manifest.files -Force
 $report|ConvertTo-Json -Depth 18|Set-Content -LiteralPath $path -Encoding UTF8
 Write-Output ($Name+': '+$report.checks+' checks passed')
}
if(-not $Native){
 foreach($suite in @(
  @('masked-runtime','MaskedRuntimeTests','tests/MaskedISMCTS/RuntimeChecks.cs'),
  @('era-runtime','EraRuntimeTests','tests/MaskedISMCTS/EraChecks.cs'),
  @('industry-runtime','IndustryRuntimeTests','tests/MaskedISMCTS/IndustryChecks.cs'),
  @('economy-runtime','EconomyRuntimeTests','tests/MaskedISMCTS/EconomyChecks.cs'),
  @('local-ai-execution','AiExecutionTests','tests/BrassLocalAI/AiExecutionTests.cs'),
  @('local-ai-conversion-0','ConversionTests','tests/BrassLocalAI/ConversionTests.cs'),
  @('local-ai-credit-0','CreditTests','tests/BrassLocalAI/CreditTests.cs'),
  @('local-ai-cash-beer-0','CashBeerTests','tests/BrassLocalAI/CashBeerTests.cs'))){
  if($Suites.Count -and $suite[0] -notin $Suites){continue}
  $exe=Compile-Tests $suite[1] @($suite[2]);$out=Join-Path $build ($suite[0]+'.json')
  & $exe $out $root;if($LASTEXITCODE -ne 0){throw ('Runtime checks failed: '+$suite[0])};Publish-Test $suite[0]
 }
 return
}
# Private native player; never touch another thread's player or patch-build.
$player=Join-Path $root 'artifacts/era-player';$short=Join-Path $root 'artifacts/er'
if(Get-Process BrassBirminghamUnity -ErrorAction SilentlyContinue){throw 'Another game/test player is running'}
if(-not(Test-Path -LiteralPath $player)){
 New-Item -ItemType Directory -Path $player -Force|Out-Null
 foreach($name in 'BrassBirminghamUnity.exe','UnityPlayer.dll','UnityCrashHandler64.exe'){Copy-Item -LiteralPath (Join-Path $root $name) -Destination $player}
 New-Item -ItemType Junction -Path (Join-Path $player 'MonoBleedingEdge') -Target (Join-Path $root 'MonoBleedingEdge')|Out-Null
 $data=Join-Path $player 'BrassBirminghamUnity_Data';New-Item -ItemType Directory -Path $data -Force|Out-Null
 foreach($entry in Get-ChildItem -LiteralPath (Join-Path $root 'BrassBirminghamUnity_Data')){
  if($entry.Name -eq 'Managed'){continue}
  if($entry.PSIsContainer){New-Item -ItemType Junction -Path (Join-Path $data $entry.Name) -Target $entry.FullName|Out-Null}
  else{Copy-Item -LiteralPath $entry.FullName -Destination $data}
 }
}
if(-not(Test-Path -LiteralPath $short)){New-Item -ItemType Junction -Path $short -Target $player|Out-Null}
if((Get-Item -LiteralPath $short).Target -ne $player){throw 'Unexpected native player junction'}
$nativeBuild=Join-Path $short 'artifacts/patch-build';$nativeManaged=Join-Path $nativeBuild 'Managed';$dataManaged=Join-Path $short 'BrassBirminghamUnity_Data/Managed'
New-Item -ItemType Directory -Path $nativeManaged,$dataManaged -Force|Out-Null
Compile-Tests 'LocalAiUnityTests' @('tests/BrassLocalAI/LocalAiUnityTests.cs','tests/BrassLocalAI/AiExecutionResourceTests.cs') -Library|Out-Null
foreach($destination in $nativeManaged,$dataManaged){Get-ChildItem -LiteralPath $managed -File|Where-Object Extension -in '.dll','.exe'|Copy-Item -Destination $destination -Force}
$nativeOutput=Join-Path $nativeBuild 'local-ai-native.json'
if(Test-Path -LiteralPath $nativeOutput){Remove-Item -LiteralPath $nativeOutput}
$process=Start-Process -FilePath (Join-Path $short 'BrassBirminghamUnity.exe') -ArgumentList @('-batchmode','-logFile',('"'+(Join-Path $build 'era-native.log')+'"'),'--brass-local-ai-test','--brass-masked-quick') -WorkingDirectory $short -WindowStyle Hidden -PassThru
$timer=[Diagnostics.Stopwatch]::StartNew()
while(-not $process.HasExited){if($timer.Elapsed.TotalMinutes -gt 10){$process.Kill();throw 'Native test timed out'};Start-Sleep -Milliseconds 500;$process.Refresh()}
Copy-Item -LiteralPath $nativeOutput -Destination (Join-Path $build 'local-ai-native.json') -Force
Publish-Test 'local-ai-native'


