$ErrorActionPreference='Stop'
$root=Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
# Keep copied .NET Framework executable/output paths below Windows MAX_PATH.
$auditRoot=Join-Path $root ('artifacts/re-'+[Guid]::NewGuid().ToString('N').Substring(0,8))
New-Item -ItemType Directory -Path $auditRoot -Force|Out-Null
$compiler=Join-Path $root 'artifacts/patch-tools/microsoft.net.compilers.toolset.4.12.0/tasks/net472/csc.exe'
$mockChild=Join-Path $auditRoot 'MockChild.exe'
$mockCompiler=Join-Path $auditRoot 'MockCompiler.exe'
$childSource=Join-Path $auditRoot 'MockChild.cs'
$compilerSource=Join-Path $auditRoot 'MockCompiler.cs'
@'
using System;
using System.IO;
class MockChild {
 static int Main(string[] args) {
  string mode=Environment.GetEnvironmentVariable("RUNNER_EVIDENCE_MODE");
  if(mode=="no_output")return 0;
  File.WriteAllText(args[0],"{\"passed\":true,\"checks\":1}");
  if(mode=="hash_change")File.AppendAllText(Path.Combine(AppDomain.CurrentDomain.BaseDirectory,"BrassLegacyAI.dll"),"changed");
  return mode=="nonzero_write"?17:0;
 }
}
'@|Set-Content -LiteralPath $childSource -Encoding UTF8
@'
using System;
using System.IO;
using System.Linq;
class MockCompiler {
 static int Main(string[] args) {
  if(Environment.GetEnvironmentVariable("RUNNER_EVIDENCE_MODE")=="compiler_failure")return 19;
  string output=File.ReadAllLines(args[0].Substring(1)).Single(l=>l.StartsWith("/out:" )).Substring(5).Trim('"');
  File.Copy(Environment.GetEnvironmentVariable("RUNNER_EVIDENCE_CHILD"),output,true);
  return 0;
 }
}
'@|Set-Content -LiteralPath $compilerSource -Encoding UTF8
& $compiler '/nologo' '/target:exe' ('/out:'+$mockChild) $childSource
if($LASTEXITCODE -ne 0){throw 'Mock child compilation failed'}
& $compiler '/nologo' '/target:exe' ('/out:'+$mockCompiler) $compilerSource
if($LASTEXITCODE -ne 0){throw 'Mock compiler compilation failed'}
$env:RUNNER_EVIDENCE_CHILD=$mockChild
$cases=@(
 @{script='run-gameplay-matrices.ps1';args=@('-Players','2');reports=@('local-ai-runtime-2.json','local-ai-identical-2.json');runtime='artifacts/local-ai-gameplay-matrix-mock-2';privateReports=@('local-ai-runtime-2.json','local-ai-identical-2.json')},
 @{script='run-gameplay-live.ps1';args=@();reports=@('local-ai-gameplay-live.json');runtime='artifacts/local-ai-gameplay-runtime';privateReports=@('gameplay-result.json')},
 @{script='run-tactical.ps1';args=@();reports=@('local-ai-tactical.json');runtime='artifacts/local-ai-tactical-runtime';privateReports=@('tactical-result.json')},
 @{script='run-search-gameplay.ps1';args=@();reports=@('local-ai-search-gameplay.json');runtime='artifacts/patch-build/Managed';privateReports=@()},
 @{script='run-runtime.ps1';args=@('-GameplayOnly');reports=@('local-ai-runtime-0.json');runtime='artifacts/patch-build/Managed';privateReports=@()}
)
$results=@();$checks=0;$caseNumber=0
foreach($case in $cases){foreach($mode in @('success','no_output','nonzero_write','compiler_failure','hash_change')){
 $caseNumber++
 $fixture=Join-Path $auditRoot ('c'+$caseNumber)
 $fixtureScripts=Join-Path $fixture 'tests/BrassLocalAI'
 $fixtureBuild=Join-Path $fixture 'artifacts/patch-build'
 $fixtureManaged=Join-Path $fixtureBuild 'Managed'
 $fixtureCompiler=Join-Path $fixture 'artifacts/patch-tools/microsoft.net.compilers.toolset.4.12.0/tasks/net472'
 $fixtureRuntime=Join-Path $fixture $case.runtime
 New-Item -ItemType Directory -Path $fixtureScripts,$fixtureManaged,$fixtureCompiler,$fixtureRuntime -Force|Out-Null
 Copy-Item -LiteralPath (Join-Path $PSScriptRoot $case.script) -Destination $fixtureScripts
 Copy-Item -LiteralPath $mockCompiler -Destination (Join-Path $fixtureCompiler 'csc.exe')
 $pins=@(foreach($name in @('Assembly-CSharp.dll','BrassPatch.dll','BrassLegacyAI.dll')){
  [IO.File]::WriteAllText((Join-Path $fixtureManaged $name),'mock candidate '+$name)
  [pscustomobject]@{name=$name;sha256=(Get-FileHash -LiteralPath (Join-Path $fixtureManaged $name)).Hash}
 })
 @{version='mock';files=$pins}|ConvertTo-Json -Depth 8|Set-Content -LiteralPath (Join-Path $fixtureBuild 'build-manifest.json') -Encoding UTF8
 # Deliberately start with stale success everywhere each actual wrapper could read.
 $stale=@{passed=$true;checks=999;testedFiles=$pins}|ConvertTo-Json -Depth 8
 foreach($report in $case.reports){Set-Content -LiteralPath (Join-Path $fixtureBuild $report) -Value $stale -Encoding UTF8}
 foreach($report in $case.privateReports){Set-Content -LiteralPath (Join-Path $fixtureRuntime $report) -Value $stale -Encoding UTF8}
 $env:RUNNER_EVIDENCE_MODE=$mode
 $log=Join-Path $fixture 'runner.log'
 & (Get-Command pwsh).Source -NoProfile -File (Join-Path $fixtureScripts $case.script) @($case.args) *> $log
 $runnerExit=$LASTEXITCODE
 $reports=@(foreach($name in $case.reports){
  $report=Get-Content -LiteralPath (Join-Path $fixtureBuild $name) -Raw|ConvertFrom-Json
  [pscustomobject]@{name=$name;passed=($report.passed -eq $true);hasPins=($null -ne $report.testedFiles);status=$report.status}
 })
 $expectedSuccess=$mode -eq 'success'
 $valid=if($expectedSuccess){$runnerExit -eq 0 -and @($reports|Where-Object{-not $_.passed -or -not $_.hasPins}).Count -eq 0}else{$runnerExit -ne 0 -and @($reports|Where-Object{$_.passed}).Count -eq 0}
 $checks++
 $results+= [pscustomobject]@{runner=$case.script;mode=$mode;passed=$valid;runnerExit=$runnerExit;reports=$reports;log=$log}
 Write-Output ($case.script+' '+$mode+' '+$(if($valid){'PASS'}else{'FAIL'}))
}}
$passed=@($results|Where-Object{-not $_.passed}).Count -eq 0
$resultPath=Join-Path $auditRoot 'runner-evidence-result.json'
@{passed=$passed;checks=$checks;results=$results;scope='Copied wrappers, mock compiler and children, fake candidate trees only; no game execution or real patch-build output changes'}|ConvertTo-Json -Depth 15|Set-Content -LiteralPath $resultPath -Encoding UTF8
Write-Output $resultPath
if(-not $passed){throw 'Runner evidence regression failed; inspect isolated report'}
