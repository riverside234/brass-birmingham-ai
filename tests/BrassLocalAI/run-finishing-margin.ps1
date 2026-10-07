param([string]$BuildDirectory='artifacts/patch-releases/1.4.27',[string]$Label='baseline',[switch]$Candidate)
$ErrorActionPreference='Stop'
$root=Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$build=Join-Path $root $BuildDirectory
$output=Join-Path $root ('artifacts/game-finishing-margin/'+$Label)
$player=Join-Path $root 'artifacts/fm-player'
if(Get-Process BrassBirminghamUnity -ErrorAction SilentlyContinue){throw 'Another game/test player is running; leave it untouched.'}
if(-not(Test-Path -LiteralPath (Join-Path $player 'BrassBirminghamUnity_Data/globalgamemanagers'))){
 New-Item -ItemType Directory -Path $player -Force|Out-Null
 foreach($name in 'BrassBirminghamUnity.exe','UnityPlayer.dll','UnityCrashHandler64.exe'){Copy-Item -LiteralPath (Join-Path $root $name) -Destination $player}
 if(-not(Test-Path -LiteralPath (Join-Path $player 'MonoBleedingEdge'))){New-Item -ItemType Junction -Path (Join-Path $player 'MonoBleedingEdge') -Target (Join-Path $root 'MonoBleedingEdge')|Out-Null}
 $data=Join-Path $player 'BrassBirminghamUnity_Data';New-Item -ItemType Directory -Path $data -Force|Out-Null
 foreach($entry in Get-ChildItem -LiteralPath (Join-Path $root 'BrassBirminghamUnity_Data')){
  if($entry.Name -eq 'Managed'){continue}
  if($entry.PSIsContainer){if(-not(Test-Path -LiteralPath (Join-Path $data $entry.Name))){New-Item -ItemType Junction -Path (Join-Path $data $entry.Name) -Target $entry.FullName|Out-Null}}else{Copy-Item -LiteralPath $entry.FullName -Destination $data -Force}
 }
}
$nativeBuild=Join-Path $player 'artifacts/patch-build';$nativeManaged=Join-Path $nativeBuild 'Managed';$dataManaged=Join-Path $player 'BrassBirminghamUnity_Data/Managed'
New-Item -ItemType Directory -Path $output,$nativeManaged,$dataManaged -Force|Out-Null
foreach($destination in $nativeManaged,$dataManaged){
 Get-ChildItem -LiteralPath (Join-Path $root 'BrassBirminghamUnity_Data/Managed') -Filter '*.dll'|Where-Object Name -notmatch 'Tests|Regression'|Copy-Item -Destination $destination -Force
 foreach($name in 'Assembly-CSharp.dll','BrassPatch.dll','BrassLegacyAI.dll'){Copy-Item -LiteralPath (Join-Path $build ('Managed/'+$name)) -Destination $destination -Force}
}
Copy-Item -LiteralPath (Join-Path $root 'artifacts/game-finishing-margin/deadline-fixtures.json') -Destination $nativeBuild -Force
$compiler=Join-Path $root 'artifacts/patch-tools/microsoft.net.compilers.toolset.4.12.0/tasks/net472/csc.exe'
$rsp=Join-Path $output 'native.rsp'
@('/nologo','/target:library','/langversion:7.3','/nostdlib+','/deterministic+','/optimize+',('/out:"'+(Join-Path $nativeManaged 'LocalAiUnityTests.dll')+'"'))+
 $(if($Candidate){@('/define:FINISHING_MARGIN')}else{@()})+
 @(Get-ChildItem -LiteralPath $nativeManaged -Filter '*.dll'|Where-Object Name -notmatch 'Tests|Regression'|ForEach-Object{'/reference:"'+$_.FullName+'"'})+
 @('"'+(Join-Path $PSScriptRoot 'FinishingMarginNativeTests.cs')+'"')|Set-Content -LiteralPath $rsp -Encoding UTF8
& $compiler ('@'+$rsp);if($LASTEXITCODE -ne 0){throw 'Deadline native harness compilation failed.'}
Copy-Item -LiteralPath (Join-Path $nativeManaged 'LocalAiUnityTests.dll') -Destination $dataManaged -Force
$nativeOutput=Join-Path $nativeBuild 'local-ai-native.json';if(Test-Path -LiteralPath $nativeOutput){Remove-Item -LiteralPath $nativeOutput}
$process=Start-Process -FilePath (Join-Path $player 'BrassBirminghamUnity.exe') -ArgumentList @('-batchmode','-logFile',('"'+(Join-Path $output 'native.log')+'"'),'--brass-local-ai-test') -WorkingDirectory $player -WindowStyle Hidden -PassThru
$timer=[Diagnostics.Stopwatch]::StartNew()
while(-not $process.HasExited){if($timer.Elapsed.TotalMinutes -gt 8){$process.Kill();throw 'Deadline smoke timed out.'};Start-Sleep -Milliseconds 500;$process.Refresh()}
Copy-Item -LiteralPath $nativeOutput -Destination (Join-Path $output 'report.json') -Force
Copy-Item -LiteralPath (Join-Path $nativeBuild 'deadline-fixtures.json') -Destination (Join-Path $output 'deadline-fixtures.json') -Force
$report=Get-Content -LiteralPath (Join-Path $output 'report.json') -Raw|ConvertFrom-Json
if(-not $report.passed){throw ('Native deadline smoke failed: '+$report.error)}
$report|Add-Member -NotePropertyName testedFiles -NotePropertyValue @('Assembly-CSharp.dll','BrassPatch.dll','BrassLegacyAI.dll'|ForEach-Object{[ordered]@{name=$_;sha256=(Get-FileHash -LiteralPath (Join-Path $dataManaged $_)).Hash}}) -Force
$report|ConvertTo-Json -Depth 16|Set-Content -LiteralPath (Join-Path $output 'report.json') -Encoding UTF8
Write-Output ($Label+': '+$report.checks+' native checks; '+$report.rows.Count+' timed searches. '+(Join-Path $output 'report.json'))
