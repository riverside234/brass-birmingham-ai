param([switch]$Smoke)
$ErrorActionPreference='Stop'
. (Join-Path $PSScriptRoot 'patch-install-common.ps1')
$root=Split-Path -Parent $PSScriptRoot
$build=Join-Path $root 'artifacts/patch-build'
$manifest=Read-BrassJson (Join-Path $build 'build-manifest.json')
$validation=Read-BrassJson (Join-Path $build 'validation.json')
$record=Read-BrassJson (Join-Path $root 'artifacts/patch-installation.json')
Assert-BrassCandidate $manifest $validation $build
if(Get-Process BrassBirminghamUnity -ErrorAction SilentlyContinue){throw 'Close the game before the installed-player check.'}
if($record.status -ne 'installed' -or $record.version -notin @('1.3.0','1.3.1','1.3.2','1.4.0','1.4.1','1.4.2','1.4.3','1.4.4','1.4.5')){throw 'Combined release is not installed'}
function Verify-Installed {
 foreach($file in $manifest.files){
  if((Get-FileHash -LiteralPath (Join-Path $root ('BrassBirminghamUnity_Data/Managed/'+$file.name))).Hash -ne $file.sha256){throw 'Installed file differs from candidate'}
  if(@($record.files|Where-Object {$_.name -eq $file.name -and $_.sha256 -eq $file.sha256}).Count -ne 1){throw 'Installation record differs'}
 }
}
Verify-Installed
$managed=Join-Path $build 'Managed'
$rsp=Join-Path $build 'installed-ai-tests.rsp'
@('/nologo','/target:library','/langversion:7.3','/nostdlib+','/deterministic+',('/out:"'+(Join-Path $managed 'LocalAiUnityTests.dll')+'"'))+@(Get-ChildItem -LiteralPath $managed -Filter '*.dll'|Where-Object {$_.Name -notmatch 'Tests|Regression'}|ForEach-Object {'/reference:"'+$_.FullName+'"'})+@('"'+(Join-Path $root 'tests/BrassLocalAI/LocalAiUnityTests.cs')+'"')|Set-Content -LiteralPath $rsp -Encoding UTF8
& (Join-Path $root 'artifacts/patch-tools/microsoft.net.compilers.toolset.4.12.0/tasks/net472/csc.exe') ('@'+$rsp)
if($LASTEXITCODE -ne 0){throw 'Installed-player test compilation failed'}
$path=Join-Path $build 'local-ai-native.json'
$isolated=[IO.File]::ReadAllBytes($path)
try{
 Remove-Item -LiteralPath $path
 $arguments=@('-batchmode','-logFile',('"'+(Join-Path $build 'local-ai-installed.log')+'"'),'--brass-local-ai-test')
 if($Smoke){$arguments+='--brass-local-ai-smoke'}
 $process=Start-Process -FilePath (Join-Path $root 'BrassBirminghamUnity.exe') -ArgumentList $arguments -WorkingDirectory $root -WindowStyle Hidden -PassThru
 $watch=[Diagnostics.Stopwatch]::StartNew()
 while(-not $process.HasExited){if($watch.Elapsed.TotalMinutes -gt 32){$process.Kill();throw 'Installed-player test timed out'};Start-Sleep -Milliseconds 500;$process.Refresh()}
 $result=Read-BrassJson $path
 if(-not $result -or -not $result.passed -or ($Smoke -and ($result.mode -ne 'startup-menu-smoke' -or $result.checks -lt 45)) -or (-not $Smoke -and $result.turns.Count -ne 108)){throw ('Installed AI test failed: '+$result.error)}
 Verify-Installed
 $result|Add-Member -NotePropertyName testedFiles -NotePropertyValue $manifest.files -Force
 $result|Add-Member -NotePropertyName installedLocation -NotePropertyValue $root -Force
 $result|ConvertTo-Json -Depth 15|Set-Content -LiteralPath (Join-Path $build 'local-ai-installed.json') -Encoding UTF8
 Write-Output $(if($Smoke){'Installed-player startup, strategy-selection and game-creation smoke checks passed.'}else{'Installed-player menu, selection, persistence and 108 native AI turn checks passed.'})
}finally{[IO.File]::WriteAllBytes($path,$isolated)}
