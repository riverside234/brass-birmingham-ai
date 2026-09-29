$ErrorActionPreference='Stop'
. (Join-Path $PSScriptRoot 'patch-install-common.ps1')
$root=Split-Path -Parent $PSScriptRoot
if(Get-Process -Name BrassBirminghamUnity -ErrorAction SilentlyContinue){throw 'Close the game before running the isolated Unity self-test.'}
$build=Join-Path $root 'artifacts/patch-build'
$recordPath=Join-Path $root 'artifacts/patch-installation.json'
$validationPath=Join-Path $build 'validation.json'
$record=Read-BrassJson $recordPath
$manifest=Read-BrassJson (Join-Path $build 'build-manifest.json')
$validation=Read-BrassJson $validationPath
Assert-BrassCandidate $manifest $validation $build
if(-not $record -or $record.gameRoot -ne $root -or $record.status -eq 'restored'){throw 'No current installation record for this game.'}
foreach($file in $manifest.files){
    $recorded=@($record.files|Where-Object {$_.name -eq $file.name})
    if($recorded.Count -ne 1 -or $recorded[0].sha256 -ne $file.sha256){throw 'Installed record belongs to a different candidate build.'}
    if((Get-FileHash -LiteralPath (Join-Path $root ('BrassBirminghamUnity_Data/Managed/'+$file.name))).Hash -ne $file.sha256){throw 'Installed files differ from the candidate being validated.'}
}
$result=Join-Path $build 'unity-selftest-result.json'
$paths=@(Get-BrassSaveDataPaths @($record))
$record|Add-Member -NotePropertyName saveDataPaths -NotePropertyValue $paths -Force
# The isolated test never opens player saves. Its runtime account's persistentDataPath
# is diagnostic information, not an additional real player save directory.
Write-BrassJsonAtomic $recordPath $record
if(Test-Path -LiteralPath $result){Remove-Item -LiteralPath $result}
$log=Join-Path $build 'unity-selftest.log'
$arguments=@('-batchmode','-nographics','-logFile',('"'+$log+'"'),'--brass-patch-selftest')
$process=Start-Process -FilePath (Join-Path $root 'BrassBirminghamUnity.exe') -ArgumentList $arguments -WorkingDirectory $root -WindowStyle Hidden -PassThru
$timer=[Diagnostics.Stopwatch]::StartNew()
while(-not $process.HasExited){
    if($timer.Elapsed.TotalMinutes -gt 5){$process.Kill();throw 'Isolated Unity self-test exceeded five minutes; inspect the log.'}
    Start-Sleep -Seconds 2
    $process.Refresh()
}
if(-not(Test-Path -LiteralPath $result)){throw ('Unity did not produce its result. Inspect '+$log)}
$data=Get-Content -LiteralPath $result -Raw|ConvertFrom-Json
if(-not $data.passed){throw ('Unity self-test failed: '+$data.error)}
foreach($file in $manifest.files){if((Get-FileHash -LiteralPath (Join-Path $root ('BrassBirminghamUnity_Data/Managed/'+$file.name))).Hash -ne $file.sha256){throw 'Installed files changed during Unity validation.'}}
$currentManifest=Read-BrassJson (Join-Path $build 'build-manifest.json')
$validation=Read-BrassJson $validationPath
Assert-BrassCandidate $currentManifest $validation $build
foreach($file in $manifest.files){if($file.sha256 -notin @(Get-BrassHashes @($currentManifest) $file.name)){throw 'Candidate changed during Unity validation.'}}
$data|Add-Member -NotePropertyName testedFiles -NotePropertyValue $manifest.files -Force
$validation|Add-Member -NotePropertyName unityPlayerValidation -NotePropertyValue $data -Force
Write-BrassJsonAtomic $validationPath $validation
$data|ConvertTo-Json -Depth 8
