$ErrorActionPreference = 'Stop'
$workspace = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$review = Join-Path $workspace 'artifacts/reference-review'
$managed = Join-Path $workspace 'BrassBirminghamUnity_Data/Managed'
if (Get-Process -Name BrassBirminghamUnity -ErrorAction SilentlyContinue) { throw 'The game is running; no review test started.' }
$gameHash = (Get-FileHash (Join-Path $managed 'Assembly-CSharp.dll')).Hash
$helperHash = (Get-FileHash (Join-Path $managed 'BrassPatch.dll')).Hash
if ($gameHash -ne '34DD7A8200DA1BEFE41FF7532014E2CBFA6E6EEECCF2BCCAF3CB70DC7E272B5A') { throw 'Unexpected installed game build.' }
if ($helperHash -ne '4D574D81F40D7A2CFFD58AB73C9087BA219FF25456D3A9AF9080B05E8A1AB76D') { throw 'Unexpected installed patch build.' }
$validationPath = Join-Path $workspace 'artifacts/patch-build/validation.json'
$validationHash = (Get-FileHash -LiteralPath $validationPath).Hash
$compiler = Join-Path $workspace 'artifacts/patch-tools/microsoft.net.compilers.toolset.4.12.0/tasks/net472/csc.exe'
$testAssembly = Join-Path $review 'UnityPresentationRegression.dll'
$rsp = Join-Path $review 'phases-native.rsp'
$lines = @('/nologo','/target:library','/langversion:7.3','/nostdlib+','/deterministic+',('/out:"' + $testAssembly + '"'))
$lines += @(Get-ChildItem -LiteralPath $managed -Filter '*.dll' | ForEach-Object { '/reference:"' + $_.FullName + '"' })
$lines += '"' + (Join-Path $PSScriptRoot 'phases-native-probe.cs') + '"'
$lines | Set-Content -LiteralPath $rsp -Encoding UTF8
& $compiler ('@' + $rsp)
if ($LASTEXITCODE -ne 0) { throw 'Review native probe compilation failed.' }
$target = Join-Path $workspace 'artifacts/patch-build/Managed/UnityPresentationRegression.dll'
$selfTestResult = Join-Path $workspace 'artifacts/patch-build/unity-selftest-result.json'
$selfTestLog = Join-Path $workspace 'artifacts/patch-build/unity-selftest.log'
$backup = Join-Path $review ('native-backup-' + [Guid]::NewGuid().ToString('N').Substring(0,8))
New-Item -ItemType Directory -Path $backup | Out-Null
$records = @()
foreach ($path in @($target,$selfTestResult,$selfTestLog)) {
    $copy = Join-Path $backup ([IO.Path]::GetFileName($path))
    Copy-Item -LiteralPath $path -Destination $copy
    $records += [pscustomobject]@{path=$path; backup=$copy; hash=(Get-FileHash -LiteralPath $path).Hash}
}
$log = Join-Path $review 'phases-native-unity.log'
$runError = $null; $exitCode = $null; $restored = @()
try {
    Copy-Item -LiteralPath $testAssembly -Destination $target -Force
    $process = Start-Process -FilePath (Join-Path $workspace 'BrassBirminghamUnity.exe') -WorkingDirectory $workspace -WindowStyle Hidden -PassThru -ArgumentList @('-batchmode','-nographics','--brass-patch-selftest','-logFile',('"' + $log + '"'))
    if (-not $process.WaitForExit(45000)) { $process.Kill(); $process.WaitForExit(); throw 'Isolated Unity review probe timed out.' }
    $exitCode = $process.ExitCode
    Copy-Item -LiteralPath $selfTestResult -Destination (Join-Path $review 'phases-native-selftest-result.json') -Force
} catch { $runError = $_.Exception.ToString() }
finally {
    foreach ($record in $records) {
        Copy-Item -LiteralPath $record.backup -Destination $record.path -Force
        $match = (Get-FileHash -LiteralPath $record.path).Hash -eq $record.hash
        $restored += [ordered]@{path=$record.path; originalSha256=$record.hash; restored=$match}
        if (-not $match) { throw 'An original self-test artifact was not restored exactly.' }
    }
    $gameUnchanged = (Get-FileHash (Join-Path $managed 'Assembly-CSharp.dll')).Hash -eq $gameHash
    $helperUnchanged = (Get-FileHash (Join-Path $managed 'BrassPatch.dll')).Hash -eq $helperHash
    $validationUnchanged = (Get-FileHash -LiteralPath $validationPath).Hash -eq $validationHash
    [ordered]@{exitCode=$exitCode; runError=$runError; gameAssemblyUnchanged=$gameUnchanged; helperAssemblyUnchanged=$helperUnchanged; validationUnchanged=$validationUnchanged; originalValidationSha256=$validationHash; restoredArtifacts=$restored; originalGameSha256=$gameHash; originalHelperSha256=$helperHash} | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $review 'phases-native-run.json') -Encoding UTF8
    if (-not $gameUnchanged -or -not $helperUnchanged) { throw 'Installed binaries changed during review.' }
    if (-not $validationUnchanged) { throw 'Validation record changed during review.' }
}
if ($runError) { throw $runError }
Get-Content -LiteralPath (Join-Path $review 'phases-native-results.json')
