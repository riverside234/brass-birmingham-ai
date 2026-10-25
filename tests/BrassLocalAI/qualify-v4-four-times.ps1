param([string]$BuildDirectory='artifacts/v4-four-times-build')
$ErrorActionPreference='Stop'
$timingRoot=Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$timingBuild=Join-Path $timingRoot $BuildDirectory
. (Join-Path $timingRoot 'scripts/patch-install-common.ps1')
$candidate=Read-BrassJson (Join-Path $timingBuild 'build-manifest.json')
$baseline=Join-Path $timingRoot ('artifacts/patch-releases/'+$candidate.baseVersion)
$previous=Read-BrassJson (Join-Path $baseline 'build-manifest.json')
$old=Read-BrassJson (Join-Path $baseline 'validation.json')
if($candidate.version -ne '1.4.33' -or $previous.version -ne '1.4.32' -or -not $old.passed){throw 'Accepted 1.4.32 baseline required.'}
foreach($entry in (Read-BrassJson (Join-Path $baseline 'archive-manifest.json')).files){
 $path=[IO.Path]::GetFullPath((Join-Path $baseline $entry.path))
 if(-not $path.StartsWith([IO.Path]::GetFullPath($baseline)+[IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase) -or (Get-FileHash -LiteralPath $path).Hash -ne $entry.sha256){throw 'Baseline archive changed.'}
}
$changed=@()
foreach($pin in $candidate.sourcePins){
 if((Get-FileHash -LiteralPath (Join-Path $timingBuild ('runtime/'+$pin.path))).Hash -ne $pin.sha256){throw 'Compiled source snapshot changed.'}
 $prior=@($previous.sourcePins|Where-Object path -eq $pin.path)
 if($prior.Count -ne 1 -or $prior[0].sha256 -ne $pin.sha256){
  if($pin.path -notin $candidate.sourceScope -or (Get-FileHash -LiteralPath (Join-Path $timingRoot $pin.path)).Hash -ne $pin.sha256){throw 'Unrelated runtime source change.'}
  $changed+=$pin.path
 }
}
foreach($pin in $previous.sourcePins){if(@($candidate.sourcePins|Where-Object path -eq $pin.path).Count -ne 1){throw 'Baseline source missing.'}}
foreach($name in 'Assembly-CSharp.dll','BrassPatch.dll'){if(($candidate.files|Where-Object name -eq $name).sha256 -ne ($previous.files|Where-Object name -eq $name).sha256){throw 'Game or helper binary changed.'}}
$compiler=Join-Path $timingRoot 'artifacts/patch-tools/microsoft.net.compilers.toolset.4.12.0/tasks/net472/csc.exe'
Copy-Item -LiteralPath (Join-Path $timingRoot 'artifacts/patch-tools/mono.cecil.0.11.6/lib/net40/Mono.Cecil.dll') -Destination $timingBuild -Force
Copy-Item -LiteralPath (Join-Path $timingRoot 'BrassBirminghamUnity_Data/Managed/Newtonsoft.Json.dll') -Destination $timingBuild -Force
$checker=Join-Path $timingBuild 'V4FourTimesPreservedRuntimeChecks.exe'
@('/nologo','/target:exe','/langversion:7.3','/deterministic+',('/out:"'+$checker+'"'),('/reference:"'+(Join-Path $timingBuild 'Mono.Cecil.dll')+'"'),('/reference:"'+(Join-Path $timingBuild 'Newtonsoft.Json.dll')+'"'),('"'+(Join-Path $PSScriptRoot 'V4FourTimesPreservedRuntimeChecks.cs')+'"'))|Set-Content -LiteralPath (Join-Path $timingBuild 'preservation.rsp') -Encoding utf8
& $compiler ('@'+(Join-Path $timingBuild 'preservation.rsp'))
if($LASTEXITCODE -ne 0){throw 'Preservation checker compilation failed.'}
& $checker (Join-Path $baseline 'Managed') (Join-Path $timingBuild 'Managed') (Join-Path $timingBuild 'preserved-runtime.json')
if($LASTEXITCODE -ne 0){throw 'A protected method or embedded profile changed.'}
$preserved=Read-BrassJson (Join-Path $timingBuild 'preserved-runtime.json')
$menu=Read-BrassJson (Join-Path $timingBuild 'menu-native.json')
$visual=Read-BrassJson (Join-Path $timingBuild 'visual-checks.json')
foreach($capture in $visual.cases){if((Get-FileHash -LiteralPath (Join-Path $timingRoot ('artifacts/card-tracker-player/artifacts/patch-build/'+$capture.name))).Hash -ne $capture.sha256){throw 'Reviewed capture changed.'}}
$retained=[ordered]@{passed=$true;release=$candidate.baseVersion;validationSha256=(Get-FileHash -LiteralPath (Join-Path $baseline 'validation.json')).Hash;files=$previous.files;cardTrackerFiles=$old.cardTracker.testedFiles;preservedExistingMethods=($preserved.assemblies|Measure-Object unchangedMethods -Sum).Sum;scope='Game, Statistics, Game History, Card Tracker, search algorithm/coordinator, worker count and embedded profiles preserved. Previous reports retain their original tested-file fingerprints. Fresh native menu and nine live V4 turn cases qualify the changed clock integration.'}
$ai=$old.localAi
$ai.files=$candidate.files;$ai.sourcePins=$candidate.sourcePins
$ai.resourceTests=$menu.resourceTests;$ai.executionStatistics=$menu.executionStatistics
$ai|Add-Member -NotePropertyName retainedFrom -NotePropertyValue $candidate.baseVersion -Force
$ai|Add-Member -NotePropertyName retainedEvidence -NotePropertyValue $retained -Force
$ai|Add-Member -NotePropertyName searchTimeNative -NotePropertyValue $menu -Force
$ai|Add-Member -NotePropertyName turnOptionsMs -NotePropertyValue $candidate.turnOptionsMs -Force
$ai|ConvertTo-Json -Depth 45|Set-Content -LiteralPath (Join-Path $timingBuild 'local-ai-qualification.json') -Encoding utf8
[ordered]@{passed=$true;validatedUtc=[DateTime]::UtcNow.ToString('o');files=$candidate.files;sourceScope=$changed;preservedRuntime=$preserved;visualChecks=$visual;menu=$menu;statistics=$old.statistics;history=$old.history;cardTracker=$old.cardTracker;retainedUi=$retained;localAi=$ai}|ConvertTo-Json -Depth 48|Set-Content -LiteralPath (Join-Path $timingBuild 'validation.json') -Encoding utf8
Assert-BrassCandidate $candidate (Read-BrassJson (Join-Path $timingBuild 'validation.json')) $timingBuild
Write-Output ('Qualified four V4 search times 1.4.33; '+$retained.preservedExistingMethods+' existing methods preserved.')
