param([string]$BuildDirectory='artifacts/statistics-geography-build')
$ErrorActionPreference='Stop'
$geoRoot=Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
$geoBuild=Join-Path $geoRoot $BuildDirectory
. (Join-Path $geoRoot 'scripts/patch-install-common.ps1')
$candidate=Read-BrassJson (Join-Path $geoBuild 'build-manifest.json')
$baseline=Join-Path $geoRoot ('artifacts/patch-releases/'+$candidate.baseVersion)
$previous=Read-BrassJson (Join-Path $baseline 'build-manifest.json')
$old=Read-BrassJson (Join-Path $baseline 'validation.json')
if($candidate.version -ne '1.4.29' -or $previous.version -ne '1.4.27' -or -not $old.passed){throw 'Accepted 1.4.27 baseline required.'}
foreach($pin in (Read-BrassJson (Join-Path $baseline 'archive-manifest.json')).files){
 $path=[IO.Path]::GetFullPath((Join-Path $baseline $pin.path))
 if(-not $path.StartsWith([IO.Path]::GetFullPath($baseline)+[IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase) -or (Get-FileHash -LiteralPath $path).Hash -ne $pin.sha256){throw 'Accepted archive changed.'}
}
$changed=@()
foreach($pin in $candidate.sourcePins){
 if((Get-FileHash -LiteralPath (Join-Path $geoBuild ('runtime/'+$pin.path))).Hash -ne $pin.sha256){throw 'Compiled source snapshot changed.'}
 $prior=@($previous.sourcePins|Where-Object path -eq $pin.path)
 if($prior.Count -ne 1 -or $prior[0].sha256 -ne $pin.sha256){
  if($pin.path -notin $candidate.sourceScope -or $pin.path -notlike 'src/BrassPatch/Statistics/*' -or (Get-FileHash -LiteralPath (Join-Path $geoRoot $pin.path)).Hash -ne $pin.sha256){throw 'Unrelated runtime source change.'}
  $changed+=$pin.path
 }
}
foreach($pin in $previous.sourcePins){if(@($candidate.sourcePins|Where-Object path -eq $pin.path).Count -ne 1){throw 'Baseline source missing.'}}
foreach($name in 'Assembly-CSharp.dll','BrassLegacyAI.dll'){if(($candidate.files|Where-Object name -eq $name).sha256 -ne ($previous.files|Where-Object name -eq $name).sha256){throw 'Game or AI binary changed.'}}
$compiler=Join-Path $geoRoot 'artifacts/patch-tools/microsoft.net.compilers.toolset.4.12.0/tasks/net472/csc.exe'
Copy-Item -LiteralPath (Join-Path $geoRoot 'artifacts/patch-tools/mono.cecil.0.11.6/lib/net40/Mono.Cecil.dll') -Destination $geoBuild -Force
Copy-Item -LiteralPath (Join-Path $geoRoot 'BrassBirminghamUnity_Data/Managed/Newtonsoft.Json.dll') -Destination $geoBuild -Force
$checker=Join-Path $geoBuild 'StatisticsGeographyPreservedRuntimeChecks.exe'
@('/nologo','/target:exe','/langversion:7.3','/deterministic+',('/out:"'+$checker+'"'),('/reference:"'+(Join-Path $geoBuild 'Mono.Cecil.dll')+'"'),('/reference:"'+(Join-Path $geoBuild 'Newtonsoft.Json.dll')+'"'),('"'+(Join-Path $PSScriptRoot 'StatisticsGeographyPreservedRuntimeChecks.cs')+'"'))|Set-Content -LiteralPath (Join-Path $geoBuild 'preservation.rsp') -Encoding utf8
& $compiler ('@'+(Join-Path $geoBuild 'preservation.rsp'))
if($LASTEXITCODE -ne 0){throw 'Preservation checker compilation failed.'}
& $checker (Join-Path $baseline 'Managed') (Join-Path $geoBuild 'Managed') (Join-Path $geoBuild 'preserved-runtime.json')
if($LASTEXITCODE -ne 0){throw 'A method or resource outside the Geography scope changed.'}
$preserved=Read-BrassJson (Join-Path $geoBuild 'preserved-runtime.json')
$reports=@{}
foreach($name in 'statistics','history'){
 $report=Read-BrassJson (Join-Path $geoBuild ($name+'-native.json'))
 if(-not $report.passed -or $report.captureSkipped -or @($report.nativeErrors|Where-Object {$_}).Count){throw ('Native suite failed: '+$name)}
 foreach($file in $candidate.files){if(@($report.testedFiles|Where-Object {$_.name -eq $file.name -and $_.sha256 -eq $file.sha256}).Count -ne 1){throw 'Native report belongs to another candidate.'}}
 $reports[$name]=$report
}
# This file is deliberately authored only after inspecting the current captures.
$visual=Read-BrassJson (Join-Path $geoBuild 'visual-checks.json')
if(-not $visual.passed -or $visual.cases.Count -lt 12){throw 'Manual visual review is incomplete.'}
foreach($file in $candidate.files){if(@($visual.testedFiles|Where-Object {$_.name -eq $file.name -and $_.sha256 -eq $file.sha256}).Count -ne 1){throw 'Visual review belongs to another build.'}}
foreach($capture in $visual.cases){if((Get-FileHash -LiteralPath (Join-Path $geoRoot ('artifacts/card-tracker-player/artifacts/patch-build/'+$capture.name))).Hash -ne $capture.sha256){throw 'Reviewed capture changed.'}}
$retained=[ordered]@{passed=$true;release=$candidate.baseVersion;validationSha256=(Get-FileHash -LiteralPath (Join-Path $baseline 'validation.json')).Hash;files=$previous.files;menuFiles=$old.menu.testedFiles;trackerFiles=$old.cardTracker.testedFiles;preservedExistingMethods=($preserved.assemblies|Measure-Object unchangedMethods -Sum).Sum;scope='Gameplay, AI, History persistence, Card Tracker, selection UI and live-presentation suspension methods/resources preserved. Original menu, tracker and AI execution reports retain their original tested-file hashes. Fresh native Statistics and History suites qualify the shared Geography UI.'}
$ai=$old.localAi;$ai.files=$candidate.files;$ai.sourcePins=$candidate.sourcePins
$ai|Add-Member -NotePropertyName retainedFrom -NotePropertyValue $candidate.baseVersion -Force
$ai|Add-Member -NotePropertyName retainedEvidence -NotePropertyValue $retained -Force
$ai|ConvertTo-Json -Depth 40|Set-Content -LiteralPath (Join-Path $geoBuild 'local-ai-qualification.json') -Encoding utf8
[ordered]@{passed=$true;validatedUtc=[DateTime]::UtcNow.ToString('o');files=$candidate.files;sourceScope=$changed;preservedRuntime=$preserved;visualChecks=$visual;statistics=$reports.statistics;history=$reports.history;retainedUi=$retained;menu=$old.menu;cardTracker=$old.cardTracker;localAi=$ai}|ConvertTo-Json -Depth 42|Set-Content -LiteralPath (Join-Path $geoBuild 'validation.json') -Encoding utf8
Assert-BrassCandidate $candidate (Read-BrassJson (Join-Path $geoBuild 'validation.json')) $geoBuild
Write-Output ('Qualified Geography 1.4.29; '+$retained.preservedExistingMethods+' existing methods preserved.')
