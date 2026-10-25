param([string]$BuildDirectory='artifacts/network-input-build')
$ErrorActionPreference='Stop'
$root=Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
$build=Join-Path $root $BuildDirectory
. (Join-Path $root 'scripts/patch-install-common.ps1')
$candidate=Read-BrassJson (Join-Path $build 'build-manifest.json')
$baseline=Join-Path $root ('artifacts/patch-releases/'+$candidate.baseVersion)
$previous=Read-BrassJson (Join-Path $baseline 'build-manifest.json')
$old=Read-BrassJson (Join-Path $baseline 'validation.json')
if($candidate.version -ne '1.4.34' -or $previous.version -ne '1.4.33' -or -not $old.passed){throw 'Accepted 1.4.33 baseline required.'}
foreach($entry in (Read-BrassJson (Join-Path $baseline 'archive-manifest.json')).files){
 $path=[IO.Path]::GetFullPath((Join-Path $baseline $entry.path))
 if(-not $path.StartsWith([IO.Path]::GetFullPath($baseline)+[IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase) -or (Get-FileHash -LiteralPath $path).Hash -ne $entry.sha256){throw 'Baseline archive changed.'}
}
$changed=@()
foreach($pin in $candidate.sourcePins){
 if((Get-FileHash -LiteralPath (Join-Path $build ('runtime/'+$pin.path))).Hash -ne $pin.sha256){throw 'Compiled source snapshot changed.'}
 $prior=@($previous.sourcePins|Where-Object path -eq $pin.path)
 if($prior.Count -ne 1 -or $prior[0].sha256 -ne $pin.sha256){
  if($pin.path -notin $candidate.sourceScope -or (Get-FileHash -LiteralPath (Join-Path $root $pin.path)).Hash -ne $pin.sha256){throw 'Unrelated runtime source change.'}
  $changed+=$pin.path
 }
}
foreach($pin in $previous.sourcePins){if(@($candidate.sourcePins|Where-Object path -eq $pin.path).Count -ne 1){throw 'Baseline source missing.'}}
$compiler=Join-Path $root 'artifacts/patch-tools/microsoft.net.compilers.toolset.4.12.0/tasks/net472/csc.exe'
Copy-Item -LiteralPath (Join-Path $root 'artifacts/patch-tools/mono.cecil.0.11.6/lib/net40/Mono.Cecil.dll') -Destination $build -Force
Copy-Item -LiteralPath (Join-Path $root 'BrassBirminghamUnity_Data/Managed/Newtonsoft.Json.dll') -Destination $build -Force
$checker=Join-Path $build 'NetworkInputPreservedRuntimeChecks.exe'
@('/nologo','/target:exe','/langversion:7.3','/deterministic+',('/out:"'+$checker+'"'),('/reference:"'+(Join-Path $build 'Mono.Cecil.dll')+'"'),('/reference:"'+(Join-Path $build 'Newtonsoft.Json.dll')+'"'),('"'+(Join-Path $PSScriptRoot 'NetworkInputPreservedRuntimeChecks.cs')+'"'))|Set-Content -LiteralPath (Join-Path $build 'preservation.rsp') -Encoding utf8
& $compiler ('@'+(Join-Path $build 'preservation.rsp'))
if($LASTEXITCODE -ne 0){throw 'Preservation checker compilation failed.'}
& $checker (Join-Path $baseline 'Managed') (Join-Path $build 'Managed') (Join-Path $build 'preserved-runtime.json')
if($LASTEXITCODE -ne 0){throw 'A protected method or embedded profile changed.'}
$preserved=Read-BrassJson (Join-Path $build 'preserved-runtime.json')
$menu=Read-BrassJson (Join-Path $build 'menu-native.json')
$link=Read-BrassJson (Join-Path $build 'link-native.json')
$presentation=Read-BrassJson (Join-Path $build 'presentation-native.json')
$reproduction=Read-BrassJson (Join-Path $root 'artifacts/network-input-baseline/link-native.json')
foreach($report in $menu,$link,$presentation,$reproduction){if(-not $report -or -not $report.passed){throw 'Complete successful native reports are required before qualification.'}}
foreach($file in $previous.files){if(@($reproduction.testedFiles|Where-Object {$_.name -eq $file.name -and $_.sha256 -eq $file.sha256}).Count -ne 1){throw 'Reproduction did not use the accepted baseline.'}}
Copy-Item -LiteralPath (Join-Path $root 'artifacts/network-input-baseline/link-native.json') -Destination (Join-Path $build 'baseline-reproduction.json') -Force
$retained=[ordered]@{passed=$true;release=$candidate.baseVersion;validationSha256=(Get-FileHash -LiteralPath (Join-Path $baseline 'validation.json')).Hash;files=$previous.files;cardTrackerFiles=$old.cardTracker.testedFiles;preservedExistingMethods=($preserved.assemblies|Measure-Object unchangedMethods -Sum).Sum;scope='Only two previously unguarded Network entry methods change. All prior helper methods, AI methods and 48 profiles are preserved. Previous Statistics, History, Card Tracker and visual reports retain their original tested-file fingerprints. Fresh native Network, information panel and V4 turn tests qualify the candidate.'}
$ai=$old.localAi
$ai.files=$candidate.files;$ai.sourcePins=$candidate.sourcePins
$ai.resourceTests=$menu.resourceTests;$ai.executionStatistics=$menu.executionStatistics
$ai|Add-Member -NotePropertyName retainedFrom -NotePropertyValue $candidate.baseVersion -Force
$ai|Add-Member -NotePropertyName retainedEvidence -NotePropertyValue $retained -Force
$ai|Add-Member -NotePropertyName networkNative -NotePropertyValue $link -Force
$ai|Add-Member -NotePropertyName searchTimeNative -NotePropertyValue $menu -Force
$ai|ConvertTo-Json -Depth 45|Set-Content -LiteralPath (Join-Path $build 'local-ai-qualification.json') -Encoding utf8
$qualified=[ordered]@{passed=$true;validatedUtc=[DateTime]::UtcNow.ToString('o');files=$candidate.files;sourceScope=$changed;preservedRuntime=$preserved;baselineReproduction=$reproduction;networkInput=$link;presentation=$presentation;menu=$menu;visualChecks=$old.visualChecks;statistics=$old.statistics;history=$old.history;cardTracker=$old.cardTracker;retainedUi=$retained;localAi=$ai}|ConvertTo-Json -Depth 48|ConvertFrom-Json
Assert-BrassCandidate $candidate $qualified $build
$qualified|ConvertTo-Json -Depth 48|Set-Content -LiteralPath (Join-Path $build 'validation.json') -Encoding utf8
Write-Output ('Qualified Network input fix 1.4.34; '+$retained.preservedExistingMethods+' existing methods preserved.')
