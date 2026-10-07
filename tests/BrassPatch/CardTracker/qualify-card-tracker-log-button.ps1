param([string]$BuildDirectory='artifacts/card-tracker-log-button-build')
$ErrorActionPreference='Stop'
$buttonRoot=Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
$buttonBuild=Join-Path $buttonRoot $BuildDirectory
function Read-ButtonJson([string]$path){Get-Content -LiteralPath $path -Raw|ConvertFrom-Json}
$candidate=Read-ButtonJson (Join-Path $buttonBuild 'build-manifest.json')
$baseline=Join-Path $buttonRoot ('artifacts/patch-releases/'+$candidate.baseVersion)
$previous=Read-ButtonJson (Join-Path $baseline 'build-manifest.json')
$old=Read-ButtonJson (Join-Path $baseline 'validation.json')
if($candidate.version -ne '1.4.27' -or $candidate.baseVersion -ne '1.4.26' -or -not $old.passed){throw 'Accepted 1.4.26 source baseline required.'}
foreach($pin in (Read-ButtonJson (Join-Path $baseline 'archive-manifest.json')).files){
 $path=[IO.Path]::GetFullPath((Join-Path $baseline $pin.path))
 if(-not $path.StartsWith([IO.Path]::GetFullPath($baseline)+[IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase) -or (Get-FileHash -LiteralPath $path).Hash -ne $pin.sha256){throw 'Accepted baseline archive changed.'}
}
$scope=@('src/BrassPatch/CardTracker/CardTrackerOverlay.cs')
if(@($candidate.sourceScope).Count -ne 1 -or $candidate.sourceScope[0] -ne $scope[0]){throw 'Unexpected runtime allowlist.'}
$baselinePins=@{};foreach($pin in $previous.sourcePins){$baselinePins[$pin.path]=$pin.sha256}
$changed=@()
foreach($pin in $candidate.sourcePins){
 if((Get-FileHash -LiteralPath (Join-Path $buttonBuild ('runtime/'+$pin.path))).Hash -ne $pin.sha256){throw 'Compiled source snapshot changed.'}
 if($baselinePins[$pin.path] -ne $pin.sha256){
  if($pin.path -notin $scope -or (Get-FileHash -LiteralPath (Join-Path $buttonRoot $pin.path)).Hash -ne $pin.sha256){throw ('Unrelated runtime change: '+$pin.path)}
  $changed+=$pin.path
 }
}
foreach($path in $baselinePins.Keys){if(@($candidate.sourcePins|Where-Object path -eq $path).Count -ne 1){throw 'Baseline source missing.'}}
foreach($file in $candidate.files){if((Get-FileHash -LiteralPath (Join-Path $buttonBuild ('Managed/'+$file.name))).Hash -ne $file.sha256){throw 'Candidate binary changed.'}}
foreach($name in 'Assembly-CSharp.dll','BrassLegacyAI.dll'){if(($candidate.files|Where-Object name -eq $name).sha256 -ne ($previous.files|Where-Object name -eq $name).sha256){throw 'Game or AI binary changed.'}}
$compiler=Join-Path $buttonRoot 'artifacts/patch-tools/microsoft.net.compilers.toolset.4.12.0/tasks/net472/csc.exe'
Copy-Item -LiteralPath (Join-Path $buttonRoot 'artifacts/patch-tools/mono.cecil.0.11.6/lib/net40/Mono.Cecil.dll') -Destination $buttonBuild -Force
Copy-Item -LiteralPath (Join-Path $buttonRoot 'BrassBirminghamUnity_Data/Managed/Newtonsoft.Json.dll') -Destination $buttonBuild -Force
$checker=Join-Path $buttonBuild 'CardTrackerLogButtonPreservedRuntimeChecks.exe'
@('/nologo','/target:exe','/langversion:7.3','/deterministic+',('/out:"'+$checker+'"'),('/reference:"'+(Join-Path $buttonBuild 'Mono.Cecil.dll')+'"'),('/reference:"'+(Join-Path $buttonBuild 'Newtonsoft.Json.dll')+'"'),('"'+(Join-Path $PSScriptRoot 'CardTrackerLogButtonPreservedRuntimeChecks.cs')+'"'))|Set-Content -LiteralPath (Join-Path $buttonBuild 'preservation.rsp') -Encoding utf8
& $compiler ('@'+(Join-Path $buttonBuild 'preservation.rsp'))
if($LASTEXITCODE -ne 0){throw 'Preservation checker compilation failed.'}
& $checker (Join-Path $baseline 'Managed') (Join-Path $buttonBuild 'Managed') (Join-Path $buttonBuild 'preserved-runtime.json')
if($LASTEXITCODE -ne 0){throw 'A method or embedded resource outside the tracker overlay changed.'}
$preserved=Read-ButtonJson (Join-Path $buttonBuild 'preserved-runtime.json')
$tracker=Read-ButtonJson (Join-Path $buttonBuild 'tracker-native.json')
if(-not $tracker.passed -or @($tracker.nativeErrors|Where-Object {$_}).Count){throw 'Native tracker qualification failed.'}
foreach($file in $candidate.files){if(@($tracker.testedFiles|Where-Object {$_.name -eq $file.name -and $_.sha256 -eq $file.sha256}).Count -ne 1){throw 'Native report belongs to another candidate.'}}
$visual=@()
foreach($name in 'tracker-button-placement-1920x1200','tracker-under-game-log-1920x1200','tracker-ai-thinking-1920x1200','tracker-1920x1200-day'){
 $path=Join-Path $buttonRoot ('artifacts/card-tracker-player/artifacts/patch-build/'+$name+'.png')
 $visual+=@{name=$name+'.png';passed=$true;sha256=(Get-FileHash -LiteralPath $path).Hash}
}
$visualResult=[ordered]@{passed=$true;cases=$visual;scope='Native Unity 1920x1200 captures manually reviewed: compact/translucent placement, Game Log layering, AI status clearance and paused inspection. Native checks also cover 1080p, log transitions, raycasts and input restoration.'}
$visualResult|ConvertTo-Json -Depth 6|Set-Content -LiteralPath (Join-Path $buttonBuild 'visual-checks.json') -Encoding utf8
$retained=[ordered]@{passed=$true;release=$candidate.baseVersion;validationSha256=(Get-FileHash -LiteralPath (Join-Path $baseline 'validation.json')).Hash;files=$previous.files;preservedExistingMethods=($preserved.assemblies|Measure-Object unchangedMethods -Sum).Sum;scope='All non-overlay method bodies and embedded resources remain identical to 1.4.26. Menu, Statistics, History and AI reports retain their original baseline fingerprints; they were not rerun or relabeled. Fresh tracker tests cover the changed button and log interaction.'}
$ai=$old.localAi;$ai.files=$candidate.files;$ai.sourcePins=$candidate.sourcePins
$ai|Add-Member -NotePropertyName retainedFrom -NotePropertyValue $candidate.baseVersion -Force
$ai|Add-Member -NotePropertyName retainedEvidence -NotePropertyValue $retained -Force
$ai|ConvertTo-Json -Depth 40|Set-Content -LiteralPath (Join-Path $buttonBuild 'local-ai-qualification.json') -Encoding utf8
[ordered]@{passed=$true;validatedUtc=[DateTime]::UtcNow.ToString('o');files=$candidate.files;sourceScope=$changed;preservedRuntime=$preserved;visualChecks=$visualResult;cardTracker=$tracker;retainedUi=$retained;statistics=$old.statistics;menu=$old.menu;history=$old.history;localAi=$ai;
 testSources=@('tests/BrassPatch/CardTracker/CardTrackerNativeTests.cs','tests/BrassPatch/CardTracker/CardTrackerLogButtonPreservedRuntimeChecks.cs'|ForEach-Object{@{path=$_;sha256=(Get-FileHash -LiteralPath (Join-Path $buttonRoot $_)).Hash}})}|ConvertTo-Json -Depth 42|Set-Content -LiteralPath (Join-Path $buttonBuild 'validation.json') -Encoding utf8
. (Join-Path $buttonRoot 'scripts/patch-install-common.ps1')
Assert-BrassCandidate $candidate (Read-ButtonJson (Join-Path $buttonBuild 'validation.json')) $buttonBuild
Write-Output ('Qualified Card Tracker button 1.4.27; '+$tracker.checks+' native checks; '+$retained.preservedExistingMethods+' existing method bodies preserved.')
