$ErrorActionPreference='Stop'
$trackerRoot=Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
$trackerBuild=Join-Path $trackerRoot 'artifacts/card-tracker-build'
function Read-TrackerReport([string]$path){Get-Content -LiteralPath $path -Raw|ConvertFrom-Json}
$trackerCandidate=Read-TrackerReport (Join-Path $trackerBuild 'build-manifest.json')
$trackerBase=Join-Path $trackerBuild ('baseline/'+$trackerCandidate.baseVersion)
if(-not(Test-Path -LiteralPath $trackerBase)){$trackerBase=Join-Path $trackerRoot ('artifacts/patch-releases/'+$trackerCandidate.baseVersion)}
$trackerPrevious=Read-TrackerReport (Join-Path $trackerBase 'build-manifest.json')
$trackerOld=Read-TrackerReport (Join-Path $trackerBase 'validation.json')
if($trackerCandidate.version -ne '1.4.13' -or $trackerCandidate.baseVersion -ne '1.4.12' -or -not $trackerOld.passed -or -not $trackerOld.localAi.passed){throw 'Qualified 1.4.12 and tracker appearance candidate 1.4.13 are required.'}
$trackerInventory=Read-TrackerReport (Join-Path $trackerBase 'archive-manifest.json')
foreach($entry in $trackerInventory.files){if((Get-FileHash -LiteralPath (Join-Path $trackerBase $entry.path)).Hash -ne $entry.sha256){throw 'Accepted baseline snapshot changed.'}}
function Assert-TrackerFiles($actual){foreach($file in $trackerCandidate.files){if(@($actual|Where-Object {$_.name -eq $file.name -and $_.sha256 -eq $file.sha256}).Count -ne 1){throw ('Evidence fingerprint differs: '+$file.name)}}}
foreach($file in $trackerCandidate.files){if((Get-FileHash -LiteralPath (Join-Path $trackerBuild ('Managed/'+$file.name))).Hash -ne $file.sha256){throw 'Candidate changed.'}}
if(($trackerCandidate.files|Where-Object name -eq 'BrassLegacyAI.dll').sha256 -ne ($trackerPrevious.files|Where-Object name -eq 'BrassLegacyAI.dll').sha256){throw 'Tracker changed the accepted AI binary.'}
if(($trackerCandidate.files|Where-Object name -eq 'Assembly-CSharp.dll').sha256 -ne ($trackerPrevious.files|Where-Object name -eq 'Assembly-CSharp.dll').sha256){throw 'Appearance revision changed the patched game assembly.'}
$trackerModified=@()
foreach($pin in $trackerCandidate.sourcePins){
 if((Get-FileHash -LiteralPath (Join-Path $trackerBuild ('runtime/'+$pin.path))).Hash -ne $pin.sha256){throw 'Compiled runtime snapshot changed.'}
 $before=@($trackerPrevious.sourcePins|Where-Object path -eq $pin.path)
 if($before.Count -ne 1 -or $before[0].sha256 -ne $pin.sha256){
  if($pin.path -notin @('src/BrassPatch/CardTracker/CardTrackerOverlay.cs','src/BrassPatch/CardTracker/CardTrackerWidgets.cs')){throw ('Unrelated runtime source changed: '+$pin.path)}
  if((Get-FileHash -LiteralPath (Join-Path $trackerRoot $pin.path)).Hash -ne $pin.sha256){throw ('Tracker source differs from compiled snapshot: '+$pin.path)}
  $trackerModified+=$pin.path
 }
}
foreach($pin in $trackerPrevious.sourcePins){if(@($trackerCandidate.sourcePins|Where-Object path -eq $pin.path).Count -ne 1){throw 'Baseline runtime source is missing.'}}
$trackerCompiler=Join-Path $trackerRoot 'artifacts/patch-tools/microsoft.net.compilers.toolset.4.12.0/tasks/net472/csc.exe'
$trackerPreservation=Join-Path $trackerBuild 'PreservedRuntimeChecks.exe'
@('/nologo','/target:exe','/langversion:7.3','/deterministic+',('/out:"'+$trackerPreservation+'"'),('/reference:"'+(Join-Path $trackerBuild 'Mono.Cecil.dll')+'"'),('/reference:"'+(Join-Path $trackerBuild 'Newtonsoft.Json.dll')+'"'),('"'+(Join-Path $PSScriptRoot 'PreservedRuntimeChecks.cs')+'"'))|Set-Content -LiteralPath (Join-Path $trackerBuild 'preservation.rsp') -Encoding utf8
& $trackerCompiler ('@'+(Join-Path $trackerBuild 'preservation.rsp'))
if($LASTEXITCODE -ne 0){throw 'Preservation checks did not compile.'}
& $trackerPreservation (Join-Path $trackerBase 'Managed') (Join-Path $trackerBuild 'Managed') (Join-Path $trackerBuild 'preserved-runtime.json') appearance
if($LASTEXITCODE -ne 0){throw 'An unrelated existing method changed.'}
$trackerPreserved=Read-TrackerReport (Join-Path $trackerBuild 'preserved-runtime.json')
$trackerReports=@{}
foreach($name in 'tracker','menu'){
 $item=Read-TrackerReport (Join-Path $trackerBuild ($name+'-native.json'))
 if(-not $item.passed -or @($item.nativeErrors|Where-Object {$_}).Count){throw ('Failed native evidence: '+$name)}
 Assert-TrackerFiles $item.testedFiles;$trackerReports[$name]=$item
}
if($trackerReports.tracker.checks -lt 530 -or $trackerReports.menu.checks -lt 259){throw 'Affected native coverage is incomplete.'}
foreach($name in 'AI turn disables tracker and rejects stale/direct clicks','turn/scoring animation disables tracker even for a human','replay disables tracker','in-progress human action disables tracker','opening freezes gameplay and all DOTween clocks','inspection leaves game and public history unchanged','disabled view restores pause state','projection reconciles with independent public discard-pile oracle'){
 if(-not @($trackerReports.tracker.cases|Where-Object {$_.name -eq $name -and $_.passed}).Count){throw ('Required acceptance evidence missing: '+$name)}
}
$trackerManagedBaseline=Join-Path $trackerRoot 'artifacts/patch-releases/1.4.10/validation.json'
$trackerManaged=Read-TrackerReport $trackerManagedBaseline
if(-not $trackerManaged.passed -or $trackerManaged.suites.Count -ne 12){throw 'Accepted managed rules/persistence baseline is missing.'}
$trackerRetained=[ordered]@{
 release='1.4.12';validationSha256=(Get-FileHash -LiteralPath (Join-Path $trackerBase 'validation.json')).Hash
 unchangedFiles=@($trackerCandidate.files|Where-Object name -in 'BrassLegacyAI.dll','Assembly-CSharp.dll')
 preservedExistingMethods=($trackerPreserved.assemblies|Measure-Object -Property unchangedMethods -Sum).Sum
 scope='Only CardTrackerOverlay and CardTrackerGlyph appearance/layout sources changed. The accepted AI and game DLLs are byte-identical to 1.4.12. Cecil verifies all helper methods outside CardTracker are unchanged. Public projection, pause, eligibility, rules, AI bridge, save/persistence, statistics and city highlights are preserved. Fresh tracker checks include actual visible hand cards, opacity, native screenshots and input/lifecycle behavior; fresh startup/menu/resource checks cover the complete candidate. Other accepted 1.4.12 regression evidence is retained for the unchanged implementation.'
}
$trackerAi=$trackerOld.localAi
$trackerAi.files=$trackerCandidate.files;$trackerAi.sourcePins=$trackerCandidate.sourcePins
$trackerAi.retainedEvidence=$trackerRetained
$trackerAi|Add-Member -NotePropertyName retainedFrom -NotePropertyValue '1.4.12' -Force
$trackerAi|ConvertTo-Json -Depth 26|Set-Content -LiteralPath (Join-Path $trackerBuild 'local-ai-qualification.json') -Encoding utf8
$trackerValidation=[ordered]@{
 passed=$true;validatedUtc=[DateTime]::UtcNow.ToString('o');files=$trackerCandidate.files
 cardTracker=$trackerReports.tracker;menu=$trackerReports.menu
 retainedUi=[ordered]@{passed=$true;release='1.4.12';statistics=$trackerOld.statistics;presentation=$trackerOld.presentation;linkReplay=$trackerOld.linkReplay;scope=$trackerRetained.scope}
 preservedRuntime=$trackerPreserved;localAi=$trackerAi;retainedEvidence=$trackerRetained
 managedRules=[ordered]@{passed=$true;retainedFrom='1.4.10';validationSha256=(Get-FileHash -LiteralPath $trackerManagedBaseline).Hash;suites=$trackerManaged.suites;scope=$trackerRetained.scope}
 sourceScope=$trackerModified
 evidence=@($trackerReports.Keys|Sort-Object|ForEach-Object{@{name=$_;sha256=(Get-FileHash -LiteralPath (Join-Path $trackerBuild ($_+'-native.json'))).Hash}})
}
$trackerValidation|ConvertTo-Json -Depth 30|Set-Content -LiteralPath (Join-Path $trackerBuild 'validation.json') -Encoding utf8
. (Join-Path $trackerRoot 'scripts/patch-install-common.ps1')
Assert-BrassCandidate $trackerCandidate (Read-TrackerReport (Join-Path $trackerBuild 'validation.json')) $trackerBuild
Write-Output ('Qualified Card Tracker appearance 1.4.13 with '+$trackerReports.tracker.checks+' tracker checks and '+$trackerRetained.preservedExistingMethods+' preserved method bodies.')
