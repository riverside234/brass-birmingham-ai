$ErrorActionPreference='Stop'
Set-StrictMode -Version Latest
$root=Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$build=Join-Path $root 'artifacts/patch-build'
$archive=Join-Path $root 'artifacts/patch-releases/1.4.6'
$candidate=Get-Content -LiteralPath (Join-Path $build 'build-manifest.json') -Raw|ConvertFrom-Json
$previous=Get-Content -LiteralPath (Join-Path $archive 'build-manifest.json') -Raw|ConvertFrom-Json
$old=Get-Content -LiteralPath (Join-Path $archive 'validation.json') -Raw|ConvertFrom-Json
if($candidate.version -ne '1.4.7' -or -not $old.passed -or -not $old.localAi.passed -or -not $old.unityPlayerValidation.passed){throw 'Qualified 1.4.6 and candidate 1.4.7 are required'}
function Assert-Files($actual,$expected=$candidate.files){
 foreach($file in $expected){if(@($actual|Where-Object {$_.name -eq $file.name -and $_.sha256 -eq $file.sha256}).Count -ne 1){throw 'Evidence fingerprint mismatch'}}
}
$inventory=Get-Content -LiteralPath (Join-Path $archive 'archive-manifest.json') -Raw|ConvertFrom-Json
foreach($entry in $inventory.files){if((Get-FileHash -LiteralPath (Join-Path $archive $entry.path)).Hash -ne $entry.sha256){throw ('Prior release archive changed: '+$entry.path)}}
foreach($file in $candidate.files){if((Get-FileHash -LiteralPath (Join-Path $build ('Managed/'+$file.name))).Hash -ne $file.sha256){throw 'Candidate changed'}}
Assert-Files $candidate.files @($previous.files|Where-Object name -ne 'BrassLegacyAI.dll')
$allowed=@('Policy/Coordinator.cs','Unity/LocalTurnBridge.cs','Unity/AiTurnControls.cs')
$added=@('Execution/AiExecutionScheduler.cs','Unity/AiExecutionResources.cs','Unity/AiOptionalSimulation.cs')
$changed=@();$pins=@()
foreach($name in 'BrassPatch','BrassLocalAI'){
 $source=Join-Path $root ('src/'+$name);$prior=Join-Path $archive $name
 $files=@(Get-ChildItem -LiteralPath $source -Recurse -File|Where-Object Extension -in '.cs','.json')
 $previousFiles=@(Get-ChildItem -LiteralPath $prior -Recurse -File|Where-Object Extension -in '.cs','.json')
 if($files.Count -ne $previousFiles.Count+$(if($name -eq 'BrassLocalAI'){$added.Count}else{0})){throw 'Unexpected runtime source inventory'}
 foreach($file in $files){
  $relative=[IO.Path]::GetRelativePath($source,$file.FullName).Replace('\','/');$hash=(Get-FileHash -LiteralPath $file.FullName).Hash
  $oldPath=Join-Path $prior $relative
  if(-not (Test-Path -LiteralPath $oldPath)){
   if($name -ne 'BrassLocalAI' -or $relative -notin $added){throw ('Unexpected runtime source: '+$relative)}
  }elseif((Get-FileHash -LiteralPath $oldPath).Hash -ne $hash){
   if($name -ne 'BrassLocalAI' -or $relative -notin $allowed){throw ('Out-of-scope runtime change: '+$name+'/'+$relative)}
   $changed+=$relative
  }
  $pins+=[ordered]@{path=('src/'+$name+'/'+$relative);sha256=$hash}
 }
}
if((($changed|Sort-Object)-join '|') -ne (($allowed|Sort-Object)-join '|')){throw 'Expected coordinator, bridge, and status-cache changes'}
foreach($pin in $pins){if(@($candidate.sourcePins|Where-Object {$_.path -eq $pin.path -and $_.sha256 -eq $pin.sha256}).Count -ne 1){throw 'Current runtime source differs from compiled source'}}
if($candidate.sourcePins.Count -ne $pins.Count){throw 'Compilation source inventory differs'}
$reports=@{}
foreach($name in 'local-ai-execution','local-ai-runtime-0','local-ai-native','local-ai-gameplay-live','local-ai-v2-save-compatibility','local-ai-v3-save-compatibility','local-ai-1.4.3-save-compatibility'){
 $item=Get-Content -LiteralPath (Join-Path $build ($name+'.json')) -Raw|ConvertFrom-Json
 if(-not $item.passed){throw ('Failed evidence: '+$name)};Assert-Files $item.testedFiles;$reports[$name]=$item
}
$execution=$reports['local-ai-execution'];$native=$reports['local-ai-native'];$runtime=$reports['local-ai-runtime-0']
if($execution.checks -lt 10 -or $execution.statistics.maximumRunning -ne 1 -or $execution.statistics.running -or $execution.statistics.submitted -ne $execution.statistics.completed){throw 'Async backend/cancellation/cleanup coverage incomplete'}
if(-not $runtime.saveTestsRun -or $runtime.checks -lt 2339){throw 'Fresh S1/time-budget/save coverage incomplete'}
if($native.turns.Count -ne 108 -or @($native.nativeErrors).Count -or -not $native.resourceTests.passed -or $native.resourceTests.checks -lt 20){throw 'Fresh native resources and 108-turn matrix required'}
if($native.executionStatistics.maximumRunning -ne 1 -or $native.executionStatistics.running -or $native.executionStatistics.submitted -ne $native.executionStatistics.completed){throw 'Native worker retirement coverage incomplete'}
foreach($count in 2,3,4){foreach($intro in $false,$true){foreach($style in 'balanced','rail-tempo','cotton','goods-beer','supplier','pottery-flex'){foreach($difficulty in 'easy','mid','hard'){
 $turn=@($native.turns|Where-Object {$_.count -eq $count -and $_.intro -eq $intro -and $_.key -eq $style -and $_.difficulty -eq $difficulty})
 $ms=switch($difficulty){'easy'{2000}'mid'{5000}'hard'{12000}}
 if($turn.Count -ne 1 -or $turn[0].thinkingMilliseconds -ne $ms -or $turn[0].workLimit -ne 240 -or -not $turn[0].inputGuardTested -or $turn[0].lastRequestMilliseconds -gt $ms){throw 'Native configuration or deadline contract missing'}
}}}}
if(@($native.turns|Where-Object initialQueueDelayTested).Count -ne 1 -or @($native.turns|Where-Object replanInjected).Count -ne 1){throw 'Initial queue exclusion and shared-deadline replan must be exercised'}
foreach($name in 'local-ai-v2-save-compatibility','local-ai-v3-save-compatibility','local-ai-1.4.3-save-compatibility'){if($reports[$name].checks -lt 54){throw 'Older save producer coverage incomplete'}}
& (Join-Path $PSScriptRoot 'verify-h5-provenance.ps1')
$h5=Get-Content -LiteralPath (Join-Path $build 'local-ai-h5.json') -Raw|ConvertFrom-Json
if(-not $h5.passed -or $h5.v3Profiles.Count -ne 6){throw 'Immutable policy/profile provenance failed'}
$retained=[ordered]@{release='1.4.6';validationSha256=(Get-FileHash -LiteralPath (Join-Path $archive 'validation.json')).Hash;
 unchangedFiles=@($candidate.files|Where-Object name -ne 'BrassLegacyAI.dll');
 scope='Game engine and BrassPatch helper byte-identical to 1.4.6; all rule, replay, persistence, menu and policy sources preserved. Archived rules/link/presentation evidence retained for those unchanged binaries. Coordinator/resource execution, 108 native turns, menu, budgets and old-save compatibility rerun freshly.'}
$testPins=@(Get-ChildItem -LiteralPath $PSScriptRoot -File|Where-Object Name -in 'AiExecutionTests.cs','AiExecutionResourceTests.cs','LocalAiUnityTests.cs','run-execution.ps1','run-unity.ps1','qualify-execution-release.ps1'|ForEach-Object{
 [ordered]@{path=('tests/BrassLocalAI/'+$_.Name);sha256=(Get-FileHash -LiteralPath $_.FullName).Hash}
})
$qualification=[ordered]@{passed=$true;qualifiedUtc=[DateTime]::UtcNow.ToString('o');files=$candidate.files;sourcePins=$pins;testPins=$testPins;changedSources=$changed;addedSources=$added;h5Provenance=$h5;retainedEvidence=$retained;
 evidence=@($reports.Keys|Sort-Object|ForEach-Object{[ordered]@{name=$_;sha256=(Get-FileHash -LiteralPath (Join-Path $build ($_+'.json'))).Hash}});
 resourceTests=$native.resourceTests;resourceMeasurements=$native.resourceMeasurements;executionStatistics=$native.executionStatistics;
 scope='Backend-independent execution/resource lifecycle; actual async/GPU completion gates the next turn. 30 FPS during computation; captured FPS/VSync restored for presentation and all teardown. Registered optional work only; no global physics/time changes, forced GC, thread-pool reset, policy/budget/save-format change, or model-weight unloading.'}
$qualification|ConvertTo-Json -Depth 22|Set-Content -LiteralPath (Join-Path $build 'local-ai-qualification.json') -Encoding UTF8
$unity=$old.unityPlayerValidation
$unity|Add-Member -NotePropertyName retainedFrom -NotePropertyValue '1.4.6' -Force
$unity|Add-Member -NotePropertyName scope -NotePropertyValue $retained.scope -Force
$unity.testedFiles=$retained.unchangedFiles
[ordered]@{passed=$true;validatedUtc=[DateTime]::UtcNow.ToString('o');files=$candidate.files;localAi=$qualification;retainedEvidence=$retained;unityPlayerValidation=$unity}|ConvertTo-Json -Depth 24|Set-Content -LiteralPath (Join-Path $build 'validation.json') -Encoding UTF8
Write-Output 'Qualified 1.4.7: generic resource lifecycle, async retirement, 108 native configurations and preserved frozen policy verified.'
