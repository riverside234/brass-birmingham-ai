$ErrorActionPreference='Stop'
$root=Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$build=Join-Path $root 'artifacts/patch-build'
$archive=Join-Path $root 'artifacts/patch-releases/1.4.8'
$candidate=Get-Content -LiteralPath (Join-Path $build 'build-manifest.json') -Raw|ConvertFrom-Json
$prior=Get-Content -LiteralPath (Join-Path $archive 'build-manifest.json') -Raw|ConvertFrom-Json
$old=Get-Content -LiteralPath (Join-Path $archive 'validation.json') -Raw|ConvertFrom-Json
if($candidate.version -ne '1.4.9' -or -not $old.passed -or -not $old.statistics.passed){throw 'Qualified 1.4.8 and candidate 1.4.9 are required'}
function Assert-Files($actual,$expected=$candidate.files){
 foreach($file in $expected){if(@($actual|Where-Object {$_.name -eq $file.name -and $_.sha256 -eq $file.sha256}).Count -ne 1){throw ('Evidence fingerprint differs: '+$file.name)}}
}
$inventory=Get-Content -LiteralPath (Join-Path $archive 'archive-manifest.json') -Raw|ConvertFrom-Json
foreach($entry in $inventory.files){if((Get-FileHash -LiteralPath (Join-Path $archive $entry.path)).Hash -ne $entry.sha256){throw 'Previous archive changed'}}
$unchanged=@($candidate.files|Where-Object name -ne 'BrassLegacyAI.dll')
Assert-Files $prior.files $unchanged
foreach($file in $candidate.files){if((Get-FileHash -LiteralPath (Join-Path $build ('Managed/'+$file.name))).Hash -ne $file.sha256){throw 'Candidate changed'}}
foreach($pin in $candidate.sourcePins){if((Get-FileHash -LiteralPath (Join-Path $root $pin.path)).Hash -ne $pin.sha256){throw ('Compiled source changed: '+$pin.path)}}
$reports=@{}
foreach($name in 'masked-runtime','local-ai-execution','local-ai-native','local-ai-conversion-0','local-ai-credit-0','local-ai-cash-beer-0','local-ai-v2-save-compatibility','local-ai-v3-save-compatibility','local-ai-1.4.3-save-compatibility'){
 $item=Get-Content -LiteralPath (Join-Path $build ($name+'.json')) -Raw|ConvertFrom-Json
 if(-not $item.passed){throw ('Failed evidence: '+$name)};Assert-Files $item.testedFiles;$reports[$name]=$item
}
$native=$reports['local-ai-native'];$execution=$reports['local-ai-execution'];$masked=$reports['masked-runtime']
if($masked.iterations -lt 2 -or $masked.backend -ne 'masked_ismcts' -or $execution.checks -lt 10){throw 'Masked/backend correctness is incomplete'}
if($native.turns.Count -ne 8 -or @($native.nativeErrors).Count -or -not $native.resourceTests.passed -or $native.resourceTests.checks -lt 20){throw 'Compact native/resource coverage is incomplete'}
if(($native.turns|Select-Object -ExpandProperty key -Unique).Count -ne 6){throw 'All six styles must play'}
foreach($difficulty in 'easy','mid','hard'){
 $ms=switch($difficulty){'easy'{2000}'mid'{5000}'hard'{12000}}
 $turns=@($native.turns|Where-Object difficulty -eq $difficulty)
 if(-not $turns.Count -or @($turns|Where-Object {$_.thinkingMilliseconds -ne $ms -or $_.workLimit -ne 1000000 -or -not $_.inputGuardTested -or $_.lastRequestMilliseconds -gt $ms}).Count){throw 'Clock/guard contract differs'}
}
foreach($stats in $native.executionStatistics,$execution.statistics){if($stats.maximumRunning -ne 1 -or $stats.running -or $stats.submitted -ne $stats.completed){throw 'Backend retirement is incomplete'}}
foreach($name in 'local-ai-v2-save-compatibility','local-ai-v3-save-compatibility','local-ai-1.4.3-save-compatibility'){if($reports[$name].checks -lt 54){throw 'Old save compatibility is incomplete'}}
$retained=[ordered]@{release='1.4.8';validationSha256=(Get-FileHash -LiteralPath (Join-Path $archive 'validation.json')).Hash;unchangedFiles=$unchanged;
 scope='Only Assembly-CSharp and BrassPatch evidence is retained, for byte-identical binaries. New AI helper has fresh runtime, native turn/resource, conversion/credit and saved-profile checks. No previous S1 strength qualification is transferred.'}
$qualification=[ordered]@{passed=$true;qualifiedUtc=[DateTime]::UtcNow.ToString('o');files=$candidate.files;sourcePins=$candidate.sourcePins;masked=$masked;
 evidence=@($reports.Keys|Sort-Object|ForEach-Object{[ordered]@{name=$_;sha256=(Get-FileHash -LiteralPath (Join-Path $build ($_+'.json'))).Hash}});
 resourceTests=$native.resourceTests;resourceMeasurements=$native.resourceMeasurements;executionStatistics=$native.executionStatistics;retainedEvidence=$retained;
 scope='Compact correctness/lifecycle qualification. All six v3 styles use Masked ISMCTS, exact 2/5/12 clocks, old 10-second saves and backend-independent resource management. Playing strength is unmeasured.'}
$qualification|ConvertTo-Json -Depth 24|Set-Content -LiteralPath (Join-Path $build 'local-ai-qualification.json') -Encoding UTF8
$statistics=$old.statistics;$statistics.testedFiles=$unchanged
$statistics|Add-Member -NotePropertyName retainedFrom -NotePropertyValue '1.4.8' -Force
$statistics|Add-Member -NotePropertyName retainedScope -NotePropertyValue $retained.scope -Force
$unity=$old.unityPlayerValidation;$unity.testedFiles=$unchanged
$unity|Add-Member -NotePropertyName retainedFrom -NotePropertyValue '1.4.8' -Force
[ordered]@{passed=$true;validatedUtc=[DateTime]::UtcNow.ToString('o');files=$candidate.files;localAi=$qualification;statistics=$statistics;unityPlayerValidation=$unity;retainedEvidence=$retained}|ConvertTo-Json -Depth 26|Set-Content -LiteralPath (Join-Path $build 'validation.json') -Encoding UTF8
Write-Output 'Qualified 1.4.9 Masked ISMCTS with compact fresh checks and byte-identical UI/rules evidence.'
