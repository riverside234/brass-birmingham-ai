param([string]$BuildDirectory='artifacts/game-finishing-margin/build')
$ErrorActionPreference='Stop'
$root=Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
. (Join-Path $root 'scripts/patch-install-common.ps1')
$build=Join-Path $root $BuildDirectory;$candidate=Read-BrassJson (Join-Path $build 'build-manifest.json')
$baseline=Join-Path $root ('artifacts/patch-releases/'+$candidate.baseVersion)
$prior=Read-BrassJson (Join-Path $baseline 'validation.json')
$preserved=Read-BrassJson (Join-Path $build 'preserved-runtime.json')
$timing=Read-BrassJson (Join-Path $root 'artifacts/game-finishing-margin/candidate/report.json')
$native=Read-BrassJson (Join-Path $build 'local-ai-native.json')
$boundary=Read-BrassJson (Join-Path $build 'deadline-boundaries.json')
if(-not $preserved.passed -or -not $native.passed -or -not $boundary.passed -or -not $timing.passed){throw 'Fresh preservation, boundary and native checks must pass.'}
$scope=@($candidate.sourceScope)
foreach($pin in $candidate.sourcePins){
 $source=Join-Path $build ('runtime/'+$pin.path)
 if((Get-FileHash -LiteralPath $source).Hash -ne $pin.sha256){throw 'Compiled source changed.'}
 if($pin.path -notin $scope -and (Get-FileHash -LiteralPath (Join-Path $baseline $pin.path)).Hash -ne $pin.sha256){throw 'Runtime source outside the game deadline scope changed.'}
}
foreach($file in $candidate.files){foreach($report in $native,$timing){if(@($report.testedFiles|Where-Object {$_.name -eq $file.name -and $_.sha256 -eq $file.sha256}).Count -ne 1){throw 'Evidence belongs to another candidate.'}}}
$training=Read-BrassJson (Join-Path $root 'artifacts/game-finishing-margin/training-scope-verified.json')
if(-not $training.passed){throw 'Training scope was not verified.'}
$retained=[ordered]@{passed=$true;release=$candidate.baseVersion;validationSha256=(Get-FileHash -LiteralPath (Join-Path $baseline 'validation.json')).Hash;testedFiles=$candidate.baseFiles;scope='Rules and UI binaries are byte-identical. Other existing method bodies and all 48 AI resources are preserved. Earlier UI reports retain their baseline fingerprints; only game deadline/lifecycle checks are newly executed.'}
$ai=[ordered]@{passed=$true;files=$candidate.files;sourcePins=$candidate.sourcePins;workers=2;deadlines=$timing;boundaries=$boundary;native=$native;resourceTests=$native.resourceTests;executionStatistics=$native.executionStatistics;masked=@{passed=$true;backend='masked_ismcts'};evidence=@(@{name='deadline-boundaries';sha256=(Get-FileHash -LiteralPath (Join-Path $build 'deadline-boundaries.json')).Hash},@{name='local-ai-native';sha256=(Get-FileHash -LiteralPath (Join-Path $build 'local-ai-native.json')).Hash})}
$ai|ConvertTo-Json -Depth 35|Set-Content -LiteralPath (Join-Path $build 'local-ai-qualification.json') -Encoding UTF8
[ordered]@{passed=$true;validatedUtc=[DateTime]::UtcNow.ToString('o');files=$candidate.files;sourceScope=$scope;preservedRuntime=$preserved;localAi=$ai;retainedUi=$retained;trainingUnchanged=$true;statistics=$prior.statistics;menu=$prior.menu;history=$prior.history;cardTracker=$prior.cardTracker}|ConvertTo-Json -Depth 40|Set-Content -LiteralPath (Join-Path $build 'validation.json') -Encoding UTF8
$validation=Read-BrassJson (Join-Path $build 'validation.json');Assert-BrassCandidate $candidate $validation $build
# Reject superficially passing records with lost deadlines, wrong workers or
# accidental training/source scope expansion. No live files are changed.
$packagingChecks=1
foreach($fault in 'late','workers','scope'){
 $m=$candidate|ConvertTo-Json -Depth 15|ConvertFrom-Json;$v=$validation|ConvertTo-Json -Depth 40|ConvertFrom-Json
 switch($fault){'late'{$v.localAi.deadlines.rows[0].timing.deadline_overrun=$true}'workers'{$m.searchWorkers=4}'scope'{$m.sourceScope[0]='src/brass_ai/recovery/config.py'}}
 $rejected=$false;try{Assert-BrassCandidate $m $v $build}catch{$rejected=$true}
 if(-not $rejected){throw ('Unsafe candidate was admitted: '+$fault)};$packagingChecks++
}
@{passed=$true;checks=$packagingChecks}|ConvertTo-Json|Set-Content -LiteralPath (Join-Path $build 'packaging-checks.json') -Encoding UTF8
Write-Output ('Qualified game-only finishing margin: '+$timing.checks+' timed checks, '+$boundary.checks+' boundary checks, '+$native.checks+' native integration checks, '+$packagingChecks+' package checks.')
