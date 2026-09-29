$ErrorActionPreference='Stop'
Set-StrictMode -Version Latest
$root=Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$build=Join-Path $root 'artifacts/patch-build'
$prior=Join-Path $root 'artifacts/patch-releases/1.3.1'
$candidate=Get-Content -LiteralPath (Join-Path $build 'build-manifest.json') -Raw|ConvertFrom-Json
if($candidate.version -ne '1.3.2'){throw 'This scoped qualification applies only to the 1.3.2 AI/gameplay repair'}
foreach($name in 'validation.json','local-ai-qualification.json'){
 @{passed=$false;status='qualifying';files=$candidate.files}|ConvertTo-Json -Depth 10|Set-Content -LiteralPath (Join-Path $build $name) -Encoding UTF8
}
$previous=Get-Content -LiteralPath (Join-Path $prior 'build-manifest.json') -Raw|ConvertFrom-Json
$oldValidation=Get-Content -LiteralPath (Join-Path $prior 'validation.json') -Raw|ConvertFrom-Json
if(-not $oldValidation.passed -or -not $oldValidation.unityPlayerValidation.passed){throw 'Previous rules qualification is missing'}
function Assert-Files($files){
 foreach($file in $candidate.files){
  if(@($files|Where-Object {$_.name -eq $file.name -and $_.sha256 -eq $file.sha256}).Count -ne 1){throw 'Evidence belongs to a different candidate'}
  if((Get-FileHash -LiteralPath (Join-Path $build ('Managed/'+$file.name))).Hash -ne $file.sha256){throw 'Candidate changed'}
 }
}
function Get-StrategyKeys { @('balanced','rail-tempo','cotton','goods-beer','supplier','pottery-flex') }
function Assert-ScenarioSet([string]$label,[string[]]$actual,[string[]]$expected){
 $seen=[Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
 foreach($key in $actual){if(-not $seen.Add($key)){throw ($label+' has a duplicate scenario: '+$key)}}
 if($actual.Count -ne $expected.Count -or -not $seen.SetEquals($expected)){throw ($label+' scenario identities do not match the required coverage')}
}
function Assert-NativeCoverage($result){
 $expected=@(foreach($count in 2,3,4){foreach($intro in $false,$true){foreach($key in Get-StrategyKeys){"$count|$intro|$key"}}})
 $actual=@(foreach($turn in $result.turns){
  if($turn.intro -isnot [bool] -or $turn.inputGuardTested -cne $true -or
     $turn.humanTransitionTested -cne ($turn.count -eq 2 -and $turn.key -ceq 'balanced') -or
     $turn.thinkingMilliseconds -lt 10000 -or $turn.workLimit -lt 1024){throw 'Native input/transition or search-allowance coverage is incomplete'}
  '{0}|{1}|{2}' -f $turn.count,$turn.intro,$turn.key
 })
 if($result.mode -cne 'full' -or @($result.nativeErrors).Count){throw 'Native evidence is not a clean full run'}
 Assert-ScenarioSet 'Native turns' $actual $expected
}
function Assert-LiveCoverage($result){
 $keys=@(Get-StrategyKeys)
 $expected=@(foreach($count in 2,3,4){foreach($intro in $false,$true){"$count|$intro"}})
 $actual=@(foreach($match in $result.matches){
  if($match.intro -isnot [bool] -or $match.count -notin 2,3,4){throw 'Live game has an invalid configuration'}
  $offset=if($match.intro){2}else{0}
  $styles=@(for($seat=1;$seat -lt $match.count;$seat++){$keys[($seat+$match.count+$offset)%6]})
  Assert-ScenarioSet 'Live strategy assignments' @($match.strategies) $styles
  if($match.aiActions -le 0 -or $match.humanActions -le 0 -or
     ($match.intro -and $match.railTurns -ne 0) -or (-not $match.intro -and $match.railTurns -le 0)){throw 'Live game participation or era coverage is incomplete'}
  '{0}|{1}' -f $match.count,$match.intro
 })
 Assert-ScenarioSet 'Live games' $actual $expected
}
function Assert-MatrixCoverage($result,[int]$count,[string]$kind){
 $colors=switch($count){2{@(3,1)}3{@(3,0,2)}4{@(2,0,3,1)}default{throw 'Unsupported matrix player count'}}
 $expected=@(foreach($intro in $false,$true){foreach($key in Get-StrategyKeys){
  if($kind -eq 'identical'){"$count|$intro|$key|41024|-1"}
  elseif($kind -eq 'runtime'){
   foreach($seed in 41022,41023){foreach($color in $colors){$seat=($color+$seed%4)%4;"$count|$intro|$key|$seed|$seat"}}
  }else{throw 'Unsupported matrix kind'}
 }})
 $actual=@(foreach($match in $result.matches){
  if($match.count -ne $count -or $match.work -ne 1 -or $match.intro -isnot [bool]){throw 'Matrix row belongs to another player count, work limit or game mode'}
  $seats=@($colors|ForEach-Object {($_+$match.seed%4)%4})
  Assert-ScenarioSet 'Matrix terminal seats' @($match.result.seats|ForEach-Object {[string]$_.seat}) @($seats|ForEach-Object {[string]$_})
  '{0}|{1}|{2}|{3}|{4}' -f $match.count,$match.intro,$match.candidate,$match.seed,$match.candidateSeat
 })
 Assert-ScenarioSet ($kind+' '+$count+'-player matrix') $actual $expected
}
Assert-Files $candidate.files
foreach($file in $previous.files){
 if((Get-FileHash -LiteralPath (Join-Path $prior $file.name)).Hash -ne $file.sha256){throw 'Archived release changed'}
 if(@($oldValidation.files|Where-Object {$_.name -eq $file.name -and $_.sha256 -eq $file.sha256}).Count -ne 1){throw 'Archived validation fingerprint mismatch'}
 if($file.name -ne 'BrassLegacyAI.dll' -and @($candidate.files|Where-Object {$_.name -eq $file.name -and $_.sha256 -eq $file.sha256}).Count -ne 1){throw 'Rules/helper binary changed: run fresh rules qualification instead of inheriting evidence'}
}
# Restrict this release to the reviewed planner and UI repairs. The exact same
# rules/helper binaries retain their old validation; save-specific work is excluded.
$sourceChanges=@();$allowed=@('Policy/Planner.cs','Unity/LocalMenu.cs','Unity/LocalTurnBridge.cs','Unity/TurnInputGuard.cs')
$sourceRoot=Join-Path $root 'src/BrassLocalAI'
$oldSourceRoot=Join-Path $prior 'BrassLocalAI'
$currentSources=@(Get-ChildItem -LiteralPath $sourceRoot -Recurse -Filter '*.cs' -File)
$oldSources=@(Get-ChildItem -LiteralPath $oldSourceRoot -Recurse -Filter '*.cs' -File)
if($currentSources.Count -ne $oldSources.Count){throw 'AI source set changed beyond this repair'}
foreach($file in $currentSources){
 $relative=[IO.Path]::GetRelativePath($sourceRoot,$file.FullName).Replace('\','/')
 $old=Join-Path $oldSourceRoot $relative
 if(-not(Test-Path -LiteralPath $old)){throw 'Unexpected AI source file'}
 if((Get-FileHash -LiteralPath $file.FullName).Hash -ne (Get-FileHash -LiteralPath $old).Hash){
  if($relative -notin $allowed){throw ('Unreviewed AI source change: '+$relative)}
  $sourceChanges+=$relative
 }
}
& (Join-Path $PSScriptRoot 'verify-h5-provenance.ps1')
$h5=Get-Content -LiteralPath (Join-Path $build 'local-ai-h5.json') -Raw|ConvertFrom-Json
$evidence=@{}
foreach($name in @('local-ai-runtime-0','local-ai-native','local-ai-tactical','local-ai-search-gameplay','local-ai-gameplay-live')){
 $item=Get-Content -LiteralPath (Join-Path $build ($name+'.json')) -Raw|ConvertFrom-Json
 if(-not $item.passed){throw ('Failed evidence: '+$name)}
 Assert-Files $item.testedFiles
 $evidence[$name]=$item
}
if($evidence['local-ai-runtime-0'].saveTestsRun -or $evidence['local-ai-runtime-0'].checks -lt 1000){throw 'Gameplay runtime coverage is incomplete'}
Assert-NativeCoverage $evidence['local-ai-native']
Assert-LiveCoverage $evidence['local-ai-gameplay-live']
if($evidence['local-ai-gameplay-live'].matches.Count -ne 6 -or $evidence['local-ai-tactical'].checks -lt 62 -or $evidence['local-ai-search-gameplay'].checks -lt 1400){throw 'Gameplay/search/tactical coverage is incomplete'}
$matrices=@();$total=0
foreach($count in 2,3,4){foreach($kind in 'runtime','identical'){
 $path=Join-Path $build ("local-ai-$kind-$count.json")
 $result=Get-Content -LiteralPath $path -Raw|ConvertFrom-Json
 $expected=if($kind -eq 'runtime'){$count*24}else{12}
 if(-not $result.passed -or $result.matches.Count -ne $expected -or $result.workLimit -ne 1 -or $result.saveTestsRun){throw 'Fresh full-game matrix is incomplete'}
 Assert-MatrixCoverage $result $count $kind
 Assert-Files $result.testedFiles
 $total+=$result.matches.Count
 $matrices+=[ordered]@{kind=$kind;players=$count;matches=$result.matches.Count;assertions=$result.checks;workLimit=1;sha256=(Get-FileHash -LiteralPath $path).Hash;runtime=$result.runtime}
}}
$retained=[ordered]@{release='1.3.1';validationSha256=(Get-FileHash -LiteralPath (Join-Path $prior 'validation.json')).Hash;
 unchangedFiles=@($candidate.files|Where-Object name -ne 'BrassLegacyAI.dll');scope='Existing rules, persistence and city rendering evidence retained for byte-identical engine/helper DLLs; not rerun or relabeled as new tests.'}
$qualification=[ordered]@{passed=$true;qualifiedUtc=[DateTime]::UtcNow.ToString('o');files=$candidate.files;fullGames=$total;matrices=$matrices;
 gameplayLiveGames=$evidence['local-ai-gameplay-live'].matches;gameplayAssertions=$evidence['local-ai-gameplay-live'].checks;
 tacticalAssertions=$evidence['local-ai-tactical'].checks;searchAssertions=$evidence['local-ai-search-gameplay'].checks;smokeAssertions=$evidence['local-ai-runtime-0'].checks;
 nativeTurns=36;nativeAssertions=$evidence['local-ai-native'].checks;h5Provenance=$h5;changedSources=$sourceChanges;retainedRulesEvidence=$retained;
 scope='Fresh 252 complete strategy games at fixed one-node work, six live in-memory games, S1/tactical/settlement regressions, and 36 native UI turns on final binaries. Reliability evidence; no new strength claim. Save-game review and dedicated save tests excluded.'}
$qualification|ConvertTo-Json -Depth 15|Set-Content -LiteralPath (Join-Path $build 'local-ai-qualification.json') -Encoding UTF8
$unity=$oldValidation.unityPlayerValidation
$unity|Add-Member -NotePropertyName retainedFrom -NotePropertyValue '1.3.1' -Force
$unity|Add-Member -NotePropertyName scope -NotePropertyValue $retained.scope -Force
$unity.testedFiles=$retained.unchangedFiles
[ordered]@{passed=$true;validatedUtc=[DateTime]::UtcNow.ToString('o');files=$candidate.files;suites=@($evidence.Keys|Sort-Object);localAi=$qualification;retainedRulesEvidence=$retained;unityPlayerValidation=$unity}|ConvertTo-Json -Depth 18|Set-Content -LiteralPath (Join-Path $build 'validation.json') -Encoding UTF8
Write-Output 'Qualified 1.3.2: fresh AI/gameplay evidence and unchanged rules/helper fingerprints; save-game work excluded.'
