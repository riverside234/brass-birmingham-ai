$ErrorActionPreference='Stop'
Set-StrictMode -Version Latest
$root=Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$build=Join-Path $root 'artifacts/patch-build'
$prior=Join-Path $root 'artifacts/patch-releases/1.3.2'
$candidate=Get-Content -LiteralPath (Join-Path $build 'build-manifest.json') -Raw|ConvertFrom-Json
if($candidate.version -ne '1.4.0'){throw 'This qualifier is for v3 cash management in package 1.4.0'}
foreach($name in 'validation.json','local-ai-qualification.json'){@{passed=$false;status='qualifying'}|ConvertTo-Json|Set-Content -LiteralPath (Join-Path $build $name)}
$old=Get-Content -LiteralPath (Join-Path $prior 'validation.json') -Raw|ConvertFrom-Json
$previous=Get-Content -LiteralPath (Join-Path $prior 'build-manifest.json') -Raw|ConvertFrom-Json
if(-not $old.passed -or -not $old.unityPlayerValidation.passed){throw 'Qualified prior rules evidence is required'}
function Assert-Files($files){
 foreach($file in $candidate.files){
  if(@($files|Where-Object {$_.name -eq $file.name -and $_.sha256 -eq $file.sha256}).Count -ne 1){throw 'Evidence has the wrong fingerprint'}
  if((Get-FileHash -LiteralPath (Join-Path $build ('Managed/'+$file.name))).Hash -ne $file.sha256){throw 'Candidate changed'}
 }
}
function Assert-Set([string]$label,[string[]]$actual,[string[]]$expected){
 $set=[Collections.Generic.HashSet[string]]::new([StringComparer]::Ordinal)
 foreach($key in $actual){if(-not $set.Add($key)){throw ($label+' duplicate scenario')}}
 if($actual.Count -ne $expected.Count -or -not $set.SetEquals($expected)){throw ($label+' coverage is incomplete')}
}
foreach($file in $previous.files){
 if((Get-FileHash -LiteralPath (Join-Path $prior $file.name)).Hash -ne $file.sha256){throw 'Archive changed'}
 if(@($old.files|Where-Object {$_.name -eq $file.name -and $_.sha256 -eq $file.sha256}).Count -ne 1){throw 'Previous evidence mismatch'}
 if($file.name -ne 'BrassLegacyAI.dll' -and @($candidate.files|Where-Object {$_.name -eq $file.name -and $_.sha256 -eq $file.sha256}).Count -ne 1){throw 'Run fresh rules/UI-helper qualification for changed engine binaries'}
}
$allowed=@('Policy/CashManagement.cs','Policy/Planner.cs','Policy/Evaluator.cs','Policy/ProfileCatalog.cs','Policy/LocalGameState.cs','Unity/LocalTurnBridge.cs')
$sourceRoot=Join-Path $root 'src/BrassLocalAI';$oldRoot=Join-Path $prior 'BrassLocalAI';$changes=@();$sourcePins=@()
foreach($file in Get-ChildItem -LiteralPath $oldRoot -Recurse -Filter '*.cs'){
 $rel=[IO.Path]::GetRelativePath($oldRoot,$file.FullName)
 if(-not(Test-Path -LiteralPath (Join-Path $sourceRoot $rel))){throw 'Unexpected deleted source'}
}
foreach($file in Get-ChildItem -LiteralPath $sourceRoot -Recurse -Filter '*.cs'){
 $rel=[IO.Path]::GetRelativePath($sourceRoot,$file.FullName).Replace('\','/');$path=Join-Path $oldRoot $rel;$hash=(Get-FileHash -LiteralPath $file.FullName).Hash
 if(-not(Test-Path -LiteralPath $path) -or $hash -ne (Get-FileHash -LiteralPath $path).Hash){if($rel -notin $allowed){throw ('Out-of-scope source: '+$rel)};$changes+=$rel}
 $sourcePins+=[ordered]@{path=('src/BrassLocalAI/'+$rel);sha256=$hash}
}
& (Join-Path $PSScriptRoot 'verify-h5-provenance.ps1')
$h5=Get-Content -LiteralPath (Join-Path $build 'local-ai-h5.json') -Raw|ConvertFrom-Json
if(-not $h5.passed -or $h5.v3Profiles.Count -ne 6){throw 'Versioned profile provenance failed'}
$names=@('local-ai-runtime-0','local-ai-search-gameplay','local-ai-tactical','local-ai-native','local-ai-gameplay-live','local-ai-cash-regression','local-ai-cash-matches','local-ai-cash-live-budget','local-ai-v2-save-compatibility')
$evidence=@{}
foreach($name in $names){
 $path=Join-Path $build ($name+'.json');$item=Get-Content -LiteralPath $path -Raw|ConvertFrom-Json
 if(-not $item.passed){throw ('Failed evidence: '+$name)};Assert-Files $item.testedFiles;$evidence[$name]=$item
}
if(-not $evidence['local-ai-runtime-0'].saveTestsRun -or $evidence['local-ai-runtime-0'].checks -lt 1030 -or $evidence['local-ai-search-gameplay'].checks -lt 1441 -or $evidence['local-ai-tactical'].checks -lt 62){throw 'Core S1/settlement/tactics/save checks incomplete'}
$keys=@('balanced','rail-tempo','cotton','goods-beer','supplier','pottery-flex')
$native=$evidence['local-ai-native']
if($native.mode -ne 'full' -or @($native.nativeErrors).Count){throw 'Native run not clean/full'}
Assert-Set 'Native turns' @($native.turns|ForEach-Object {if(-not $_.inputGuardTested -or $_.thinkingMilliseconds -lt 10000 -or $_.workLimit -lt 1024){throw 'Native turn contract missing'};'{0}|{1}|{2}' -f $_.count,$_.intro,$_.key}) @(foreach($n in 2,3,4){foreach($intro in $false,$true){foreach($key in $keys){"$n|$intro|$key"}}})
Assert-Set 'Live games' @($evidence['local-ai-gameplay-live'].matches|ForEach-Object {if($_.aiActions -le 0 -or $_.humanActions -le 0){throw 'Missing live participation'};'{0}|{1}' -f $_.count,$_.intro}) @(foreach($n in 2,3,4){foreach($intro in $false,$true){"$n|$intro"}})
$cash=$evidence['local-ai-cash-regression']
if($cash.checks -lt 294){throw 'Cash recovery/borrowing-floor regressions are incomplete'}
Assert-Set 'Cash cases' @($cash.cases|Where-Object {$_.PSObject.Properties['key']}|ForEach-Object {'{0}|{1}|{2}' -f $_.key,$_.actions,$_.work}) @(foreach($key in $keys){foreach($actions in 1,2){foreach($work in 1,128){"$key|$actions|$work"}}})
$matches=$evidence['local-ai-cash-matches'].matches
$expected=@(foreach($key in $keys){"41024|$key|3|1"})+@(foreach($seed in 52028,52029){foreach($version in 2,3){"$seed|mixed|$version|48"}})
Assert-Set 'Four-AI tables' @($matches|ForEach-Object {
 if($_.actions -ne 124 -or $_.turns.Count -ne 4 -or @($_.turns|Where-Object {$_ -ne 16}).Count -or ($_.version -eq 3 -and $_.avoidableShortfalls -ne 0)){throw 'Four-player cash/progress requirement failed'}
 Assert-Set 'Terminal seats' @($_.result.seats|ForEach-Object {[string]$_.seat}) @('0','1','2','3')
 '{0}|{1}|{2}|{3}' -f $_.seed,$_.style,$_.version,$_.work
}) $expected
$timed=$evidence['local-ai-cash-live-budget'].matches
Assert-Set 'Deployed-budget tables' @($timed|ForEach-Object {
 if(-not $_.timed -or $_.work -ne 1024 -or $_.milliseconds -ne 10000 -or $_.actions -ne 124 -or $_.turns.Count -ne 4 -or @($_.turns|Where-Object {$_ -ne 16}).Count -or ($_.version -eq 3 -and $_.avoidableShortfalls -ne 0)){throw 'Deployed-budget four-AI game failed'}
 Assert-Set 'Timed terminal seats' @($_.result.seats|ForEach-Object {[string]$_.seat}) @('0','1','2','3')
 '{0}|{1}|{2}' -f $_.seed,$_.style,$_.version
}) @('52030|mixed|2','52030|mixed|3')
$compat=$evidence['local-ai-v2-save-compatibility']
if($compat.checks -lt 48){throw 'Save compatibility checks incomplete'}
Assert-Set 'Old saves' @($compat.games|ForEach-Object strategy) $keys
foreach($file in $previous.files){if(@($compat.producerFiles|Where-Object {$_.name -eq $file.name -and $_.sha256 -eq $file.sha256}).Count -ne 1){throw 'V2 fixtures were not made by the archived runtime'}}
$retained=[ordered]@{release='1.3.2';validationSha256=(Get-FileHash -LiteralPath (Join-Path $prior 'validation.json')).Hash;unchangedFiles=@($candidate.files|Where-Object name -ne 'BrassLegacyAI.dll');scope='Earlier rules, persistence-helper and city-rendering checks retained only for byte-identical engine/helper DLLs.'}
$qualification=[ordered]@{passed=$true;qualifiedUtc=[DateTime]::UtcNow.ToString('o');files=$candidate.files;changedSources=$changes;sourcePins=$sourcePins;h5Provenance=$h5;retainedRulesEvidence=$retained;
 evidence=@($names|ForEach-Object {[ordered]@{name=$_;sha256=(Get-FileHash -LiteralPath (Join-Path $build ($_+'.json'))).Hash;checks=$evidence[$_].checks}});
 scope='V3 cash regressions, six all-style four-AI tables, two paired mixed-table cash comparisons plus one pair at the deployed ten-second/1024-node budget, six live mixed games, 36 native turns, original S1/tactical/settlement checks and generated v2 save compatibility. Small cash/reliability sample, no strength certification.'}
$qualification|ConvertTo-Json -Depth 18|Set-Content -LiteralPath (Join-Path $build 'local-ai-qualification.json') -Encoding UTF8
$unity=$old.unityPlayerValidation;$unity|Add-Member -NotePropertyName retainedFrom -NotePropertyValue '1.3.2' -Force;$unity|Add-Member -NotePropertyName scope -NotePropertyValue $retained.scope -Force;$unity.testedFiles=$retained.unchangedFiles
[ordered]@{passed=$true;files=$candidate.files;localAi=$qualification;retainedRulesEvidence=$retained;unityPlayerValidation=$unity}|ConvertTo-Json -Depth 20|Set-Content -LiteralPath (Join-Path $build 'validation.json') -Encoding UTF8
Write-Output 'Qualified v3 cash management, package 1.4.0, with prior rules/helper binaries unchanged.'
