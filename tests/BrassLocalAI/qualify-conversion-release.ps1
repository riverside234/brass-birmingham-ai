$ErrorActionPreference='Stop'
Set-StrictMode -Version Latest
$root=Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$build=Join-Path $root 'artifacts/patch-build'
$prior=Join-Path $root 'artifacts/patch-releases/1.4.0'
$candidate=Get-Content -LiteralPath (Join-Path $build 'build-manifest.json') -Raw|ConvertFrom-Json
if($candidate.version -ne '1.4.1'){throw 'This qualifier is for v3 conversion and cash management in package 1.4.1'}
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
$allowed=@('Policy/Conversion.cs','Policy/CashManagement.cs','Policy/Planner.cs','Policy/Evaluator.cs','Policy/ProfileCatalog.cs','Policy/LocalGameState.cs','Unity/LocalTurnBridge.cs')
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
$names=@('local-ai-runtime-0','local-ai-search-gameplay','local-ai-tactical','local-ai-native','local-ai-gameplay-live','local-ai-cash-regression','local-ai-conversion-0','local-ai-cash-beer-0','local-ai-demand-0','local-ai-route-0','local-ai-credit-0','local-ai-conversion-matches','local-ai-conversion-live-budget','local-ai-v2-save-compatibility','local-ai-v3-save-compatibility')
$evidence=@{}
foreach($name in $names){
 $path=Join-Path $build ($name+'.json');$item=Get-Content -LiteralPath $path -Raw|ConvertFrom-Json
 if(-not $item.passed){throw ('Failed evidence: '+$name)};Assert-Files $item.testedFiles;$evidence[$name]=$item
}
if(-not $evidence['local-ai-runtime-0'].saveTestsRun -or $evidence['local-ai-runtime-0'].checks -lt 1000 -or $evidence['local-ai-search-gameplay'].checks -lt 1000 -or $evidence['local-ai-tactical'].checks -lt 62){throw 'Core S1/settlement/tactics/save checks incomplete'}
$keys=@('balanced','rail-tempo','cotton','goods-beer','supplier','pottery-flex')
$native=$evidence['local-ai-native']
if($native.mode -ne 'full' -or @($native.nativeErrors).Count){throw 'Native run not clean/full'}
Assert-Set 'Native turns' @($native.turns|ForEach-Object {if(-not $_.inputGuardTested -or $_.thinkingMilliseconds -lt 10000 -or $_.workLimit -lt 1024){throw 'Native turn contract missing'};'{0}|{1}|{2}' -f $_.count,$_.intro,$_.key}) @(foreach($n in 2,3,4){foreach($intro in $false,$true){foreach($key in $keys){"$n|$intro|$key"}}})
Assert-Set 'Live games' @($evidence['local-ai-gameplay-live'].matches|ForEach-Object {if($_.aiActions -le 0 -or $_.humanActions -le 0){throw 'Missing live participation'};'{0}|{1}' -f $_.count,$_.intro}) @(foreach($n in 2,3,4){foreach($intro in $false,$true){"$n|$intro"}})
$cash=$evidence['local-ai-cash-regression']
if($cash.checks -lt 294){throw 'Cash recovery/borrowing-floor regressions are incomplete'}
Assert-Set 'Cash cases' @($cash.cases|Where-Object {$_.PSObject.Properties['key']}|ForEach-Object {'{0}|{1}|{2}' -f $_.key,$_.actions,$_.work}) @(foreach($key in $keys){foreach($actions in 1,2){foreach($work in 1,128){"$key|$actions|$work"}}})
$conversion=$evidence['local-ai-conversion-0']
$expected=@(foreach($rail in $false,$true){foreach($key in $keys){"deadline-sale|$rail|$key"}})+@(foreach($region in 'Gloucester','Warrington','Shrewsbury','Nottingham','Oxford'){foreach($key in $keys){"merchant-beer|$region|$key"}})+@(foreach($scenario in 'batch-sale-and-brewery','brewery-over-merchant'){foreach($key in $keys){"$scenario|$key"}})
Assert-Set 'Conversion fixtures' @($conversion.cases|ForEach-Object {
 if($_.scenario -eq 'deadline-sale'){"$($_.scenario)|$($_.rail)|$($_.key)"}elseif($_.scenario -eq 'merchant-beer'){"$($_.scenario)|$($_.bonusRegion)|$($_.key)"}else{"$($_.scenario)|$($_.key)"}
}) $expected
$research=Get-Content -LiteralPath (Join-Path $build 'local-ai-conversion-1.json') -Raw|ConvertFrom-Json
if(-not $research.passed -or ($research.cases|ConvertTo-Json -Depth 10 -Compress) -cne ($conversion.cases|ConvertTo-Json -Depth 10 -Compress)){throw 'Research/deployment fixture decisions differ'}
$current=Get-Content -LiteralPath (Join-Path $root 'training/opponents/current.json') -Raw|ConvertFrom-Json
$bank=Get-Content -LiteralPath (Join-Path $root ('training/opponents/bank/'+$current.bank+'/manifest.json')) -Raw|ConvertFrom-Json
foreach($file in $bank.payload.runtime_files){if(@($research.testedFiles|Where-Object {$_.name -eq $file.name -and $_.sha256 -eq $file.sha256}).Count -ne 1){throw 'Research fixtures did not test frozen V3 runtime'}}
$cashBeer=$evidence['local-ai-cash-beer-0']
Assert-Set 'Exhausted-budget cash/beer' @($cashBeer.cases|ForEach-Object {"$($_.key)|$($_.brewery)"}) @(foreach($key in $keys){foreach($brewery in $true,$false){"$key|$brewery"}})
$researchCashBeer=Get-Content -LiteralPath (Join-Path $build 'local-ai-cash-beer-1.json') -Raw|ConvertFrom-Json
if(-not $researchCashBeer.passed -or ($researchCashBeer.cases|ConvertTo-Json -Depth 10 -Compress) -cne ($cashBeer.cases|ConvertTo-Json -Depth 10 -Compress)){throw 'Research/deployment cash/beer fallbacks differ'}
foreach($file in $bank.payload.runtime_files){if(@($researchCashBeer.testedFiles|Where-Object {$_.name -eq $file.name -and $_.sha256 -eq $file.sha256}).Count -ne 1){throw 'Cash/beer fixtures did not test frozen V3 runtime'}}
$demand=$evidence['local-ai-demand-0'];$researchDemand=Get-Content -LiteralPath (Join-Path $build 'local-ai-demand-1.json') -Raw|ConvertFrom-Json
if($demand.checks -lt 19 -or -not $researchDemand.passed -or $researchDemand.checks -ne $demand.checks -or $researchDemand.signature -cne $demand.signature -or ($researchDemand.cases|ConvertTo-Json -Depth 10 -Compress) -cne ($demand.cases|ConvertTo-Json -Depth 10 -Compress)){throw 'Unsupported-brewery regression failed or differs by runtime'}
Assert-Set 'Demand work budgets' @($demand.cases|ForEach-Object {[string]$_.work}) @('64','240')
foreach($file in $bank.payload.runtime_files){if(@($researchDemand.testedFiles|Where-Object {$_.name -eq $file.name -and $_.sha256 -eq $file.sha256}).Count -ne 1){throw 'Demand fixture did not test frozen V3 runtime'}}
$credit=$evidence['local-ai-credit-0'];$researchCredit=Get-Content -LiteralPath (Join-Path $build 'local-ai-credit-1.json') -Raw|ConvertFrom-Json
Assert-Set 'Credit access and emergency borrowing' @($credit.cases|ForEach-Object {"$($_.key)|$($_.cash)"}) @(foreach($key in $keys){foreach($cash in 12,0){"$key|$cash"}})
if($credit.checks -lt 72 -or -not $researchCredit.passed -or ($researchCredit.cases|ConvertTo-Json -Depth 10 -Compress) -cne ($credit.cases|ConvertTo-Json -Depth 10 -Compress)){throw 'Credit preservation regression failed or differs by runtime'}
foreach($file in $bank.payload.runtime_files){if(@($researchCredit.testedFiles|Where-Object {$_.name -eq $file.name -and $_.sha256 -eq $file.sha256}).Count -ne 1){throw 'Credit fixtures did not test frozen V3 runtime'}}
$report=Get-Content -LiteralPath (Join-Path $root $current.qualification) -Raw|ConvertFrom-Json
$route=$evidence['local-ai-route-0'];$researchRoute=Get-Content -LiteralPath (Join-Path $build 'local-ai-route-1.json') -Raw|ConvertFrom-Json
if($route.checks -lt 24 -or -not $researchRoute.passed -or ($researchRoute.cases|ConvertTo-Json -Depth 10 -Compress) -cne ($route.cases|ConvertTo-Json -Depth 10 -Compress)){throw 'Merchant route regression differs by runtime'}
Assert-Set 'Merchant route budgets' @($route.cases|ForEach-Object {[string]$_.work}) @('64','240')
foreach($file in $bank.payload.runtime_files){if(@($researchRoute.testedFiles|Where-Object {$_.name -eq $file.name -and $_.sha256 -eq $file.sha256}).Count -ne 1){throw 'Route fixtures did not test frozen V3 runtime'}}
if($report.bank -ne $current.bank -or $report.failed_games -ne 0 -or @($report.profiles.PSObject.Properties|Where-Object {-not $_.Value.niche_qualified -or $_.Value.economic_stalls.Count}).Count){throw 'V3 bank admission incomplete'}
foreach($name in 'local-ai-conversion-matches','local-ai-conversion-live-budget'){
 $matches=$evidence[$name].matches
 foreach($match in $matches){
  if($match.actions -ne 124 -or $match.turns.Count -ne 4 -or @($match.turns|Where-Object {$_ -ne 16}).Count -or $match.avoidableShortfalls -ne 0){throw 'Four-player conversion/cash lifecycle failed'}
  Assert-Set 'Terminal seats' @($match.result.seats|ForEach-Object {[string]$_.seat}) @('0','1','2','3')
  if($match.eraReports.Count -ne 2){throw 'Missing era-end conversion measurement'}
 }
 if($name -eq 'local-ai-conversion-matches'){
  Assert-Set 'Conversion matches' @($matches|ForEach-Object {if($_.timed -or $_.work -ne 240){throw 'Comparison budget changed'};"$($_.seed)|$($_.revised)"}) @('72001|False','72001|True','72002|False','72002|True')
 }else{
  if($matches.Count -ne 1 -or -not $matches[0].timed -or -not $matches[0].revised -or $matches[0].seed -ne 72003 -or $matches[0].work -ne 1024 -or $matches[0].milliseconds -ne 10000){throw 'Normal-budget game incomplete'}
 }
}
foreach($name in 'local-ai-v2-save-compatibility','local-ai-v3-save-compatibility'){
 $compat=$evidence[$name]
 if($compat.checks -lt 48){throw 'Save compatibility checks incomplete'}
 Assert-Set 'Old saves' @($compat.games|ForEach-Object strategy) $keys
 $producer=if($name -eq 'local-ai-v2-save-compatibility'){'1.3.2'}else{'1.4.0'}
 $source=Get-Content -LiteralPath (Join-Path $root ('artifacts/patch-releases/'+$producer+'/build-manifest.json')) -Raw|ConvertFrom-Json
 foreach($file in $source.files){if(@($compat.producerFiles|Where-Object {$_.name -eq $file.name -and $_.sha256 -eq $file.sha256}).Count -ne 1){throw 'Old fixtures not made by archived binaries'}}
}
$retained=[ordered]@{release='1.4.0';validationSha256=(Get-FileHash -LiteralPath (Join-Path $prior 'validation.json')).Hash;unchangedFiles=@($candidate.files|Where-Object name -ne 'BrassLegacyAI.dll');scope='Earlier rules, persistence-helper and city-rendering checks retained only for byte-identical engine/helper DLLs.'}
$qualification=[ordered]@{passed=$true;qualifiedUtc=[DateTime]::UtcNow.ToString('o');files=$candidate.files;changedSources=$changes;sourcePins=$sourcePins;h5Provenance=$h5;retainedRulesEvidence=$retained;
 evidence=@($names|ForEach-Object {[ordered]@{name=$_;sha256=(Get-FileHash -LiteralPath (Join-Path $build ($_+'.json'))).Hash;checks=$evidence[$_].checks}});
 scope='V3 deadline and merchant-beer fixtures on identical research/deployment policy; cash, save and native UI regressions; complete four-player conversion games; separately admitted immutable V3 research bank.'}
$qualification|ConvertTo-Json -Depth 18|Set-Content -LiteralPath (Join-Path $build 'local-ai-qualification.json') -Encoding UTF8
$unity=$old.unityPlayerValidation;$unity|Add-Member -NotePropertyName retainedFrom -NotePropertyValue '1.4.0' -Force;$unity|Add-Member -NotePropertyName scope -NotePropertyValue $retained.scope -Force;$unity.testedFiles=$retained.unchangedFiles
[ordered]@{passed=$true;files=$candidate.files;localAi=$qualification;retainedRulesEvidence=$retained;unityPlayerValidation=$unity}|ConvertTo-Json -Depth 20|Set-Content -LiteralPath (Join-Path $build 'validation.json') -Encoding UTF8
Write-Output 'Qualified v3 conversion and research integration, package 1.4.1, with prior rules/helper binaries unchanged.'
