param([string]$BuildDirectory='artifacts/economy-v3/package-1.4.20-v4',[string]$Bank='')
$ErrorActionPreference='Stop'
$root=Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
. (Join-Path $root 'scripts/patch-install-common.ps1')
$build=Join-Path $root $BuildDirectory
$candidate=Read-BrassJson (Join-Path $build 'build-manifest.json')
$userSelectedV4=$candidate.version -eq '1.4.20' -and $candidate.selectionBasis -eq 'user-directed-masked-v4/1.0.0'
$archive=Join-Path $root ('artifacts/patch-releases/'+$candidate.baseVersion)
$prior=Read-BrassJson (Join-Path $archive 'build-manifest.json')
$old=Read-BrassJson (Join-Path $archive 'validation.json')
$inventory=Read-BrassJson (Join-Path $archive 'archive-manifest.json')
if($candidate.version -notin @('1.4.15','1.4.16','1.4.18','1.4.20') -or -not $old.passed -or -not $inventory){throw 'Candidate and qualified UI baseline are required'}
$gameWorkers=if($candidate.version -in @('1.4.16','1.4.18','1.4.20')){2}else{4}
foreach($entry in $inventory.files){if((Get-FileHash -LiteralPath (Join-Path $archive $entry.path)).Hash -ne $entry.sha256){throw 'Baseline archive changed'}}
function Assert-Files($actual,$expected=$candidate.files){
 foreach($file in $expected){if(@($actual|Where-Object {$_.name -eq $file.name -and $_.sha256 -eq $file.sha256}).Count -ne 1){throw ('Evidence fingerprint differs: '+$file.name)}}
}
$unchanged=@($candidate.files|Where-Object name -ne 'BrassLegacyAI.dll')
Assert-Files $prior.files $unchanged
foreach($file in $candidate.files){if((Get-FileHash -LiteralPath (Join-Path $build ('Managed/'+$file.name))).Hash -ne $file.sha256){throw 'Candidate binary changed'}}
foreach($pin in $candidate.sourcePins){
 foreach($path in (Join-Path $root $pin.path),(Join-Path $build ('source/'+$pin.path))){if((Get-FileHash -LiteralPath $path).Hash -ne $pin.sha256){throw ('Compiled source changed: '+$pin.path)}}
}
$reports=@{}
$reportNames=@('masked-runtime','era-runtime','local-ai-execution','local-ai-native','local-ai-conversion-0','local-ai-credit-0','local-ai-cash-beer-0','save-1.3.2','save-1.4.0','save-1.4.3','save-1.4.9','save-1.4.11')
if($candidate.version -in @('1.4.16','1.4.18','1.4.20')){$reportNames+='save-1.4.15'}
if($candidate.version -in @('1.4.18','1.4.20')){$reportNames+=@('industry-runtime','save-1.4.17')}
if($candidate.version -eq '1.4.20'){$reportNames+=@('economy-runtime','save-1.4.18')}
foreach($name in $reportNames){
 $item=Read-BrassJson (Join-Path $build ($name+'.json'))
 if(-not $item.passed){throw ('Failed evidence: '+$name)};Assert-Files $item.testedFiles;$reports[$name]=$item
}
$native=$reports['local-ai-native'];$execution=$reports['local-ai-execution'];$masked=$reports['masked-runtime'];$era=$reports['era-runtime']
if($masked.iterations -lt 8 -or $masked.backend -ne 'masked_ismcts' -or $execution.checks -lt 10 -or $era.checks -lt 19 -or $era.workers -ne 4 -or $era.cpu_batching.pending){throw 'Search/forest correctness is incomplete'}
if($candidate.version -in @('1.4.16','1.4.18','1.4.20') -and ($candidate.searchWorkers -ne 2 -or $era.game_workers -ne 2 -or $era.checks -lt 22 -or $reports['save-1.4.15'].checks -lt 252)){throw 'Two game workers and saved era profiles must pass'}
if($native.turns.Count -ne 8 -or @($native.nativeErrors).Count -or -not $native.resourceTests.passed -or $native.resourceTests.checks -lt 20){throw 'Native/resource coverage is incomplete'}
if(($native.turns|Select-Object -ExpandProperty key -Unique).Count -ne 6){throw 'All six styles must play'}
foreach($difficulty in 'easy','mid','hard'){
 $ms=switch($difficulty){'easy'{2000}'mid'{5000}'hard'{12000}}
 $turns=@($native.turns|Where-Object difficulty -eq $difficulty)
 if(-not $turns.Count -or @($turns|Where-Object {$_.thinkingMilliseconds -ne $ms -or $_.workLimit -ne 1000000 -or -not $_.inputGuardTested -or $_.lastRequestMilliseconds -gt $ms}).Count){throw 'Clock/guard contract differs'}
}
foreach($stats in $native.executionStatistics,$execution.statistics){if($stats.maximumRunning -ne 1 -or $stats.running -or $stats.submitted -ne $stats.completed){throw 'Backend retirement is incomplete'}}
foreach($name in 'save-1.3.2','save-1.4.0','save-1.4.3'){if($reports[$name].checks -lt 54){throw 'Older save compatibility is incomplete'}}
foreach($name in 'save-1.4.9','save-1.4.11'){if($reports[$name].checks -lt 252){throw 'Saved Masked profiles/clocks are incomplete'}}
# Replay check counts depend on the selected continuation length. The v4
# merchant fixture starts inside a sale so a legitimate early-Rail build does
# not make a beer-payment test fail. Verify every named case, not path length.
$conversion=$reports['local-ai-conversion-0']
$minimumConversion=if($userSelectedV4){400}else{435}
if($conversion.checks -lt $minimumConversion -or $reports['local-ai-credit-0'].checks -lt 72 -or $reports['local-ai-cash-beer-0'].checks -lt 60){throw 'Cash/conversion regression coverage is incomplete'}
$conversionCoverage=$null
if($userSelectedV4){
 if(@($conversion.cases).Count -ne 54){throw 'All v4 conversion scenarios are required'}
 foreach($style in 'balanced','rail-tempo','cotton','goods-beer','supplier','pottery-flex'){
  $cases=@($conversion.cases|Where-Object key -eq $style)
  foreach($rail in $false,$true){if(@($cases|Where-Object {$_.scenario -eq 'deadline-sale' -and $_.rail -eq $rail}).Count -ne 1){throw 'Canal/Rail deadline sale coverage is required for every style'}}
  foreach($merchant in 'Gloucester','Warrington','Shrewsbury','Nottingham','Oxford'){if(@($cases|Where-Object {$_.scenario -eq 'merchant-beer' -and $_.bonusRegion -eq $merchant}).Count -ne 1){throw 'Each merchant bonus must be checked for every style'}}
  foreach($scenario in 'batch-sale-and-brewery','brewery-over-merchant'){if(@($cases|Where-Object scenario -eq $scenario).Count -ne 1){throw 'Grouped sale and brewery/merchant tradeoff coverage is required for every style'}}
 }
 $conversionCoverage=[ordered]@{passed=$true;styles=6;deadlineSales=12;merchantBonuses=30;batchSales=6;breweryTradeoffs=6;checks=$conversion.checks;scope='All named conversion scenarios; merchant payment is tested within an already initiated sale.'}
}
$current=Read-BrassJson (Join-Path $root 'training/opponents/current.json')
if(-not $Bank){$Bank=$current.bank}
$bankManifest=Read-BrassJson (Join-Path $root ('training/opponents/bank/'+$Bank+'/manifest.json'))
$screen=Read-BrassJson (Join-Path $root $current.qualification)
$expectedGames=switch($candidate.version){'1.4.20'{24}'1.4.18'{24}default{8}}
if($userSelectedV4){
 if($Bank -ne 'bank-3dc65253a610191d66a5' -or $candidate.selectedBank -ne $Bank -or $current.bank -ne $Bank -or $current.format -ne 'masked-v4-bank/1.0.0' -or
  $screen.format -ne 'user-directed-masked-v4/1.0.0' -or -not $screen.passed -or $screen.bank -ne $Bank -or $screen.generation -ne 'v4' -or
  -not $screen.fixes.productive_progress -or -not $screen.fixes.opening_candidate_bias -or $screen.strength_promoted -or $screen.fresh_strength_games -ne 0 -or $screen.seconds -ne 1 -or $screen.search_workers -ne 4){throw 'Exact user-selected v4 bank/progress/clock contract is required'}
}elseif($current.bank -ne $Bank -or -not $screen.passed -or $screen.new_bank -ne $Bank -or $screen.games -ne $expectedGames -or @($screen.failures).Count -or $screen.seconds -ne 1 -or $screen.search_workers -ne 4){throw 'Latest bank smoke evidence is incomplete'}
if((Get-FileHash -LiteralPath (Join-Path $root $current.qualification)).Hash -ne $current.qualification_sha256 -or (Get-FileHash -LiteralPath (Join-Path $root ('training/opponents/bank/'+$Bank+'/manifest.json'))).Hash -ne $current.manifest_sha256){throw 'Bank receipt changed'}
$parity=@()
foreach($pin in $candidate.sourcePins|Where-Object path -like 'src/BrassMaskedSearch/*.cs'){
 if($bankManifest.payload.source_files.PSObject.Properties[$pin.path].Value -ne $pin.sha256){throw 'Game/research search source differs'};$parity+=$pin.path
}
foreach($file in 'Evaluator.cs','Conversion.cs','CashManagement.cs','Tactics.cs'){
 $path='src/BrassLegacyAI/'+$file;$gamePath='src/BrassLocalAI/Policy/'+$file
 $gamePin=@($candidate.sourcePins|Where-Object path -eq $gamePath)
 if($gamePin.Count -ne 1 -or $bankManifest.payload.source_files.PSObject.Properties[$path].Value -ne $gamePin[0].sha256){throw 'Game/research evaluation source differs'};$parity+=$gamePath
}
$retained=[ordered]@{passed=$true;release=$candidate.baseVersion;validationSha256=(Get-FileHash -LiteralPath (Join-Path $archive 'validation.json')).Hash;testedFiles=$unchanged;
 scope='Qualified installed UI/rules retained only for byte-identical binaries; the changed AI has fresh checks.'}
if($candidate.version -eq '1.4.18' -and (-not $screen.promote -or $reports['industry-runtime'].checks -lt 16 -or $reports['save-1.4.17'].checks -lt 252)){throw 'Both controls, industry timing and current saved profiles must pass'}
if($userSelectedV4){
 if($reports['economy-runtime'].checks -lt 37 -or $reports['industry-runtime'].checks -lt 17 -or $reports['save-1.4.18'].checks -lt 252){throw 'V4 progress, labels and saved-profile correctness must pass'}
}elseif($candidate.version -eq '1.4.20'){
 if($screen.format -ne 'masked-economy-smoke/1.3.0'){throw 'Previous/economic comparisons with all three metric gates are required'}
 if(-not $screen.promote -or $reports['economy-runtime'].checks -lt 46 -or $reports['industry-runtime'].checks -lt 17 -or $reports['save-1.4.18'].checks -lt 252){throw 'Economy, six strategies and prior installed profiles must pass'}
 $controls=@('bank-3dc65253a610191d66a5','bank-f5107c644af2fbf4ac99')
 if(@($screen.comparisons.PSObject.Properties).Count -ne $controls.Count){throw 'Both previous and economic controls are required'}
 foreach($control in $controls){
  $comparison=$screen.comparisons.PSObject.Properties[$control].Value
  if(-not $comparison -or $comparison.games -ne 12 -or -not $comparison.eligible -or -not $comparison.more_wins -or $comparison.policies.new.win_share -le $comparison.policies.old.win_share){throw 'New styles must win more against every control'}
  if(-not $comparison.all_metrics_improved -or $comparison.policies.new.first_places -le $comparison.policies.old.first_places -or $comparison.policies.new.mean_vp -le $comparison.policies.old.mean_vp -or $comparison.policies.new.mean_rank -ge $comparison.policies.old.mean_rank){throw 'Each control must improve on first places, VP AND rank'}
 }
}
$qualification=[ordered]@{passed=$true;qualifiedUtc=[DateTime]::UtcNow.ToString('o');files=$candidate.files;sourcePins=$candidate.sourcePins;masked=$masked;era=$era;industry=$reports['industry-runtime'];economy=$reports['economy-runtime'];workers=$gameWorkers;labWorkers=4;
 resourceTests=$native.resourceTests;resourceMeasurements=$native.resourceMeasurements;executionStatistics=$native.executionStatistics;
 evidence=@($reports.Keys|Sort-Object|ForEach-Object{[ordered]@{name=$_;sha256=(Get-FileHash -LiteralPath (Join-Path $build ($_+'.json'))).Hash}});
 sourceParity=[ordered]@{passed=$true;bank=$Bank;paths=$parity};retainedEvidence=$retained;
 scope='All six v3 styles, independent game/lab workers, saved profiles, clocks, legal conversion/cash, cancellation and backend restoration. Strength is measured separately against the pinned controls.'}
if($userSelectedV4){$qualification.generation='v4';$qualification.selectionBasis=$candidate.selectionBasis;$qualification.conversionCoverage=$conversionCoverage;$qualification.scope='Six v4 styles; selected previous strategy, productive intent progress and opening-biased candidates. Correctness, saves, clocks, cancellation and resources; no new strength tests.'}
Write-BrassJsonAtomic (Join-Path $build 'local-ai-qualification.json') $qualification
$smoke=if($userSelectedV4){[ordered]@{passed=$true;games=0;promote=$false;selectionBasis=$candidate.selectionBasis;generation='v4';report=$current.qualification;sha256=$current.qualification_sha256;bank=$Bank;manifestSha256=$current.manifest_sha256;freshStrengthGames=0}}else{[ordered]@{passed=$true;games=$screen.games;promote=$screen.promote;comparisons=$screen.comparisons;report=$current.qualification;sha256=$current.qualification_sha256;winner=$screen.winner;policies=$screen.policies;bank=$Bank}}
$installationBasis=if($userSelectedV4){'Explicit user request to stop strength testing and install/use the previous advanced strategy as v4, retaining productive-development and opening-candidate fixes.'}elseif($candidate.version -eq '1.4.20'){'More first places, higher VP AND better rank against EACH previous/economic revision; fresh correctness and saved-profile checks'}else{'User-directed any-metric improvement against each control; no errors; fresh correctness and saved-profile checks'}
Write-BrassJsonAtomic (Join-Path $build 'validation.json') ([ordered]@{passed=$true;validatedUtc=[DateTime]::UtcNow.ToString('o');files=$candidate.files;localAi=$qualification;retainedUi=$retained;smoke=$smoke;installationBasis=$installationBasis})
Assert-BrassCandidate $candidate (Read-BrassJson (Join-Path $build 'validation.json')) $build
Write-Output ('Qualified era-aware '+$candidate.version+': '+$gameWorkers+' game workers/four lab workers, shared search/evaluation sources, fresh correctness evidence and unchanged UI/rules.')



