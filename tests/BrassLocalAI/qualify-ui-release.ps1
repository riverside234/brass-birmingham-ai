$ErrorActionPreference='Stop'
Set-StrictMode -Version Latest
$root=Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$build=Join-Path $root 'artifacts/patch-build'
$candidate=Get-Content -LiteralPath (Join-Path $build 'build-manifest.json') -Raw|ConvertFrom-Json
if($candidate.version -ne '1.4.4'){throw 'This qualification covers the 1.4.4 style picker and per-player difficulty settings'}
function Assert-Files($files){
 foreach($file in $candidate.files){
  if(@($files|Where-Object {$_.name -eq $file.name -and $_.sha256 -eq $file.sha256}).Count -ne 1){throw 'Evidence belongs to another candidate'}
  if((Get-FileHash -LiteralPath (Join-Path $build ('Managed/'+$file.name))).Hash -ne $file.sha256){throw 'Candidate changed'}
 }
}
$validation=Get-Content -LiteralPath (Join-Path $build 'validation.json') -Raw|ConvertFrom-Json
if(-not $validation.passed -or $validation.suites.Count -ne 12){throw 'Fresh managed rules and save regression suites are required'}
Assert-Files $validation.files
$names=@('local-ai-runtime-0','local-ai-native','unity-isolated-result','presentation-lifecycle','local-ai-live','local-ai-v2-save-compatibility','local-ai-v3-save-compatibility','local-ai-1.4.3-save-compatibility')
$evidence=@{}
foreach($name in $names){
 $item=Get-Content -LiteralPath (Join-Path $build ($name+'.json')) -Raw|ConvertFrom-Json
 if(-not $item.passed){throw ('Failed evidence: '+$name)};Assert-Files $item.testedFiles;$evidence[$name]=$item
}
$runtime=$evidence['local-ai-runtime-0']
if(-not $runtime.saveTestsRun -or $runtime.checks -lt 2339){throw 'Difficulty, S1 and save validation incomplete'}
foreach($tier in 'easy','mid','hard','exhausted'){
 $account=@($runtime.budgetAccounting|Where-Object difficulty -eq $tier)
 if($account.Count -ne 1 -or $account[0].work_limit -ne 240 -or $account[0].reply_work -ne 2048 -or $account[0].nodes -le 0){throw 'Timed search accounting or exhausted-budget legal completion missing'}
}
$native=$evidence['local-ai-native']
if($native.mode -ne 'full' -or $native.turns.Count -ne 108 -or @($native.nativeErrors).Count){throw 'Native difficulty matrix incomplete'}
$keys=@('balanced','rail-tempo','cotton','goods-beer','supplier','pottery-flex')
$tiers=@{easy=2000;mid=5000;hard=12000}
foreach($count in 2,3,4){foreach($intro in $false,$true){foreach($key in $keys){foreach($tier in $tiers.Keys){
 $turn=@($native.turns|Where-Object {$_.count -eq $count -and $_.intro -eq $intro -and $_.key -eq $key -and $_.difficulty -eq $tier})
 if($turn.Count -ne 1 -or -not $turn[0].inputGuardTested -or $turn[0].thinkingMilliseconds -ne $tiers[$tier] -or $turn[0].workLimit -ne 240){throw 'Per-player native time/node/input coverage missing'}
}}}}
$replan=@($native.turns|Where-Object {$_.count -eq 2 -and -not $_.intro -and $_.key -eq 'balanced' -and $_.difficulty -eq 'mid'})
if($replan.Count -ne 1 -or -not $replan[0].replanInjected -or $replan[0].planningRequests -ne 2 -or $replan[0].lastRequestMilliseconds -gt 5000){throw 'Shared-deadline stale-continuation test missing'}
foreach($image in @('local-ai-custom-menu.png','local-ai-resume-menu.png','local-ai-style-picker.png','local-ai-expanded-picker.png','local-ai-picker-1280.png','local-ai-picker-1920.png','local-ai-picker-2560.png','local-ai-resume-picker.png','local-ai-resume-1280.png','local-ai-resume-picker-1280.png')){
 if(-not @($native.images|Where-Object {$_ -like ('*'+$image) -and (Test-Path -LiteralPath $_)}).Count){throw ('Native menu render missing: '+$image)}
}
$unity=$evidence['unity-isolated-result']
if($unity.suites.Count -ne 10 -or $unity.cityLocatorChecks -lt 46000 -or @($unity.nativeErrors).Count){throw 'Existing native rules/UI regressions incomplete'}
$presentation=$evidence['presentation-lifecycle']
if($presentation.checks -lt 184 -or $presentation.failures -ne 0 -or @($presentation.nativeErrors).Count){throw 'Presentation regression failed'}
$live=$evidence['local-ai-live']
if($live.matches.Count -ne 6){throw 'Complete save/reload games missing'}
foreach($count in 2,3,4){foreach($intro in $false,$true){
 $match=@($live.matches|Where-Object {$_.count -eq $count -and $_.intro -eq $intro})
 if($match.Count -ne 1 -or $match[0].actions -le 0 -or $match[0].reloads -le 0){throw 'Save/reload lifecycle coverage missing'}
}}
$compatibility=@{'local-ai-v2-save-compatibility'='1.3.2';'local-ai-v3-save-compatibility'='1.4.0';'local-ai-1.4.3-save-compatibility'='1.4.3'}
foreach($name in $compatibility.Keys){
 $item=$evidence[$name]
 if($item.checks -lt 54 -or (($item.games|ForEach-Object strategy|Sort-Object)-join '|') -ne (($keys|Sort-Object)-join '|')){throw 'Older save coverage incomplete'}
 $archive=Join-Path $root ('artifacts/patch-releases/'+$compatibility[$name]);$producer=Get-Content -LiteralPath (Join-Path $archive 'build-manifest.json') -Raw|ConvertFrom-Json
 foreach($file in $producer.files){
  if(@($item.producerFiles|Where-Object {$_.name -eq $file.name -and $_.sha256 -eq $file.sha256}).Count -ne 1 -or (Get-FileHash -LiteralPath (Join-Path $archive $file.name)).Hash -ne $file.sha256){throw 'Unverified save producer'}
 }
}
$pins=@();$prior=Join-Path $root 'artifacts/patch-releases/1.4.3/BrassLocalAI'
foreach($file in Get-ChildItem -LiteralPath (Join-Path $root 'src/BrassLocalAI') -Recurse -Filter '*.cs'){
 $relative=[IO.Path]::GetRelativePath((Join-Path $root 'src/BrassLocalAI'),$file.FullName).Replace('\','/');$hash=(Get-FileHash -LiteralPath $file.FullName).Hash
 if(($relative.StartsWith('Host/') -or $relative.StartsWith('Policy/')) -and $relative -notin @('Policy/LocalGameState.cs','Policy/DifficultyCatalog.cs')){
  if((Get-FileHash -LiteralPath (Join-Path $prior $relative)).Hash -ne $hash){throw ('Search policy or host changed unexpectedly: '+$relative)}
 }
 $pins+=[ordered]@{path=('src/BrassLocalAI/'+$relative);sha256=$hash}
}
& (Join-Path $PSScriptRoot 'verify-h5-provenance.ps1')
$provenance=Get-Content -LiteralPath (Join-Path $build 'local-ai-h5.json') -Raw|ConvertFrom-Json
if(-not $provenance.passed -or $provenance.v3Profiles.Count -ne 6 -or -not $provenance.v3Bank){throw 'Research bank provenance failed'}
$qualification=[ordered]@{passed=$true;qualifiedUtc=[DateTime]::UtcNow.ToString('o');files=$candidate.files;sourcePins=$pins;h5Provenance=$provenance;
 evidence=@($names|ForEach-Object {[ordered]@{name=$_;sha256=(Get-FileHash -LiteralPath (Join-Path $build ($_+'.json'))).Hash}});
 scope='Fresh rules and native UI regressions; direct style picker including 16-style fixture; 108 native S1 configurations at 2/5/12 seconds; versioned execution assignments and genuine archived save producers. Search policy unchanged; no new strength claim. No personal save opened.'}
$qualification|ConvertTo-Json -Depth 20|Set-Content -LiteralPath (Join-Path $build 'local-ai-qualification.json') -Encoding UTF8
$validation|Add-Member -NotePropertyName localAi -NotePropertyValue $qualification -Force
$unity|Add-Member -NotePropertyName retainedFrom -NotePropertyValue $null -Force
$unity|Add-Member -NotePropertyName scope -NotePropertyValue 'Fresh native regression suite against all three 1.4.4 candidate DLLs.' -Force
$validation.unityPlayerValidation=$unity
$validation|ConvertTo-Json -Depth 22|Set-Content -LiteralPath (Join-Path $build 'validation.json') -Encoding UTF8
Write-Output 'Qualified 1.4.4: direct style picker, per-player difficulty, native execution and save compatibility.'
