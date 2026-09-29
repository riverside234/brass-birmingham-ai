$ErrorActionPreference='Stop'
Set-StrictMode -Version Latest
$root=Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$build=Join-Path $root 'artifacts/patch-build'
$candidate=Get-Content -LiteralPath (Join-Path $build 'build-manifest.json') -Raw|ConvertFrom-Json
if($candidate.version -ne '1.4.2'){throw 'This qualification covers presentation, save/continue, and the 15-second allowance in 1.4.2'}
$validation=Get-Content -LiteralPath (Join-Path $build 'validation.json') -Raw|ConvertFrom-Json
function Assert-Files($files){
 foreach($file in $candidate.files){
  if(@($files|Where-Object {$_.name -eq $file.name -and $_.sha256 -eq $file.sha256}).Count -ne 1){throw 'Evidence belongs to another build'}
  if((Get-FileHash -LiteralPath (Join-Path $build ('Managed/'+$file.name))).Hash -ne $file.sha256){throw 'Candidate changed'}
 }
}
if(-not $validation.passed -or $validation.suites.Count -ne 12){throw 'Fresh managed rules/save qualification is required'}
Assert-Files $validation.files
$names=@('presentation-lifecycle','local-ai-runtime-0','local-ai-native','unity-isolated-result','local-ai-live','local-ai-v2-save-compatibility','local-ai-v3-save-compatibility','local-ai-1.4.1-save-compatibility')
$evidence=@{}
foreach($name in $names){
 $item=Get-Content -LiteralPath (Join-Path $build ($name+'.json')) -Raw|ConvertFrom-Json
 if(-not $item.passed){throw ('Failed evidence: '+$name)};Assert-Files $item.testedFiles;$evidence[$name]=$item
}
$presentation=$evidence['presentation-lifecycle']
if($presentation.checks -lt 184 -or $presentation.failures -ne 0 -or @($presentation.nativeErrors).Count){throw 'Presentation lifecycle coverage failed'}
$runtime=$evidence['local-ai-runtime-0']
if(-not $runtime.saveTestsRun -or $runtime.checks -lt 1030){throw 'Runtime/S1/save checks incomplete'}
$unity=$evidence['unity-isolated-result']
if($unity.suites.Count -ne 10 -or $unity.liquidationChecks -lt 57 -or $unity.cityLocatorChecks -lt 46000 -or @($unity.nativeErrors).Count){throw 'Fresh existing Unity suite coverage incomplete'}
$keys=@('balanced','rail-tempo','cotton','goods-beer','supplier','pottery-flex')
$native=$evidence['local-ai-native']
if($native.mode -ne 'full' -or $native.turns.Count -ne 36 -or @($native.nativeErrors).Count){throw 'Native turn matrix incomplete'}
foreach($n in 2,3,4){foreach($intro in $false,$true){foreach($key in $keys){
 $turn=@($native.turns|Where-Object {$_.count -eq $n -and $_.intro -eq $intro -and $_.key -eq $key})
 if($turn.Count -ne 1 -or -not $turn[0].inputGuardTested -or $turn[0].thinkingMilliseconds -ne 15000 -or $turn[0].workLimit -lt 1024){throw 'Native strategy input/timing contract missing'}
}}}
$live=$evidence['local-ai-live']
if($live.matches.Count -ne 6){throw 'Complete save/reload games missing'}
foreach($n in 2,3,4){foreach($intro in $false,$true){
 $match=@($live.matches|Where-Object {$_.count -eq $n -and $_.intro -eq $intro})
 if($match.Count -ne 1 -or $match[0].actions -le 0 -or $match[0].reloads -le 0){throw 'Save/reload lifecycle coverage missing'}
}}
$compatibility=@{'local-ai-v2-save-compatibility'='1.3.2';'local-ai-v3-save-compatibility'='1.4.0';'local-ai-1.4.1-save-compatibility'='1.4.1'}
foreach($name in $compatibility.Keys){
 $item=$evidence[$name];if($item.checks -lt 48 -or (($item.games|ForEach-Object strategy|Sort-Object)-join '|') -ne (($keys|Sort-Object)-join '|')){throw 'Older save coverage incomplete'}
 $archive=Join-Path $root ('artifacts/patch-releases/'+$compatibility[$name]);$producer=Get-Content -LiteralPath (Join-Path $archive 'build-manifest.json') -Raw|ConvertFrom-Json
 foreach($file in $producer.files){
  if(@($item.producerFiles|Where-Object {$_.name -eq $file.name -and $_.sha256 -eq $file.sha256}).Count -ne 1 -or (Get-FileHash -LiteralPath (Join-Path $archive $file.name)).Hash -ne $file.sha256){throw 'Compatibility fixtures have an unverified producer'}
 }
}
# Preserve the accepted research policy exactly. This release changes execution
# time and presentation/persistence, and does not claim a new strength study.
$pins=@()
foreach($area in 'Host','Policy'){
 $current=Join-Path $root ('src/BrassLocalAI/'+$area);$prior=Join-Path $root ('artifacts/patch-releases/1.4.1/BrassLocalAI/'+$area)
 $files=@(Get-ChildItem -LiteralPath $current -Recurse -Filter '*.cs')
 if($files.Count -ne @(Get-ChildItem -LiteralPath $prior -Recurse -Filter '*.cs').Count){throw 'Policy/host source set changed'}
 foreach($file in $files){
  $relative=[IO.Path]::GetRelativePath($current,$file.FullName);$hash=(Get-FileHash -LiteralPath $file.FullName).Hash
  if((Get-FileHash -LiteralPath (Join-Path $prior $relative)).Hash -ne $hash){throw ('Policy or host changed: '+$relative)}
  $pins+=[ordered]@{path=('src/BrassLocalAI/'+$area+'/'+$relative.Replace('\','/'));sha256=$hash}
 }
}
& (Join-Path $PSScriptRoot 'verify-h5-provenance.ps1')
$provenance=Get-Content -LiteralPath (Join-Path $build 'local-ai-h5.json') -Raw|ConvertFrom-Json
if(-not $provenance.passed -or $provenance.v3Profiles.Count -ne 6 -or -not $provenance.v3Bank){throw 'Research/profile provenance failed'}
$qualification=[ordered]@{passed=$true;qualifiedUtc=[DateTime]::UtcNow.ToString('o');files=$candidate.files;policySourcesUnchangedFrom='1.4.1';sourcePins=$pins;h5Provenance=$provenance;
 evidence=@($names|ForEach-Object {[ordered]@{name=$_;sha256=(Get-FileHash -LiteralPath (Join-Path $build ($_+'.json'))).Hash}});
 scope='Fresh managed rules and native UI suites; all 36 strategy/configuration turns at a 15-second allowance; six complete generated-save games; all six profiles from three archived save producers. No new AI strength claim and no user save opened.'}
$qualification|ConvertTo-Json -Depth 20|Set-Content -LiteralPath (Join-Path $build 'local-ai-qualification.json') -Encoding UTF8
$validation|Add-Member -NotePropertyName localAi -NotePropertyValue $qualification -Force
$unity|Add-Member -NotePropertyName retainedFrom -NotePropertyValue $null -Force
$unity|Add-Member -NotePropertyName scope -NotePropertyValue 'Fresh run against all three 1.4.2 candidate DLLs.' -Force
$validation.unityPlayerValidation=$unity
$validation|ConvertTo-Json -Depth 22|Set-Content -LiteralPath (Join-Path $build 'validation.json') -Encoding UTF8
Write-Output 'Qualified 1.4.2: presentation/save repairs, fresh rules checks, and 15-second native AI turns.'
