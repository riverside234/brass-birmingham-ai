$ErrorActionPreference='Stop'
Set-StrictMode -Version Latest
$root=Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$build=Join-Path $root 'artifacts/patch-build'
$archive=Join-Path $root 'artifacts/patch-releases/1.4.2'
$candidate=Get-Content -LiteralPath (Join-Path $build 'build-manifest.json') -Raw|ConvertFrom-Json
if($candidate.version -ne '1.4.3'){throw 'This qualifier covers the 1.4.3 live search-budget adjustment'}
$previous=Get-Content -LiteralPath (Join-Path $archive 'build-manifest.json') -Raw|ConvertFrom-Json
$old=Get-Content -LiteralPath (Join-Path $archive 'validation.json') -Raw|ConvertFrom-Json
if(-not $old.passed -or -not $old.localAi.passed -or -not $old.unityPlayerValidation.passed){throw 'Qualified 1.4.2 evidence is required'}
function Assert-Files($files){
 foreach($file in $candidate.files){
  if(@($files|Where-Object {$_.name -eq $file.name -and $_.sha256 -eq $file.sha256}).Count -ne 1){throw 'Evidence has the wrong candidate fingerprint'}
  if((Get-FileHash -LiteralPath (Join-Path $build ('Managed/'+$file.name))).Hash -ne $file.sha256){throw 'Candidate changed'}
 }
}
foreach($file in $previous.files){
 if((Get-FileHash -LiteralPath (Join-Path $archive $file.name)).Hash -ne $file.sha256 -or @($old.files|Where-Object {$_.name -eq $file.name -and $_.sha256 -eq $file.sha256}).Count -ne 1){throw 'Archive or previous qualification mismatch'}
 if($file.name -ne 'BrassLegacyAI.dll' -and @($candidate.files|Where-Object {$_.name -eq $file.name -and $_.sha256 -eq $file.sha256}).Count -ne 1){throw 'Changed rules/helper binaries require fresh rules/UI qualification'}
}
function Body([string]$text){[regex]::Replace([regex]::Replace($text,'(?m)^\s*//[^\r\n]*',''),'\s+','')}
$sourceRoot=Join-Path $root 'src/BrassLocalAI';$priorRoot=Join-Path $archive 'BrassLocalAI';$sourcePins=@();$changed=@()
$currentSources=@(Get-ChildItem -LiteralPath $sourceRoot -Recurse -Filter '*.cs')
if($currentSources.Count -ne @(Get-ChildItem -LiteralPath $priorRoot -Recurse -Filter '*.cs').Count){throw 'Unexpected source set change'}
foreach($file in $currentSources){
 $relative=[IO.Path]::GetRelativePath($sourceRoot,$file.FullName).Replace('\','/');$prior=Join-Path $priorRoot $relative;$hash=(Get-FileHash -LiteralPath $file.FullName).Hash
 if((Get-FileHash -LiteralPath $prior).Hash -ne $hash){
  if($relative -ne 'Unity/LocalTurnBridge.cs'){throw ('Out-of-scope AI source change: '+$relative)}
  $expected=[IO.File]::ReadAllText($prior).Replace('ThinkingMilliseconds = 15000','ThinkingMilliseconds = 10000').Replace('SearchWorkLimit = 1024','SearchWorkLimit = 240').Replace('Math.Max(ThinkingMilliseconds, bridge.profile.milliseconds)','ThinkingMilliseconds').Replace('Math.Max(SearchWorkLimit, bridge.profile.work_limit)','SearchWorkLimit')
  if((Body $expected) -cne (Body ([IO.File]::ReadAllText($file.FullName)))){throw 'AI bridge changed beyond the requested budget settings'}
  $changed+=$relative
 }
 $sourcePins+=[ordered]@{path=('src/BrassLocalAI/'+$relative);sha256=$hash}
}
if($changed.Count -ne 1){throw 'Expected one budget-only source change'}
& (Join-Path $PSScriptRoot 'verify-h5-provenance.ps1')
$h5=Get-Content -LiteralPath (Join-Path $build 'local-ai-h5.json') -Raw|ConvertFrom-Json
if(-not $h5.passed -or $h5.v3Profiles.Count -ne 6){throw 'Profile provenance failed'}
$bankRoot=Join-Path $root ('training/opponents/bank/'+$h5.v3Bank)
$keys=@('balanced','rail-tempo','cotton','goods-beer','supplier','pottery-flex')
foreach($key in $keys){
 $profile=Get-Content -LiteralPath (Join-Path $bankRoot ('source/configs/legacy-ai/'+$key+'-v3.json')) -Raw|ConvertFrom-Json
 if($profile.work_limit -ne 240 -or $profile.reply_work -ne 2048){throw 'Research bank work limits differ from this release'}
}
$names=@('local-ai-runtime-0','local-ai-native');$evidence=@{}
foreach($name in $names){$item=Get-Content -LiteralPath (Join-Path $build ($name+'.json')) -Raw|ConvertFrom-Json;if(-not $item.passed){throw ('Failed evidence: '+$name)};Assert-Files $item.testedFiles;$evidence[$name]=$item}
$runtime=$evidence['local-ai-runtime-0'];if(-not $runtime.saveTestsRun -or $runtime.checks -lt 1030){throw 'Runtime/S1/save regressions incomplete'}
$native=$evidence['local-ai-native'];if($native.mode -ne 'full' -or $native.turns.Count -ne 36 -or @($native.nativeErrors).Count){throw 'Native turn matrix incomplete'}
foreach($count in 2,3,4){foreach($intro in $false,$true){foreach($key in $keys){
 $turn=@($native.turns|Where-Object {$_.count -eq $count -and $_.intro -eq $intro -and $_.key -eq $key})
 if($turn.Count -ne 1 -or -not $turn[0].inputGuardTested -or $turn[0].thinkingMilliseconds -ne 10000 -or $turn[0].workLimit -ne 240){throw 'Native search budget or input contract failed'}
}}}
$retained=[ordered]@{release='1.4.2';validationSha256=(Get-FileHash -LiteralPath (Join-Path $archive 'validation.json')).Hash;unchangedFiles=@($candidate.files|Where-Object name -ne 'BrassLegacyAI.dll');scope='Rules, presentation and persistence-helper evidence retained only for byte-identical engine/helper DLLs; policy/host sources unchanged.'}
$qualification=[ordered]@{passed=$true;qualifiedUtc=[DateTime]::UtcNow.ToString('o');files=$candidate.files;changedSources=$changed;sourcePins=$sourcePins;h5Provenance=$h5;retainedRulesEvidence=$retained;
 evidence=@($names|ForEach-Object {[ordered]@{name=$_;sha256=(Get-FileHash -LiteralPath (Join-Path $build ($_+'.json'))).Hash}});
 scope='Only live allowance changed: 10 seconds, 240 whole-turn nodes, research-bank reply budget preserved. Fresh 36 native turns plus runtime/save checks. No new strength claim.'}
$qualification|ConvertTo-Json -Depth 20|Set-Content -LiteralPath (Join-Path $build 'local-ai-qualification.json') -Encoding UTF8
$unity=$old.unityPlayerValidation;$unity.retainedFrom='1.4.2';$unity.scope=$retained.scope;$unity.testedFiles=$retained.unchangedFiles
[ordered]@{passed=$true;files=$candidate.files;localAi=$qualification;retainedRulesEvidence=$retained;unityPlayerValidation=$unity}|ConvertTo-Json -Depth 22|Set-Content -LiteralPath (Join-Path $build 'validation.json') -Encoding UTF8
Write-Output 'Qualified 1.4.3: 10 seconds and the research-bank 240-node limit; rules/UI helper binaries unchanged.'
