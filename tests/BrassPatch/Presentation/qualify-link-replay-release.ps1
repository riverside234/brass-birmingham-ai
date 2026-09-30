$ErrorActionPreference='Stop'
Set-StrictMode -Version Latest
$root=Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
$build=Join-Path $root 'artifacts/patch-build'
$archive=Join-Path $root 'artifacts/patch-releases/1.4.5'
$candidate=Get-Content -LiteralPath (Join-Path $build 'build-manifest.json') -Raw|ConvertFrom-Json
$previous=Get-Content -LiteralPath (Join-Path $archive 'build-manifest.json') -Raw|ConvertFrom-Json
if($candidate.version -ne '1.4.6'){throw 'This qualifier covers the 1.4.6 link replay lifecycle repair'}
function Assert-Files($files,$expected=$candidate.files){
 foreach($f in $expected){if(@($files|Where-Object {$_.name -eq $f.name -and $_.sha256 -eq $f.sha256}).Count -ne 1){throw 'Evidence fingerprint mismatch'}}
}
foreach($f in $candidate.files){if((Get-FileHash -LiteralPath (Join-Path $build ('Managed/'+$f.name))).Hash -ne $f.sha256){throw 'Candidate changed'}}
Assert-Files $candidate.files @($previous.files|Where-Object name -eq 'BrassLegacyAI.dll')
$inventory=Get-Content -LiteralPath (Join-Path $archive 'archive-manifest.json') -Raw|ConvertFrom-Json
foreach($entry in $inventory.files){if((Get-FileHash -LiteralPath (Join-Path $archive $entry.path)).Hash -ne $entry.sha256){throw ('Prior archive changed: '+$entry.path)}}
$pins=@()
foreach($name in 'BrassLocalAI','BrassPatch'){
 $current=Join-Path $root ('src/'+$name);$prior=Join-Path $archive $name
 $oldFiles=@(Get-ChildItem -LiteralPath $prior -Recurse -File|Where-Object Extension -in '.cs','.json')
 $newFiles=@(Get-ChildItem -LiteralPath $current -Recurse -File|Where-Object Extension -in '.cs','.json')
 if($newFiles.Count -ne $oldFiles.Count + $(if($name -eq 'BrassPatch'){1}else{0})){throw 'Unexpected runtime source set change'}
 foreach($file in $newFiles){
  $relative=[IO.Path]::GetRelativePath($current,$file.FullName).Replace('\','/');$old=Join-Path $prior $relative
  $hash=(Get-FileHash -LiteralPath $file.FullName).Hash
  if($name -ne 'BrassPatch' -or $relative -notin @('Presentation/LinkReplay.cs','Presentation/manifest.json')){
   if(-not (Test-Path -LiteralPath $old) -or (Get-FileHash -LiteralPath $old).Hash -ne $hash){throw ('Out-of-scope runtime change: '+$name+'/'+$relative)}
  }
  $pins+=[ordered]@{path=('src/'+$name+'/'+$relative);sha256=$hash}
 }
}
$oldHooks=Get-Content -LiteralPath (Join-Path $archive 'BrassPatch/Presentation/manifest.json') -Raw|ConvertFrom-Json
$newHooks=Get-Content -LiteralPath (Join-Path $root 'src/BrassPatch/Presentation/manifest.json') -Raw|ConvertFrom-Json
foreach($hook in $oldHooks){if(@($newHooks|Where-Object {($_|ConvertTo-Json -Depth 8 -Compress) -eq ($hook|ConvertTo-Json -Depth 8 -Compress)}).Count -ne 1){throw 'Existing presentation hook changed'}}
$added=@($newHooks|Where-Object helperType -eq 'BrassPatch.Presentation.LinkReplay')
if($added.Count -ne 7 -or $newHooks.Count -ne $oldHooks.Count+7){throw 'Expected seven scoped link lifecycle hooks'}
$woven=Get-Content -LiteralPath (Join-Path $build 'Managed/Assembly-CSharp.dll.manifest.json') -Raw|ConvertFrom-Json
$priorWoven=Get-Content -LiteralPath (Join-Path $archive 'Assembly-CSharp.dll.manifest.json') -Raw|ConvertFrom-Json
if(@($priorWoven.methods|Where-Object {$_ -notin $woven.methods}).Count -or $woven.methods.Count -ne $priorWoven.methods.Count+7 -or $woven.gameOwnedBonusSites -ne $priorWoven.gameOwnedBonusSites){throw 'Prior hook retention failed'}
$validation=Get-Content -LiteralPath (Join-Path $build 'validation.json') -Raw|ConvertFrom-Json
if(-not $validation.passed -or $validation.suites.Count -ne 12){throw 'Fresh managed regression suites required'}
Assert-Files $validation.files
$reports=@{}
foreach($name in 'link-replay','presentation-lifecycle','unity-isolated-result','local-ai-menu','local-ai-runtime-0'){
 $item=Get-Content -LiteralPath (Join-Path $build ($name+'.json')) -Raw|ConvertFrom-Json
 if(-not $item.passed){throw ('Failed evidence: '+$name)};Assert-Files $item.testedFiles;$reports[$name]=$item
 if($name -ne 'local-ai-runtime-0' -and @($item.nativeErrors).Count){throw 'Unexpected native errors'}
}
if($reports['link-replay'].checks -lt 765 -or $reports['link-replay'].failures){throw 'Link replay coverage incomplete'}
if($reports['presentation-lifecycle'].checks -lt 184 -or $reports['presentation-lifecycle'].failures){throw 'Presentation regressions incomplete'}
if($reports['unity-isolated-result'].suites.Count -ne 10 -or $reports['unity-isolated-result'].cityLocatorChecks -lt 46000){throw 'Existing native regression coverage incomplete'}
if($reports['local-ai-menu'].mode -ne 'startup-menu-smoke' -or $reports['local-ai-menu'].checks -lt 239){throw 'First-open and picker regression coverage incomplete'}
if($reports['local-ai-runtime-0'].checks -lt 2339 -or -not $reports['local-ai-runtime-0'].saveTestsRun){throw 'Fresh S1/runtime/save regression coverage incomplete'}
$baseline=Get-Content -LiteralPath (Join-Path $build 'link-replay-baseline.json') -Raw|ConvertFrom-Json
Assert-Files $baseline.testedFiles $previous.files
if($baseline.passed -or $baseline.error -or $baseline.checks -ne 18 -or $baseline.failures -ne 12){throw 'Native regression must reproduce on 1.4.5'}
foreach($name in 'normal placed-link replay completes once','link Play is queued before the cancellation test','scoring still raises the link token above the board'){
 if(@($baseline.cases|Where-Object {$_.name -eq $name -and $_.passed}).Count -ne 2){throw 'Baseline positive control failed'}
}
foreach($name in 'killed placed-link replay returns control once','queued Play cannot resume an intentionally cancelled link replay','repeated link highlights own exactly one pulse','clearing highlights leaves no orphan pulse','interrupted scoring restores link token and VP label parents and geometry','rail placement after interrupted Canal scoring returns control'){
 if(@($baseline.cases|Where-Object {$_.name -eq $name -and -not $_.passed}).Count -ne 2){throw 'Expected interruption/scoring/highlight regression missing'}
}
& (Join-Path $root 'tests/BrassLocalAI/verify-h5-provenance.ps1')
$h5=Get-Content -LiteralPath (Join-Path $build 'local-ai-h5.json') -Raw|ConvertFrom-Json
if(-not $h5.passed -or $h5.v3Profiles.Count -ne 6){throw 'Frozen research provenance failed'}
$retained=[ordered]@{release='1.4.5';aiDllSha256=(@($previous.files|Where-Object name -eq 'BrassLegacyAI.dll')[0].sha256);
 validationSha256=(Get-FileHash -LiteralPath (Join-Path $archive 'validation.json')).Hash;
 scope='AI binary and all existing runtime sources unchanged. Prior 108-turn strategy/difficulty and archived save compatibility evidence retained; new link replay, rules, presentation, menu and S1/save runtime tests rerun against all three candidate DLLs.'}
$qualification=[ordered]@{passed=$true;qualifiedUtc=[DateTime]::UtcNow.ToString('o');files=$candidate.files;sourcePins=$pins;addedHooks=$added;h5Provenance=$h5;retainedEvidence=$retained;
 evidence=@($reports.Keys|Sort-Object|ForEach-Object {[ordered]@{name=$_;sha256=(Get-FileHash -LiteralPath (Join-Path $build ($_+'.json'))).Hash}});
 baselineSha256=(Get-FileHash -LiteralPath (Join-Path $build 'link-replay-baseline.json')).Hash;
 testSourceSha256=(Get-FileHash -LiteralPath (Join-Path $PSScriptRoot 'LinkReplayTests.cs')).Hash;
 scope='Seven presentation hooks: owned link placement and highlight/scoring lifetimes, exactly-once interrupted link replay recovery, and stale/intentional-cancel protection. No rule, AI, input, save-format or training-policy changes.'}
$qualification|ConvertTo-Json -Depth 20|Set-Content -LiteralPath (Join-Path $build 'local-ai-qualification.json') -Encoding UTF8
$unity=$reports['unity-isolated-result']
$unity|Add-Member -NotePropertyName retainedFrom -NotePropertyValue $null -Force
$unity|Add-Member -NotePropertyName scope -NotePropertyValue 'Fresh native regression suite against all three 1.4.6 candidate DLLs.' -Force
$validation|Add-Member -NotePropertyName localAi -NotePropertyValue $qualification -Force
$validation|Add-Member -NotePropertyName retainedEvidence -NotePropertyValue $retained -Force
$validation.unityPlayerValidation=$unity
$validation|ConvertTo-Json -Depth 22|Set-Content -LiteralPath (Join-Path $build 'validation.json') -Encoding UTF8
Write-Output 'Qualified 1.4.6: link/scoring interruption reproduced and repaired; fresh rules, presentation, menu, S1 and save regressions passed.'
