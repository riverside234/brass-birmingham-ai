$ErrorActionPreference='Stop'
Set-StrictMode -Version Latest
$root=Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$build=Join-Path $root 'artifacts/patch-build'
$archive=Join-Path $root 'artifacts/patch-releases/1.4.4'
$candidate=Get-Content -LiteralPath (Join-Path $build 'build-manifest.json') -Raw|ConvertFrom-Json
$previous=Get-Content -LiteralPath (Join-Path $archive 'build-manifest.json') -Raw|ConvertFrom-Json
$old=Get-Content -LiteralPath (Join-Path $archive 'validation.json') -Raw|ConvertFrom-Json
if($candidate.version -ne '1.4.5'){throw 'This qualifier covers only the 1.4.5 inactive-menu initialization fix'}
if(-not $old.passed -or -not $old.localAi.passed -or -not $old.unityPlayerValidation.passed){throw 'Qualified 1.4.4 evidence is required'}
$inventory=Get-Content -LiteralPath (Join-Path $archive 'archive-manifest.json') -Raw|ConvertFrom-Json
foreach($entry in $inventory.files){if((Get-FileHash -LiteralPath (Join-Path $archive $entry.path)).Hash -ne $entry.sha256){throw ('Prior archive changed: '+$entry.path)}}
function Assert-Files($files,$expected){
 foreach($file in $expected){if(@($files|Where-Object {$_.name -eq $file.name -and $_.sha256 -eq $file.sha256}).Count -ne 1){throw 'Evidence fingerprint mismatch'}}
}
Assert-Files $old.files $previous.files
foreach($file in $candidate.files){
 if((Get-FileHash -LiteralPath (Join-Path $build ('Managed/'+$file.name))).Hash -ne $file.sha256){throw 'Candidate changed'}
 if($file.name -ne 'BrassLegacyAI.dll'){Assert-Files $previous.files @($file)}
}
function Body([string]$value){[regex]::Replace([regex]::Replace($value,'(?m)^\s*//[^\r\n]*',''),'\s+','')}
$sourceRoot=Join-Path $root 'src/BrassLocalAI';$priorRoot=Join-Path $archive 'BrassLocalAI'
$sources=@(Get-ChildItem -LiteralPath $sourceRoot -Recurse -Filter '*.cs');$pins=@();$changed=@()
if($sources.Count -ne @(Get-ChildItem -LiteralPath $priorRoot -Recurse -Filter '*.cs').Count){throw 'Unexpected runtime source set change'}
foreach($file in $sources){
 $relative=[IO.Path]::GetRelativePath($sourceRoot,$file.FullName).Replace('\','/');$prior=Join-Path $priorRoot $relative
 $hash=(Get-FileHash -LiteralPath $file.FullName).Hash
 if($hash -ne (Get-FileHash -LiteralPath $prior).Hash){
  $expected=[IO.File]::ReadAllText($prior)
  switch($relative){
   'Unity/LocalMenu.cs' {$expected=$expected.Replace('if (view.GetComponentInParent<CreateOfflineGameView>() == null) return;','if (!view.GetComponentsInParent<CreateOfflineGameView>(true).Any()) return;')}
   'Unity/AiSetupRow.cs' {$expected=$expected.Replace('menu = view.GetComponentInParent<Frameworks.View.Menu.CreateGame.CreateOfflineGameView>();','menu = view.GetComponentsInParent<Frameworks.View.Menu.CreateGame.CreateOfflineGameView>(true).FirstOrDefault();')}
   default {throw ('Out-of-scope runtime change: '+$relative)}
  }
  if((Body $expected) -cne (Body ([IO.File]::ReadAllText($file.FullName)))){throw ('Change exceeds inactive-parent lookup: '+$relative)}
  $changed+=$relative
 }
 $pins+=[ordered]@{path=('src/BrassLocalAI/'+$relative);sha256=$hash}
}
if($changed.Count -ne 2){throw 'Expected the two inactive-parent lookup corrections'}
if((Get-FileHash -LiteralPath (Join-Path $sourceRoot 'Unity/manifest.json')).Hash -ne (Get-FileHash -LiteralPath (Join-Path $priorRoot 'Unity/manifest.json')).Hash){throw 'Hook allowlist changed'}
$baseline=Get-Content -LiteralPath (Join-Path $build 'local-ai-cold-menu-baseline.json') -Raw|ConvertFrom-Json
if($baseline.passed -or $baseline.error -notlike '*cold-menu AI controls bind before the view is enabled*'){throw 'The regression must fail on 1.4.4'}
Assert-Files $baseline.testedFiles $previous.files
$menu=Get-Content -LiteralPath (Join-Path $build 'local-ai-menu.json') -Raw|ConvertFrom-Json
if(-not $menu.passed -or $menu.mode -ne 'startup-menu-smoke' -or $menu.checks -lt 239 -or @($menu.nativeErrors).Count){throw 'Fresh cold-menu and existing menu regressions must pass'}
Assert-Files $menu.testedFiles $candidate.files
foreach($name in 'local-ai-cold-menu.png','local-ai-cold-resume-menu.png','local-ai-style-picker.png','local-ai-expanded-picker.png','local-ai-picker-1280.png','local-ai-picker-1920.png','local-ai-picker-2560.png','local-ai-resume-picker-1280.png'){
 if(-not @($menu.images|Where-Object {$_ -like ('*'+$name) -and (Test-Path -LiteralPath $_)}).Count){throw ('Menu render missing: '+$name)}
}
& (Join-Path $PSScriptRoot 'verify-h5-provenance.ps1')
$h5=Get-Content -LiteralPath (Join-Path $build 'local-ai-h5.json') -Raw|ConvertFrom-Json
if(-not $h5.passed -or $h5.v3Profiles.Count -ne 6){throw 'Frozen research provenance failed'}
$retained=[ordered]@{release='1.4.4';validationSha256=(Get-FileHash -LiteralPath (Join-Path $archive 'validation.json')).Hash;unchangedFiles=@($candidate.files|Where-Object name -ne 'BrassLegacyAI.dll');scope='Engine/helper binaries and all AI search, execution, persistence and input-guard sources unchanged; only two inactive-parent lookups changed. Prior rules, 108-turn AI and save/presentation evidence retained; menu initialization and interaction rerun freshly.'}
$qualification=[ordered]@{passed=$true;qualifiedUtc=[DateTime]::UtcNow.ToString('o');files=$candidate.files;changedSources=$changed;sourcePins=$pins;h5Provenance=$h5;retainedEvidence=$retained;nativeMenu=$menu;
 testSourceSha256=(Get-FileHash -LiteralPath (Join-Path $PSScriptRoot 'LocalAiUnityTests.cs')).Hash;
 baselineSha256=(Get-FileHash -LiteralPath (Join-Path $build 'local-ai-cold-menu-baseline.json')).Hash;
 scope='Regression reproduced on 1.4.4. Cold creation before activation, cached reopening and newly created Continue view pass without clicking or refreshing. Existing direct-picker, difficulty, save/Continue, keyboard, pointer, fade and resolution checks pass.'}
$qualification|ConvertTo-Json -Depth 20|Set-Content -LiteralPath (Join-Path $build 'local-ai-qualification.json') -Encoding UTF8
$unity=$old.unityPlayerValidation;$unity.retainedFrom='1.4.4';$unity.scope=$retained.scope;$unity.testedFiles=$retained.unchangedFiles
[ordered]@{passed=$true;validatedUtc=[DateTime]::UtcNow.ToString('o');files=$candidate.files;localAi=$qualification;retainedEvidence=$retained;unityPlayerValidation=$unity}|ConvertTo-Json -Depth 22|Set-Content -LiteralPath (Join-Path $build 'validation.json') -Encoding UTF8
Write-Output 'Qualified 1.4.5: cold-menu initialization repaired; fresh menu regression and unchanged-runtime checks passed.'
