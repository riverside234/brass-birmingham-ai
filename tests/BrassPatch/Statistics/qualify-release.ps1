$ErrorActionPreference='Stop'
Set-StrictMode -Version Latest
$root=Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
$build=Join-Path $root 'artifacts/patch-build'
$prior=Join-Path $root 'artifacts/patch-releases/1.4.7'
function Read-Report([string]$path){Get-Content -LiteralPath $path -Raw|ConvertFrom-Json}
$candidate=Read-Report (Join-Path $build 'build-manifest.json')
$previous=Read-Report (Join-Path $prior 'build-manifest.json')
$old=Read-Report (Join-Path $prior 'validation.json')
$validation=Read-Report (Join-Path $build 'validation.json')
if($candidate.version -ne '1.4.8' -or -not $old.passed -or -not $old.localAi.passed -or -not $validation.passed){throw 'Qualified 1.4.7 and fresh managed 1.4.8 validation are required.'}
function Assert-Files($files){
 foreach($file in $candidate.files){if(@($files|Where-Object {$_.name -eq $file.name -and $_.sha256 -eq $file.sha256}).Count -ne 1){throw 'Evidence belongs to a different candidate.'}}
}
Assert-Files $validation.files
$archive=Read-Report (Join-Path $prior 'archive-manifest.json')
foreach($file in $archive.files){if((Get-FileHash -LiteralPath (Join-Path $prior $file.path)).Hash -ne $file.sha256){throw 'Prior archive changed.'}}
foreach($file in $candidate.files){if((Get-FileHash -LiteralPath (Join-Path $build ('Managed/'+$file.name))).Hash -ne $file.sha256){throw 'Candidate changed.'}}
if(($candidate.files|Where-Object name -eq 'BrassLegacyAI.dll').sha256 -ne ($previous.files|Where-Object name -eq 'BrassLegacyAI.dll').sha256){throw 'AI policy/execution binary must remain identical to 1.4.7.'}
$added=@();$modified=@()
foreach($pin in $candidate.sourcePins) {
 if((Get-FileHash -LiteralPath (Join-Path $root $pin.path)).Hash -ne $pin.sha256){throw 'Runtime source differs from compiled source.'}
 $before=@($previous.sourcePins|Where-Object path -eq $pin.path)
 if(-not $before.Count){
  if(-not $pin.path.StartsWith('src/BrassPatch/Statistics/')){throw ('Unexpected new runtime source: '+$pin.path)}
  $added+=$pin.path
 }elseif($before[0].sha256 -ne $pin.sha256){
  if($pin.path -ne 'src/BrassPatch/Presentation/manifest.json'){throw ('Unrelated runtime change: '+$pin.path)}
  $modified+=$pin.path
 }
}
foreach($pin in $previous.sourcePins){if(@($candidate.sourcePins|Where-Object path -eq $pin.path).Count -ne 1){throw 'A prior runtime source was removed.'}}
$oldHooks=Read-Report (Join-Path $prior 'BrassPatch/Presentation/manifest.json')
$newHooks=Read-Report (Join-Path $root 'src/BrassPatch/Presentation/manifest.json')
if($oldHooks.Count -ne $newHooks.Count){throw 'Unexpected presentation hook inventory.'}
for($i=0;$i -lt $oldHooks.Count;$i++){
 if($oldHooks[$i].targetMethod -eq 'ProcessInput'){
  if($newHooks[$i].helperType -ne 'BrassPatch.Statistics.StatisticsEntryPoints' -or $newHooks[$i].mode -ne 'replace'){throw 'Statistics input gate is missing.'}
 }elseif(($oldHooks[$i]|ConvertTo-Json -Depth 8 -Compress) -ne ($newHooks[$i]|ConvertTo-Json -Depth 8 -Compress)){throw 'An existing presentation hook was modified.'}
}
$evidence=@{}
foreach($name in 'statistics-native','presentation-lifecycle','link-replay','unity-isolated-result','local-ai-menu','local-ai-runtime-0'){
 $report=Read-Report (Join-Path $build ($name+'.json'))
 if(-not $report.passed){throw ('Failed fresh qualification: '+$name)}
 Assert-Files $report.testedFiles;$evidence[$name]=$report
}
$statistics=$evidence['statistics-native']
if($statistics.matches -lt 9 -or $statistics.pages -ne 10 -or $statistics.checks -lt 509 -or @($statistics.nativeErrors).Count){throw 'Completed matches, draw cases, all pages and lifecycle coverage are required.'}
if($evidence['link-replay'].checks -lt 765 -or $evidence['presentation-lifecycle'].checks -lt 180){throw 'Fresh presentation/link coverage is incomplete.'}
$retained=[ordered]@{
 release='1.4.7';validationSha256=(Get-FileHash -LiteralPath (Join-Path $prior 'validation.json')).Hash
 unchangedAI=($candidate.files|Where-Object name -eq 'BrassLegacyAI.dll')
 scope='AI policy, profiles, difficulty/search budgets, execution resources and all existing rules/persistence/presentation implementation sources are unchanged. The 108 native AI configurations and resource-retirement matrix remain qualified for that identical AI DLL. Rules/state/save/city checks, input/presentation/link checks, AI runtime/save/menu smoke and statistics were rerun on this complete candidate.'
}
$ai=$old.localAi
$ai.files=$candidate.files;$ai.sourcePins=$candidate.sourcePins
$ai|Add-Member -NotePropertyName statisticsIntegration -NotePropertyValue $retained -Force
$ai|Add-Member -NotePropertyName retainedFrom -NotePropertyValue '1.4.7' -Force
$ai|ConvertTo-Json -Depth 26|Set-Content -LiteralPath (Join-Path $build 'local-ai-qualification.json') -Encoding UTF8
$validation|Add-Member -NotePropertyName localAi -NotePropertyValue $ai -Force
$validation|Add-Member -NotePropertyName statistics -NotePropertyValue $statistics -Force
$validation|Add-Member -NotePropertyName unityPlayerValidation -NotePropertyValue $evidence['unity-isolated-result'] -Force
$validation|Add-Member -NotePropertyName retainedEvidence -NotePropertyValue $retained -Force
$validation|Add-Member -NotePropertyName statisticsScope -NotePropertyValue @{added=$added;modified=$modified;evidence=@($evidence.Keys|Sort-Object|ForEach-Object {@{name=$_;sha256=(Get-FileHash -LiteralPath (Join-Path $build ($_+'.json'))).Hash}})} -Force
$validation|ConvertTo-Json -Depth 30|Set-Content -LiteralPath (Join-Path $build 'validation.json') -Encoding UTF8
Write-Output 'Qualified 1.4.8: all ten statistics pages, nine completed matches, fresh rules/presentation/AI compatibility and preserved AI execution policy.'
