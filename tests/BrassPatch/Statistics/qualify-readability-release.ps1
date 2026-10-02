$ErrorActionPreference='Stop'
Set-StrictMode -Version Latest
$root=Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
$build=Join-Path $root 'artifacts/patch-build'
$archive=Join-Path $root 'artifacts/patch-releases/1.4.9'
function Read-Report([string]$path){Get-Content -LiteralPath $path -Raw|ConvertFrom-Json}
$candidate=Read-Report (Join-Path $build 'build-manifest.json')
$previous=Read-Report (Join-Path $archive 'build-manifest.json')
$old=Read-Report (Join-Path $archive 'validation.json')
$validation=Read-Report (Join-Path $build 'validation.json')
if($candidate.version -ne '1.4.10' -or -not $old.passed -or -not $old.localAi.passed -or -not $validation.passed -or $validation.suites.Count -ne 12){throw 'Qualified 1.4.9 and fresh managed 1.4.10 validation are required.'}
function Assert-Files($files){
 foreach($file in $candidate.files){if(@($files|Where-Object {$_.name -eq $file.name -and $_.sha256 -eq $file.sha256}).Count -ne 1){throw 'Evidence belongs to a different candidate.'}}
}
Assert-Files $validation.files
& (Join-Path $root 'scripts/stage-archived-patch.ps1') -Version 1.4.9 -VerifyOnly
foreach($file in $candidate.files){if((Get-FileHash -LiteralPath (Join-Path $build ('Managed/'+$file.name))).Hash -ne $file.sha256){throw 'Candidate changed.'}}
foreach($name in 'BrassLegacyAI.dll','Assembly-CSharp.dll'){
 if(($candidate.files|Where-Object name -eq $name).sha256 -ne ($previous.files|Where-Object name -eq $name).sha256){throw ('Unrelated binary changed: '+$name)}
}
$modified=@()
foreach($pin in $candidate.sourcePins){
 if((Get-FileHash -LiteralPath (Join-Path $root $pin.path)).Hash -ne $pin.sha256){throw 'Compiled source changed.'}
 $before=@($previous.sourcePins|Where-Object path -eq $pin.path)
 if($before.Count -ne 1){throw ('Unexpected new runtime source: '+$pin.path)}
 if($before[0].sha256 -ne $pin.sha256){
  if($pin.path -notin @('src/BrassPatch/Statistics/StatisticsPanel.cs','src/BrassPatch/Statistics/StatisticsWidgets.cs')){throw ('Unrelated runtime change: '+$pin.path)}
  $modified+=$pin.path
 }
}
if($previous.sourcePins.Count -ne $candidate.sourcePins.Count -or $modified.Count -ne 2){throw 'Runtime source scope differs.'}
$reports=@{}
foreach($name in 'statistics-native','presentation-lifecycle','link-replay','local-ai-menu'){
 $item=Read-Report (Join-Path $build ($name+'.json'))
 if(-not $item.passed){throw ('Failed fresh evidence: '+$name)}
 Assert-Files $item.testedFiles;$reports[$name]=$item
}
$statistics=$reports['statistics-native']
if($statistics.matches -lt 9 -or $statistics.pages -ne 10 -or $statistics.checks -lt 600 -or @($statistics.nativeErrors).Count){throw 'Native completed-match/readability coverage is incomplete.'}
if(@($statistics.cases|Where-Object name -eq 'visible text uses a readable physical size').Count -lt 50){throw 'All-page/resolution/text-size qualification is missing.'}
if($reports['presentation-lifecycle'].checks -lt 180 -or $reports['link-replay'].checks -lt 765 -or $reports['local-ai-menu'].checks -lt 259){throw 'Fresh presentation/link/menu integration coverage is incomplete.'}
$retained=[ordered]@{
 release='1.4.9';validationSha256=(Get-FileHash -LiteralPath (Join-Path $archive 'validation.json')).Hash
 unchangedFiles=@($candidate.files|Where-Object name -in 'BrassLegacyAI.dll','Assembly-CSharp.dll')
 scope='Only statistics layout/widgets changed. AI DLL and patched game assembly are byte-identical to 1.4.9, and all AI, rules, persistence, statistics collection/projection/page sources and hooks are unchanged. Archived Masked ISMCTS, six-style clock/resource and save compatibility evidence is retained for that identical implementation. Native statistics, input/presentation/link/menu integration and managed regression suites were rerun on the complete candidate. No new playing-strength claim.'
}
$ai=$old.localAi
$ai.files=$candidate.files;$ai.sourcePins=$candidate.sourcePins
$ai.retainedEvidence=$retained
$ai|Add-Member -NotePropertyName retainedFrom -NotePropertyValue '1.4.9' -Force
$ai|ConvertTo-Json -Depth 26|Set-Content -LiteralPath (Join-Path $build 'local-ai-qualification.json') -Encoding UTF8
$validation|Add-Member -NotePropertyName localAi -NotePropertyValue $ai -Force
$validation|Add-Member -NotePropertyName statistics -NotePropertyValue $statistics -Force
$validation|Add-Member -NotePropertyName retainedEvidence -NotePropertyValue $retained -Force
$validation|Add-Member -NotePropertyName statisticsScope -NotePropertyValue @{
 modified=$modified
 evidence=@($reports.Keys|Sort-Object|ForEach-Object {@{name=$_;sha256=(Get-FileHash -LiteralPath (Join-Path $build ($_+'.json'))).Hash}})
} -Force
$validation|ConvertTo-Json -Depth 30|Set-Content -LiteralPath (Join-Path $build 'validation.json') -Encoding UTF8
Write-Output 'Qualified 1.4.10: readable statistics at three resolutions, native input/lifecycle integration and preserved 1.4.9 AI/rules.'
