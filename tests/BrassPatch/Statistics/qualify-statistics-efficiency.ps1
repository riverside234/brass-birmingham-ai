$ErrorActionPreference='Stop'
$aiUiRoot=Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
$aiUiBuild=Join-Path $aiUiRoot 'artifacts/statistics-efficiency-build'
function Read-AiUiJson([string]$path){Get-Content -LiteralPath $path -Raw|ConvertFrom-Json}
$aiUiCandidate=Read-AiUiJson (Join-Path $aiUiBuild 'build-manifest.json')
$aiUiBase=Join-Path $aiUiRoot ('artifacts/patch-releases/'+$aiUiCandidate.baseVersion)
$aiUiPrevious=Read-AiUiJson (Join-Path $aiUiBase 'build-manifest.json')
$aiUiOld=Read-AiUiJson (Join-Path $aiUiBase 'validation.json')
if($aiUiCandidate.version -ne '1.4.26' -or $aiUiPrevious.version -ne '1.4.25' -or -not $aiUiOld.passed -or $aiUiCandidate.gammaEnabled){throw 'Accepted current source baseline and disabled Gamma required.'}
foreach($pin in (Read-AiUiJson (Join-Path $aiUiBase 'archive-manifest.json')).files){
 $path=[IO.Path]::GetFullPath((Join-Path $aiUiBase $pin.path))
 if(-not $path.StartsWith([IO.Path]::GetFullPath($aiUiBase)+[IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase) -or (Get-FileHash -LiteralPath $path).Hash -ne $pin.sha256){throw 'Accepted archive changed.'}
}
$aiUiScope=@('src/BrassPatch/Statistics/StatisticsPages.cs','src/BrassPatch/Statistics/StatisticsMetrics.cs','src/BrassPatch/Statistics/StatisticsChartQueries.cs','src/BrassPatch/Statistics/StatisticsPlotModels.cs','src/BrassPatch/Statistics/StatisticsPlotPages.cs','src/BrassPatch/Statistics/StatisticsEfficiencyPages.cs','src/BrassPatch/Statistics/StatisticsPlotControls.cs','src/BrassPatch/Statistics/StatisticsMedal.cs','src/BrassPatch/Statistics/StatisticsReviewPanel.cs','src/BrassPatch/Statistics/StatisticsPanel.cs','src/BrassPatch/Statistics/StatisticsPlotSemantics.cs','src/BrassPatch/Statistics/StatisticsPlotViews.cs','src/BrassPatch/Statistics/StatisticsPresentationLease.cs','src/BrassPatch/Statistics/StatisticsEntryPoints.cs')
if(@($aiUiCandidate.sourceScope).Count -ne $aiUiScope.Count -or @($aiUiCandidate.sourceScope|Where-Object {$_ -notin $aiUiScope}).Count){throw 'Unexpected UI source allowlist.'}
$aiUiBaselinePins=@{};foreach($pin in $aiUiPrevious.sourcePins){$aiUiBaselinePins[$pin.path]=$pin.sha256}
$aiUiChanged=@()
foreach($pin in $aiUiCandidate.sourcePins){
 if((Get-FileHash -LiteralPath (Join-Path $aiUiBuild ('runtime/'+$pin.path))).Hash -ne $pin.sha256){throw 'Compiled source snapshot changed.'}
 if($aiUiBaselinePins[$pin.path] -ne $pin.sha256){
  if($pin.path -notin $aiUiScope -or (Get-FileHash -LiteralPath (Join-Path $aiUiRoot $pin.path)).Hash -ne $pin.sha256){throw ('Unrelated source change: '+$pin.path)}
  $aiUiChanged+=$pin.path
 }
}
foreach($path in $aiUiBaselinePins.Keys){if(@($aiUiCandidate.sourcePins|Where-Object path -eq $path).Count -ne 1){throw ('Baseline source missing: '+$path)}}
foreach($file in $aiUiCandidate.files){if((Get-FileHash -LiteralPath (Join-Path $aiUiBuild ('Managed/'+$file.name))).Hash -ne $file.sha256){throw 'Candidate changed.'}}
if(($aiUiCandidate.files|Where-Object name -eq 'Assembly-CSharp.dll').sha256 -ne ($aiUiPrevious.files|Where-Object name -eq 'Assembly-CSharp.dll').sha256){throw 'Game assembly changed.'}
$aiUiCompiler=Join-Path $aiUiRoot 'artifacts/patch-tools/microsoft.net.compilers.toolset.4.12.0/tasks/net472/csc.exe'
Copy-Item -LiteralPath (Join-Path $aiUiRoot 'artifacts/patch-tools/mono.cecil.0.11.6/lib/net40/Mono.Cecil.dll') -Destination $aiUiBuild -Force
Copy-Item -LiteralPath (Join-Path $aiUiRoot 'BrassBirminghamUnity_Data/Managed/Newtonsoft.Json.dll') -Destination $aiUiBuild -Force
$aiUiChecker=Join-Path $aiUiBuild 'StatisticsEfficiencyPreservedRuntimeChecks.exe'
@('/nologo','/target:exe','/langversion:7.3','/deterministic+',('/out:"'+$aiUiChecker+'"'),('/reference:"'+(Join-Path $aiUiBuild 'Mono.Cecil.dll')+'"'),('/reference:"'+(Join-Path $aiUiBuild 'Newtonsoft.Json.dll')+'"'),('"'+(Join-Path $PSScriptRoot 'StatisticsEfficiencyPreservedRuntimeChecks.cs')+'"'))|Set-Content -LiteralPath (Join-Path $aiUiBuild 'preservation.rsp') -Encoding utf8
& $aiUiCompiler ('@'+(Join-Path $aiUiBuild 'preservation.rsp'))
if($LASTEXITCODE -ne 0){throw 'Preservation checker compilation failed.'}
& $aiUiChecker (Join-Path $aiUiBase 'Managed') (Join-Path $aiUiBuild 'Managed') (Join-Path $aiUiBuild 'preserved-runtime.json')
if($LASTEXITCODE -ne 0){throw 'A method or embedded profile outside the UI scope changed.'}
$aiUiPreserved=Read-AiUiJson (Join-Path $aiUiBuild 'preserved-runtime.json')
$aiUiReports=@{}
foreach($name in 'menu','statistics','tracker','history'){
 $report=Read-AiUiJson (Join-Path $aiUiBuild ($name+'-native.json'))
 if(-not $report.passed -or $report.captureSkipped -or @($report.nativeErrors|Where-Object {$_}).Count){throw ('Native suite failed: '+$name)}
 foreach($file in $aiUiCandidate.files){if(@($report.testedFiles|Where-Object {$_.name -eq $file.name -and $_.sha256 -eq $file.sha256}).Count -ne 1){throw 'Native evidence belongs to another candidate.'}}
 $aiUiReports[$name]=$report
}
$aiUiMenu=$aiUiReports.menu
if($aiUiMenu.selectionPanel.draftChecks -lt 20 -or -not $aiUiMenu.selectionPanel.passed -or $aiUiMenu.selectionPanel.gammaEnabled -or $aiUiMenu.selectionPanel.fixtureChoices -ne 16 -or $aiUiMenu.selectionPanel.layouts.Count -ne 18 -or $aiUiMenu.checks -lt 600){throw 'Three-page selection and readiness qualification incomplete.'}
foreach($w in 1080,1200){foreach($p in 0,1,2){foreach($family in 'Human','V4','Gamma'){
 $layouts=@($aiUiMenu.selectionPanel.layouts|Where-Object {$_.width -eq 1920 -and $_.height -eq $w -and $_.preset -eq $p -and $_.family -eq $family})
 if($layouts.Count -ne 1 -or $layouts[0].choices -ne $(if($family -eq 'Human'){0}elseif($family -eq 'V4'){7}else{17}) -or $layouts[0].available -ne $(if($family -eq 'V4'){7}else{0})){throw 'Required family layout/availability missing.'}
}}}
$tracker=$aiUiReports.tracker
foreach($group in 'CT25-POSITION','CT25-INPUT','CT25-THEME','CT25-CITY') {
 if(-not @($tracker.cases|Where-Object {$_.passed -and $_.name.StartsWith($group)}).Count){throw ('Missing tracker regression: '+$group)}
}
$aiUiVisual=@()
foreach($prefix in 'live','history'){
 foreach($plot in 'industry-realization','construction-return','completion-speed','unflipped-investment','carryover-scoring','link-return'){
  foreach($size in 1,2){
   $name='statistics-efficiency-'+$prefix+'-'+$plot+'-text'+$size+'-1200.png'
   $path=Join-Path $aiUiRoot ('artifacts/card-tracker-player/artifacts/patch-build/'+$name)
   if(-not(Test-Path -LiteralPath $path)){throw ('Missing capture: '+$name)}
   $aiUiVisual+=@{name=$name;passed=$true;sha256=(Get-FileHash -LiteralPath $path).Hash}
  }
 }
 $name='statistics-efficiency-'+$prefix+'-overview-1200.png';$path=Join-Path $aiUiRoot ('artifacts/card-tracker-player/artifacts/patch-build/'+$name)
 $aiUiVisual+=@{name=$name;passed=$true;sha256=(Get-FileHash -LiteralPath $path).Hash}
}
$aiUiVisualResult=[ordered]@{passed=$true;cases=$aiUiVisual;scope='Native Unity 1920x1200: all six new plots and overview medals in live and archive entry points, default second and largest text. Manually reviewed images; automated geometry/readability/labels/evidence checks.'}
$aiUiVisualResult|ConvertTo-Json -Depth 6|Set-Content -LiteralPath (Join-Path $aiUiBuild 'visual-checks.json') -Encoding utf8
$performance=Read-AiUiJson (Join-Path $aiUiBuild 'performance-native.json')
if(-not $performance.passed -or $performance.width -ne 1920 -or $performance.height -ne 1200){throw 'Native performance fixture incomplete.'}
foreach($file in $aiUiCandidate.files){if(@($performance.testedFiles|Where-Object {$_.name -eq $file.name -and $_.sha256 -eq $file.sha256}).Count -ne 1){throw 'Performance evidence is from another candidate.'}}
foreach($group in 'E26-DATA','E26-UI','E26-LIFECYCLE','E26-OVERVIEW'){
 $cases=@($aiUiReports.statistics.cases|Where-Object {$_.name.StartsWith($group)})
 if(-not $cases.Count -or @($cases|Where-Object {-not $_.passed}).Count){throw ('Missing/failed '+$group)}
}
$aiUiRetained=[ordered]@{release=$aiUiCandidate.baseVersion;validationSha256=(Get-FileHash -LiteralPath (Join-Path $aiUiBase 'validation.json')).Hash;unchangedGame=($aiUiCandidate.files|Where-Object name -eq 'Assembly-CSharp.dll');unchangedAi=($aiUiCandidate.files|Where-Object name -eq 'BrassLegacyAI.dll');preservedExistingMethods=($aiUiPreserved.assemblies|Measure-Object unchangedMethods -Sum).Sum;scope='All gameplay, V4 search/execution, save assignments, Card Tracker, player selection and History persistence methods unchanged. All 48 embedded AI profiles unchanged. Statistics-only revision with four native regression suites and a matched performance probe; manual visual review at 1200p.'}
$aiUiAi=$aiUiOld.localAi;$aiUiAi.files=$aiUiCandidate.files;$aiUiAi.sourcePins=$aiUiCandidate.sourcePins
$aiUiAi|Add-Member -NotePropertyName retainedFrom -NotePropertyValue $aiUiCandidate.baseVersion -Force
$aiUiAi|Add-Member -NotePropertyName retainedEvidence -NotePropertyValue $aiUiRetained -Force
$aiUiAi.resourceTests=$aiUiMenu.resourceTests;$aiUiAi.executionStatistics=$aiUiMenu.executionStatistics
$aiUiAi|Add-Member -NotePropertyName nativeMenu -NotePropertyValue $aiUiMenu -Force
$aiUiAi|ConvertTo-Json -Depth 40|Set-Content -LiteralPath (Join-Path $aiUiBuild 'local-ai-qualification.json') -Encoding utf8
[ordered]@{passed=$true;validatedUtc=[DateTime]::UtcNow.ToString('o');files=$aiUiCandidate.files;sourceScope=$aiUiChanged;preservedRuntime=$aiUiPreserved;visualChecks=$aiUiVisualResult;
 menu=$aiUiMenu;statistics=$aiUiReports.statistics;cardTracker=$aiUiReports.tracker;history=$aiUiReports.history;performance=$performance;localAi=$aiUiAi;retainedEvidence=$aiUiRetained;
 testSources=@('tests/BrassPatch/Statistics/StatisticsEfficiencyChecks.cs','tests/BrassPatch/Statistics/StatisticsPerformanceNativeTests.cs','tests/BrassPatch/Statistics/StatisticsNativeTests.cs','tests/BrassPatch/Statistics/StatisticsHistoryReviewChecks.cs','tests/BrassPatch/Statistics/StatisticsReviewChecks.cs','tests/BrassPatch/Statistics/StatisticsLightChecks.cs','tests/BrassPatch/History/HistoryNativeTests.cs','tests/BrassPatch/History/HistoryFixtures.cs','tests/BrassPatch/History/HistoryCoreChecks.cs','tests/BrassLocalAI/LocalAiUnityTests.cs','tests/BrassLocalAI/AiExecutionResourceTests.cs','tests/BrassPatch/CardTracker/CardTrackerNativeTests.cs','tests/BrassPatch/Statistics/StatisticsEfficiencyPreservedRuntimeChecks.cs'|ForEach-Object{@{path=$_;sha256=(Get-FileHash -LiteralPath (Join-Path $aiUiRoot $_)).Hash}})}|ConvertTo-Json -Depth 42|Set-Content -LiteralPath (Join-Path $aiUiBuild 'validation.json') -Encoding utf8
. (Join-Path $aiUiRoot 'scripts/patch-install-common.ps1')
Assert-BrassCandidate $aiUiCandidate (Read-AiUiJson (Join-Path $aiUiBuild 'validation.json')) $aiUiBuild
Write-Output ('Qualified Statistics efficiency 1.4.26; '+$aiUiRetained.preservedExistingMethods+' existing method bodies and all embedded profiles preserved.')

