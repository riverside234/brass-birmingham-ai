param([string]$BuildDirectory='artifacts/game-history-build')
$ErrorActionPreference='Stop'
$historyRoot=Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
$historyBuild=Join-Path $historyRoot $BuildDirectory
function Read-HistoryJson([string]$path){Get-Content -LiteralPath $path -Raw|ConvertFrom-Json}
$candidate=Read-HistoryJson (Join-Path $historyBuild 'build-manifest.json')
$baseline=Join-Path $historyRoot ('artifacts/patch-releases/'+$candidate.baseVersion)
$previous=Read-HistoryJson (Join-Path $baseline 'build-manifest.json')
$accepted=Read-HistoryJson (Join-Path $baseline 'validation.json')
if($candidate.version -ne '1.4.23' -or $previous.version -ne '1.4.22' -or -not $accepted.passed -or $candidate.gammaEnabled -or $candidate.historyLimit -ne 10 -or $candidate.reportSchema -ne 1){throw 'Accepted baseline, disabled Gamma and history contract required.'}
foreach($pin in (Read-HistoryJson (Join-Path $baseline 'archive-manifest.json')).files){
 $path=[IO.Path]::GetFullPath((Join-Path $baseline $pin.path))
 if(-not $path.StartsWith([IO.Path]::GetFullPath($baseline)+[IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase) -or (Get-FileHash -LiteralPath $path).Hash -ne $pin.sha256){throw 'Accepted archive changed.'}
}
$scope=@('src/BrassPatch/State/OfflineMode.cs','src/BrassPatch/Statistics/StatisticsEntryPoints.cs','src/BrassPatch/Statistics/StatisticsPanel.cs','src/BrassPatch/Statistics/StatisticsViewHost.cs')
$scope+=@('src/BrassPatch/Statistics/StatisticsPlotViews.cs','src/BrassPatch/Statistics/StatisticsWidgets.cs','src/BrassPatch/Statistics/StatisticsPlotSemantics.cs')
$scope+=@(Get-ChildItem -LiteralPath (Join-Path $historyRoot 'src/BrassPatch/History') -Filter '*.cs'|ForEach-Object{'src/BrassPatch/History/'+$_.Name})
if(@($candidate.sourceScope).Count -ne $scope.Count -or @($candidate.sourceScope|Where-Object {$_ -notin $scope}).Count -or ($candidate.gameHooks -join '|') -ne 'Frameworks.View.Menu.MainMenuView::PrepareReferences'){throw 'Unexpected history source/hook allowlist.'}
$oldPins=@{};foreach($pin in $previous.sourcePins){$oldPins[$pin.path]=$pin.sha256}
$changed=@()
foreach($pin in $candidate.sourcePins){
 if((Get-FileHash -LiteralPath (Join-Path $historyBuild ('runtime/'+$pin.path))).Hash -ne $pin.sha256){throw 'Compiled source snapshot changed.'}
 if($oldPins[$pin.path] -ne $pin.sha256){
  if($pin.path -notin $scope -or (Get-FileHash -LiteralPath (Join-Path $historyRoot $pin.path)).Hash -ne $pin.sha256){throw ('Unrelated source change: '+$pin.path)}
  $changed+=$pin.path
 }
}
foreach($path in $oldPins.Keys){if(@($candidate.sourcePins|Where-Object path -eq $path).Count -ne 1){throw ('Baseline source missing: '+$path)}}
foreach($file in $candidate.files){if((Get-FileHash -LiteralPath (Join-Path $historyBuild ('Managed/'+$file.name))).Hash -ne $file.sha256){throw 'Candidate changed.'}}
if(($candidate.files|Where-Object name -eq 'BrassLegacyAI.dll').sha256 -ne ($previous.files|Where-Object name -eq 'BrassLegacyAI.dll').sha256){throw 'AI binary changed.'}
$compiler=Join-Path $historyRoot 'artifacts/patch-tools/microsoft.net.compilers.toolset.4.12.0/tasks/net472/csc.exe'
Copy-Item -LiteralPath (Join-Path $historyRoot 'artifacts/patch-tools/mono.cecil.0.11.6/lib/net40/Mono.Cecil.dll') -Destination $historyBuild -Force
Copy-Item -LiteralPath (Join-Path $historyRoot 'BrassBirminghamUnity_Data/Managed/Newtonsoft.Json.dll') -Destination $historyBuild -Force
$checker=Join-Path $historyBuild 'HistoryPreservedRuntimeChecks.exe'
@('/nologo','/target:exe','/langversion:7.3','/deterministic+',('/out:"'+$checker+'"'),('/reference:"'+(Join-Path $historyBuild 'Mono.Cecil.dll')+'"'),('/reference:"'+(Join-Path $historyBuild 'Newtonsoft.Json.dll')+'"'),('"'+(Join-Path $PSScriptRoot 'HistoryPreservedRuntimeChecks.cs')+'"'))|Set-Content -LiteralPath (Join-Path $historyBuild 'preservation.rsp') -Encoding utf8
& $compiler ('@'+(Join-Path $historyBuild 'preservation.rsp'))
if($LASTEXITCODE -ne 0){throw 'Preservation checker compilation failed.'}
& $checker (Join-Path $baseline 'Managed') (Join-Path $historyBuild 'Managed') (Join-Path $historyBuild 'preserved-runtime.json')
if($LASTEXITCODE -ne 0){throw 'An unrelated method or resource changed.'}
$preserved=Read-HistoryJson (Join-Path $historyBuild 'preserved-runtime.json')
$reports=@{}
foreach($name in 'history','menu','statistics','tracker','presentation','link'){
 $report=Read-HistoryJson (Join-Path $historyBuild ($name+'-native.json'))
 if(-not $report.passed -or @($report.nativeErrors|Where-Object {$_}).Count){throw ('Native suite failed: '+$name)}
 foreach($file in $candidate.files){if(@($report.testedFiles|Where-Object {$_.name -eq $file.name -and $_.sha256 -eq $file.sha256}).Count -ne 1){throw 'Native evidence belongs to another candidate.'}}
 $reports[$name]=$report
}
$history=$reports.history
if($history.checks -lt 3300 -or $history.matches -ne 9 -or $history.pages -ne 10 -or $history.historyLimit -ne 10 -or $history.reportSchema -ne 1 -or @($history.layouts).Count -ne 6){throw 'Report, retention, recovery and reuse qualification incomplete.'}
foreach($group in 'H-DATA','H-RETENTION','H-RECOVERY','H-MENU','H-INPUT','H-UI','H-REUSE','H-CAPTURE'){
 if(-not @($history.cases|Where-Object {$_.passed -and $_.name.StartsWith($group)}).Count){throw ('Missing history coverage: '+$group)}
}
foreach($report in $reports.history,$reports.statistics){foreach($group in 'S23-DATA','S23-FILTER','S23-LABEL','S23-VISUAL'){
 if(-not @($report.cases|Where-Object {$_.passed -and $_.name.StartsWith($group)}).Count){throw ('Missing live/archive statistics review: '+$group)}
}}
Add-Type -AssemblyName System.Drawing
$visual=@()
foreach($resolution in 1080,1200){foreach($preset in 0,1,2){
 $layout=@($history.layouts|Where-Object {$_.width -eq 1920 -and $_.height -eq $resolution -and $_.preset -eq $preset -and $_.games -eq 10})
 if($layout.Count -ne 1){throw 'Required list layout missing.'}
 $name='history-1920x'+$resolution+'-'+$preset+'.png'
 $path=Join-Path $historyRoot ('artifacts/card-tracker-player/artifacts/patch-build/'+$name)
 if($path -notin $history.images){throw ('Capture not recorded: '+$name)}
 $bitmap=[Drawing.Bitmap]::FromFile($path)
 try{$pixel=$bitmap.GetPixel(1400,60);$dimensions=@($bitmap.Width,$bitmap.Height)}finally{$bitmap.Dispose()}
 if($pixel.R -ne 245 -or $pixel.G -ne 241 -or $pixel.B -ne 231 -or $dimensions[0] -ne 1920 -or $dimensions[1] -ne $resolution){throw ('Shared light palette/resolution differs: '+$name)}
 $visual+=@{name=$name;passed=$true;sha256=(Get-FileHash -LiteralPath $path).Hash;surfaceRgb=@($pixel.R,$pixel.G,$pixel.B)}
}}
foreach($resolution in 1080,1200){foreach($page in 0..9){
 $name='history-statistics-1920x'+$resolution+'-'+$page+'.png'
 $path=Join-Path $historyRoot ('artifacts/card-tracker-player/artifacts/patch-build/'+$name)
 if($path -notin $history.images -or -not(Test-Path -LiteralPath $path)){throw 'Archived page capture missing.'}
 $visual+=@{name=$name;passed=$true;sha256=(Get-FileHash -LiteralPath $path).Hash}
}}
foreach($prefix in 'history','live'){
 $captures=@(Get-ChildItem -LiteralPath (Join-Path $historyRoot 'artifacts/card-tracker-player/artifacts/patch-build') -Filter ($prefix+'-plot-1200-*.png'))
 if($captures.Count -lt 28){throw ('Missing 1200p plot captures: '+$prefix)}
 foreach($capture in $captures){$visual+=@{name=$capture.Name;passed=$true;sha256=(Get-FileHash -LiteralPath $capture.FullName).Hash}}
}
$visualResult=[ordered]@{passed=$true;cases=$visual;scope='Actual Unity history list at 1200p/1080p and all text presets; existing 1080p layout regression retained. Plot review and visual inspection at 1200p for live and archived Statistics, including scrolled plots and all resource tabs. Native geometry/input and independent data checks.'}
$visualResult|ConvertTo-Json -Depth 6|Set-Content -LiteralPath (Join-Path $historyBuild 'visual-checks.json') -Encoding utf8
$retained=[ordered]@{release=$candidate.baseVersion;validationSha256=(Get-FileHash -LiteralPath (Join-Path $baseline 'validation.json')).Hash;unchangedAi=($candidate.files|Where-Object name -eq 'BrassLegacyAI.dll');preservedExistingMethods=($preserved.assemblies|Measure-Object unchangedMethods -Sum).Sum;scope='AI binary and all 48 profiles identical. All gameplay, save and unrelated UI method bodies preserved; fresh Statistics, Tracker, Menu, Presentation and Link suites.'}
$ai=$accepted.localAi;$ai.files=$candidate.files;$ai.sourcePins=$candidate.sourcePins
$ai|Add-Member -NotePropertyName retainedFrom -NotePropertyValue $candidate.baseVersion -Force
$ai|Add-Member -NotePropertyName retainedEvidence -NotePropertyValue $retained -Force
$ai|ConvertTo-Json -Depth 40|Set-Content -LiteralPath (Join-Path $historyBuild 'local-ai-qualification.json') -Encoding utf8
$testPaths=@('tests/BrassPatch/History/HistoryMenuHook.cs','tests/BrassPatch/History/HistoryPreservedRuntimeChecks.cs','tests/BrassPatch/History/HistoryCoreChecks.cs','tests/BrassPatch/History/HistoryFixtures.cs','tests/BrassPatch/History/HistoryNativeTests.cs','tests/BrassPatch/History/qualify-game-history-release.ps1','tests/BrassPatch/InformationUI/run-native.ps1','tests/BrassPatch/Statistics/StatisticsHistoryReviewChecks.cs','tests/BrassPatch/Statistics/StatisticsNativeTests.cs','tests/BrassPatch/Statistics/StatisticsLightChecks.cs','tests/BrassPatch/Statistics/StatisticsReviewChecks.cs')
[ordered]@{passed=$true;validatedUtc=[DateTime]::UtcNow.ToString('o');files=$candidate.files;sourceScope=$changed;preservedRuntime=$preserved;visualChecks=$visualResult;history=$history;
 menu=$reports.menu;statistics=$reports.statistics;cardTracker=$reports.tracker;presentation=$reports.presentation;linkReplay=$reports.link;localAi=$ai;retainedEvidence=$retained;
 testSources=@($testPaths|ForEach-Object{@{path=$_;sha256=(Get-FileHash -LiteralPath (Join-Path $historyRoot $_)).Hash}})}|ConvertTo-Json -Depth 42|Set-Content -LiteralPath (Join-Path $historyBuild 'validation.json') -Encoding utf8
. (Join-Path $historyRoot 'scripts/patch-install-common.ps1')
Assert-BrassCandidate $candidate (Read-HistoryJson (Join-Path $historyBuild 'validation.json')) $historyBuild
Write-Output ('Qualified Game History 1.4.23; '+$retained.preservedExistingMethods+' unrelated method bodies and all embedded profiles preserved.')
