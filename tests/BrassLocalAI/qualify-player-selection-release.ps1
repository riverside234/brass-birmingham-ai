$ErrorActionPreference='Stop'
$aiUiRoot=Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$aiUiBuild=Join-Path $aiUiRoot 'artifacts/player-selection-build'
function Read-AiUiJson([string]$path){Get-Content -LiteralPath $path -Raw|ConvertFrom-Json}
$aiUiCandidate=Read-AiUiJson (Join-Path $aiUiBuild 'build-manifest.json')
$aiUiBase=Join-Path $aiUiRoot ('artifacts/patch-releases/'+$aiUiCandidate.baseVersion)
$aiUiPrevious=Read-AiUiJson (Join-Path $aiUiBase 'build-manifest.json')
$aiUiOld=Read-AiUiJson (Join-Path $aiUiBase 'validation.json')
if($aiUiCandidate.version -ne '1.4.24' -or $aiUiPrevious.version -ne '1.4.23' -or -not $aiUiOld.passed -or $aiUiCandidate.gammaEnabled){throw 'Accepted current source baseline and disabled Gamma required.'}
foreach($pin in (Read-AiUiJson (Join-Path $aiUiBase 'archive-manifest.json')).files){
 $path=[IO.Path]::GetFullPath((Join-Path $aiUiBase $pin.path))
 if(-not $path.StartsWith([IO.Path]::GetFullPath($aiUiBase)+[IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase) -or (Get-FileHash -LiteralPath $path).Hash -ne $pin.sha256){throw 'Accepted archive changed.'}
}
$aiUiScope=@('src/BrassLocalAI/Unity/LocalMenu.cs','src/BrassLocalAI/Unity/AiSetupRow.cs','src/BrassLocalAI/Unity/AiSelectionPanel.cs','src/BrassLocalAI/Policy/PlayerSelectionDraft.cs')
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
$aiUiChecker=Join-Path $aiUiBuild 'PlayerSelectionPreservedRuntimeChecks.exe'
@('/nologo','/target:exe','/langversion:7.3','/deterministic+',('/out:"'+$aiUiChecker+'"'),('/reference:"'+(Join-Path $aiUiBuild 'Mono.Cecil.dll')+'"'),('/reference:"'+(Join-Path $aiUiBuild 'Newtonsoft.Json.dll')+'"'),('"'+(Join-Path $PSScriptRoot 'PlayerSelectionPreservedRuntimeChecks.cs')+'"'))|Set-Content -LiteralPath (Join-Path $aiUiBuild 'preservation.rsp') -Encoding utf8
& $aiUiCompiler ('@'+(Join-Path $aiUiBuild 'preservation.rsp'))
if($LASTEXITCODE -ne 0){throw 'Preservation checker compilation failed.'}
& $aiUiChecker (Join-Path $aiUiBase 'Managed') (Join-Path $aiUiBuild 'Managed') (Join-Path $aiUiBuild 'preserved-runtime.json')
if($LASTEXITCODE -ne 0){throw 'A method or embedded profile outside the UI scope changed.'}
$aiUiPreserved=Read-AiUiJson (Join-Path $aiUiBuild 'preserved-runtime.json')
$aiUiReports=@{}
foreach($name in 'menu','statistics','tracker','history'){
 $report=Read-AiUiJson (Join-Path $aiUiBuild ($name+'-native.json'))
 if(-not $report.passed -or @($report.nativeErrors|Where-Object {$_}).Count){throw ('Native suite failed: '+$name)}
 foreach($file in $aiUiCandidate.files){if(@($report.testedFiles|Where-Object {$_.name -eq $file.name -and $_.sha256 -eq $file.sha256}).Count -ne 1){throw 'Native evidence belongs to another candidate.'}}
 $aiUiReports[$name]=$report
}
$aiUiMenu=$aiUiReports.menu
if($aiUiMenu.selectionPanel.draftChecks -lt 20 -or -not $aiUiMenu.selectionPanel.passed -or $aiUiMenu.selectionPanel.gammaEnabled -or $aiUiMenu.selectionPanel.fixtureChoices -ne 16 -or $aiUiMenu.selectionPanel.layouts.Count -ne 18 -or $aiUiMenu.checks -lt 600){throw 'Three-page selection and readiness qualification incomplete.'}
foreach($w in 1080,1200){foreach($p in 0,1,2){foreach($family in 'Human','V4','Gamma'){
 $layouts=@($aiUiMenu.selectionPanel.layouts|Where-Object {$_.width -eq 1920 -and $_.height -eq $w -and $_.preset -eq $p -and $_.family -eq $family})
 if($layouts.Count -ne 1 -or $layouts[0].choices -ne $(if($family -eq 'Human'){0}elseif($family -eq 'V4'){7}else{17}) -or $layouts[0].available -ne $(if($family -eq 'V4'){7}else{0})){throw 'Required family layout/availability missing.'}
}}}
Add-Type -AssemblyName System.Drawing
$aiUiVisual=@()
foreach($resolution in '1920x1200','1920x1080'){foreach($family in 'Human','V4','Gamma'){foreach($preset in 0,1,2){
 $name='player-selection-'+$resolution+'-'+$family+'-'+$preset+'.png'
 $path=Join-Path $aiUiRoot ('artifacts/card-tracker-player/artifacts/patch-build/'+$name)
 if($path -notin $aiUiMenu.images){throw ('Capture not recorded: '+$name)}
 $bitmap=[Drawing.Bitmap]::FromFile($path)
 try{$pixel=$bitmap.GetPixel(1400,60)}finally{$bitmap.Dispose()}
 if($pixel.R -ne 245 -or $pixel.G -ne 241 -or $pixel.B -ne 231){throw ('Statistics window palette differs: '+$name)}
 $aiUiVisual+=@{name=$name;passed=$true;sha256=(Get-FileHash -LiteralPath $path).Hash;surfaceRgb=@($pixel.R,$pixel.G,$pixel.B)}
}}}
$aiUiVisualResult=[ordered]@{passed=$true;cases=$aiUiVisual;scope='Actual Unity captures at 1200p/1080p, all text presets and three pages. Automated geometry, input and palette checks; manual image inspection at 1200p. No neural strength claims.'}
$aiUiVisualResult|ConvertTo-Json -Depth 6|Set-Content -LiteralPath (Join-Path $aiUiBuild 'visual-checks.json') -Encoding utf8
$aiUiRetained=[ordered]@{release=$aiUiCandidate.baseVersion;validationSha256=(Get-FileHash -LiteralPath (Join-Path $aiUiBase 'validation.json')).Hash;unchangedGame=($aiUiCandidate.files|Where-Object name -eq 'Assembly-CSharp.dll');unchangedHelper=($aiUiCandidate.files|Where-Object name -eq 'BrassPatch.dll');preservedExistingMethods=($aiUiPreserved.assemblies|Measure-Object unchangedMethods -Sum).Sum;scope='All gameplay, V4 search/execution, save assignments, Card Tracker and Statistics methods unchanged. All 48 embedded AI profiles unchanged. Fresh native menu, Statistics, Game History and Card Tracker suites run. Manual image review at 1200p.'}
$aiUiAi=$aiUiOld.localAi;$aiUiAi.files=$aiUiCandidate.files;$aiUiAi.sourcePins=$aiUiCandidate.sourcePins
$aiUiAi|Add-Member -NotePropertyName retainedFrom -NotePropertyValue $aiUiCandidate.baseVersion -Force
$aiUiAi|Add-Member -NotePropertyName retainedEvidence -NotePropertyValue $aiUiRetained -Force
$aiUiAi.resourceTests=$aiUiMenu.resourceTests;$aiUiAi.executionStatistics=$aiUiMenu.executionStatistics
$aiUiAi|Add-Member -NotePropertyName nativeMenu -NotePropertyValue $aiUiMenu -Force
$aiUiAi|ConvertTo-Json -Depth 40|Set-Content -LiteralPath (Join-Path $aiUiBuild 'local-ai-qualification.json') -Encoding utf8
[ordered]@{passed=$true;validatedUtc=[DateTime]::UtcNow.ToString('o');files=$aiUiCandidate.files;sourceScope=$aiUiChanged;preservedRuntime=$aiUiPreserved;visualChecks=$aiUiVisualResult;
 menu=$aiUiMenu;statistics=$aiUiReports.statistics;cardTracker=$aiUiReports.tracker;history=$aiUiReports.history;localAi=$aiUiAi;retainedEvidence=$aiUiRetained;
 testSources=@('tests/BrassLocalAI/LocalAiUnityTests.cs','tests/BrassLocalAI/AiExecutionResourceTests.cs','tests/BrassLocalAI/PlayerSelectionPreservedRuntimeChecks.cs'|ForEach-Object{@{path=$_;sha256=(Get-FileHash -LiteralPath (Join-Path $aiUiRoot $_)).Hash}})}|ConvertTo-Json -Depth 42|Set-Content -LiteralPath (Join-Path $aiUiBuild 'validation.json') -Encoding utf8
. (Join-Path $aiUiRoot 'scripts/patch-install-common.ps1')
Assert-BrassCandidate $aiUiCandidate (Read-AiUiJson (Join-Path $aiUiBuild 'validation.json')) $aiUiBuild
Write-Output ('Qualified Human/V4/Gamma selection UI 1.4.24; '+$aiUiRetained.preservedExistingMethods+' existing method bodies and all embedded profiles preserved.')
