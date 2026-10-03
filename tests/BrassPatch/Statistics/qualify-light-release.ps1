$ErrorActionPreference='Stop'
$statRoot=Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
$statBuild=Join-Path $statRoot 'artifacts/statistics-light-build'
function Read-StatJson([string]$path){Get-Content -LiteralPath $path -Raw|ConvertFrom-Json}
$statCandidate=Read-StatJson (Join-Path $statBuild 'build-manifest.json')
$statBase=Join-Path $statRoot ('artifacts/patch-releases/'+$statCandidate.baseVersion)
$statPrevious=Read-StatJson (Join-Path $statBase 'build-manifest.json')
$statOld=Read-StatJson (Join-Path $statBase 'validation.json')
if($statCandidate.version -ne '1.4.21' -or $statPrevious.version -ne '1.4.20' -or $statCandidate.helperSourceBaseVersion -ne '1.4.19' -or -not $statOld.passed){throw 'Qualified current game/AI and helper baselines required.'}
$statBaselinePins=@{}
foreach($version in '1.4.19','1.4.20'){
 $archive=Join-Path $statRoot ('artifacts/patch-releases/'+$version)
 foreach($pin in (Read-StatJson (Join-Path $archive 'archive-manifest.json')).files){
  $path=[IO.Path]::GetFullPath((Join-Path $archive $pin.path))
  if(-not $path.StartsWith([IO.Path]::GetFullPath($archive)+[IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase) -or (Get-FileHash -LiteralPath $path).Hash -ne $pin.sha256){throw 'Accepted archive changed.'}
 }
 foreach($pin in (Read-StatJson (Join-Path $archive 'build-manifest.json')).sourcePins){$statBaselinePins[$pin.path]=$pin.sha256}
}
foreach($file in $statCandidate.files){if((Get-FileHash -LiteralPath (Join-Path $statBuild ('Managed/'+$file.name))).Hash -ne $file.sha256){throw 'Candidate changed.'}}
foreach($name in 'Assembly-CSharp.dll','BrassLegacyAI.dll'){if(($statCandidate.files|Where-Object name -eq $name).sha256 -ne ($statPrevious.files|Where-Object name -eq $name).sha256){throw 'Accepted game or AI binary changed.'}}
$statModified=@()
foreach($pin in $statCandidate.sourcePins){
 if((Get-FileHash -LiteralPath (Join-Path $statBuild ('runtime/'+$pin.path))).Hash -ne $pin.sha256){throw 'Compiled snapshot changed.'}
 if($statBaselinePins[$pin.path] -ne $pin.sha256){
  if($pin.path -notlike 'src/BrassPatch/Statistics/*' -or $pin.path -notin $statCandidate.sourceScope -or (Get-FileHash -LiteralPath (Join-Path $statRoot $pin.path)).Hash -ne $pin.sha256){throw ('Unexpected source change: '+$pin.path)}
  $statModified+=$pin.path
 }
}
foreach($path in $statBaselinePins.Keys){if($path -notin $statCandidate.retiredSources -and @($statCandidate.sourcePins|Where-Object path -eq $path).Count -ne 1){throw ('Baseline source missing: '+$path)}}
if(($statCandidate.retiredSources -join '|') -ne 'src/BrassPatch/Statistics/StatisticsInsights.cs'){throw 'Only the retired advice engine may be removed.'}
$statCompiler=Join-Path $statRoot 'artifacts/patch-tools/microsoft.net.compilers.toolset.4.12.0/tasks/net472/csc.exe'
Copy-Item -LiteralPath (Join-Path $statRoot 'artifacts/patch-tools/mono.cecil.0.11.6/lib/net40/Mono.Cecil.dll') -Destination $statBuild -Force
Copy-Item -LiteralPath (Join-Path $statRoot 'BrassBirminghamUnity_Data/Managed/Newtonsoft.Json.dll') -Destination $statBuild -Force
$statCheck=Join-Path $statBuild 'StatisticsPreservedRuntimeChecks.exe'
@('/nologo','/target:exe','/langversion:7.3','/deterministic+',('/out:"'+$statCheck+'"'),('/reference:"'+(Join-Path $statBuild 'Mono.Cecil.dll')+'"'),('/reference:"'+(Join-Path $statBuild 'Newtonsoft.Json.dll')+'"'),('"'+(Join-Path $statRoot 'tests/BrassPatch/Statistics/StatisticsPreservedRuntimeChecks.cs')+'"'))|Set-Content -LiteralPath (Join-Path $statBuild 'preservation.rsp') -Encoding utf8
& $statCompiler ('@'+(Join-Path $statBuild 'preservation.rsp'))
if($LASTEXITCODE -ne 0){throw 'Preservation checker compilation failed.'}
& $statCheck (Join-Path $statBase 'Managed') (Join-Path $statBuild 'Managed') (Join-Path $statBuild 'preserved-runtime.json') statistics
if($LASTEXITCODE -ne 0){throw 'A method outside Statistics changed.'}
$statPreserved=Read-StatJson (Join-Path $statBuild 'preserved-runtime.json')
$statReports=@{}
foreach($name in 'statistics','tracker','presentation','link','menu'){
 $report=Read-StatJson (Join-Path $statBuild ($name+'-native.json'))
 if(-not $report.passed -or @($report.nativeErrors|Where-Object {$_}).Count){throw ('Native suite failed: '+$name)}
 foreach($file in $statCandidate.files){if(@($report.testedFiles|Where-Object {$_.name -eq $file.name -and $_.sha256 -eq $file.sha256}).Count -ne 1){throw 'Native evidence belongs to another candidate.'}}
 $statReports[$name]=$report
}
foreach($id in 'SD01','SD02','SD03','SD04','SD05','SD06','SD07','SD08','SD09','SD10','SD11','N1','N2','N3','N4','N5','N6','N7'){
 if(-not @($statReports.statistics.cases|Where-Object {$_.passed -and $_.name.Contains($id)}).Count){throw ('Required regression missing: '+$id)}
}
Add-Type -AssemblyName System.Drawing
$statVisualCases=@()
foreach($resolution in '1920x1200','1920x1080'){
 foreach($name in @("statistics-${resolution}-0-larger.png","statistics-light-${resolution}-resources-Market-default.png","statistics-light-${resolution}-resources-Merchants-default.png","statistics-light-${resolution}-efficiency-0-actions.png")){
 $path=Join-Path $statRoot ('artifacts/card-tracker-player/artifacts/patch-build/'+$name)
 $bitmap=[Drawing.Bitmap]::FromFile($path)
 try{$pixel=$bitmap.GetPixel(1400,60)}finally{$bitmap.Dispose()}
 $pass=$pixel.R -gt 210 -and $pixel.G -gt 205 -and $pixel.B -gt 190
 if(-not $pass){throw ('Light surface is missing in native capture: '+$name)}
 $statVisualCases+=@{name=$name;passed=$pass;sha256=(Get-FileHash -LiteralPath $path).Hash;surfaceRgb=@($pixel.R,$pixel.G,$pixel.B)}
 }
}
[ordered]@{passed=$true;cases=$statVisualCases;definition='Native screenshot surface pixels; numerical/interaction layout assertions are in statistics-native.json. Human visual review covers page and detail captures.'}|ConvertTo-Json -Depth 7|Set-Content -LiteralPath (Join-Path $statBuild 'visual-checks.json') -Encoding utf8
$statRetained=[ordered]@{release=$statCandidate.baseVersion;validationSha256=(Get-FileHash -LiteralPath (Join-Path $statBase 'validation.json')).Hash;unchangedFiles=@($statCandidate.files|Where-Object name -in 'Assembly-CSharp.dll','BrassLegacyAI.dll');preservedExistingMethods=($statPreserved.assemblies|Measure-Object unchangedMethods -Sum).Sum;scope='Game and AI binaries match qualified 1.4.20 exactly. Every helper method outside Statistics is unchanged, including Card Tracker, menu, animations and link replay. Five native UI suites rerun.'}
$statAi=$statOld.localAi;$statAi.files=$statCandidate.files;$statAi.sourcePins=$statCandidate.sourcePins
$statAi|Add-Member -NotePropertyName retainedFrom -NotePropertyValue $statCandidate.baseVersion -Force
$statAi|Add-Member -NotePropertyName retainedEvidence -NotePropertyValue $statRetained -Force
$statAi|ConvertTo-Json -Depth 28|Set-Content -LiteralPath (Join-Path $statBuild 'local-ai-qualification.json') -Encoding utf8
[ordered]@{passed=$true;validatedUtc=[DateTime]::UtcNow.ToString('o');files=$statCandidate.files;statistics=$statReports.statistics;cardTracker=$statReports.tracker;presentation=$statReports.presentation;linkReplay=$statReports.link;menu=$statReports.menu;preservedRuntime=$statPreserved;visualChecks=(Read-StatJson (Join-Path $statBuild 'visual-checks.json'));localAi=$statAi;retainedEvidence=$statRetained;sourceScope=$statModified;retiredSources=$statCandidate.retiredSources;evidence=@($statReports.Keys|Sort-Object|ForEach-Object{@{name=$_;sha256=(Get-FileHash -LiteralPath (Join-Path $statBuild ($_+'-native.json'))).Hash}})}|ConvertTo-Json -Depth 30|Set-Content -LiteralPath (Join-Path $statBuild 'validation.json') -Encoding utf8
. (Join-Path $statRoot 'scripts/patch-install-common.ps1')
Assert-BrassCandidate $statCandidate (Read-StatJson (Join-Path $statBuild 'validation.json')) $statBuild
Write-Output ('Qualified statistics light theme 1.4.21; '+$statRetained.preservedExistingMethods+' existing method bodies preserved.')
