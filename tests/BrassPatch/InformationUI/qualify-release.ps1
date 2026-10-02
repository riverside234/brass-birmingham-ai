$ErrorActionPreference='Stop'
$uiRoot=Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
$uiBuild=Join-Path $uiRoot 'artifacts/information-ui-build'
function Read-UiJson([string]$path){Get-Content -LiteralPath $path -Raw|ConvertFrom-Json}
$uiCandidate=Read-UiJson (Join-Path $uiBuild 'build-manifest.json')
$uiBase=Join-Path $uiRoot ('artifacts/patch-releases/'+$uiCandidate.baseVersion)
$uiPrevious=Read-UiJson (Join-Path $uiBase 'build-manifest.json')
$uiOld=Read-UiJson (Join-Path $uiBase 'validation.json')
if($uiCandidate.version -ne '1.4.19' -or $uiPrevious.version -ne '1.4.18' -or $uiCandidate.helperSourceBaseVersion -ne '1.4.17' -or -not $uiOld.passed){throw 'Qualified UI and AI baselines required.'}
$uiInventory=Read-UiJson (Join-Path $uiBase 'archive-manifest.json')
foreach($pin in $uiInventory.files){
 $path=[IO.Path]::GetFullPath((Join-Path $uiBase $pin.path))
 if(-not $path.StartsWith([IO.Path]::GetFullPath($uiBase)+[IO.Path]::DirectorySeparatorChar,[StringComparison]::OrdinalIgnoreCase) -or (Get-FileHash -LiteralPath $path).Hash -ne $pin.sha256){throw 'Accepted archive changed.'}
}
foreach($file in $uiCandidate.files){if((Get-FileHash -LiteralPath (Join-Path $uiBuild ('Managed/'+$file.name))).Hash -ne $file.sha256){throw 'Candidate changed.'}}
foreach($name in 'Assembly-CSharp.dll','BrassLegacyAI.dll'){if(($uiCandidate.files|Where-Object name -eq $name).sha256 -ne ($uiPrevious.files|Where-Object name -eq $name).sha256){throw 'Accepted game or AI binary changed.'}}
$uiBaselinePins=@{}
foreach($version in '1.4.17','1.4.18'){
 $archive=Join-Path $uiRoot ('artifacts/patch-releases/'+$version)
 $inventory=Read-UiJson (Join-Path $archive 'archive-manifest.json')
 foreach($pin in $inventory.files){if((Get-FileHash -LiteralPath (Join-Path $archive $pin.path)).Hash -ne $pin.sha256){throw 'Accepted source archive changed.'}}
 $sourceManifest=Read-UiJson (Join-Path $archive 'build-manifest.json')
 foreach($pin in $sourceManifest.sourcePins){$uiBaselinePins[$pin.path]=$pin.sha256}
}
$uiScope=@('src/BrassPatch/CardTracker/CardTrackerOverlay.cs','src/BrassPatch/Statistics/StatisticsEntryPoints.cs','src/BrassPatch/Statistics/StatisticsEntrySurface.cs','src/BrassPatch/Statistics/StatisticsPanel.cs')
$uiModified=@()
foreach($pin in $uiCandidate.sourcePins){
 if((Get-FileHash -LiteralPath (Join-Path $uiBuild ('runtime/'+$pin.path))).Hash -ne $pin.sha256){throw 'Compiled source snapshot changed.'}
 if($uiBaselinePins[$pin.path] -ne $pin.sha256){
  if($pin.path -notin $uiScope -or (Get-FileHash -LiteralPath (Join-Path $uiRoot $pin.path)).Hash -ne $pin.sha256){throw ('Unexpected runtime source change: '+$pin.path)}
  $uiModified+=$pin.path
 }
}
foreach($path in $uiBaselinePins.Keys){if(@($uiCandidate.sourcePins|Where-Object path -eq $path).Count -ne 1){throw ('Baseline source missing: '+$path)}}
$uiCompiler=Join-Path $uiRoot 'artifacts/patch-tools/microsoft.net.compilers.toolset.4.12.0/tasks/net472/csc.exe'
Copy-Item -LiteralPath (Join-Path $uiRoot 'artifacts/patch-tools/mono.cecil.0.11.6/lib/net40/Mono.Cecil.dll') -Destination $uiBuild -Force
Copy-Item -LiteralPath (Join-Path $uiRoot 'BrassBirminghamUnity_Data/Managed/Newtonsoft.Json.dll') -Destination $uiBuild -Force
$uiPreservation=Join-Path $uiBuild 'StatisticsPreservedRuntimeChecks.exe'
@('/nologo','/target:exe','/langversion:7.3','/deterministic+',('/out:"'+$uiPreservation+'"'),('/reference:"'+(Join-Path $uiBuild 'Mono.Cecil.dll')+'"'),('/reference:"'+(Join-Path $uiBuild 'Newtonsoft.Json.dll')+'"'),('"'+(Join-Path $uiRoot 'tests/BrassPatch/Statistics/StatisticsPreservedRuntimeChecks.cs')+'"'))|Set-Content -LiteralPath (Join-Path $uiBuild 'preservation.rsp') -Encoding utf8
& $uiCompiler ('@'+(Join-Path $uiBuild 'preservation.rsp'))
if($LASTEXITCODE -ne 0){throw 'Preservation checks did not compile.'}
& $uiPreservation (Join-Path $uiBase 'Managed') (Join-Path $uiBuild 'Managed') (Join-Path $uiBuild 'preserved-runtime.json') information-ui
if($LASTEXITCODE -ne 0){throw 'An unrelated method body changed.'}
$uiPreserved=Read-UiJson (Join-Path $uiBuild 'preserved-runtime.json')
$uiReports=@{}
foreach($name in 'statistics','tracker','presentation','link','menu'){
 $report=Read-UiJson (Join-Path $uiBuild ($name+'-native.json'))
 if(-not $report.passed -or @($report.nativeErrors|Where-Object {$_}).Count){throw ('Failed native qualification: '+$name)}
 foreach($file in $uiCandidate.files){if(@($report.testedFiles|Where-Object {$_.name -eq $file.name -and $_.sha256 -eq $file.sha256}).Count -ne 1){throw 'Native evidence belongs to another build.'}}
 $uiReports[$name]=$report
}
foreach($name in 'tracker stays fixed during another player''s market animation','tracker occupies fixed bottom-right position above action band','fixed tracker button clears the native action controls','tracker defaults to the second text size','tracker toolbar uses compact physical dimensions','wild cards are explicitly labeled with both card types','wild holder counts explain possession, rather than cards played','full wild-card type labels fit the compact strip','largest text retains complete wild-card labels','raised and rotated hand moves only the legend, not tracker button'){
 if(-not @($uiReports.tracker.cases|Where-Object {$_.name -eq $name -and $_.passed}).Count){throw ('Tracker acceptance case missing: '+$name)}
}
foreach($name in 'statistics defaults to the second text size','statistics entry sits below original results controls without overlap','light results entry stays within screen with readable fixed size','statistics entry uses light brass surface and dark readable text','results statistics control remains fully visible and interactive','board statistics button is hidden behind original results modal','board statistics entry returns in the bottom row beside original summary button'){
 if(-not @($uiReports.statistics.cases|Where-Object {$_.name -eq $name -and $_.passed}).Count){throw ('Statistics acceptance case missing: '+$name)}
}
Add-Type -AssemblyName System.Drawing
$uiVisualCases=@()
foreach($layout in $uiReports.statistics.layouts){
 $name='statistics-results-entry-'+$layout.width+'x'+$layout.height+'.png'
 $imagePath=Join-Path $uiRoot ('artifacts/card-tracker-player/artifacts/patch-build/'+$name)
 $bounds=@($uiReports.statistics.cases|Where-Object {$_.name -eq 'statistics entry sits below original results controls without overlap' -and $_.detail.y -eq $layout.height})[0].detail
 $bitmap=[Drawing.Bitmap]::FromFile($imagePath)
 try{$pixel=$bitmap.GetPixel([int]($bounds.xMin+$bounds.width/2),[int]($layout.height-$bounds.yMin-$bounds.height+8))}finally{$bitmap.Dispose()}
 $pass=$pixel.R -ge 200 -and $pixel.G -ge 180 -and $pixel.B -ge 115
 if(-not $pass){throw ('Statistics entry is dark in its native capture: '+$name)}
 $uiVisualCases+=@{name=$name;passed=$pass;sha256=(Get-FileHash -LiteralPath $imagePath).Hash;topSurfaceRgb=@($pixel.R,$pixel.G,$pixel.B)}
}
[ordered]@{passed=$true;cases=$uiVisualCases;definition='Encoded native screenshot pixels sampled inside the light button surface, outside text and border.'}|ConvertTo-Json -Depth 6|Set-Content -LiteralPath (Join-Path $uiBuild 'visual-checks.json') -Encoding utf8
$uiRetained=[ordered]@{release=$uiCandidate.baseVersion;validationSha256=(Get-FileHash -LiteralPath (Join-Path $uiBase 'validation.json')).Hash;unchangedFiles=@($uiCandidate.files|Where-Object name -in 'Assembly-CSharp.dll','BrassLegacyAI.dll');preservedExistingMethods=($uiPreserved.assemblies|Measure-Object unchangedMethods -Sum).Sum;scope='Game and AI binaries are byte-identical to accepted 1.4.18. Every helper method outside the four revised UI source files is verified unchanged. Five native UI suites were rerun on this build.'}
$uiAi=$uiOld.localAi
$uiAi.files=$uiCandidate.files;$uiAi.sourcePins=$uiCandidate.sourcePins
$uiAi|Add-Member -NotePropertyName retainedFrom -NotePropertyValue $uiCandidate.baseVersion -Force
$uiAi|Add-Member -NotePropertyName retainedEvidence -NotePropertyValue $uiRetained -Force
$uiAi|ConvertTo-Json -Depth 28|Set-Content -LiteralPath (Join-Path $uiBuild 'local-ai-qualification.json') -Encoding utf8
[ordered]@{passed=$true;validatedUtc=[DateTime]::UtcNow.ToString('o');files=$uiCandidate.files;statistics=$uiReports.statistics;cardTracker=$uiReports.tracker;presentation=$uiReports.presentation;linkReplay=$uiReports.link;menu=$uiReports.menu;preservedRuntime=$uiPreserved;visualChecks=(Read-UiJson (Join-Path $uiBuild 'visual-checks.json'));localAi=$uiAi;retainedEvidence=$uiRetained;sourceScope=$uiModified;evidence=@($uiReports.Keys|Sort-Object|ForEach-Object{@{name=$_;sha256=(Get-FileHash -LiteralPath (Join-Path $uiBuild ($_+'-native.json'))).Hash}})}|ConvertTo-Json -Depth 30|Set-Content -LiteralPath (Join-Path $uiBuild 'validation.json') -Encoding utf8
. (Join-Path $uiRoot 'scripts/patch-install-common.ps1')
Assert-BrassCandidate $uiCandidate (Read-UiJson (Join-Path $uiBuild 'validation.json')) $uiBuild
Write-Output ('Qualified information UI 1.4.19; '+$uiRetained.preservedExistingMethods+' existing method bodies preserved.')
