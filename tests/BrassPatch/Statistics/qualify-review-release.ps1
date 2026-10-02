$ErrorActionPreference='Stop'
$reviewRoot=Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
$reviewBuild=Join-Path $reviewRoot 'artifacts/statistics-review-build'
function Read-ReviewJson([string]$path){Get-Content -LiteralPath $path -Raw|ConvertFrom-Json}
$reviewCandidate=Read-ReviewJson (Join-Path $reviewBuild 'build-manifest.json')
$reviewBase=Join-Path $reviewRoot ('artifacts/patch-releases/'+$reviewCandidate.baseVersion)
$reviewHelperBase=Join-Path $reviewRoot ('artifacts/patch-releases/'+$reviewCandidate.helperSourceBaseVersion)
$reviewPrevious=Read-ReviewJson (Join-Path $reviewBase 'build-manifest.json')
$reviewOld=Read-ReviewJson (Join-Path $reviewBase 'validation.json')
if($reviewCandidate.version -ne '1.4.17' -or $reviewPrevious.version -ne '1.4.16' -or -not $reviewOld.passed){throw 'Accepted 1.4.16 and statistics 1.4.17 are required.'}
$reviewBaselinePins=@{}
foreach($archive in $reviewHelperBase,$reviewBase){
 $inventory=Read-ReviewJson (Join-Path $archive 'archive-manifest.json')
 foreach($pin in $inventory.files){if((Get-FileHash -LiteralPath (Join-Path $archive $pin.path)).Hash -ne $pin.sha256){throw 'Accepted archive changed.'}}
 $manifest=Read-ReviewJson (Join-Path $archive 'build-manifest.json')
 foreach($pin in $manifest.sourcePins){$reviewBaselinePins[$pin.path]=$pin.sha256}
}
foreach($file in $reviewCandidate.files){if((Get-FileHash -LiteralPath (Join-Path $reviewBuild ('Managed/'+$file.name))).Hash -ne $file.sha256){throw 'Candidate changed.'}}
foreach($name in 'Assembly-CSharp.dll','BrassLegacyAI.dll'){if(($reviewCandidate.files|Where-Object name -eq $name).sha256 -ne ($reviewPrevious.files|Where-Object name -eq $name).sha256){throw 'An accepted runtime binary changed.'}}
$reviewModified=@()
foreach($pin in $reviewCandidate.sourcePins){
 if((Get-FileHash -LiteralPath (Join-Path $reviewBuild ('runtime/'+$pin.path))).Hash -ne $pin.sha256){throw 'Pinned compiled source changed.'}
 if($reviewBaselinePins[$pin.path] -ne $pin.sha256){
  if($pin.path -notlike 'src/BrassPatch/Statistics/*'){throw ('Unrelated runtime source changed: '+$pin.path)}
  if((Get-FileHash -LiteralPath (Join-Path $reviewRoot $pin.path)).Hash -ne $pin.sha256){throw 'Working statistics source differs from compiled source.'}
  $reviewModified+=$pin.path
 }
}
foreach($path in $reviewBaselinePins.Keys){if(@($reviewCandidate.sourcePins|Where-Object path -eq $path).Count -ne 1){throw ('Baseline source pin missing: '+$path)}}
$reviewCompiler=Join-Path $reviewRoot 'artifacts/patch-tools/microsoft.net.compilers.toolset.4.12.0/tasks/net472/csc.exe'
Copy-Item -LiteralPath (Join-Path $reviewRoot 'artifacts/patch-tools/mono.cecil.0.11.6/lib/net40/Mono.Cecil.dll') -Destination $reviewBuild -Force
Copy-Item -LiteralPath (Join-Path $reviewRoot 'BrassBirminghamUnity_Data/Managed/Newtonsoft.Json.dll') -Destination $reviewBuild -Force
$reviewPreservation=Join-Path $reviewBuild 'StatisticsPreservedRuntimeChecks.exe'
@('/nologo','/target:exe','/langversion:7.3','/deterministic+',('/out:"'+$reviewPreservation+'"'),('/reference:"'+(Join-Path $reviewBuild 'Mono.Cecil.dll')+'"'),('/reference:"'+(Join-Path $reviewBuild 'Newtonsoft.Json.dll')+'"'),('"'+(Join-Path $PSScriptRoot 'StatisticsPreservedRuntimeChecks.cs')+'"'))|Set-Content -LiteralPath (Join-Path $reviewBuild 'preservation.rsp') -Encoding utf8
& $reviewCompiler ('@'+(Join-Path $reviewBuild 'preservation.rsp'))
if($LASTEXITCODE -ne 0){throw 'Preservation checks failed to compile.'}
& $reviewPreservation (Join-Path $reviewBase 'Managed') (Join-Path $reviewBuild 'Managed') (Join-Path $reviewBuild 'preserved-runtime.json') statistics
if($LASTEXITCODE -ne 0){throw 'An unrelated method body changed.'}
$reviewPreserved=Read-ReviewJson (Join-Path $reviewBuild 'preserved-runtime.json')
$reviewReports=@{}
foreach($name in 'statistics','tracker','presentation','link','menu'){
 $report=Read-ReviewJson (Join-Path $reviewBuild ($name+'-native.json'))
 if(-not $report.passed -or @($report.nativeErrors|Where-Object {$_}).Count){throw ('Failed native qualification: '+$name)}
 foreach($file in $reviewCandidate.files){if(@($report.testedFiles|Where-Object {$_.name -eq $file.name -and $_.sha256 -eq $file.sha256}).Count -ne 1){throw 'Regression evidence is for another build.'}}
 $reviewReports[$name]=$report
}
foreach($name in 'signed gap including opening and penalties reconciles to 23 − 21','outgoing matrix includes selected supplier and both consumers','incoming source bars partition the same five matrix units','outgoing source bars and evidence retain both consumers','outgoing headlines and source ledger match the exact flow','selected supplier uses one outgoing unit rather than three incoming units','source industry and region filters retain opponent consumers','consuming action filter applies to both resource plots','Rail resource scope does not exclude a source built in Canal','loan recovery uses pre-loan level, not marker or cash','missing history suppresses detailed charts and unsupported advice','Match review opens first without changing the original page IDs','resource and direction controls use typed scopes','review drill opens exact record evidence in whole-match scope','each player gets every actual payday, including zero payments','action spending independently includes base and market payments once'){
 if(-not @($reviewReports.statistics.cases|Where-Object {$_.name -eq $name -and $_.passed}).Count){throw ('Required acceptance case missing: '+$name)}
}
$reviewRetained=[ordered]@{release='1.4.16';validationSha256=(Get-FileHash -LiteralPath (Join-Path $reviewBase 'validation.json')).Hash;unchangedFiles=@($reviewCandidate.files|Where-Object name -in 'Assembly-CSharp.dll','BrassLegacyAI.dll');preservedExistingMethods=($reviewPreserved.assemblies|Measure-Object unchangedMethods -Sum).Sum;scope='Game and AI DLLs are byte-identical to accepted 1.4.16. All helper method bodies outside Statistics are verified unchanged. Rules, persistence, Card Tracker, city styling, replay, AI search budgets and the two-worker game policy are retained. All five native UI suites were rerun on this candidate.'}
$reviewAi=$reviewOld.localAi
$reviewAi.files=$reviewCandidate.files;$reviewAi.sourcePins=$reviewCandidate.sourcePins
$reviewAi|Add-Member -NotePropertyName retainedFrom -NotePropertyValue '1.4.16' -Force
$reviewAi|Add-Member -NotePropertyName retainedEvidence -NotePropertyValue $reviewRetained -Force
$reviewAi|ConvertTo-Json -Depth 26|Set-Content -LiteralPath (Join-Path $reviewBuild 'local-ai-qualification.json') -Encoding utf8
$reviewValidation=[ordered]@{passed=$true;validatedUtc=[DateTime]::UtcNow.ToString('o');files=$reviewCandidate.files;statistics=$reviewReports.statistics;cardTracker=$reviewReports.tracker;presentation=$reviewReports.presentation;linkReplay=$reviewReports.link;menu=$reviewReports.menu;preservedRuntime=$reviewPreserved;localAi=$reviewAi;retainedEvidence=$reviewRetained;sourceScope=$reviewModified;evidence=@($reviewReports.Keys|Sort-Object|ForEach-Object{@{name=$_;sha256=(Get-FileHash -LiteralPath (Join-Path $reviewBuild ($_+'-native.json'))).Hash}})}
$reviewValidation|ConvertTo-Json -Depth 30|Set-Content -LiteralPath (Join-Path $reviewBuild 'validation.json') -Encoding utf8
. (Join-Path $reviewRoot 'scripts/patch-install-common.ps1')
Assert-BrassCandidate $reviewCandidate (Read-ReviewJson (Join-Path $reviewBuild 'validation.json')) $reviewBuild
Write-Output ('Qualified statistics 1.4.17: '+$reviewReports.statistics.checks+' statistics checks; '+$reviewRetained.preservedExistingMethods+' existing method bodies preserved.')
