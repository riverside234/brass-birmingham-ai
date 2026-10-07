param([string]$BuildDirectory='artifacts/information-ui-build')
$ErrorActionPreference='Stop'
$trackerRoot=Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
$trackerBuild=Join-Path $trackerRoot $BuildDirectory
$trackerManifest=Get-Content -LiteralPath (Join-Path $trackerBuild 'build-manifest.json') -Raw|ConvertFrom-Json
$trackerBase=Join-Path $trackerBuild ('baseline/'+$trackerManifest.baseVersion)
if(-not(Test-Path -LiteralPath $trackerBase)){$trackerBase=Join-Path $trackerRoot ('artifacts/patch-releases/'+$trackerManifest.baseVersion)}
$trackerFixture=Join-Path $trackerRoot ('artifacts/information-ui-packaging/'+[Guid]::NewGuid().ToString('N'))
$trackerFixtureGame=Join-Path $trackerFixture 'game'
$trackerFixtureManaged=Join-Path $trackerFixtureGame 'BrassBirminghamUnity_Data/Managed'
$trackerFixtureBuild=Join-Path $trackerFixture 'artifacts/information-ui-build'
$trackerFixtureScripts=Join-Path $trackerFixture 'scripts'
$trackerFixtureBaseline=Join-Path $trackerFixture 'artifacts/patch-baseline'
$trackerFixtureSaves=Join-Path $trackerFixture 'artificial-saves'
New-Item -ItemType Directory -Path $trackerFixtureManaged,(Join-Path $trackerFixtureBuild 'Managed'),$trackerFixtureScripts,$trackerFixtureBaseline,$trackerFixtureSaves -Force|Out-Null
foreach($name in 'install-combined-patch.ps1','patch-install-common.ps1'){Copy-Item -LiteralPath (Join-Path $trackerRoot ('scripts/'+$name)) -Destination $trackerFixtureScripts}
Copy-Item -LiteralPath (Join-Path $trackerRoot 'artifacts/patch-baseline/Assembly-CSharp.dll') -Destination $trackerFixtureBaseline
foreach($name in 'build-manifest.json','validation.json'){Copy-Item -LiteralPath (Join-Path $trackerBuild $name) -Destination $trackerFixtureBuild}
foreach($file in $trackerManifest.files){Copy-Item -LiteralPath (Join-Path $trackerBuild ('Managed/'+$file.name)) -Destination (Join-Path $trackerFixtureBuild 'Managed')}
$trackerRecordPath=Join-Path $trackerFixture 'artifacts/patch-installation.json'
$trackerPendingPath=Join-Path $trackerFixture 'artifacts/patch-installation.pending.json'
$trackerChecks=0
$trackerCases=@()
function Check-TrackerPackage([bool]$pass,[string]$name){$script:trackerChecks++;$script:trackerCases+=@{name=$name;passed=$pass};if(-not $pass){throw $name}}
function Reset-TrackerPackage {
 foreach($file in $trackerManifest.baseFiles){Copy-Item -LiteralPath (Join-Path $trackerBase ('Managed/'+$file.name)) -Destination (Join-Path $trackerFixtureManaged $file.name) -Force}
 [ordered]@{version=$trackerManifest.baseVersion;status='installed';gameRoot=$trackerFixtureGame;stockSha256=$trackerManifest.stockSha256;files=$trackerManifest.baseFiles;onlineEnabled=$false;saveDataPaths=@($trackerFixtureSaves)}|ConvertTo-Json -Depth 12|Set-Content -LiteralPath $trackerRecordPath -Encoding utf8
 if(Test-Path -LiteralPath $trackerPendingPath){Remove-Item -LiteralPath $trackerPendingPath}
}
function Run-TrackerPackage([bool]$success){
 $output=& pwsh -NoProfile -File (Join-Path $trackerFixtureScripts 'install-combined-patch.ps1') -GameRoot $trackerFixtureGame -BuildDirectory $trackerFixtureBuild 2>&1
 Check-TrackerPackage (($LASTEXITCODE -eq 0) -eq $success) ('installer expected success='+$success+': '+($output -join ' '))
}
function Assert-TrackerPackage($files){foreach($file in $files){Check-TrackerPackage ((Get-FileHash -LiteralPath (Join-Path $trackerFixtureManaged $file.name)).Hash -eq $file.sha256) ('published fingerprint '+$file.name)}}
Reset-TrackerPackage
Run-TrackerPackage $true;Assert-TrackerPackage $trackerManifest.files
$trackerInstalled=Get-Content -LiteralPath $trackerRecordPath -Raw|ConvertFrom-Json
Check-TrackerPackage ($trackerInstalled.version -eq $trackerManifest.version -and $trackerInstalled.status -eq 'installed' -and -not $trackerInstalled.onlineEnabled) 'new installed version and offline status'
Check-TrackerPackage ($trackerInstalled.saveDataPaths.Count -eq 1 -and $trackerInstalled.saveDataPaths[0] -eq $trackerFixtureSaves) 'configured save path preserved without reading game saves'
foreach($file in $trackerManifest.baseFiles){Check-TrackerPackage ((Get-FileHash -LiteralPath (Join-Path $trackerInstalled.previousReleaseBackup $file.name)).Hash -eq $file.sha256) ('recoverable baseline backup '+$file.name)}
Check-TrackerPackage (-not(Test-Path -LiteralPath $trackerPendingPath)) 'complete transaction removes pending record'
Run-TrackerPackage $true;Assert-TrackerPackage $trackerManifest.files
$trackerHelper=Join-Path $trackerFixtureBuild 'Managed/BrassPatch.dll'
$trackerStream=[IO.File]::OpenWrite($trackerHelper);try{$trackerStream.WriteByte(0)}finally{$trackerStream.Dispose()}
Run-TrackerPackage $false;Assert-TrackerPackage $trackerManifest.files
Copy-Item -LiteralPath (Join-Path $trackerBuild 'Managed/BrassPatch.dll') -Destination $trackerHelper -Force
$trackerValidationPath=Join-Path $trackerFixtureBuild 'validation.json'
$trackerScenarios=@('failed-tracker','wrong-fingerprint','failed-statistics','failed-preservation','failed-visual')
if($trackerManifest.version -in '1.4.23','1.4.24','1.4.25','1.4.26','1.4.27','1.4.29'){$trackerScenarios+=@('failed-history','wrong-history-limit','wrong-history-schema','wrong-history-fingerprint')}
if($trackerManifest.version -in '1.4.24','1.4.25','1.4.26','1.4.27','1.4.29'){$trackerScenarios+=@('failed-draft','gamma-enabled','wrong-player-layout')}
if($trackerManifest.version -in '1.4.25','1.4.26'){$trackerScenarios+='failed-tracker-position'}
if($trackerManifest.version -in '1.4.25','1.4.26','1.4.27','1.4.29'){$trackerScenarios+='failed-tracker-theme'}
foreach($scenario in $trackerScenarios){
 $trackerValidation=Get-Content -LiteralPath (Join-Path $trackerBuild 'validation.json') -Raw|ConvertFrom-Json
 if($scenario -eq 'failed-tracker-position'){foreach($case in $trackerValidation.cardTracker.cases|Where-Object name -Like 'CT25-POSITION*'){$case.passed=$false}}elseif($scenario -eq 'failed-tracker-theme'){foreach($case in $trackerValidation.cardTracker.cases|Where-Object name -Like 'CT25-THEME*'){$case.passed=$false}}elseif($scenario -eq 'failed-draft'){$trackerValidation.menu.selectionPanel.draftChecks=0}elseif($scenario -eq 'gamma-enabled'){$trackerValidation.menu.selectionPanel.gammaEnabled=$true}elseif($scenario -eq 'wrong-player-layout'){$trackerValidation.menu.selectionPanel.layouts[0].choices=99}elseif($scenario -eq 'failed-tracker'){$trackerValidation.cardTracker.passed=$false}elseif($scenario -eq 'failed-statistics'){$trackerValidation.statistics.passed=$false}elseif($scenario -eq 'failed-preservation'){$trackerValidation.preservedRuntime.passed=$false}elseif($scenario -eq 'failed-visual'){$trackerValidation.visualChecks.passed=$false}elseif($scenario -eq 'failed-history'){$trackerValidation.history.passed=$false}elseif($scenario -eq 'wrong-history-limit'){$trackerValidation.history.historyLimit=11}elseif($scenario -eq 'wrong-history-schema'){$trackerValidation.history.reportSchema=99}elseif($scenario -eq 'wrong-history-fingerprint'){$trackerValidation.history.testedFiles[0].sha256='unexpected'}else{$trackerValidation.cardTracker.testedFiles[0].sha256='unexpected'}
 $trackerValidation|ConvertTo-Json -Depth 30|Set-Content -LiteralPath $trackerValidationPath -Encoding utf8
 Run-TrackerPackage $false;Assert-TrackerPackage $trackerManifest.files
}
if($trackerManifest.version -in '1.4.27','1.4.29'){
 foreach($scenario in 'CT27-LAYER','CT27-STYLE','CT27-POSITION'){
  $trackerValidation=Get-Content -LiteralPath (Join-Path $trackerBuild 'validation.json') -Raw|ConvertFrom-Json
  foreach($case in $trackerValidation.cardTracker.cases|Where-Object {$_.name.StartsWith($scenario)}){$case.passed=$false}
  $trackerValidation|ConvertTo-Json -Depth 42|Set-Content -LiteralPath $trackerValidationPath -Encoding utf8
  Run-TrackerPackage $false;Assert-TrackerPackage $trackerManifest.files
 }
}
if($trackerManifest.version -eq '1.4.29'){
 foreach($suite in 'statistics','history'){
  foreach($group in 'GEO29-DATA','GEO29-UI','GEO29-TAG','GEO29-INPUT'){
   $trackerValidation=Get-Content -LiteralPath (Join-Path $trackerBuild 'validation.json') -Raw|ConvertFrom-Json
   foreach($case in $trackerValidation.$suite.cases|Where-Object {$_.name.StartsWith($group)}){$case.passed=$false}
   $trackerValidation|ConvertTo-Json -Depth 42|Set-Content -LiteralPath $trackerValidationPath -Encoding utf8
   Run-TrackerPackage $false;Assert-TrackerPackage $trackerManifest.files
  }
 }
}
if($trackerManifest.version -eq '1.4.26'){
 foreach($scenario in 'E26-DATA','E26-LIFECYCLE','E26-UI','performance'){
  $trackerValidation=Get-Content -LiteralPath (Join-Path $trackerBuild 'validation.json') -Raw|ConvertFrom-Json
  if($scenario -eq 'performance'){$trackerValidation.performance.passed=$false}else{foreach($case in $trackerValidation.statistics.cases|Where-Object {$_.name.StartsWith($scenario)}){$case.passed=$false}}
  $trackerValidation|ConvertTo-Json -Depth 30|Set-Content -LiteralPath $trackerValidationPath -Encoding utf8
  Run-TrackerPackage $false;Assert-TrackerPackage $trackerManifest.files
 }
}
Copy-Item -LiteralPath (Join-Path $trackerBuild 'validation.json') -Destination $trackerValidationPath -Force
$trackerRecord=Get-Content -LiteralPath $trackerRecordPath -Raw|ConvertFrom-Json;$trackerRecord.version='future-release'
$trackerRecord|ConvertTo-Json -Depth 16|Set-Content -LiteralPath $trackerRecordPath -Encoding utf8
Run-TrackerPackage $false;Assert-TrackerPackage $trackerManifest.files
Reset-TrackerPackage
$trackerUnknown=Join-Path $trackerFixtureManaged 'Assembly-CSharp.dll'
$trackerStream=[IO.File]::OpenWrite($trackerUnknown);try{$trackerStream.WriteByte(0)}finally{$trackerStream.Dispose()}
Run-TrackerPackage $false
foreach($file in $trackerManifest.baseFiles|Where-Object name -ne 'Assembly-CSharp.dll'){Check-TrackerPackage ((Get-FileHash -LiteralPath (Join-Path $trackerFixtureManaged $file.name)).Hash -eq $file.sha256) ('unrecognized game assembly leaves dependency '+$file.name)}
Reset-TrackerPackage
$trackerLock=[IO.File]::Open((Join-Path $trackerFixtureManaged 'Assembly-CSharp.dll'),[IO.FileMode]::Open,[IO.FileAccess]::Read,[IO.FileShare]::Read)
try{Run-TrackerPackage $false;Assert-TrackerPackage $trackerManifest.baseFiles}finally{$trackerLock.Dispose()}
Check-TrackerPackage ((Get-Content -LiteralPath $trackerRecordPath -Raw|ConvertFrom-Json).version -eq $trackerManifest.baseVersion) 'blocked replacement keeps baseline installation record'
Run-TrackerPackage $true;Assert-TrackerPackage $trackerManifest.files
Check-TrackerPackage (-not(Test-Path -LiteralPath $trackerPendingPath)) 'retry finishes retained transaction after lock release'
foreach($file in $trackerManifest.files){Check-TrackerPackage ((Get-FileHash -LiteralPath (Join-Path $trackerBuild ('Managed/'+$file.name))).Hash -eq $file.sha256) ('qualification candidate unchanged '+$file.name)}
[ordered]@{passed=$true;checks=$trackerChecks;cases=$trackerCases;fixture=$trackerFixture;testedFiles=$trackerManifest.files}|ConvertTo-Json -Depth 18|Set-Content -LiteralPath (Join-Path $trackerBuild 'packaging-results.json') -Encoding utf8
Write-Output ('Information UI packaging: '+$trackerChecks+' checks passed using artificial files only.')
