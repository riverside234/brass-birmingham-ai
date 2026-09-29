param([string]$Managed,[string]$Output)
$ErrorActionPreference='Stop'
$root=Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
if(-not $Managed){$Managed=Join-Path $root 'artifacts/patch-build/Managed'}
if(-not $Output){$Output=Join-Path $root 'artifacts/patch-build/local-ai-search-gameplay.json'}
$testedFiles=@(foreach($name in @('Assembly-CSharp.dll','BrassPatch.dll','BrassLegacyAI.dll')){[pscustomobject]@{name=$name;sha256=(Get-FileHash -LiteralPath (Join-Path $Managed $name)).Hash}})
@{passed=$false;status='running';testedFiles=$testedFiles}|ConvertTo-Json -Depth 10|Set-Content -LiteralPath $Output -Encoding UTF8
$exe=Join-Path $Managed 'SearchGameplayTests.exe'
$rsp=Join-Path (Split-Path -Parent $Output) 'search-gameplay-tests.rsp'
@('/nologo','/target:exe','/langversion:7.3','/nostdlib+','/deterministic+',('/out:"'+$exe+'"'))+@(Get-ChildItem -LiteralPath $Managed -Filter '*.dll'|Where-Object {$_.Name -notmatch 'Tests|Regression'}|ForEach-Object{'/reference:"'+$_.FullName+'"'})+@('"'+(Join-Path $PSScriptRoot 'SearchGameplayTests.cs')+'"')|Set-Content -LiteralPath $rsp -Encoding UTF8
& (Join-Path $root 'artifacts/patch-tools/microsoft.net.compilers.toolset.4.12.0/tasks/net472/csc.exe') ('@'+$rsp)
if($LASTEXITCODE -ne 0){throw 'Search gameplay test compilation failed'}
& $exe $Output
if($LASTEXITCODE -ne 0){
 @{passed=$false;status='process-failed'}|ConvertTo-Json|Set-Content -LiteralPath $Output -Encoding UTF8
 throw 'Search gameplay tests failed'
}
$result=Get-Content -LiteralPath $Output -Raw|ConvertFrom-Json
if(-not $result.passed){throw 'Search gameplay tests produced no fresh success'}
foreach($file in $testedFiles){if((Get-FileHash -LiteralPath (Join-Path $Managed $file.name)).Hash -ne $file.sha256){
 @{passed=$false;status='candidate-changed'}|ConvertTo-Json|Set-Content -LiteralPath $Output -Encoding UTF8
 throw 'Search gameplay candidate changed during testing'
}}
$result|Add-Member -NotePropertyName testedFiles -NotePropertyValue $testedFiles -Force
$result|ConvertTo-Json -Depth 20|Set-Content -LiteralPath $Output -Encoding UTF8
