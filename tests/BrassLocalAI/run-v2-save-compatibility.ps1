param([ValidateSet('1.3.2','1.4.0','1.4.1','1.4.3')][string]$PreviousRelease='1.3.2')
$ErrorActionPreference='Stop'
$root=Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$build=Join-Path $root 'artifacts/patch-build'
$managed=Join-Path $build 'Managed'
$manifest=Get-Content -LiteralPath (Join-Path $build 'build-manifest.json') -Raw|ConvertFrom-Json
$prior=Join-Path $root ('artifacts/patch-releases/'+$PreviousRelease)
$old=Get-Content -LiteralPath (Join-Path $prior 'build-manifest.json') -Raw|ConvertFrom-Json
$fixture=Join-Path $root ('artifacts/local-ai-save-compatibility/'+[Guid]::NewGuid().ToString('N'))
$v2=Join-Path $fixture 'v2';$v3=Join-Path $fixture 'v3';$saves=Join-Path $fixture 'saves'
New-Item -ItemType Directory -Path $v2,$v3,$saves|Out-Null
$output=Join-Path $build $(if($PreviousRelease -eq '1.3.2'){'local-ai-v2-save-compatibility.json'}elseif($PreviousRelease -eq '1.4.0'){'local-ai-v3-save-compatibility.json'}else{'local-ai-'+$PreviousRelease+'-save-compatibility.json'})
@{passed=$false;status='running'}|ConvertTo-Json|Set-Content -LiteralPath $output
foreach($dir in $v2,$v3){Get-ChildItem -LiteralPath $managed -Filter '*.dll'|Copy-Item -Destination $dir}
foreach($file in $old.files){if((Get-FileHash -LiteralPath (Join-Path $prior $file.name)).Hash -ne $file.sha256){throw 'Archive changed'};Copy-Item -LiteralPath (Join-Path $prior $file.name) -Destination $v2 -Force}
foreach($pair in @(@($v2,'V2SaveFixture'),@($v3,'SaveCompatibilityTests'))){
 $dir=$pair[0];$class=$pair[1];$exe=Join-Path $dir ($class+'.exe');$rsp=Join-Path $dir 'test.rsp'
 @('/nologo','/target:exe','/langversion:7.3','/nostdlib+','/deterministic+',('/out:"'+$exe+'"'))+@(Get-ChildItem -LiteralPath $dir -Filter '*.dll'|Where-Object {$_.Name -notmatch 'Tests|Regression'}|ForEach-Object {'/reference:"'+$_.FullName+'"'})+@('"'+(Join-Path $PSScriptRoot ($class+'.cs'))+'"')|Set-Content -LiteralPath $rsp -Encoding UTF8
 & (Join-Path $root 'artifacts/patch-tools/microsoft.net.compilers.toolset.4.12.0/tasks/net472/csc.exe') ('@'+$rsp)
 if($LASTEXITCODE -ne 0){throw 'Compatibility test compilation failed'}
}
$index=Join-Path $fixture 'v2-fixtures.json';$raw=Join-Path $fixture 'result.json'
& (Join-Path $v2 'V2SaveFixture.exe') $saves $index
if($LASTEXITCODE -ne 0){throw 'V2 fixture creation failed'}
& (Join-Path $v3 'SaveCompatibilityTests.exe') $saves $index $raw
if($LASTEXITCODE -ne 0){throw 'V3 save compatibility failed'}
$result=Get-Content -LiteralPath $raw -Raw|ConvertFrom-Json
if(-not $result.passed){throw 'Compatibility did not pass'}
foreach($file in $manifest.files){foreach($dir in $managed,$v3){if((Get-FileHash -LiteralPath (Join-Path $dir $file.name)).Hash -ne $file.sha256){throw 'Candidate changed'}}}
$result|Add-Member -NotePropertyName testedFiles -NotePropertyValue $manifest.files
$result|Add-Member -NotePropertyName producerFiles -NotePropertyValue $old.files
$result|ConvertTo-Json -Depth 15|Set-Content -LiteralPath $output -Encoding UTF8
Write-Output ('V2 save compatibility passed: '+$result.checks+' assertions.')
