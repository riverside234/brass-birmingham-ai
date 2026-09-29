param([string]$Root=(Split-Path -Parent $PSScriptRoot))
$ErrorActionPreference='Stop'
$build=Join-Path $Root 'artifacts/patch-build'
$managed=Join-Path $build 'Managed'
$compiler=Join-Path $Root 'artifacts/patch-tools/microsoft.net.compilers.toolset.4.12.0/tasks/net472/csc.exe'
$helper=Join-Path $managed 'BrassLegacyAI.dll'
$references=@(Get-ChildItem -LiteralPath $managed -Filter '*.dll' | Where-Object Name -notin @('BrassLegacyAI.dll') | Where-Object { $_.Name -notmatch 'Tests|Regression' } | ForEach-Object {'/reference:"'+$_.FullName+'"'})
$sources=@(Get-ChildItem -LiteralPath (Join-Path $Root 'src/BrassLocalAI') -Filter '*.cs' -Recurse | ForEach-Object {'"'+$_.FullName+'"'})
$resources=@(foreach($key in 'balanced','rail-tempo','cotton','goods-beer','supplier','pottery-flex'){foreach($version in 'v2','v3'){
 '/resource:"'+(Join-Path $Root ('configs/legacy-ai/'+$key+'-'+$version+'.json'))+'",BrassLegacyAI.Profiles.'+$key+'-'+$version+'.json'
}})
$resources+=@(foreach($key in 'balanced','rail-tempo','cotton','goods-beer','supplier','pottery-flex'){
 '/resource:"'+(Join-Path $Root ('configs/legacy-ai/archive/1.4.0/'+$key+'-v3.json'))+'",BrassLegacyAI.Profiles.'+$key+'-v3-1.4.0.json'
})
$rsp=Join-Path $build 'local-ai.rsp'
@('/nologo','/target:library','/langversion:7.3','/nostdlib+','/deterministic+','/optimize+',('/out:"'+$helper+'"'))+$references+$sources+$resources | Set-Content -LiteralPath $rsp -Encoding UTF8
& $compiler ('@'+$rsp)
if($LASTEXITCODE -ne 0){throw 'Local AI compilation failed'}
$base=Join-Path $managed 'Assembly-CSharp.dll'
$staged=Join-Path $managed 'Assembly-CSharp.local-ai.dll'
& (Join-Path $build 'BrassPatch.Weaver.exe') '--overlay' $base $helper (Join-Path $Root 'src/BrassLocalAI/Unity') $staged ((Get-FileHash -LiteralPath $base).Hash)
if($LASTEXITCODE -ne 0){throw 'Local AI weaving failed'}
Copy-Item -LiteralPath $staged -Destination $base -Force
Copy-Item -LiteralPath ($staged+'.manifest.json') -Destination ($base+'.manifest.json') -Force
Remove-Item -LiteralPath $staged,($staged+'.manifest.json')
Write-Output 'Built local AI integration with embedded, versioned profiles.'
