param([string]$GameRoot=(Split-Path -Parent $PSScriptRoot))
$ErrorActionPreference='Stop'
$repo=Split-Path -Parent $PSScriptRoot
$toolsDir=Join-Path $repo 'artifacts/patch-tools'
$out=Join-Path $repo 'artifacts/patch-build'
$managed=Join-Path $GameRoot 'BrassBirminghamUnity_Data/Managed'
$stock=Join-Path $repo 'artifacts/patch-baseline/Assembly-CSharp.dll'
$expected='600EBF309670B2726001C16A3375F8BBF3DE76A23C0202BE7CF198BA338981FB'
New-Item -ItemType Directory -Force -Path $toolsDir,$out,(Join-Path $out 'Managed'),(Split-Path $stock) | Out-Null
if(-not (Test-Path -LiteralPath $stock)) {
    $installed=Join-Path $managed 'Assembly-CSharp.dll'
    if((Get-FileHash -LiteralPath $installed).Hash -ne $expected){throw 'A verified stock assembly is required for the first build.'}
    Copy-Item -LiteralPath $installed -Destination $stock
}
if((Get-FileHash -LiteralPath $stock).Hash -ne $expected){throw 'Stock backup fingerprint mismatch.'}
Add-Type -AssemblyName System.IO.Compression.FileSystem
$packages=@(
 @{Name='mono.cecil';Version='0.11.6';Hash='D2A23832AAA948BA9A01ACC42B5726E34C5F995958F1B30D45C0E7C70B3A72D5'},
 @{Name='microsoft.net.compilers.toolset';Version='4.12.0';Hash='FE24EF31A6FFCB7C49383D2FD362763DEE291AD9B9D98CC0C19EF80203B99EBC'}
)
foreach($package in $packages){
    $id=$package.Name+'.'+$package.Version
    $archive=Join-Path $toolsDir ($id+'.nupkg')
    if(-not(Test-Path -LiteralPath $archive)){
        Invoke-WebRequest -Uri ('https://api.nuget.org/v3-flatcontainer/'+$package.Name+'/'+$package.Version+'/'+$id+'.nupkg') -OutFile $archive
    }
    if((Get-FileHash -LiteralPath $archive).Hash -ne $package.Hash){throw "Tool package hash mismatch: $id"}
    $dest=Join-Path $toolsDir $id
    if(-not(Test-Path -LiteralPath $dest)){[IO.Compression.ZipFile]::ExtractToDirectory($archive,$dest)}
}
$compiler=Join-Path $toolsDir 'microsoft.net.compilers.toolset.4.12.0/tasks/net472/csc.exe'
$cecil=Join-Path $toolsDir 'mono.cecil.0.11.6/lib/net40/Mono.Cecil.dll'
$references=@(Get-ChildItem -LiteralPath $managed -Filter '*.dll' | Where-Object Name -notin @('Assembly-CSharp.dll','BrassPatch.dll','BrassLegacyAI.dll') | ForEach-Object {'/reference:"'+$_.FullName+'"'})
$references+='/reference:"'+$stock+'"'
$sources=@(Get-ChildItem -LiteralPath (Join-Path $repo 'src/BrassPatch') -Recurse -Filter '*.cs' | ForEach-Object {'"'+$_.FullName+'"'})
$helper=Join-Path $out 'Managed/BrassPatch.dll'
$response=@('/nologo','/target:library','/langversion:7.3','/nostdlib+','/deterministic+','/optimize+',('/out:"'+$helper+'"'))+$references+$sources
$rsp=Join-Path $out 'helper.rsp'
$response | Set-Content -LiteralPath $rsp -Encoding UTF8
& $compiler ('@'+$rsp)
if($LASTEXITCODE -ne 0){throw 'Helper compilation failed.'}
$weaver=Join-Path $out 'BrassPatch.Weaver.exe'
$response=@('/nologo','/target:exe','/langversion:7.3','/deterministic+',('/out:"'+$weaver+'"'),('/reference:"'+$cecil+'"'),('/reference:"'+(Join-Path $managed 'Newtonsoft.Json.dll')+'"'),('"'+(Join-Path $repo 'src/BrassPatch.Weaver/Program.cs')+'"'))
$rsp=Join-Path $out 'weaver.rsp'
$response|Set-Content -LiteralPath $rsp -Encoding UTF8
& $compiler ('@'+$rsp)
if($LASTEXITCODE -ne 0){throw 'Weaver compilation failed.'}
Copy-Item -LiteralPath $cecil -Destination (Join-Path $out 'Mono.Cecil.dll') -Force
Copy-Item -LiteralPath (Join-Path $managed 'Newtonsoft.Json.dll') -Destination (Join-Path $out 'Newtonsoft.Json.dll') -Force
Get-ChildItem -LiteralPath $managed -Filter '*.dll' | Where-Object Name -notin @('Assembly-CSharp.dll','BrassPatch.dll','BrassLegacyAI.dll') | Copy-Item -Destination (Join-Path $out 'Managed') -Force
& $weaver $stock $helper (Join-Path $repo 'src/BrassPatch') (Join-Path $out 'Managed/Assembly-CSharp.dll')
if($LASTEXITCODE -ne 0){throw 'Weaving failed. Installed game was not changed.'}
& (Join-Path $PSScriptRoot 'build-local-ai.ps1') -Root $repo
if($LASTEXITCODE -ne 0){throw 'Local AI build failed'}
$manifest=[ordered]@{version='1.4.4';stockSha256=$expected;files=@(Get-Item $helper,(Join-Path $out 'Managed/BrassLegacyAI.dll'),(Join-Path $out 'Managed/Assembly-CSharp.dll')|ForEach-Object{[ordered]@{name=$_.Name;sha256=(Get-FileHash -LiteralPath $_.FullName).Hash}})}
$manifest|ConvertTo-Json -Depth 5|Set-Content -LiteralPath (Join-Path $out 'build-manifest.json') -Encoding UTF8
Write-Output "Candidate patch built in $out. The installed game has not been modified."
