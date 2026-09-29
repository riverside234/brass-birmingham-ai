param([string]$GameRoot=(Split-Path -Parent $PSScriptRoot))
$ErrorActionPreference='Stop'
$repo=Split-Path -Parent $PSScriptRoot
$managed=Join-Path $GameRoot 'BrassBirminghamUnity_Data/Managed'
$profile=Get-Content -LiteralPath (Join-Path $repo 'configs/rules/steam-brass-patch-1.1.0.json') -Raw | ConvertFrom-Json
foreach($file in $profile.files){
    if((Get-FileHash -LiteralPath (Join-Path $GameRoot $file.path)).Hash -ne $file.sha256){throw "Rules fingerprint mismatch: $($file.path)"}
}
$dependencies=Get-Content -LiteralPath (Join-Path $repo 'configs/rules/runtime-dependencies-1.1.0.json') -Raw | ConvertFrom-Json
if(@(Get-ChildItem -LiteralPath $managed -Filter '*.dll').Count -ne $dependencies.files.Count){throw 'Runtime dependency inventory changed'}
foreach($file in $dependencies.files){
    if((Get-FileHash -LiteralPath (Join-Path $managed $file.name)).Hash -ne $file.sha256){throw "Runtime dependency changed: $($file.name)"}
}
$compiler=Join-Path $repo 'artifacts/patch-tools/microsoft.net.compilers.toolset.4.12.0/tasks/net472/csc.exe'
if(-not(Test-Path -LiteralPath $compiler)){
    $toolsDirectory=Join-Path $repo 'artifacts/patch-tools'
    New-Item -ItemType Directory -Force -Path $toolsDirectory|Out-Null
    $archive=Join-Path $toolsDirectory 'microsoft.net.compilers.toolset.4.12.0.nupkg'
    if(-not(Test-Path -LiteralPath $archive)){
        Invoke-WebRequest -Uri 'https://api.nuget.org/v3-flatcontainer/microsoft.net.compilers.toolset/4.12.0/microsoft.net.compilers.toolset.4.12.0.nupkg' -OutFile $archive
    }
    if((Get-FileHash -LiteralPath $archive).Hash -ne 'FE24EF31A6FFCB7C49383D2FD362763DEE291AD9B9D98CC0C19EF80203B99EBC'){throw 'Compiler package fingerprint mismatch'}
    Add-Type -AssemblyName System.IO.Compression.FileSystem
    [IO.Compression.ZipFile]::ExtractToDirectory($archive,(Join-Path $toolsDirectory 'microsoft.net.compilers.toolset.4.12.0'))
}
$out=Join-Path $repo 'artifacts/r1-host/guarded'
New-Item -ItemType Directory -Force -Path $out | Out-Null
Get-ChildItem -LiteralPath $managed -Filter '*.dll' | Copy-Item -Destination $out -Force
$references=@(Get-ChildItem -LiteralPath $managed -Filter '*.dll' | ForEach-Object {'/reference:"'+$_.FullName+'"'})
$sources=@(Get-ChildItem -LiteralPath (Join-Path $repo 'src/BrassHost') -Filter '*.cs' | Sort-Object Name | ForEach-Object {'"'+$_.FullName+'"'})
$response=@('/nologo','/target:exe','/langversion:7.3','/nostdlib+','/deterministic+','/optimize+',('/out:"'+(Join-Path $out 'BrassHost.exe')+'"'))+$references+$sources
$rsp=Join-Path $repo 'artifacts/r1-host/host.rsp'
$response | Set-Content -LiteralPath $rsp -Encoding UTF8
& $compiler ('@'+$rsp)
if($LASTEXITCODE -ne 0){throw 'Host compilation failed.'}
$runtimeFiles=@(Get-ChildItem -LiteralPath $out -Filter '*.dll')+@(Get-Item -LiteralPath (Join-Path $out 'BrassHost.exe'))
$sourcePins=@(Get-ChildItem -LiteralPath (Join-Path $repo 'src/BrassHost') -Filter '*.cs' | Sort-Object Name | ForEach-Object{[ordered]@{name=$_.Name;sha256=(Get-FileHash -LiteralPath $_.FullName).Hash.ToLowerInvariant()}})
$manifest=[ordered]@{format='brass-runtime';version=1;rules=$profile.id;compiler_package='Microsoft.Net.Compilers.Toolset/4.12.0';sources=$sourcePins;files=@($runtimeFiles | Sort-Object Name | ForEach-Object{[ordered]@{name=$_.Name;sha256=(Get-FileHash -LiteralPath $_.FullName).Hash.ToLowerInvariant()}})}
$manifest|ConvertTo-Json -Depth 5|Set-Content -LiteralPath (Join-Path $out 'runtime.json') -Encoding UTF8
Write-Output "Built isolated host: $out. Installed game unchanged."
