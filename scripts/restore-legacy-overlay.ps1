# Restore only a known R3 overlay to the pinned 1.1.0 rules patch. Never restores
# unpatched stock over the rules fixes and never reads/writes user game saves.
$ErrorActionPreference='Stop'
$repo=Split-Path -Parent $PSScriptRoot
$combinedRecord=Join-Path $repo 'artifacts/patch-installation.json'
if(Test-Path -LiteralPath $combinedRecord){
    $release=Get-Content -LiteralPath $combinedRecord -Raw|ConvertFrom-Json
    if($release.status -eq 'installed' -and [version]$release.version -ge [version]'1.2.0'){
        throw 'This restore command is only for the research overlay on 1.1.0. Use the combined patch installer/recovery scripts for this installation; no files were changed.'
    }
}
if(Get-Process BrassBirminghamUnity -ErrorAction SilentlyContinue){throw 'Close the game before restoring the optional overlay.'}
$managed=Join-Path $repo 'BrassBirminghamUnity_Data/Managed'
$baseline=Join-Path $repo 'artifacts/r1-host/guarded/Assembly-CSharp.dll'
$expected='1c8cef281b81474200268eed23448faa3f5247f8ea49409a69efc3982c5a2d0a'
if((Get-FileHash -LiteralPath $baseline).Hash.ToLowerInvariant() -ne $expected){throw 'The private pinned rules baseline is missing or changed.'}
$manifest=Get-Content -LiteralPath (Join-Path $repo 'artifacts/legacy-ai/overlay/manifest.json') -Raw|ConvertFrom-Json
$target=Join-Path $managed 'Assembly-CSharp.dll'
$actual=(Get-FileHash -LiteralPath $target).Hash.ToLowerInvariant()
$overlayHash=($manifest.files|Where-Object name -eq 'Assembly-CSharp.dll').sha256
if($actual -ne $expected -and $actual -ne $overlayHash){throw 'Unknown installed assembly; preserve it and inspect before any restore.'}
if($actual -ne $expected){Copy-Item -LiteralPath $baseline -Destination $target -Force}
$helper=Join-Path $managed 'BrassLegacyAI.dll'
if(Test-Path -LiteralPath $helper){
    $helperHash=($manifest.files|Where-Object name -eq 'BrassLegacyAI.dll').sha256
    if((Get-FileHash -LiteralPath $helper).Hash.ToLowerInvariant() -ne $helperHash){throw 'Unknown helper retained; rules assembly has been restored.'}
    Remove-Item -LiteralPath $helper
}
Write-Output 'Pinned rules patch 1.1.0 restored; legacy sidecars/configuration and all game saves retained.'
