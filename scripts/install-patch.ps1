param([string]$GameRoot=(Split-Path -Parent $PSScriptRoot),[switch]$SkipValidation,[string[]]$SaveDataPaths=@())
$ErrorActionPreference='Stop'
. (Join-Path $PSScriptRoot 'patch-install-common.ps1')
if($SkipValidation){throw 'Installation without validation is not supported.'}
$repo=Split-Path -Parent $PSScriptRoot
$game=(Resolve-Path -LiteralPath $GameRoot).Path
$managed=Join-Path $game 'BrassBirminghamUnity_Data/Managed'
$build=Join-Path $repo 'artifacts/patch-build'
$original=Join-Path $repo 'artifacts/patch-baseline/Assembly-CSharp.dll'
$recordPath=Join-Path $repo 'artifacts/patch-installation.json'
$intentPath=Join-Path $repo 'artifacts/patch-installation.pending.json'
$manifest=Read-BrassJson (Join-Path $build 'build-manifest.json')
if($manifest -and $manifest.version -in @('1.3.0','1.3.1','1.3.2','1.4.0','1.4.1','1.4.2','1.4.3','1.4.4','1.4.5','1.4.6','1.4.7')){
    & (Join-Path $PSScriptRoot 'install-combined-patch.ps1') -GameRoot $GameRoot -SaveDataPaths $SaveDataPaths
    return
}
$validation=Read-BrassJson (Join-Path $build 'validation.json')
Assert-BrassCandidate $manifest $validation $build
if(Get-Process -Name BrassBirminghamUnity -ErrorAction SilentlyContinue){throw 'Close Brass: Birmingham before installing. No files were changed.'}
if((Get-FileHash -LiteralPath $original).Hash -ne $BrassStockSha256){throw 'Original backup hash mismatch.'}
$previous=Read-BrassJson $recordPath
$pending=Read-BrassJson $intentPath
foreach($item in @($previous,$pending)){if($item -and $item.gameRoot -ne $game){throw 'Installation record belongs to a different game directory.'}}
$paths=@(Get-BrassSaveDataPaths @($previous,$pending) $SaveDataPaths)
$known=@($previous,$pending,$manifest)
$assembly=Join-Path $managed 'Assembly-CSharp.dll'
$installedHash=(Get-FileHash -LiteralPath $assembly).Hash
if($installedHash -notin (@($BrassStockSha256)+@(Get-BrassHashes $known 'Assembly-CSharp.dll'))){throw 'Installed game differs from stock and known patch files. Refusing to overwrite it.'}
$helper=Join-Path $managed 'BrassPatch.dll'
$priorFiles=@([ordered]@{name='Assembly-CSharp.dll';sha256=$installedHash})
if(Test-Path -LiteralPath $helper){
    $helperHash=(Get-FileHash -LiteralPath $helper).Hash
    if($helperHash -notin @(Get-BrassHashes $known 'BrassPatch.dll')){throw 'An unrecognized helper already exists.'}
    $priorFiles+=([ordered]@{name='BrassPatch.dll';sha256=$helperHash})
}
# The durable intent recognizes both sides of an interrupted upgrade, including a first install.
$intent=[ordered]@{operation='install';createdUtc=[DateTime]::UtcNow.ToString('o');gameRoot=$game;stockBackup=$original;stockSha256=$BrassStockSha256;priorFiles=$priorFiles;candidateFiles=$manifest.files;saveDataPaths=$paths}
Write-BrassJsonAtomic $intentPath $intent
try{
    Copy-BrassAtomic (Join-Path $build 'Managed/BrassPatch.dll') $helper
    Copy-BrassAtomic (Join-Path $build 'Managed/Assembly-CSharp.dll') $assembly
    foreach($file in $manifest.files){if((Get-FileHash -LiteralPath (Join-Path $managed $file.name)).Hash -ne $file.sha256){throw 'Installed hash mismatch.'}}
    $record=[ordered]@{version=$manifest.version;status='installed';installedUtc=[DateTime]::UtcNow.ToString('o');gameRoot=$game;stockBackup=$original;stockSha256=$BrassStockSha256;files=$manifest.files;onlineEnabled=$false;saveDataPaths=$paths}
    Write-BrassJsonAtomic $recordPath $record
    Remove-Item -LiteralPath $intentPath
}catch{
    $installError=$_
    # Restore one coherent stock installation. Keep the intent so a failed recovery can be retried.
    try{
        Assert-BrassNoPendingSaves $paths
        Copy-BrassAtomic $original $assembly
        if((Get-FileHash -LiteralPath $assembly).Hash -ne $BrassStockSha256){throw 'Stock rollback hash mismatch.'}
        if(Test-Path -LiteralPath $helper){
            if((Get-FileHash -LiteralPath $helper).Hash -notin @(Get-BrassHashes @($intent) 'BrassPatch.dll')){throw 'Helper changed during installation; it was retained.'}
            Remove-Item -LiteralPath $helper
        }
    }catch{Write-Warning ('Automatic stock recovery could not complete; the stock backup and recovery intent were retained. '+$_.Exception.Message)}
    throw $installError
}
Write-Output 'Installed the verified local-game patch. Online play is disabled. Original backup retained.'
