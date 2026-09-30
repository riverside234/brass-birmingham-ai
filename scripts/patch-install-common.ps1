# Shared packaging helpers. Records and DLL replacements are published atomically.
$BrassStockSha256='600EBF309670B2726001C16A3375F8BBF3DE76A23C0202BE7CF198BA338981FB'

function Read-BrassJson([string]$Path){
    if(-not(Test-Path -LiteralPath $Path)){return $null}
    try{return Get-Content -LiteralPath $Path -Raw|ConvertFrom-Json}
    catch{Write-Warning ('Unreadable record retained for inspection: '+$Path);return $null}
}

function Invoke-BrassAtomicReplace([string]$Temporary,[string]$Destination){
    # Windows scanners can briefly prevent replacement even after our own handles
    # are closed. Retry the same atomic operation; never delete the live target.
    for($attempt=0;$attempt -lt 6;$attempt++){
        try{[IO.File]::Replace($Temporary,$Destination,[NullString]::Value);return}
        catch{
            $cause=$_.Exception.GetBaseException()
            $code=$cause.HResult -band 0xffff
            if($attempt -eq 5 -or $cause -isnot [IO.IOException] -or $code -notin @(32,33,1175) -or -not(Test-Path -LiteralPath $Temporary)){throw}
            Start-Sleep -Milliseconds (200*($attempt+1))
        }
    }
}

function Write-BrassJsonAtomic([string]$Path,$Value){
    $temporary=$Path+'.'+[Guid]::NewGuid().ToString('N')+'.tmp'
    try{
        $bytes=[Text.Encoding]::UTF8.GetBytes(($Value|ConvertTo-Json -Depth 15))
        $stream=[IO.File]::Open($temporary,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::None)
        try{$stream.Write($bytes,0,$bytes.Length);$stream.Flush($true)}finally{$stream.Dispose()}
        if(Test-Path -LiteralPath $Path){Invoke-BrassAtomicReplace $temporary $Path}
        else{[IO.File]::Move($temporary,$Path)}
    }finally{if(Test-Path -LiteralPath $temporary){Remove-Item -LiteralPath $temporary}}
}

function Copy-BrassAtomic([string]$Source,[string]$Destination){
    $temporary=$Destination+'.'+[Guid]::NewGuid().ToString('N')+'.tmp'
    try{
        Copy-Item -LiteralPath $Source -Destination $temporary
        $stream=[IO.File]::Open($temporary,[IO.FileMode]::Open,[IO.FileAccess]::ReadWrite,[IO.FileShare]::None)
        try{$stream.Flush($true)}finally{$stream.Dispose()}
        if(Test-Path -LiteralPath $Destination){Invoke-BrassAtomicReplace $temporary $Destination}
        else{[IO.File]::Move($temporary,$Destination)}
    }finally{if(Test-Path -LiteralPath $temporary){Remove-Item -LiteralPath $temporary}}
}

function Get-BrassHashes($Records,[string]$Name){
    @($Records|Where-Object {$null -ne $_}|ForEach-Object {
        (@($_.files)+@($_.priorFiles)+@($_.candidateFiles))|Where-Object {$_.name -eq $Name}|ForEach-Object {$_.sha256}
    }|Where-Object {$_}|Sort-Object -Unique)
}

function Get-BrassSaveDataPaths($Records,[string[]]$Additional=@()){
    $paths=@($Additional)
    foreach($item in $Records){if($null -ne $item){$paths+=@($item.saveDataPaths);$paths+=@($item.persistentDataPath)}}
    # An explicit or previously recorded player directory takes precedence over the
    # account running the packaging tool (which may be a sandbox or administrator).
    if(-not @($paths|Where-Object {$_}).Count){
        $profilePath=[Environment]::GetFolderPath('UserProfile')
        if($profilePath){$paths+=Join-Path $profilePath 'AppData/LocalLow/Cublo/Brass Birmingham'}
    }
    @($paths|Where-Object {$_}|ForEach-Object {[IO.Path]::GetFullPath($_).TrimEnd([char[]]@('\','/'))}|Sort-Object -Unique)
}

function Assert-BrassNoPendingSaves([string[]]$Paths){
    foreach($savePath in $Paths){
        if(Test-Path -LiteralPath $savePath){
            $pending=@(Get-ChildItem -LiteralPath $savePath -Recurse -File -Filter 'brasspatch-save-v1.pending' -ErrorAction Stop)
            if($pending.Count){throw ('A local save has a pending recovery journal under '+$savePath+'. Reopen it with the patch to finish recovery before restoring stock files. No stock restore was performed.')}
        }
    }
}

function Assert-BrassCandidate($Manifest,$Validation,[string]$Build){
    if(-not $Manifest -or $Manifest.stockSha256 -ne $BrassStockSha256){throw 'Candidate is not based on the supported stock game.'}
    if(-not $Validation -or -not $Validation.passed){throw 'Candidate validation must pass before installation.'}
    $expectedFiles='Assembly-CSharp.dll|BrassPatch.dll'
    if($Manifest.version -in @('1.3.0','1.3.1','1.3.2','1.4.0','1.4.1','1.4.2','1.4.3','1.4.4')){
        $expectedFiles='Assembly-CSharp.dll|BrassLegacyAI.dll|BrassPatch.dll'
        if(-not $Validation.PSObject.Properties['localAi'] -or -not $Validation.localAi.passed){throw 'Combined AI and native UI qualification must pass before installation.'}
    }
    if((@($Manifest.files|ForEach-Object {$_.name}|Sort-Object)-join '|') -ne $expectedFiles){throw 'Unexpected candidate file list.'}
    foreach($file in $Manifest.files){
        if((Get-FileHash -LiteralPath (Join-Path $Build ('Managed/'+$file.name))).Hash -ne $file.sha256){throw ('Changed candidate: '+$file.name)}
        $checked=@($Validation.files|Where-Object {$_.name -eq $file.name})
        if($checked.Count -ne 1 -or $checked[0].sha256 -ne $file.sha256){throw 'Validation belongs to a different build.'}
    }
}

function Assert-BrassNoNewAiSaves([string[]]$Paths){
    foreach($savePath in $Paths){
        if(Test-Path -LiteralPath $savePath){
            foreach($file in Get-ChildItem -LiteralPath $savePath -Recurse -File -Filter 'metadata.storage' -ErrorAction Stop){
                if([IO.File]::ReadAllText($file.FullName).Contains('"brasspatch_ai"')){
                    throw 'A saved game uses the six-strategy AI. Keep a compatible combined patch installed to resume it. Restoring an older runtime would change its AI; no restore was performed.'
                }
            }
        }
    }
}
