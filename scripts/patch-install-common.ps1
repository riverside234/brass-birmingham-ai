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
    if($Manifest.version -in @('1.3.0','1.3.1','1.3.2','1.4.0','1.4.1','1.4.2','1.4.3','1.4.4','1.4.5','1.4.6','1.4.7','1.4.8','1.4.9','1.4.10','1.4.11','1.4.12','1.4.13','1.4.15','1.4.16','1.4.17','1.4.18','1.4.19','1.4.20','1.4.21')){
        $expectedFiles='Assembly-CSharp.dll|BrassLegacyAI.dll|BrassPatch.dll'
        if(-not $Validation.PSObject.Properties['localAi'] -or -not $Validation.localAi.passed){throw 'Combined AI and native UI qualification must pass before installation.'}
    }
    if((@($Manifest.files|ForEach-Object {$_.name}|Sort-Object)-join '|') -ne $expectedFiles){throw 'Unexpected candidate file list.'}
    if($Manifest.version -in @('1.4.7','1.4.8','1.4.9','1.4.10','1.4.11','1.4.12','1.4.13','1.4.15','1.4.16','1.4.17','1.4.18','1.4.19','1.4.20','1.4.21')){
        if(-not $Manifest.PSObject.Properties['sourcePins'] -or -not $Validation.localAi.PSObject.Properties['sourcePins'] -or $Manifest.sourcePins.Count -ne $Validation.localAi.sourcePins.Count){throw 'Compiled runtime-source qualification is missing.'}
        foreach($pin in $Manifest.sourcePins){if(@($Validation.localAi.sourcePins|Where-Object {$_.path -eq $pin.path -and $_.sha256 -eq $pin.sha256}).Count -ne 1){throw 'Qualified runtime sources differ from compilation.'}}
        if(-not $Validation.localAi.resourceTests.passed -or $Validation.localAi.resourceTests.checks -lt 20 -or $Validation.localAi.executionStatistics.maximumRunning -ne 1 -or $Validation.localAi.executionStatistics.running){throw 'General AI execution/resource qualification must pass.'}
    }
    if($Manifest.version -in @('1.4.8','1.4.10')){
        if(-not $Validation.PSObject.Properties['statistics'] -or -not $Validation.statistics.passed -or $Validation.statistics.matches -lt 6 -or $Validation.statistics.pages -ne 10){throw 'Native completed-match statistics qualification must pass.'}
        if($Manifest.version -eq '1.4.11'){
        if(-not $Validation.localAi.masked.passed -or $Validation.localAi.masked.backend -ne 'masked_ismcts' -or $Validation.localAi.masked.iterations -lt 8){throw 'Repaired search correctness is incomplete.'}
        if(-not $Validation.statistics.passed -or $Validation.statistics.retainedFrom -ne '1.4.10'){throw 'Preserved statistics readability evidence is missing.'}
        foreach($file in $Manifest.files|Where-Object name -ne 'BrassLegacyAI.dll'){
            if(@($Validation.statistics.testedFiles|Where-Object {$_.name -eq $file.name -and $_.sha256 -eq $file.sha256}).Count -ne 1){throw 'Preserved UI/rules binaries differ.'}
        }
    }
    foreach($file in $Manifest.files){if(@($Validation.statistics.testedFiles|Where-Object {$_.name -eq $file.name -and $_.sha256 -eq $file.sha256}).Count -ne 1){throw 'Statistics qualification belongs to a different build.'}}
    }
    if($Manifest.version -eq '1.4.9'){
        if(-not $Validation.localAi.masked.passed -or $Validation.localAi.masked.backend -ne 'masked_ismcts' -or $Validation.localAi.masked.iterations -lt 2){throw 'Masked ISMCTS runtime qualification must pass.'}
        if(-not $Validation.statistics.passed -or $Validation.statistics.retainedFrom -ne '1.4.8'){throw 'Retained statistics evidence is missing.'}
        foreach($file in $Manifest.files|Where-Object name -ne 'BrassLegacyAI.dll'){
            if(@($Validation.statistics.testedFiles|Where-Object {$_.name -eq $file.name -and $_.sha256 -eq $file.sha256}).Count -ne 1){throw 'Retained UI/rules evidence differs from the unchanged binary.'}
        }
    }
    if($Manifest.version -eq '1.4.10'){
        if(-not $Validation.localAi.masked.passed -or $Validation.localAi.masked.backend -ne 'masked_ismcts'){throw 'Preserved Masked ISMCTS qualification is missing.'}
        if($Validation.localAi.retainedEvidence.release -ne '1.4.9' -or $Validation.statistics.checks -lt 600){throw 'Statistics readability/native qualification is incomplete.'}
    }
    if($Manifest.version -eq '1.4.12'){
        if($Manifest.baseVersion -ne '1.4.11' -or -not $Validation.PSObject.Properties['cardTracker'] -or -not $Validation.cardTracker.passed -or $Validation.cardTracker.checks -lt 500 -or -not $Validation.preservedRuntime.passed){throw 'Native Card Tracker and preserved-runtime qualification must pass.'}
        if(-not $Validation.localAi.masked.passed -or $Validation.localAi.masked.backend -ne 'masked_ismcts' -or $Validation.localAi.retainedFrom -ne $Manifest.baseVersion){throw 'Accepted AI qualification is missing.'}
        if(($Manifest.files|Where-Object name -eq 'BrassLegacyAI.dll').sha256 -ne ($Manifest.baseFiles|Where-Object name -eq 'BrassLegacyAI.dll').sha256){throw 'Card Tracker must preserve the accepted AI binary.'}
        if(-not $Validation.statistics.passed -or $Validation.statistics.checks -lt 600 -or -not $Validation.presentation.passed -or $Validation.presentation.checks -lt 180 -or -not $Validation.linkReplay.passed -or $Validation.linkReplay.checks -lt 765 -or -not $Validation.menu.passed -or $Validation.menu.checks -lt 259){throw 'Affected UI regression qualification is incomplete.'}
        foreach($report in $Validation.cardTracker,$Validation.statistics,$Validation.presentation,$Validation.linkReplay,$Validation.menu){
            foreach($file in $Manifest.files){if(@($report.testedFiles|Where-Object {$_.name -eq $file.name -and $_.sha256 -eq $file.sha256}).Count -ne 1){throw 'UI qualification belongs to a different candidate.'}}
        }
    }
    if($Manifest.version -eq '1.4.13'){
        if($Manifest.baseVersion -ne '1.4.12' -or -not $Validation.cardTracker.passed -or $Validation.cardTracker.checks -lt 530 -or -not $Validation.menu.passed -or $Validation.menu.checks -lt 259 -or -not $Validation.preservedRuntime.passed){throw 'Card Tracker appearance and preserved-runtime qualification must pass.'}
        if(-not $Validation.localAi.masked.passed -or $Validation.localAi.masked.backend -ne 'masked_ismcts' -or $Validation.localAi.retainedFrom -ne $Manifest.baseVersion){throw 'Accepted AI qualification is missing.'}
        foreach($name in 'BrassLegacyAI.dll','Assembly-CSharp.dll'){
            if(($Manifest.files|Where-Object name -eq $name).sha256 -ne ($Manifest.baseFiles|Where-Object name -eq $name).sha256){throw 'Appearance revision must preserve the accepted AI and game binaries.'}
        }
        if(-not $Validation.retainedUi.passed -or $Validation.retainedUi.release -ne $Manifest.baseVersion -or -not $Validation.retainedUi.statistics.passed -or $Validation.retainedUi.statistics.checks -lt 600 -or -not $Validation.retainedUi.presentation.passed -or $Validation.retainedUi.presentation.checks -lt 180 -or -not $Validation.retainedUi.linkReplay.passed -or $Validation.retainedUi.linkReplay.checks -lt 765){throw 'Preserved UI regression evidence is incomplete.'}
        if(@($Validation.sourceScope|Where-Object {$_ -notin @('src/BrassPatch/CardTracker/CardTrackerOverlay.cs','src/BrassPatch/CardTracker/CardTrackerWidgets.cs')}).Count){throw 'Appearance qualification contains an unrelated source change.'}
        foreach($report in $Validation.cardTracker,$Validation.menu){
            foreach($file in $Manifest.files){if(@($report.testedFiles|Where-Object {$_.name -eq $file.name -and $_.sha256 -eq $file.sha256}).Count -ne 1){throw 'Appearance qualification belongs to a different candidate.'}}
        }
    }
    if($Manifest.version -in @('1.4.15','1.4.16','1.4.18','1.4.20')){
        $gameWorkers=if($Manifest.version -in @('1.4.16','1.4.18','1.4.20')){2}else{4}
        if(-not $Validation.localAi.masked.passed -or $Validation.localAi.masked.backend -ne 'masked_ismcts' -or $Validation.localAi.masked.iterations -lt 8 -or -not $Validation.localAi.era.passed -or $Validation.localAi.era.checks -lt 19 -or $Validation.localAi.workers -ne $gameWorkers -or $Validation.localAi.era.cpu_batching.pending){throw 'Era-aware search and worker qualification must pass.'}
        if($Manifest.version -in @('1.4.16','1.4.18','1.4.20') -and ($Manifest.searchWorkers -ne 2 -or $Validation.localAi.era.game_workers -ne 2 -or $Validation.localAi.era.checks -lt 22 -or $Validation.localAi.labWorkers -ne 4)){throw 'Game two-worker/lab four-worker separation must pass.'}
        $userSelectedV4=$Manifest.version -eq '1.4.20' -and $Manifest.selectionBasis -eq 'user-directed-masked-v4/1.0.0'
        $expectedGames=if($userSelectedV4){0}else{switch($Manifest.version){'1.4.20'{24}'1.4.18'{24}default{8}}}
        if(-not $Validation.retainedUi.passed -or $Validation.retainedUi.release -ne $Manifest.baseVersion -or -not $Validation.localAi.sourceParity.passed -or -not $Validation.smoke.passed -or $Validation.smoke.games -ne $expectedGames -or $Validation.smoke.bank -ne $Validation.localAi.sourceParity.bank){throw 'Preserved UI, game/research parity and smoke qualification must pass.'}
        if($Manifest.version -eq '1.4.18' -and (-not $Validation.localAi.industry.passed -or $Validation.localAi.industry.checks -lt 16 -or -not $Validation.smoke.promote -or @($Validation.smoke.comparisons.PSObject.Properties).Count -ne 2)){throw 'Useful development/group-sale correctness and both comparison controls must pass.'}
        if($userSelectedV4){
            if($Manifest.generation -ne 'v4' -or $Manifest.selectedBank -ne 'bank-3dc65253a610191d66a5' -or $Validation.localAi.selectionBasis -ne $Manifest.selectionBasis -or
              $Validation.smoke.selectionBasis -ne $Manifest.selectionBasis -or $Validation.smoke.generation -ne 'v4' -or $Validation.smoke.bank -ne $Manifest.selectedBank -or $Validation.smoke.promote -or
              $Validation.smoke.freshStrengthGames -ne 0 -or -not $Validation.localAi.economy.passed -or $Validation.localAi.economy.checks -lt 37 -or
              -not $Validation.localAi.industry.passed -or $Validation.localAi.industry.checks -lt 17){throw 'Exact user-selected v4 strategy, progress and correctness qualification is required.'}
            $coverage=$Validation.localAi.conversionCoverage
            if(-not $coverage.passed -or $coverage.styles -ne 6 -or $coverage.deadlineSales -ne 12 -or $coverage.merchantBonuses -ne 30 -or $coverage.batchSales -ne 6 -or $coverage.breweryTradeoffs -ne 6 -or $coverage.checks -lt 400){throw 'All six styles need complete named conversion coverage.'}
        }elseif($Manifest.version -eq '1.4.20'){
            if(-not $Validation.localAi.economy.passed -or $Validation.localAi.economy.checks -lt 46 -or -not $Validation.localAi.industry.passed -or $Validation.localAi.industry.checks -lt 17 -or -not $Validation.smoke.promote){throw 'Six-style economy/development qualification and win-focused promotion must pass.'}
            $controls=@('bank-3dc65253a610191d66a5','bank-f5107c644af2fbf4ac99')
            if(@($Validation.smoke.comparisons.PSObject.Properties).Count -ne $controls.Count){throw 'Both previous and economic comparison controls are required.'}
            foreach($control in $controls){
                $comparison=$Validation.smoke.comparisons.PSObject.Properties[$control].Value
                if(-not $comparison -or $comparison.games -ne 12 -or -not $comparison.eligible -or -not $comparison.more_wins -or $comparison.policies.new.win_share -le $comparison.policies.old.win_share){throw 'Tournament-style revision must win more against every control.'}
                if(-not $comparison.all_metrics_improved -or $comparison.policies.new.first_places -le $comparison.policies.old.first_places -or $comparison.policies.new.mean_vp -le $comparison.policies.old.mean_vp -or $comparison.policies.new.mean_rank -ge $comparison.policies.old.mean_rank){throw 'Each control must improve on first places, VP AND rank.'}
            }
        }
        foreach($file in $Manifest.files|Where-Object name -ne 'BrassLegacyAI.dll'){
            if(@($Manifest.baseFiles|Where-Object {$_.name -eq $file.name -and $_.sha256 -eq $file.sha256}).Count -ne 1 -or @($Validation.retainedUi.testedFiles|Where-Object {$_.name -eq $file.name -and $_.sha256 -eq $file.sha256}).Count -ne 1){throw 'Era revision must preserve the qualified UI/rules binaries.'}
        }
    }
    if($Manifest.version -eq '1.4.17'){
        if($Manifest.baseVersion -ne '1.4.16' -or $Manifest.helperSourceBaseVersion -ne '1.4.13' -or -not $Validation.preservedRuntime.passed){throw 'Statistics review requires its accepted runtime and preserved method qualification.'}
        foreach($name in 'Assembly-CSharp.dll','BrassLegacyAI.dll'){
            if(($Manifest.files|Where-Object name -eq $name).sha256 -ne ($Manifest.baseFiles|Where-Object name -eq $name).sha256){throw 'Statistics review must retain the accepted game and AI binaries.'}
        }
        if(-not $Validation.localAi.masked.passed -or $Validation.localAi.masked.backend -ne 'masked_ismcts' -or $Validation.localAi.retainedFrom -ne $Manifest.baseVersion -or $Validation.localAi.workers -ne 2){throw 'Accepted AI execution qualification is missing.'}
        if(-not $Validation.statistics.passed -or $Validation.statistics.pages -ne 11 -or $Validation.statistics.matches -ne 9 -or $Validation.statistics.checks -lt 2500){throw 'Expanded statistics qualification is incomplete.'}
        if(-not $Validation.cardTracker.passed -or $Validation.cardTracker.checks -lt 539 -or -not $Validation.presentation.passed -or $Validation.presentation.checks -lt 184 -or -not $Validation.linkReplay.passed -or $Validation.linkReplay.checks -lt 765 -or -not $Validation.menu.passed -or $Validation.menu.checks -lt 259){throw 'Affected UI regression qualification is incomplete.'}
        if(@($Validation.sourceScope|Where-Object {$_ -notlike 'src/BrassPatch/Statistics/*'}).Count){throw 'Statistics qualification contains an unrelated runtime source change.'}
        foreach($report in $Validation.statistics,$Validation.cardTracker,$Validation.presentation,$Validation.linkReplay,$Validation.menu){
            foreach($file in $Manifest.files){if(@($report.testedFiles|Where-Object {$_.name -eq $file.name -and $_.sha256 -eq $file.sha256}).Count -ne 1){throw 'Statistics regression evidence belongs to a different candidate.'}}
        }
    }
    if($Manifest.version -eq '1.4.19'){
        if($Manifest.baseVersion -ne '1.4.18' -or $Manifest.helperSourceBaseVersion -ne '1.4.17' -or -not $Validation.preservedRuntime.passed -or -not $Validation.visualChecks.passed){throw 'Information UI requires accepted 1.4.18, preserved runtime and native visual qualification.'}
        foreach($name in 'Assembly-CSharp.dll','BrassLegacyAI.dll'){
            if(($Manifest.files|Where-Object name -eq $name).sha256 -ne ($Manifest.baseFiles|Where-Object name -eq $name).sha256){throw 'Information UI must preserve the accepted game and AI binaries.'}
        }
        if(-not $Validation.localAi.masked.passed -or $Validation.localAi.masked.backend -ne 'masked_ismcts' -or $Validation.localAi.retainedFrom -ne $Manifest.baseVersion -or $Validation.localAi.workers -ne 2){throw 'Accepted AI execution qualification is missing.'}
        if(-not $Validation.statistics.passed -or $Validation.statistics.pages -ne 11 -or $Validation.statistics.matches -ne 9 -or $Validation.statistics.checks -lt 2500 -or -not $Validation.cardTracker.passed -or $Validation.cardTracker.checks -lt 430){throw 'Updated information UI native qualification is incomplete.'}
        foreach($report in $Validation.statistics,$Validation.cardTracker){
            if(@($report.layouts).Count -ne 2 -or @($report.layouts|Where-Object {$_.width -eq 1920 -and $_.height -eq 1200}).Count -ne 1 -or @($report.layouts|Where-Object {$_.width -eq 1920 -and $_.height -eq 1080}).Count -ne 1){throw 'Requested 1200p and 1080p UI coverage is missing.'}
        }
        if(-not $Validation.presentation.passed -or $Validation.presentation.checks -lt 184 -or -not $Validation.linkReplay.passed -or $Validation.linkReplay.checks -lt 765 -or -not $Validation.menu.passed -or $Validation.menu.checks -lt 259){throw 'Preserved UI regression qualification is incomplete.'}
        $uiScope=@('src/BrassPatch/CardTracker/CardTrackerOverlay.cs','src/BrassPatch/Statistics/StatisticsEntryPoints.cs','src/BrassPatch/Statistics/StatisticsEntrySurface.cs','src/BrassPatch/Statistics/StatisticsPanel.cs')
        if(@($Validation.sourceScope|Where-Object {$_ -notin $uiScope}).Count -or @($Manifest.sourceScope|Where-Object {$_ -notin $uiScope}).Count){throw 'Information UI contains an unrelated source change.'}
        foreach($report in $Validation.statistics,$Validation.cardTracker,$Validation.presentation,$Validation.linkReplay,$Validation.menu){
            foreach($file in $Manifest.files){if(@($report.testedFiles|Where-Object {$_.name -eq $file.name -and $_.sha256 -eq $file.sha256}).Count -ne 1){throw 'Information UI evidence belongs to a different candidate.'}}
        }
    }
    if($Manifest.version -eq '1.4.21'){
        if($Manifest.baseVersion -ne '1.4.20' -or $Manifest.helperSourceBaseVersion -ne '1.4.19' -or -not $Validation.preservedRuntime.passed -or -not $Validation.visualChecks.passed){throw 'Statistics revision requires accepted 1.4.20, preserved runtime and native visual qualification.'}
        foreach($name in 'Assembly-CSharp.dll','BrassLegacyAI.dll'){
            if(($Manifest.files|Where-Object name -eq $name).sha256 -ne ($Manifest.baseFiles|Where-Object name -eq $name).sha256){throw 'Statistics revision must preserve the current game and AI binaries.'}
        }
        if(-not $Validation.localAi.masked.passed -or $Validation.localAi.masked.backend -ne 'masked_ismcts' -or $Validation.localAi.retainedFrom -ne $Manifest.baseVersion -or $Validation.localAi.workers -ne 2){throw 'Accepted AI execution qualification is missing.'}
        if(-not $Validation.statistics.passed -or $Validation.statistics.pages -ne 10 -or $Validation.statistics.matches -ne 9 -or $Validation.statistics.checks -lt 2700 -or -not $Validation.cardTracker.passed -or $Validation.cardTracker.checks -lt 430){throw 'Statistics and preserved Card Tracker qualification are incomplete.'}
        foreach($report in $Validation.statistics,$Validation.cardTracker){
            if(@($report.layouts).Count -ne 2 -or @($report.layouts|Where-Object {$_.width -eq 1920 -and $_.height -eq 1200}).Count -ne 1 -or @($report.layouts|Where-Object {$_.width -eq 1920 -and $_.height -eq 1080}).Count -ne 1){throw 'Requested 1200p and 1080p coverage is missing.'}
        }
        if(-not $Validation.presentation.passed -or $Validation.presentation.checks -lt 184 -or -not $Validation.linkReplay.passed -or $Validation.linkReplay.checks -lt 765 -or -not $Validation.menu.passed -or $Validation.menu.checks -lt 259){throw 'Preserved UI regression qualification is incomplete.'}
        if(@($Validation.sourceScope|Where-Object {$_ -notlike 'src/BrassPatch/Statistics/*'}).Count -or @($Manifest.sourceScope|Where-Object {$_ -notlike 'src/BrassPatch/Statistics/*'}).Count -or ($Manifest.retiredSources -join '|') -ne 'src/BrassPatch/Statistics/StatisticsInsights.cs'){throw 'Statistics revision contains an unrelated source change.'}
        foreach($id in 'SD01','SD02','SD03','SD04','SD05','SD06','SD07','SD08','SD09','SD10','SD11','N1','N2','N3','N4','N5','N6','N7'){
            if(-not @($Validation.statistics.cases|Where-Object {$_.passed -and $_.name.Contains($id)}).Count){throw ('Required statistics regression missing: '+$id)}
        }
        foreach($report in $Validation.statistics,$Validation.cardTracker,$Validation.presentation,$Validation.linkReplay,$Validation.menu){
            foreach($file in $Manifest.files){if(@($report.testedFiles|Where-Object {$_.name -eq $file.name -and $_.sha256 -eq $file.sha256}).Count -ne 1){throw 'Statistics/native preservation evidence belongs to another candidate.'}}
        }
    }
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
