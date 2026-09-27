$ErrorActionPreference = 'Stop'
$assemblyPath = Join-Path (Split-Path -Parent $PSScriptRoot) 'BrassBirminghamUnity_Data/Managed/Assembly-CSharp.dll'
$hash = '600EBF309670B2726001C16A3375F8BBF3DE76A23C0202BE7CF198BA338981FB'
if ((Get-FileHash -LiteralPath $assemblyPath -Algorithm SHA256).Hash -ne $hash) { throw 'Assembly changed' }
$assembly = [System.Reflection.Assembly]::LoadFrom($assemblyPath)
$failures = [System.Collections.Generic.List[object]]::new()
$counts = [ordered]@{market=0;income=0;build=0;develop=0;overbuild=0;aiControls=0}
function Check($group, $name, $expected, $actual) {
    $counts[$group]++
    if ($expected -ne $actual) { $failures.Add([ordered]@{group=$group;case=$name;expected=$expected;actual=$actual}) }
}
function New-Game([int]$players=2) {
    $config = [BoardGameRules.Entities.Game.GameStartConfiguration]::new([BoardGameRules.Entities.Common.NumberOfPlayers]$players)
    $config.Seed = 20260927
    $game = [BoardGameRules.Entities.Game.BoardGameRulesGame]::new($config)
    $game.StartGame()
    return $game
}
function Card-At($player, $region) {
    $card = [BoardGameRules.Entities.Cards.LocationCard]::new('extended-build-card', [BoardGameRules.Entities.Common.NumberOfPlayers]::Two, $region.Name, $region.Color)
    $player.Hand.Clear(); $player.Hand.Add($card)
    return $card
}
function Set-MarketStock($market, [int]$stock) {
    [void]$market.Buy(100)
    [void]$market.Sell($stock)
}
# Printed market ladders: paired prices 1..7 (coal), 1..5 (iron), unlimited 8/6.
foreach ($kind in 'Coal','Iron') {
    $ladder = if ($kind -eq 'Coal') { @(1,1,2,2,3,3,4,4,5,5,6,6,7,7) } else { @(1,1,2,2,3,3,4,4,5,5) }
    $outside = if ($kind -eq 'Coal') {8} else {6}
    for ($stock=0; $stock -le $ladder.Count; $stock++) {
        for ($qty=1; $qty -le 16; $qty++) {
            $market = if ($kind -eq 'Coal') {[BoardGameRules.Entities.Markets.CoalMarket]::new()} else {[BoardGameRules.Entities.Markets.IronMarket]::new()}
            Set-MarketStock $market $stock
            $expectedBuy=0; $expectedSell=0
            for ($n=0; $n -lt $qty; $n++) {
                $expectedBuy += if ($n -lt $stock) {$ladder[$ladder.Count-$stock+$n]} else {$outside}
                if ($n -lt $ladder.Count-$stock) {$expectedSell += $ladder[$ladder.Count-$stock-$n-1]}
            }
            Check market "$kind stock=$stock quantity=$qty buy" $expectedBuy $market.GetBuyingPrice($qty)
            Check market "$kind stock=$stock quantity=$qty sell" $expectedSell $market.GetSellingPrice($qty)
        }
    }
}
# Fixed printed-income band endpoints; independently check every marker and 3-level loan landing.
$upper = @{}; for($level=-10;$level -le 0;$level++){$upper[$level]=$level+10}
for($level=1;$level -le 10;$level++){$upper[$level]=10+2*$level}
for($level=11;$level -le 20;$level++){$upper[$level]=30+3*($level-10)}
for($level=21;$level -le 29;$level++){$upper[$level]=60+4*($level-20)}
$upper[30]=99
for($marker=0;$marker -le 99;$marker++) {
    $expectedLevel = -10
    while($marker -gt $upper[$expectedLevel]){$expectedLevel++}
    Check income "marker=$marker level" $expectedLevel ([BoardGameRules.Entities.NumericalResources.IncomeTrack]::GetCurrentIncomeLevel($marker))
    Check income "marker=$marker threshold" $upper[$expectedLevel] ([BoardGameRules.Entities.NumericalResources.IncomeTrack]::GetIncomeThreshold($marker))
    if($expectedLevel -ge -7) { Check income "marker=$marker loan" $upper[$expectedLevel-3] ([BoardGameRules.Entities.NumericalResources.IncomeTrack]::DecreaseIncomeLevelBy($marker,3)) }
}
# Build every available next-level variant with current-era gates and exact cash/one-short.
# A common fully connected board excludes known B02 nearest-stock defect from this matrix.
foreach($players in 2,3,4) {
    foreach($era in [BoardGameRules.Entities.GamePhases.GameEra]::CanalEra,[BoardGameRules.Entities.GamePhases.GameEra]::RailEra) {
        foreach($buildingType in [Enum]::GetValues([BoardGameRules.Entities.Board.BuildingSlots.BuildingType])) {
            if($buildingType.ToString() -eq 'Undefined'){continue}
            $seedGame=New-Game $players
            $allTiles=@($seedGame.GameProgressInformation.CurrentPlayer.Mat.GetIndustryList($buildingType))
            for($tileIndex=0;$tileIndex -lt $allTiles.Count;$tileIndex++) {
                foreach($marketStock in 0,1,8) {
                    $game=New-Game $players; $board=$game.Board; $player=$game.GameProgressInformation.CurrentPlayer
                    $game.GameProgressInformation.CurrentGameEra=$era
                    foreach($region in $board.Regions){foreach($link in $region.Links){if(-not $link.HasOwner -and $link.GetType().Name -eq $(if($era.ToString() -eq 'CanalEra'){'Canal'}else{'Rail'})){$link.SetOwner($player)}}}
                    for($i=0;$i -lt $tileIndex;$i++){$player.Mat.DevelopIndustry($buildingType)}
                    $tile=$player.Mat.GetFirstBuildingOfType($buildingType)
                    $region=@($board.Regions|Where-Object{ @($_.BuildingSlots|Where-Object{$_.BuildingTypes.Count -eq 1 -and $_.BuildingTypes.Contains($buildingType)}).Count -gt 0 })[0]
                    $slot=@($region.BuildingSlots|Where-Object{$_.BuildingTypes.Count -eq 1 -and $_.BuildingTypes.Contains($buildingType)})[0]
                    $card=Card-At $player $region
                    Set-MarketStock $board.CoalMarket $marketStock; Set-MarketStock $board.IronMarket $marketStock
                    $cost=$tile.MoneyCost+$board.CoalMarket.GetBuyingPrice($tile.CoalCost)+$board.IronMarket.GetBuyingPrice($tile.IronCost)
                    $legalEra= -not (($tile.IsForCanalEra -and $era.ToString() -eq 'RailEra') -or ($tile.IsForRailEra -and $era.ToString() -eq 'CanalEra'))
                    $player.Money=$cost
                    $actual=[BoardGameRules.Entities.Actions.BuildAction]::CanUseBuildActionForGivenSlot($slot,$tile,$player,$board,$game.GameProgressInformation)
                    Check build "$players $era $buildingType tileIndex=$tileIndex market=$marketStock exactcash" $legalEra $actual
                    $player.Money=$cost-1
                    Check build "$players $era $buildingType tileIndex=$tileIndex market=$marketStock shortcash" $false ([BoardGameRules.Entities.Actions.BuildAction]::CanUseBuildActionForGivenSlot($slot,$tile,$player,$board,$game.GameProgressInformation))
                    if($legalEra) {
                        $player.Money=$cost
                        $action=$game.TurnFlow.StartBuildAction()
                        [BrassApplication.AI.AIHelpers.BuildActionAIHelper]::BuildIndustry($action,$card,$slot,$buildingType,$board,$player)
                        Check build "$players $era $buildingType tileIndex=$tileIndex market=$marketStock executionResourceDebt" 0 ($action.NeededCoal+$action.NeededIron)
                        Check build "$players $era $buildingType tileIndex=$tileIndex market=$marketStock placement" $tile.UniqueId $slot.BuildedIndustry.UniqueId
                        if($buildingType.ToString() -eq 'Brewery'){Check build "$players $era brewery beer" $(if($era.ToString() -eq 'CanalEra'){1}else{2}) $tile.AvailableBeer}
                        Check build "$players $era $buildingType round-spend" $cost $game.PlayersTrack.SpendMoneyInCurrentRound[$player.Color]
                        Check build "$players $era $buildingType card discard" 0 $player.Hand.Count
                        $expectedRevenue=0; $expectedIncome=10
                        if($buildingType.ToString() -in @('CoalMine','IronWorks')) {
                            $capacity=if($buildingType.ToString() -eq 'CoalMine'){14}else{10}
                            $ladder=if($capacity -eq 14){@(1,1,2,2,3,3,4,4,5,5,6,6,7,7)}else{@(1,1,2,2,3,3,4,4,5,5)}
                            $output=$tile.GetMaxResource(); $sold=[Math]::Min($capacity-$marketStock,$output)
                            for($n=0;$n -lt $sold;$n++){$expectedRevenue+=$ladder[$capacity-$marketStock-$n-1]}
                            Check build "$players $era $buildingType remaining output" ($output-$sold) $tile.GetResource()
                            Check build "$players $era $buildingType flip" ($sold -eq $output) $tile.IsFlipped
                            if($sold -eq $output){$expectedIncome+=$tile.IncomeIncrease}
                        }
                        Check build "$players $era $buildingType market revenue" $expectedRevenue $player.Money
                        Check build "$players $era $buildingType income" $expectedIncome $player.IncomeMarkerPosition
                    }
                }
            }
        }
    }
}
# Development affordability and payment across market boundaries and free board iron.
foreach($stock in 0,1,2,5,10) {
    foreach($freeIron in 0,1,2) {
        $game=New-Game; $board=$game.Board; $player=$game.GameProgressInformation.CurrentPlayer
        Set-MarketStock $board.IronMarket $stock
        if($freeIron -gt 0) {
            $other=@($game.PlayersTrack.Players|Where-Object Color -ne $player.Color)[0]
            $iron=[BoardGameRules.Entities.Industries.IronWorks]::new('extended-dev-iron',$other.Color,2,5,1,3,$true,$false,$false,7,1,0,$freeIron)
            $board.GetRegionByName([BoardGameRules.Entities.Board.Regions.RegionName]::Birmingham).BuildingSlots[2].BuildIndustry($iron)
        }
        $action=$game.TurnFlow.DevelopAction
        $oneCost=$board.IronMarket.GetBuyingPrice([Math]::Max(0,1-$freeIron)); $twoCost=$board.IronMarket.GetBuyingPrice([Math]::Max(0,2-$freeIron))
        for($money=0;$money -le 12;$money++) {
            $player.Money=$money
            Check develop "market=$stock free=$freeIron cash=$money first affordability" ($money -ge $oneCost) ($action.GetDevelopActionNotPossibleReasons($game.GameProgressInformation).Count -eq 0)
            Check develop "market=$stock free=$freeIron cash=$money second affordability" ($money -ge $twoCost) ($action.HasPlayerIronForSecondDevelop($player))
        }
        $player.Money=$twoCost
        $card=$player.Hand[0]
        $beforeMat=$player.Mat.GetIndustryList([BoardGameRules.Entities.Board.BuildingSlots.BuildingType]::CottonMill).Count
        [BrassApplication.AI.AIHelpers.DevelopActionAIHelper]::DevelopIndustry($action,$card,[BoardGameRules.Entities.Board.BuildingSlots.BuildingType]::CottonMill,[BoardGameRules.Entities.Board.BuildingSlots.BuildingType]::CottonMill,$board,$player)
        Check develop "market=$stock free=$freeIron two removed" ($beforeMat-2) $player.Mat.GetIndustryList([BoardGameRules.Entities.Board.BuildingSlots.BuildingType]::CottonMill).Count
        Check develop "market=$stock free=$freeIron money spent" 0 $player.Money
        Check develop "market=$stock free=$freeIron iron paid" 0 $action.NeededIron
        Check develop "market=$stock free=$freeIron counter" 2 $action.DevelopCounter
    }
}
# Tile-development permissions from the printed lightbulb rule and icons: Pottery I/III prohibited.
$game=New-Game; $player=$game.GameProgressInformation.CurrentPlayer
foreach($type in [Enum]::GetValues([BoardGameRules.Entities.Board.BuildingSlots.BuildingType])) {
    if($type.ToString() -eq 'Undefined'){continue}
    while($player.Mat.GetFirstBuildingOfType($type)) {
        $tile=$player.Mat.GetFirstBuildingOfType($type)
        $expected= -not ($type.ToString() -eq 'Pottery' -and $tile.Level -in @(1,3))
        Check develop "$type level=$($tile.Level) permission" $expected $player.Mat.CanIndustryBeDeveloped($type)
        $player.Mat.DevelopIndustry($type)
    }
    Check develop "$type exhausted permission" $false $player.Mat.CanIndustryBeDeveloped($type)
}
# Overbuild ownership/resource restrictions, without known B08 dual-icon preference case.
foreach($type in [Enum]::GetValues([BoardGameRules.Entities.Board.BuildingSlots.BuildingType])) {
    if($type.ToString() -eq 'Undefined'){continue}
    foreach($ownership in 'own','opponent') {
        foreach($oldStock in 0,1) {
            foreach($marketStock in 0,1) {
                $game=New-Game; $board=$game.Board; $player=$game.GameProgressInformation.CurrentPlayer
                $game.GameProgressInformation.CurrentGameEra=[BoardGameRules.Entities.GamePhases.GameEra]::RailEra
                $other=@($game.PlayersTrack.Players|Where-Object Color -ne $player.Color)[0]
                foreach($region in $board.Regions){foreach($link in $region.Links){if(-not $link.HasOwner -and $link.GetType().Name -eq 'Rail'){$link.SetOwner($other)}}}
                $owner=if($ownership -eq 'own'){$player}else{$other}
                $old=$owner.Mat.GetFirstBuildingOfType($type)
                [void]$owner.Mat.RemoveGivenBuildingFromMat($old)
                while($player.Mat.GetFirstBuildingOfType($type).Level -le $old.Level){$player.Mat.DevelopIndustry($type)}
                $new=$player.Mat.GetFirstBuildingOfType($type)
                $region=@($board.Regions|Where-Object{ @($_.BuildingSlots|Where-Object{$_.BuildingTypes.Count -eq 1 -and $_.BuildingTypes.Contains($type)}).Count -gt 0 })[0]
                $slot=@($region.BuildingSlots|Where-Object{$_.BuildingTypes.Count -eq 1 -and $_.BuildingTypes.Contains($type)})[0]
                $slot.BuildIndustry($old); $old.Build()
                switch($type.ToString()) {'CoalMine'{$old.AvailableCoal=$oldStock};'IronWorks'{$old.AvailableIron=$oldStock};'Brewery'{$old.AvailableBeer=$oldStock}}
                $old.IsFlipped=($oldStock -eq 0)
                Set-MarketStock $board.CoalMarket $marketStock; Set-MarketStock $board.IronMarket $marketStock
                $card=Card-At $player $region; $player.Money=100
                $expected=($ownership -eq 'own') -or (($type.ToString() -in @('CoalMine','IronWorks')) -and $oldStock -eq 0 -and $marketStock -eq 0)
                $actual=[BoardGameRules.Entities.Actions.BuildAction]::CanUseBuildActionForGivenSlot($slot,$new,$player,$board,$game.GameProgressInformation)
                Check overbuild "$type $ownership stock=$oldStock market=$marketStock eligibility" $expected $actual
                if($expected) {
                    $action=$game.TurnFlow.StartBuildAction()
                    [BrassApplication.AI.AIHelpers.BuildActionAIHelper]::BuildIndustry($action,$card,$slot,$type,$board,$player)
                    Check overbuild "$type $ownership stock=$oldStock market=$marketStock replacement" $new.UniqueId $slot.BuildedIndustry.UniqueId
                    Check overbuild "$type $ownership stock=$oldStock market=$marketStock removed" $true $board.RemovedIndustries.Contains($old)
                    $expectedIncome=10; if($new.IsFlipped){$expectedIncome+=$new.IncomeIncrease}
                    Check overbuild "$type $ownership stock=$oldStock market=$marketStock no old-income reward" $expectedIncome $player.IncomeMarkerPosition
                }
            }
        }
    }
}
# AI empty-manufacturer edge: public normal decision entry point, with an ordinary 3/4p card.
$aiCases=[System.Collections.Generic.List[object]]::new()
foreach($players in 3,4) {
    $game=New-Game $players; $player=$game.GameProgressInformation.CurrentPlayer; $board=$game.Board
    $game.GameProgressInformation.CurrentGameEra=[BoardGameRules.Entities.GamePhases.GameEra]::RailEra
    $game.GameProgressInformation.CurrentRoundNumber=1
    $game.TurnFlow.StartTurn($player,1,[BoardGameRules.Entities.GamePhases.GameEra]::RailEra)
    $player.Money=0; $player.IncomeMarkerPosition=1
    $manufacture=[BoardGameRules.Entities.Board.BuildingSlots.BuildingType]::Manufacturer
    $allManufact=@($player.Mat.GetIndustryList($manufacture))
    while($player.Mat.GetFirstBuildingOfType($manufacture)){$player.Mat.DevelopIndustry($manufacture)}
    $factory=[BoardGameRules.Entities.Cards.CardsFactory]::new($game.Randomer)
    $goodsCard=@($factory.GetLocationAndIndustryCards([BoardGameRules.Entities.Common.NumberOfPlayers]$players)|Where-Object{$_ -is [BoardGameRules.Entities.Cards.IndustryCard] -and $_.BuildingTypes.Contains($manufacture)})[0]
    $player.Hand[0]=$goodsCard
    $other=@($game.PlayersTrack.Players|Where-Object Color -ne $player.Color)[0]
    $iron=[BoardGameRules.Entities.Industries.IronWorks]::new('extended-ai-iron',$other.Color,2,5,1,3,$true,$false,$false,7,1,0,1)
    $board.GetRegionByName([BoardGameRules.Entities.Board.Regions.RegionName]::Birmingham).BuildingSlots[2].BuildIndustry($iron)
    $metadata=[BrassApplication.Game.GameMetadata]::new($game.GameStartConfiguration,[BrassApplication.Game.GameType]::Local)
    $app=[BrassApplication.Game.BrassApplicationGame]::new($game,$metadata)
    $behavior=[BrassApplication.AI.AiBehaviors.HardAiBehavior]::new('probe')
    $errorText=$null
    try{$move=$behavior.GetNextAIMove($app)}catch{$errorText=$_.Exception.ToString()}
    $controls=[ordered]@{}
    $game.GameProgressInformation.CurrentRoundNumber=2
    $controlError=$null;try{$move=$behavior.GetNextAIMove($app)}catch{$controlError=$_.Exception.Message}
    $controls.secondRailRound=$controlError
    Check aiControls "$players second rail round no exception" $true ($null -eq $controlError)
    $game.GameProgressInformation.CurrentRoundNumber=1
    $player.Mat.GetIndustryList($manufacture).Add($allManufact[0])
    $controlError=$null;try{$move=$behavior.GetNextAIMove($app)}catch{$controlError=$_.Exception.Message}
    $controls.manufacturerAvailable=$controlError
    Check aiControls "$players manufacturer available no exception" $true ($null -eq $controlError)
    $player.Mat.GetIndustryList($manufacture).Clear()
    $easy=[BrassApplication.AI.AiBehaviors.EasyAiBehavior]::new('probe')
    $controlError=$null;try{$move=$easy.GetNextAIMove($app)}catch{$controlError=$_.Exception.Message}
    $controls.easyDifficulty=$controlError
    Check aiControls "$players easy no exception" $true ($null -eq $controlError)
    $aiCases.Add([ordered]@{players=$players;case='empty manufacturing stack first rail round';normalDecisionEntry='HardAiBehavior.GetNextAIMove';exception=$errorText;controls=$controls;limitation='Legally attainable stock/input context arranged in memory; not reproduced from a naturally played original-AI match.'})
}
[ordered]@{assemblySha256=$hash;method='Original DLL, isolated in-memory fixtures, no saves/game files touched';assertionCounts=$counts;failures=$failures;aiEdges=$aiCases} | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $PSScriptRoot 'extended-build-results.json') -Encoding UTF8
[ordered]@{counts=$counts;failures=$failures;aiEdges=$aiCases}|ConvertTo-Json -Depth 8
