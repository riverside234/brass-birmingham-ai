$ErrorActionPreference = 'Stop'
# Isolated, in-memory scenarios against the installed game DLL. Does not launch Unity,
# read saves, modify game assemblies, or persist any game state.
$gameRoot = Split-Path -Parent $PSScriptRoot
$assemblyPath = Join-Path $gameRoot 'BrassBirminghamUnity_Data/Managed/Assembly-CSharp.dll'
$expectedHash = '600EBF309670B2726001C16A3375F8BBF3DE76A23C0202BE7CF198BA338981FB'
if ((Get-FileHash -LiteralPath $assemblyPath -Algorithm SHA256).Hash -ne $expectedHash) { throw 'Unexpected DLL; re-audit before running.' }
[void][System.Reflection.Assembly]::LoadFrom($assemblyPath)
$results = [System.Collections.Generic.List[object]]::new()
function New-ProbeGame([int]$count) {
    $config = [BoardGameRules.Entities.Game.GameStartConfiguration]::new([BoardGameRules.Entities.Common.NumberOfPlayers]$count)
    $config.Seed = 20260927
    $game = [BoardGameRules.Entities.Game.BoardGameRulesGame]::new($config)
    $game.StartGame()
    $game.GameProgressInformation.CurrentGameEra = [BoardGameRules.Entities.GamePhases.GameEra]::RailEra
    return $game
}
function Add-LocationCard($player, $region) {
    $card = [BoardGameRules.Entities.Cards.LocationCard]::new('probe-location', [BoardGameRules.Entities.Common.NumberOfPlayers]::Two, $region.Name, $region.Color)
    $player.Hand.Clear()
    $player.Hand.Add($card)
    return $card
}
foreach ($count in 2,3,4) {
    $game = New-ProbeGame $count
    $board = $game.Board
    $player = $game.GameProgressInformation.CurrentPlayer
    $other = @($game.PlayersTrack.Players | Where-Object Color -ne $player.Color)[0]
    $player.Money = 22
    $coventry = $board.GetRegionByName([BoardGameRules.Entities.Board.Regions.RegionName]::Coventry)
    $nuneaton = $board.GetRegionByName([BoardGameRules.Entities.Board.Regions.RegionName]::Nuneaton)
    $potteryType = [BoardGameRules.Entities.Board.BuildingSlots.BuildingType]::Pottery
    $player.Mat.DevelopIndustry($potteryType)
    $player.Mat.DevelopIndustry($potteryType)
    $pottery = $player.Mat.GetFirstBuildingOfType($potteryType)
    $card = Add-LocationCard $player $coventry
    $nearest = [BoardGameRules.Entities.Industries.CoalMine]::new('probe-nearest', $other.Color, 2, 2, 1, 7, $true, $false, $false, 7, 0, 0, 1)
    $farther = [BoardGameRules.Entities.Industries.CoalMine]::new('probe-farther', $other.Color, 2, 2, 1, 7, $true, $false, $false, 7, 0, 0, 1)
    $coventry.BuildingSlots[1].BuildIndustry($nearest)
    $nuneaton.BuildingSlots[1].BuildIndustry($farther)
    $nearest.Build()
    $farther.Build()
    $board.GetLinkById('RailCoventryNuneaton').SetOwner($other)
    $slot = $coventry.BuildingSlots[0]
    $action = $game.TurnFlow.BuildAction
    $before = $action.GetCantUseBuildActionReasonForGivenSlot($slot, $pottery, $player, $board, $game.GameProgressInformation)
    $canBuild = [BoardGameRules.Entities.Actions.BuildAction]::CanUseBuildActionForGivenSlot($slot, $pottery, $player, $board, $game.GameProgressInformation)
    $nearestEligible = [BoardGameRules.Entities.Consumables.ConsumingCoal]::CanCoalBeSuppliedToRegionFromSource($nearest, $coventry, $board)
    $fartherEligible = [BoardGameRules.Entities.Consumables.ConsumingCoal]::CanCoalBeSuppliedToRegionFromSource($farther, $coventry, $board)
    # The domain execution succeeds despite the availability predicate above rejecting it.
    $action.DiscardCard($card, $player)
    $action.Build($slot, $potteryType)
    $action.SupplyCoal($nearest, 1)
    $fartherAfterNearest = [BoardGameRules.Entities.Consumables.ConsumingCoal]::CanCoalBeSuppliedToRegionFromSource($farther, $coventry, $board)
    $action.SupplyCoal($farther, 1)
    $results.Add([ordered]@{ case='two-coal-build-nearest-exhaustion'; players=$count; target='Coventry Pottery III'; expectedUiAllowsBuild=$true; actualUiAllowsBuild=$canBuild; actualReasons=@($before | ForEach-Object ToString); coalBefore=@{nearest=1;farther=1}; nearestEligibleInitially=$nearestEligible; fartherEligibleInitially=$fartherEligible; fartherEligibleAfterNearestConsumed=$fartherAfterNearest; executionNeededCoalAfter=$action.NeededCoal; moneyAfter=$player.Money; bothMinesFlipped=($nearest.IsFlipped -and $farther.IsFlipped) })

    $game = New-ProbeGame $count
    $board = $game.Board
    $player = $game.GameProgressInformation.CurrentPlayer
    $player.Money = 100
    $cannock = $board.GetRegionByName([BoardGameRules.Entities.Board.Regions.RegionName]::Cannock)
    $coalType = [BoardGameRules.Entities.Board.BuildingSlots.BuildingType]::CoalMine
    $player.Mat.DevelopIndustry($coalType)
    $oldMine = $player.Mat.GetFirstBuildingOfType($coalType)
    [void]$player.Mat.RemoveGivenBuildingFromMat($oldMine)
    $cannock.BuildingSlots[0].BuildIndustry($oldMine)
    $oldMine.Build()
    $player.Mat.DevelopIndustry($coalType)
    $newMine = $player.Mat.GetFirstBuildingOfType($coalType)
    $card = Add-LocationCard $player $cannock
    $slot = $cannock.BuildingSlots[0]
    $action = $game.TurnFlow.BuildAction
    $overbuildEligible = [BoardGameRules.Entities.Actions.BuildAction]::GetPossibleSlotsToOverBuildInWithGivenCardByGivenPlayer($card, $player, $board).Contains($slot)
    $before = $action.GetCantUseBuildActionReasonForGivenSlot($slot, $newMine, $player, $board, $game.GameProgressInformation)
    $canBuild = [BoardGameRules.Entities.Actions.BuildAction]::CanUseBuildActionForGivenSlot($slot, $newMine, $player, $board, $game.GameProgressInformation)
    $action.DiscardCard($card, $player)
    $errorText = $null
    try { $action.Build($slot, $coalType) } catch { $errorText = $_.Exception.InnerException.Message }
    $results.Add([ordered]@{ case='overbuild-occupied-dual-slot'; players=$count; target='Cannock Coal II to III'; emptySingleSlot=($null -eq $cannock.BuildingSlots[1].BuildedIndustry); expectedUiAllowsBuild=$true; actualUiAllowsBuild=$canBuild; overbuildPredicateAllows=$overbuildEligible; actualReasons=@($before | ForEach-Object ToString); executionError=$errorText })
}
[ordered]@{ assemblySha256=$expectedHash; method='isolated hand-constructed in-memory rule states using installed DLL; no Unity UI test'; cases=$results } | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $PSScriptRoot 'build_resources-results.json') -Encoding UTF8
$results | ConvertTo-Json -Depth 8
