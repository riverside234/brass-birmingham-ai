$ErrorActionPreference = 'Stop'
$gameRoot = Split-Path -Parent $PSScriptRoot
$assemblyPath = Join-Path $gameRoot 'BrassBirminghamUnity_Data\Managed\Assembly-CSharp.dll'
$hash = (Get-FileHash -LiteralPath $assemblyPath -Algorithm SHA256).Hash
if ($hash -ne '600EBF309670B2726001C16A3375F8BBF3DE76A23C0202BE7CF198BA338981FB') { throw 'Assembly version changed.' }
[void][Reflection.Assembly]::LoadFrom($assemblyPath)

function New-ProbeGame {
    $cfg = [BoardGameRules.Entities.Game.GameStartConfiguration]::new([BoardGameRules.Entities.Common.NumberOfPlayers]::Two)
    $cfg.Seed = 20260927
    $g = [BoardGameRules.Entities.Game.BoardGameRulesGame]::new($cfg)
    $g.StartGame()
    return $g
}

function Add-ProbeIndustry($g, $regionName, $buildingType, $level) {
    $p = $g.GameProgressInformation.CurrentPlayer
    $industry = @($p.Mat.GetIndustryList($buildingType) | Where-Object Level -eq $level)[0]
    [void]$p.Mat.RemoveGivenBuildingFromMat($industry)
    $region = $g.Board.GetRegionByName($regionName)
    $slot = @($region.BuildingSlots | Where-Object { $_.BuildingTypes.Contains($buildingType) -and $null -eq $_.BuildedIndustry })[0]
    $slot.BuildIndustry($industry)
    return $slot
}

# Fixture uses a possible two-player merchant arrangement: manufactured-goods
# and the all-industries tile at Oxford. Industry/resource state is arranged
# directly in memory; execution uses the installed DLL's original AI helper.
$g = New-ProbeGame
$p = $g.GameProgressInformation.CurrentPlayer
$bt = [BoardGameRules.Entities.Board.BuildingSlots.BuildingType]::Manufacturer
$slot = Add-ProbeIndustry $g ([BoardGameRules.Entities.Board.Regions.RegionName]::Birmingham) $bt 5
$oxford = $g.Board.GetRegionByName([BoardGameRules.Entities.Board.Regions.RegionName]::Oxford)
$allMerchants = @($g.Board.Regions | Where-Object { $_ -is [BoardGameRules.Entities.Board.Regions.BorderRegion] } | ForEach-Object { $_.MerchantSlots })
$manufacturedTile = @($allMerchants.MerchantTile | Where-Object { $null -ne $_ -and $_.BuildingTypes.Count -eq 1 -and $_.BuildingTypes.Contains($bt) })[0]
$wildTile = @($allMerchants.MerchantTile | Where-Object { $null -ne $_ -and $_.BuildingTypes.Count -eq 3 })[0]
foreach ($merchant in $allMerchants) {
    $merchant.SetMerchant([BoardGameRules.Entities.Board.Merchants.MerchantTile]::new([BoardGameRules.Entities.Common.NumberOfPlayers]::Two,[System.Collections.Generic.List[BoardGameRules.Entities.Board.BuildingSlots.BuildingType]]::new()))
}
$oxford.MerchantSlots[0].SetMerchant($manufacturedTile)
$oxford.MerchantSlots[1].SetMerchant($wildTile)
$g.Board.GetLinkById('CanalBirminghamOxford').SetOwner($p)
$action = $g.TurnFlow.StartSellAction()
$beforeIncome = $p.IncomeMarkerPosition
$beforeBeer = ($oxford.MerchantSlots | Measure-Object -Property AvailableBeer -Sum).Sum
$uiReasons = @($action.GetSellActionNotPossibleReasons($g.Board,$g.GameProgressInformation) | ForEach-Object { $_.ToString() })
$slotEligible = $action.GetValidBuildingSlotsForSellingIndustries($g.Board,$p).Contains($slot)
$logger = [BoardGameRules.Utils.Logging.LogFactory]::BuildLogger.Invoke([BoardGameRules.Entities.Actions.SellAction])
[BrassApplication.AI.AIHelpers.SellActionAIHelper]::SellIndustries($action,$p.Hand[0],$g.Board,$p,$g.DevelopMerchantBonus,$logger)
$merchantProbe = [ordered]@{
    name = 'Original AI sells one level V manufacturer using two distinct merchants'
    fixtureNote = 'Two Oxford merchant tiles accept manufactured goods; no brewery exists; arranged in memory.'
    industryBeerCost = $slot.BuildedIndustry.CostOfBeerBarrelsToSellIndustry
    uiPreflightReasons = $uiReasons
    engineEligible = $slotEligible
    beforeMerchantBeer = $beforeBeer
    afterMerchantBeer = ($oxford.MerchantSlots | Measure-Object -Property AvailableBeer -Sum).Sum
    industryFlipped = $slot.BuildedIndustry.IsFlipped
    incomeMarkerDelta = $p.IncomeMarkerPosition - $beforeIncome
    industryIncome = $slot.BuildedIndustry.IncomeIncrease
    expected = 'Sale unavailable without at least one brewery beer; one industry may use only the selected merchant beer.'
}

# Confirm no-development state is expressible and the core end-turn guard
# rejects unresolved bonus. Human UI call path is assessed statically.
$g2 = New-ProbeGame
$p2 = $g2.GameProgressInformation.CurrentPlayer
foreach ($type in [Enum]::GetValues([BoardGameRules.Entities.Board.BuildingSlots.BuildingType])) {
    if ($type -ne [BoardGameRules.Entities.Board.BuildingSlots.BuildingType]::Undefined) { $p2.Mat.GetIndustryList($type).Clear() }
}
$g2.ApplyEvent([BoardGameRules.Entities.Board.Events.MerchantBonusReceived]::new([BoardGameRules.Entities.Board.Regions.BonusType]::Develop,$p2.Color,[BoardGameRules.Entities.Board.Regions.RegionName]::Gloucester))
$endTurnError = $null
try { $g2.TurnFlow.EndTurn() } catch { $endTurnError = $_.Exception.InnerException.Message }
$developProbe = [ordered]@{
    name = 'Gloucester bonus with no remaining eligible tiles'
    cannotDevelopReasons = @($g2.DevelopMerchantBonus.GetDevelopMerchantBonusNotPossibleReasons($g2.GameProgressInformation) | ForEach-Object { $_.ToString() })
    bonusPending = $g2.DevelopMerchantBonus.MerchantDevelopmentNeeded
    endTurnError = $endTurnError
    uiReachability = 'Static call-chain finding; no Unity UI automation performed.'
}

# The proposed Birmingham-Dudley rail is incident to a Dudley mine. A
# Redditch mine is a further rail away, but both are tied from Birmingham.
$g3 = New-ProbeGame
$p3 = $g3.GameProgressInformation.CurrentPlayer
$g3.GameProgressInformation.CurrentGameEra = [BoardGameRules.Entities.GamePhases.GameEra]::RailEra
$p3.Money = 30
$coalType = [BoardGameRules.Entities.Board.BuildingSlots.BuildingType]::CoalMine
$nearSlot = Add-ProbeIndustry $g3 ([BoardGameRules.Entities.Board.Regions.RegionName]::Dudley) $coalType 2
$farSlot = Add-ProbeIndustry $g3 ([BoardGameRules.Entities.Board.Regions.RegionName]::Redditch) $coalType 2
$g3.Board.GetLinkById('RailBirminghamRedditch').SetOwner($p3)
$link = $g3.Board.GetLinkById('RailBirminghamDudley')
$network = $g3.TurnFlow.StartNetworkAction()
$network.DiscardCard($p3.Hand[0],$p3)
$network.PlaceLink($link)
$targetRegions = @($link.Region1,$link.Region2,$link.Region3) | Where-Object { $null -ne $_ }
$validCoal = @($g3.Board.GetUnflippedCoalMines() | Where-Object {
    $mine = $_
    @($targetRegions | Where-Object { $network.GeneralAction.ConsumingCoal.CanCoalBeSuppliedToRegionFromSource($mine,$_) }).Count -gt 0
})
$nearCoalBefore = $nearSlot.BuildedIndustry.AvailableCoal
$farCoalBefore = $farSlot.BuildedIndustry.AvailableCoal
$network.SupplyCoal($farSlot.BuildedIndustry,1)
$coalProbe = [ordered]@{
    name = 'Rail coal chooser accepts a mine farther from the new link'
    fixtureNote = 'New Birmingham-Dudley rail; existing Birmingham-Redditch rail; coal mines in Dudley and Redditch.'
    allowedMineRegions = @($validCoal | ForEach-Object { $g3.Board.GetRegionContainingGivenIndustry($_).Name.ToString() })
    dudleyDistancesToEnds = @($targetRegions | ForEach-Object { $g3.Board.GetRouteLength($g3.Board.GetRegionContainingGivenIndustry($nearSlot.BuildedIndustry),$_) })
    redditchDistancesToEnds = @($targetRegions | ForEach-Object { $g3.Board.GetRouteLength($g3.Board.GetRegionContainingGivenIndustry($farSlot.BuildedIndustry),$_) })
    coalConsumedAtDudley = $nearCoalBefore - $nearSlot.BuildedIndustry.AvailableCoal
    coalConsumedAtRedditch = $farCoalBefore - $farSlot.BuildedIndustry.AvailableCoal
    expected = 'Only Dudley is closest to the new link; compare each mine minimum distance across all ends, rather than per-endpoint winners.'
}
$report = [ordered]@{
    assemblySha256 = $hash
    utc = [DateTime]::UtcNow.ToString('o')
    runtime = [Runtime.InteropServices.RuntimeInformation]::FrameworkDescription
    method = 'Read original installed assembly; mutate only isolated in-memory fixtures; no game saves or binaries modified.'
    results = @($merchantProbe,$developProbe,$coalProbe)
}
$json = $report | ConvertTo-Json -Depth 10
$json | Set-Content -LiteralPath (Join-Path $PSScriptRoot 'network-sell-probe-results.json') -Encoding utf8
$json
