$ErrorActionPreference = 'Stop'
$gameRoot = Split-Path -Parent $PSScriptRoot
$assemblyPath = Join-Path $gameRoot 'BrassBirminghamUnity_Data\Managed\Assembly-CSharp.dll'
$hash = (Get-FileHash -LiteralPath $assemblyPath -Algorithm SHA256).Hash
if ($hash -ne '600EBF309670B2726001C16A3375F8BBF3DE76A23C0202BE7CF198BA338981FB') { throw 'Unreviewed assembly version' }
[void][System.Reflection.Assembly]::LoadFrom($assemblyPath)
$cases = @()
foreach ($count in @(2,3,4)) {
    $config = [BoardGameRules.Entities.Game.GameStartConfiguration]::new([BoardGameRules.Entities.Common.NumberOfPlayers]$count)
    $config.Seed = 20260927
    $game = [BoardGameRules.Entities.Game.BoardGameRulesGame]::new($config)
    $game.StartGame()
    $players = $game.PlayersTrack.Players
    $merchants = @($game.Board.Regions | Where-Object { $_ -is [BoardGameRules.Entities.Board.Regions.BorderRegion] } | ForEach-Object { $_.MerchantSlots } | Where-Object { $null -ne $_.MerchantTile })
    $totalCards = $game.Board.CardDeck.Count + ($players | ForEach-Object { $_.Hand.Count + $_.DiscardPile.Count } | Measure-Object -Sum).Sum
    $expectedCards = @{2=40;3=54;4=64}[$count]
    $perPlayer = @($players | ForEach-Object {
        $p = $_
        $tiles = 0
        foreach ($type in @('CoalMine','IronWorks','Pottery','Brewery','CottonMill','Manufacturer')) { $tiles += $p.Mat.GetIndustryList([BoardGameRules.Entities.Board.BuildingSlots.BuildingType]::$type).Count }
        if ($p.Money -ne 17 -or $p.Hand.Count -ne 8 -or $p.DiscardPile.Count -ne 1 -or $p.IncomeMarkerPosition -ne 10 -or $tiles -ne 45 -or $p.LinkTileCount -ne 14) { throw 'Setup invariant failed' }
        [ordered]@{color=$p.Color.ToString();money=$p.Money;hand=$p.Hand.Count;discard=$p.DiscardPile.Count;incomeMarker=$p.IncomeMarkerPosition;industryTiles=$tiles;linkTiles=$p.LinkTileCount}
    })
    if ($totalCards -ne $expectedCards -or $game.TurnFlow.GetMaxActionsCount() -ne 1) { throw 'Deck/first-round invariant failed' }
    $cases += [ordered]@{players=$count;normalCards=$totalCards;remainingDeck=$game.Board.CardDeck.Count;merchantTiles=$merchants.Count;merchantBeer=@($game.Board.GetMerchantsWithBeer()).Count;wildLocations=$game.Board.WildLocationCardDeck.Count;wildIndustries=$game.Board.WildIndustryCardDeck.Count;coalMarket=$game.Board.CoalMarket.AvailableCoal;ironMarket=$game.Board.IronMarket.AvailableIron;rounds=$game.GameFlow.GetNumberOfRounds($config.NumberOfPlayers);perPlayer=$perPlayer}
}
$result = [ordered]@{assemblySha256=$hash;runtime=[System.Runtime.InteropServices.RuntimeInformation]::FrameworkDescription;setupCases=$cases}
$json = ConvertTo-Json -InputObject $result -Depth 10
$json | Set-Content -LiteralPath (Join-Path $PSScriptRoot 'baseline-probe-results.json') -Encoding utf8
$json
