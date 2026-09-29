$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$assemblyPath = Join-Path $root 'BrassBirminghamUnity_Data/Managed/Assembly-CSharp.dll'
$helperPath = Join-Path $root 'BrassBirminghamUnity_Data/Managed/BrassPatch.dll'
$hash = (Get-FileHash -LiteralPath $assemblyPath -Algorithm SHA256).Hash
$helperHash = (Get-FileHash -LiteralPath $helperPath -Algorithm SHA256).Hash
if ($hash -ne '34DD7A8200DA1BEFE41FF7532014E2CBFA6E6EEECCF2BCCAF3CB70DC7E272B5A' -or
    $helperHash -ne '4D574D81F40D7A2CFFD58AB73C9087BA219FF25456D3A9AF9080B05E8A1AB76D') { throw 'Unreviewed patched assembly version' }
[void][System.Reflection.Assembly]::LoadFrom($helperPath)
[void][System.Reflection.Assembly]::LoadFrom($assemblyPath)
function Key($value) {
    $s = $value.ToString()
    if ($s -eq 'Central') { return 'northern' }
    if ($s -eq 'SouthWest') { return 'southern' }
    return $s.Substring(0,1).ToLowerInvariant() + $s.Substring(1)
}
$cases = @()
foreach ($count in @(2,3,4)) {
    $config = [BoardGameRules.Entities.Game.GameStartConfiguration]::new([BoardGameRules.Entities.Common.NumberOfPlayers]$count)
    $config.Seed = 20260927
    $game = [BoardGameRules.Entities.Game.BoardGameRulesGame]::new($config)
    $game.StartGame()
    $p = $game.PlayersTrack.Players[0]
    $industries = @()
    foreach ($type in @('CoalMine','IronWorks','Pottery','Brewery','CottonMill','Manufacturer')) {
        foreach ($tile in $p.Mat.GetIndustryList([BoardGameRules.Entities.Board.BuildingSlots.BuildingType]::$type)) {
            $tile.Build()
            $industries += [ordered]@{
                type=(Key $tile.BuildingType);level=$tile.Level;vp=$tile.VPForBuilding;linkVP=$tile.VPPerLink;income=$tile.IncomeIncrease
                canDevelop=$tile.CanBeDeveloped;canalEra=(-not $tile.IsForRailEra);railEra=(-not $tile.IsForCanalEra)
                cost=$tile.MoneyCost;costCoal=$tile.CoalCost;costIron=$tile.IronCost
                beersToSell=$(if($tile -is [BoardGameRules.Entities.Industries.BaseSellableIndustry]){$tile.GetBeerNeeded()}else{$null})
                resourceCubes=$tile.GetResource()
            }
        }
    }
    $cities = [ordered]@{}
    $merchants = [ordered]@{}
    $merchantTiles = @()
    foreach ($r in $game.Board.Regions) {
        if ($r -is [BoardGameRules.Entities.Board.Regions.BorderRegion]) {
            $merchants[(Key $r.Name)] = [ordered]@{slots=$r.MerchantSlots.Count;minPlayers=[int]$r.MerchantSlots[0].MinNumberOfPlayers;bonusType=$r.MerchantBeerBonusType.ToString().ToLowerInvariant()}
            foreach ($slot in $r.MerchantSlots) {
                if ($null -ne $slot.MerchantTile) {
                    $types = @($slot.MerchantTile.BuildingTypes | ForEach-Object { Key $_ })
                    $merchantTiles += [ordered]@{location=(Key $r.Name);type=$(if($types.Count -eq 0){'blank'}elseif($types.Count -eq 3){'any'}else{$types[0]});minPlayers=[int]$slot.MerchantTile.MinNumberOfPlayers;beer=$slot.AvailableBeer}
                }
            }
        } else {
            $cities[(Key $r.Name)] = @($r.BuildingSlots | ForEach-Object { ,@($_.BuildingTypes | ForEach-Object {Key $_}) })
        }
    }
    $links = @($game.Board.Regions | ForEach-Object { $_.Links } | Sort-Object LinkId -Unique | ForEach-Object {
        $link = $_
        $ends = @($link.Region1,$link.Region2,$link.Region3) | Where-Object {$null -ne $_} | ForEach-Object {Key $_.Name} | Sort-Object
        [ordered]@{era=$link.GetType().Name.ToLowerInvariant();ends=@($ends)}
    })
    $cards = @($game.Board.CardDeck) + @($game.PlayersTrack.Players | ForEach-Object { $_.Hand; $_.DiscardPile })
    $cardNames = @($cards | ForEach-Object {
        if ($_ -is [BoardGameRules.Entities.Cards.LocationCard]) { 'location:'+(Key $_.Region) }
        elseif ($_ -is [BoardGameRules.Entities.Cards.IndustryCard]) { 'industry:'+(@($_.BuildingTypes | ForEach-Object {Key $_} | Sort-Object) -join '+') }
        else { throw 'Unexpected normal-deck card' }
    })
    $players = @($game.PlayersTrack.Players | ForEach-Object { [ordered]@{money=$_.Money;incomeMarker=$_.IncomeMarkerPosition;hand=$_.Hand.Count;discard=$_.DiscardPile.Count;linkTiles=$_.LinkTileCount} })
    $cases += [ordered]@{players=$count;industries=$industries;cities=$cities;merchants=$merchants;merchantTiles=$merchantTiles;links=$links;cards=$cardNames;playerSetup=$players;coalMarket=$game.Board.CoalMarket.AvailableCoal;ironMarket=$game.Board.IronMarket.AvailableIron;wildIndustry=$game.Board.WildIndustryCardDeck.Count;wildLocation=$game.Board.WildLocationCardDeck.Count}
}
$result = [ordered]@{assemblySha256=$hash;helperSha256=$helperHash;runtime=[System.Runtime.InteropServices.RuntimeInformation]::FrameworkDescription;seed=20260927;cases=$cases}
$result | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath (Join-Path $PSScriptRoot 'components-dll-data.json') -Encoding utf8
Write-Output 'Component dump complete for current patched DLL: 2, 3, and 4 players. No game files or saves changed.'
