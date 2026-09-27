$ErrorActionPreference='Stop'
$root=Split-Path -Parent $PSScriptRoot
$dll=Join-Path $root 'BrassBirminghamUnity_Data/Managed/Assembly-CSharp.dll'
$hash=(Get-FileHash -LiteralPath $dll -Algorithm SHA256).Hash
if($hash -ne '600EBF309670B2726001C16A3375F8BBF3DE76A23C0202BE7CF198BA338981FB'){throw 'Assembly mismatch'}
[void][Reflection.Assembly]::LoadFrom($dll)
$cfg=[BoardGameRules.Entities.Game.GameStartConfiguration]::new([BoardGameRules.Entities.Common.NumberOfPlayers]::Four)
$cfg.Seed=984130
$g=[BoardGameRules.Entities.Game.BoardGameRulesGame]::new($cfg)
$g.StartGame()
$p=$g.PlayersTrack.GetPlayerByColor([BoardGameRules.Entities.Players.PlayerColor]::Red)
$brew=$p.Mat.GetIndustryList([BoardGameRules.Entities.Board.BuildingSlots.BuildingType]::Brewery)|Where-Object UniqueId -eq 'industry944625454'
$cotton=$p.Mat.GetIndustryList([BoardGameRules.Entities.Board.BuildingSlots.BuildingType]::CottonMill)|Where-Object UniqueId -eq 'industry944625454'
if(-not $brew -or -not $cotton){throw 'Expected startup collision missing'}
# Arrange only placement and era; preserve the original colliding objects and IDs.
$g.GameProgressInformation.CurrentGameEra=[BoardGameRules.Entities.GamePhases.GameEra]::RailEra
$slots=@($g.Board.Regions.BuildingSlots)
$cottonSlot=$slots|Where-Object {$_.CanBuildIndustryType($cotton.BuildingType)}|Select-Object -First 1
$brewSlot=$slots|Where-Object {$_.CanBuildIndustryType($brew.BuildingType)}|Select-Object -Last 1
[void]$p.Mat.RemoveGivenBuildingFromMat($brew)
[void]$p.Mat.RemoveGivenBuildingFromMat($cotton)
$cottonSlot.BuildIndustry($cotton);$cotton.Build()
$brewSlot.BuildIndustry($brew);$brew.Build()
$cottonRegion=$g.Board.GetRegionContainingGivenIndustry($cotton)
$brewRegion=$g.Board.GetRegionContainingGivenIndustry($brew)
$beerBefore=$brew.AvailableBeer
$incomeBefore=$p.IncomeMarkerPosition
$g.TurnFlow.StartSellAction()
$consuming=$g.TurnFlow.CurrentAction.GeneralAction.ConsumingBeer
$regions=[Collections.Generic.List[BoardGameRules.Entities.Board.Regions.Region]]::new()
$regions.Add($brewRegion)
$consuming.SupplyBeer($brew,$beerBefore,$beerBefore,$p,$regions)
$result=[ordered]@{
 assemblySha256=$hash;runtime=[Runtime.InteropServices.RuntimeInformation]::FrameworkDescription
 seed=984130;collisionId=$brew.UniqueId
 method='Original seed-generated tiles, placed in an isolated Rail fixture without altering IDs; invoke original ConsumingBeer.SupplyBeer to exhaust the brewery. No complete legal action history to this placement is claimed.'
 cottonRegion=$cottonRegion.Name.ToString();breweryRegion=$brewRegion.Name.ToString()
 breweryBeerBefore=$beerBefore;breweryBeerAfter=$brew.AvailableBeer
 breweryFlipped=$brew.IsFlipped;cottonFlippedWithoutSale=$cotton.IsFlipped
 incomeIncrease=$p.IncomeMarkerPosition-$incomeBefore
 emittedFlip=@($g.GetUncommittedEvents()|Where-Object {$_ -is [BoardGameRules.Entities.Actions.Events.IndustryFlipped]}|ForEach-Object {[ordered]@{id=$_.IndustryUniqueId;region=$_.RegionName.ToString()}})
}
if($brew.IsFlipped -or -not $cotton.IsFlipped -or $brew.AvailableBeer -ne 0){throw 'Collision did not reproduce expected wrong flip'}
$result|ConvertTo-Json -Depth 8|Set-Content -LiteralPath (Join-Path $PSScriptRoot 'extended-identity-impact-results.json') -Encoding utf8
$result|ConvertTo-Json -Depth 8
