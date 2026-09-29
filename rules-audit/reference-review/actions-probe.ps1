using namespace BoardGameRules.Entities.Actions
using namespace BoardGameRules.Entities.Board.BuildingSlots
using namespace BoardGameRules.Entities.Board.Regions
using namespace BoardGameRules.Entities.Cards
using namespace BoardGameRules.Entities.Common
using namespace BoardGameRules.Entities.Game
using namespace BoardGameRules.Entities.GamePhases
using namespace BrassApplication.AI.AIHelpers

$ErrorActionPreference='Stop'
$gameRoot=Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$managed=Join-Path $gameRoot 'BrassBirminghamUnity_Data/Managed'
$dll=Join-Path $managed 'Assembly-CSharp.dll'
$helper=Join-Path $managed 'BrassPatch.dll'
$hash=(Get-FileHash -LiteralPath $dll -Algorithm SHA256).Hash
$helperHash=(Get-FileHash -LiteralPath $helper -Algorithm SHA256).Hash
if($hash -ne '34DD7A8200DA1BEFE41FF7532014E2CBFA6E6EEECCF2BCCAF3CB70DC7E272B5A' -or $helperHash -ne '4D574D81F40D7A2CFFD58AB73C9087BA219FF25456D3A9AF9080B05E8A1AB76D'){throw 'Current installed fingerprints changed.'}
[void][Reflection.Assembly]::LoadFrom($dll)
[void][Reflection.Assembly]::LoadFrom($helper)
$checks=[Collections.Generic.List[object]]::new()
function Check($name,$expected,$actual){$checks.Add([ordered]@{name=$name;expected=$expected;actual=$actual;passed=($expected -eq $actual)})}
function New-Game($count){$cfg=[GameStartConfiguration]::new([NumberOfPlayers]$count);$cfg.Seed=20260927;$g=[BoardGameRulesGame]::new($cfg);$g.StartGame();return $g}
function Put($g,$owner,$regionName,$type,$level){$tile=@($owner.Mat.GetIndustryList($type)|Where-Object Level -eq $level)[0];[void]$owner.Mat.RemoveGivenBuildingFromMat($tile);$slot=@($g.Board.GetRegionByName($regionName).BuildingSlots|Where-Object{$null -eq $_.BuildedIndustry -and $_.BuildingTypes.Contains($type)})[0];$slot.BuildIndustry($tile);$tile.Build();return $tile}
function Location-Card($p,$r){$card=[LocationCard]::new('review-card',[NumberOfPlayers]::Two,$r.Name,$r.Color);$p.Hand.Clear();$p.Hand.Add($card);return $card}

foreach($count in 2,3,4){
    # An industry card can start anywhere when the player has no board presence.
    # Once a tile exists, a disconnected industry-card build is illegal.
    $g=New-Game $count;$p=$g.GameProgressInformation.CurrentPlayer;$b=$g.Board
    $type=[BuildingType]::CottonMill;$r=$b.GetRegionByName([RegionName]::Worcester)
    $slot=@($r.BuildingSlots|Where-Object{$_.BuildingTypes.Count -eq 1 -and $_.BuildingTypes.Contains($type)})[0]
    $types=[Collections.Generic.List[BuildingType]]::new();$types.Add($type)
    $card=[IndustryCard]::new('review-industry',[NumberOfPlayers]::Two,$types)
    $p.Hand.Clear();$p.Hand.Add($card);$p.Money=100
    $tile=$p.Mat.GetFirstBuildingOfType($type)
    Check "$count empty-network industry card" $true ([BuildAction]::CanUseBuildActionForGivenSlot($slot,$tile,$p,$b,$g.GameProgressInformation))
    [void](Put $g $p ([RegionName]::Stafford) ([BuildingType]::Brewery) 1)
    Check "$count disconnected industry card" $false ([BuildAction]::CanUseBuildActionForGivenSlot($slot,$tile,$p,$b,$g.GameProgressInformation))
    [void](Location-Card $p $r)
    Check "$count disconnected location card" $true ([BuildAction]::CanUseBuildActionForGivenSlot($slot,$tile,$p,$b,$g.GameProgressInformation))

    # Two-coal construction must deplete the near mine before the farther mine.
    $g=New-Game $count;$p=$g.GameProgressInformation.CurrentPlayer;$b=$g.Board
    $op=@($g.PlayersTrack.Players|Where-Object Color -ne $p.Color)[0]
    $g.GameProgressInformation.CurrentGameEra=[GameEra]::RailEra;$p.Money=22
    $p.Mat.DevelopIndustry([BuildingType]::Pottery);$p.Mat.DevelopIndustry([BuildingType]::Pottery)
    $near=Put $g $op ([RegionName]::Coventry) ([BuildingType]::CoalMine) 2
    $far=Put $g $op ([RegionName]::Nuneaton) ([BuildingType]::CoalMine) 2
    $near.AvailableCoal=1;$far.AvailableCoal=1
    $b.GetLinkById('RailCoventryNuneaton').SetOwner($p)
    $r=$b.GetRegionByName([RegionName]::Coventry);$card=Location-Card $p $r
    $slot=@($r.BuildingSlots|Where-Object{$_.BuildingTypes.Contains([BuildingType]::Pottery)})[0]
    $tile=$p.Mat.GetFirstBuildingOfType([BuildingType]::Pottery)
    Check "$count two-mine build affordability" $true ([BuildAction]::CanUseBuildActionForGivenSlot($slot,$tile,$p,$b,$g.GameProgressInformation))
    $action=$g.TurnFlow.StartBuildAction()
    [BuildActionAIHelper]::BuildIndustry($action,$card,$slot,[BuildingType]::Pottery,$b,$p)
    Check "$count AI build paid resources" 0 ($action.NeededCoal+$action.NeededIron)
    Check "$count AI build paid exact cash" 0 $p.Money
    Check "$count AI build depleted both mines" $true ($near.IsFlipped -and $far.IsFlipped)

    # Rail source selection compares all endpoints globally.
    $g=New-Game $count;$p=$g.GameProgressInformation.CurrentPlayer;$b=$g.Board
    $g.GameProgressInformation.CurrentGameEra=[GameEra]::RailEra;$p.Money=30
    $near=Put $g $p ([RegionName]::Dudley) ([BuildingType]::CoalMine) 2
    $far=Put $g $p ([RegionName]::Redditch) ([BuildingType]::CoalMine) 2
    $b.GetLinkById('RailBirminghamRedditch').SetOwner($p)
    $link=$b.GetLinkById('RailBirminghamDudley');$action=$g.TurnFlow.StartNetworkAction()
    Check "$count rail candidate offered" $true $action.GetValidConnectionSlots($b,$g.GameProgressInformation).Contains($link)
    $action.DiscardCard($p.Hand[0],$p);$action.PlaceLink($link)
    $sources=[BrassPatch.ResourcesFixes]::ClosestCoalMines($b,[Region[]]@($link.Region1,$link.Region2,$link.Region3))
    Check "$count rail global nearest" $near.UniqueId $sources[0].UniqueId
    $rejected=$false;try{$action.SupplyCoal($far,1)}catch{$rejected=$true}
    Check "$count rail farther mine rejected" $true $rejected
    $stock=$near.AvailableCoal;$action.SupplyCoal($near,1)
    Check "$count rail coal settled" 0 $action.NeededCoal
    Check "$count rail nearest consumed" ($stock-1) $near.AvailableCoal

    # Develop can combine one board iron with one market iron, regardless of
    # board connectivity, and the ordinary helper re-queries after depletion.
    $g=New-Game $count;$p=$g.GameProgressInformation.CurrentPlayer;$b=$g.Board
    $op=@($g.PlayersTrack.Players|Where-Object Color -ne $p.Color)[0]
    $iron=Put $g $op ([RegionName]::Birmingham) ([BuildingType]::IronWorks) 1
    $iron.AvailableIron=1;[void]$b.IronMarket.Buy(100);[void]$b.IronMarket.Sell(1)
    $price=$b.IronMarket.IronPrice();$p.Money=$price;$action=$g.TurnFlow.DevelopAction
    Check "$count mixed-iron develop affordable" $true $action.HasPlayerIronForSecondDevelop($p)
    $p.Money=$price-1;Check "$count mixed-iron second develop short cash" $false $action.HasPlayerIronForSecondDevelop($p)
    $p.Money=$price;$before=$p.Mat.GetIndustryList([BuildingType]::CottonMill).Count
    [DevelopActionAIHelper]::DevelopIndustry($action,$p.Hand[0],[BuildingType]::CottonMill,[BuildingType]::CottonMill,$b,$p)
    Check "$count develop removes exactly two" ($before-2) $p.Mat.GetIndustryList([BuildingType]::CottonMill).Count
    Check "$count develop clears iron debt" 0 $action.NeededIron
    Check "$count develop spends exact cash" 0 $p.Money
    Check "$count develop board works flips" $true $iron.IsFlipped
    Check "$count forbidden Pottery I not developable" $false $p.Mat.CanIndustryBeDeveloped([BuildingType]::Pottery)
}
$report=[ordered]@{utc=[DateTime]::UtcNow.ToString('o');assemblySha256=$hash;helperSha256=$helperHash;method='Focused actual-DLL predicates and original AI helpers; arranged independent in-memory fixtures; no user saves or Unity UI';assertions=$checks.Count;failures=@($checks|Where-Object{-not $_.passed});checks=$checks}
$report|ConvertTo-Json -Depth 8|Set-Content -LiteralPath (Join-Path $PSScriptRoot 'actions-results.json') -Encoding utf8
[pscustomobject]$report|Select-Object utc,assemblySha256,helperSha256,assertions,@{n='failures';e={$_.failures.Count}}|ConvertTo-Json
if($report.failures.Count -gt 0){exit 1}
