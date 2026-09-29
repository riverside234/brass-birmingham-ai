$ErrorActionPreference='Stop'
$root=Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$assemblyPath=Join-Path $root 'BrassBirminghamUnity_Data/Managed/Assembly-CSharp.dll'
if((Get-FileHash $assemblyPath).Hash -ne '34DD7A8200DA1BEFE41FF7532014E2CBFA6E6EEECCF2BCCAF3CB70DC7E272B5A'){throw 'Unexpected installed patch'}
if((Get-FileHash (Join-Path $root 'BrassBirminghamUnity_Data/Managed/BrassPatch.dll')).Hash -ne '4D574D81F40D7A2CFFD58AB73C9087BA219FF25456D3A9AF9080B05E8A1AB76D'){throw 'Unexpected helper patch'}
[void][Reflection.Assembly]::LoadFrom($assemblyPath)
$results=@()
foreach($players in 2,3,4){
 $config=[BoardGameRules.Entities.Game.GameStartConfiguration]::new([BoardGameRules.Entities.Common.NumberOfPlayers]$players);$config.Seed=20260928
 $g=[BoardGameRules.Entities.Game.BoardGameRulesGame]::new($config);$g.StartGame()
 $target=$g.GameProgressInformation.CurrentPlayer.Color;$step=0;$trace=@();$allActions=0
 while($g.GameProgressInformation.CurrentRoundNumber -lt 7 -or $g.GameProgressInformation.CurrentPlayer.Color -ne $target){
  if($g.GameProgressInformation.CurrentRoundNumber -gt 7){throw 'Missed target turn'}
  $p=$g.GameProgressInformation.CurrentPlayer
  while($g.TurnFlow.GetRemainingActionsCount() -gt 0){
   $before=$p.Money;$kind='Pass';$quantity=0;$beforeHand=$p.Hand.Count;$beforeActions=$g.TurnFlow.GetRemainingActionsCount()
   if($p.Color -eq $target -and $step -lt 3){
    $a=$g.TurnFlow.StartLoanAction()
    if($a.GetLoanActionNotPossibleReasons($g.GameProgressInformation).Count){throw 'Illegal loan in trace'}
    $a.DiscardCard($p.Hand[0],$p);$a.Loan($p.Color);$kind='Loan';$step++
   }elseif($p.Color -eq $target -and $step -lt 11){
    $type=if($step -lt 7){[BoardGameRules.Entities.Board.BuildingSlots.BuildingType]::Brewery}else{[BoardGameRules.Entities.Board.BuildingSlots.BuildingType]::CoalMine}
    $quantity=if($step -eq 6 -or $step -eq 10){1}else{2}
    $a=$g.TurnFlow.StartDevelopAction()
    if($a.GetDevelopActionNotPossibleReasons($g.GameProgressInformation).Count){throw 'Illegal Develop in trace'}
    if($p.Money -lt $g.Board.IronMarket.IronPrice($quantity)){throw 'Unaffordable Develop in trace'}
    $a.DiscardCard($p.Hand[0],$p)
    for($n=0;$n -lt $quantity;$n++){
     if(-not $p.Mat.CanIndustryBeDeveloped($type)){throw 'Prohibited Develop in trace'}
     $a.Develop($type)
    }
    $a.SupplyIron($g.Board.IronMarket,$quantity)
    if($a.NeededIron -ne 0){throw 'Unpaid iron'}
    $kind='Develop '+$type;$step++
   }else{$g.TurnFlow.StartPassAction().Pass($p.Hand[0],$p)}
   $g.TurnFlow.ConfirmCurrentAction();$allActions++
   if($p.Money -lt 0 -or $p.Hand.Count -ne $beforeHand-1){throw 'Invalid action result'}
   if($g.TurnFlow.GetRemainingActionsCount() -ne $beforeActions-1){throw 'Invalid action-count decrement'}
   if($p.Color -eq $target){$trace+=[ordered]@{round=$g.GameProgressInformation.CurrentRoundNumber;action=$kind;tiles=$quantity;cashBefore=$before;cashAfter=$p.Money;income=$p.IncomeLevel}}
  }
  $g.GameFlow.EndTurn()
  if($g.GameFlow.IsLastTurnOver()){$g.GameFlow.EndRound()}
 }
 $p=$g.GameProgressInformation.CurrentPlayer
 $app=[BrassApplication.Game.BrassApplicationGame]::new($g,[BrassApplication.Game.GameMetadata]::new($config,[BrassApplication.Game.GameType]::Local))
 $errorText=$null
 try{[void][BrassApplication.AI.AiBehaviors.HardAiBehavior]::new('trace').GetNextAIMove($app)}catch{$errorText=$_.Exception.ToString()}
 $copy=[BoardGameRules.Entities.Game.BoardGameRulesGame]::new($config)
 foreach($event in $g.GetUncommittedEvents()){$copy.ApplyEvent($event)}
 $copyPlayer=$copy.GameProgressInformation.CurrentPlayer
 if($copyPlayer.Money -ne $p.Money -or $copyPlayer.IncomeLevel -ne $p.IncomeLevel -or $copy.GameProgressInformation.CurrentRoundNumber -ne 7 -or $copyPlayer.Color -ne $p.Color){throw 'Event replay changed the trigger state'}
 $replayError=$null
 $copyApp=[BrassApplication.Game.BrassApplicationGame]::new($copy,[BrassApplication.Game.GameMetadata]::new($config,[BrassApplication.Game.GameType]::Local))
 try{[void][BrassApplication.AI.AiBehaviors.HardAiBehavior]::new('replay').GetNextAIMove($copyApp)}catch{$replayError=$_.Exception.ToString()}
 $results+=[ordered]@{players=$players;seed=$config.Seed;target=$target.ToString();round=$g.GameProgressInformation.CurrentRoundNumber;era=$g.GameProgressInformation.CurrentGameEra.ToString();money=$p.Money;income=$p.IncomeLevel;breweryTiles=$p.Mat.GetIndustryList([BoardGameRules.Entities.Board.BuildingSlots.BuildingType]::Brewery).Count;coalTiles=$p.Mat.GetIndustryList([BoardGameRules.Entities.Board.BuildingSlots.BuildingType]::CoalMine).Count;hand=$p.Hand.Count;totalLegalActions=$allActions;revision=$g.Revision;targetActions=$trace;exception=$errorText;eventReplayException=$replayError;limitation='Legal scripted choices, not a history chosen by the original Hard AI policy. No direct board, mat, cash, income, or round mutation; in-memory event replay, not a user save.'}
}
$results|ConvertTo-Json -Depth 9|Set-Content -LiteralPath (Join-Path $PSScriptRoot 'ai-brewery-legal-results.json') -Encoding UTF8
$results|ForEach-Object {[pscustomobject]$_}|Select-Object players,round,money,income,breweryTiles,coalTiles,totalLegalActions,@{n='failed';e={$null -ne $_.exception}}|ConvertTo-Json
