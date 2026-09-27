param([ValidateSet(2,3,4)][int]$Players=4,[int]$GamesPerDifficulty=10,[int]$Seed=270900,[string]$OutputTag='extended-ai')
$ErrorActionPreference='Stop'
$root=Split-Path -Parent $PSScriptRoot
$dll=Join-Path $root 'BrassBirminghamUnity_Data/Managed/Assembly-CSharp.dll'
$hash=(Get-FileHash -LiteralPath $dll -Algorithm SHA256).Hash
if($hash -ne '600EBF309670B2726001C16A3375F8BBF3DE76A23C0202BE7CF198BA338981FB'){throw 'Assembly mismatch'}
[void][Reflection.Assembly]::LoadFrom($dll)
$runs=[Collections.Generic.List[object]]::new()
foreach($difficulty in @('Easy','Medium','Hard')){
 for($index=0;$index -lt $GamesPerDifficulty;$index++){
  $config=[BoardGameRules.Entities.Game.GameStartConfiguration]::new([BoardGameRules.Entities.Common.NumberOfPlayers]$Players)
  $config.Seed=$Seed+$index
  $metadata=[BrassApplication.Game.GameMetadata]::new($config,[BrassApplication.Game.GameType]::Local)
  $metadata.GameId='extended-audit-'+$Players+'-'+$difficulty+'-'+$config.Seed
  foreach($pc in $config.Players){
   $pm=[BrassApplication.Game.PlayerMetadata]::new()
   $pm.PlayerNetworkId='audit-'+$pc.Color
   $pm.NickName=$pm.PlayerNetworkId
   $pm.PlayerStartConfiguration=$pc
   $pm.AiType=[BrassApplication.AI.AIType]::$difficulty
   $pm.AIBehaviour=[BrassApplication.AI.AiBehaviors.AiBehaviorFactory]::CreateAiBehavior($pm)
   $metadata.JoinGame($pm)
  }
  $g=[BoardGameRules.Entities.Game.BoardGameRulesGame]::new($config)
  $app=[BrassApplication.Game.BrassApplicationGame]::new($g,$metadata)
  $g.StartGame()
  $turns=0;$actions=0;$checks=0;$debts=0;$scoringChecks=0;$errorDetail=$null;$recent=@();$actionCounts=@{}
  $debtSelector=[BrassApplication.AI.RuleBasedSystem.RuleSelectors.RuleSelectorFactory]::CreatePrioritySelector([BrassApplication.AI.RuleBasedSystem.RuleSetFactory]::CreateSellIndustryRuleSet())
  try {
   while(-not $g.GameProgressInformation.IsGameOver){
    if($turns -gt 100 -or $actions -gt 200){throw 'Progress bound exceeded'}
    $p=$g.GameProgressInformation.CurrentPlayer
    $ai=$metadata.GetPlayerMetadataByColor($p.Color).AIBehaviour
    while($g.TurnFlow.GetRemainingActionsCount() -gt 0){
     $handBefore=$p.Hand.Count
     $revBefore=$g.Revision
     $move=$ai.GetNextAIMove($app)
     if($null -eq $move){throw 'AI returned null'}
     $move.Invoke()
     $a=$g.TurnFlow.CurrentAction
     if($null -eq $a){throw 'AI completed a no-action delegate'}
     $name=$a.GetType().Name
     if(-not $actionCounts.ContainsKey($name)){$actionCounts[$name]=0};$actionCounts[$name]++
     $events=@($g.GetUncommittedEvents() | Where-Object {$_.Metadata.Revision -gt $revBefore})
     $recent=@($events | ForEach-Object {$_.GetType().Name})
     if($handBefore-$p.Hand.Count -ne 1){throw "Unexpected hand delta for $name"}
     $checks++
     $discardCount=@($events | Where-Object {$_ -is [BoardGameRules.Entities.Actions.Events.CardDiscarded]}).Count
     if($discardCount -ne 1){throw "Expected one action discard for $name, got $discardCount"}
     $checks++
     foreach($field in @('NeededCoal','NeededIron','NeededBeer')){
      if($a.PSObject.Properties[$field]){if($a.$field -ne 0){throw "AI finishes $name with $field=$($a.$field)"};$checks++}
     }
     foreach($player in $g.PlayersTrack.Players){
      if($player.Money -lt 0 -or $player.IncomeMarkerPosition -lt 0 -or $player.IncomeMarkerPosition -gt 99 -or $player.Hand.Count -gt 8 -or $player.LinkTileCount -lt 0){throw 'Player invariant failed'}
      $checks++
     }
     foreach($slot in $g.Board.Regions.BuildingSlots){
      $tile=$slot.BuildedIndustry
      if($null -ne $tile){
       if($tile.GetResource() -lt 0){throw 'Negative resource stock'}
       if($tile.IsFlipped -and $tile.GetResource() -gt 0){throw 'Flipped industry retains resources'}
       $checks+=2
      }
     }
     $g.TurnFlow.ConfirmCurrentAction();$actions++
    }
    $g.GameFlow.EndTurn();$turns++
    if($g.GameFlow.IsLastTurnOver()){
     if($g.GameFlow.IsSellIndustryNecessary()){$debts++;$debtSelector.Select($app).MakeAction($app,$false).Invoke()}
     $isEraEnd=$g.GameProgressInformation.CurrentRoundNumber -eq $g.GameFlow.GetNumberOfRounds($config.NumberOfPlayers)
     $expectedVP=@{}
     if($isEraEnd){
      foreach($player in $g.PlayersTrack.Players){
       $vp=$player.VP
       foreach($region in $g.Board.Regions){foreach($slot in $region.BuildingSlots){$tile=$slot.BuildedIndustry;if($tile -and $tile.Color -eq $player.Color -and $tile.IsFlipped){$vp+=$tile.VPForBuilding}}}
       $seen=[Collections.Generic.HashSet[string]]::new()
       foreach($region in $g.Board.Regions){foreach($link in $region.Links){
        if($link.Owner -and $link.Owner.Color -eq $player.Color -and $seen.Add($link.LinkId)){
         foreach($end in @($link.Region1,$link.Region2,$link.Region3)){
          if($null -eq $end){continue}
          if($end -is [BoardGameRules.Entities.Board.Regions.BorderRegion]){$vp+=2}else{foreach($slot in $end.BuildingSlots){if($slot.BuildedIndustry -and $slot.BuildedIndustry.IsFlipped){$vp+=$slot.BuildedIndustry.VPPerLink}}}
         }
        }
       }}
       $expectedVP[$player.Color]=$vp
      }
     }
     $g.GameFlow.EndRound()
     if($isEraEnd -and $debts -eq 0){foreach($player in $g.PlayersTrack.Players){if($player.VP -ne $expectedVP[$player.Color]){throw "Era scoring mismatch for $($player.Color)"};$scoringChecks++}}
    }
   }
  }catch{$errorDetail=[ordered]@{message=$_.Exception.ToString();round=$g.GameProgressInformation.CurrentRoundNumber;era=$g.GameProgressInformation.CurrentGameEra.ToString();player=$g.GameProgressInformation.CurrentPlayer.Color.ToString();recentEvents=$recent;revision=$g.Revision}}
  $runs.Add([ordered]@{players=$Players;difficulty=$difficulty;seed=$config.Seed;complete=$g.GameProgressInformation.IsGameOver;turns=$turns;actions=$actions;checks=$checks;scoringChecks=$scoringChecks;debtPhases=$debts;actionCounts=$actionCounts;error=$errorDetail})
 }
}
$report=[ordered]@{assemblySha256=$hash;runtime=[Runtime.InteropServices.RuntimeInformation]::FrameworkDescription;purpose='Original AI multi-seed progression with action payment, card, resource, and independently recomputed era score checks; no Unity or saves.';runs=@($runs.ToArray())}
$report|ConvertTo-Json -Depth 12|Set-Content -LiteralPath (Join-Path $PSScriptRoot "$OutputTag-$Players-results.json") -Encoding utf8
[ordered]@{players=$Players;runs=$runs.Count;complete=@($runs|Where-Object complete).Count;errors=@($runs|Where-Object error).Count;actions=($runs|ForEach-Object {$_.actions}|Measure-Object -Sum).Sum;checks=($runs|ForEach-Object {$_.checks}|Measure-Object -Sum).Sum;scoringChecks=($runs|ForEach-Object {$_.scoringChecks}|Measure-Object -Sum).Sum}|ConvertTo-Json
