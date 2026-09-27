$ErrorActionPreference='Stop'
$null = . (Join-Path $PSScriptRoot 'extended-phases-probes.ps1')
function Snapshot-Game($app) {
 $g=$app.Game; $progress=$g.GameProgressInformation
 $players=@($g.PlayersTrack.Players | ForEach-Object {
  $p=$_; $mat=@()
  foreach($type in [Enum]::GetValues([BoardGameRules.Entities.Board.BuildingSlots.BuildingType])) {
   if($type.ToString() -ne 'Undefined') { $mat+=@($p.Mat.GetIndustryList($type) | ForEach-Object {$_.UniqueId}) }
  }
  [ordered]@{ color=$p.Color.ToString(); money=$p.Money; incomeMarker=$p.IncomeMarkerPosition; vp=$p.VP; bonusVp=$p.VPFromBonusesInCurrentEra; links=$p.LinkTileCount; spend=$g.PlayersTrack.SpendMoneyInCurrentRound[$p.Color]; hand=@($p.Hand|ForEach-Object{$_.UniqueId}); discards=@($p.DiscardPile|ForEach-Object{$_.UniqueId}); mat=$mat }
 })
 $board=@($g.Board.Regions|ForEach-Object {$r=$_; foreach($s in $r.BuildingSlots) { $i=$s.BuildedIndustry; [ordered]@{slot=$s.UniqueId; industry=$i.UniqueId; flipped=$i.IsFlipped; resource=$(if($null -ne $i){$i.GetResource()}else{$null})} } })
 $links=@($g.Board.Regions | ForEach-Object {$_.Links} | Sort-Object LinkId -Unique | ForEach-Object {[ordered]@{id=$_.LinkId;owner=$_.Owner.Color}})
 $merchants=@($g.Board.Regions|ForEach-Object {$_.MerchantSlots}|Where-Object{$null -ne $_}|ForEach-Object {[ordered]@{id=$_.UniqueId; beer=$_.AvailableBeer; types=@($_.MerchantTile.BuildingTypes|Where-Object{$null -ne $_}|ForEach-Object{$_.ToString()})}})
 return [ordered]@{ revision=$g.Revision; era=$progress.CurrentGameEra.ToString(); round=$progress.CurrentRoundNumber; turn=$progress.CurrentTurnNumber; player=$progress.CurrentPlayer.Color.ToString(); actions=$g.TurnFlow.GetRemainingActionsCount(); gameOver=$progress.IsGameOver; players=$players; board=$board; links=$links; merchants=$merchants; deck=@($g.Board.CardDeck|ForEach-Object{$_.UniqueId}); wilds=@($g.Board.WildLocationCardDeck|ForEach-Object{$_.UniqueId})+@($g.Board.WildIndustryCardDeck|ForEach-Object{$_.UniqueId}); removed=@($g.Board.RemovedIndustries|ForEach-Object{$_.UniqueId}); coalSpace=$g.Board.CoalMarket.AmountOfResourcePossibleToSell(); ironSpace=$g.Board.IronMarket.AmountOfResourcePossibleToSell(); rngCount=$g.Randomer.RandomerInitData.FastForwardCount; ranking=@($progress.EndGameRanking|Where-Object{$null -ne $_}|ForEach-Object{$_.ToString()}); endingPoints=@(if($null -ne $progress.EndGamePoints){foreach($entry in $progress.EndGamePoints.GetEnumerator()){[ordered]@{color=$entry.Key.ToString();values=@($entry.Value.Item1,$entry.Value.Item2,$entry.Value.Item3,$entry.Value.Item4,$entry.Value.Item5)}}}); introPoints=@(if($null -ne $progress.IntroductoryPoints){foreach($entry in $progress.IntroductoryPoints.GetEnumerator()){[ordered]@{color=$entry.Key.ToString();values=@($entry.Value.Item1,$entry.Value.Item2,$entry.Value.Item3)}}}) }
}
function Different-Keys($a,$b) {
 return @($a.Keys|Where-Object {(ConvertTo-Json -InputObject $a[$_] -Depth 15 -Compress) -ne (ConvertTo-Json -InputObject $b[$_] -Depth 15 -Compress)})
}
$runs=@()
foreach($scenario in @(@(2,20260927),@(3,20260927),@(4,20260927),@(3,270906),@(4,270904))) {
 $count=$scenario[0]; $caseSeed=$scenario[1]
 $case=New-Case $caseSeed $false $count; $app=$case.App; $repo=$case.Repo; $turns=0; $deltas=@(); $debtRounds=0
 while(-not $app.Game.GameProgressInformation.IsGameOver) {
  $p=$app.Game.GameProgressInformation.CurrentPlayer; $behavior=$app.Metadata.GetPlayerMetadataByColor($p.Color).AIBehaviour
  while($app.Game.TurnFlow.GetRemainingActionsCount() -gt 0) { $behavior.GetNextAIMove($app).Invoke(); $app.Game.TurnFlow.ConfirmCurrentAction() }
  $app.Game.GameFlow.EndTurn()
  if($app.Game.GameFlow.IsLastTurnOver()) {
   if($app.Game.GameFlow.IsSellIndustryNecessary()) { $debtRounds++; $selector=[BrassApplication.AI.RuleBasedSystem.RuleSelectors.RuleSelectorFactory]::CreatePrioritySelector([BrassApplication.AI.RuleBasedSystem.RuleSetFactory]::CreateSellIndustryRuleSet()); $selector.Select($app).MakeAction($app,$false).Invoke() }
   $app.Game.GameFlow.EndRound()
  }
  $repo.Save($app); $turns++
  $before=Snapshot-Game $app; $reloaded=$repo.Load($app.Metadata.GameId); $after=Snapshot-Game $reloaded
  $keys=Different-Keys $before $after
  if($keys.Count -gt 0) {
   $beforePlayerCore=@($before.players|ForEach-Object{$p=$_;$core=[ordered]@{};foreach($key in $p.Keys){if($key -ne 'bonusVp'){$core[$key]=$p[$key]}};$core})
   $afterPlayerCore=@($after.players|ForEach-Object{$p=$_;$core=[ordered]@{};foreach($key in $p.Keys){if($key -ne 'bonusVp'){$core[$key]=$p[$key]}};$core})
   $otherPlayerFieldsMatch=(ConvertTo-Json -InputObject $beforePlayerCore -Depth 15 -Compress) -eq (ConvertTo-Json -InputObject $afterPlayerCore -Depth 15 -Compress)
   $deltas += [ordered]@{turn=$turns; era=$before.era; round=$before.round; differences=$keys; otherPlayerFieldsMatch=$otherPlayerFieldsMatch; originalBonusVp=@($before.players|ForEach-Object{$_.bonusVp}); loadedBonusVp=@($after.players|ForEach-Object{$_.bonusVp})}
  }
 }
 $runs += [ordered]@{players=$count; seed=$caseSeed; turns=$turns; savedLoadedTurns=$turns; debtRounds=$debtRounds; differences=$deltas}
}
$case=New-Case 20260927
$history=[BrassApplication.GameHistory.BrassGameHistory]::new(); $history.Repository=$case.Repo; $history.SetGame($case.App)
$before=Snapshot-Game $history.CurrentGame
$p=$history.CurrentGame.Game.GameProgressInformation.CurrentPlayer
$a=$history.CurrentGame.Game.TurnFlow.StartBuildAction(); $a.DiscardCard($p.Hand[0],$p)
$history.CurrentGame.Game.TurnFlow.CancelCurrentAction(); $history.CancelAction()
$cancelKeys=@(Different-Keys $before (Snapshot-Game $history.CurrentGame))
$p=$history.CurrentGame.Game.GameProgressInformation.CurrentPlayer
$behavior=$history.CurrentGame.Metadata.GetPlayerMetadataByColor($p.Color).AIBehaviour
$behavior.GetNextAIMove($history.CurrentGame).Invoke(); $history.CurrentGame.Game.TurnFlow.ConfirmCurrentAction()
$history.UndoAction()
$undoKeys=@(Different-Keys $before (Snapshot-Game $history.CurrentGame))
$case=New-Case 20260927
while($case.App.Game.GameProgressInformation.CurrentGameEra.ToString() -eq 'CanalEra') {
 $p=$case.App.Game.GameProgressInformation.CurrentPlayer; $behavior=$case.App.Metadata.GetPlayerMetadataByColor($p.Color).AIBehaviour
 while($case.App.Game.TurnFlow.GetRemainingActionsCount() -gt 0){$behavior.GetNextAIMove($case.App).Invoke();$case.App.Game.TurnFlow.ConfirmCurrentAction()}
 Complete-Turn $case.App; $case.Repo.Save($case.App)
}
$resumeBranches=@()
$resumed=$case.Repo.Load($case.App.Metadata.GameId)
foreach($branch in @([pscustomobject]@{label='uninterrupted';app=$case.App},[pscustomobject]@{label='resumed';app=$resumed})) {
 while(-not $branch.app.Game.GameProgressInformation.IsGameOver){Pass-AllActions $branch.app; Complete-Turn $branch.app}
 $resumeBranches += [ordered]@{branch=$branch.label; finalPlayers=@($branch.app.Game.PlayersTrack.Players|ForEach-Object{$p=$_;$points=$branch.app.Game.GameProgressInformation.EndGamePoints[$p.Color];[ordered]@{color=$p.Color.ToString();totalVp=$p.VP;railBonusVp=$points.Item3;displayedCanalVp=$p.VP-$points.Item1-$points.Item2-$points.Item3}})}
}
$result=[ordered]@{purpose='Original-AI full games, comparing original state to InMemory repository Save/Load after every completed turn. Includes scoring counters, decks, hands, mats, resources, links, merchants, round order and PRNG advancement.'; runs=$runs; partialBuildCancelDifferenceKeys=$cancelKeys; confirmedFirstActionUndoDifferenceKeys=$undoKeys; railResumeScoring=$resumeBranches}
$json=ConvertTo-Json $result -Depth 12
$json | Set-Content -LiteralPath (Join-Path $PSScriptRoot 'extended-phases-roundtrip-results.json') -Encoding utf8
$json
