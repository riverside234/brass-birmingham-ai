param([int]$Seed = 20260927)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$dll = Join-Path $root 'BrassBirminghamUnity_Data\Managed\Assembly-CSharp.dll'
if ((Get-FileHash -LiteralPath $dll -Algorithm SHA256).Hash -ne '600EBF309670B2726001C16A3375F8BBF3DE76A23C0202BE7CF198BA338981FB') { throw 'Wrong DLL version' }
[void][Reflection.Assembly]::LoadFrom($dll)
$stub = @'
using System;
using System.Collections.Generic;
using System.Reflection;
using System.Reflection.Emit;
using BoardGameRules.Entities.EventSourcing.Events;
using BrassApplication.Game;
using BrassApplication.UseCases.Game.Replay;
public class ExtendedReplayProbeOutput : IReplayOutput {
 public IEvent LastEvent;
 public IEventReplayDelegate Callback;
 public void UpdateCurrentPlayer(string s) {}
 public void UpdateCurrentPlayerAction(IEvent e) {}
 public void UpdateCurrentPlayerActionText(string s) {}
 public void UpdateGamePresentation(PlayerMetadata p) {}
 public void ReplayEvent(IEvent e, IEventReplayDelegate d) { LastEvent=e; Callback=d; }
 public void CancelCurrentReplay() {}
 public void ShowReplaySummary(IList<IEvent> es, Action a) { if(a!=null) a(); }
 public void ShowIndustrySellingSummary(IEvent e, Action a) { if(a!=null) a(); }
 public void UpdateCurrentPlayerActionsView() {}
 public void FinishReplay() {}
}
public static class ExtendedProbeOutputFactory {
 public static object NoOp(Type iface) {
  AssemblyBuilder ab = AppDomain.CurrentDomain.DefineDynamicAssembly(new AssemblyName("ExtendedProbeDummy"+Guid.NewGuid().ToString("N")), AssemblyBuilderAccess.Run);
  TypeBuilder tb = ab.DefineDynamicModule("probe").DefineType("Dummy", TypeAttributes.Public);
  tb.AddInterfaceImplementation(iface);
  foreach(MethodInfo m in iface.GetMethods()) {
   ParameterInfo[] ps=m.GetParameters(); Type[] pt=new Type[ps.Length];
   for(int i=0;i<pt.Length;i++) pt[i]=ps[i].ParameterType;
   MethodBuilder mb=tb.DefineMethod(m.Name,MethodAttributes.Public|MethodAttributes.Virtual,m.ReturnType,pt);
   mb.GetILGenerator().Emit(OpCodes.Ret); tb.DefineMethodOverride(mb,m);
  }
  return Activator.CreateInstance(tb.CreateType());
 }
}
'@
Add-Type -TypeDefinition $stub -ReferencedAssemblies $dll
function New-Case([int]$caseSeed, [bool]$intro = $false, [int]$players = 2) {
 $config = [BoardGameRules.Entities.Game.GameStartConfiguration]::new([BoardGameRules.Entities.Common.NumberOfPlayers]$players)
 $config.Seed = $caseSeed; $config.IsIntroductoryGame = $intro
 $metadata = [BrassApplication.Game.GameMetadata]::new($config, [BrassApplication.Game.GameType]::Local)
 foreach ($pc in $config.Players) {
  $pm = [BrassApplication.Game.PlayerMetadata]::new()
  $pm.PlayerStartConfiguration = $pc; $pm.PlayerNetworkId = 'probe-' + $pc.Color; $pm.NickName = 'Probe ' + $pc.Color
  $pm.AiType = [BrassApplication.AI.AIType]::Hard
  $pm.AIBehaviour = [BrassApplication.AI.AiBehaviors.AiBehaviorFactory]::CreateAiBehavior($pm)
  $metadata.JoinGame($pm)
 }
 $repo = [Frameworks.Persistence.Repository.BaseRepository]::new([Frameworks.Persistence.RepositoryBackends.InMemory.InMemoryGameStorage]::new(), [Frameworks.Persistence.RepositoryBackends.InMemory.InMemorySnapshotStorage]::new(), [Frameworks.Persistence.GameFactory.BrassBirminghamFactory]::new())
 $app = $repo.Create($metadata); $app.Game.StartGame(); $repo.Save($app)
 return [pscustomobject]@{ Repo = $repo; App = $app }
}
function Complete-Turn($app) {
 $app.Game.GameFlow.EndTurn()
 if ($app.Game.GameFlow.IsLastTurnOver()) {
  if ($app.Game.GameFlow.IsSellIndustryNecessary()) {
   $selector = [BrassApplication.AI.RuleBasedSystem.RuleSelectors.RuleSelectorFactory]::CreatePrioritySelector([BrassApplication.AI.RuleBasedSystem.RuleSetFactory]::CreateSellIndustryRuleSet())
   $selector.Select($app).MakeAction($app,$false).Invoke()
  }
  $app.Game.GameFlow.EndRound()
 }
}
function Pass-AllActions($app) {
 while ($app.Game.TurnFlow.GetRemainingActionsCount() -gt 0) {
  $p = $app.Game.GameProgressInformation.CurrentPlayer
  $a = $app.Game.TurnFlow.StartPassAction(); $a.Pass($p.Hand[0], $p); $app.Game.TurnFlow.ConfirmCurrentAction()
 }
}
$case = New-Case $Seed $true
while (-not $case.App.Game.GameProgressInformation.IsGameOver) { Pass-AllActions $case.App; Complete-Turn $case.App; $case.Repo.Save($case.App) }
$introBefore = $case.App.Game.GameProgressInformation.IntroductoryPoints
$loaded = $case.Repo.Load($case.App.Metadata.GameId)
$history = [BrassApplication.GameHistory.BrassGameHistory]::new(); $history.Repository = $case.Repo; $history.SetGame($loaded)
$reviewPresenter = [ExtendedProbeOutputFactory]::NoOp([BrassApplication.UseCases.Game.FinishedGameReview.IFinishedGameReviewOutput])
$review = [BrassApplication.UseCases.Game.FinishedGameReview.FinishedGameReviewUseCase]::new($reviewPresenter, $history)
$reviewError = $null
try { $review.ShowGameSummaryTable() } catch { $reviewError = $_.Exception.InnerException.ToString() }
$history.SetGame($case.App)
$review.ShowGameSummaryTable()
$intro = [ordered]@{ originalSummarySucceeds = $true; originalIntroductoryPointsCount = $introBefore.Count; reloadedIntroductoryPointsIsNull = $null -eq $loaded.Game.GameProgressInformation.IntroductoryPoints; originalVp = @($case.App.Game.PlayersTrack.Players | ForEach-Object { $_.VP }); loadedVp = @($loaded.Game.PlayersTrack.Players | ForEach-Object { $_.VP }); reviewError = $reviewError; turns = 20 }
$replayResult = $null
for ($offset = 0; $offset -lt 20 -and $null -eq $replayResult; $offset++) {
 $case = New-Case ($Seed+$offset)
 $app = $case.App; $repo = $case.Repo
 $turn = 0
 while (-not $app.Game.GameProgressInformation.IsGameOver) {
  $startRevision = $app.Game.Revision
  $p = $app.Game.GameProgressInformation.CurrentPlayer
  $behavior = $app.Metadata.GetPlayerMetadataByColor($p.Color).AIBehaviour
  while ($app.Game.TurnFlow.GetRemainingActionsCount() -gt 0) { $behavior.GetNextAIMove($app).Invoke(); $app.Game.TurnFlow.ConfirmCurrentAction() }
  $target = @($app.Game.GetUncommittedEvents() | Where-Object { $_.GetType().Name -eq 'MerchantBonusReceived' -and $_.BonusType.ToString() -eq 'Develop' })
  Complete-Turn $app; $turn++
  $repo.Save($app)
  if ($target.Count -gt 0 -and -not $app.Game.GameProgressInformation.IsGameOver) {
   $history = [BrassApplication.GameHistory.BrassGameHistory]::new(); $history.Repository = $repo; $history.SetGame($app)
   $pm = $app.Metadata.GetPlayerMetadataByColor($app.Game.GameProgressInformation.CurrentPlayer.Color)
   $pm.LastSeenEventRevision = $startRevision
   $presenter = [ExtendedReplayProbeOutput]::new()
   $replay = [BrassApplication.UseCases.Game.Replay.ReplayUseCase]::new($presenter,$history)
   $latestFlagBefore = $app.Game.DevelopMerchantBonus.MerchantDevelopmentNeeded
   $replay.ReplayUnseenEvents($pm, $null)
   $eventsDone = 0
   while ($replay.GetIsReplayInProgress() -and $presenter.LastEvent.Metadata.Revision -le $target[0].Metadata.Revision) {
    $presenter.Callback.EventReplayCompleted($presenter.LastEvent); $eventsDone++
    if ($eventsDone -gt 200) { throw 'Replay guard' }
   }
   $beforeSkipEvent = $presenter.LastEvent.GetType().Name
   $staticBeforeSkip = [BoardGameRules.Entities.Actions.DevelopMerchantBonus]::Instance.MerchantDevelopmentNeeded
   $replay.SkipReplay()
   while ($replay.GetIsReplayInProgress()) { $replay.SkipReplay() }
   $staticAfterSkip = [BoardGameRules.Entities.Actions.DevelopMerchantBonus]::Instance.MerchantDevelopmentNeeded
   $ownAfterSkip = $history.CurrentGame.Game.DevelopMerchantBonus.MerchantDevelopmentNeeded
   $sameInstance = [object]::ReferenceEquals([BoardGameRules.Entities.Actions.DevelopMerchantBonus]::Instance, $history.CurrentGame.Game.DevelopMerchantBonus)
   $nextTurnError = $null
   try { Pass-AllActions $history.CurrentGame; $history.CurrentGame.Game.GameFlow.EndTurn() } catch { $nextTurnError = $_.Exception.InnerException.ToString() }
   $replayResult = [ordered]@{ seed = $Seed+$offset; priorAiTurns = $turn; merchantBonusRevision = $target[0].Metadata.Revision; latestOwnFlagBefore = $latestFlagBefore; pausedAtNextEvent = $beforeSkipEvent; staticFlagBeforeSkip = $staticBeforeSkip; staticFlagAfterSkip = $staticAfterSkip; latestOwnFlagAfterSkip = $ownAfterSkip; instanceMatchesCurrentGame = $sameInstance; resumedGameIsLatest = [object]::ReferenceEquals($app,$history.CurrentGame); nextTurnError = $nextTurnError }
   $controls = @()
   foreach ($mode in @('completeAllEvents','skipBeforeBonus')) {
    $controlApp = $repo.Load($app.Metadata.GameId)
    $controlHistory = [BrassApplication.GameHistory.BrassGameHistory]::new(); $controlHistory.Repository=$repo; $controlHistory.SetGame($controlApp)
    $controlPm=$controlApp.Metadata.GetPlayerMetadataByColor($controlApp.Game.GameProgressInformation.CurrentPlayer.Color); $controlPm.LastSeenEventRevision=$startRevision
    $controlPresenter=[ExtendedReplayProbeOutput]::new()
    $controlReplay=[BrassApplication.UseCases.Game.Replay.ReplayUseCase]::new($controlPresenter,$controlHistory)
    $controlReplay.PresentEvents($repo.Advanced.GameStorage.LoadEvents($app.Metadata.GameId,$startRevision,[int]::MaxValue),$controlPm,$null)
    if($mode -eq 'skipBeforeBonus') { while($controlReplay.GetIsReplayInProgress()) { $controlReplay.SkipReplay() } }
    else { while($controlReplay.GetIsReplayInProgress()) { $controlPresenter.Callback.EventReplayCompleted($controlPresenter.LastEvent) } }
    $controlFlag=[BoardGameRules.Entities.Actions.DevelopMerchantBonus]::Instance.MerchantDevelopmentNeeded
    $controlError=$null
    try { Pass-AllActions $controlHistory.CurrentGame; $controlHistory.CurrentGame.Game.GameFlow.EndTurn() } catch { $controlError=$_.Exception.InnerException.ToString() }
    $controls += [ordered]@{mode=$mode; staticFlag=$controlFlag; endTurnError=$controlError}
   }
   $replayResult['controls']=$controls
   break
  }
 }
}
$result = [ordered]@{ assemblySha256 = (Get-FileHash -LiteralPath $dll -Algorithm SHA256).Hash; introFinishedReload = $intro; skipReplayAfterGloucester = $replayResult }
$json = ConvertTo-Json $result -Depth 12
$json | Set-Content -LiteralPath (Join-Path $PSScriptRoot 'extended-phases-results.json') -Encoding utf8
$json
