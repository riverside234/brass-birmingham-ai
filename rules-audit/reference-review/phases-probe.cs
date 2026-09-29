using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using System.IO;
using System.Threading.Tasks;
using BoardGameRules.Entities.Actions.Events;
using BoardGameRules.Entities.Cards;
using BoardGameRules.Entities.Common;
using BoardGameRules.Entities.EventSourcing.Events;
using BoardGameRules.Entities.Game;
using BrassApplication.AI;
using BrassApplication.Game;
using BrassApplication.GameHistory;
using BrassApplication.UseCases.Game.Actions;
using BrassApplication.UseCases.Game.Actions.ActionInProgress;
using BrassApplication.UseCases.Game.Actions.BaseAction;
using BrassApplication.UseCases.Game.Actions.ScoutAction;
using BrassApplication.UseCases.Game.Replay;
using Frameworks.Persistence.GameFactory;
using Frameworks.Persistence.Repository;
using Frameworks.Persistence.RepositoryBackends.InMemory;
using Frameworks.Persistence.RepositoryBackends.Filesystem;

public static class ReferencePhasesProbe
{
    private sealed class Case
    {
        public BrassApplicationGame App;
        public BaseRepository Repo;
        public BrassGameHistory History;
        public ScoutActionUseCase Scout;
        public Presenter Output;
        public ActionInProgressUseCase CancelInput;
    }
    private static Case NewCase(string path)
    {
        var config = new GameStartConfiguration(NumberOfPlayers.Two); config.Seed = 20260927;
        var metadata = new GameMetadata(config, GameType.Local);
        foreach (var player in config.Players) metadata.JoinGame(new PlayerMetadata { PlayerStartConfiguration = player, PlayerNetworkId = "probe-" + player.Color, NickName = "Probe " + player.Color, AiType = AIType.Human });
        var serializer = new Frameworks.Persistence.Serialization.JsonSerializer();
        var repo = new BaseRepository(new FilesystemGameStorage(path, serializer), new FilesystemSnapshotStorage(path, serializer), new BrassBirminghamFactory());
        var app = repo.Create(metadata); app.Game.StartGame(); repo.Save(app);
        var history = new BrassGameHistory { Repository = repo }; history.SetGame(app);
        var cancel = new ActionInProgressUseCase(new ProgressOutput());
        var output = new Presenter { CancelInput = cancel };
        var scout = new ScoutActionUseCase(output, history);
        return new Case { App = app, Repo = repo, History = history, Scout = scout, Output = output, CancelInput = cancel };
    }
    private static Dictionary<string, object> Snapshot(Case c)
    {
        var game = c.History.CurrentGame.Game;
        var player = game.GameProgressInformation.CurrentPlayer;
        return new Dictionary<string, object> {
            { "revision", game.Revision }, { "remainingActions", game.TurnFlow.GetRemainingActionsCount() },
            { "handCount", player.Hand.Count }, { "handIds", player.Hand.Select(x => x.UniqueId).ToArray() },
            { "wildCards", player.Hand.Count(x => x is WildIndustryCard || x is WildLocationCard) },
            { "discardCount", player.DiscardPile.Count }, { "distinctDiscardIds", player.DiscardPile.Select(x => x.UniqueId).Distinct().Count() },
            { "eventNames", game.GetUncommittedEvents().Select(e => e.GetType().Name).ToArray() },
            { "eventRevisions", game.GetUncommittedEvents().Select(e => e.Metadata.Revision).ToArray() },
            { "currentAction", game.TurnFlow.CurrentAction == null ? null : game.TurnFlow.CurrentAction.GetType().Name },
            { "drawDeckCount", game.Board.CardDeck.Count }
        };
    }
    private static void QueueScout(Case c, bool duplicate)
    {
        c.Scout.StartScoutAction();
        var p = c.History.CurrentGame.Game.GameProgressInformation.CurrentPlayer;
        c.Scout.SelectCard(p.Hand[0]);
        BaseCard one = p.Hand[0], two = duplicate ? one : p.Hand[1];
        c.Scout.SelectCard(one); c.Scout.SelectCard(two);
        if (c.Output.QueuedCompletion == null) throw new Exception("Scout completion was not queued");
    }
    public static object[] Run(string artifactRoot)
    {
        var results = new List<object>();
        foreach (string mode in new[] { "normalDistinctCards", "duplicateTradeCard", "cancelThenLateCompletion", "cancelPassThenLateCompletion", "cancelStartNextScoutThenOldCompletion" })
        {
            string fixturePath = Path.Combine(artifactRoot, Guid.NewGuid().ToString("N").Substring(0, 8));
            Case c = NewCase(fixturePath); QueueScout(c, mode == "duplicateTradeCard");
            Action originalCallback = c.Output.QueuedCompletion;
            var before = Snapshot(c);
            string callbackError = null;
            if (mode.StartsWith("cancel")) c.CancelInput.CancelAction();
            var afterCancel = Snapshot(c);
            if (mode == "cancelPassThenLateCompletion")
            {
                var game = c.History.CurrentGame.Game;
                var p = game.GameProgressInformation.CurrentPlayer;
                game.TurnFlow.StartPassAction().Pass(p.Hand[0], p); game.TurnFlow.ConfirmCurrentAction();
            }
            if (mode == "cancelStartNextScoutThenOldCompletion")
            {
                c.Scout.StartScoutAction();
                c.Scout.SelectCard(c.History.CurrentGame.Game.GameProgressInformation.CurrentPlayer.Hand[0]);
            }
            var beforeCallback = Snapshot(c);
            try { originalCallback(); } catch (Exception ex) { callbackError = ex.GetType().Name + ": " + ex.Message; }
            var afterCallback = Snapshot(c);
            string endTurnError = null, saveError = null, historyError = null;
            int liveRevision = -1, reloadRevision = -1;
            Dictionary<string, object> afterEndTurn = null;
            if (c.History.CurrentGame.Game.TurnFlow.GetRemainingActionsCount() == 0)
            {
                try
                {
                    c.History.CurrentGame.Game.GameFlow.EndTurn();
                    var current = c.History.CurrentGame;
                    liveRevision = current.Game.Revision;
                    afterEndTurn = Snapshot(c);
                    try { c.Repo.Save(current); } catch (Exception ex) { saveError = ex.GetType().Name + ": " + ex.Message; }
                    var loaded = c.Repo.Load(current.Metadata.GameId);
                    reloadRevision = loaded.Game.Revision;
                    if (saveError == null)
                    {
                        var history2 = new BrassGameHistory { Repository = c.Repo }; history2.SetGame(loaded);
                        history2.MoveGameToRevision(liveRevision);
                    }
                }
                catch (Exception ex) { historyError = ex.GetType().Name + ": " + ex.Message; }
            }
            results.Add(new Dictionary<string, object> {
                { "mode", mode }, { "before", before }, { "afterCancel", afterCancel }, { "beforeCallback", beforeCallback },
                { "afterCallback", afterCallback }, { "callbackError", callbackError }, { "endTurnError", endTurnError }, { "afterEndTurn", afterEndTurn },
                { "liveRevisionAfterEndTurn", liveRevision }, { "reloadRevision", reloadRevision }, { "saveError", saveError }, { "historyError", historyError }, { "fixturePath", fixturePath }
            });
        }
        return results.ToArray();
    }
    private sealed class ProgressOutput : IActionInProgressOutput { }
    private sealed class Presenter : IScoutActionOutput
    {
        public ActionInProgressUseCase CancelInput;
        public Action QueuedCompletion;
        public void ShowPlayerInfoToSelectCardToDiscard(IActionInProgressCancelDelegate input, int count, int max) { CancelInput.SetCancelActionDelegate(input); }
        public void ShowCardsToDiscard(ISelectCardDelegate input) { }
        public void HideCards() { }
        public void ShowExchangeForWildCardsView(BrassApplicationGame game, ISelectCardDelegate input) { }
        public void SetupCardInSlot(int slotIndex, BaseCard card, Action callback) { if (slotIndex == 1) QueuedCompletion = callback; }
        public void HideExchangeForWildCardsView() { }
        public void ReplayEvents(ReadOnlyCollection<IEvent> events) { }
        public void EnableCardsLayoutGroup() { }
        public void HideActionInProgressView() { CancelInput.SetCancelActionDelegate(null); }
        public void ShowPlayerActionsButtons() { }
        public void UpdateGameStatus(PlayerMetadata metadata) { }
        public void PresentEvents(ReadOnlyCollection<IEvent> events, PlayerMetadata metadata, IReplayDelegate replay) { replay.ReplayCompleted(); }
        public void RestoreCardsView() { }
        public void ShowEndTurnButton() { }
        public void HideReplayView() { }
        public Task WaitForTweensEnd() { return Task.FromResult(true); }
        public void HideIronDelivery() { }
    }
}
