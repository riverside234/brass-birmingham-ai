using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using BoardGameRules.Entities.Actions;
using BoardGameRules.Entities.Board;
using BoardGameRules.Entities.Board.BuildingSlots;
using BoardGameRules.Entities.Board.Regions;
using BoardGameRules.Entities.Cards;
using BoardGameRules.Entities.Common;
using BoardGameRules.Entities.EventSourcing.Events;
using BoardGameRules.Entities.Game;
using BoardGameRules.Entities.GamePhases;
using BoardGameRules.Entities.Industries;
using BoardGameRules.Entities.Players;
using BrassApplication.AI;
using BrassApplication.Game;
using BrassApplication.GameHistory;
using BrassApplication.UseCases.Game.Actions;
using BrassApplication.UseCases.Game.Actions.BaseAction;
using BrassApplication.UseCases.Game.Actions.BuildAction;
using BrassApplication.UseCases.Game.Actions.CoalDelivery;
using BrassApplication.UseCases.Game.Actions.IronDelivery;
using BrassApplication.UseCases.Game.RegionDetails;
using BrassApplication.UseCases.Game.Replay;
using BrassApplication.UseCases.Game.Board;
using BrassApplication.UseCases.Game.PlayerActions;
using BrassApplication.UseCases.Game.PlayGame;
using BrassApplication.UseCases.Common.SaveGame;
using InterfaceAdapters.Game.Board;
using InterfaceAdapters.Game.PlayerActions;
using BrassLegacyAI;
using BrassLegacyAI.Native;
using BrassPatch;
using BrassPatch.Persistence;
using Frameworks.Persistence.GameFactory;
using Frameworks.Persistence.Repository;
using Frameworks.Persistence.RepositoryBackends.InMemory;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

// Calls the shipped city/build/cancel and history entry points. The sinks only
// replace presentation; all event mutation, reconstruction and input guards are real.
public static class TurnHistoryRegression
{
    static int checks;
    static void Check(bool ok, string message) { checks++; if (!ok) throw new Exception(message); }
    public sealed class Fixture
    {
        public BrassGameHistory History;
        public BrassApplicationGame App { get { return History.CurrentGame; } }
        public int Checkpoint, AiTurns;
        public Fixture(int seats, bool native, bool rail = false, bool humanFirst = true)
        {
            var config = new GameStartConfiguration((NumberOfPlayers)seats) { Seed = 102601, IsIntroductoryGame = false };
            var metadata = new GameMetadata(config, GameType.Local);
            foreach (var start in config.Players)
                metadata.JoinGame(new PlayerMetadata { PlayerStartConfiguration = start, PlayerNetworkId = "test-" + start.Color,
                    NickName = start.Color.ToString(), AiType = start.Color == PlayerColor.Yellow ? AIType.Human : AIType.Hard });
            var state = native ? NativeGameState.Create(metadata, p => "easy") : LocalGameState.Create(metadata, p => "balanced", p => "easy");
            SaveExtensions.Set(metadata, state);
            var repository = new BaseRepository(new InMemoryGameStorage(), new InMemorySnapshotStorage(), new BrassBirminghamFactory());
            History = new BrassGameHistory { Repository = repository };
            var app = repository.Create(metadata); app.Game.StartGame();
            state["game_id"] = app.Metadata.GameId; state["ready"] = true; SaveExtensions.Set(app.Metadata, state);
            History.SetGame(app);
            while ((rail && app.Game.GameProgressInformation.CurrentGameEra != GameEra.RailEra) ||
                app.Game.GameProgressInformation.CurrentPlayer.Color != PlayerColor.Yellow ||
                app.Game.GameProgressInformation.CurrentRoundNumber < 2) { PassTurn(app); End(app); }
            // A real persisted human-End checkpoint, followed by unsaved AI
            // turns and the next human turn. Research setup is not a gameplay save.
            if (humanFirst) { PassTurn(app); End(app); }
            repository.Save(app); Checkpoint = app.Game.Revision;
            while (app.Game.GameProgressInformation.CurrentPlayer.Color != PlayerColor.Yellow)
            { PassTurn(app); End(app); AiTurns++; }
            if (humanFirst && AiTurns == 0) throw new Exception("fixture needs unsaved AI turns");
        }
        public string Snapshot()
        {
            var g = App.Game;
            var events = History.Repository.Advanced.GameStorage.LoadEvents(App.Metadata.GameId, 0, g.Revision)
                .Concat(g.GetUncommittedEvents()).Where(e => e.Metadata.Revision <= g.Revision).OrderBy(e => e.Metadata.Revision);
            return JsonConvert.SerializeObject(new { g.Revision, actor = g.GameProgressInformation.CurrentPlayer.Color,
                era = g.GameProgressInformation.CurrentGameEra, round = g.GameProgressInformation.CurrentRoundNumber,
                turn = g.GameProgressInformation.CurrentTurnNumber, remaining = g.TurnFlow.GetRemainingActionsCount(),
                players = g.PlayersTrack.Players.Select(p => new { p.Color, p.Money, p.VP, p.IncomeMarkerPosition,
                    hand = p.Hand.Select(c => c.UniqueId), discard = p.DiscardPile.Select(c => c.UniqueId) }),
                events = events.Select(e => new { type = e.GetType().FullName, data = JToken.FromObject(e) }) });
        }
        public BuildActionUseCase City()
        {
            var output = new BuildSink(); var build = new BuildActionUseCase(output, History);
            var board = new BoardUseCase(new BoardSink(build), History, null);
            board.SetAllowBuidlingFromBoard(true); board.RegionSelected(RegionName.Birmingham);
            Check(ReferenceEquals(App.Game.TurnFlow.CurrentAction, App.Game.TurnFlow.BuildAction), "city starts current human Build");
            Check(output.Cancel == build && output.Details == 1, "city installs cancel route and opens details");
            return build;
        }
    }
    public static void PassOne(BrassApplicationGame app, bool confirm = true)
    {
        var player = app.Game.GameProgressInformation.CurrentPlayer;
        app.Game.TurnFlow.StartPassAction().Pass(player.Hand[0], player);
        if (confirm) app.Game.TurnFlow.ConfirmCurrentAction();
    }
    public static void PassTurn(BrassApplicationGame app) { while (app.Game.TurnFlow.GetRemainingActionsCount() > 0) PassOne(app); }
    public static void End(BrassApplicationGame app)
    { app.Game.GameFlow.EndTurn(); if (app.Game.GameFlow.IsLastTurnOver()) app.Game.GameFlow.EndRound(); }

    public static JObject Run(bool reproduce)
    {
        checks = 0;
        if (reproduce)
        {
            var f = new Fixture(4, true); int before = f.App.Game.Revision;
            f.City().CancelAction();
            var actor = f.App.Game.GameProgressInformation.CurrentPlayer.Color;
            Check(actor != PlayerColor.Yellow && f.App.Game.Revision < before, "baseline reproduces actor switch and history rewind");
            return new JObject { ["passed"] = true, ["bug_reproduced"] = true, ["before_actor"] = "Yellow", ["after_actor"] = actor.ToString(),
                ["before_revision"] = before, ["after_revision"] = f.App.Game.Revision, ["checks"] = checks };
        }
        var rows = new JArray();
        foreach (int seats in new[] { 2, 3, 4 })
        foreach (bool native in seats == 4 ? new[] { false, true } : new[] { false })
        foreach (bool rail in new[] { false, true })
        {
            var f = new Fixture(seats, native, rail); string before = f.Snapshot(); int revision = f.App.Game.Revision;
            Check(!f.History.IsCancelActionAvailable() && !f.History.IsUndoActionAvailable(), "earlier AI events are not a human action");
            var original = f.App;
            for (int n = 0; n < 3; n++)
            {
                var build = f.City();
                if (n == 1) new RegionDetailsUseCase(new DetailsSink(build), f.History).BackToBoard();
                else build.CancelAction();
                Check(ReferenceEquals(original, f.App) && f.Snapshot() == before, "no-op city cancel keeps human, identity and every AI event");
                Check(f.App.Game.TurnFlow.CurrentAction == null, "city cancel releases action selection");
                var actions = new ActionsSink(); new PlayerActionsUseCase(actions, f.History).StartPassAction();
                Check(actions.Passes == 1, "human action input remains enabled after city cancel");
            }
            // Card discard changes state before confirmation. Cancel must remove
            // only this action, retaining prior players' entire unsaved turns.
            PassOne(f.App, false); Check(f.History.IsCancelActionAvailable(), "partial human action can cancel");
            new CancelUseCase(f.History).CancelAction();
            Check(f.Snapshot() == before, "partial first action rolls back exactly to current TurnStarted");
            Check(!f.History.IsUndoActionAvailable(), "cannot undo the preceding AI action after cancellation");
            PassOne(f.App); string afterFirst = f.Snapshot();
            var second = f.City(); second.CancelAction();
            Check(f.Snapshot() == afterFirst, "second-action city cancel retains confirmed first action");
            PassOne(f.App, false); new CancelUseCase(f.History).CancelAction();
            Check(f.Snapshot() == afterFirst, "partial second action rolls back exactly to first action boundary");
            PassOne(f.App); Check(f.History.IsUndoActionAvailable(), "second action can undo"); f.History.UndoAction();
            Check(f.Snapshot() == afterFirst, "undo second retains first action");
            f.History.UndoAction(); Check(f.Snapshot() == before, "undo first stays in current human turn");
            Check(!f.History.IsUndoActionAvailable() && !f.History.IsCancelActionAvailable(), "turn boundary blocks further rollback");
            // Replay views must remain non-interactive and return to the live game.
            var latest = f.App; f.History.MoveGameToRevision(revision - 1);
            Check(!f.History.IsCancelActionAvailable() && !f.History.IsUndoActionAvailable(), "replay cannot cancel or undo");
            f.History.MoveToLatestGame(); Check(ReferenceEquals(latest, f.App) && f.Snapshot() == before, "replay returns to live human state");
            Check(f.History.Repository.Load(f.App.Metadata.GameId).Game.Revision == f.Checkpoint, "no cancel/undo/AI gameplay saves");
            var assignments = LocalGameState.Read(f.App.Metadata)["assignments"].DeepClone();
            PassTurn(f.App); End(f.App); f.History.Repository.Save(f.App);
            int savedRevision = f.App.Game.Revision;
            f.History.SetGame(f.History.Repository.Load(f.App.Metadata.GameId));
            Check(f.App.Game.Revision == savedRevision && JToken.DeepEquals(LocalGameState.Read(f.App.Metadata)["assignments"], assignments), "human-End save/load keeps contiguous history and seat assignments");
            rows.Add(new JObject { ["seats"] = seats, ["family"] = native ? "nn" : "v4", ["era"] = rail ? "Rail" : "Canal", ["unsaved_ai_turns"] = f.AiTurns });
        }
        var fresh = new Fixture(2, false, humanFirst: false); string start = fresh.Snapshot();
        fresh.City().CancelAction(); Check(fresh.Snapshot() == start, "city cancel with empty pending history stays unchanged");
        PassOne(fresh.App, false); new CancelUseCase(fresh.History).CancelAction();
        Check(fresh.Snapshot() == start, "partial action falls back to persisted boundary when no TurnStarted is pending");
        CompletedTurnCheckpoints(false); CompletedTurnCheckpoints(true);
        return new JObject { ["passed"] = true, ["checks"] = checks, ["cases"] = rows, ["human_and_ai_end_checkpoint_tests"] = true };
    }
    static void CompletedTurnCheckpoints(bool native)
    {
        var f = new Fixture(4, native, humanFirst: false);
        var output = new CheckpointSink(f.History);
        var play = new PlayGameUseCase(output, f.History, null, null);
        int checkpoint = f.App.Game.Revision;
        for (int turn = 0; turn < 3; turn++)
        {
            bool human = f.App.Metadata.GetPlayerMetadataByColor(f.App.Game.GameProgressInformation.CurrentPlayer.Color).AiType == AIType.Human;
            Check(human == (turn == 0), "fixture runs human then two AI turns");
            PassTurn(f.App);
            Check(output.Saves == turn && f.History.Repository.Load(f.App.Metadata.GameId).Game.Revision == checkpoint, "no save during individual human/AI actions");
            play.EndTurn(); checkpoint = f.App.Game.Revision;
            Check(output.Saves == turn + 1 && output.Starts == turn, "one completed-turn checkpoint before next actor dispatch");
            play.EndTurn(); Check(output.Saves == turn + 1 && f.App.Game.Revision == checkpoint, "duplicate End blocked during save");
            var saved = f.History.Repository.Load(f.App.Metadata.GameId);
            Check(saved.Game.Revision == checkpoint && saved.Game.GameProgressInformation.CurrentPlayer.Color == f.App.Game.GameProgressInformation.CurrentPlayer.Color, "checkpoint includes completed turn and next actor");
            Check(saved.Game.TurnFlow.CurrentAction == null, "checkpoint has no partially completed action");
            output.Complete(); Check(output.Starts == turn + 1, "next turn starts after successful save callback");
        }
        f.History.SetGame(f.History.Repository.Load(f.App.Metadata.GameId));
        Check(f.App.Game.Revision == checkpoint && NativeGameState.IsNative(LocalGameState.Read(f.App.Metadata)) == native, "resume retains latest completed AI turn and table family");
    }
    public static int Main(string[] args)
    {
        try { var result = Run(args.Contains("--reproduce")); File.WriteAllText(args[0], result.ToString()); Console.WriteLine(result.ToString()); return 0; }
        catch (Exception error) { File.WriteAllText(args[0], new JObject { ["passed"] = false, ["checks"] = checks, ["error"] = error.ToString() }.ToString()); Console.Error.WriteLine(error); return 1; }
    }
    sealed class CancelUseCase : BaseActionUseCase
    { public CancelUseCase(BrassGameHistory history) : base(new BuildSink(), history) { } }
    sealed class BoardSink : BoardPresenter, IBoardOutput
    {
        readonly BuildActionUseCase build;
        public BoardSink(BuildActionUseCase value) : base(null, null) { build = value; }
        void IBoardOutput.StartBuildActionFromBoard(RegionName region) { build.StartBuildActionFromBoard(region); }
    }
    sealed class ActionsSink : PlayerActionsPresenter, IPlayerActionsOutput
    {
        public int Passes; public ActionsSink() : base(null, null) { }
        void IPlayerActionsOutput.StartPassAction() { Passes++; }
    }
    sealed class CheckpointSink : IPlayGameOutput
    {
        readonly BrassGameHistory history; ISaveGameDelegate pending;
        public int Saves, Starts;
        public CheckpointSink(BrassGameHistory value) { history = value; }
        public void SaveGame(BrassApplicationGame app, ISaveGameDelegate callback) { history.Repository.Save(app); Saves++; pending = callback; }
        public void Complete() { var callback = pending; pending = null; callback.SaveGameComplete(true); }
        public void StartHumanTurn() { Starts++; } public void StartAITurn(PlayerMetadata player) { Starts++; }
        public void StartSellingIndustries() { throw new Exception("unexpected debt"); }
        public void StartGameReview(PlayerMetadata player) { throw new Exception("unexpected end"); }
        public void StartNetworkHumanTurn() { throw new Exception("unexpected network"); }
        public void StartNetworkSpectatorTurn(PlayerMetadata player) { throw new Exception("unexpected network"); }
    }
    sealed class DetailsSink : IRegionDetailsOutput
    {
        readonly BuildActionUseCase build; public DetailsSink(BuildActionUseCase value) { build = value; }
        public void CancelAction() { build.CancelAction(); } public void BackToBoard() { }
    }
    public sealed class BuildSink : IBuildActionOutput
    {
        public IActionInProgressCancelDelegate Cancel; public int Details;
        public void HighlightValidBuildingSlots(Board b, List<BuildingSlot> s, ISelectBuildingDelegate d, Action<RegionName> a) { }
        public void HighlightValidBuildingSlotsInRegion(Board b, List<BuildingSlot> s, ISelectBuildingDelegate d, Action<RegionName> a, RegionName r) { }
        public void TurnOffSlotsHighlightning(Board b) { }
        public void ShowPlayerInfoToSelectBuildingSlot(IActionInProgressCancelDelegate d, int c, int m) { Cancel = d; }
        public void ShowValidCardsToBuild(List<BaseCard> c, ISelectCardDelegate d) { }
        public void HideCards() { }
        public void ShowRegionDetailsView(RegionName r, BrassApplicationGame g, IDictionary<Tuple<BuildingSlot, BuildingType>, IList<CantUseBuildActionReason>> x) { Details++; }
        public void HideRegionDetailsView() { } public void HideIronDelivery() { } public void HideCoalDelivery() { }
        public void ShowPlayerInfoToSelectIndustry() { } public void ShowPlayerInfoToSelectCardToDiscard() { }
        public void StartIronDeliveryUseCase(int n, IIronDeliveryResultDelegate d) { }
        public void StartCoalDeliveryUseCase(int n, IEnumerable<CoalMine> c, ICoalDeliveryResultDelegate d) { }
        public void SetFastForwardButtonActive(bool a) { }
        public void HideActionInProgressView() { } public void ShowPlayerActionsButtons() { } public void UpdateGameStatus(PlayerMetadata p) { }
        public void PresentEvents(ReadOnlyCollection<IEvent> e, PlayerMetadata p, IReplayDelegate d) { }
        public void RestoreCardsView() { } public void ShowEndTurnButton() { } public void HideReplayView() { }
        public Task WaitForTweensEnd() { return Task.FromResult(0); }
    }
}
