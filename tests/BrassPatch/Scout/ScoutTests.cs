using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using BoardGameRules.Entities.Cards;
using BoardGameRules.Entities.Common;
using BoardGameRules.Entities.EventSourcing.Events;
using BoardGameRules.Entities.Game;
using BoardGameRules.Entities.Players;
using BrassApplication.AI;
using BrassApplication.Game;
using BrassApplication.GameHistory;
using BrassApplication.UseCases.Game.Actions;
using BrassApplication.UseCases.Game.Actions.ActionInProgress;
using BrassApplication.UseCases.Game.Actions.BaseAction;
using BrassApplication.UseCases.Game.Actions.PassAction;
using BrassApplication.UseCases.Game.Actions.ScoutAction;
using BrassApplication.UseCases.Game.Replay;
using BrassApplication.UseCases.Game.HandOfCards;
using BrassPatch.Scout;
using Frameworks.Persistence.GameFactory;
using Frameworks.Persistence.Repository;
using Frameworks.Persistence.RepositoryBackends.InMemory;
using Frameworks.Persistence.RepositoryBackends.Filesystem;

namespace BrassPatch.Tests.Scout
{
    public static class ScoutTests
    {
        private static int checks;
        private static void Check(bool result, string label) { checks++; if (!result) throw new Exception(label); }
        private static void Reject(Action action, string label)
        { bool rejected = false; try { action(); } catch (InvalidOperationException) { rejected = true; } Check(rejected, label); }
        private sealed class Case
        {
            public BaseRepository Repo;
            public BrassGameHistory History;
            public ScoutActionUseCase Scout;
            public PassActionUseCase Pass;
            public Presenter Output;
            public ActionInProgressUseCase Cancel;
            public BrassApplicationGame App { get { return History.CurrentGame; } }
            public Player Player { get { return App.Game.GameProgressInformation.CurrentPlayer; } }
        }
        private static Case NewCase(int players = 2, bool filesystem = false)
        {
            var config = new GameStartConfiguration((NumberOfPlayers)players); config.Seed = 20260927;
            var metadata = new GameMetadata(config, GameType.Local);
            foreach (var p in config.Players) metadata.JoinGame(new PlayerMetadata { PlayerStartConfiguration = p, PlayerNetworkId = "scout-test-" + p.Color, NickName = "Test " + p.Color, AiType = AIType.Human });
            BaseRepository repo;
            if (filesystem)
            {
                string path = Path.Combine(Environment.CurrentDirectory, "artifacts", "scout-tests", Guid.NewGuid().ToString("N").Substring(0, 8));
                var serializer = new Frameworks.Persistence.Serialization.JsonSerializer();
                repo = new BaseRepository(new FilesystemGameStorage(path, serializer), new FilesystemSnapshotStorage(path, serializer), new BrassBirminghamFactory());
            }
            else repo = new BaseRepository(new InMemoryGameStorage(), new InMemorySnapshotStorage(), new BrassBirminghamFactory());
            var app = repo.Create(metadata); app.Game.StartGame(); repo.Save(app);
            var c = new Case { Repo = repo, History = new BrassGameHistory { Repository = repo } }; c.History.SetGame(app);
            c.Cancel = new ActionInProgressUseCase(new Progress()); c.Output = new Presenter { Cancel = c.Cancel };
            c.Scout = new ScoutActionUseCase(c.Output, c.History); c.Pass = new PassActionUseCase(c.Output, c.History);
            return c;
        }
        private static string Cards(IEnumerable<BaseCard> cards) { return String.Join(",", cards.Select(x => x.UniqueId)); }
        private static string State(BrassApplicationGame app)
        {
            var g = app.Game;
            return g.Revision + ":" + g.TurnFlow.GetRemainingActionsCount() + ":" + (g.TurnFlow.CurrentAction == null ? "none" : g.TurnFlow.CurrentAction.GetType().Name)
                + ":" + g.GameProgressInformation.CurrentPlayer.Color + ":" + Cards(g.Board.CardDeck) + ":" + Cards(g.Board.WildIndustryCardDeck) + ":" + Cards(g.Board.WildLocationCardDeck)
                + ":" + String.Join("|", g.PlayersTrack.Players.Select(p => p.Color + ":" + Cards(p.Hand) + ":" + Cards(p.DiscardPile)))
                + ":" + String.Join(",", g.GetUncommittedEvents().Select(e => e.GetType().Name + "@" + e.Metadata.Revision));
        }
        private static string RulesState(BrassApplicationGame app)
        {
            var g = app.Game;
            return g.Revision + ":" + g.TurnFlow.GetRemainingActionsCount() + ":" + g.GameProgressInformation.CurrentPlayer.Color
                + ":" + Cards(g.Board.CardDeck) + ":" + Cards(g.Board.WildIndustryCardDeck) + ":" + Cards(g.Board.WildLocationCardDeck)
                + ":" + String.Join("|", g.PlayersTrack.Players.Select(p => p.Color + ":" + Cards(p.Hand) + ":" + Cards(p.DiscardPile)));
        }
        private static Action Queue(Case c, bool reverse = false)
        {
            c.Scout.StartScoutAction(); c.Scout.SelectCard(c.Player.Hand[0]);
            BaseCard a = c.Player.Hand[0], b = c.Player.Hand[1];
            c.Scout.SelectCard(reverse ? b : a); c.Scout.SelectCard(reverse ? a : b);
            Check(c.Output.Callbacks.Count != 0, "second trade card queues completion");
            return c.Output.Callbacks.Last();
        }
        private static void SaveReload(Case c)
        {
            c.App.Game.GameFlow.EndTurn(); var events = c.App.Game.GetUncommittedEvents();
            Check(events.Zip(events.Skip(1), (a, b) => b.Metadata.Revision == a.Metadata.Revision + 1).All(x => x), "events remain contiguous before save");
            string expected = RulesState(c.App); c.Repo.Save(c.App); var loaded = c.Repo.Load(c.App.Metadata.GameId);
            Check(RulesState(loaded) == expected, "actual repository save/reload preserves Scout/Pass state");
        }
        public static int Run()
        {
            checks = 0;
            foreach (int players in new[] { 2, 3, 4 }) foreach (bool reverse in new[] { false, true })
            {
                Case c = NewCase(players); Action complete = Queue(c, reverse); complete();
                Check(c.Player.Hand.Count == 7, "Scout consumes three cards and grants two");
                Check(c.Player.Hand.Count(x => x is WildIndustryCard || x is WildLocationCard) == 2, "one of each wild granted");
                Check(c.Player.DiscardPile.Count == 4 && c.Player.DiscardPile.Select(x => x.UniqueId).Distinct().Count() == 4, "three distinct new discards plus setup discard");
                Check(c.App.Game.TurnFlow.GetRemainingActionsCount() == 0, "normal completion consumes exactly one action");
                string state = State(c.App); int ui = c.Output.Calls; complete(); c.Scout.SelectCard(c.Player.Hand[0]);
                Check(State(c.App) == state && c.Output.Calls == ui, "completed callback/extra selection has no state or UI effect"); SaveReload(c);
            }
            Cancellation(); Reentry(); DuplicateSelection(); DomainValidation(); ConfirmationGuards(); WildStartGuard(); ActualReplay();
            return checks;
        }
        private static void Cancellation()
        {
            for (int cutoff = 0; cutoff <= 3; cutoff++)
            {
                Case c = NewCase(); string initial = State(c.App); c.Scout.StartScoutAction();
                if (cutoff >= 1) c.Scout.SelectCard(c.Player.Hand[0]);
                if (cutoff >= 2) c.Scout.SelectCard(c.Player.Hand[0]);
                if (cutoff >= 3) c.Scout.SelectCard(c.Player.Hand[1]);
                c.Cancel.CancelAction(); Check(State(c.App) == initial, "cancel cutoff " + cutoff + " restores beginning of action");
                int ui = c.Output.Calls;
                foreach (Action callback in c.Output.Callbacks) callback();
                c.Scout.CancelAction(); Check(State(c.App) == initial && c.Output.Calls == ui, "cancelled/duplicate callbacks no-op at cutoff " + cutoff);
            }
            Case pass = NewCase(filesystem: true); Action late = Queue(pass); pass.Cancel.CancelAction();
            pass.Pass.StartPassAction(); pass.Pass.SelectCard(pass.Player.Hand[0]); string before = State(pass.App); int calls = pass.Output.Calls;
            late(); Check(State(pass.App) == before && pass.Output.Calls == calls, "cancel then completed Pass cannot be reconfirmed"); SaveReload(pass);
            Case undone = NewCase(); Action done = Queue(undone); done(); undone.History.UndoAction(); string undo = State(undone.App); done();
            Check(State(undone.App) == undo && undone.App.Game.TurnFlow.GetRemainingActionsCount() == 1, "callback after Undo cannot reapply Scout");
        }
        private static void Reentry()
        {
            Case restart = NewCase(); Action old = Queue(restart); Action current = Queue(restart); string before = State(restart.App); int calls = restart.Output.Calls;
            old(); Check(State(restart.App) == before && restart.Output.Calls == calls, "restart invalidates old generation"); current(); Check(restart.Player.Hand.Count == 7, "restarted Scout commits normally");
            Case next = NewCase(); Action first = Queue(next); object oldToken = ScoutFixes.CaptureOperation(next.Scout); next.Cancel.CancelAction(); Action second = Queue(next);
            ScoutFixes.InvalidateOperation(next.Scout, oldToken); first(); second(); Check(next.Player.Hand.Count == 7, "old destroyed-view token cannot invalidate new generation");
            Case destroyed = NewCase(); Action pending = Queue(destroyed); ScoutFixes.InvalidateOperation(destroyed.Scout, ScoutFixes.CaptureOperation(destroyed.Scout)); before = State(destroyed.App); calls = destroyed.Output.Calls;
            pending(); Check(State(destroyed.App) == before && destroyed.Output.Calls == calls, "current destroyed-view token prevents completion");
            destroyed.Scout.CancelAction(); Check(destroyed.App.Game.TurnFlow.CurrentAction == null && destroyed.Player.Hand.Count == 8, "destroyed attempt can still be explicitly cancelled");
            Case reopened = NewCase(); Action obsolete = Queue(reopened); ScoutFixes.InvalidateOperation(reopened.Scout, ScoutFixes.CaptureOperation(reopened.Scout));
            Action renewed = Queue(reopened); obsolete(); renewed(); Check(reopened.Player.Hand.Count == 7, "destroyed attempt can be cancelled and restarted without closing game");
            Case navigation = NewCase(); Action stale = Queue(navigation); Case other = NewCase(3); navigation.History.SetGame(other.App); before = State(other.App); calls = navigation.Output.Calls;
            stale(); Check(State(other.App) == before && navigation.Output.Calls == calls, "navigation blocks stale UI and rules effects");
            Case reentrant = NewCase(); Action finish = Queue(reentrant); var captured = reentrant.App; Case destination = NewCase(3);
            reentrant.Output.HideAction = delegate { reentrant.History.SetGame(destination.App); };
            int beforeCalls = reentrant.Output.Calls; string destinationState = State(destination.App); finish();
            Check(captured.Game.TurnFlow.GetRemainingActionsCount() == 0 && captured.Game.TurnFlow.CurrentAction == null, "captured game confirmed before navigation presentation");
            Check(State(destination.App) == destinationState && reentrant.Output.Calls == beforeCalls + 1, "synchronous hide navigation cannot alter destination or continue old UI");
            Case twice = NewCase(); Action once = Queue(twice); twice.Output.HideAction = once; once();
            Check(twice.Player.Hand.Count == 7 && twice.App.Game.TurnFlow.GetRemainingActionsCount() == 0, "synchronous completion reentrancy commits once");
        }
        private static void DuplicateSelection()
        {
            Case c = NewCase(); c.Scout.StartScoutAction(); c.Scout.SelectCard(c.Player.Hand[0]);
            BaseCard a = c.Player.Hand[0], b = c.Player.Hand[1]; c.Scout.SelectCard(a); string before = State(c.App); int calls = c.Output.Calls;
            c.Scout.SelectCard(a); Check(c.Output.Slots.SequenceEqual(new[] { 0 }) && c.Output.Callbacks.Count == 0 && c.Output.Calls == calls && State(c.App) == before, "duplicate selection ignored before queueing");
            c.Scout.SelectCard(b); before = State(c.App); calls = c.Output.Calls; c.Scout.SelectCard(c.Player.Hand[2]);
            Check(State(c.App) == before && c.Output.Calls == calls && c.Output.Callbacks.Count == 1, "fourth card cannot change queued selection");
            c.Output.Callbacks[0](); Check(c.Player.Hand.Count == 7 && c.Player.DiscardPile.Select(x => x.UniqueId).Distinct().Count() == 4, "legal selection after duplicate conserves cards");
            Case foreign = NewCase(); Case sameIds = NewCase(); foreign.Scout.StartScoutAction(); before = State(foreign.App); calls = foreign.Output.Calls;
            Check(foreign.Player.Hand[0].UniqueId == sameIds.Player.Hand[0].UniqueId, "foreign game fixture includes matching ordinal ID");
            foreign.Scout.SelectCard(sameIds.Player.Hand[0]); foreign.Scout.SelectCard(null);
            foreign.Scout.SelectCard(foreign.App.Game.PlayersTrack.Players.First(p => !Object.ReferenceEquals(p, foreign.Player)).Hand[0]);
            Check(State(foreign.App) == before && foreign.Output.Calls == calls, "foreign/same-ID/null selections have no effects");
        }
        private static void DomainValidation()
        {
            foreach (string mode in new[] { "duplicate", "foreign", "otherPlayer", "actionCard", "nullList", "nullCard", "threeCards", "emptyWildIndustry", "emptyWildLocation", "wrongPlayer", "noDiscard" })
            {
                Case c = NewCase(); var player = c.Player; var action = c.App.Game.TurnFlow.StartScoutAction(); BaseCard spent = player.Hand[0];
                if (mode != "noDiscard") action.DiscardCard(spent, player);
                var selected = new List<BaseCard> { player.Hand[0], player.Hand[1] }; Player actor = player;
                if (mode == "duplicate") selected[1] = selected[0];
                if (mode == "foreign") { Case other = NewCase(); selected[0] = other.Player.Hand.First(x => x.UniqueId == selected[0].UniqueId); }
                if (mode == "otherPlayer") selected[0] = c.App.Game.PlayersTrack.Players.First(p => !Object.ReferenceEquals(p, player)).Hand[0];
                if (mode == "actionCard") selected[0] = spent;
                if (mode == "nullList") selected = null;
                if (mode == "nullCard") selected[0] = null;
                if (mode == "threeCards") selected.Add(player.Hand[2]);
                if (mode == "emptyWildIndustry") c.App.Game.Board.WildIndustryCardDeck.Clear();
                if (mode == "emptyWildLocation") c.App.Game.Board.WildLocationCardDeck.Clear();
                if (mode == "wrongPlayer") actor = c.App.Game.PlayersTrack.Players.First(p => !Object.ReferenceEquals(p, player));
                string before = State(c.App); Reject(delegate { action.Scout(selected, actor); }, "domain rejects " + mode);
                Check(State(c.App) == before, "rejected exchange leaves revision/events/cards untouched: " + mode);
            }
            Case valid = NewCase(); var actorPlayer = valid.Player; var scout = valid.App.Game.TurnFlow.StartScoutAction(); scout.DiscardCard(actorPlayer.Hand[0], actorPlayer);
            scout.Scout(new List<BaseCard> { actorPlayer.Hand[0], actorPlayer.Hand[1] }, actorPlayer); valid.App.Game.TurnFlow.ConfirmCurrentAction();
            Check(actorPlayer.Hand.Count == 7 && valid.App.Game.TurnFlow.GetRemainingActionsCount() == 0, "AI-style direct domain Scout remains valid");
            Case card = NewCase(); var active = card.App.Game.TurnFlow.StartScoutAction(); Case foreignCard = NewCase(); string unchanged = State(card.App);
            Reject(delegate { active.DiscardCard(foreignCard.Player.Hand[0], card.Player); }, "foreign action card rejected"); Check(State(card.App) == unchanged, "invalid action card leaves events/revision untouched");
        }
        private static void ConfirmationGuards()
        {
            Case c = NewCase(); string before = State(c.App); Reject(c.App.Game.TurnFlow.ConfirmCurrentAction, "missing action cannot be confirmed"); Check(State(c.App) == before, "missing-action rejection cannot create revision gap");
            c.Pass.StartPassAction(); c.Pass.SelectCard(c.Player.Hand[0]); c.App.Game.TurnFlow.StartPassAction(); before = State(c.App);
            Reject(c.App.Game.TurnFlow.ConfirmCurrentAction, "exhausted turn cannot be confirmed"); Check(State(c.App) == before, "zero-action rejection precedes action reset/event mutation");
        }
        private static void WildStartGuard()
        {
            Case c = NewCase(); Queue(c)(); PlayerColor color = c.Player.Color;
            c.App.Game.GameFlow.EndTurn();
            while (c.Player.Color != color)
            {
                while (c.App.Game.TurnFlow.GetRemainingActionsCount() > 0) { c.Pass.StartPassAction(); c.Pass.SelectCard(c.Player.Hand[0]); }
                c.App.Game.GameFlow.EndTurn(); if (c.App.Game.GameFlow.IsLastTurnOver()) c.App.Game.GameFlow.EndRound();
            }
            string before = State(c.App); Reject(c.Scout.StartScoutAction, "held wild prevents Scout start before discard"); Check(State(c.App) == before, "failed Scout start leaves state intact");
            var action = c.App.Game.TurnFlow.StartScoutAction(); before = State(c.App);
            Reject(delegate { action.DiscardCard(c.Player.Hand.First(x => x is WildIndustryCard), c.Player); }, "cannot bypass no-wild eligibility by using wild as action card");
            Check(State(c.App) == before, "wild bypass rejection conserves hand/deck/revision");
        }
        private static void ActualReplay()
        {
            Case c = NewCase(); var startingApp = c.App;
            PlayerMetadata metadata = c.App.Metadata.GetPlayerMetadataByColor(c.Player.Color); metadata.LastSeenEventRevision = c.App.Game.Revision;
            var handOutput = new HandOutput(); var hand = new HandOfCardsUseCase(handOutput, c.History); hand.UpdateCardsForPlayer(metadata);
            BaseCard first = handOutput.Cards[0], second = handOutput.Cards[1], third = handOutput.Cards[2];
            var replayOutput = new ReplayOutput { Hand = hand }; var replay = new ReplayUseCase(replayOutput, c.History);
            c.Output.OnPresent = delegate(ReadOnlyCollection<IEvent> events, PlayerMetadata player, IReplayDelegate done) { replay.PresentEvents(events, player, done); };
            c.Scout.StartScoutAction(); hand.SetOnCardClickActionDelegate(c.Scout); hand.OnCardSelected(first);
            Check(!replay.GetIsReplayInProgress() && replayOutput.Events == 1, "actual ReplayUseCase processes first Scout discard");
            Check(Object.ReferenceEquals(c.App, startingApp) && Object.ReferenceEquals(handOutput.Cards[1], second), "replay restores latest instance and Scout keeps hand card references");
            hand.OnCardSelected(second); hand.OnCardSelected(third); c.Output.Callbacks.Last()();
            Check(c.Player.Hand.Count == 7 && c.App.Game.TurnFlow.GetRemainingActionsCount() == 0 && !replay.GetIsReplayInProgress(), "live hand selections after actual replay complete Scout");
            Check(replayOutput.Events == 3 && handOutput.Cards.Count == 7, "wild exchange/action decrement replay and hand refresh complete");
        }
        private sealed class HandOutput : IHandOfCardsOutput
        {
            public List<BaseCard> Cards;
            public void UpdateHandCardsView(List<BaseCard> cards, bool hidden) { Cards = new List<BaseCard>(cards); }
            public void OnLocationCardClicked(BoardGameRules.Entities.Board.Regions.RegionName region, List<BoardGameRules.Entities.Board.BuildingSlots.BuildingSlot> slots, BoardGameRules.Entities.Board.Board board) { }
            public string GetRegionFriendlyName(string region) { return region; }
        }
        private sealed class ReplayOutput : IReplayOutput
        {
            public HandOfCardsUseCase Hand; public int Events;
            public void UpdateCurrentPlayer(string text) { }
            public void UpdateCurrentPlayerAction(IEvent e) { }
            public void UpdateCurrentPlayerActionText(string text) { }
            public void UpdateGamePresentation(PlayerMetadata player) { Hand.UpdateCards(player); }
            public void ReplayEvent(IEvent e, IEventReplayDelegate done) { Events++; done.EventReplayCompleted(e); }
            public void CancelCurrentReplay() { }
            public void ShowReplaySummary(IList<IEvent> events, Action done) { done(); }
            public void ShowIndustrySellingSummary(IEvent e, Action done) { done(); }
            public void UpdateCurrentPlayerActionsView() { }
            public void FinishReplay() { }
        }
        private sealed class Progress : IActionInProgressOutput { }
        private sealed class Presenter : IScoutActionOutput, IPassActionOutput
        {
            public ActionInProgressUseCase Cancel; public readonly List<Action> Callbacks = new List<Action>(); public readonly List<int> Slots = new List<int>();
            public int Calls; public Action HideAction;
            public Action<ReadOnlyCollection<IEvent>, PlayerMetadata, IReplayDelegate> OnPresent;
            public void ShowPlayerInfoToSelectCardToDiscard(IActionInProgressCancelDelegate input, int count, int max) { Calls++; Cancel.SetCancelActionDelegate(input); }
            public void ShowCardsToDiscard(ISelectCardDelegate input) { Calls++; }
            public void HideCards() { Calls++; }
            public void ShowExchangeForWildCardsView(BrassApplicationGame game, ISelectCardDelegate input) { Calls++; }
            public void SetupCardInSlot(int slot, BaseCard card, Action callback) { Calls++; Slots.Add(slot); if (slot == 1) Callbacks.Add(callback); }
            public void HideExchangeForWildCardsView() { Calls++; if (HideAction != null) HideAction(); }
            public void ReplayEvents(ReadOnlyCollection<IEvent> events) { Calls++; }
            public void EnableCardsLayoutGroup() { Calls++; }
            public void HideActionInProgressView() { Calls++; Cancel.SetCancelActionDelegate(null); }
            public void ShowPlayerActionsButtons() { Calls++; }
            public void UpdateGameStatus(PlayerMetadata metadata) { Calls++; }
            public void PresentEvents(ReadOnlyCollection<IEvent> events, PlayerMetadata metadata, IReplayDelegate replay) { Calls++; if (OnPresent != null) OnPresent(events, metadata, replay); else replay.ReplayCompleted(); }
            public void RestoreCardsView() { Calls++; }
            public void ShowEndTurnButton() { Calls++; }
            public void HideReplayView() { Calls++; }
            public Task WaitForTweensEnd() { return Task.FromResult(true); }
            public void HideIronDelivery() { Calls++; }
        }
    }
}
