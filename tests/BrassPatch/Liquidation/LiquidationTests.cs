using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using BoardGameRules.Entities.Actions.Events;
using BoardGameRules.Entities.Board.BuildingSlots;
using BoardGameRules.Entities.Common;
using BoardGameRules.Entities.Game;
using BoardGameRules.Entities.GamePhases.Events;
using BoardGameRules.Entities.NumericalResources.Events;
using BoardGameRules.Entities.Players;
using BoardGameRules.Entities.SellIndustry.Events;
using BrassApplication.AI;
using BrassApplication.ApplicationSettings;
using BrassApplication.Game;
using BrassApplication.GameHistory;
using BrassApplication.Network;
using BrassApplication.Persistence.RepositoryBackends;
using BrassApplication.UseCases.Common;
using BrassApplication.UseCases.Common.SaveGame;
using BrassApplication.UseCases.Game.ExitToMenu;
using BrassApplication.UseCases.Game.PlayGame;
using BrassApplication.UseCases.Game.SellIndustryFlow;
using BrassPatch.Liquidation;
using Frameworks.Persistence.GameFactory;
using Frameworks.Persistence.Repository;
using Frameworks.Persistence.RepositoryBackends.InMemory;
using InterfaceAdapters.Common;
using InterfaceAdapters.Game.SellIndustryFlow;
using InterfaceAdapters.Routing;

namespace BrassPatch.Tests.Liquidation
{
    public static class LiquidationTests
    {
        private static int checks;
        private static void Check(bool condition, string description) { checks++; if (!condition) throw new Exception(description); }
        private static void Reject(Action action, string description) { bool rejected = false; try { action(); } catch (InvalidOperationException) { rejected = true; } Check(rejected, description); }

        public static int Run()
        {
            checks = 0;
            DomainAndReload();
            DelayedHumanChoices();
            NoAssetsAndMixedAi();
            SaveFailure();
            NetworkRouting();
            NavigationCallbacks();
            return checks;
        }

        private static Case NewCase(int count = 2)
        {
            GameStartConfiguration config = new GameStartConfiguration((NumberOfPlayers)count); config.Seed = 20261001;
            GameMetadata metadata = new GameMetadata(config, GameType.Local);
            foreach (PlayerStartConfiguration p in config.Players) metadata.JoinGame(new PlayerMetadata { PlayerStartConfiguration = p, PlayerNetworkId = "test-" + p.Color, NickName = "Player " + p.Color, AiType = AIType.Human });
            BaseRepository repo = new BaseRepository(new InMemoryGameStorage(), new InMemorySnapshotStorage(), new BrassBirminghamFactory());
            BrassApplicationGame app = repo.Create(metadata); app.Game.StartGame(); repo.Save(app);
            return new Case { App = app, Repo = repo };
        }
        private static void Debt(Case c, Player p, int marker = 0)
        {
            c.App.Game.ApplyEvent(new MoneySpent(p.Color, p.Money));
            c.App.Game.ApplyEvent(new IncomeMarkerPositionChanged(p.Color, marker - p.IncomeMarkerPosition, ""));
        }
        private static BuildingSlot Build(Case c, Player p, BuildingType type)
        {
            BuildingSlot slot = c.App.Game.Board.Regions.SelectMany(r => r.BuildingSlots).First(s => s.BuildedIndustry == null && s.CanBuildIndustryType(type));
            c.App.Game.ApplyEvent(new IndustryBuilt(p.Color, type, slot.UniqueId)); return slot;
        }
        private static void PassActions(BrassApplicationGame app)
        {
            while (app.Game.TurnFlow.GetRemainingActionsCount() > 0)
            {
                Player p = app.Game.GameProgressInformation.CurrentPlayer;
                app.Game.TurnFlow.StartPassAction().Pass(p.Hand[0], p); app.Game.TurnFlow.ConfirmCurrentAction();
            }
        }
        private static void ReachFinalTurn(Case c)
        {
            while (c.App.Game.GameProgressInformation.CurrentTurnNumber < (int)c.App.Game.GameStartConfiguration.NumberOfPlayers) { PassActions(c.App); c.App.Game.GameFlow.EndTurn(); }
            PassActions(c.App);
        }
        private static void DomainAndReload()
        {
            Case c = NewCase(); Player p = c.App.Game.PlayersTrack.Players[0], other = c.App.Game.PlayersTrack.Players[1];
            BuildingSlot first = Build(c, p, BuildingType.Pottery), second = Build(c, p, BuildingType.CottonMill), opponent = Build(c, other, BuildingType.CoalMine);
            Debt(c, p); var flow = c.App.Game.SellIndustryFlow;
            int revision = c.App.Game.Revision;
            Reject(() => flow.SellIndustry(p.Color, first), "Selling outside settlement is rejected");
            Check(c.App.Game.Revision == revision, "Rejected non-phase sale adds no event");
            flow.StartIndustrySelling();
            Check(flow.RemainingDebtToPay(p.Color) == 10, "Initial debt");
            revision = c.App.Game.Revision;
            Reject(() => flow.SellIndustry(p.Color, opponent), "Opponent tile rejected");
            Reject(() => c.App.Game.GameFlow.EndRound(), "Unresolved phase blocks EndRound");
            Check(c.App.Game.GameFlow.IsLastTurnOver(), "Rejected EndRound retains pending-round flag");
            Check(c.App.Game.Revision == revision, "Invalid sale/EndRound leave revision untouched");
            c.Repo.Save(c.App); BrassApplicationGame loaded = c.Repo.Load(c.App.Metadata.GameId);
            Check(LiquidationFixes.IsPending(loaded.Game.SellIndustryFlow), "Phase is reconstructed after initial save");
            Check(loaded.Game.SellIndustryFlow.RemainingDebtToPay(p.Color) == 10 && loaded.Game.GameFlow.IsSellIndustryNecessary(), "Initial debt and round flags reconstruct");
            Check(loaded.Game.GameProgressInformation.CurrentPlayer.Color == p.Color, "Decision owner reconstructs");
            flow.SellIndustry(p.Color, first);
            int proceeds = flow.GetSoldIndustriesInLastRound()[p.Color][0].MoneyCost / 2;
            c.Repo.Save(c.App); loaded = c.Repo.Load(c.App.Metadata.GameId);
            Check(loaded.Game.SellIndustryFlow.RemainingDebtToPay(p.Color) == 10 - proceeds, "Partial debt reconstructs");
            Check(loaded.Game.SellIndustryFlow.GetSoldIndustriesInLastRound()[p.Color].Count == 1, "Sold history reconstructs");
            Check(loaded.Game.PlayersTrack.GetPlayerByColor(p.Color).Money == proceeds, "Partial sale money reconstructs once");
            Check(loaded.Game.Board.GetBuildingSlotByUniqueId(first.UniqueId).BuildedIndustry == null, "Liquidated tile remains removed");
            int secondProceeds = second.BuildedIndustry.MoneyCost / 2;
            flow.SellIndustry(p.Color, second);
            Check(flow.RemainingDebtToPay(p.Color) == 0 && p.Money == proceeds + secondProceeds, "Covered debt retains full sale proceeds");
            revision = c.App.Game.Revision;
            Reject(() => flow.SellIndustry(p.Color, opponent), "Extra sale cannot occur after debt is covered");
            Check(c.App.Game.Revision == revision, "Rejected extra sale leaves event stream unchanged");
            c.App.Game.GameFlow.EndRound(); c.Repo.Save(c.App); loaded = c.Repo.Load(c.App.Metadata.GameId);
            Check(!LiquidationFixes.IsPending(loaded.Game.SellIndustryFlow), "Ended phase reconstructs as ended");
            Check(loaded.Game.PlayersTrack.GetPlayerByColor(p.Color).Money == proceeds + secondProceeds - 10, "Surplus is retained after income settlement");
            Check(loaded.Game.SellIndustryFlow.GetSoldIndustriesInLastRound()[p.Color].Count == 2, "Final liquidation summary reconstructs");
        }

        private static UiHarness Harness(Case c, string networkUser = null)
        {
            BrassGameHistory history = new BrassGameHistory { Repository = c.Repo }; history.SetGame(c.App);
            FakeRouter router = new FakeRouter(); FakePlayOutput output = new FakePlayOutput(router, c);
            PlayGameUseCase play = new PlayGameUseCase(output, history, networkUser == null ? null : new FakeNetworkClient(networkUser), null);
            SellIndustryFlowUseCase sell = new SellIndustryFlowUseCase(new SellIndustryFlowPresenter(router), history);
            output.Sell = sell; router.Builder.Play = play;
            return new UiHarness { Router = router, Output = output, Play = play, History = history };
        }
        private static void DelayedHumanChoices()
        {
            Case c = NewCase(); Player p = c.App.Game.PlayersTrack.Players[0];
            Build(c, p, BuildingType.Pottery); Build(c, p, BuildingType.CottonMill); Debt(c, p); ReachFinalTurn(c);
            UiHarness h = Harness(c); int round = c.App.Game.GameProgressInformation.CurrentRoundNumber;
            h.Play.EndTurn();
            Check(h.Output.SaveCalls == 1 && h.Router.Popup == null, "Initial pending state saves before showing choices");
            Check(c.App.Game.GameProgressInformation.CurrentRoundNumber == round, "Delayed save does not advance the round");
            h.Play.EndTurn(); Check(h.Output.SaveCalls == 1, "Duplicate EndTurn during save is ignored");
            h.Output.CompleteSave();
            Check(h.Router.Popup.Text.Contains("income debt") && h.Router.Popup.Text.Contains("£10"), "Human sees remaining debt and tile choice");
            int revision = c.App.Game.Revision; string initialText = h.Router.Popup.Text;
            h.Router.Popup.Cancel(); Check(c.App.Game.Revision == revision, "Cycling choices does not mutate gameplay");
            h.Router.CompleteHide(); Check(h.Router.Popup.Text != initialText, "Next industry offers another owned tile");
            Action doubleClick = h.Router.Popup.Confirm; doubleClick(); doubleClick();
            Check(h.Output.SaveCalls == 1, "Choice waits for presentation completion");
            h.Router.CompleteHide(); Check(h.Output.SaveCalls == 2, "Confirmed choice saves exactly once");
            h.Output.CompleteSave(); Check(c.App.Game.GameProgressInformation.CurrentRoundNumber == round, "Remaining debt waits for another choice");
            h.Router.Popup.Confirm(); h.Router.CompleteHide(); h.Output.CompleteSave();
            Check(h.Output.HasPendingSave, "Covering debt ends round and schedules final save");
            Check(c.App.Game.GameProgressInformation.CurrentRoundNumber == round + 1, "Round resumes once after last choice");
            h.Output.CompleteSave(); Check(h.Output.HumanTurns == 1, "Next human turn starts once");
            int ended = c.Repo.Advanced.GameStorage.LoadEvents(c.App.Metadata.GameId, 0, int.MaxValue).Count(e => e is RoundEnded);
            Check(ended == 1, "Exactly one RoundEnded event is emitted");
            // Resume a saved pending phase in fresh application use cases.
            Case paused = NewCase(); Player debtor = paused.App.Game.PlayersTrack.Players[0]; Build(paused, debtor, BuildingType.Pottery); Debt(paused, debtor); ReachFinalTurn(paused);
            UiHarness before = Harness(paused); before.Play.EndTurn(); before.Output.CompleteSave();
            paused.App = paused.Repo.Load(paused.App.Metadata.GameId); UiHarness resumed = Harness(paused); resumed.Play.PlayGame();
            Check(resumed.Router.Popup != null && resumed.Router.Popup.Text.Contains("£10"), "Reload resumes human prompt instead of a new action turn");
            Check(paused.App.Game.TurnFlow.GetRemainingActionsCount() == 0, "Reloaded settlement does not grant action points");
        }

        private static void NoAssetsAndMixedAi()
        {
            Case c = NewCase(4); Player p = c.App.Game.PlayersTrack.Players[0], human = c.App.Game.PlayersTrack.Players[1], human2 = c.App.Game.PlayersTrack.Players[2], noAssets = c.App.Game.PlayersTrack.Players[3];
            c.App.Metadata.GetPlayerMetadataByColor(p.Color).AiType = AIType.Hard;
            Build(c, p, BuildingType.Pottery); Build(c, p, BuildingType.CottonMill); Build(c, human, BuildingType.Pottery); Build(c, human, BuildingType.CottonMill);
            Build(c, human2, BuildingType.Pottery); Build(c, human2, BuildingType.CottonMill);
            Debt(c, p); Debt(c, human); Debt(c, human2); Debt(c, noAssets); c.App.Game.ApplyEvent(new VictoryPointsChanged(noAssets.Color, 7)); ReachFinalTurn(c);
            UiHarness h = Harness(c); h.Play.EndTurn(); h.Output.CompleteSave();
            while (h.Output.HasPendingSave) h.Output.CompleteSave();
            Check(h.Router.Popup != null && h.Router.Popup.Text.Contains(human.Color.ToString()), "AI liquidates only its tiles before next human prompt");
            Check(c.App.Game.Board.GetIndustriesOnBoardOfGivenPlayer(human.Color).Count() == 2, "Human alternatives remain intact");
            int guard = 0; bool sawSecondHuman = false;
            while (c.App.Game.GameProgressInformation.CurrentRoundNumber == 1 || h.Output.HasPendingSave)
            {
                if (++guard > 20) throw new Exception("Liquidation callback loop");
                if (h.Output.HasPendingSave) h.Output.CompleteSave();
                else { if (h.Router.Popup.Text.Contains(human2.Color.ToString())) sawSecondHuman = true; h.Router.Popup.Confirm(); h.Router.CompleteHide(); }
            }
            Check(sawSecondHuman, "Each indebted human receives their own choice");
            Check(noAssets.VP == 0, "No-assets debt loses available VP without a negative total");
            Check(c.App.Game.SellIndustryFlow.GetLostVpsInLastRound()[noAssets.Color] == 7, "Shortfall summary records VP loss");
            BrassApplicationGame loaded = c.Repo.Load(c.App.Metadata.GameId);
            Check(loaded.Game.SellIndustryFlow.GetLostVpsInLastRound()[noAssets.Color] == 7, "No-assets loss summary survives reload");
            Check(!LiquidationFixes.IsPending(loaded.Game.SellIndustryFlow), "Mixed phase completes and reloads");
        }

        private static void SaveFailure()
        {
            Case c = NewCase(); Player debtor = c.App.Game.PlayersTrack.Players[0]; Build(c, debtor, BuildingType.Pottery); Debt(c, debtor); ReachFinalTurn(c);
            UiHarness h = Harness(c); h.Play.EndTurn(); int revision = c.App.Game.Revision;
            h.Output.CompleteSave(false);
            Check(h.Router.Popup.Text.Contains("could not be saved"), "Failed save offers retry");
            h.Play.EndTurn(); Check(h.Output.SaveCalls == 1, "Failed-save modal blocks duplicate EndTurn");
            h.Router.Popup.Confirm(); h.Router.CompleteHide(); Check(h.Output.SaveCalls == 2, "Retry saves original pending event batch");
            h.Output.CompleteSave();
            Check(c.App.Game.Revision == revision, "Save retry does not repeat turn or sale events");
            Check(h.Router.Popup.Text.Contains("income debt"), "Successful retry resumes debt choice");
        }

        private static void NetworkRouting()
        {
            Case c = NewCase(); Player debtor = c.App.Game.PlayersTrack.Players[0];
            Build(c, debtor, BuildingType.Pottery); Debt(c, debtor); ReachFinalTurn(c); c.App.Game.GameFlow.EndTurn();
            c.App.Metadata.GameType = GameType.Network;
            UiHarness h = Harness(c, "remote"); int revision = c.App.Game.Revision;
            h.Play.PlayGame();
            Check(h.Router.Popup.Text.Contains("Online play is disabled"), "Opening a network game shows the local-only notice");
            Check(c.App.Game.Revision == revision && h.Output.SaveCalls == 0 && h.Output.SpectatorTurns == 0, "Network guard makes no gameplay, save, or spectator call");
            h.Play.EndTurn();
            Check(c.App.Game.Revision == revision && h.Output.SaveCalls == 0, "Network EndTurn cannot mutate or save a game");
            h.Play.SaveGameComplete(true);
            Check(c.App.Game.Revision == revision && h.Output.SaveCalls == 0, "Network save callback cannot resume play");
            h.Router.Popup.Confirm(); h.Router.CompleteHide();
            Check(h.Router.Builder.ExitOutput.MenuCalls == 1, "The local-only notice returns to the menu without saving");
        }
        private static void NavigationCallbacks()
        {
            Case c = NewCase(); Player debtor = c.App.Game.PlayersTrack.Players[0]; Build(c, debtor, BuildingType.Pottery); Debt(c, debtor); ReachFinalTurn(c);
            UiHarness h = Harness(c); h.Play.EndTurn(); h.Output.CompleteSave(); Action obsoleteChoice = h.Router.Popup.Confirm;
            Case different = NewCase(); h.History.SetGame(different.App); h.Play.PlayGame();
            Check(h.Output.HumanTurns == 1, "A reused use case does not retain another game's pending modal");
            int revision = c.App.Game.Revision; obsoleteChoice(); h.Router.CompleteHide();
            Check(c.App.Game.Revision == revision && h.Output.SaveCalls == 1, "Obsolete modal callbacks cannot sell or save after game navigation");
            Case saving = NewCase(); ReachFinalTurn(saving); UiHarness pending = Harness(saving); pending.Play.EndTurn();
            pending.History.SetGame(different.App); pending.Play.PlayGame();
            Check(pending.Output.HumanTurns == 1, "A reused use case does not retain another game's save-in-progress flag");
            pending.Output.CompleteSave(); Check(pending.Output.HumanTurns == 1, "An obsolete save callback cannot advance another game");
            Case failing = NewCase(); ReachFinalTurn(failing); UiHarness staleFailure = Harness(failing); staleFailure.Play.EndTurn();
            staleFailure.History.SetGame(different.App); staleFailure.Output.CompleteSave(false);
            Check(staleFailure.Router.Popup == null, "An obsolete failed save cannot open a retry dialog in another game");
        }

        private sealed class Case { public BrassApplicationGame App; public BaseRepository Repo; }
        private sealed class UiHarness { public FakeRouter Router; public FakePlayOutput Output; public PlayGameUseCase Play; public BrassGameHistory History; }
        private sealed class FakePlayOutput : IPlayGameOutput
        {
            private readonly IRouter _router; private readonly Case c; private ISaveGameDelegate pending;
            public SellIndustryFlowUseCase Sell; public int SaveCalls, HumanTurns, SpectatorTurns; public bool HasPendingSave { get { return pending != null; } }
            public FakePlayOutput(IRouter router, Case c) { _router = router; this.c = c; }
            public void CompleteSave(bool successful = true) { ISaveGameDelegate callback = pending; pending = null; if (successful) c.Repo.Save(c.App); new InterfaceAdapters.Common.SaveGame.SaveGamePresenter(_router, new ImmediateDispatcher()).SaveGameCompleted(successful, callback); }
            public void SaveGame(BrassApplicationGame app, ISaveGameDelegate callback) { if (pending != null) throw new Exception("Overlapping saves"); SaveCalls++; pending = callback; }
            public void StartSellingIndustries() { Sell.StartSellIndustriesFlow(); }
            public void StartHumanTurn() { HumanTurns++; }
            public void StartAITurn(PlayerMetadata metadata) { }
            public void StartGameReview(PlayerMetadata metadata) { }
            public void StartNetworkHumanTurn() { }
            public void StartNetworkSpectatorTurn(PlayerMetadata metadata) { SpectatorTurns++; }
        }
        private sealed class FakeNetworkClient : INetworkBackendClient, INetworkBackendUserManagement
        {
            private readonly string id; public FakeNetworkClient(string id) { this.id = id; }
            public INetworkBackendUserManagement UserManagement { get { return this; } }
            public IGameStorage GameStorage { get { return null; } } public string AuthToken { get; set; }
            public string UserId { get { return id; } } public string UserNickname { get { return id; } }
            public Task Register(string email, string nickname, string password, Action<IEnumerable<NetworkBackendStatus>> done) { throw new NotSupportedException(); }
            public Task Login(string email, string password, Action<IEnumerable<NetworkBackendStatus>, string> done) { throw new NotSupportedException(); }
            public Task Logout() { throw new NotSupportedException(); }
            public Task ResetPassword(string email, Action<IEnumerable<NetworkBackendStatus>> done) { throw new NotSupportedException(); }
            public Task Me(Action<IEnumerable<NetworkBackendStatus>, string> done) { throw new NotSupportedException(); }
            public Task RegisterDeviceForPushNotification(string id, string platform, Action<IEnumerable<NetworkBackendStatus>> done) { throw new NotSupportedException(); }
            public Task UnregisterDeviceForPushNotification(string id, string platform, Action<IEnumerable<NetworkBackendStatus>> done) { throw new NotSupportedException(); }
        }
        private sealed class FakeRouter : IRouter
        {
            public FakeBuilder Builder = new FakeBuilder(); public FakeQuestion Popup; private FakeAnimation hide;
            public IUseCaseBuilder UseCaseBuilder { get { return Builder; } set { } }
            public IAnimation ShowView<T>() where T : class, IBaseView { Popup = new FakeQuestion(); return new FakeAnimation(); }
            public IAnimation HideView<T>() where T : class, IBaseView { hide = new FakeAnimation(); return hide; }
            public void CompleteHide() { FakeAnimation old = hide; hide = null; old.Complete(); }
            public T GetView<T>() where T : class, IBaseView { return Popup as T; }
            public void MoveOnTop<T>() where T : class, IBaseView { }
            public bool IsViewVisible<T>() where T : class, IBaseView { return Popup is T; }
            public bool IsViewCached<T>() where T : class, IBaseView { return Popup != null; }
            public void ProcessInput() { }
            public void RegisterKey(KeyEnum key, IBaseView view) { }
            public void UnregisterKey(KeyEnum key, IBaseView view) { }
            public void UnregisterAllKeys(IBaseView view) { }
        }
        private sealed class FakeBuilder : IUseCaseBuilder
        {
            public PlayGameUseCase Play;
            public FakeExitOutput ExitOutput = new FakeExitOutput();
            public TUseCase GetUseCase<TUseCase>() where TUseCase : IUseCase
            {
                if (typeof(TUseCase) == typeof(ExitToMenuUseCase)) return (TUseCase)(object)new ExitToMenuUseCase(ExitOutput, null, null, null, null, null, null);
                return (TUseCase)(object)Play;
            }
            public void BuildUseCases() { } public void RestoreUseCases() { } public void RestoreUseCaseOfType(Type type) { } public IController GetController(Type type) { return null; }
        }
        private sealed class ImmediateDispatcher : InterfaceAdapters.Utils.MainThreadDispatcher.IMainThreadDispatcher
        {
            public bool isMainThread { get { return true; } }
            public void InvokeAsync(Action action) { action(); }
            public void Invoke(Action action) { action(); }
        }
        private sealed class FakeExitOutput : IExitToMenuOutput
        {
            public int MenuCalls; public void ShowLoadingViewAndLoadMenuScene() { MenuCalls++; }
            public string GetTranslation(string key) { return key; } public void HideView() { }
            public void RefreshSettings(IAudioSettings audio, ILanguageSettings language, IGameSettings game) { }
        }
        private sealed class FakeAnimation : IAnimation
        {
            private Action callbacks; public void Complete() { if (callbacks != null) callbacks(); }
            public IAnimation Play() { return this; } public IAnimation Kill() { return this; } public IAnimation OnComplete(Action a) { callbacks += a; return this; }
            public IAnimation OnKill(Action a) { return this; } public IAnimation AppendCallback(Action a) { callbacks += a; return this; } public bool IsPlaying() { return false; }
        }
        private sealed class FakeQuestion : IQuestionPopupView
        {
            public string Text; public Action Confirm, Cancel;
            public void Setup(string text, bool negative, string yes, string no, Action confirm, Action cancel) { Text = text; Confirm = confirm; Cancel = cancel; }
            public IInputRegistration InputRegistration { get; set; } public bool DisableMenusUnderneath { get; set; } public bool DestroyWhenClosed { get; set; } public IController Controller { get; set; }
            public void HandleKeyDown(KeyEnum key) { } public void HandleKey(KeyEnum key) { } public void HandleKeyUp(KeyEnum key) { }
            public void EnableView() { } public void DisableView() { } public IAnimation CreateShowViewAnimation() { return new FakeAnimation(); }
            public IAnimation CreateHideViewAnimation() { return new FakeAnimation(); } public IAnimation CreateShowViewImmediatelyAnimation() { return new FakeAnimation(); }
            public void SetInteractable(bool value) { } public bool IsViewVisible() { return true; } public void MoveToTop() { } public void MoveToBottom() { }
        }
    }
}

