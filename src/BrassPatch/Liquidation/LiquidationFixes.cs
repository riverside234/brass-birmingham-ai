using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Runtime.CompilerServices;
using BoardGameRules.Entities.Board.BuildingSlots;
using BoardGameRules.Entities.EventSourcing.Aggregates;
using BoardGameRules.Entities.Game;
using BoardGameRules.Entities.GamePhases;
using BoardGameRules.Entities.GamePhases.Events;
using BoardGameRules.Entities.Industries;
using BoardGameRules.Entities.NumericalResources.Events;
using BoardGameRules.Entities.Players;
using BoardGameRules.Entities.SellIndustry;
using BoardGameRules.Entities.SellIndustry.Events;
using BrassApplication.AI;
using BrassApplication.Game;
using BrassApplication.GameHistory;
using BrassApplication.UseCases.Common.SaveGame;
using BrassApplication.UseCases.Game.ExitToMenu;
using BrassApplication.UseCases.Game.PlayGame;
using BrassApplication.UseCases.Game.SellIndustryFlow;
using InterfaceAdapters.Common;
using InterfaceAdapters.Routing;

namespace BrassPatch.Liquidation
{
    /// <summary>Round-end liquidation is an event-reconstructed phase, not a synchronous AI shortcut.</summary>
    public static class LiquidationFixes
    {
        private sealed class Phase { public bool Active; }
        private static readonly ConditionalWeakTable<SellIndustryFlow, Phase> Phases = new ConditionalWeakTable<SellIndustryFlow, Phase>();
        private static readonly ConditionalWeakTable<PlayGameUseCase, UiState> UiStates = new ConditionalWeakTable<PlayGameUseCase, UiState>();
        private sealed class UiState { public BrassApplicationGame App; public bool PromptOpen; public bool SaveInProgress; }
        private sealed class SaveCallback : ISaveGameDelegate
        {
            private readonly PlayGameUseCase useCase;
            private readonly BrassApplicationGame app;
            public SaveCallback(PlayGameUseCase useCase, BrassApplicationGame app) { this.useCase = useCase; this.app = app; }
            public bool IsCurrent { get { return Object.ReferenceEquals(Get<IGameHistory>(useCase, "_gameHistory").CurrentGame, app); } }
            public void SaveGameComplete(bool successful)
            {
                if (IsCurrent) LiquidationFixes.SaveGameComplete(useCase, successful);
            }
        }
        private const BindingFlags Flags = BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic;

        private static FieldInfo Field(object value, string name)
        {
            for (Type t = value.GetType(); t != null; t = t.BaseType)
            {
                FieldInfo f = t.GetField(name, Flags | BindingFlags.DeclaredOnly);
                if (f != null) return f;
            }
            throw new MissingFieldException(value.GetType().FullName, name);
        }
        private static T Get<T>(object value, string name) { return (T)Field(value, name).GetValue(value); }
        private static void Set(object value, string name, object next) { Field(value, name).SetValue(value, next); }
        private static BoardGameRulesGame Owner(AggregateRootClient value) { return Get<BoardGameRulesGame>(value, "_aggregateRoot"); }
        private static void Original(object value, string method, params object[] args)
        {
            MethodInfo m = value.GetType().GetMethods(Flags).Single(x => x.Name == "BrassPatch_Original_" + method && x.GetParameters().Length == args.Length);
            try { m.Invoke(value, args); }
            catch (TargetInvocationException ex) { if (ex.InnerException != null) System.Runtime.ExceptionServices.ExceptionDispatchInfo.Capture(ex.InnerException).Throw(); throw; }
        }
        public static bool IsPending(SellIndustryFlow flow) { return Phases.GetOrCreateValue(flow).Active; }
        private static UiState State(PlayGameUseCase useCase)
        {
            UiState state = UiStates.GetOrCreateValue(useCase);
            BrassApplicationGame app = Get<IGameHistory>(useCase, "_gameHistory").CurrentGame;
            if (!Object.ReferenceEquals(state.App, app)) { state.App = app; state.PromptOpen = false; state.SaveInProgress = false; }
            return state;
        }

        public static void StartIndustrySelling(SellIndustryFlow flow)
        {
            if (IsPending(flow)) throw new InvalidOperationException("Industry liquidation is already pending.");
            Owner(flow).ApplyEvent(new IndustrySellingStarted());
        }

        public static void ApplyStarted(SellIndustryFlow flow, IndustrySellingStarted e)
        {
            BoardGameRulesGame game = Owner(flow);
            Set(flow, "<_moneyToPayBySelling>k__BackingField", game.PlayersTrack.Players.ToDictionary(p => p.Color, p => Math.Max(0, -(p.IncomeLevel + p.Money))));
            Set(flow, "<_soldInLastRound>k__BackingField", game.PlayersTrack.Players.ToDictionary(p => p.Color, p => new List<BaseIndustry>()));
            Set(flow, "<_lostVpsInLastRound>k__BackingField", game.PlayersTrack.Players.ToDictionary(p => p.Color, p => 0));
            Phases.GetOrCreateValue(flow).Active = true;
            Set(game.GameFlow, "_shouldEndRoundBeCalled", true);
            Set(game.GameFlow, "_isSellIndustryNecessary", true);
            FocusNextDebtor(game);
        }

        public static void SellIndustry(SellIndustryFlow flow, PlayerColor color, BuildingSlot slot)
        {
            BoardGameRulesGame game = Owner(flow);
            if (!IsPending(flow)) throw new InvalidOperationException("No industry liquidation is pending.");
            if (slot == null || !game.Board.Regions.SelectMany(r => r.BuildingSlots).Any(s => Object.ReferenceEquals(s, slot)))
                throw new InvalidOperationException("Select an industry on this game's board.");
            if (slot.BuildedIndustry == null || slot.BuildedIndustry.Color != color)
                throw new InvalidOperationException("Only the indebted player's own industries may be liquidated.");
            if (!flow._moneyToPayBySelling.ContainsKey(color) || flow._moneyToPayBySelling[color] <= 0)
                throw new InvalidOperationException("This player's debt is already covered.");
            game.ApplyEvent(new IndustrySold(color, slot.BuildedIndustry.MoneyCost / 2, slot.UniqueId));
        }

        public static void ApplySold(SellIndustryFlow flow, IndustrySold e)
        {
            BoardGameRulesGame game = Owner(flow);
            BuildingSlot slot = game.Board.GetBuildingSlotByUniqueId(e.SlotUniqueId);
            // Existing saves contain the same event sequence. Reconstruction never emits new events.
            if (!IsPending(flow) || slot == null || slot.BuildedIndustry == null)
                throw new InvalidOperationException("Invalid persisted industry liquidation.");
            flow._moneyToPayBySelling[e.PlayerColor] = Math.Max(0, flow._moneyToPayBySelling[e.PlayerColor] - e.MoneyToGet);
            flow._soldInLastRound[e.PlayerColor].Add(slot.BuildedIndustry);
            game.PlayersTrack.GetPlayerByColor(e.PlayerColor).Money += e.MoneyToGet;
            slot.BuildedIndustry = null;
            FocusNextDebtor(game);
        }

        public static void EndIndustrySelling(SellIndustryFlow flow)
        {
            if (!IsPending(flow)) throw new InvalidOperationException("No industry liquidation is pending.");
            if (!flow.CanIndustrySellingBeCompleted()) throw new InvalidOperationException("Industry liquidation still requires a choice.");
            BoardGameRulesGame game = Owner(flow);
            Dictionary<PlayerColor, int> losses = new Dictionary<PlayerColor, int>();
            foreach (KeyValuePair<PlayerColor, int> debt in flow._moneyToPayBySelling)
            {
                int loss = Math.Min(game.PlayersTrack.GetPlayerByColor(debt.Key).VP, debt.Value);
                losses[debt.Key] = loss;
                if (loss > 0) game.ApplyEvent(new VictoryPointsChanged(debt.Key, -loss));
            }
            game.ApplyEvent(new IndustrySellingEnded(flow._soldInLastRound.ToDictionary(x => x.Key, x => new List<BaseIndustry>(x.Value)), losses));
        }

        public static void ApplyEnded(SellIndustryFlow flow, IndustrySellingEnded e)
        {
            Set(flow, "<_soldInLastRound>k__BackingField", e.SoldInLastRound.ToDictionary(x => x.Key, x => new List<BaseIndustry>(x.Value)));
            Set(flow, "<_lostVpsInLastRound>k__BackingField", new Dictionary<PlayerColor, int>(e.LostVpsInLastRound));
            Phases.GetOrCreateValue(flow).Active = false;
            Set(Owner(flow).GameFlow, "_isSellIndustryNecessary", false);
        }

        private static void FocusNextDebtor(BoardGameRulesGame game)
        {
            Player next = game.PlayersTrack.Players.FirstOrDefault(p => !game.SellIndustryFlow.CanIndustrySellingBeCompleted(p.Color));
            if (next != null) game.GameProgressInformation.CurrentPlayer = next;
        }

        public static void AfterTurnEnded(GameFlow flow, TurnEnded e)
        {
            BoardGameRulesGame game = Owner(flow);
            Set(flow, "_shouldEndRoundBeCalled", game.GameProgressInformation.CurrentTurnNumber == (int)game.GameStartConfiguration.NumberOfPlayers);
        }
        public static void AfterRoundEnded(GameFlow flow, RoundEnded e)
        {
            Set(flow, "_shouldEndRoundBeCalled", false);
            Set(flow, "_isSellIndustryNecessary", false);
        }
        public static void EndRound(GameFlow flow)
        {
            SellIndustryFlow debt = Owner(flow).SellIndustryFlow;
            if (IsPending(debt) && !debt.CanIndustrySellingBeCompleted())
                throw new InvalidOperationException("Finish the pending industry liquidation before ending the round.");
            Original(flow, "EndRound");
        }

        public static void EndTurn(PlayGameUseCase useCase)
        {
            if (BlockNetwork(useCase)) return;
            UiState state = State(useCase);
            if (state.SaveInProgress || state.PromptOpen) return;
            BrassApplicationGame app = Get<IGameHistory>(useCase, "_gameHistory").CurrentGame;
            // Capture the player who clicked End before EndTurn/EndRound changes
            // the actor. This is the one automatic checkpoint: the completed
            // human turn and resulting next-player/round boundary, before AI play.
            bool humanFinished = app.Metadata.GetPlayerMetadataByColor(app.Game.GameProgressInformation.CurrentPlayer.Color).AiType == AIType.Human;
            if (!app.Game.GameFlow.IsLastTurnOver()) app.Game.GameFlow.EndTurn();
            if (app.Game.GameFlow.IsLastTurnOver() && !IsPending(app.Game.SellIndustryFlow)) app.Game.GameFlow.EndRound();
            if (humanFinished) Save(useCase); else useCase.PlayGame();
        }

        private static void Save(PlayGameUseCase useCase)
        {
            UiState state = State(useCase);
            state.SaveInProgress = true;
            Get<IPlayGameOutput>(useCase, "_presenter").SaveGame(state.App, new SaveCallback(useCase, state.App));
        }

        public static bool HandlesSaveFailure(ISaveGameDelegate callback) { return callback is SaveCallback; }
        public static bool IsCurrentSave(ISaveGameDelegate callback) { return callback is SaveCallback && ((SaveCallback)callback).IsCurrent; }

        public static void SaveGameComplete(PlayGameUseCase useCase, bool successful)
        {
            if (BlockNetwork(useCase)) return;
            UiState state = State(useCase);
            state.SaveInProgress = false;
            if (successful) { useCase.PlayGame(); return; }
            IRouter router = RouterFor(Get<IPlayGameOutput>(useCase, "_presenter"));
            state.PromptOpen = true;
            BrassApplicationGame app = state.App;
            Action retry = delegate
            {
                if (!Object.ReferenceEquals(Get<IGameHistory>(useCase, "_gameHistory").CurrentGame, app)) return;
                state.PromptOpen = false; Save(useCase);
            };
            ShowQuestion(router, "The game could not be saved. Retry before continuing.", "Retry save", "Retry save", retry, retry);
        }

        public static void PlayGame(PlayGameUseCase useCase)
        {
            if (BlockNetwork(useCase)) return;
            UiState state = State(useCase);
            if (state.SaveInProgress || state.PromptOpen) return;
            BrassApplicationGame app = Get<IGameHistory>(useCase, "_gameHistory").CurrentGame;
            SellIndustryFlow flow = app.Game.SellIndustryFlow;
            if (!IsPending(flow)) { Original(useCase, "PlayGame"); return; }
            IPlayGameOutput presenter = Get<IPlayGameOutput>(useCase, "_presenter");
            FocusNextDebtor(app.Game);
            var aiState = BrassPatch.Persistence.SaveExtensions.Get(app.Metadata);
            if ((string)aiState?["family"] == "nn") {
                // The qualified public-state contract resolves debt in color
                // order. Focus is presentation state; each sale remains a normal
                // persisted event. Apply the same order to human and NN debtors.
                var next = app.Game.PlayersTrack.Players.OrderBy(p => (int)p.Color)
                    .FirstOrDefault(p => !flow.CanIndustrySellingBeCompleted(p.Color));
                if (next != null) app.Game.GameProgressInformation.CurrentPlayer = next;
            }
            Player current = app.Game.GameProgressInformation.CurrentPlayer;
            PlayerMetadata metadata = app.Metadata.GetPlayerMetadataByColor(current.Color);
            if (flow.CanIndustrySellingBeCompleted()) { app.Game.GameFlow.EndRound(); useCase.PlayGame(); return; }
            if (metadata.AiType == AIType.Human) { presenter.StartSellingIndustries(); return; }
            if ((string)aiState?["family"] == "nn") {
                // Late binding avoids a helper dependency cycle. An unavailable
                // native route stops play; it must never fall through to V4.
                var type = Type.GetType("BrassLegacyAI.LocalTurnBridge, BrassLegacyAI", true);
                if (!(bool)type.GetMethod("PlayNativeDebt").Invoke(null, new object[] { useCase }))
                    throw new InvalidOperationException("NN table has no native debt handler.");
                return;
            }
            // Preserve the shipped AI's valuation, but never let it liquidate a human player's tile.
            BuildingSlot choice = Eligible(app, current.Color).OrderBy(s =>
                ((app.Game.GameProgressInformation.CurrentGameEra == GameEra.CanalEra && s.BuildedIndustry.Level == 2) ? 2 : 1) *
                (s.BuildedIndustry.VPForBuilding + s.BuildedIndustry.VPPerLink)).ThenBy(s => s.UniqueId, StringComparer.Ordinal).First();
            flow.SellIndustry(current.Color, choice);
            useCase.PlayGame();
        }

        public static void StartSellIndustriesFlow(SellIndustryFlowUseCase useCase)
        {
            IRouter router = RouterFor(Get<ISellIndustryFlowOutput>(useCase, "_presenter"));
            PlayGameUseCase play = router.UseCaseBuilder.GetUseCase<PlayGameUseCase>();
            if (BlockNetwork(play)) return;
            BrassApplicationGame app = Get<IGameHistory>(useCase, "_gameHistory").CurrentGame;
            if (!IsPending(app.Game.SellIndustryFlow)) { play.PlayGame(); return; }
            ShowChoice(play, router, app, 0);
        }

        private static List<BuildingSlot> Eligible(BrassApplicationGame app, PlayerColor color)
        {
            return app.Game.Board.Regions.SelectMany(r => r.BuildingSlots).Where(s => s.BuildedIndustry != null && s.BuildedIndustry.Color == color).ToList();
        }

        private static void ShowChoice(PlayGameUseCase play, IRouter router, BrassApplicationGame app, int index)
        {
            if (!Object.ReferenceEquals(Get<IGameHistory>(play, "_gameHistory").CurrentGame, app)) return;
            UiState state = State(play);
            SellIndustryFlow flow = app.Game.SellIndustryFlow;
            if (!IsPending(flow) || flow.CanIndustrySellingBeCompleted()) { state.PromptOpen = false; play.PlayGame(); return; }
            Player player = app.Game.GameProgressInformation.CurrentPlayer;
            List<BuildingSlot> choices = Eligible(app, player.Color);
            BuildingSlot slot = choices[index % choices.Count];
            BaseIndustry industry = slot.BuildedIndustry;
            int proceeds = industry.MoneyCost / 2;
            int debt = flow.RemainingDebtToPay(player.Color);
            string text = app.Metadata.GetPlayerMetadataByColor(player.Color).NickName + " (" + player.Color + ") must cover £" + debt + " of income debt.\n\n" +
                app.Game.Board.GetRegionContainingGivenSlot(slot).Name + ": " + industry.BuildingType + " level " + industry.Level +
                (industry.IsFlipped ? " (flipped)" : " (unflipped)") + "\nReceive £" + proceeds + "; debt remaining £" + Math.Max(0, debt - proceeds) +
                ".\n\nIndustry " + (index % choices.Count + 1) + " of " + choices.Count + ". Use Next industry to choose another.";
            state.PromptOpen = true;
            ShowQuestion(router, text, "Liquidate £" + proceeds, "Next industry", delegate
            {
                if (!Object.ReferenceEquals(Get<IGameHistory>(play, "_gameHistory").CurrentGame, app)) return;
                state.PromptOpen = false;
                flow.SellIndustry(player.Color, slot);
                play.PlayGame();
            }, () => ShowChoice(play, router, app, index + 1));
        }

        private static IRouter RouterFor(object presenter) { return Get<IRouter>(presenter, "_router"); }
        private static bool BlockNetwork(PlayGameUseCase useCase)
        {
            if (Get<IGameHistory>(useCase, "_gameHistory").CurrentGame.Metadata.GameType != GameType.Network) return false;
            UiState state = State(useCase);
            if (state.PromptOpen) return true;
            state.PromptOpen = true;
            IRouter router = RouterFor(Get<IPlayGameOutput>(useCase, "_presenter"));
            Action back = delegate { router.UseCaseBuilder.GetUseCase<ExitToMenuUseCase>().BackToMenu(); };
            ShowQuestion(router, "Online play is disabled while the local rules patch is installed. Continue with a local game, or restore the original game files to play online.", "Return to menu", "Return to menu", back, back);
            return true;
        }
        private static void ShowQuestion(IRouter router, string text, string confirm, string alternate, Action onConfirm, Action onAlternate)
        {
            router.ShowView<IQuestionPopupView>();
            IQuestionPopupView view = router.GetView<IQuestionPopupView>();
            // The existing popup is destroyed after its hide animation. Defer the next prompt until then,
            // so keyboard cleanup cannot remove the next prompt's callbacks and listeners cannot accumulate.
            bool consumed = false;
            Action<Action> finish = callback =>
            {
                if (consumed) return;
                consumed = true;
                view.SetInteractable(false);
                router.HideView<IQuestionPopupView>().AppendCallback(callback);
            };
            view.Setup(text, true, confirm, alternate, () => finish(onConfirm), () => finish(onAlternate));
        }
    }
}
