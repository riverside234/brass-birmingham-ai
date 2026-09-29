using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Runtime.CompilerServices;
using BoardGameRules.Entities.Actions;
using BoardGameRules.Entities.Cards;
using BoardGameRules.Entities.Game;
using BoardGameRules.Entities.Players;
using BoardGameRules.Entities.TurnPhases;
using BrassApplication.Game;
using BrassApplication.GameHistory;
using BrassApplication.UseCases.Game.Actions.ScoutAction;
using InterfaceAdapters.Game.Actions.ScoutAction;

namespace BrassPatch.Scout
{
    public static class ScoutFixes
    {
        private sealed class Operation
        {
            public BrassApplicationGame App;
            public BoardGameRules.Entities.Actions.ScoutAction Action;
            public Player Player;
            public readonly List<BaseCard> Selection = new List<BaseCard>();
            public bool Active = true;
            public bool Queued;
        }
        private sealed class State { public Operation Current; }
        private static readonly ConditionalWeakTable<ScoutActionUseCase, State> States = new ConditionalWeakTable<ScoutActionUseCase, State>();
        private static IGameHistory History(ScoutActionUseCase owner) { return ReflectionAccess.Get<IGameHistory>(owner, "_gameHistory"); }
        private static IScoutActionOutput Output(ScoutActionUseCase owner) { return ReflectionAccess.Get<IScoutActionOutput>(owner, "_presenter"); }
        private static BoardGameRulesGame Game(BaseAction action) { return ReflectionAccess.Get<BoardGameRulesGame>(action, "AggregateRoot"); }
        private static bool Current(ScoutActionUseCase owner, Operation op)
        {
            return op != null && op.Active && OwnsCurrentAction(owner, op);
        }
        private static bool OwnsCurrentAction(ScoutActionUseCase owner, Operation op)
        {
            return op != null && Object.ReferenceEquals(States.GetOrCreateValue(owner).Current, op)
                && Object.ReferenceEquals(History(owner).CurrentGame, op.App)
                && Object.ReferenceEquals(op.App.Game.TurnFlow.CurrentAction, op.Action)
                && Object.ReferenceEquals(op.App.Game.GameProgressInformation.CurrentPlayer, op.Player)
                && op.App.Game.TurnFlow.GetRemainingActionsCount() > 0;
        }
        private static bool Owned(Player player, BaseCard card)
        {
            return card != null && player.Hand.Any(x => Object.ReferenceEquals(x, card))
                && player.Hand.Count(x => x.UniqueId == card.UniqueId) == 1;
        }
        private static void Invoke(ScoutActionUseCase owner, string name)
        {
            MethodInfo method = typeof(ScoutActionUseCase).GetMethod(name, BindingFlags.Instance | BindingFlags.NonPublic);
            try { method.Invoke(owner, null); }
            catch (TargetInvocationException ex) { System.Runtime.ExceptionServices.ExceptionDispatchInfo.Capture(ex.InnerException ?? ex).Throw(); throw; }
        }

        public static void StartScoutAction(ScoutActionUseCase owner)
        {
            State state = States.GetOrCreateValue(owner);
            if (OwnsCurrentAction(owner, state.Current)) CancelAction(owner);
            else if (state.Current != null) state.Current.Active = false;
            BrassApplicationGame app = History(owner).CurrentGame;
            if (app == null || app.Game.TurnFlow.GetRemainingActionsCount() <= 0 || app.Game.TurnFlow.CurrentAction != null)
                throw new InvalidOperationException("A Scout action cannot be started now.");
            var action = app.Game.TurnFlow.ScoutAction;
            if (action.GetScoutActionNotPossibleReasons(app.Game.Board, app.Game.GameProgressInformation).Count != 0)
                throw new InvalidOperationException("The Scout action is unavailable.");
            state.Current = new Operation { App = app, Action = action, Player = app.Game.GameProgressInformation.CurrentPlayer };
            ReflectionAccess.Set(owner, "_discardedCardCount", 0);
            ReflectionAccess.Set(owner, "_isScoutPanelVisible", false);
            ReflectionAccess.Original(owner, "StartScoutAction", Type.EmptyTypes);
        }

        public static void SelectCard(ScoutActionUseCase owner, BaseCard card)
        {
            Operation op = States.GetOrCreateValue(owner).Current;
            if (!Current(owner, op) || op.Queued || !Owned(op.Player, card) || op.Selection.Any(x => x.UniqueId == card.UniqueId)) return;
            int count = ReflectionAccess.Get<int>(owner, "_discardedCardCount");
            if (count == 0)
            {
                // The domain gate checks eligibility before even the action card is discarded.
                op.Action.DiscardCard(card, op.Player);
                ReflectionAccess.Set(owner, "_discardedCardCount", 1);
                ReflectionAccess.Set(owner, "_isScoutPanelVisible", true);
                Output(owner).ShowExchangeForWildCardsView(op.App, owner);
                Invoke(owner, "PresentEvents");
                return;
            }
            if (count < 1 || count > 2) return;
            op.Selection.Add(card);
            ReflectionAccess.Set(owner, "_cardsToDiscard", op.Selection);
            ReflectionAccess.Set(owner, "_discardedCardCount", count + 1);
            if (count == 1) Output(owner).SetupCardInSlot(0, card, delegate { });
            else
            {
                op.Queued = true;
                Output(owner).SetupCardInSlot(1, card, delegate { Complete(owner, op); });
            }
        }

        private static void Complete(ScoutActionUseCase owner, Operation op)
        {
            if (!Current(owner, op) || !op.Queued) return;
            // Recheck the entire command before changing presentation or committing events.
            ValidateExchange(op.Action, op.Selection, op.Player);
            op.Active = false; // One-shot even if a presenter synchronously reenters.
            op.Action.Scout(new List<BaseCard>(op.Selection), op.Player);
            // Commit only the captured flow, before any presentation can navigate or
            // synchronously start a different action/game.
            op.App.Game.TurnFlow.ConfirmCurrentAction();
            ReflectionAccess.Set(owner, "_discardedCardCount", 0);
            ReflectionAccess.Set(owner, "_isScoutPanelVisible", false);
            Output(owner).HideExchangeForWildCardsView();
            if (!DisplayCurrent(owner, op)) return;
            Output(owner).RestoreCardsView();
            if (!DisplayCurrent(owner, op)) return;
            Output(owner).HideActionInProgressView();
            if (!DisplayCurrent(owner, op)) return;
            Output(owner).PresentEvents(op.App.Game.GetUncommittedEvents(), op.App.Metadata.GetPlayerMetadataByColor(op.Player.Color), owner);
        }

        private static bool DisplayCurrent(ScoutActionUseCase owner, Operation op)
        {
            return Object.ReferenceEquals(States.GetOrCreateValue(owner).Current, op)
                && Object.ReferenceEquals(History(owner).CurrentGame, op.App)
                && Object.ReferenceEquals(op.App.Game.GameProgressInformation.CurrentPlayer, op.Player)
                && op.App.Game.TurnFlow.CurrentAction == null;
        }

        public static void CancelAction(ScoutActionUseCase owner)
        {
            Operation op = States.GetOrCreateValue(owner).Current;
            bool current = OwnsCurrentAction(owner, op);
            if (op != null) op.Active = false;
            if (current) ReflectionAccess.Original(owner, "CancelAction", Type.EmptyTypes);
        }

        // All newly queued closures carry an Operation. Old unbound callbacks must not
        // acquire the current operation merely because another Scout has started.
        public static void LegacyComplete(ScoutActionUseCase owner) { }

        public static object CaptureOperation(ScoutActionUseCase owner) { return owner == null ? null : States.GetOrCreateValue(owner).Current; }
        public static void InvalidateOperation(ScoutActionUseCase owner, object token)
        {
            if (owner == null || token == null) return;
            Operation op = States.GetOrCreateValue(owner).Current;
            if (Object.ReferenceEquals(op, token)) op.Active = false;
        }
        public static void DiscardCard(BaseAction action, BaseCard card, Player player)
        {
            var scout = action as BoardGameRules.Entities.Actions.ScoutAction;
            if (scout != null)
            {
                BoardGameRulesGame game = Game(action);
                if (player == null || !Object.ReferenceEquals(game.GameProgressInformation.CurrentPlayer, player)
                    || !Object.ReferenceEquals(game.TurnFlow.CurrentAction, action) || game.TurnFlow.GetRemainingActionsCount() <= 0
                    || !Owned(player, card) || ReflectionAccess.Get<BaseCard>(action, "_chosenCard") != null
                    || scout.GetScoutActionNotPossibleReasons(game.Board, game.GameProgressInformation).Count != 0)
                    throw new InvalidOperationException("The Scout action card or starting state is invalid.");
            }
            MethodInfo original = typeof(BaseAction).GetMethod("BrassPatch_Original_DiscardCard", ReflectionAccess.Flags, null, new[] { typeof(BaseCard), typeof(Player) }, null);
            try { original.Invoke(action, new object[] { card, player }); }
            catch (TargetInvocationException ex) { System.Runtime.ExceptionServices.ExceptionDispatchInfo.Capture(ex.InnerException ?? ex).Throw(); throw; }
        }

        private static void ValidateExchange(BoardGameRules.Entities.Actions.ScoutAction action, List<BaseCard> cards, Player player)
        {
            BoardGameRulesGame game = Game(action);
            BaseCard chosen = ReflectionAccess.Get<BaseCard>(action, "_chosenCard");
            if (player == null || !Object.ReferenceEquals(game.GameProgressInformation.CurrentPlayer, player)
                || !Object.ReferenceEquals(game.TurnFlow.CurrentAction, action) || game.TurnFlow.GetRemainingActionsCount() <= 0
                || !Object.ReferenceEquals(ReflectionAccess.Get<Player>(action, "_player"), player) || chosen == null
                || player.Hand.Any(x => x.UniqueId == chosen.UniqueId)
                || !player.DiscardPile.Any(x => Object.ReferenceEquals(x, chosen))
                || cards == null || cards.Count != 2 || cards.Any(x => !Owned(player, x))
                || cards[0].UniqueId == cards[1].UniqueId || cards.Any(x => x.UniqueId == chosen.UniqueId)
                || player.Hand.Any(x => x is WildIndustryCard || x is WildLocationCard)
                || game.Board.WildIndustryCardDeck.Count == 0 || game.Board.WildLocationCardDeck.Count == 0)
                throw new InvalidOperationException("Scout requires two distinct remaining cards and both wild cards.");
        }

        public static void Scout(BoardGameRules.Entities.Actions.ScoutAction action, List<BaseCard> cards, Player player)
        {
            ValidateExchange(action, cards, player);
            ReflectionAccess.Original(action, "Scout", new[] { typeof(List<BaseCard>), typeof(Player) }, new List<BaseCard>(cards), player);
        }

        public static void ConfirmCurrentAction(TurnFlow flow)
        {
            if (flow.CurrentAction == null || flow.GetRemainingActionsCount() <= 0)
                throw new InvalidOperationException("There is no current action to confirm.");
            ReflectionAccess.Original(flow, "ConfirmCurrentAction", Type.EmptyTypes);
        }
    }
}
