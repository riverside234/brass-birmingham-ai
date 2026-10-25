using System;
using System.Linq;
using System.Runtime.CompilerServices;
using BoardGameRules.Entities.Board.Links;
using BoardGameRules.Entities.Cards;
using BoardGameRules.Entities.Game;
using BoardGameRules.Entities.Players;
using BrassApplication.GameHistory;
using BrassApplication.UseCases.Game.Actions.NetworkAction;
using NetworkAction = BoardGameRules.Entities.Actions.NetworkAction;

namespace BrassPatch.NetworkInput
{
    public static class NetworkInputFixes
    {
        sealed class Selection { public bool Busy; }
        static readonly ConditionalWeakTable<NetworkActionUseCase, Selection> Selections =
            new ConditionalWeakTable<NetworkActionUseCase, Selection>();

        // Link buttons can outlive the prompt that installed their listeners.
        // A completed action has already cleared its player and discarded card.
        public static void LinkSelected(NetworkActionUseCase owner, Link link)
        {
            var selection = Selections.GetOrCreateValue(owner);
            if (selection.Busy || link == null) return;
            var history = ReflectionAccess.Get<IGameHistory>(owner, "_gameHistory");
            var app = history == null ? null : history.CurrentGame;
            var action = ReflectionAccess.Get<NetworkAction>(owner, "_networkAction");
            if (app == null || action == null ||
                ReflectionAccess.Get<NetworkActionUseCaseState>(owner, "_networkActionUseCaseState") != NetworkActionUseCaseState.SelectLink ||
                !ReferenceEquals(app.Game.TurnFlow.CurrentAction, action) ||
                !ReferenceEquals(ReflectionAccess.Get<object>(action, "AggregateRoot"), app.Game) ||
                !ReferenceEquals(ReflectionAccess.Get<Player>(owner, "_currentPlayer"), app.Game.GameProgressInformation.CurrentPlayer) ||
                !IsPrepared(action, app.Game) || ReflectionAccess.Get<BaseCard>(action, "_chosenCard") == null ||
                action.NeededCoal != 0 || action.NeededBeer != 0 ||
                (history is BrassGameHistory && ReflectionAccess.Get<bool>(history, "_movedBackInHistory")) ||
                !IsCurrentLink(app.Game, link) ||
                !action.GetValidConnectionSlots(app.Game.Board, app.Game.GameProgressInformation).Contains(link)) return;

            selection.Busy = true;
            try { ReflectionAccess.Original(owner, "LinkSelected", new[] { typeof(Link) }, link); }
            finally { selection.Busy = false; }
        }

        // Validate before the original increments _linkCounter. In the vendor
        // code a late call increments it, then dereferences the cleared player;
        // that invisible partial mutation breaks the next player's Network action.
        public static void BeforePlaceLink(NetworkAction action, Link link)
        {
            if (link == null) throw new InvalidOperationException("Given link is null.");
            var game = ReflectionAccess.Get<object>(action, "AggregateRoot") as BoardGameRulesGame;
            if (game == null || !IsPrepared(action, game))
                throw new InvalidOperationException("Network placement requires the current player's prepared Network action.");
            if (!IsCurrentLink(game, link))
                throw new InvalidOperationException("Network placement requires an unowned link on the current board.");
        }

        static bool IsPrepared(NetworkAction action, BoardGameRulesGame game)
        {
            return ReferenceEquals(game.TurnFlow.CurrentAction, action) &&
                ReflectionAccess.Get<Player>(action, "_player") != null &&
                ReferenceEquals(ReflectionAccess.Get<Player>(action, "_player"), game.GameProgressInformation.CurrentPlayer);
        }

        static bool IsCurrentLink(BoardGameRulesGame game, Link link)
        {
            return !link.HasOwner && game.Board.Regions.SelectMany(r => r.Links).Any(l => ReferenceEquals(l, link));
        }
    }
}
