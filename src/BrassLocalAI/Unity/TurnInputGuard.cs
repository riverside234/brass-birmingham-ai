using System;
using BrassApplication.AI;
using BrassApplication.Game;
using BrassApplication.GameHistory;
using BrassApplication.UseCases.Game.Board;
using BrassApplication.UseCases.Game.PlayerActions;
using BrassApplication.UseCases.Game.HandOfCards;
using BrassApplication.UseCases.Game.Replay;
using BoardGameRules.Entities.Board.Regions;
using BoardGameRules.Entities.Cards;
using BrassPatch;
using BrassPatch.Persistence;

namespace BrassLegacyAI
{
    // Stock bots ran synchronously, so these human input routes never used to run
    // between planning and committing. They must reject clicks on an AI's turn.
    public static class TurnInputGuard
    {
        static bool HumanInput(object owner)
        {
            var history = ReflectionAccess.Get<IGameHistory>(owner, "_gameHistory");
            var app = history.CurrentGame;
            // Replays include the human's own action animations. Their historical
            // actor can still be Human, but that state must not accept new input.
            return app != null && (SaveExtensions.Get(app.Metadata) == null ||
                (!(history is BrassGameHistory games && ReflectionAccess.Get<bool>(games, "_movedBackInHistory")) &&
                !app.Game.GameProgressInformation.IsGameOver &&
                app.Metadata.GetPlayerMetadataByColor(app.Game.GameProgressInformation.CurrentPlayer.Color).AiType == AIType.Human));
        }
        static void Action(PlayerActionsUseCase owner, string method)
        { if (HumanInput(owner)) ReflectionAccess.Original(owner, method, Type.EmptyTypes); }
        public static void Build(PlayerActionsUseCase owner) { Action(owner, "StartBuildAction"); }
        public static void Network(PlayerActionsUseCase owner) { Action(owner, "StartNetworkAction"); }
        public static void Develop(PlayerActionsUseCase owner) { Action(owner, "StartDevelopAction"); }
        public static void Sell(PlayerActionsUseCase owner) { Action(owner, "StartSellAction"); }
        public static void Loan(PlayerActionsUseCase owner) { Action(owner, "StartLoanAction"); }
        public static void Scout(PlayerActionsUseCase owner) { Action(owner, "StartScoutAction"); }
        public static void Pass(PlayerActionsUseCase owner) { Action(owner, "StartPassAction"); }
        public static void Undo(PlayerActionsUseCase owner) { Action(owner, "UndoAction"); }
        public static void EndTurn(PlayerActionsUseCase owner)
        {
            if (!HumanInput(owner)) return;
            var app = ReflectionAccess.Get<IGameHistory>(owner, "_gameHistory").CurrentGame;
            if (app.Game.TurnFlow.GetRemainingActionsCount() == 0 && app.Game.TurnFlow.CurrentAction == null)
                ReflectionAccess.Original(owner, "EndTurn", Type.EmptyTypes);
        }
        public static void Region(BoardUseCase owner, RegionName region)
        {
            if (HumanInput(owner)) ReflectionAccess.Original(owner, "RegionSelected", new[] { typeof(RegionName) }, region);
        }
        public static void Card(HandOfCardsUseCase owner, BaseCard card)
        {
            if (HumanInput(owner)) ReflectionAccess.Original(owner, "OnCardSelected", new[] { typeof(BaseCard) }, card);
        }
        public static void SkipReplay(ReplayUseCase owner)
        {
            // Skip to me is a replay control. It may never replace the live game
            // while the worker is using it, including the scheduled-start frame.
            var history = ReflectionAccess.Get<IGameHistory>(owner, "_gameHistory");
            var app = history.CurrentGame;
            if (LocalTurnBridge.IsThinking || (app != null && history is BrassGameHistory games && ReferenceEquals(app, ReflectionAccess.Get<BrassApplicationGame>(games, "<LatestGame>k__BackingField")) &&
                app.Metadata.GetPlayerMetadataByColor(app.Game.GameProgressInformation.CurrentPlayer.Color).AiType != AIType.Human &&
                app.Game.TurnFlow.GetRemainingActionsCount() > 0)) return;
            if (ReflectionAccess.Get<bool>(owner, "_isReplayInProgress")) ReflectionAccess.Original(owner, "SkipReplay", Type.EmptyTypes);
        }
    }
}
