using System;
using System.Linq;
using BoardGameRules.Entities.Board;
using BoardGameRules.Entities.Board.Events;
using BoardGameRules.Entities.Board.Merchants;
using BoardGameRules.Entities.Board.Regions;
using BoardGameRules.Entities.NumericalResources;
using BoardGameRules.Entities.NumericalResources.Events;
using Frameworks.View.Game.Replay;
using InterfaceAdapters.Common;
using InterfaceAdapters.Game.Board;
using InterfaceAdapters.Game.PlayersSummary;
using InterfaceAdapters.Routing;
using Utils;

namespace BrassPatch
{
    public static class PresentationFixes
    {
        public static int MerchantVpTotal(int current, RegionName region) { return current + (region == RegionName.Shrewsbury ? 4 : 3); }
        public static int CappedIncome(int current, int gain) { return IncomeTrack.ChangeIncomeMarkerPositionBy(current, gain); }
        public static void ClampMerchantIncome(Board board, MerchantBonusReceived change)
        {
            if (change.BonusType != BonusType.Income) return;
            var players = ReflectionAccess.Get<PlayersTrack>(board, "_playersTrack");
            var player = players.GetPlayerByColor(change.PlayerColor);
            player.IncomeMarkerPosition = CappedIncome(player.IncomeMarkerPosition, 0);
        }
        private static IAnimationSequence Sequence(ReplayView view)
        {
            return (IAnimationSequence)view.GetType().GetMethod("CreateAnimationSequence", ReflectionAccess.Flags).Invoke(view, null);
        }
        private static IAnimation IncomeAnimation(IRouter router, int target, BoardGameRules.Entities.Players.PlayerColor color)
        {
            int level = IncomeTrack.GetCurrentIncomeLevel(target);
            return router.GetView<IPlayersSummaryView>().CreateChangeIncomeAnimation(
                (level > 0 ? "+" : "") + level, target + "/" + IncomeTrack.GetIncomeThreshold(target), color);
        }
        public static IAnimation MerchantAnimation(ReplayView view, MerchantBonusReceived change)
        {
            var router = ReflectionAccess.Get<IRouter>(view, "_router");
            var player = Global.Instance.BrassGameHistory.CurrentGame.Game.PlayersTrack.GetPlayerByColor(change.PlayerColor);
            switch (change.BonusType)
            {
                case BonusType.VP: return router.GetView<IPlayersSummaryView>().CreateChangeVPAnimation(MerchantVpTotal(player.VP, change.RegionName).ToString(), change.PlayerColor);
                case BonusType.Money: return router.GetView<IPlayersSummaryView>().CreateChangeMoneyAnimation((player.Money + 5).ToString(), change.PlayerColor);
                case BonusType.Income:
                    var sequence = Sequence(view);
                    sequence.Append(IncomeAnimation(router, CappedIncome(player.IncomeMarkerPosition, 2), change.PlayerColor));
                    return sequence;
                default: return Sequence(view);
            }
        }
        public static IAnimation IncomeMarkerAnimation(ReplayView view, IncomeMarkerPositionChanged change)
        {
            var router = ReflectionAccess.Get<IRouter>(view, "_router");
            var game = Global.Instance.BrassGameHistory.CurrentGame.Game;
            int current = game.PlayersTrack.GetPlayerByColor(change.Player).IncomeMarkerPosition;
            int target = CappedIncome(current, change.NumberOfPoints);
            var sequence = Sequence(view);
            sequence.Append(IncomeAnimation(router, target, change.Player));
            // Legacy events contain only an ID here. An ambiguous visual source must not stall replay.
            var sources = game.Board.Regions.SelectMany(r => r.BuildingSlots).Select(s => s.BuildedIndustry)
                .Where(i => i != null && i.UniqueId == change.SourceUniqueId).ToList();
            var industry = sources.Count == 1 ? sources[0] : null;
            var slot = industry == null ? null : game.Board.GetBuildingSlotContainingGivenIndustry(industry);
            if (slot != null) sequence.Insert(0f, router.GetView<IBoardView>().CreateIncomeMarkerPositionChangedAnimation(target - current, slot.UniqueId));
            return sequence;
        }
    }
}
