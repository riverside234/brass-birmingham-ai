using System;
using System.Collections.Generic;
using System.Linq;
using BoardGameRules.Entities.Actions.Events;
using BoardGameRules.Entities.Board;
using BoardGameRules.Entities.Board.BuildingSlots;
using BoardGameRules.Entities.Board.Regions;
using BoardGameRules.Entities.Common;
using BoardGameRules.Entities.EventSourcing.Aggregates;
using BoardGameRules.Entities.EventSourcing.Events;
using BoardGameRules.Entities.Game;
using BoardGameRules.Entities.Industries;
using BoardGameRules.Entities.NumericalResources;

namespace BrassPatch
{
    /// <summary>Legacy IDs stay unchanged; context-free lookups must never guess between colliding objects.</summary>
    public static class IdentityCompatibility
    {
        private static T Single<T>(IEnumerable<T> matches, string id, string kind) where T : class
        {
            var candidates = matches.Take(2).ToList();
            if (candidates.Count > 1)
                throw new InvalidOperationException("Ambiguous legacy " + kind + " ID: " + id +
                    ". This action cannot safely identify its target. Keep the original save for recovery.");
            return candidates.SingleOrDefault();
        }

        private static IEnumerable<BaseIndustry> Industries(Board board)
        {
            return board.Regions.SelectMany(r => r.BuildingSlots).Select(s => s.BuildedIndustry).Where(i => i != null);
        }

        public static ICoalSource Coal(Board board, string id)
        {
            return Single(Industries(board).OfType<ICoalSource>().Concat(new[] { board.CoalMarket }).Where(i => i.UniqueId == id), id, "coal source");
        }

        public static IIronSource Iron(Board board, string id)
        {
            return Single(Industries(board).OfType<IIronSource>().Concat(new[] { board.IronMarket }).Where(i => i.UniqueId == id), id, "iron source");
        }

        public static IBeerSource Beer(Board board, string id)
        {
            return Single(Industries(board).OfType<IBeerSource>().Concat(board.GetBorderRegions(r => true)
                .SelectMany(r => r.MerchantSlots).Cast<IBeerSource>()).Where(i => i.UniqueId == id), id, "beer source");
        }

        public static BaseIndustry BoardIndustry(Board board, string id)
        {
            return Single(Industries(board).Where(i => i.UniqueId == id), id, "industry");
        }

        public static BaseIndustry MatIndustry(PlayersTrack players, string id)
        {
            var types = Enum.GetValues(typeof(BuildingType)).Cast<BuildingType>().Where(t => t != BuildingType.Undefined);
            return Single(players.Players.SelectMany(p => types.SelectMany(t => p.Mat.GetIndustryList(t)))
                .Where(i => i.UniqueId == id), id, "player-mat industry");
        }

        public static void BeforeEvent(AggregateRoot aggregate, IEvent change)
        {
            var game = aggregate as BoardGameRulesGame;
            if (game == null || game.Board == null) return;
            var coal = change as CoalSupplied;
            if (coal != null) { Coal(game.Board, coal.SourceUniqueId); return; }
            var iron = change as IronSupplied;
            if (iron != null) { Iron(game.Board, iron.SourceId); return; }
            var beer = change as BeerSupplied;
            if (beer != null) Beer(game.Board, beer.BeerSourceId);
        }
    }
}
