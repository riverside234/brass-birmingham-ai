using System;
using System.Globalization;
using System.Linq;
using System.Runtime.CompilerServices;
using BoardGameRules.Entities.Actions;
using BoardGameRules.Entities.Actions.Events;
using BoardGameRules.Entities.Board;
using BoardGameRules.Entities.Board.BuildingSlots;
using BoardGameRules.Entities.Board.Regions;
using BoardGameRules.Entities.Game;
using BoardGameRules.Entities.Industries;
using BoardGameRules.Utils;

namespace BrassPatch
{
    public static class IdentityFixes
    {
        private sealed class VersionState { public int Version; public long Ordinal; }
        private static readonly ConditionalWeakTable<IRandomer, VersionState> Versions = new ConditionalWeakTable<IRandomer, VersionState>();
        [ThreadStatic] private static int initializingVersion;
        public static int Version(GameStartConfiguration config)
        {
            return (int)config.GetType().GetField("BrassRulesVersion").GetValue(config);
        }
        public static void StartGame(BoardGameRulesGame game)
        {
            // Old saves reconstruct GameStarted through Init; they never call StartGame.
            game.GameStartConfiguration.GetType().GetField("BrassRulesVersion").SetValue(game.GameStartConfiguration, 1);
        }
        public static void Init(BoardGameRulesGame game, GameStartConfiguration config)
        {
            int version = Version(config);
            if (version < 0 || version > 1) throw new InvalidOperationException("Unsupported Brass rules version; install its matching patch before loading this game.");
            int previous = initializingVersion;
            initializingVersion = version;
            try { ReflectionAccess.Original(game, "Init", new[] { typeof(GameStartConfiguration) }, config); }
            finally { initializingVersion = previous; }
        }
        private static string Next(IRandomer randomer, string category)
        {
            if (randomer == null) return "";
            var state = Versions.GetValue(randomer, key => new VersionState { Version = initializingVersion });
            // Exactly one original draw per ID preserves board setup, shuffles and AI randomness.
            int randomValue = randomer.Next();
            long ordinal = ++state.Ordinal;
            return state.Version == 0 ? category + randomValue.ToString(CultureInfo.InvariantCulture)
                : category + "v1_" + ordinal.ToString(CultureInfo.InvariantCulture);
        }
        public static string Industry(IRandomer randomer) { return Next(randomer, "industry"); }
        public static string Card(IRandomer randomer) { return Next(randomer, "card"); }
        public static string Merchant(IRandomer randomer) { return Next(randomer, "merchantSlot"); }
        public static string Slot(IRandomer randomer) { return Next(randomer, "buildingSlot"); }
        public static Region RegionForIndustry(Board board, BaseIndustry industry)
        {
            var slot = SlotForIndustry(board, industry);
            return slot == null ? null : board.Regions.First(r => r.BuildingSlots.Any(s => ReferenceEquals(s, slot)));
        }
        public static BuildingSlot SlotForIndustry(Board board, BaseIndustry industry)
        {
            if (industry == null) return null;
            var slots = board.Regions.SelectMany(r => r.BuildingSlots).Where(s => s.BuildedIndustry != null).ToList();
            var exact = slots.FirstOrDefault(s => ReferenceEquals(s.BuildedIndustry, industry));
            if (exact != null) return exact;
            var matches = slots.Where(s => s.BuildedIndustry.UniqueId == industry.UniqueId && s.BuildedIndustry.BuildingType == industry.BuildingType && s.BuildedIndustry.Color == industry.Color).ToList();
            if (matches.Count > 1) throw new InvalidOperationException("Ambiguous legacy industry ID: " + industry.UniqueId + ". Keep the original save for recovery.");
            return matches.SingleOrDefault();
        }
        public static void IndustryFlipped(GeneralAction action, IndustryFlipped change)
        {
            var board = ReflectionAccess.Get<Board>(action, "Board");
            var matches = board.GetRegionByName(change.RegionName).BuildingSlots
                .Select(s => s.BuildedIndustry).Where(i => i != null && i.UniqueId == change.IndustryUniqueId).ToList();
            if (matches.Count != 1) throw new InvalidOperationException("Missing or ambiguous industry flip in " + change.RegionName + ": " + change.IndustryUniqueId + ". Keep the original save for recovery.");
            matches[0].IsFlipped = true;
        }
    }
}
