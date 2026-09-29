using System;
using System.Linq;
using Newtonsoft.Json.Linq;
using BoardGameRules.Entities.Common;
using BoardGameRules.Entities.Game;
using BoardGameRules.Entities.Players;

namespace BrassHost
{
    public sealed partial class Host
    {
        public static JObject Configuration(GameStartConfiguration config)
        {
            return new JObject { ["player_count"] = (int)config.NumberOfPlayers,
                ["introductory"] = config.IsIntroductoryGame,
                ["seats"] = new JArray(config.Players.OrderBy(p => p.SittingOrder).Select(p =>
                    new JObject { ["color"] = (int)p.Color, ["sitting_order"] = p.SittingOrder })) };
        }
        static GameStartConfiguration ReadConfiguration(int seed, JObject setup)
        {
            var token = setup?["configuration"] ?? setup?["position"]?["configuration"];
            if (token == null) return new GameStartConfiguration(NumberOfPlayers.Four) { Seed = seed };
            int count = (int)token["player_count"];
            if (!Wire.ValidPlayerCount(count)) throw new Exception("unsupported_player_count");
            var seats = token["seats"].Select(p => new PlayerStartConfiguration((PlayerColor)(int)p["color"], (int)p["sitting_order"])).ToList();
            if (seats.Count != count || seats.Select(p => p.Color).Distinct().Count() != count ||
                seats.Any(p => (int)p.Color < 0 || (int)p.Color > 3) ||
                !seats.Select(p => p.SittingOrder).OrderBy(x => x).SequenceEqual(Enumerable.Range(0, count)))
                throw new Exception("invalid_player_mapping");
            return new GameStartConfiguration((NumberOfPlayers)count) { Seed = seed, Players = seats,
                IsIntroductoryGame = (bool)token["introductory"] };
        }
        public static JObject Composition(JObject configuration)
        {
            // Counts from a separate standard setup are public; no live deck order or hand is read.
            var h = new Host(1, new JObject { ["configuration"] = configuration.DeepClone() });
            var cards = h.Game.Board.CardDeck.Concat(h.Seats.SelectMany(p => p.Hand.Concat(p.DiscardPile)));
            var result = new JObject();
            foreach (var group in cards.GroupBy(Wire.Card).OrderBy(g => g.Key, StringComparer.Ordinal)) result[group.Key] = group.Count();
            return result;
        }
        void CaptureDiscardStarts()
        {
            discardStarts = new int[4];
            foreach (var p in Seats) discardStarts[(int)p.Color] = p.DiscardPile.Count;
        }
    }
}
