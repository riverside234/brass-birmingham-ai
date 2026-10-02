using System;
using System.Collections.Generic;
using System.Linq;
using BoardGameRules.Entities.Actions.Events;
using BoardGameRules.Entities.Board.BuildingSlots;
using BoardGameRules.Entities.Board.Regions;
using BoardGameRules.Entities.Cards;
using BoardGameRules.Entities.EventSourcing.Events;
using BoardGameRules.Entities.GamePhases.Events;
using BoardGameRules.Entities.Players;
using BrassApplication.GameHistory;

namespace BrassPatch.CardTracker
{
    public sealed class TrackedCard
    {
        public string Key, Name;
        public int Total, Played;
        public RegionName? City;
        public int Unseen { get { return Total - Played; } }
    }
    public static class FourPlayerCardManifest
    {
        public static List<TrackedCard> Create()
        {
            var cards = new List<TrackedCard>();
            string[] cities = { "Belper", "Birmingham", "BurtonOnTrent", "Cannock", "Coalbrookdale", "Coventry", "Derby", "Dudley",
                "Kidderminster", "Leek", "Nuneaton", "Redditch", "Stafford", "StokeOnTrent", "Stone", "Tamworth", "Uttoxeter", "Walsall", "Wolverhampton", "Worcester" };
            int[] totals = { 2,3,2,2,3,3,3,2,2,2,1,1,2,3,2,1,2,1,2,2 };
            for (int i = 0; i < cities.Length; i++) {
                var city = (RegionName)Enum.Parse(typeof(RegionName), cities[i]);
                cards.Add(new TrackedCard { Key = "location:" + city, City = city, Total = totals[i],
                    Name = cities[i] == "BurtonOnTrent" ? "Burton-on-Trent" : cities[i] == "StokeOnTrent" ? "Stoke-on-Trent" : cities[i] });
            }
            cards.Add(new TrackedCard { Key = "industry:Brewery", Name = "Brewery", Total = 5 });
            cards.Add(new TrackedCard { Key = "industry:CoalMine", Name = "Coal Mine", Total = 3 });
            cards.Add(new TrackedCard { Key = "industry:IronWorks", Name = "Iron Works", Total = 4 });
            cards.Add(new TrackedCard { Key = "industry:CottonMill+Manufacturer", Name = "Cotton / Manufacturer", Total = 8 });
            cards.Add(new TrackedCard { Key = "industry:Pottery", Name = "Pottery", Total = 3 });
            return cards;
        }
        public static string IndustryKey(BuildingType type)
        { return "industry:" + (type == BuildingType.CottonMill || type == BuildingType.Manufacturer ? "CottonMill+Manufacturer" : type.ToString()); }
        public static string CardKey(BaseCard card)
        {
            var location = card as LocationCard;
            if (location != null) return "location:" + location.Region;
            var industry = card as IndustryCard;
            if (industry != null) return "industry:" + String.Join("+", industry.BuildingTypes.Select(t => t.ToString()).OrderBy(t => t, StringComparer.Ordinal));
            if (card is WildLocationCard) return "wild:location";
            if (card is WildIndustryCard) return "wild:industry";
            throw new InvalidOperationException("Unknown public card type.");
        }
    }
    public sealed class CardTrackerProjection
    {
        public readonly List<TrackedCard> Cards = FourPlayerCardManifest.Create();
        public readonly Dictionary<PlayerColor, bool[]> Wild = new Dictionary<PlayerColor, bool[]>();
        public bool Complete, Rail, Ended;
        public int Revision;
        public string Availability;
        public int Played { get { return Cards.Sum(c => c.Played); } }
        public int Unseen { get { return 64 - Played; } }
        public TrackedCard Find(string key) { return Cards.First(c => c.Key == key); }
        static string Signature(IEvent change)
        {
            // Compare only the fields this PUBLIC projection is permitted to consume.
            // Never serialize GameStarted's seed or private configuration/card graph.
            var discard = change as CardDiscarded;
            if (discard != null) return "discard:" + discard.Player + ":" + discard.CardToDiscard.UniqueId + ":" + FourPlayerCardManifest.CardKey(discard.CardToDiscard);
            var scout = change as WildCardsTaken;
            if (scout != null) return "scout:" + scout.Player + ":" + String.Join(",", scout.CardsToTrade.Select(c => c.UniqueId + ":" + FourPlayerCardManifest.CardKey(c)));
            var setup = change as GameStarted;
            if (setup != null) return "start:" + (int)setup.GameStartConfiguration.NumberOfPlayers + ":" + setup.GameStartConfiguration.IsIntroductoryGame;
            return change.GetType().FullName;
        }
        public static CardTrackerProjection Build(IEnumerable<IEvent> persisted, IEnumerable<IEvent> pending, int revision, IEnumerable<PlayerColor> players)
        {
            var result = new CardTrackerProjection { Revision = revision, Availability = "Public card history" };
            try {
                foreach (var player in players) result.Wild.Add(player, new bool[2]);
                if (result.Wild.Count != 4) throw new InvalidOperationException("Card Tracker requires four players.");
                var events = new SortedDictionary<int,IEvent>();
                foreach (var change in persisted.Concat(pending)) {
                    if (change == null || change.Metadata == null) throw new InvalidOperationException("Invalid event record.");
                    int at = change.Metadata.Revision;
                    if (at > revision) continue;
                    IEvent previous;
                    if (events.TryGetValue(at, out previous)) {
                        if (Signature(previous) != Signature(change)) throw new InvalidOperationException("Conflicting public history at revision " + at + ".");
                    } else events.Add(at, change);
                }
                if (events.Count != revision || !events.ContainsKey(1) || !(events[1] is GameStarted))
                    throw new InvalidOperationException("History incomplete.");
                var started = (GameStarted)events[1];
                if ((int)started.GameStartConfiguration.NumberOfPlayers != 4) throw new InvalidOperationException("Card Tracker requires four players.");
                var seen = new HashSet<string>(StringComparer.Ordinal);
                int expected = 1;
                foreach (var pair in events) {
                    if (pair.Key != expected++) throw new InvalidOperationException("History incomplete.");
                    var change = pair.Value;
                    if (change is RailEraStarted) {
                        result.Rail = true; seen.Clear();
                        foreach (var card in result.Cards) card.Played = 0;
                    }
                    var discarded = change as CardDiscarded;
                    if (discarded != null) result.Discard(discarded.CardToDiscard, discarded.Player, seen);
                    var scout = change as WildCardsTaken;
                    if (scout != null) {
                        var held = result.Wild[scout.Player];
                        if (held[0] || held[1] || scout.CardsToTrade.Count != 2) throw new InvalidOperationException("Invalid public Scout record.");
                        foreach (var card in scout.CardsToTrade) result.Discard(card, scout.Player, seen);
                        held[0] = held[1] = true;
                    }
                    if (change is GameEnded) result.Ended = true;
                }
                result.Complete = true;
            } catch (Exception error) {
                // Partially folded records must never look like complete counts.
                foreach (var card in result.Cards) card.Played = 0;
                foreach (var held in result.Wild.Values) held[0] = held[1] = false;
                result.Availability = error.Message; result.Complete = false;
            }
            return result;
        }
        void Discard(BaseCard card, PlayerColor player, HashSet<string> seen)
        {
            string key = FourPlayerCardManifest.CardKey(card);
            if (key.StartsWith("wild:", StringComparison.Ordinal)) {
                int type = key == "wild:location" ? 0 : 1;
                if (!Wild[player][type]) throw new InvalidOperationException("Wild acquisition history incomplete.");
                Wild[player][type] = false; return;
            }
            if (String.IsNullOrEmpty(card.UniqueId) || !seen.Add(card.UniqueId))
                throw new InvalidOperationException("Duplicate ordinary card in this era.");
            var tracked = Cards.FirstOrDefault(c => c.Key == key);
            if (tracked == null || tracked.Played >= tracked.Total) throw new InvalidOperationException("Card count exceeds the verified deck.");
            tracked.Played++;
        }
        public static CardTrackerProjection Capture(IGameHistory history)
        {
            var app = history.CurrentGame;
            try {
                var result = Build(history.Repository.Advanced.GameStorage.LoadEvents(app.Metadata.GameId, 0, app.Game.Revision),
                    app.Game.GetUncommittedEvents(), app.Game.Revision, app.Metadata.PlayerMetadata.Select(p => p.PlayerStartConfiguration.Color));
                if(!result.Complete)result.Rail=app.Game.GameProgressInformation.CurrentGameEra.ToString()=="RailEra";
                return result;
            } catch (Exception error) {
                return new CardTrackerProjection { Revision = app.Game.Revision, Availability = "History unavailable: " + error.Message,
                    Rail = app.Game.GameProgressInformation.CurrentGameEra.ToString() == "RailEra" };
            }
        }
        public string Description(TrackedCard card)
        {
            if (!Complete) return card.Name + "\nCounts unknown: " + Availability + "\nNo private cards are inspected.";
            string text = card.Name + (card.City.HasValue ? " — Location cards" : " — Industry cards") +
                "\nPlayed: " + card.Played + " / " + card.Total + "    Unseen: " + card.Unseen;
            if (Rail) text += "\nUnseen cards are in hands or the draw deck.";
            else text += "\nUnseen includes the four hidden Canal setup discards. Playable copies for this category: " +
                Math.Max(0,card.Unseen-4) + "–" + card.Unseen + ".";
            if (card.Key == "industry:CottonMill+Manufacturer") text += "\nCotton Mill and Manufacturer share these same eight dual-purpose cards.";
            if (card.City.HasValue && card.Unseen == 0) text += "\nOrdinary city cards are depleted. Wild Location and legal industry-card builds can still reach this city.";
            return text;
        }
    }
}
