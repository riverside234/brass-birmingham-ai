using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Security.Cryptography;
using System.Text;
using BoardGameRules.Entities.Players;
using BrassPatch.Statistics;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using Newtonsoft.Json.Serialization;

namespace BrassPatch.History
{
    // The collection contract is explicit; the scalar contract is limited to
    // the existing value-only Statistics DTOs and excludes computed getters.
    public static class GameHistoryStatisticsCodec
    {
        public const int Schema = 1;
        public const string Collector = "statistics/1.4.21";
        public const int MaxReportBytes = 8 * 1024 * 1024;
        static readonly Type[] scalarTypes = { typeof(StatisticsPlayer), typeof(StatisticsAsset), typeof(StatisticsLink),
            typeof(StatisticsAction), typeof(StatisticsEntry), typeof(StatisticsSample), typeof(StatisticsCity),
            typeof(StatisticsConnection), typeof(StatisticsBoundary), typeof(StatisticsMatTile), typeof(StatisticsStock),
            typeof(StatisticsLinkContribution), typeof(StatisticsLocation) };
        sealed class Scalars : DefaultContractResolver
        {
            protected override IList<JsonProperty> CreateProperties(Type type, MemberSerialization mode)
            {
                if (!scalarTypes.Contains(type)) throw new InvalidDataException("Unexpected report object: " + type.FullName);
                return type.GetProperties(BindingFlags.Public | BindingFlags.Instance).Where(p => p.GetSetMethod(true) != null)
                    .OrderBy(p => p.Name, StringComparer.Ordinal).Select(p => {
                        var property = base.CreateProperty(p, mode); property.Writable = property.Readable = true;
                        property.Required = Required.AllowNull; return property;
                    }).ToList();
            }
        }
        static readonly Scalars scalarResolver = new Scalars();
        static JsonSerializer Serializer()
        { return JsonSerializer.Create(new JsonSerializerSettings { ContractResolver = scalarResolver, TypeNameHandling = TypeNameHandling.None,
            MissingMemberHandling = MissingMemberHandling.Error, DateParseHandling = DateParseHandling.None, Culture = CultureInfo.InvariantCulture }); }
        static JArray Values<T>(IEnumerable<T> values) { return JArray.FromObject(values, Serializer()); }
        static List<T> Read<T>(JToken token) { var serializer=Serializer(); return Array(token).Select(v => v.ToObject<T>(serializer)).ToList(); }
        static JArray Array(JToken token) { var result = token as JArray; if (result == null || result.Count > 50000) throw new InvalidDataException("Missing or oversized report collection."); return result; }
        static void Fields(JObject value, params string[] names)
        { if (value == null || value.Properties().Count() != names.Length || names.Any(n => value.Property(n) == null)) throw new InvalidDataException("Report fields do not match the supported schema."); }
        public static string Digest(string value)
        { using (var sha = SHA256.Create()) return BitConverter.ToString(sha.ComputeHash(Encoding.UTF8.GetBytes(value))).Replace("-", "").ToLowerInvariant(); }
        public static string Key(string gameId)
        { if (String.IsNullOrWhiteSpace(gameId) || gameId.Length > 1024) throw new InvalidDataException("Invalid game identity."); return Digest(gameId); }
        public static JObject Encode(CompletedMatchStatistics report)
        {
            Validate(report);
            var actions = new JArray();
            foreach (var action in report.actions) {
                var item = JObject.FromObject(action, Serializer()); item["notes"] = new JArray(action.notes); item["cards"] = new JArray(action.cards); actions.Add(item);
            }
            var rounds = new JArray(report.rounds.Select(r => new JObject { ["Era"] = r.Era, ["Round"] = r.Round,
                ["order"] = new JArray(r.order.Select(c => (int)c)),
                ["spending"] = new JArray(r.spending.OrderBy(p => (int)p.Key).Select(p => new JObject { ["player"] = (int)p.Key, ["cash"] = p.Value })) }));
            var value = new JObject { ["GameId"] = report.GameId, ["FinalRevision"] = report.FinalRevision,
                ["Introductory"] = report.Introductory, ["CompleteHistory"] = report.CompleteHistory,
                ["Availability"] = report.Availability, ["CollectionMilliseconds"] = report.CollectionMilliseconds,
                ["players"] = Values(report.players), ["assets"] = Values(report.assets), ["links"] = Values(report.links),
                ["actions"] = actions, ["entries"] = Values(report.entries), ["samples"] = Values(report.samples), ["rounds"] = rounds,
                ["cities"] = Values(report.cities), ["connections"] = Values(report.connections), ["boundaries"] = Values(report.boundaries),
                ["matTiles"] = Values(report.matTiles), ["contributions"] = Values(report.contributions), ["locations"] = Values(report.locations),
                ["stocks"] = Values(report.stocks), ["warnings"] = new JArray(report.warnings) };
            if (Encoding.UTF8.GetByteCount(value.ToString(Formatting.None)) > MaxReportBytes) throw new InvalidDataException("Statistics report exceeds its storage limit.");
            return value;
        }
        public static CompletedMatchStatistics Decode(JObject value)
        {
            Fields(value, "GameId", "FinalRevision", "Introductory", "CompleteHistory", "Availability", "CollectionMilliseconds",
                "players", "assets", "links", "actions", "entries", "samples", "rounds", "cities", "connections", "boundaries",
                "matTiles", "contributions", "locations", "stocks", "warnings");
            var report = new CompletedMatchStatistics { GameId = (string)value["GameId"], FinalRevision = (int)value["FinalRevision"],
                Introductory = (bool)value["Introductory"], CompleteHistory = (bool)value["CompleteHistory"],
                Availability = (string)value["Availability"], CollectionMilliseconds = (double)value["CollectionMilliseconds"] };
            report.players.AddRange(Read<StatisticsPlayer>(value["players"])); report.assets.AddRange(Read<StatisticsAsset>(value["assets"]));
            report.links.AddRange(Read<StatisticsLink>(value["links"]));
            var actionSerializer=Serializer();
            foreach (var token in Array(value["actions"])) {
                var item = (JObject)token.DeepClone(); var notes = Array(item["notes"]); var cards = Array(item["cards"]);
                item.Remove("notes"); item.Remove("cards"); var action = item.ToObject<StatisticsAction>(actionSerializer);
                action.notes.AddRange(notes.Select(n => (string)n)); action.cards.AddRange(cards.Select(c => (string)c)); report.actions.Add(action);
            }
            report.entries.AddRange(Read<StatisticsEntry>(value["entries"])); report.samples.AddRange(Read<StatisticsSample>(value["samples"]));
            foreach (var token in Array(value["rounds"])) {
                var item = token as JObject; Fields(item, "Era", "Round", "order", "spending");
                var round = new StatisticsRound { Era = (int)item["Era"], Round = (int)item["Round"] };
                round.order.AddRange(Array(item["order"]).Select(c => (PlayerColor)(int)c));
                foreach (var amount in Array(item["spending"])) { Fields(amount as JObject, "player", "cash"); round.spending.Add((PlayerColor)(int)amount["player"], (int)amount["cash"]); }
                report.rounds.Add(round);
            }
            report.cities.AddRange(Read<StatisticsCity>(value["cities"])); report.connections.AddRange(Read<StatisticsConnection>(value["connections"]));
            report.boundaries.AddRange(Read<StatisticsBoundary>(value["boundaries"])); report.matTiles.AddRange(Read<StatisticsMatTile>(value["matTiles"]));
            report.contributions.AddRange(Read<StatisticsLinkContribution>(value["contributions"])); report.locations.AddRange(Read<StatisticsLocation>(value["locations"]));
            report.stocks.AddRange(Read<StatisticsStock>(value["stocks"])); report.warnings.AddRange(Array(value["warnings"]).Select(w => (string)w));
            Validate(report); return report;
        }
        public static JObject Envelope(CompletedMatchStatistics report, long sequence, DateTime utc, string producer)
        {
            var payload = Encode(report);
            return new JObject { ["schema"] = Schema, ["collector"] = Collector, ["producer"] = producer,
                ["key"] = Key(report.GameId), ["sequence"] = sequence, ["recordedUtc"] = utc.ToUniversalTime().ToString("o", CultureInfo.InvariantCulture),
                ["payload"] = payload, ["sha256"] = Digest(payload.ToString(Formatting.None)) };
        }
        public static GameHistoryRecord Record(JObject envelope)
        {
            Header(envelope); var report = Decode((JObject)envelope["payload"]);
            return new GameHistoryRecord { Key = (string)envelope["key"], Sequence = (long)envelope["sequence"], RecordedUtc = Date(envelope),
                Producer = (string)envelope["producer"], Report = report };
        }
        public static GameHistorySummary Summary(JObject envelope)
        {
            var result = new GameHistorySummary { Key = (string)envelope["key"], Sequence = (long)envelope["sequence"], Players = new List<StatisticsPlayer>().AsReadOnly() };
            try {
                result.RecordedUtc=Date(envelope);Header(envelope);var payload = (JObject)envelope["payload"];
                var players = Read<StatisticsPlayer>(payload["players"]); ValidatePlayers(players);
                result.Introductory = (bool)payload["Introductory"]; result.CompleteHistory = (bool)payload["CompleteHistory"]; result.Players = players.AsReadOnly();
            } catch (Exception) { result.Unavailable = "This statistics report is unreadable or uses an unsupported format."; }
            return result;
        }
        static DateTime Date(JObject envelope)
        {
            DateTime utc;
            if (!DateTime.TryParseExact((string)envelope["recordedUtc"], "o", CultureInfo.InvariantCulture, DateTimeStyles.RoundtripKind, out utc) || utc.Kind != DateTimeKind.Utc)
                throw new InvalidDataException("Invalid recorded date.");
            return utc;
        }
        public static void Header(JObject envelope)
        {
            Fields(envelope, "schema", "collector", "producer", "key", "sequence", "recordedUtc", "payload", "sha256");
            if ((int)envelope["schema"] != Schema || (string)envelope["collector"] != Collector) throw new InvalidDataException("Unsupported statistics report format.");
            var payload = envelope["payload"] as JObject;
            if (payload == null || (long)envelope["sequence"] <= 0 || (string)envelope["key"] != Key((string)payload["GameId"]) ||
                (string)envelope["sha256"] != Digest(payload.ToString(Formatting.None))) throw new InvalidDataException("Statistics report identity or integrity differs.");
            Date(envelope);
        }
        static void ValidatePlayers(IList<StatisticsPlayer> players)
        {
            if (players.Count < 2 || players.Count > 4 || players.Any(p => p == null || !Enum.IsDefined(typeof(PlayerColor), p.Color) || p.Rank < 0 || p.Rank > players.Count) || players.Select(p => p.Color).Distinct().Count() != players.Count)
                throw new InvalidDataException("Invalid statistics player lineup.");
        }
        static HashSet<string> Keys<T>(IEnumerable<T> values, Func<T, string> key)
        { var list = values.Select(key).ToArray(); if (list.Any(String.IsNullOrWhiteSpace) || list.Distinct(StringComparer.Ordinal).Count() != list.Length) throw new InvalidDataException("Missing or duplicate statistics evidence identity."); return new HashSet<string>(list, StringComparer.Ordinal); }
        public static void Validate(CompletedMatchStatistics report)
        {
            if (report == null || report.FinalRevision <= 0 || Double.IsNaN(report.CollectionMilliseconds) || Double.IsInfinity(report.CollectionMilliseconds)) throw new InvalidDataException("Statistics lack a terminal revision.");
            Key(report.GameId); ValidatePlayers(report.players); var colors = new HashSet<PlayerColor>(report.players.Select(p => p.Color));
            var actions = Keys(report.actions, a => a.Key); var assets = Keys(report.assets, a => a.Key); var links = Keys(report.links, l => l.Key);
            IEnumerable<StatisticsPosition> positions = report.actions.Cast<StatisticsPosition>().Concat(report.assets).Concat(report.links).Concat(report.entries)
                .Concat(report.samples).Concat(report.connections).Concat(report.boundaries).Concat(report.matTiles).Concat(report.stocks);
            if (positions.Any(p => p == null || !colors.Contains(p.Player) || p.Revision < 0 || p.Revision > report.FinalRevision || p.Era < 0 || p.Era > 2)) throw new InvalidDataException("Invalid statistics position.");
            if (report.entries.Any(e => !String.IsNullOrEmpty(e.ActionKey) && !actions.Contains(e.ActionKey)) || report.connections.Any(c => !String.IsNullOrEmpty(c.ActionKey) && !actions.Contains(c.ActionKey)) ||
                report.contributions.Any(c => !links.Contains(c.LinkKey) || !colors.Contains(c.Owner) || c.Supplier.HasValue && !colors.Contains(c.Supplier.Value)) ||
                report.stocks.Any(s => !assets.Contains(s.AssetKey)) || report.cities.Any(c => !colors.Contains(c.Player)) ||
                report.entries.Any(e=>e.Supplier.HasValue&&!colors.Contains(e.Supplier.Value)) || report.assets.Any(a=>a.FlipActor.HasValue&&!colors.Contains(a.FlipActor.Value)) ||
                report.rounds.Any(r => r.order.Any(c => !colors.Contains(c)) || r.order.Distinct().Count() != r.order.Count || r.spending.Keys.Any(c => !colors.Contains(c))))
                throw new InvalidDataException("Statistics evidence references an unknown record or player.");
        }
    }
}
