using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Newtonsoft.Json.Linq;
using BrassHost;
using BrassPatch.Persistence;
using BrassApplication.AI;
using BrassApplication.Game;
using BoardGameRules.Entities.EventSourcing.Events;

namespace BrassLegacyAI
{
    public static class LocalGameState
    {
        public const string Format = "brass-local-ai-save/1.0.0";
        static string AssignmentHash(JObject state)
        {
            return Wire.Hash(new JObject { ["format"] = state["format"], ["runtime"] = state["runtime"],
                ["rules"] = state["rules"], ["schema"] = state["schema"], ["configuration"] = state["configuration"], ["assignments"] = state["assignments"] });
        }
        public static JObject Create(GameMetadata metadata, Func<PlayerMetadata, string> key)
        {
            var config = metadata.GameStartConfiguration;
            var assignments = new JArray(metadata.PlayerMetadata.Select(p => {
                var row = new JObject { ["color"] = (int)p.PlayerStartConfiguration.Color, ["sitting_order"] = p.PlayerStartConfiguration.SittingOrder,
                    ["kind"] = p.AiType == AIType.Human ? "human" : "legacy" };
                if (p.AiType != AIType.Human) {
                    string strategy = key(p); var profile = ProfileCatalog.Snapshot(strategy, (int)config.NumberOfPlayers, config.IsIntroductoryGame);
                    row["strategy"] = strategy; row["profile"] = profile; row["profile_sha256"] = Wire.Hash(profile); row["source_sha256"] = ProfileCatalog.SourceHash(strategy);
                }
                return row;
            }));
            var state = new JObject { ["format"] = Format, ["runtime"] = ProfileCatalog.Runtime, ["rules"] = Wire.Rules, ["schema"] = Wire.Version,
                ["game_id"] = metadata.GameId, ["created_utc"] = DateTime.UtcNow.ToString("o"), ["ready"] = false,
                ["configuration"] = Host.Configuration(config), ["assignments"] = assignments, ["records"] = new JArray() };
            state["assignment_sha256"] = AssignmentHash(state); state["records_sha256"] = Wire.Hash(state["records"]);
            return state;
        }
        public static JObject Read(GameMetadata metadata, bool allowPending = false)
        {
            var state = SaveExtensions.Get(metadata);
            if (state == null) return null;
            string runtime=(string)state["runtime"];
            if ((string)state["format"] != Format || (runtime != ProfileCatalog.Runtime && runtime != ProfileCatalog.PreviousRuntime && runtime != ProfileCatalog.CashRuntime) || (string)state["rules"] != Wire.Rules ||
                (string)state["schema"] != Wire.Version || (string)state["game_id"] != metadata.GameId ||
                (!allowPending && (bool?)state["ready"] != true) || !JToken.DeepEquals(state["configuration"], Host.Configuration(metadata.GameStartConfiguration)) ||
                (string)state["assignment_sha256"] != AssignmentHash(state) || !(state["records"] is JArray) || (string)state["records_sha256"] != Wire.Hash(state["records"]))
                throw new InvalidDataException("This game's AI configuration is incomplete or incompatible. Its save has been preserved.");
            var rows = state["assignments"].ToArray();
            if (rows.Length != metadata.PlayerMetadata.Count || rows.Select(r => (int)r["color"]).Distinct().Count() != rows.Length)
                throw new InvalidDataException("Saved AI player mapping is invalid.");
            foreach (var p in metadata.PlayerMetadata)
            {
                var row = rows.Single(r => (int)r["color"] == (int)p.PlayerStartConfiguration.Color);
                bool human = p.AiType == AIType.Human;
                if ((int)row["sitting_order"] != p.PlayerStartConfiguration.SittingOrder || (string)row["kind"] != (human ? "human" : "legacy"))
                    throw new InvalidDataException("Saved AI player identity changed.");
                if (human) continue;
                string strategy = (string)row["strategy"];
                var expected = ProfileCatalog.Snapshot(strategy, (int)metadata.GameStartConfiguration.NumberOfPlayers, metadata.GameStartConfiguration.IsIntroductoryGame,runtime);
                if (!JToken.DeepEquals(row["profile"], expected) || (string)row["profile_sha256"] != Wire.Hash(expected) || (string)row["source_sha256"] != ProfileCatalog.SourceHash(strategy,runtime))
                    throw new InvalidDataException("Saved AI profile differs from this package. Restore the matching patch to resume.");
            }
            return state;
        }
        public static JObject Assignment(JObject state, int color) { return (JObject)state["assignments"].Single(p => (int)p["color"] == color); }
        public static string HistoryHash(IEnumerable<IEvent> events, int revision)
        {
            return HistoryHashes(events.Where(e => e.Metadata.Revision <= revision)).OrderBy(x => x.Key).Last().Value;
        }
        static Dictionary<int, string> HistoryHashes(IEnumerable<IEvent> events)
        {
            string hash = Wire.Hash(new JArray());
            var result = new Dictionary<int, string> { [0] = hash };
            foreach (var e in events.OrderBy(e => e.Metadata.Revision))
            {
                hash = Wire.Hash(new JObject { ["previous"] = hash, ["type"] = e.GetType().FullName, ["event"] = JToken.FromObject(e) });
                result[e.Metadata.Revision] = hash;
            }
            return result;
        }
        public static List<IEvent> History(BrassApplicationGame app, IEnumerable<IEvent> persisted)
        {
            return persisted.Concat(app.Game.GetUncommittedEvents()).Where(e => e.Metadata.Revision <= app.Game.Revision)
                .GroupBy(e => e.Metadata.Revision).Select(g => g.Last()).OrderBy(e => e.Metadata.Revision).ToList();
        }
        public static IntentState Intent(JObject state, int color, int revision, IEnumerable<IEvent> events)
        {
            var row = Assignment(state, color);
            var hashes = HistoryHashes(events);
            foreach (var record in state["records"].Where(r => (int)r["color"] == color && (int)r["revision"] <= revision).OrderByDescending(r => (int)r["revision"]))
            {
                string hash;
                if (!hashes.TryGetValue((int)record["revision"], out hash) || (string)record["history_sha256"] != hash) continue;
                var intent = record["intent"].ToObject<IntentState>();
                if (intent.seat != color || intent.profile != (string)row["profile"]["id"] || intent.revision > (int)record["revision"])
                    throw new InvalidDataException("Saved AI intent belongs to a different player or profile.");
                return intent;
            }
            return null;
        }
        public static void CommitIntent(GameMetadata metadata, IntentState intent, int revision, IList<IEvent> events)
        {
            if (intent == null) return;
            var state = Read(metadata);
            if ((string)Assignment(state, intent.seat)["profile"]["id"] != intent.profile || intent.revision > revision)
                throw new InvalidDataException("AI intent does not match the committed action.");
            var hashes = HistoryHashes(events);
            var records = new JArray(state["records"].Where(r => (int)r["revision"] <= revision && hashes.ContainsKey((int)r["revision"]) &&
                (string)r["history_sha256"] == hashes[(int)r["revision"]] &&
                !((int)r["revision"] == revision && (int)r["color"] == intent.seat)).Select(r => r.DeepClone()));
            records.Add(new JObject { ["color"] = intent.seat, ["revision"] = revision,
                ["history_sha256"] = hashes[revision], ["intent"] = JObject.FromObject(intent) });
            state["records"] = records; state["records_sha256"] = Wire.Hash(records);
            // This in-memory update is saved with the engine events by the existing recoverable save.
            SaveExtensions.Set(metadata, state);
        }
    }
}
