using System;
using System.IO;
using System.Linq;
using BrassApplication.AI;
using BrassApplication.Game;
using BrassHost;
using BrassLegacyAI.Native;
using BrassPatch.Persistence;
using Newtonsoft.Json.Linq;

namespace BrassLegacyAI
{
    // A new envelope preserves old homogeneous saves without reinterpreting them.
    // Every executable identity and search setting belongs to a physical seat.
    public static class MixedGameState
    {
        public const string Format = "brass-local-mixed-save/1.0.0";
        public const string Runtime = "brass-local-seat-routing/1.0.0";
        public static bool IsMixed(JObject state)
        { return (string)state?["format"] == Format || (string)state?["runtime"] == Runtime; }
        static string AssignmentHash(JObject state)
        {
            return Wire.Hash(new JObject { ["format"] = state["format"], ["runtime"] = state["runtime"],
                ["family"] = state["family"], ["rules"] = state["rules"], ["schema"] = state["schema"],
                ["configuration"] = state["configuration"], ["assignments"] = state["assignments"] });
        }
        public static JObject Create(GameMetadata metadata, Func<PlayerMetadata, AiChoice> choice,
            Func<PlayerMetadata, string> difficulty, Func<PlayerMetadata, int> searchK, Func<PlayerMetadata, double> temperature)
        {
            if (!NativeGameState.Supported(metadata)) throw new InvalidDataException("NN opponents require standard four-player rules.");
            var state = LocalGameState.Create(metadata, p => choice(p).Family == AiFamily.V4 ? choice(p).EntryId : "balanced", difficulty);
            foreach (var player in metadata.PlayerMetadata.Where(p => p.AiType != AIType.Human)) {
                var selected = choice(player);
                var row = LocalGameState.Assignment(state, (int)player.PlayerStartConfiguration.Color);
                if (selected.Family == AiFamily.V4) { row["backend"] = ProfileCatalog.Runtime; continue; }
                if (selected.Family != AiFamily.AlphaTest) throw new InvalidDataException("Unsupported AI family.");
                var model = NativeModel.FromChoice(selected.EntryId);
                row.Remove("profile_sha256"); row.Remove("source_sha256");
                row["kind"] = "native"; row["strategy"] = model.Key; row["backend"] = model.Backend;
                row["profile"] = model.Profile("neutral"); row["nn_focus"] = "neutral";
                row["model_id"] = model.Model; row["export_sha256"] = model.Export;
                row["nn_search_k"] = searchK(player); row["nn_temperature"] = temperature(player);
                NativeGameState.SearchK(row); NativeGameState.Temperature(row);
            }
            state["format"] = Format; state["runtime"] = Runtime; state["family"] = "per-seat";
            state["assignment_sha256"] = AssignmentHash(state);
            return state;
        }
        public static JObject Read(GameMetadata metadata, bool pending)
        {
            var state = SaveExtensions.Get(metadata);
            if (!NativeGameState.Supported(metadata) || (string)state["format"] != Format || (string)state["runtime"] != Runtime ||
                (string)state["family"] != "per-seat" || (string)state["rules"] != Wire.Rules || (string)state["schema"] != Wire.Version ||
                (string)state["game_id"] != metadata.GameId || (!pending && (bool?)state["ready"] != true) ||
                !JToken.DeepEquals(state["configuration"], Host.Configuration(metadata.GameStartConfiguration)) ||
                (string)state["assignment_sha256"] != AssignmentHash(state) || !(state["records"] is JArray) ||
                (string)state["records_sha256"] != Wire.Hash(state["records"])) throw new InvalidDataException("Incompatible per-seat AI save; save preserved.");
            var rows = state["assignments"] as JArray;
            if (rows == null || rows.Count != metadata.PlayerMetadata.Count || rows.Count != 4 ||
                rows.Select(r => (int)r["color"]).Distinct().Count() != rows.Count) throw new InvalidDataException("Invalid saved seat mapping.");
            foreach (var player in metadata.PlayerMetadata) {
                var row = LocalGameState.Assignment(state, (int)player.PlayerStartConfiguration.Color);
                string kind = (string)row["kind"];
                if ((int)row["sitting_order"] != player.PlayerStartConfiguration.SittingOrder ||
                    (player.AiType == AIType.Human) != (kind == "human")) throw new InvalidDataException("Saved controller identity changed.");
                if (kind == "human") {
                    if (row.Properties().Any(p => p.Name != "color" && p.Name != "sitting_order" && p.Name != "kind"))
                        throw new InvalidDataException("Unexpected human AI settings.");
                    continue;
                }
                DifficultyCatalog.Read(row["execution"]);
                if (kind == "native") {
                    var model = NativeModel.FromBackend((string)row["backend"]);
                    string focus = NativeGameState.Focus(state, (int)row["color"]);
                    if ((string)row["strategy"] != model.Key || (string)row["model_id"] != model.Model ||
                        (string)row["export_sha256"] != model.Export || (string)row["profile"] != model.Profile(focus) ||
                        row["nn_search_k"] == null || row["nn_temperature"] == null) throw new InvalidDataException("Saved NN identity changed.");
                    NativeGameState.SearchK(row); NativeGameState.Temperature(row);
                } else if (kind == "legacy") {
                    string strategy = (string)row["strategy"];
                    var expected = ProfileCatalog.Snapshot(strategy, 4, false);
                    if ((string)row["backend"] != ProfileCatalog.Runtime || !JToken.DeepEquals(row["profile"], expected) ||
                        (string)row["profile_sha256"] != Wire.Hash(expected) || (string)row["source_sha256"] != ProfileCatalog.SourceHash(strategy))
                        throw new InvalidDataException("Saved V4 profile changed.");
                } else throw new InvalidDataException("Unknown saved AI controller.");
            }
            return state;
        }
    }
}
