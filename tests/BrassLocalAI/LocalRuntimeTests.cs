using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Newtonsoft.Json.Linq;
using BrassHost;
using BrassLegacyAI;
using BrassPatch.Persistence;
using BrassApplication.AI;
using BrassApplication.AI.AiBehaviors;
using BrassApplication.Game;
using Frameworks.Persistence.GameFactory;
using Frameworks.Persistence.Repository;
using Frameworks.Persistence.RepositoryBackends.Filesystem;

public static class LocalRuntimeTests
{
    static int checks;
    static readonly List<object> budgetAccounting = new List<object>();
    static void Check(bool condition, string message) { checks++; if (!condition) throw new Exception(message); }
    static JObject Configuration(int count, bool intro, int rotate = 0)
    {
        var colors = (count == 2 ? new[] { 3, 1 } : count == 3 ? new[] { 3, 0, 2 } : new[] { 2, 0, 3, 1 });
        return new JObject { ["player_count"] = count, ["introductory"] = intro,
            ["seats"] = new JArray(colors.Select((c, i) => new JObject { ["color"] = (c + rotate) % 4, ["sitting_order"] = i })) };
    }
    static Host NewHost(int count, bool intro, int seed, int rotate = 0)
    { return new Host(seed, new JObject { ["configuration"] = Configuration(count, intro, rotate) }); }
    static IEnumerable<int[]> Permutations(int length, int[] prefix = null)
    {
        prefix = prefix ?? new int[0];
        if (prefix.Length == length) { yield return prefix; yield break; }
        foreach (int color in Enumerable.Range(0, 4).Except(prefix))
            foreach (var result in Permutations(length, prefix.Concat(new[] { color }).ToArray())) yield return result;
    }
    static void PublicInformationAndMappings()
    {
        foreach (int count in new[] { 2, 3, 4 }) foreach (bool intro in new[] { false, true }) foreach (var colors in Permutations(count))
        {
            var config = new JObject { ["player_count"] = count, ["introductory"] = intro,
                ["seats"] = new JArray(colors.Select((color, i) => new JObject { ["color"] = color, ["sitting_order"] = i })) };
            var host = new Host(32027, new JObject { ["configuration"] = config });
            var observation = (JObject)Wire.Token(host.Observe(host.Actor)); var legal = host.Choices().Select(c => c.id).ToArray();
            var sampled = new Host(1, Belief.Setup(observation, Host.Composition(config), 92));
            Check(sampled.Choices().Select(c => c.id).SequenceEqual(legal), "all color/subset permutations reconstruct legal choices");
            var opponent = host.Game.PlayersTrack.Players.First(p => (int)p.Color != host.Actor);
            var card = opponent.Hand[0]; opponent.Hand[0] = host.Game.Board.CardDeck[0]; host.Game.Board.CardDeck[0] = card;
            card = opponent.DiscardPile[0]; opponent.DiscardPile[0] = host.Game.Board.CardDeck[1]; host.Game.Board.CardDeck[1] = card;
            host.Game.Board.CardDeck.Reverse();
            Check(JToken.DeepEquals(observation, Wire.Token(host.Observe(host.Actor))), "opponent hand, hidden initial discard, future deck order stay private");
            Check(host.Choices().Select(c => c.id).SequenceEqual(legal), "hidden-world replacement leaves legal choices unchanged");
        }
    }
    static void Smoke()
    {
        foreach (int count in new[] { 2, 3, 4 }) foreach (bool intro in new[] { false, true }) foreach (int rotate in Enumerable.Range(0, 4))
        {
            var h = NewHost(count, intro, 31004, rotate); var composition = Host.Composition(Configuration(count, intro, rotate));
            var view = (JObject)Wire.Token(h.Observe(h.Actor));
            var sampled = new Host(1, Belief.Setup(view, composition, 99));
            Check(sampled.Choices().Select(c => c.id).SequenceEqual(h.Choices().Select(c => c.id)), "reconstruction legal choices " + count);
            var canonical = Wire.Hash(h.Canonical());
            foreach (string key in ProfileCatalog.Keys)
            {
                var profile = ProfileCatalog.Create(key, count, intro); profile.work_limit = 1;
                var decision = new Planner(composition).Decide(view, h.Choices(), profile, true);
                Check(decision.selected.actions.Count == h.Remaining, "whole initial turn " + key);
                foreach (var action in decision.selected.actions) foreach (var choice in action) sampled.SearchApply(choice);
                Check(sampled.Remaining == 0, "initial turn applied " + key);
                sampled = new Host(1, Belief.Setup(view, composition, 99));
                Check(Wire.Hash(h.Canonical()) == canonical, "search mutated live state");
                Check((string)ProfileCatalog.Snapshot(key, count, intro)["mode"] == "S1", "catalog S1");
            }
        }
    }
    static void Persistence(string directory)
    {
        Directory.CreateDirectory(directory);
        var h = NewHost(3, true, 91233);
        var metadata = new GameMetadata(h.Game.GameStartConfiguration, GameType.Local);
        foreach (var start in metadata.GameStartConfiguration.Players)
        {
            var p = new PlayerMetadata { PlayerStartConfiguration = start, PlayerNetworkId = "Player", NickName = "Local AI test", AiType = start.SittingOrder == 0 ? AIType.Human : AIType.Hard };
            p.AIBehaviour = AiBehaviorFactory.CreateAiBehavior(p); metadata.JoinGame(p);
        }
        var state = LocalGameState.Create(metadata, p => p.PlayerStartConfiguration.SittingOrder == 1 ? "cotton" : "supplier", p => p.PlayerStartConfiguration.SittingOrder == 1 ? "easy" : "mid");
        SaveExtensions.Set(metadata, state);
        var repository = new BaseRepository(new FilesystemGameStorage(directory, new Frameworks.Persistence.Serialization.JsonSerializer()), new FilesystemSnapshotStorage(directory, new Frameworks.Persistence.Serialization.JsonSerializer()), new BrassBirminghamFactory());
        var app = repository.Create(metadata); app.Game.StartGame();
        state["game_id"] = app.Metadata.GameId; state["ready"] = true; SaveExtensions.Set(app.Metadata, state); repository.Save(app);
        var loaded = repository.Load(app.Metadata.GameId);
        Check(JToken.DeepEquals(LocalGameState.Read(loaded.Metadata), LocalGameState.Read(app.Metadata)), "assignment roundtrip");
        Check(LocalGameState.Difficulty(LocalGameState.Read(loaded.Metadata), LocalGameState.Assignment(LocalGameState.Read(loaded.Metadata), (int)loaded.Metadata.PlayerMetadata[1].PlayerStartConfiguration.Color)) == "easy", "per-player difficulty survives real save reload");
        Check(SaveExtensions.Get((GameMetadata)loaded.Metadata.Clone()) != null, "clone retains extension");
        loaded.Metadata.PlayerMetadata[1].LastSeenEventRevision = 1; repository.Save(loaded.Metadata.PlayerMetadata[1]);
        loaded = repository.Load(app.Metadata.GameId);
        Check(JToken.DeepEquals(LocalGameState.Read(loaded.Metadata), LocalGameState.Read(app.Metadata)), "metadata-only save retains extension");
        Check(loaded.Metadata.PlayerMetadata[0].LastSeenEventRevision==0 && loaded.Metadata.PlayerMetadata[1].LastSeenEventRevision==1,"offline shared ID keeps per-seat replay boundary");
        var completed=(GameMetadata)loaded.Metadata.Clone(); completed.GameId="completed";
        completed.GameProgressInformation.IsGameOver=true;
        var completedState=LocalGameState.Read(loaded.Metadata); completedState["game_id"]=completed.GameId; completedState["created_utc"]=DateTime.UtcNow.AddDays(1).ToString("o");
        SaveExtensions.Set(completed,completedState); repository.Advanced.GameStorage.SaveGame(completed.GameId,new List<BoardGameRules.Entities.EventSourcing.Events.IEvent>(),completed);
        var history=new BrassApplication.GameHistory.BrassGameHistory{Repository=repository};
        Check(LocalMenu.FindResume(history).GameId==app.Metadata.GameId,"Continue finds unfinished game behind newer completed game");
        int color = (int)loaded.Metadata.PlayerMetadata[1].PlayerStartConfiguration.Color;
        string profileId = (string)LocalGameState.Assignment(state, color)["profile"]["id"];
        var events = repository.Advanced.GameStorage.LoadEvents(app.Metadata.GameId, 0, app.Game.Revision);
        LocalGameState.CommitIntent(loaded.Metadata, new IntentState { profile = profileId, seat = color, revision = app.Game.Revision }, app.Game.Revision, events);
        repository.Save(loaded); loaded = repository.Load(app.Metadata.GameId);
        Check(LocalGameState.Intent(LocalGameState.Read(loaded.Metadata), color, loaded.Game.Revision, events) != null, "intent saved with events");
        Check(LocalGameState.Intent(LocalGameState.Read(loaded.Metadata), color, 0, events) == null, "future intent excluded");
        Check(LocalGameState.Intent(LocalGameState.Read(loaded.Metadata), color, loaded.Game.Revision, new BoardGameRules.Entities.EventSourcing.Events.IEvent[0]) == null, "other history excluded");
        var corrupt = LocalGameState.Read(loaded.Metadata); corrupt["assignments"][1]["strategy"] = "balanced"; SaveExtensions.Set(loaded.Metadata, corrupt);
        bool rejected = false; try { LocalGameState.Read(loaded.Metadata); } catch (InvalidDataException) { rejected = true; }
        Check(rejected, "corrupt assignments rejected");
        Check(File.Exists(Path.Combine(directory, app.Metadata.GameId, "events.storage")), "save preserved on validation failure");
    }
    static void DifficultyContracts()
    {
        foreach (int count in new[] { 2, 3, 4 }) foreach (bool intro in new[] { false, true })
        foreach (string style in ProfileCatalog.Keys) foreach (string difficulty in DifficultyCatalog.Keys)
        {
            var config = new BoardGameRules.Entities.Game.GameStartConfiguration((BoardGameRules.Entities.Common.NumberOfPlayers)count) { IsIntroductoryGame = intro };
            var metadata = new GameMetadata(config, GameType.Local) { GameId = "difficulty-contract" };
            foreach (var start in config.Players) metadata.JoinGame(new PlayerMetadata { PlayerStartConfiguration = start, AiType = AIType.Hard });
            var state = LocalGameState.Create(metadata, p => style, p => difficulty); state["ready"] = true; SaveExtensions.Set(metadata, state);
            var saved = LocalGameState.Read(metadata); var assignment = LocalGameState.Assignment(saved, (int)config.Players[0].Color);
            Check(LocalGameState.Difficulty(saved, assignment) == difficulty, "all 108 configurations retain the selected difficulty");
            Check((int)assignment["execution"]["search_ms"] == DifficultyCatalog.Milliseconds(difficulty), "saved allowance matches preset");
            Check(JToken.DeepEquals(assignment["profile"], ProfileCatalog.Snapshot(style, count, intro)), "difficulty leaves frozen profile untouched");
            Check((int)assignment["profile"]["work_limit"] == 240 && (int)assignment["profile"]["reply_work"] == 2048, "research node limits unchanged across tiers");
            // Re-sign deliberate invalid data so this exercises semantic validation,
            // not only the generic assignment hash rejection.
            foreach (string fault in new[] { "missing", "unknown", "time", "version", "type", "old-with-execution" }) {
                var broken = (JObject)saved.DeepClone(); var row = (JObject)broken["assignments"][0];
                switch (fault) {
                    case "missing": row.Remove("execution"); break;
                    case "unknown": row["execution"]["difficulty"] = "impossible"; break;
                    case "time": row["execution"]["search_ms"] = 17; break;
                    case "version": row["execution"]["version"] = "future/9"; break;
                    case "type": row["execution"]["search_ms"] = "5000"; break;
                    case "old-with-execution": broken["format"] = LocalGameState.PreviousFormat; break;
                }
                broken["assignment_sha256"] = AssignmentHash(broken); SaveExtensions.Set(metadata, broken);
                bool rejected = false; try { LocalGameState.Read(metadata); } catch (InvalidDataException) { rejected = true; }
                Check(rejected, "invalid execution contract rejected: " + fault);
            }
            var older = (JObject)saved.DeepClone(); older["format"] = LocalGameState.PreviousFormat;
            foreach (JObject row in older["assignments"]) row.Remove("execution");
            older["assignment_sha256"] = AssignmentHash(older); SaveExtensions.Set(metadata, older);
            var read = LocalGameState.Read(metadata);
            Check(LocalGameState.Difficulty(read, LocalGameState.Assignment(read, (int)config.Players[0].Color)) == DifficultyCatalog.Existing, "old format has explicit compatibility setting");
            Check(DifficultyCatalog.Milliseconds(DifficultyCatalog.Existing) == 10000 && JToken.DeepEquals(read, older), "old format reads at 10s without rewriting data");
        }
    }
    static string AssignmentHash(JObject state)
    {
        return Wire.Hash(new JObject { ["format"] = state["format"], ["runtime"] = state["runtime"], ["rules"] = state["rules"],
            ["schema"] = state["schema"], ["configuration"] = state["configuration"], ["assignments"] = state["assignments"] });
    }
    // Ported from the R3 S1 behavioral fixtures, against the deployed runtime.
    static void S1Contracts()
    {
        Action<JObject, string[]> hand = (v, cards) => {
            v["deck_count"] = (int)v["deck_count"] + v["own_hand"].Count() - cards.Length;
            Wire.Player(v, (int)v["seat"])["hand_count"] = cards.Length; v["own_hand"] = new JArray(cards.OrderBy(c => c));
        };
        Action<JObject, string, string, int, bool> put = (v, slot, kind, seat, flipped) => {
            var tiles = (JArray)Wire.Player(v, seat)["mat"].Single(m => (string)m["type"] == kind)["tiles"];
            var tile = (JObject)tiles[0].DeepClone(); tiles.RemoveAt(0); tile["flipped"] = flipped;
            if (kind == "CoalMine") tile["resource"] = 2;
            v["slots"].Single(s => (string)s["id"] == slot)["tile"] = tile;
        };
        foreach (string fixture in new[] { "loan-build", "develop-build" })
        {
            var source = new Host(1301); var view = (JObject)Wire.Token(source.Observe(source.Actor)); int actor = source.Actor;
            view["context"]["round"] = 2; view["context"]["actions_remaining"] = 2;
            var me = Wire.Player(view, actor);
            if (fixture == "loan-build")
            {
                me["cash"] = 0; put(view, "Coalbrookdale:2", "CoalMine", (actor + 1) % 4, false);
                view["links"].Single(l => (string)l["id"] == "CanalCoalbrookdaleShrewsbury")["owner"] = actor;
                view["iron_market"] = 0; hand(view, new[] { "location:Coalbrookdale", "location:Redditch" });
            }
            else
            {
                me["cash"] = 30; put(view, "Worcester:0", "CottonMill", actor, true); me["income_marker"] = 15;
                foreach (var mat in me["mat"].Where(m => (string)m["type"] != "CottonMill")) mat["tiles"] = new JArray();
                hand(view, new[] { "location:Kidderminster", "location:Nuneaton" });
                foreach (string id in new[] { "CanalWorcesterSouthWest", "CanalWorcesterGloucester" }) view["links"].Single(l => (string)l["id"] == id)["owner"] = actor;
                foreach (var merchant in view["merchants"].Where(m => ((string)m["id"]).StartsWith("Gloucester:"))) { merchant["types"] = new JArray("CottonMill"); merchant["beer"] = 1; }
            }
            var composition = Host.Composition(Host.Configuration(source.Game.GameStartConfiguration));
            var h = new Host(1, Belief.Setup(view, composition, 2)); view = (JObject)Wire.Token(h.Observe(actor));
            var before = Wire.Hash(h.Canonical()); var profile = ProfileCatalog.Create("balanced", 4);
            profile.work_limit = 1024; profile.milliseconds = 10000; var planner = new Planner(composition);
            var decision = planner.Decide(view, h.Choices(), profile, true, 1, default(System.Threading.CancellationToken), p => Check(p.actions.Count == 2, "S1 publishes only complete two-action incumbents"));
            Check(decision.mode == "S1" && decision.selected.actions.Count == 2, "deployed S1 compares a complete turn");
            Check(decision.selected.actions[0][0].StartsWith(fixture == "loan-build" ? "loan|" : "develop|") && decision.selected.actions[1][0].StartsWith("build|"), "S1 discovers enabling setup sequence: " + fixture);
            Check(decision.alternatives.Length > 1 && decision.nodes > 10, "S1 widens beyond its first greedy incumbent");
            foreach (var plan in decision.alternatives)
            {
                Check(plan.actions.Count == 2, "S1 alternatives include both ordinary actions");
                var replay = new Host(1, Belief.Setup(view, composition, 17));
                foreach (var action in plan.actions) foreach (string id in action) replay.SearchApply(id);
                Check(replay.Remaining == 0 && replay.Actor == actor, "complete plan stays within current turn and does not replenish cards");
                Check(decision.selected.score >= plan.score - 0.000001, "Balanced chooses by complete-turn score");
            }
            var again = planner.Decide(view, h.Choices(), profile, true);
            Check(again.selected.signature == decision.selected.signature && before == Wire.Hash(h.Canonical()), "S1 is repeatable and does not mutate live state");
            Console.WriteLine("S1 contract passed: " + fixture + ", nodes=" + decision.nodes);
        }
    }
    static void BudgetAccounting()
    {
        // Exercise the deployed bank setting with real deadlines, including the
        // bounded legal completion after the remaining search allowance is zero.
        foreach (string difficulty in DifficultyCatalog.Keys.Concat(new[] { "exhausted" }))
        {
            var source = NewHost(4, false, 1301); var composition = Host.Composition(Configuration(4, false));
            var view = (JObject)Wire.Token(source.Observe(source.Actor));
            view["context"]["round"] = 2; view["context"]["actions_remaining"] = 2;
            var host = new Host(1, Belief.Setup(view, composition, 2)); view = (JObject)Wire.Token(host.Observe(host.Actor));
            var before = Wire.Hash(host.Canonical()); var profile = ProfileCatalog.Create("balanced", 4);
            profile.milliseconds = difficulty == "exhausted" ? 0 : DifficultyCatalog.Milliseconds(difficulty);
            var decision = new Planner(composition).Decide(view, host.Choices(), profile, false);
            Check(decision.mode == "S1" && decision.selected.actions.Count == 2 && decision.nodes > 0, "timed bank search retains a complete incumbent: " + difficulty);
            Check(Wire.Hash(host.Canonical()) == before, "timed bank search leaves the live input unchanged");
            foreach (var action in decision.selected.actions) foreach (var id in action) host.SearchApply(id);
            Check(host.Remaining == 0, "timed bank incumbent is legal through both actions: " + difficulty);
            budgetAccounting.Add(new { difficulty, allowance_ms = profile.milliseconds, work_limit = profile.work_limit, reply_work = profile.reply_work, decision.nodes, decision.failures, decision.elapsed_ms, decision.overrun, decision.status });
        }
    }
    static object Match(int count, bool intro, int seed, string candidate, int candidateSeat, int work)
    {
        var h = NewHost(count, intro, seed, seed % 4);
        var composition = Host.Composition(Configuration(count, intro, seed % 4)); var planner = new Planner(composition);
        var intents = new Dictionary<int, IntentState>(); int turns = 0, actions = 0, debts = 0;
        var watch = System.Diagnostics.Stopwatch.StartNew(); var kinds = new HashSet<string>();
        while (!h.Terminal)
        {
            if (turns++ > 400) throw new Exception("match watchdog");
            int seat = h.Actor;
            var key = candidateSeat < 0 || seat == candidateSeat ? candidate : ProfileCatalog.Keys[(seat + seed) % 6];
            var profile = ProfileCatalog.Create(key, count, intro); profile.work_limit = work;
            var view = (JObject)Wire.Token(h.Observe(seat)); IntentState prior; intents.TryGetValue(seat, out prior);
            var decision = planner.Decide(view, h.Choices(), profile, true, 1, default(System.Threading.CancellationToken), null, prior);
            intents[seat] = decision.selected.intent;
            if (h.Phase == "debt") debts++;
            foreach (var action in decision.selected.actions)
            {
                kinds.Add(action[0].Split('|')[0]);
                foreach (var choice in action) h.SearchApply(choice);
                actions++;
            }
            h.Advance();
            Check(h.Game.PlayersTrack.Players.All(p => p.IncomeMarkerPosition >= 0 && p.IncomeMarkerPosition <= 99), "income bounds");
        }
        var result = (JObject)Wire.Token(h.Result());
        Check(result["seats"].Count() == count && Math.Abs(result["seats"].Sum(s => (double)s["utility"]) - 1) < 1e-9, "terminal ranking");
        return new { count, intro, seed, candidate, candidateSeat, work, actions, turns, debts, kinds = kinds.OrderBy(x => x).ToArray(), elapsed_ms = watch.ElapsedMilliseconds, result };
    }
    public static int Main(string[] args)
    {
        try
        {
            string output = args[0];
            bool gameplayOnly = args.Length == 2 && args[1] == "gameplay-only";
            if (args.Length == 1 || gameplayOnly) { Smoke(); PublicInformationAndMappings(); S1Contracts(); BudgetAccounting(); if (!gameplayOnly) { DifficultyContracts(); Persistence(Path.Combine(Path.GetDirectoryName(output), "save-" + Guid.NewGuid().ToString("N"))); } }
            var matches = new List<object>();
            if (args.Length > 1 && !gameplayOnly)
            {
                int count = int.Parse(args[1]), work = args.Length > 2 ? int.Parse(args[2]) : 48;
                if (args.Length > 3 && args[3] == "identical")
                {
                    foreach (bool intro in new[] { false, true }) foreach (string key in ProfileCatalog.Keys)
                    {
                        matches.Add(Match(count, intro, 41024, key, -1, work));
                        Console.WriteLine("Completed identical table " + matches.Count + ": " + count + "/" + intro + "/" + key);
                        File.WriteAllText(output, Wire.Json(new { passed = false, running = true, checks, matches }));
                    }
                    File.WriteAllText(output, Wire.Json(new { passed = true, checks, matches })); return 0;
                }
                foreach (bool intro in new[] { false, true }) foreach (string key in ProfileCatalog.Keys)
                    foreach (int seed in new[] { 41022, 41023 })
                    {
                        foreach (int seat in Configuration(count, intro, seed % 4)["seats"].Select(x => (int)x["color"]))
                        {
                            matches.Add(Match(count, intro, seed, key, seat, work));
                            Console.WriteLine("Completed " + matches.Count + " games: " + count + "/" + intro + "/" + key + "/" + seed + "/" + seat);
                            File.WriteAllText(output, Wire.Json(new { passed = false, running = true, checks, matches }));
                        }
                    }
            }
            File.WriteAllText(output, Wire.Json(new { passed = true, checks, matches, budgetAccounting })); Console.WriteLine("PASS " + checks); return 0;
        }
        catch (Exception error) { Console.Error.WriteLine(error); if (args.Length > 0) File.WriteAllText(args[0], Wire.Json(new { passed = false, checks, error = error.ToString() })); return 1; }
    }
}
