using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using BoardGameRules.Entities.Common;
using BoardGameRules.Entities.Game;
using BrassApplication.AI;
using BrassApplication.AI.AiBehaviors;
using BrassApplication.AI.RuleBasedSystem;
using BrassApplication.AI.RuleBasedSystem.RuleSelectors;
using BrassApplication.Game;
using BrassHost;
using BrassLegacyAI;
using BrassPatch.Persistence;
using Frameworks.Persistence.GameFactory;
using Frameworks.Persistence.Repository;
using Frameworks.Persistence.RepositoryBackends.Filesystem;
using Newtonsoft.Json.Linq;

public static class LocalLiveTests
{
    static int checks;
    static void Check(bool value, string message) { checks++; if (!value) throw new Exception(message); }
    static void Coordinator()
    {
        var h = new Host(327); var view = (JObject)Wire.Token(h.Observe(h.Actor)); var composition = Host.Composition(Host.Configuration(h.Game.GameStartConfiguration));
        var profile = ProfileCatalog.Create("balanced", 4, false); profile.milliseconds = 0; profile.work_limit = 1;
        var ticket = Ticket.Create(h.GameId, h.Revision, h.Actor, h.Phase, profile); ticket.generation = 18; ticket.profile_fingerprint = "fixture";
        string before = Wire.Hash(h.Canonical());
        foreach (string field in new[] { "game_id", "revision", "actor", "request_id", "profile", "phase", "generation", "profile_fingerprint", "schema", "rules" })
        {
            using (var coordinator = new TurnCoordinator(ticket, view, h.Choices(), profile, composition))
            {
                Check(coordinator.Worker.Wait(30000), "zero-budget complete incumbent terminates");
                var stale = JObject.FromObject(ticket); stale[field] = stale[field].Type == JTokenType.Integer ? (JToken)((long)stale[field] + 1) : (JToken)"stale";
                Check(coordinator.Take(stale.ToObject<Ticket>(), true) == null && coordinator.Take(ticket, true) == null, "stale " + field + " consumes request without applying");
            }
        }
        using (var coordinator = new TurnCoordinator(ticket, view, h.Choices(), profile, composition))
        { coordinator.Dispose(); Check(coordinator.Worker.Wait(30000) && coordinator.Take(ticket, true) == null, "disposed worker cannot return late plan"); }
        using (var coordinator = new TurnCoordinator(ticket, view, new List<Choice>(), profile, composition))
        { Check(coordinator.Worker.Wait(30000) && coordinator.Failure != null && coordinator.Take(ticket, true) == null, "worker failure is explicit"); }
        Check(Wire.Hash(h.Canonical()) == before, "asynchronous requests leave real state and RNG unchanged");
    }
    static object Match(string directory, int count, bool intro)
    {
        var config = new GameStartConfiguration((NumberOfPlayers)count) { Seed = 32700 + count, IsIntroductoryGame = intro };
        var metadata = new GameMetadata(config, GameType.Local);
        foreach (var start in config.Players)
        {
            var p = new PlayerMetadata { PlayerStartConfiguration = start, PlayerNetworkId = "live-" + start.Color, NickName = "Scripted test", AiType = start.SittingOrder == 0 ? AIType.Human : AIType.Hard };
            p.AIBehaviour = AiBehaviorFactory.CreateAiBehavior(p); metadata.JoinGame(p);
        }
        var state = LocalGameState.Create(metadata, p => ProfileCatalog.Keys[(p.PlayerStartConfiguration.SittingOrder + count + (intro ? 2 : 0)) % 6], p => DifficultyCatalog.Keys[(p.PlayerStartConfiguration.SittingOrder - 1 + 3) % 3]); SaveExtensions.Set(metadata, state);
        var serializer = new Frameworks.Persistence.Serialization.JsonSerializer();
        var repository = new BaseRepository(new FilesystemGameStorage(directory, serializer), new FilesystemSnapshotStorage(directory, serializer), new BrassBirminghamFactory());
        var app = repository.Create(metadata); app.Game.StartGame(); state["game_id"] = app.Metadata.GameId; state["ready"] = true; SaveExtensions.Set(app.Metadata, state); repository.Save(app);
        var composition = Host.Composition(Host.Configuration(config)); var planner = new Planner(composition); var scripted = new HardAiBehavior("scripted-human");
        int turns = 0, actions = 0, reloads = 0, debts = 0, railTurns = 0; var watch = System.Diagnostics.Stopwatch.StartNew();
        while (!app.Game.GameProgressInformation.IsGameOver)
        {
            Check(turns++ < 100, "live progress watchdog");
            if (app.Game.GameProgressInformation.CurrentGameEra.ToString() == "RailEra") railTurns++;
            int seat = (int)app.Game.GameProgressInformation.CurrentPlayer.Color;
            while (app.Game.TurnFlow.GetRemainingActionsCount() > 0)
            {
                if (app.Metadata.GetPlayerMetadataByColor((BoardGameRules.Entities.Players.PlayerColor)seat).AiType == AIType.Human)
                { scripted.GetNextAIMove(app)(); app.Game.TurnFlow.ConfirmCurrentAction(); actions++; continue; }
                var events = LocalGameState.History(app, repository.Advanced.GameStorage.LoadEvents(app.Metadata.GameId, 0, app.Game.Revision));
                var live = Host.Capture(app, events); state = LocalGameState.Read(app.Metadata); var row = LocalGameState.Assignment(state, seat);
                var profile = Profile.Read((JObject)row["profile"]); profile.work_limit = 1;
                var decision = planner.Decide((JObject)Wire.Token(live.Observe(seat)), live.Choices(), profile, true, 1, default(System.Threading.CancellationToken), null, LocalGameState.Intent(state, seat, app.Game.Revision, events));
                foreach (var path in decision.selected.actions)
                {
                    events = LocalGameState.History(app, repository.Advanced.GameStorage.LoadEvents(app.Metadata.GameId, 0, app.Game.Revision)); live = Host.Capture(app, events);
                    string before = Wire.Hash(live.Canonical()); live.ValidateLiveAction(path, composition); Check(Wire.Hash(live.Canonical()) == before, "complete action validation is isolated");
                    int remaining = app.Game.TurnFlow.GetRemainingActionsCount(); var command = new LegacyBehavior(app, () => live.LiveCommit(path, composition)).GetNextAIMove(app); command();
                    Check(app.Game.TurnFlow.GetRemainingActionsCount() == remaining - 1, "live action confirms once");
                    bool rejected = false; try { command(); } catch (InvalidOperationException) { rejected = true; } Check(rejected, "reused prepared action rejected");
                    LocalGameState.CommitIntent(app.Metadata, decision.selected.intent, app.Game.Revision, LocalGameState.History(app, events)); actions++;
                }
            }
            app.Game.GameFlow.EndTurn();
            if (app.Game.GameFlow.IsLastTurnOver())
            {
                if (app.Game.GameFlow.IsSellIndustryNecessary()) { debts++; var selector = RuleSelectorFactory.CreatePrioritySelector(RuleSetFactory.CreateSellIndustryRuleSet()); selector.Select(app).MakeAction(app).Invoke(); }
                app.Game.GameFlow.EndRound();
            }
            if (!app.Game.GameProgressInformation.IsGameOver)
            {
                var events = LocalGameState.History(app, repository.Advanced.GameStorage.LoadEvents(app.Metadata.GameId, 0, app.Game.Revision));
                var before = Wire.Token(Host.Capture(app, events).Observe((int)app.Game.GameProgressInformation.CurrentPlayer.Color)); state = LocalGameState.Read(app.Metadata);
                repository.Save(app); app = repository.Load(app.Metadata.GameId);
                var after = Wire.Token(Host.Capture(app, repository.Advanced.GameStorage.LoadEvents(app.Metadata.GameId, 0, app.Game.Revision)).Observe((int)app.Game.GameProgressInformation.CurrentPlayer.Color));
                Check(JToken.DeepEquals(before, after) && JToken.DeepEquals(state, LocalGameState.Read(app.Metadata)), "every turn resumes observation and per-seat intent unchanged"); reloads++;
            }
        }
        Check(intro ? railTurns == 0 : railTurns > 0, "correct introductory/full era boundary");
        return new { count, intro, turns, actions, reloads, debts, railTurns, elapsed_ms = watch.ElapsedMilliseconds, human = "Scripted original Hard behavior, not a physical mouse playthrough" };
    }
    public static int Main(string[] args)
    {
        try
        {
            Coordinator(); var matches = new List<object>();
            foreach (int count in new[] { 2, 3, 4 }) foreach (bool intro in new[] { false, true })
            {
                matches.Add(Match(Path.Combine(Path.GetDirectoryName(args[0]), "mixed-" + Guid.NewGuid().ToString("N").Substring(0, 8)), count, intro));
                File.WriteAllText(args[0], Wire.Json(new { passed = false, running = true, checks, matches })); Console.WriteLine("Completed live mixed game " + count + "/" + intro);
            }
            File.WriteAllText(args[0], Wire.Json(new { passed = true, checks, matches })); return 0;
        }
        catch (Exception error) { File.WriteAllText(args[0], Wire.Json(new { passed = false, checks, error = error.ToString() })); Console.Error.WriteLine(error); return 1; }
    }
}
