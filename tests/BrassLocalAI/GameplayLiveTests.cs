using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using BoardGameRules.Entities.Common;
using BoardGameRules.Entities.Game;
using BoardGameRules.Entities.EventSourcing.Events;
using BrassApplication.AI;
using BrassApplication.AI.AiBehaviors;
using BrassApplication.AI.RuleBasedSystem;
using BrassApplication.AI.RuleBasedSystem.RuleSelectors;
using BrassApplication.Game;
using BrassHost;
using BrassLegacyAI;
using BrassPatch.Persistence;
using Newtonsoft.Json.Linq;

// Runs the real game and AI bridge command boundary entirely in memory.
// Report JSON is test output only: no repository, game-save serializer or reload is used.
public static class GameplayLiveTests
{
    static int checks;
    static readonly HashSet<string> testedStrategies = new HashSet<string>();
    static void Check(bool value, string message) { checks++; if (!value) throw new Exception(message); }
    static List<IEvent> Events(BrassApplicationGame app)
    { return LocalGameState.History(app, Enumerable.Empty<IEvent>()); }

    static void Coordinator()
    {
        var h = new Host(327); var view = (JObject)Wire.Token(h.Observe(h.Actor));
        var composition = Host.Composition(Host.Configuration(h.Game.GameStartConfiguration));
        var profile = ProfileCatalog.Create("balanced", 4); profile.milliseconds = 0; profile.work_limit = 1;
        var ticket = Ticket.Create(h.GameId, h.Revision, h.Actor, h.Phase, profile);
        ticket.generation = 18; ticket.profile_fingerprint = "gameplay-fixture";
        string before = Wire.Hash(h.Canonical());
        foreach (string field in new[] { "game_id", "revision", "actor", "request_id", "profile", "phase", "generation", "profile_fingerprint", "protocol", "schema", "rules" })
        {
            using (var coordinator = new TurnCoordinator(ticket, view, h.Choices(), profile, composition))
            {
                Check(coordinator.Worker.Wait(30000), "zero-budget complete incumbent terminates");
                var stale = JObject.FromObject(ticket);
                stale[field] = stale[field].Type == JTokenType.Integer ? (JToken)((long)stale[field] + 1) : (JToken)"stale";
                Check(coordinator.Take(stale.ToObject<Ticket>()) == null && coordinator.Take(ticket) == null, "stale " + field + " permanently rejects request");
            }
        }
        using (var coordinator = new TurnCoordinator(ticket, view, h.Choices(), profile, composition))
        {
            coordinator.Dispose();
            Check(coordinator.Worker.Wait(30000) && coordinator.Take(ticket) == null, "disposed worker cannot return late plan");
        }
        using (var coordinator = new TurnCoordinator(ticket, view, new List<Choice>(), profile, composition))
        { Check(coordinator.Worker.Wait(30000) && coordinator.Failure != null && coordinator.Take(ticket) == null, "worker failure is explicit"); }
        using (var coordinator = new TurnCoordinator(ticket, view, h.Choices(), profile, composition))
        {
            Check(coordinator.Worker.Wait(30000), "valid worker completes");
            var plan = coordinator.Take(ticket);
            Check(plan != null && plan.actions.Count == h.Remaining && coordinator.Take(ticket) == null, "completed turn can be consumed exactly once");
        }
        Check(Wire.Hash(h.Canonical()) == before, "asynchronous requests leave live state and RNG unchanged");
    }

    static object Match(int count, bool intro)
    {
        var config = new GameStartConfiguration((NumberOfPlayers)count) { Seed = 32700 + count, IsIntroductoryGame = intro };
        var metadata = new GameMetadata(config, GameType.Local);
        foreach (var start in config.Players)
        {
            var player = new PlayerMetadata { PlayerStartConfiguration = start, PlayerNetworkId = "gameplay-" + start.Color,
                NickName = "Scripted gameplay test", AiType = start.SittingOrder == 0 ? AIType.Human : AIType.Hard };
            player.AIBehaviour = AiBehaviorFactory.CreateAiBehavior(player); metadata.JoinGame(player);
        }
        var state = LocalGameState.Create(metadata, p => ProfileCatalog.Keys[(p.PlayerStartConfiguration.SittingOrder + count + (intro ? 2 : 0)) % 6]);
        SaveExtensions.Set(metadata, state);
        var app = new BrassApplicationGame(new BoardGameRulesGame(config), metadata);
        app.Game.StartGame(); state["game_id"] = app.Metadata.GameId; state["ready"] = true; SaveExtensions.Set(app.Metadata, state);
        var composition = Host.Composition(Host.Configuration(config)); var planner = new Planner(composition);
        var scripted = new HardAiBehavior("scripted-human");
        var strategies = state["assignments"].Where(r => (string)r["kind"] == "legacy").Select(r => (string)r["strategy"]).ToArray();
        foreach (string key in strategies) testedStrategies.Add(key);
        int turns = 0, actions = 0, aiActions = 0, humanActions = 0, debts = 0, railTurns = 0;
        var watch = System.Diagnostics.Stopwatch.StartNew();
        while (!app.Game.GameProgressInformation.IsGameOver)
        {
            Check(turns++ < 100, "live progress watchdog");
            if (app.Game.GameProgressInformation.CurrentGameEra.ToString() == "RailEra") railTurns++;
            int seat = (int)app.Game.GameProgressInformation.CurrentPlayer.Color;
            while (app.Game.TurnFlow.GetRemainingActionsCount() > 0)
            {
                if (app.Metadata.GetPlayerMetadataByColor((BoardGameRules.Entities.Players.PlayerColor)seat).AiType == AIType.Human)
                {
                    int remaining = app.Game.TurnFlow.GetRemainingActionsCount();
                    scripted.GetNextAIMove(app)(); app.Game.TurnFlow.ConfirmCurrentAction();
                    Check(app.Game.TurnFlow.GetRemainingActionsCount() == remaining - 1, "scripted human action completes once");
                    actions++; humanActions++; continue;
                }
                var events = Events(app); var live = Host.Capture(app, events);
                state = LocalGameState.Read(app.Metadata); var row = LocalGameState.Assignment(state, seat);
                var profile = Profile.Read((JObject)row["profile"]); profile.work_limit = 1;
                var priorIntent = LocalGameState.Intent(state, seat, app.Game.Revision, events);
                var decision = planner.Decide((JObject)Wire.Token(live.Observe(seat)), live.Choices(), profile, true, 1,
                    default(System.Threading.CancellationToken), null, priorIntent);
                Check(decision.selected.actions.Count == app.Game.TurnFlow.GetRemainingActionsCount(), "S1 returns the complete remaining turn");
                foreach (var path in decision.selected.actions)
                {
                    events = Events(app); live = Host.Capture(app, events);
                    string before = Wire.Hash(live.Canonical()); live.ValidateLiveAction(path, composition);
                    Check(Wire.Hash(live.Canonical()) == before, "complete action validation is isolated");
                    int remaining = app.Game.TurnFlow.GetRemainingActionsCount();
                    var command = new LegacyBehavior(app, () => live.LiveCommit(path, composition)).GetNextAIMove(app); command();
                    Check(app.Game.TurnFlow.GetRemainingActionsCount() == remaining - 1, "live action confirms once");
                    bool rejected = false; try { command(); } catch (InvalidOperationException) { rejected = true; }
                    Check(rejected, "reused prepared action rejected");
                    LocalGameState.CommitIntent(app.Metadata, decision.selected.intent, app.Game.Revision, Events(app));
                    var nextIntent = LocalGameState.Intent(LocalGameState.Read(app.Metadata), seat, app.Game.Revision, Events(app));
                    Check(nextIntent != null && nextIntent.profile == profile.id && nextIntent.seat == seat, "committed intent remains available in memory to its seat");
                    actions++; aiActions++;
                }
            }
            Check(app.Game.TurnFlow.CurrentAction == null, "completed turn has no unfinished action");
            app.Game.GameFlow.EndTurn();
            if (app.Game.GameFlow.IsLastTurnOver())
            {
                if (app.Game.GameFlow.IsSellIndustryNecessary())
                {
                    debts++; var selector = RuleSelectorFactory.CreatePrioritySelector(RuleSetFactory.CreateSellIndustryRuleSet());
                    selector.Select(app).MakeAction(app).Invoke();
                    Check(app.Game.SellIndustryFlow.CanIndustrySellingBeCompleted(), "end-round liquidation settles every debt");
                }
                app.Game.GameFlow.EndRound();
            }
        }
        Check(intro ? railTurns == 0 : railTurns > 0, "correct introductory/full era boundary");
        Check(aiActions > 0 && humanActions > 0, "both strategy bots and scripted human participated");
        Check(LocalGameState.Read(app.Metadata)["records"].Any(), "complete game retained in-memory tactical intent");
        return new { count, intro, turns, actions, aiActions, humanActions, debts, railTurns, strategies,
            elapsed_ms = watch.ElapsedMilliseconds, human = "Scripted original Hard behavior; no physical mouse playthrough",
            execution = "Real application game in memory; Host.Capture, ValidateLiveAction and LiveCommit; no save/reload" };
    }

    public static int Main(string[] args)
    {
        var matches = new List<object>();
        try
        {
            Coordinator();
            foreach (int count in new[] { 2, 3, 4 }) foreach (bool intro in new[] { false, true })
            {
                matches.Add(Match(count, intro));
                File.WriteAllText(args[0], Wire.Json(new { passed = false, running = true, checks, matches }));
                Console.WriteLine("Completed in-memory gameplay game " + count + "/" + intro);
            }
            Check(testedStrategies.SetEquals(ProfileCatalog.Keys), "six mixed games cover all six strategy profiles");
            File.WriteAllText(args[0], Wire.Json(new { passed = true, checks, matches, scope = "Live gameplay and coordinator lifecycle; save-game work excluded" }));
            return 0;
        }
        catch (Exception error)
        { File.WriteAllText(args[0], Wire.Json(new { passed = false, checks, matches, error = error.ToString() })); Console.Error.WriteLine(error); return 1; }
    }
}
