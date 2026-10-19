using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using BrassHost;
using BrassPatch;
using BrassPatch.Persistence;
using BrassApplication.Game;
using BrassApplication.GameHistory;
using BrassApplication.UseCases.Game.ExitToMenu;
using BrassApplication.UseCases.Game.PlayAiTurn;
using BrassApplication.UseCases.Game.Board;
using InterfaceAdapters.Game.PlayAiTurn;
using InterfaceAdapters.Game.PlayGame;
using InterfaceAdapters.Game.PlayerActions;
using InterfaceAdapters.Routing;
using Newtonsoft.Json.Linq;
using UnityEngine;
using BrassLegacyAI.Execution;

namespace BrassLegacyAI
{
    public sealed class LocalTurnBridge : MonoBehaviour
    {
        static LocalTurnBridge active;
        static long generation;
        IGameHistory history;
        IPlayAITurnOutput presenter;
        IRouter router;
        BrassApplicationGame app;
        Profile profile;
        JObject composition;
        TurnCoordinator coordinator;
        AiExecutionScope resources;
        JObject pendingState;
        Ticket ticket;
        IntentState intent;
        readonly Queue<string[]> continuation = new Queue<string[]>();
        readonly Stopwatch clock = new Stopwatch();
        public const double ThinkingMilliseconds = DifficultyCatalog.ExistingMilliseconds;
        // A safety guard, not the old beam width/work target. The turn clock
        // controls normal execution at the 6/10/15-second settings.
        public const int SearchWorkLimit = 1000000;
        double budget;
        int planningRequests;
        double lastRequestMilliseconds;
        int revision, actor, replans;
        long startedGeneration;
        bool endTurn, stopped;
        string strategy;
        AiTurnControls controls;
        public static bool IsThinking { get { return active != null && !active.stopped; } }
        // Kept for binary compatibility; no UI or stale callback may truncate search.
        public static void PlayNow() { }
        public static void CancelActive()
        {
            generation++;
            if (!ReferenceEquals(active, null)) CancelUnity();
            // The history/menu hooks also own future backend scopes. The guarded
            // non-inlined call preserves the managed, non-Unity rules harness.
            if (AiExecutionResources.ScopeCount > 0) CancelResourcesUnity();
        }
        // History is also used by the managed rules harness without a Unity player.
        [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
        static void CancelUnity() { active.Stop(); }
        [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
        static void CancelResourcesUnity() { AiExecutionResources.Reset(); }
        public static void OnSetGame(BrassGameHistory owner, BrassApplicationGame game) { CancelActive(); }
        public static void OnMoveHistory(BrassGameHistory owner, int revision) { CancelActive(); }
        public static void OnExit(ExitToMenuUseCase owner) { CancelActive(); }
        public static void Schedule(PlayGamePresenter owner, PlayerMetadata player)
        {
            var router = ReflectionAccess.Get<IRouter>(owner, "_router");
            var useCase = router.UseCaseBuilder.GetUseCase<PlayAiTurnUseCase>();
            var history = ReflectionAccess.Get<IGameHistory>(useCase, "_gameHistory");
            if (history.CurrentGame == null) return;
            if (router.IsViewVisible<IPlayerActionsView>()) router.HideView<IPlayerActionsView>();
            router.ShowView<IPlayAiTurnView>();
            router.GetView<IPlayAiTurnView>().SetCurrentPlayerName(player.NickName);
            var dispatch = new GameObject("Brass AI turn dispatch").AddComponent<AiTurnDispatch>();
            dispatch.Initialize(useCase, history, router, history.CurrentGame, generation);
        }
        internal static bool CurrentGeneration(long value) { return value == generation; }
        public static void PlayAITurn(PlayAiTurnUseCase owner, bool endTurn)
        {
            var history = ReflectionAccess.Get<IGameHistory>(owner, "_gameHistory");
            var app = history.CurrentGame;
            if (app == null) return;
            if (SaveExtensions.Get(app.Metadata) == null)
            {
                ReflectionAccess.Original(owner, "PlayAITurn", new[] { typeof(bool) }, endTurn); return;
            }
            var presenter = ReflectionAccess.Get<IPlayAITurnOutput>(owner, "_presenter");
            var router = ReflectionAccess.Get<IRouter>(presenter, "_router");
            try
            {
                var state = LocalGameState.Read(app.Metadata);
                int actor = (int)app.Game.GameProgressInformation.CurrentPlayer.Color;
                var assignment = LocalGameState.Assignment(state, actor);
                if ((string)assignment["kind"] != "legacy") throw new InvalidOperationException("An AI turn was requested for a human player.");
                if (active != null && ReferenceEquals(active.app, app) && active.revision == app.Game.Revision) return;
                CancelActive();
                var go = new GameObject("Brass strategy AI turn");
                var bridge = active = go.AddComponent<LocalTurnBridge>();
                bridge.history = history; bridge.presenter = presenter; bridge.router = router; bridge.app = app;
                bridge.revision = app.Game.Revision; bridge.actor = actor; bridge.endTurn = endTurn; bridge.startedGeneration = generation;
                bridge.strategy = (string)assignment["strategy"]; bridge.profile = ProfileCatalog.ForPlay((JObject)assignment["profile"]);
                // Apply the requested live allowance on the detached execution copy.
                // Keep the bank's 240-node planning setting, including its bounded completion work.
                string difficulty = LocalGameState.Difficulty(state, assignment);
                bridge.budget = DifficultyCatalog.Milliseconds(difficulty);
                bridge.profile.work_limit = SearchWorkLimit;
                bridge.composition = Host.Composition((JObject)state["configuration"]);
                bridge.controls = AiTurnControls.Attach(router, ProfileCatalog.Label(bridge.strategy) + " · " + DifficultyCatalog.Label(difficulty));
                bridge.StartRequest(state);
            }
            catch (Exception error)
            {
                CancelActive();
                LocalMenu.ShowError(router, "The selected AI could not start. No substitute AI will play. Return to the menu to reload the last saved turn.\n" + error.Message, error);
            }
        }
        List<BoardGameRules.Entities.EventSourcing.Events.IEvent> Events()
        {
            return LocalGameState.History(app, history.Repository.Advanced.GameStorage.LoadEvents(app.Metadata.GameId, 0, app.Game.Revision));
        }
        void StartRequest(JObject state)
        {
            coordinator?.Dispose(); coordinator = null;
            if (resources == null) resources = AiExecutionResources.BeginComputation();
            else resources.WaitingForExecution();
            pendingState = state;
            if (AiExecutionScheduler.IsIdle) LaunchRequest();
        }
        void LaunchRequest()
        {
            var state = pendingState; pendingState = null;
            var events = Events(); var host = Host.Capture(app, events);
            if (host.Phase != "action" || app.Game.TurnFlow.CurrentAction != null) throw new InvalidOperationException("AI planning requires a completed action boundary.");
            var view = (JObject)Wire.Token(host.Observe(actor));
            // Capturing/serializing the live board is preparation, not thinking time.
            if (!clock.IsRunning) clock.Start();
            var cfg = Profile.Read(JObject.FromObject(profile)); ProfileCatalog.ConfigureClock(cfg,budget,LocalGameState.Difficulty(state,LocalGameState.Assignment(state,actor))); cfg.milliseconds = Math.Max(0, budget - clock.Elapsed.TotalMilliseconds);
            planningRequests++; lastRequestMilliseconds = cfg.milliseconds;
            ticket = Ticket.Create(host.GameId, app.Game.Revision, actor, host.Phase, cfg);
            ticket.profile_fingerprint = (string)LocalGameState.Assignment(state, actor)["profile_sha256"];
            ticket.generation = startedGeneration;
            coordinator = new TurnCoordinator(ticket, view, host.Choices(), cfg, composition, LocalGameState.Intent(state, actor, app.Game.Revision, events));
            resources.Observe(coordinator.Execution);
        }
        void Update()
        {
            if (stopped) return;
            if (!CurrentGeneration(startedGeneration) || !ReferenceEquals(history.CurrentGame, app) ||
                app.Game.Revision != revision || (int)app.Game.GameProgressInformation.CurrentPlayer.Color != actor)
            { Stop(); return; }
            try
            {
                if (pendingState != null)
                {
                    if (controls != null) controls.SetApplying(false);
                    if (AiExecutionScheduler.IsIdle) LaunchRequest();
                    return;
                }
                if (controls != null) controls.SetApplying(continuation.Count > 0);
                if (continuation.Count == 0)
                {
                    var plan = coordinator.Take(ticket, app.Game.Revision, actor, generation);
                    if (plan == null)
                    {
                        if (coordinator.Finished && !coordinator.Ready) throw new InvalidOperationException(coordinator.Failure ?? "The AI did not produce a complete turn.");
                        return;
                    }
                    foreach (var action in plan.actions) continuation.Enqueue(action);
                    if (controls != null) controls.SetApplying(true);
                    intent = plan.intent;
                    if (continuation.Count == 0) throw new InvalidOperationException("The AI returned an empty turn.");
                }
                // Restore rendering and registered effects before committing or
                // dispatching move/replay presentation, even on a deadline take.
                resources.Presenting();
                var path = continuation.Dequeue(); var events = Events(); var live = Host.Capture(app, events);
                // Validate before any mutation. A stale continuation can be replaced within the same allowance.
                try { live.ValidateLiveAction(path, composition); }
                catch (InvalidOperationException)
                {
                    if (++replans > 1) throw;
                    continuation.Clear(); StartRequest(LocalGameState.Read(app.Metadata)); return;
                }
                new LegacyBehavior(app, () => live.LiveCommit(path, composition)).GetNextAIMove(app)();
                revision = app.Game.Revision;
                LocalGameState.CommitIntent(app.Metadata, intent, revision, Events());
                if (app.Game.TurnFlow.GetRemainingActionsCount() == 0)
                {
                    bool advance = endTurn; var output = presenter;
                    Stop(); if (advance) output.EndTurn(); return;
                }
                if (continuation.Count == 0) StartRequest(LocalGameState.Read(app.Metadata));
            }
            catch (Exception error)
            {
                Stop();
                LocalMenu.ShowError(router, "The AI turn stopped. Return to the menu and reload the last saved turn.\n" + error.Message, error);
            }
        }
        void Stop()
        {
            stopped = true; coordinator?.Dispose(); coordinator = null;
            resources?.Dispose(); resources = null; pendingState = null; continuation.Clear();
            if (ReferenceEquals(active, this)) active = null;
            if (controls != null) Destroy(controls.gameObject);
            Destroy(gameObject);
        }
        void OnDestroy()
        {
            stopped = true; coordinator?.Dispose(); coordinator = null;
            resources?.Dispose(); resources = null; pendingState = null; continuation.Clear();
            if (ReferenceEquals(active, this)) active = null;
            if (controls != null) Destroy(controls.gameObject);
            app = null; history = null; presenter = null; router = null; composition = null; intent = null;
        }
    }
    public sealed class AiTurnDispatch : MonoBehaviour
    {
        PlayAiTurnUseCase owner; IGameHistory history; IRouter router; BrassApplicationGame game; long generation; int revision;
        public void Initialize(PlayAiTurnUseCase owner, IGameHistory history, IRouter router, BrassApplicationGame game, long generation)
        { this.owner = owner; this.history = history; this.router = router; this.game = game; this.generation = generation; revision = game.Game.Revision; }
        void Update()
        {
            Destroy(gameObject);
            if (owner == null || !LocalTurnBridge.CurrentGeneration(generation) || !ReferenceEquals(history.CurrentGame, game) || game.Game.Revision != revision) return;
            owner.PlayAITurn();
        }
    }
}
