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
using BrassLegacyAI.Native;
using BrassApplication.UseCases.Game.PlayGame;

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
        NativeJob nativeBackend;
        AiExecutionJob nativeExecution;
        NativeValidation nativeValidation;
        AiExecutionJob validationExecution;
        string[] validatingPath;
        readonly string validationLifetime=Guid.NewGuid().ToString("N");
        int nativeAction;
        bool native;
        PlayGameUseCase debtOwner;
        string instance;
        int nativeAllowance;
        JArray nativeRecall = new JArray();
        AiExecutionScope resources;
        JObject pendingState;
        Ticket ticket;
        IntentState intent;
        readonly Queue<string[]> continuation = new Queue<string[]>();
        Stopwatch clock = new Stopwatch();
        public const double ThinkingMilliseconds = DifficultyCatalog.ExistingMilliseconds;
        // A safety guard, not the old beam width/work target. The turn clock
        // controls normal execution at the 4/8/10/12-second settings.
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
                bool native = NativeGameState.IsNative(state);
                if ((string)assignment["kind"] != (native ? "native" : "legacy")) throw new InvalidOperationException("An AI turn was requested for a human player.");
                if (active != null && ReferenceEquals(active.app, app) && active.revision == app.Game.Revision) return;
                CancelActive();
                var go = new GameObject("Brass strategy AI turn");
                var bridge = active = go.AddComponent<LocalTurnBridge>();
                bridge.history = history; bridge.presenter = presenter; bridge.router = router; bridge.app = app;
                bridge.revision = app.Game.Revision; bridge.actor = actor; bridge.endTurn = endTurn; bridge.startedGeneration = generation;
                bridge.native = native; bridge.instance = Guid.NewGuid().ToString("N");
                if(native){var lease=NativeTurnClock.For(app,actor,false);bridge.instance=lease.Instance;bridge.clock=lease.Clock;}
                bridge.strategy = (string)assignment["strategy"]; if (!native) bridge.profile = ProfileCatalog.ForPlay((JObject)assignment["profile"]);
                // Apply the requested live allowance on the detached execution copy.
                // Keep the bank's 240-node planning setting, including its bounded completion work.
                string difficulty = LocalGameState.Difficulty(state, assignment);
                bridge.budget = DifficultyCatalog.Milliseconds(difficulty);
                bridge.nativeAllowance = (int)bridge.budget;
                if (!native) bridge.profile.work_limit = SearchWorkLimit;
                bridge.composition = Host.Composition((JObject)state["configuration"]);
                bridge.controls = AiTurnControls.Attach(router, (native ? "Alpha Test" : ProfileCatalog.Label(bridge.strategy)) + " · " + DifficultyCatalog.Label(difficulty));
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
            if (native) { LaunchNative(state); return; }
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
        void LaunchNative(JObject state)
        {
            var events=Events();
            var input=NativeInput.Capture(app,events,actor,new JArray());
            if ((string)input["view"]["context"]["phase"] != (debtOwner==null?"action":"debt")) throw new InvalidOperationException("Native action boundary changed");
            if (planningRequests==0) {
                var saved=NativeGameState.Clock(state,(JObject)input["view"],LocalGameState.HistoryHash(events,revision));
                if(saved!=null && !clock.IsRunning) {
                    budget=Math.Min(budget,(int)saved["remaining_ms"]);
                    nativeRecall=(JArray)saved["recall"].DeepClone();
                    input["recall"]=nativeRecall.DeepClone();
                }
            }
            var hand=input["view"]["own_hand"];
            if(nativeRecall.Count==0 || !JToken.DeepEquals(nativeRecall.Last["hand"],hand))
                nativeRecall.Add(new JObject{["revision"]=input["view"]["context"]["revision"],["hand"]=hand.DeepClone()});
            input["recall"]=nativeRecall.DeepClone();
            planningRequests++;
            nativeAction=0;
            nativeBackend=new NativeJob(NativeInput.Request(input,instance,app.Metadata.GameId,startedGeneration+1,nativeAllowance),clock,(int)budget,NativeGameState.SearchK(state),NativeGameState.Temperature(state));
            nativeExecution=AiExecutionScheduler.Submit(nativeBackend);resources.Observe(nativeExecution);
        }
        // Called by the existing liquidation route before its V4 policy. No
        // single-seat substitution is possible in an NN table, including failure.
        public static bool PlayNativeDebt(PlayGameUseCase owner)
        {
            var history=ReflectionAccess.Get<IGameHistory>(owner,"_gameHistory");var app=history.CurrentGame;
            if(app==null || !NativeGameState.IsNative(SaveExtensions.Get(app.Metadata)))return false;
            var output=ReflectionAccess.Get<BrassApplication.UseCases.Game.PlayGame.IPlayGameOutput>(owner,"_presenter");
            var router=ReflectionAccess.Get<IRouter>(output,"_router");
            try {
                var state=LocalGameState.Read(app.Metadata);int actor=(int)app.Game.GameProgressInformation.CurrentPlayer.Color;
                if(active!=null && ReferenceEquals(active.app,app) && active.revision==app.Game.Revision)return true;
                CancelActive();
                var bridge=active=new GameObject("Brass native debt").AddComponent<LocalTurnBridge>();
                bridge.native=true;bridge.history=history;bridge.app=app;bridge.actor=actor;bridge.revision=app.Game.Revision;bridge.router=router;
                bridge.debtOwner=owner;bridge.instance=Guid.NewGuid().ToString("N");bridge.startedGeneration=generation;
                var lease=NativeTurnClock.For(app,actor,true);bridge.instance=lease.Instance;bridge.clock=lease.Clock;
                bridge.budget=bridge.nativeAllowance=DifficultyCatalog.Milliseconds(LocalGameState.Difficulty(state,LocalGameState.Assignment(state,actor)));
                bridge.composition=Host.Composition((JObject)state["configuration"]);bridge.controls=AiTurnControls.Attach(router,"Alpha Test · liquidation");bridge.StartRequest(state);
            } catch(Exception error){CancelActive();LocalMenu.ShowError(router,"Alpha Test stopped. The NN table and save are preserved.\n"+error.Message,error);}
            return true;
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
                    if(native) {
                        if(validatingPath!=null)goto NativeReady;
                        if(!nativeExecution.Completion.IsCompleted)return;
                        if(nativeExecution.Failure!=null || nativeExecution.CancellationRequested)throw new InvalidOperationException(nativeExecution.Failure??"Native request cancelled");
                        foreach(var action in NativeInput.Actions(nativeBackend.Result))continuation.Enqueue(action);
                    } else {
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
                }
                NativeReady:
                // Each ordinary action gets its own detached snapshot. In
                // particular action two is captured AFTER action one's actual
                // events, never validated speculatively on the first root.
                if(native && debtOwner==null && NativeValidation.Enabled) {
                    if(validatingPath==null) {
                        var captured=Host.Capture(app,Events());
                        if(captured.Phase!="action" || app.Game.TurnFlow.CurrentAction!=null)throw new InvalidOperationException("Native validation boundary changed");
                        validatingPath=continuation.Dequeue();
                        nativeValidation=new NativeValidation((JObject)Wire.Token(captured.Observe(actor)),composition,validatingPath,ValidationBinding(captured));
                        validationExecution=AiExecutionScheduler.Submit(nativeValidation);resources.Observe(validationExecution);
                        return;
                    }
                    if(!validationExecution.Completion.IsCompleted)return;
                    if(validationExecution.CancellationRequested)throw new InvalidOperationException("Native validation cancelled");
                    if(validationExecution.Failure!=null) {
                        nativeBackend.RecordFailure("detached-validation",validationExecution.Failure,validatingPath);
                        if(++replans>1)throw new InvalidOperationException(validationExecution.Failure);
                        validatingPath=null;nativeValidation=null;validationExecution=null;
                        continuation.Clear();StartRequest(LocalGameState.Read(app.Metadata));return;
                    }
                }
                // Restore rendering and registered effects before committing or
                // dispatching move/replay presentation, even on a deadline take.
                resources.Presenting();
                var path = validatingPath??continuation.Dequeue(); var events = Events(); var live = Host.Capture(app, events);
                // Validate before any mutation. A stale continuation can be replaced within the same allowance.
                try { if(native) {if(debtOwner!=null || !NativeValidation.Enabled)live.ValidatePersonalityAction(path, composition);} else live.ValidateLiveAction(path, composition); }
                catch (InvalidOperationException)
                {
                    if (++replans > 1) throw;
                    continuation.Clear(); StartRequest(LocalGameState.Read(app.Metadata)); return;
                }
                if(native && !NativePackage.Enabled)throw new InvalidOperationException("Alpha Test was disabled; stop this NN table or replace the whole table with V4.");
                // Reading the native save validates current model/export/family
                // identities. Revision/actor/game reference/generation were
                // checked at entry; receipt equality covers all remaining
                // complete-action validator inputs immediately before mutation.
                if(native) {
                    var currentState=LocalGameState.Read(app.Metadata);
                    if(!NativeGameState.IsNative(currentState) || (string)LocalGameState.Assignment(currentState,actor)["kind"]!="native" ||
                        DifficultyCatalog.Milliseconds(LocalGameState.Difficulty(currentState,LocalGameState.Assignment(currentState,actor)))!=nativeAllowance)
                        throw new InvalidOperationException("Native assignment changed before commit");
                }
                new LegacyBehavior(app, () => {
                    if(native && debtOwner==null && NativeValidation.Enabled)nativeValidation.Commit(live,composition,path,ValidationBinding(live),stopped || !CurrentGeneration(startedGeneration) || validationExecution.CancellationRequested);
                    else if(native)live.CommitPersonalityAction(path,composition);
                    else live.LiveCommit(path, composition);
                }).GetNextAIMove(app)();
                validatingPath=null;nativeValidation=null;validationExecution=null;nativeAction++;
                revision = app.Game.Revision;
                if(native && debtOwner!=null) {
                    if(!app.Game.SellIndustryFlow.CanIndustrySellingBeCompleted((BoardGameRules.Entities.Players.PlayerColor)actor)) {
                        var debtEvents=Events();var debtView=(JObject)NativeInput.Capture(app,debtEvents,actor,new JArray())["view"];
                        NativeGameState.Record(app.Metadata,debtView,LocalGameState.HistoryHash(debtEvents,revision),
                            Math.Max(0,(int)(budget-clock.Elapsed.TotalMilliseconds)),nativeRecall);
                    }
                    var owner=debtOwner;Stop();
                    owner.PlayGame();return;
                }
                if(native && app.Game.TurnFlow.GetRemainingActionsCount()>0) {
                    var nowEvents=Events();var current=NativeInput.Capture(app,nowEvents,actor,new JArray());
                    // An internally projected root is a hypothesis. A legal
                    // but different live successor must replan on the original
                    // turn clock rather than execute that hypothetical action.
                    if(continuation.Count>0 && !NativeInput.ContinuationMatches(nativeBackend.Result,current,nativeRecall))
                        continuation.Clear();
                    NativeGameState.Record(app.Metadata,(JObject)current["view"],LocalGameState.HistoryHash(nowEvents,revision),
                        Math.Max(0,(int)(budget-clock.Elapsed.TotalMilliseconds)),nativeRecall);
                }
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
                if(native && nativeBackend!=null)nativeBackend.RecordFailure("live-validation-or-commit",error.ToString(),validatingPath);
                Stop();
                LocalMenu.ShowError(router, "The AI turn stopped. Return to the menu and reload the last saved turn.\n" + error.Message, error);
            }
        }
        string ValidationBinding(Host live)
        {
            return NativeValidation.Binding(validationLifetime,instance,app.Metadata.GameId,generation,planningRequests,nativeAction,
                app.Game.Revision,live.Actor,live.Phase,nativeBackend.Result);
        }
        void Stop()
        {
            stopped = true; coordinator?.Dispose(); coordinator = null; nativeExecution?.Cancel();validationExecution?.Cancel();
            resources?.Dispose(); resources = null; pendingState = null; continuation.Clear();
            if (ReferenceEquals(active, this)) active = null;
            if (controls != null) Destroy(controls.gameObject);
            Destroy(gameObject);
        }
        void OnDestroy()
        {
            stopped = true; coordinator?.Dispose(); coordinator = null; nativeExecution?.Cancel();validationExecution?.Cancel();
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
