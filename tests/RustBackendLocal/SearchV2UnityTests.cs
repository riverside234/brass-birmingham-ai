using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using BrassHost;
using BrassLegacyAI;
using BrassLegacyAI.Native;
using BrassLegacyAI.Execution;
using BrassPatch;
using BrassPatch.Persistence;
using BrassApplication.AI;
using BrassApplication.Game;
using BrassApplication.UseCases.Game.PlayAiTurn;
using BrassApplication.UseCases.Game.PlayGame;
using BoardGameRules.Entities.Game;
using BoardGameRules.Entities.EventSourcing.Events;
using BoardGameRules.Entities.GamePhases.Events;
using InterfaceAdapters.Game.PlayAiTurn;
using Newtonsoft.Json.Linq;
using UnityEngine.SceneManagement;

public sealed partial class LocalAiUnityDriver
{
    IEnumerator SearchV2Cases()
    {
        string destination=Environment.GetEnvironmentVariable("BRASS_SEARCH_V2_RESULT");
        string fixture=Environment.GetEnvironmentVariable("BRASS_SEARCH_V2_EVENTS");
        string scenario=Environment.GetEnvironmentVariable("BRASS_SEARCH_V2_SCENARIO")??"normal";
        int allowance=Int32.Parse(Environment.GetEnvironmentVariable("BRASS_SEARCH_V2_MS"));
        int searchK=Int32.Parse(Environment.GetEnvironmentVariable("BRASS_SEARCH_K")??"16");
        var graceField=typeof(NativeJob).GetField("DeliveryGraceMilliseconds");
        int grace=graceField==null?0:(int)graceField.GetRawConstantValue();
        int totalAllowance=allowance+grace;
        int holdUntil=Int32.Parse(Environment.GetEnvironmentVariable("BRASS_SEARCH_HOLD_UNTIL_MS")??"0");
        Check(holdUntil>=0 && holdUntil<=(scenario=="expired"?totalAllowance+100:totalAllowance),"late probe is within its declared scope");
        double temperature=Double.Parse(Environment.GetEnvironmentVariable("BRASS_NN_TEMPERATURE")??"1.0",System.Globalization.CultureInfo.InvariantCulture);
        int priorK=searchK==32?16:searchK;
        double priorTemperature=searchK==32?temperature:temperature==1.0?1.5:1.0;
        string method=(string)new NativePackage(NativePackage.Location,searchK,temperature).Policy["method"];
        Check(!File.Exists(destination),"fresh search study output");
        yield return SceneManager.LoadSceneAsync("GameScene");yield return null;router.ReloadPrefabs();
        var view=(Frameworks.View.Game.PlayAiTurnView)router.GetView<IPlayAiTurnView>();
        var turn=new PlayAiTurnUseCase(new PlayAITurnPresenter(router),history);
        router.UseCases[typeof(PlayAiTurnUseCase)]=turn;view.Controller=new PlayAiTurnController(turn);
        var serializer=new Frameworks.Persistence.Serialization.JsonSerializer();List<IEvent> replay;
        using(var stream=File.OpenRead(fixture))replay=serializer.Deserialize<List<IEvent>>(stream);
        var game=new BoardGameRulesGame(((GameStarted)replay[0]).GameStartConfiguration);
        foreach(var e in replay)game.ApplyEvent(e);
        var metadata=new GameMetadata(game.GameStartConfiguration,GameType.Local){GameId="search-v2-"+Guid.NewGuid().ToString("N")};
        foreach(var start in game.GameStartConfiguration.Players)metadata.JoinGame(new PlayerMetadata{PlayerStartConfiguration=start,
            AiType=AIType.Hard,NickName="Alpha Test",PlayerNetworkId="local-"+start.Color});
        string difficulty=DifficultyCatalog.Keys.Single(k=>DifficultyCatalog.Milliseconds(k)==allowance);
        var app=new BrassApplicationGame(game,metadata);var state=NativeGameState.Create(metadata,p=>difficulty,searchK,temperature);
        state["game_id"]=metadata.GameId;state["ready"]=true;SaveExtensions.Set(metadata,state);
        history.Repository.Save(app);history.SetGame(app);
        int actor=(int)game.GameProgressInformation.CurrentPlayer.Color,initial=game.Revision;
        Check(game.TurnFlow.GetRemainingActionsCount()==2,"two ordinary live actions");
        var capture=Stopwatch.StartNew();var input=NativeInput.Capture(app,replay,actor,new JArray());capture.Stop();
        // Warm the other policy first, then switch under the scheduler lease.
        // This exercises table replacement without retaining two worker pools.
        var cold=Stopwatch.StartNew();var warmRequest=NativeInput.Request(input,"warm-"+Guid.NewGuid().ToString("N"),metadata.GameId,1,4000);
        var warm=new NativeJob(warmRequest,new Stopwatch(),4000,priorK,priorTemperature);
        var warmJob=AiExecutionScheduler.Submit(warm);while(!warmJob.Completion.IsCompleted)yield return null;
        Check(warmJob.Failure==null,"native warmup");cold.Stop();history.SetGame(app);
        var priorWorker=Get<Process>(NativeProcess.Shared(priorK,priorTemperature),"process");
        warmRequest=NativeInput.Request(input,"warm-selected-"+Guid.NewGuid().ToString("N"),metadata.GameId,1,4000);
        warm=new NativeJob(warmRequest,new Stopwatch(),4000,searchK,temperature);warmJob=AiExecutionScheduler.Submit(warm);
        while(!warmJob.Completion.IsCompleted)yield return null;
        Check(warmJob.Failure==null && (int)warm.Result["requested_k"]==searchK,"selected K worker warmed under scheduler");
        bool retired=false;try{retired=priorWorker.HasExited;}catch(InvalidOperationException){retired=true;}
        Check(retired,"previous K worker retired before replacement");
        var process=Process.GetCurrentProcess();var worker=Get<Process>(NativeProcess.Shared(searchK,temperature),"process");
        Check(NativeProcess.Shared(searchK,temperature).Readiness["k_by_allowance_ms"].Children<JProperty>().All(p=>(int)p.Value==searchK),"native readiness maps all budgets to selected K");
        Check((double)NativeProcess.Shared(searchK,temperature).Readiness["policy_temperature"]==temperature,"worker readiness binds selected temperature");
        process.Refresh();worker.Refresh();var cpu=process.TotalProcessorTime;var workerCpu=worker.TotalProcessorTime;
        var timer=Stopwatch.StartNew();var jobs=new List<NativeJob>();var commits=new JArray();
        int previous=initial,requests=0;bool intervened=false;BrassApplicationGame loaded=null;
        router.ShowView<IPlayAiTurnView>();turn.PlayAITurn(false);
        // Test-only late-consumption probe. Native inference/retirement and the
        // original turn stopwatch continue normally; only the bridge's Update
        // waits. No compiled runtime, rules state or search clock is changed.
        var held=typeof(LocalTurnBridge).GetField("active",ReflectionAccess.Flags).GetValue(null) as UnityEngine.Behaviour;
        var turnClock=held==null?null:Get<Stopwatch>(held,"clock");
        double resumed=-1,replyObserved=-1,originalReceived=-1;
        if(holdUntil>0){Check(held!=null && (scenario=="normal" || scenario=="expired"),"late probe owns its live bridge");held.enabled=false;}
        if(scenario=="expired")expectedError=true;
        while((LocalTurnBridge.IsThinking || !AiExecutionScheduler.IsIdle) && timer.Elapsed.TotalSeconds<15) {
            var bridge=typeof(LocalTurnBridge).GetField("active",ReflectionAccess.Flags).GetValue(null);
            if(bridge!=null){requests=Math.Max(requests,Get<int>(bridge,"planningRequests"));var job=Get<NativeJob>(bridge,"nativeBackend");if(job!=null && !jobs.Contains(job))jobs.Add(job);
                if(replyObserved<0 && job!=null && job.Result!=null && Get<AiExecutionJob>(bridge,"nativeExecution").Completion.IsCompleted)replyObserved=turnClock.Elapsed.TotalMilliseconds;}
            if(holdUntil>0 && resumed<0 && turnClock.IsRunning && turnClock.Elapsed.TotalMilliseconds>=holdUntil) {
                // A completed early response is delivered by this trusted test
                // gate now. Preserve its real native timing separately; update
                // only the test receipt timestamp to model delayed transport.
                var delayed=Get<NativeJob>(held,"nativeBackend");
                var stamp=typeof(NativeJob).GetProperty("ResponseReceivedMilliseconds");
                if(grace>0 && stamp!=null && Get<AiExecutionJob>(held,"nativeExecution").Completion.IsCompleted && delayed.Result!=null) {
                    originalReceived=(double)stamp.GetValue(delayed,null);
                    stamp.SetValue(delayed,turnClock.Elapsed.TotalMilliseconds,null);
                }
                resumed=turnClock.Elapsed.TotalMilliseconds;held.enabled=true;
            }
            if(game.Revision!=previous){commits.Add(new JObject{["ms"]=timer.Elapsed.TotalMilliseconds,["revision"]=game.Revision,["remaining"]=game.TurnFlow.GetRemainingActionsCount()});previous=game.Revision;}
            if(!intervened && game.TurnFlow.GetRemainingActionsCount()==1 && scenario!="normal") {
                intervened=true;
                if(scenario=="cancel")LocalTurnBridge.CancelActive();
                else if(scenario=="load"){loaded=history.Repository.Load(metadata.GameId);history.SetGame(loaded);}
                else throw new Exception("Unknown study scenario");
            }
            yield return null;
        }
        timer.Stop();process.Refresh();worker.Refresh();
        double turnClockMs=turnClock==null?0:turnClock.Elapsed.TotalMilliseconds;
        if(holdUntil>0)Check(resumed>=holdUntil,"late-consumption probe actually reached its target");
        expectedError=false;
        if(game.Revision!=previous)commits.Add(new JObject{["ms"]=timer.Elapsed.TotalMilliseconds,["revision"]=game.Revision,["remaining"]=game.TurnFlow.GetRemainingActionsCount()});
        Check(!LocalTurnBridge.IsThinking && AiExecutionScheduler.IsIdle && !AiExecutionResources.IsComputing,"owners and validator retired");
        Check(history.Repository.Load(metadata.GameId).Game.Revision==initial,"no action/search save");
        var replies=new JArray(jobs.Where(j=>j.Result!=null).Select(j=>j.Result.DeepClone()));
        Check(replies.All(r=>(int)r["requested_k"]==searchK),"production bridge uses saved search K");
        Check(replies.All(r=>(double)r["policy_temperature"]==temperature && (string)r["temperature_scope"]=="all_nn_policy_evaluations"),"production bridge uses saved temperature for all evaluations");
        bool focused=replies.Any(r=>(bool?)r["search_revision"]?["focus"]?["switched"]==true);
        Check(replies.Count>0,"completed production search reply");
        if(method=="search_v1")Check(replies.All(r=>r["search_revision"]==null),"search_v1 has no focus/composite revision");
        else Check(replies.All(r=>(string)r["search_revision"]?["focus"]?["method"]==method),"effective search method matches package");
        if(scenario=="expired") {
            Check(game.Revision==initial && game.TurnFlow.GetRemainingActionsCount()==2,"expired reply cannot mutate either action");
            Check(jobs.Any(j=>j.FailureBundle!=null),"expired reply retains a failure receipt");
        }else if(scenario=="normal") {
            File.WriteAllText(destination+".diagnostic.json",new JObject{["remaining"]=game.TurnFlow.GetRemainingActionsCount(),
                ["requests"]=requests,["elapsed_ms"]=timer.Elapsed.TotalMilliseconds,["commits"]=commits.DeepClone(),["replies"]=replies.DeepClone()}.ToString());
            Check(game.TurnFlow.GetRemainingActionsCount()==0 && game.Revision>initial,"both actions committed");
            if(focused)Check(requests==1 && replies.Count==1,"focused plan used one native request");
            var output=new ReleaseOutput(router,history);var play=new PlayGameUseCase(output,history,null,new Analytics());
            play.EndTurn();Check(output.Saves==1,"one completed AI turn checkpoint");
            loaded=history.Repository.Load(metadata.GameId);Check(loaded.Game.Revision==game.Revision,"checkpoint includes next-turn boundary");
            Check(NativeGameState.SearchK(NativeGameState.Read(loaded.Metadata,false))==searchK,"completed-turn save/load retains chosen K");
            Check(NativeGameState.Temperature(NativeGameState.Read(loaded.Metadata,false))==temperature,"completed-turn save/load retains chosen temperature");
        }else {
            Check(intervened && game.TurnFlow.GetRemainingActionsCount()==1,"interruption prevented action two");
            if(loaded!=null)Check(loaded.Game.Revision==initial,"load discarded unsaved action and old receipt");
        }
        Check(errors.Count==0,"no Unity errors");
        File.WriteAllText(destination,new JObject{["passed"]=true,["scenario"]=scenario,["allowance_ms"]=allowance,["search_k"]=searchK,["focused"]=focused,
            ["temperature"]=temperature,["search_method"]=method,
            ["requests"]=requests,["initial_snapshot_ms"]=capture.Elapsed.TotalMilliseconds,["cold_warmup_ms"]=cold.Elapsed.TotalMilliseconds,
            ["delivery_grace_ms"]=grace,["total_allowance_ms"]=totalAllowance,["original_response_received_clock_ms"]=originalReceived,
            ["late_delivery_hold_until_ms"]=holdUntil,["late_delivery_resumed_ms"]=resumed,["native_reply_observed_clock_ms"]=replyObserved,
            ["turn_clock_ms"]=turnClockMs,["delivery_tail_after_resume_ms"]=resumed<0?JValue.CreateNull():new JValue(turnClockMs-resumed),
            ["delivery_ms"]=timer.Elapsed.TotalMilliseconds,["unity_cpu_ms"]=(process.TotalProcessorTime-cpu).TotalMilliseconds,
            ["native_cpu_ms"]=(worker.TotalProcessorTime-workerCpu).TotalMilliseconds,["native_peak_bytes"]=worker.PeakWorkingSet64>0?new JValue(worker.PeakWorkingSet64):JValue.CreateNull(),
            ["commits"]=commits,["input"]=input,["replies"]=replies,["readiness"]=NativeProcess.Shared(searchK,temperature).Readiness}.ToString());
    }
}
