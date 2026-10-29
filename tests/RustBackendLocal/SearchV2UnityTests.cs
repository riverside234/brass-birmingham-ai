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
        var app=new BrassApplicationGame(game,metadata);var state=NativeGameState.Create(metadata,p=>allowance==4000?"easy":"mid",searchK);
        state["game_id"]=metadata.GameId;state["ready"]=true;SaveExtensions.Set(metadata,state);
        history.Repository.Save(app);history.SetGame(app);
        int actor=(int)game.GameProgressInformation.CurrentPlayer.Color,initial=game.Revision;
        Check(game.TurnFlow.GetRemainingActionsCount()==2,"two ordinary live actions");
        var capture=Stopwatch.StartNew();var input=NativeInput.Capture(app,replay,actor,new JArray());capture.Stop();
        // Warm the other policy first, then switch under the scheduler lease.
        // This exercises table replacement without retaining two worker pools.
        var cold=Stopwatch.StartNew();var warmRequest=NativeInput.Request(input,"warm-"+Guid.NewGuid().ToString("N"),metadata.GameId,1,4000);
        var warm=new NativeJob(warmRequest,new Stopwatch(),4000,searchK==8?16:8);
        var warmJob=AiExecutionScheduler.Submit(warm);while(!warmJob.Completion.IsCompleted)yield return null;
        Check(warmJob.Failure==null,"native warmup");cold.Stop();history.SetGame(app);
        var priorWorker=Get<Process>(NativeProcess.Shared(searchK==8?16:8),"process");
        warmRequest=NativeInput.Request(input,"warm-selected-"+Guid.NewGuid().ToString("N"),metadata.GameId,1,4000);
        warm=new NativeJob(warmRequest,new Stopwatch(),4000,searchK);warmJob=AiExecutionScheduler.Submit(warm);
        while(!warmJob.Completion.IsCompleted)yield return null;
        Check(warmJob.Failure==null && (int)warm.Result["requested_k"]==searchK,"selected K worker warmed under scheduler");
        bool retired=false;try{retired=priorWorker.HasExited;}catch(InvalidOperationException){retired=true;}
        Check(retired,"previous K worker retired before replacement");
        var process=Process.GetCurrentProcess();var worker=Get<Process>(NativeProcess.Shared(searchK),"process");
        Check(NativeProcess.Shared(searchK).Readiness["k_by_allowance_ms"].Children<JProperty>().All(p=>(int)p.Value==searchK),"native readiness maps all budgets to selected K");
        process.Refresh();worker.Refresh();var cpu=process.TotalProcessorTime;var workerCpu=worker.TotalProcessorTime;
        var timer=Stopwatch.StartNew();var jobs=new List<NativeJob>();var commits=new JArray();
        int previous=initial,requests=0;bool intervened=false;BrassApplicationGame loaded=null;
        router.ShowView<IPlayAiTurnView>();turn.PlayAITurn(false);
        while((LocalTurnBridge.IsThinking || !AiExecutionScheduler.IsIdle) && timer.Elapsed.TotalSeconds<15) {
            var bridge=typeof(LocalTurnBridge).GetField("active",ReflectionAccess.Flags).GetValue(null);
            if(bridge!=null){requests=Math.Max(requests,Get<int>(bridge,"planningRequests"));var job=Get<NativeJob>(bridge,"nativeBackend");if(job!=null && !jobs.Contains(job))jobs.Add(job);}
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
        if(game.Revision!=previous)commits.Add(new JObject{["ms"]=timer.Elapsed.TotalMilliseconds,["revision"]=game.Revision,["remaining"]=game.TurnFlow.GetRemainingActionsCount()});
        Check(!LocalTurnBridge.IsThinking && AiExecutionScheduler.IsIdle && !AiExecutionResources.IsComputing,"owners and validator retired");
        Check(history.Repository.Load(metadata.GameId).Game.Revision==initial,"no action/search save");
        var replies=new JArray(jobs.Where(j=>j.Result!=null).Select(j=>j.Result.DeepClone()));
        Check(replies.All(r=>(int)r["requested_k"]==searchK),"production bridge uses saved search K");
        bool focused=replies.Any(r=>(bool?)r["search_revision"]?["focus"]?["switched"]==true);
        if(scenario=="normal") {
            Check(game.TurnFlow.GetRemainingActionsCount()==0 && game.Revision>initial,"both actions committed");
            if(focused)Check(requests==1 && replies.Count==1,"focused plan used one native request");
            var output=new ReleaseOutput(router,history);var play=new PlayGameUseCase(output,history,null,new Analytics());
            play.EndTurn();Check(output.Saves==1,"one completed AI turn checkpoint");
            loaded=history.Repository.Load(metadata.GameId);Check(loaded.Game.Revision==game.Revision,"checkpoint includes next-turn boundary");
            Check(NativeGameState.SearchK(NativeGameState.Read(loaded.Metadata,false))==searchK,"completed-turn save/load retains chosen K");
        }else {
            Check(intervened && game.TurnFlow.GetRemainingActionsCount()==1,"interruption prevented action two");
            if(loaded!=null)Check(loaded.Game.Revision==initial,"load discarded unsaved action and old receipt");
        }
        Check(errors.Count==0,"no Unity errors");
        File.WriteAllText(destination,new JObject{["passed"]=true,["scenario"]=scenario,["allowance_ms"]=allowance,["search_k"]=searchK,["focused"]=focused,
            ["requests"]=requests,["initial_snapshot_ms"]=capture.Elapsed.TotalMilliseconds,["cold_warmup_ms"]=cold.Elapsed.TotalMilliseconds,
            ["delivery_ms"]=timer.Elapsed.TotalMilliseconds,["unity_cpu_ms"]=(process.TotalProcessorTime-cpu).TotalMilliseconds,
            ["native_cpu_ms"]=(worker.TotalProcessorTime-workerCpu).TotalMilliseconds,["native_peak_bytes"]=worker.PeakWorkingSet64>0?new JValue(worker.PeakWorkingSet64):JValue.CreateNull(),
            ["commits"]=commits,["input"]=input,["replies"]=replies,["readiness"]=NativeProcess.Shared(searchK).Readiness}.ToString());
    }
}
