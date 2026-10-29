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
using BoardGameRules.Entities.Game;
using BoardGameRules.Entities.EventSourcing.Events;
using BoardGameRules.Entities.GamePhases.Events;
using BrassApplication.UseCases.Game.PlayAiTurn;
using InterfaceAdapters.Game.PlayAiTurn;
using Newtonsoft.Json.Linq;
using UnityEngine;
using UnityEngine.SceneManagement;

public sealed partial class LocalAiUnityDriver
{
    static Host MatchedHost(JObject fixture) {
        // Actual vendor events, including actual draws, are needed to exercise
        // live persistence/projection. The older explicit-chance research host
        // is not a live Unity save. Generate the same deterministic legal game
        // on both arms; retain and compare the resulting permitted root hash.
        var h=new Host((int)fixture["seed"]);var random=new System.Random((int)fixture["seed"]+701);
        for(int i=0;i<4000;i++) {
            var c=(JObject)Wire.Token(h.PublicContext());
            if(h.Phase=="action" && h.Game.TurnFlow.CurrentAction==null && (string)c["era"]==(string)fixture["era"] &&
                (int)c["round"]==(int)fixture["round"] && (int)c["turn"]==2 && h.Remaining==2)return h;
            if(h.Terminal)throw new Exception("matched fixture target not reached");
            var choices=h.Choices();var legal=choices;
            if(h.Phase=="action") {
                string[] kinds=h.Player.Money<12?new[]{"loan","sell","build","network","develop","pass"}:new[]{"sell","build","network","develop","loan","pass"};
                // Prefer sales and affordable industry, with some network and
                // development diversity. All selected IDs remain rule-checked.
                string preferred=random.Next(4)==0?"network":kinds.First(k=>choices.Any(x=>x.kind==k));
                var options=choices.Where(x=>x.kind==preferred).ToList();if(options.Count>0)legal=options;
            } else if(choices.Any(x=>x.kind=="finish"))legal=choices.Where(x=>x.kind=="finish").ToList();
            h.Apply(h.Revision,legal[random.Next(legal.Count)].id);
        }
        throw new Exception("matched fixture generation guard");
    }
    IEnumerator NativeMatchedCases()
    {
        if(Environment.GetCommandLineArgs().Contains("--brass-search-v2-test")) {
            var focused=(IEnumerator)GetType().GetMethod("SearchV2Cases",ReflectionAccess.Flags).Invoke(this,null);
            while(focused.MoveNext())yield return focused.Current;yield break;
        }
        var inputs=JArray.Parse(File.ReadAllText(Environment.GetEnvironmentVariable("BRASS_L14_LIVE_FIXTURES")));
        string destination=Environment.GetEnvironmentVariable("BRASS_L14_LIVE_RESULT");
        yield return SceneManager.LoadSceneAsync("GameScene");yield return null;router.ReloadPrefabs();
        var view=(Frameworks.View.Game.PlayAiTurnView)router.GetView<IPlayAiTurnView>();
        var turn=new PlayAiTurnUseCase(new PlayAITurnPresenter(router),history);
        router.UseCases[typeof(PlayAiTurnUseCase)]=turn;view.Controller=new PlayAiTurnController(turn);
        var records=new JArray();
        int serial=0;
        foreach(JObject fixture in inputs) {
            string cache=Path.Combine(Environment.GetEnvironmentVariable("BRASS_L14_LIVE_CACHE"),(string)fixture["name"]+".json");
            var serializer=new Frameworks.Persistence.Serialization.JsonSerializer();BoardGameRulesGame game;
            if(File.Exists(cache)) {
                List<IEvent> replay;using(var stream=File.OpenRead(cache))replay=serializer.Deserialize<List<IEvent>>(stream);
                game=new BoardGameRulesGame(((GameStarted)replay[0]).GameStartConfiguration);
                foreach(var e in replay)game.ApplyEvent(e);
            }else {
                game=MatchedHost(fixture).Game;
                using(var stream=File.Create(cache))serializer.Serialize(stream,game.GetUncommittedEvents().ToList());
            }
            var metadata=new GameMetadata(game.GameStartConfiguration,GameType.Local){GameId="l14-"+Guid.NewGuid().ToString("N")};
            foreach(var start in game.GameStartConfiguration.Players)metadata.JoinGame(new PlayerMetadata{PlayerStartConfiguration=start,AiType=AIType.Hard,NickName="Alpha Test",PlayerNetworkId="local-"+start.Color});
            var app=new BrassApplicationGame(game,metadata);var state=NativeGameState.Create(metadata,p=>"easy");
            state["game_id"]=metadata.GameId;state["ready"]=true;SaveExtensions.Set(metadata,state);
            history.Repository.Save(app);history.SetGame(app);
            int actor=(int)game.GameProgressInformation.CurrentPlayer.Color,revision=game.Revision,legalCount=Host.Capture(app).Choices().Count;
            var events=LocalGameState.History(app,history.Repository.Advanced.GameStorage.LoadEvents(metadata.GameId,0,revision));
            var input=NativeInput.Capture(app,events,actor,new JArray());
            var actual=(JObject)input["view"].DeepClone();string rootHash=Wire.Hash(actual);
            if(serial==0) {
                var cold=Stopwatch.StartNew();var warm=new NativeJob(NativeInput.Request(input,"warm-"+Guid.NewGuid().ToString("N"),metadata.GameId,1,4000),new Stopwatch(),4000);
                var execution=AiExecutionScheduler.Submit(warm);while(!execution.Completion.IsCompleted)yield return null;
                Check(execution.Failure==null,"matched warmup completes: "+execution.Failure);
                File.WriteAllText(destination+".warmup.json",new JObject{["ms"]=cold.Elapsed.TotalMilliseconds,["readiness"]=NativeProcess.Shared().Readiness}.ToString());
                history.SetGame(app);
            }
            var process=Process.GetCurrentProcess();var worker=Get<Process>(NativeProcess.Shared(),"process");
            process.Refresh();worker.Refresh();var cpu=process.TotalProcessorTime;var nativeCpu=worker.TotalProcessorTime;
            long nativeBefore=worker.WorkingSet64;
            var timer=Stopwatch.StartNew();long previous=Stopwatch.GetTimestamp();
            int frames=0;double largest=0;var commits=new JArray();int last=revision;
            router.ShowView<IPlayAiTurnView>();turn.PlayAITurn(false);
            while(LocalTurnBridge.IsThinking && timer.Elapsed.TotalSeconds<15) {
                yield return null;long now=Stopwatch.GetTimestamp();largest=Math.Max(largest,(now-previous)*1000.0/Stopwatch.Frequency);previous=now;frames++;
                if(app.Game.Revision!=last){commits.Add(new JObject{["ms"]=timer.Elapsed.TotalMilliseconds,["revision"]=app.Game.Revision,["remaining_actions"]=app.Game.TurnFlow.GetRemainingActionsCount()});last=app.Game.Revision;}
            }
            timer.Stop();
            double ordinaryClockMs=NativeTurnClock.For(app,actor,false).Clock.Elapsed.TotalMilliseconds;
            process.Refresh();worker.Refresh();
            double unityCpuMs=(process.TotalProcessorTime-cpu).TotalMilliseconds,nativeCpuMs=(worker.TotalProcessorTime-nativeCpu).TotalMilliseconds;
            long nativeAfter=worker.WorkingSet64,nativePeak=worker.PeakWorkingSet64;
            Check(!LocalTurnBridge.IsThinking && app.Game.TurnFlow.GetRemainingActionsCount()==0 && app.Game.Revision>revision,"matched complete turn: "+fixture["name"]);
            Check(AiExecutionScheduler.IsIdle && !AiExecutionResources.IsComputing,"matched owners and validation retired");
            var loaded=history.Repository.Load(metadata.GameId);
            Check(loaded.Game.Revision==revision,"matched AI turn leaves its pre-AI checkpoint intact");
            records.Add(new JObject{["fixture"]=fixture["name"],["partition"]=fixture["partition"],["era"]=actual["context"]["era"],
                ["root_sha256"]=rootHash,["permitted_root"]=actual,["legal_count"]=legalCount,
                ["whole_turn_ms"]=4000,["delivery_ms"]=timer.Elapsed.TotalMilliseconds,["main_process_cpu_ms"]=unityCpuMs,
                ["native_process_cpu_ms"]=nativeCpuMs,["total_cpu_ms"]=unityCpuMs+nativeCpuMs,
                ["native_retired_bytes_before"]=nativeBefore,["native_retired_bytes_after"]=nativeAfter,["native_peak_bytes"]=nativePeak,
                ["ordinary_clock_ms"]=ordinaryClockMs,["preparation_and_dispatch_ms"]=timer.Elapsed.TotalMilliseconds-ordinaryClockMs,
                ["frames"]=frames,["max_frame_ms"]=largest,["commits"]=commits,["complete"]=true});
            File.WriteAllText(destination,new JObject{["passed"]=true,["complete"]=false,["rows"]=records}.ToString());serial++;
        }
        Check(errors.Count==0,"matched run has no Unity errors");
        File.WriteAllText(destination,new JObject{["passed"]=true,["complete"]=true,["rows"]=records}.ToString());
    }
}
