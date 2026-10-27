// Test-only event journals split a complete live-search match into short player
// invocations. They are trusted research fixtures, never policy inputs or saves.
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
using BrassPatch.Liquidation;
using BrassPatch.Persistence;
using BrassApplication.AI;
using BrassApplication.Game;
using BrassApplication.UseCases.Game.PlayAiTurn;
using BrassApplication.UseCases.Game.PlayGame;
using BoardGameRules.Entities.Common;
using BoardGameRules.Entities.Game;
using BoardGameRules.Entities.EventSourcing.Events;
using BoardGameRules.Entities.GamePhases.Events;
using InterfaceAdapters.Game.PlayAiTurn;
using Newtonsoft.Json.Linq;
using UnityEngine;

public sealed partial class LocalAiUnityDriver
{
    IEnumerator ReleaseSegment(PlayAiTurnUseCase turn,bool nn)
    {
        string prior=Environment.GetEnvironmentVariable("BRASS_RELEASE_SEGMENT_IN");
        string destination=Environment.GetEnvironmentVariable("BRASS_RELEASE_SEGMENT_OUT");
        Check(!String.IsNullOrEmpty(destination) && !File.Exists(destination),"fresh segment journal");
        var serializer=new Frameworks.Persistence.Serialization.JsonSerializer();
        JObject journal=null;BoardGameRulesGame game;GameMetadata metadata;JObject state;
        int completed=0;var rows=new JArray();
        if(!String.IsNullOrEmpty(prior)) {
            journal=JObject.Parse(File.ReadAllText(prior));Check((bool)journal["nn"]==nn && !(bool)journal["terminal"],"matching unfinished journal");
            List<IEvent> replay;using(var s=File.OpenRead(prior+".events"))replay=serializer.Deserialize<List<IEvent>>(s);
            game=new BoardGameRulesGame(((GameStarted)replay[0]).GameStartConfiguration);foreach(var e in replay)game.ApplyEvent(e);
            using(var s=File.OpenRead(prior+".metadata"))metadata=serializer.Deserialize<GameMetadata>(s);
            state=(JObject)journal["state"].DeepClone();completed=(int)journal["turns"];
            Check(game.Revision==(int)journal["revision"],"event replay preserves exact revision");
        } else {
            var config=new GameStartConfiguration((NumberOfPlayers)(nn?4:2)){Seed=nn?96261:96262,IsIntroductoryGame=false};
            metadata=new GameMetadata(config,GameType.Local){GameId="release-live-"+Guid.NewGuid().ToString("N")};int index=0;
            foreach(var p in config.Players)metadata.JoinGame(new PlayerMetadata{PlayerStartConfiguration=p,
                AiType=index++==0?AIType.Human:AIType.Hard,NickName=nn?"Alpha Test":"V4",PlayerNetworkId="release-"+p.Color});
            state=nn?NativeGameState.Create(metadata,p=>"easy"):LocalGameState.Create(metadata,p=>"balanced",p=>"easy");
            state["game_id"]=metadata.GameId;state["ready"]=true;game=new BoardGameRulesGame(config);game.StartGame();
        }
        SaveExtensions.Set(metadata,state);var app=new BrassApplicationGame(game,metadata);
        // This is fixture setup before starting the segment. Gameplay saving
        // below still runs solely through human End, verified after each action.
        history.Repository.Save(app);history.SetGame(app);int lastSaved=game.Revision;
        var output=new ReleaseOutput(router,history);var play=new PlayGameUseCase(output,history,null,new Analytics());
        router.UseCases[typeof(PlayGameUseCase)]=play;
        int humanTurns=0,aiTurns=0,actions=0,debt=0;float start=Time.realtimeSinceStartup;
        for(int i=0;i<8 && !game.GameProgressInformation.IsGameOver;i++) {
            var player=game.GameProgressInformation.CurrentPlayer;bool human=metadata.GetPlayerMetadataByColor(player.Color).AiType==AIType.Human;
            int revision=game.Revision,remaining=game.TurnFlow.GetRemainingActionsCount();string era=game.GameProgressInformation.CurrentGameEra.ToString();
            var clock=Stopwatch.StartNew();var replies=new JArray();
            if(human) {
                while(game.TurnFlow.GetRemainingActionsCount()>0){game.TurnFlow.StartPassAction().Pass(player.Hand[0],player);game.TurnFlow.ConfirmCurrentAction();}
                humanTurns++;
            } else {
                // No remaining-clock injection: real 4-second ordinary turns,
                // original action-two clock and production scheduler/commit.
                router.ShowView<IPlayAiTurnView>();turn.PlayAITurn(false);var jobs=new List<NativeJob>();
                while((LocalTurnBridge.IsThinking || !AiExecutionScheduler.IsIdle) && clock.Elapsed.TotalSeconds<14) {
                    var bridge=typeof(LocalTurnBridge).GetField("active",BrassPatch.ReflectionAccess.Flags).GetValue(null);
                    if(nn && bridge!=null){var job=Get<NativeJob>(bridge,"nativeBackend");if(job!=null && !jobs.Contains(job))jobs.Add(job);}
                    yield return null;
                }
                Check(!LocalTurnBridge.IsThinking && AiExecutionScheduler.IsIdle,"live turn retires within watchdog");
                Check(game.TurnFlow.GetRemainingActionsCount()==0,"live search completes both actions");
                foreach(var job in jobs){Check(job.Result!=null && (bool)job.Result["retired"],"native reply retired");replies.Add(job.Result.DeepClone());
                    File.AppendAllText(Output+".permitted.jsonl",new JObject{["request"]=Get<JObject>(job,"request"),["response"]=job.Result}.ToString(Newtonsoft.Json.Formatting.None)+"\n");}
                aiTurns++;
            }
            clock.Stop();actions+=remaining;
            Check(history.Repository.Load(metadata.GameId).Game.Revision==lastSaved,"AI/human actions do not save");
            int beforeSaves=output.Saves;play.EndTurn();
            Check(output.Saves==beforeSaves+(human?1:0),"exactly one checkpoint after human End");
            if(human)lastSaved=game.Revision;
            if(LiquidationFixes.IsPending(game.SellIndustryFlow)) {
                debt++;float until=Time.realtimeSinceStartup+12;
                while((LocalTurnBridge.IsThinking || !AiExecutionScheduler.IsIdle) && Time.realtimeSinceStartup<until)yield return null;
                Check(!LiquidationFixes.IsPending(game.SellIndustryFlow) && AiExecutionScheduler.IsIdle,"debt settles and retires");
            }
            Check(history.Repository.Load(metadata.GameId).Game.Revision==lastSaved,"AI End/debt never saves");
            rows.Add(new JObject{["turn"]=++completed,["era"]=era,["actor"]=(int)player.Color,["human"]=human,["before"]=revision,["after"]=game.Revision,
                ["allowance_ms"]=human?0:4000,["delivery_ms"]=clock.Elapsed.TotalMilliseconds,["actions"]=remaining,["replies"]=replies});
            Check(Time.realtimeSinceStartup-start<70,"short segment bound");yield return null;
        }
        Check(errors.Count==0,"no supported live game errors: "+String.Join("\n",errors));
        var events=LocalGameState.History(app,history.Repository.Advanced.GameStorage.LoadEvents(metadata.GameId,0,game.Revision)).ToList();
        using(var s=File.Create(destination+".events"))serializer.Serialize(s,events);
        using(var s=File.Create(destination+".metadata"))serializer.Serialize(s,metadata);
        var result=new JObject{["passed"]=true,["mode"]=nn?"nn-live":"v4-live",["nn"]=nn,["turns"]=completed,["revision"]=game.Revision,
            ["prior"]=prior,["terminal"]=game.GameProgressInformation.IsGameOver,["trace"]=rows,["actions"]=actions,["human_turns"]=humanTurns,["ai_turns"]=aiTurns,
            ["saves_after_fixture_setup"]=output.Saves,["fixture_setup_saves"]=1,["debt_phases"]=debt,["seconds"]=Time.realtimeSinceStartup-start,
            ["state"]=LocalGameState.Read(metadata),["scores"]=new JArray(game.PlayersTrack.Players.Select(p=>new JObject{["seat"]=(int)p.Color,["vp"]=p.VP,["money"]=p.Money}))};
        File.WriteAllText(destination,result.ToString());File.WriteAllText(Output+".release.json",result.ToString());
    }
}
