using System;
using System.IO;
using System.Linq;
using Newtonsoft.Json.Linq;
using BrassHost;
using BrassLegacyAI;
using BoardGameRules.Entities.Game;
using BoardGameRules.Entities.Common;
using BrassApplication.AI;
using BrassApplication.AI.AiBehaviors;
using BrassApplication.AI.RuleBasedSystem;
using BrassApplication.AI.RuleBasedSystem.RuleSelectors;
using BrassApplication.Game;
using Frameworks.Persistence.GameFactory;
using Frameworks.Persistence.Repository;
using Frameworks.Persistence.RepositoryBackends.InMemory;
public static class LiveMatchChecks
{
    public static object Run(string root)
    {
        var composition=JObject.Parse(File.ReadAllText(Path.Combine(root,"configs/contracts/v0.2.0/card-composition.json")));
        var profile=Profile.Read(JObject.Parse(File.ReadAllText(Path.Combine(root,"configs/legacy-ai/balanced-v1.json"))));profile.milliseconds=0;profile.work_limit=1;
        var profiles=new[]{"balanced-v2","cotton-v2","supplier-v2","pottery-flex-s2-v2"}.Select(n=>{var p=Profile.Read(JObject.Parse(File.ReadAllText(Path.Combine(root,"configs/legacy-ai/"+n+".json"))));p.milliseconds=0;p.work_limit=1;return p;}).ToArray();
        var intents=new IntentState[4];
        var config=new GameStartConfiguration(NumberOfPlayers.Four){Seed=2709};var metadata=new GameMetadata(config,GameType.Local);
        foreach(var start in config.Players)
        {
            var p=new PlayerMetadata{PlayerStartConfiguration=start,PlayerNetworkId="scripted-"+start.Color,NickName="Scripted "+start.Color,AiType=(int)start.Color==0?AIType.Human:AIType.Hard};p.AIBehaviour=AiBehaviorFactory.CreateAiBehavior(p);metadata.JoinGame(p);
        }
        var repository=new BaseRepository(new InMemoryGameStorage(),new InMemorySnapshotStorage(),new BrassBirminghamFactory());
        var app=repository.Create(metadata);app.Game.StartGame();repository.Save(app);
        int turns=0,actions=0,reloads=0,debts=0,railTurns=0;JObject finalProjection=null;
        var scriptedHuman=new HardAiBehavior("scripted-human");
        while(!app.Game.GameProgressInformation.IsGameOver)
        {
            if(turns>=80)throw new Exception("live_progress_limit");
            int seat=(int)app.Game.GameProgressInformation.CurrentPlayer.Color;
            if(app.Game.GameProgressInformation.CurrentGameEra.ToString()=="RailEra")railTurns++;
            while(app.Game.TurnFlow.GetRemainingActionsCount()>0)
            {
                if(seat==0){scriptedHuman.GetNextAIMove(app)();app.Game.TurnFlow.ConfirmCurrentAction();actions++;continue;}
                var events=repository.Advanced.GameStorage.LoadEvents(app.Metadata.GameId,0,app.Game.Revision);
                var live=Host.Capture(app,events);var v=(JObject)Wire.Token(live.Observe(seat));
                var d=new Planner(composition).Decide(v,live.Choices(),profiles[seat],intent:intents[seat]);intents[seat]=d.selected.intent;
                foreach(var path in d.selected.actions)
                {
                    live=Host.Capture(app,repository.Advanced.GameStorage.LoadEvents(app.Metadata.GameId,0,app.Game.Revision));
                    live.LiveCommit(path,composition);actions++;
                }
            }
            if(app.Game.GameProgressInformation.CurrentGameEra.ToString()=="RailEra" && app.Game.GameProgressInformation.CurrentRoundNumber==8 && app.Game.GameProgressInformation.CurrentTurnNumber==4)
                finalProjection=Evaluator.KnownFinalScore((JObject)Wire.Token(Host.Capture(app,repository.Advanced.GameStorage.LoadEvents(app.Metadata.GameId,0,app.Game.Revision)).Observe(seat)));
            app.Game.GameFlow.EndTurn();turns++;
            if(app.Game.GameFlow.IsLastTurnOver())
            {
                if(app.Game.GameFlow.IsSellIndustryNecessary())
                {debts++;var selector=RuleSelectorFactory.CreatePrioritySelector(RuleSetFactory.CreateSellIndustryRuleSet());selector.Select(app).MakeAction(app).Invoke();}
                app.Game.GameFlow.EndRound();
            }
            if(!app.Game.GameProgressInformation.IsGameOver && turns%4==0)
            {
                var before=Wire.Token(Host.Capture(app,repository.Advanced.GameStorage.LoadEvents(app.Metadata.GameId,0,app.Game.Revision)).Observe((int)app.Game.GameProgressInformation.CurrentPlayer.Color));
                repository.Save(app);app=repository.Load(app.Metadata.GameId);
                var after=Wire.Token(Host.Capture(app,repository.Advanced.GameStorage.LoadEvents(app.Metadata.GameId,0,app.Game.Revision)).Observe((int)app.Game.GameProgressInformation.CurrentPlayer.Color));
                if(!JToken.DeepEquals(before,after))throw new Exception("live_reload_observation_parity");reloads++;
            }
        }
        if(turns!=64 || actions!=124 || railTurns!=32 || reloads!=15)throw new Exception("live_match_coverage");
        foreach(var player in app.Game.PlayersTrack.Players)
            if((int)finalProjection["players"][(int)player.Color]["vp"]!=player.VP)throw new Exception("final_projection:"+player.Color+":"+finalProjection["players"][(int)player.Color]["vp"]+"!="+player.VP);
        return new{passed=true,turns,actions,rail_turns=railTurns,reloads,debts,final_projection_seats=4,manual_seat="scripted with stock legal actions; not a rendered human UI acceptance test"};
    }
}
