using System;
using System.Collections;
using System.Linq;
using System.Reflection;
using System.Threading.Tasks;
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
using BrassApplication.UseCases.Common.SaveGame;
using BoardGameRules.Entities.Common;
using BoardGameRules.Entities.Game;
using InterfaceAdapters.Game.PlayAiTurn;
using InterfaceAdapters.Routing;
using Newtonsoft.Json.Linq;
using UnityEngine;

public sealed partial class LocalAiUnityDriver
{
    BrassApplicationGame ValidationGame(int seed)
    {
        var config=new GameStartConfiguration((NumberOfPlayers)4){Seed=seed,IsIntroductoryGame=false};
        var metadata=new GameMetadata(config,GameType.Local);
        foreach(var start in config.Players)metadata.JoinGame(new PlayerMetadata{PlayerStartConfiguration=start,AiType=AIType.Hard,NickName="Alpha Test",PlayerNetworkId="local-"+start.Color});
        var state=NativeGameState.Create(metadata,p=>"easy");SaveExtensions.Set(metadata,state);
        var app=history.Repository.Create(metadata);app.Game.StartGame();state["game_id"]=app.Metadata.GameId;state["ready"]=true;SaveExtensions.Set(app.Metadata,state);
        while(app.Game.GameProgressInformation.CurrentRoundNumber==1) {
            var player=app.Game.GameProgressInformation.CurrentPlayer;
            app.Game.TurnFlow.StartPassAction().Pass(player.Hand[0],player);app.Game.TurnFlow.ConfirmCurrentAction();app.Game.GameFlow.EndTurn();
            if(app.Game.GameFlow.IsLastTurnOver())app.Game.GameFlow.EndRound();
        }
        history.Repository.Save(app);history.SetGame(app);return app;
    }
    IEnumerator NativeValidationLifecycleCases(PlayAiTurnUseCase turn)
    {
        var field=typeof(LocalTurnBridge).GetField("validationExecution",ReflectionAccess.Flags);
        if(field==null)yield break; // The sealed baseline has no detached validator.
        var enabled=typeof(LocalTurnBridge).Assembly.GetType("BrassLegacyAI.Native.NativeValidation").GetField("Enabled");
        if(enabled!=null && !(bool)enabled.GetRawConstantValue()) {HumanCheckpoint(93812);yield break;}
        foreach(string scenario in new[]{"cancel","load","actor","revision"}) {
            var app=HumanCheckpoint(93811);int revision=app.Game.Revision;int actor=(int)app.Game.GameProgressInformation.CurrentPlayer.Color;
            router.ShowView<IPlayAiTurnView>();turn.PlayAITurn(false);
            object bridge=null;AiExecutionJob job=null;float until=Time.realtimeSinceStartup+15;
            while(Time.realtimeSinceStartup<until) {
                bridge=typeof(LocalTurnBridge).GetField("active",ReflectionAccess.Flags).GetValue(null);
                if(bridge!=null)job=(AiExecutionJob)field.GetValue(bridge);
                if(job!=null)break;yield return null;
            }
            Check(job!=null && app.Game.Revision==revision,"validation handoff reached before mutation: "+scenario);
            bool inFlight=!job.Completion.IsCompleted;
            if(scenario=="cancel") {LocalTurnBridge.CancelActive();LocalTurnBridge.CancelActive();}
            if(scenario=="actor")app.Game.GameProgressInformation.CurrentPlayer=app.Game.PlayersTrack.Players.First(p=>(int)p.Color!=actor);
            if(scenario=="revision") {
                app.Game.ApplyEvent(new BoardGameRules.Entities.NumericalResources.Events.MoneySpent((BoardGameRules.Entities.Players.PlayerColor)actor,1));revision=app.Game.Revision;
            }
            BrassApplicationGame loaded=null;
            if(scenario=="load") {
                // Loading cancels pending AI work. It does not save that work:
                // the only checkpoint was written after the human clicked End.
                loaded=history.Repository.Load(app.Metadata.GameId);
                Check(loaded.Game.Revision==revision,"load uses the human End checkpoint, with no in-flight AI save");
                history.SetGame(loaded);
            }
            until=Time.realtimeSinceStartup+10;
            while((LocalTurnBridge.IsThinking || !AiExecutionScheduler.IsIdle) && Time.realtimeSinceStartup<until)yield return null;
            Check(!LocalTurnBridge.IsThinking && job.ResourcesReleased && AiExecutionScheduler.IsIdle && app.Game.Revision==revision,"stale/cancelled validation retires without commit: "+scenario+", in-flight="+inFlight);
            if(loaded!=null) {
                Check(loaded.Game.Revision==revision,"same-revision load does not reuse old validation");
                router.ShowView<IPlayAiTurnView>();turn.PlayAITurn(false);
                var fresh=typeof(LocalTurnBridge).GetField("active",ReflectionAccess.Flags).GetValue(null);
                Check(fresh!=null && !System.Object.ReferenceEquals(fresh,bridge),"load creates a fresh bridge lifetime");
                Check(Get<double>(fresh,"budget")==4000,"discarded AI work restarts from the saved pre-AI boundary");
                Check(field.GetValue(fresh)==null,"saved validation is never authority after load");
                LocalTurnBridge.CancelActive();until=Time.realtimeSinceStartup+10;
                while(!AiExecutionScheduler.IsIdle && Time.realtimeSinceStartup<until)yield return null;
                Check(loaded.Game.Revision==revision && AiExecutionScheduler.IsIdle,"resumed cancelled search retires without commit");
            }
        }
        // Subsequent retry tests need a fresh turn, not the expired clock of a
        // fixture that was set aside while these real-search cases ran.
        HumanCheckpoint(93812);
    }

    BrassApplicationGame HumanCheckpoint(int seed)
    {
        var app=ValidationGame(seed);
        var human=app.Game.GameProgressInformation.CurrentPlayer;
        app.Metadata.GetPlayerMetadataByColor(human.Color).AiType=AIType.Human;
        var state=NativeGameState.Create(app.Metadata,p=>"easy");state["ready"]=true;SaveExtensions.Set(app.Metadata,state);
        int diskRevision=history.Repository.Load(app.Metadata.GameId).Game.Revision;
        while(app.Game.TurnFlow.GetRemainingActionsCount()>0) {
            app.Game.TurnFlow.StartPassAction().Pass(human.Hand[0],human);app.Game.TurnFlow.ConfirmCurrentAction();
        }
        Check(history.Repository.Load(app.Metadata.GameId).Game.Revision==diskRevision,"human actions do not save before End is clicked");
        var output=new HumanCheckpointOutput(router,history);
        var play=new PlayGameUseCase(output,history,null,new Analytics());
        play.EndTurn();int checkpoint=app.Game.Revision;
        Check(output.Saves==1 && output.Starts==0,"human End writes once before dispatching the next player");
        play.EndTurn();Check(output.Saves==1 && app.Game.Revision==checkpoint,"duplicate End cannot save or advance during the checkpoint write");
        var loaded=history.Repository.Load(app.Metadata.GameId);
        Check(loaded.Game.Revision==checkpoint && app.Game.GameProgressInformation.CurrentPlayer.Color!=human.Color,"checkpoint includes the completed human turn and next actor");
        output.Complete();Check(output.Starts==1,"AI starts only after the human checkpoint completes");
        var ai=app.Game.GameProgressInformation.CurrentPlayer;
        while(app.Game.TurnFlow.GetRemainingActionsCount()>0) {
            app.Game.TurnFlow.StartPassAction().Pass(ai.Hand[0],ai);app.Game.TurnFlow.ConfirmCurrentAction();
        }
        play.EndTurn();
        int aiCheckpoint=app.Game.Revision;
        Check(output.Saves==2 && output.Starts==1 && history.Repository.Load(app.Metadata.GameId).Game.Revision==aiCheckpoint,"AI End writes one checkpoint before dispatching the next player");
        play.EndTurn();Check(output.Saves==2 && app.Game.Revision==aiCheckpoint,"duplicate AI End cannot save or advance during the checkpoint write");
        output.Complete();Check(output.Starts==2,"next player starts only after the AI checkpoint completes");
        history.SetGame(loaded);return loaded;
    }
    sealed class HumanCheckpointOutput : IPlayGameOutput
    {
        readonly IRouter _router;readonly BrassApplication.GameHistory.BrassGameHistory history;
        ISaveGameDelegate callback;public int Saves,Starts;
        public HumanCheckpointOutput(IRouter router,BrassApplication.GameHistory.BrassGameHistory history){_router=router;this.history=history;}
        public void SaveGame(BrassApplicationGame game,ISaveGameDelegate value){history.Repository.Save(game);Saves++;callback=value;}
        public void Complete(){var value=callback;callback=null;value.SaveGameComplete(true);}
        public void StartHumanTurn(){Starts++;}public void StartAITurn(PlayerMetadata p){Starts++;}public void StartSellingIndustries(){Starts++;}
        public void StartGameReview(PlayerMetadata p){Starts++;}public void StartNetworkHumanTurn(){throw new Exception("network fixture");}
        public void StartNetworkSpectatorTurn(PlayerMetadata p){throw new Exception("network fixture");}
    }
}
