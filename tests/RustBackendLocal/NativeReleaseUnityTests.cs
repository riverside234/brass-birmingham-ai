using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using BrassHost;
using BrassLegacyAI;
using BrassLegacyAI.Native;
using BrassLegacyAI.Execution;
using BrassPatch;
using BrassPatch.Persistence;
using BrassPatch.Liquidation;
using BrassApplication.AI;
using BrassApplication.Game;
using BrassApplication.UseCases.Common.SaveGame;
using BrassApplication.UseCases.Game.PlayAiTurn;
using BrassApplication.UseCases.Game.PlayGame;
using BoardGameRules.Entities.Common;
using BoardGameRules.Entities.Game;
using BoardGameRules.Entities.Players;
using InterfaceAdapters.Game.PlayAiTurn;
using InterfaceAdapters.Routing;
using Frameworks.View.Game;
using Newtonsoft.Json.Linq;
using UnityEngine;
using UnityEngine.SceneManagement;

public sealed partial class LocalAiUnityDriver
{
    // Complete-game mechanics smoke. The NN fixtures enter each ordinary turn
    // with 450 ms remaining, through the existing old-save clock adapter. This
    // never changes the shipped 4/8/10/12 s catalog and is not a strength test.
    IEnumerator NativeReleaseCases()
    {
        string mode=Environment.GetCommandLineArgs().Single(a=>a.StartsWith("--brass-release-case=")).Split('=')[1];
        router.HideView<InterfaceAdapters.Common.CreateGame.ICreateOfflineGameView>();
        yield return SceneManager.LoadSceneAsync("GameScene");yield return null;router.ReloadPrefabs();
        var turn=new PlayAiTurnUseCase(new PlayAITurnPresenter(router),history);
        router.UseCases[typeof(PlayAiTurnUseCase)]=turn;
        ((PlayAiTurnView)router.GetView<IPlayAiTurnView>()).Controller=new PlayAiTurnController(turn);
        if(mode=="failures") {
            var cases=ReleaseFailureCases(turn);while(cases.MoveNext())yield return cases.Current;yield break;
        }
        if(mode.EndsWith("-live")) {
            var cases=ReleaseSegment(turn,mode=="nn-live");while(cases.MoveNext())yield return cases.Current;yield break;
        }
        bool nn=mode.StartsWith("nn-");bool human=mode!="nn-ai";bool noHumanReview=false;
        int count=mode=="v4-two"?2:mode=="v4-three"?3:4;
        var config=new GameStartConfiguration((NumberOfPlayers)count){Seed=nn?(human?95151:95152):95153+count,IsIntroductoryGame=false};
        var metadata=new GameMetadata(config,GameType.Local);int index=0;
        foreach(var start in config.Players)metadata.JoinGame(new PlayerMetadata{PlayerStartConfiguration=start,
            AiType=human && index++==0?AIType.Human:AIType.Hard,NickName=nn?"Alpha Test":"V4",PlayerNetworkId="release-"+start.Color});
        Func<PlayerMetadata,string> difficulty=p=>DifficultyCatalog.Keys[p.PlayerStartConfiguration.SittingOrder%4];
        var state=nn?NativeGameState.Create(metadata,difficulty):LocalGameState.Create(metadata,p=>"balanced",difficulty);
        SaveExtensions.Set(metadata,state);var app=history.Repository.Create(metadata);app.Game.StartGame();
        state["game_id"]=app.Metadata.GameId;state["ready"]=true;SaveExtensions.Set(app.Metadata,state);
        history.Repository.Save(app);history.SetGame(app);
        var output=new ReleaseOutput(router,history);var play=new PlayGameUseCase(output,history,null,new Analytics());
        router.UseCases[typeof(PlayGameUseCase)]=play;
        int turns=0,aiTurns=0,humanTurns=0,actions=0,debtPhases=0,lastSaved=app.Game.Revision;var eras=new HashSet<string>();
        var rows=new JArray();float startedGame=Time.realtimeSinceStartup;
        while(!app.Game.GameProgressInformation.IsGameOver) {
            Check(turns++<80,"complete game turn bound");
            var progress=app.Game.GameProgressInformation;string era=progress.CurrentGameEra.ToString();eras.Add(era);
            var player=progress.CurrentPlayer;bool isHuman=app.Metadata.GetPlayerMetadataByColor(player.Color).AiType==AIType.Human;
            int before=app.Game.Revision,remaining=app.Game.TurnFlow.GetRemainingActionsCount();
            if(nn && !isHuman) {
                var events=LocalGameState.History(app,history.Repository.Advanced.GameStorage.LoadEvents(app.Metadata.GameId,0,before));
                var input=NativeInput.Capture(app,events,(int)player.Color,new JArray());
                NativeGameState.Record(app.Metadata,(JObject)input["view"],LocalGameState.HistoryHash(events,before),450,new JArray());
                router.ShowView<IPlayAiTurnView>();turn.PlayAITurn(false);
                float until=Time.realtimeSinceStartup+8;var jobs=new List<NativeJob>();
                while(LocalTurnBridge.IsThinking && Time.realtimeSinceStartup<until) {
                    var bridge=typeof(LocalTurnBridge).GetField("active",ReflectionAccess.Flags).GetValue(null);
                    if(bridge!=null){var job=Get<NativeJob>(bridge,"nativeBackend");if(job!=null && !jobs.Contains(job))jobs.Add(job);}
                    yield return null;
                }
                Check(!LocalTurnBridge.IsThinking && AiExecutionScheduler.IsIdle,"NN owners and validation retire each turn");
                Check(app.Game.TurnFlow.GetRemainingActionsCount()==0,"NN completes both actions: "+mode+" "+era+" turn "+turns);
                foreach(var job in jobs) {
                    Check(job.Result!=null && (bool)job.Result["retired"],"live NN job supplies a retired reply");
                    File.AppendAllText(Output+".permitted.jsonl",new JObject{["request"]=Get<JObject>(job,"request"),["response"]=job.Result}.ToString(Newtonsoft.Json.Formatting.None)+"\n");
                }
                aiTurns++;
            } else {
                // Scripted human input / V4 family settlement control. Do not
                // report these actions as decisions made by the V4 search.
                while(app.Game.TurnFlow.GetRemainingActionsCount()>0) {
                    app.Game.TurnFlow.StartPassAction().Pass(player.Hand[0],player);app.Game.TurnFlow.ConfirmCurrentAction();
                }
                if(isHuman)humanTurns++;else aiTurns++;
            }
            actions+=remaining;
            Check(history.Repository.Load(app.Metadata.GameId).Game.Revision==lastSaved,"no action writes a save before the completed turn");
            int savesBefore=output.Saves;
            try {play.EndTurn();}
            catch(InvalidOperationException error) {
                if(human || !app.Game.GameProgressInformation.IsGameOver || error.Message!="Sequence contains no matching element")throw;
                // The shipped UI requires a human and its terminal presenter
                // selects that human. This all-AI research fixture has none.
                // Domain settlement has finished; skip ONLY that known review
                // lookup, without changing any seat or save to make it pass.
                noHumanReview=true;
            }
            Check(output.Saves==savesBefore+1,"each completed human or AI turn writes one checkpoint");
            lastSaved=history.Repository.Load(app.Metadata.GameId).Game.Revision;
            if(LiquidationFixes.IsPending(app.Game.SellIndustryFlow)) {
                debtPhases++;float debtUntil=Time.realtimeSinceStartup+10;
                while((LocalTurnBridge.IsThinking || !AiExecutionScheduler.IsIdle) && Time.realtimeSinceStartup<debtUntil)yield return null;
                Check(!LocalTurnBridge.IsThinking && AiExecutionScheduler.IsIdle,"round-end debt owners retire");
            }
            Check(history.Repository.Load(app.Metadata.GameId).Game.Revision==lastSaved && output.Saves==savesBefore+1,"debt continuation does not add another checkpoint");
            Check(!LiquidationFixes.IsPending(app.Game.SellIndustryFlow),"round-end debt completed");
            rows.Add(new JObject{["turn"]=turns,["era"]=era,["actor"]=(int)player.Color,["human"]=isHuman,
                ["before"]=before,["after"]=app.Game.Revision,["allowance_ms"]=isHuman?0:DifficultyCatalog.Milliseconds(difficulty(app.Metadata.GetPlayerMetadataByColor(player.Color)))});
            Check(Time.realtimeSinceStartup-startedGame<75,"bounded complete-game smoke");yield return null;
        }
        Check(eras.Count==2 && actions>60,"complete game traverses Canal and Rail");
        Check(!LiquidationFixes.IsPending(app.Game.SellIndustryFlow),"final settlement complete");
        Check(output.Saves==humanTurns+aiTurns,"one save per completed human or AI turn across entire game");
        var saved=history.Repository.Load(app.Metadata.GameId);
        Check(NativeGameState.IsNative(LocalGameState.Read(saved.Metadata))==nn,"last checkpoint retains table family");
        Check(JToken.DeepEquals(LocalGameState.Read(saved.Metadata)["assignments"],LocalGameState.Read(app.Metadata)["assignments"]),"mixed difficulties survive checkpoint load");
        File.WriteAllText(Output+".release.json",new JObject{["passed"]=true,["mode"]=mode,["players"]=count,["seed"]=config.Seed,
            ["turns"]=turns,["actions"]=actions,["human_turns"]=humanTurns,["ai_turns"]=aiTurns,["saves_after_creation"]=output.Saves,
            ["debt_phases"]=debtPhases,
            ["all_ai_harness_only"]=!human,["unsupported_no_human_review_skipped"]=noHumanReview,
            ["eras"]=new JArray(eras),["terminal"]=true,["seconds"]=Time.realtimeSinceStartup-startedGame,
            ["nn_remaining_ms"]=nn?450:0,["v4_decisions"]="scripted Pass settlement control",
            ["scores"]=new JArray(app.Game.PlayersTrack.Players.Select(p=>new JObject{["seat"]=(int)p.Color,["vp"]=p.VP,["money"]=p.Money,["income"]=p.IncomeLevel})),["trace"]=rows}.ToString());
        Check(errors.Count==0,"no complete-game Unity errors: "+String.Join("\n",errors));
    }

    IEnumerator ReleaseFailureCases(PlayAiTurnUseCase turn)
    {
        var app=HumanCheckpoint(95159);int revision=app.Game.Revision;
        var earlier=Directory.Exists(NativeFailureBundle.Location)?Directory.GetFiles(NativeFailureBundle.Location,"native-failure-*.json"):new string[0];
        string disabled=Path.Combine(NativePackage.GameRoot,"BrassNative","disabled");
        Check(!File.Exists(disabled),"fixture begins enabled");
        router.ShowView<IPlayAiTurnView>();turn.PlayAITurn(false);yield return null;yield return null;
        File.WriteAllText(disabled,"L15 isolated disable test");expectedError=true;
        float until=Time.realtimeSinceStartup+8;
        try {
            while((LocalTurnBridge.IsThinking || !AiExecutionScheduler.IsIdle) && Time.realtimeSinceStartup<until)yield return null;
            Check(app.Game.Revision==revision && NativeGameState.IsNative(LocalGameState.Read(app.Metadata)),"disable commits no move and never changes one seat to V4");
            Check(AiExecutionScheduler.IsIdle && !AiExecutionResources.IsComputing,"disabled request retires resources");
        } finally {File.Delete(disabled);expectedError=false;}
        var bundles=Directory.GetFiles(NativeFailureBundle.Location,"native-failure-*.json").Except(earlier).ToArray();
        Check(bundles.Length>0,"live disable retains a failure bundle");
        var bundle=JObject.Parse(File.ReadAllText(bundles.Last()));
        Check(((string)bundle["error"]).Contains("disabled"),"bundle belongs to this disable failure");
        Check((bool)bundle["trusted_full_state_included"]==false && bundle["request"]["input_json"]!=null,"bundle contains only permitted request data");
        string diagnostic=Path.Combine(Path.GetDirectoryName(Output),"failure-bound-"+Guid.NewGuid().ToString("N"));
        for(int n=0;n<NativeFailureBundle.MaximumFiles+2;n++)NativeFailureBundle.Write(diagnostic,(JObject)bundle["request"],null,null,null,null,"test","expected",0,null);
        Check(Directory.GetFiles(diagnostic).Length==NativeFailureBundle.MaximumFiles,"diagnostic retention bounded without deleting earlier evidence");
        string oversized=NativeFailureBundle.Write(diagnostic+"-large",new JObject{["text"]=new string('x',NativeFailureBundle.MaximumBytes)},null,null,null,null,"test","expected",0,null);
        Check(oversized==null,"oversized diagnostic fails without replacing runtime failure");
        Check(!File.Exists(disabled) && NativePackage.Enabled,"disable fixture restored");
        File.Copy(bundles.Last(),Output+".failure.json");
        File.WriteAllText(Output+".release.json",new JObject{["passed"]=true,["mode"]="failures",["disable_no_commit"]=true,["failure_bundles"]=bundles.Length,
            ["bounded_retention"]=true,["no_single_seat_fallback"]=true}.ToString());
    }

    sealed class ReleaseOutput : IPlayGameOutput
    {
        readonly IRouter _router;readonly BrassApplication.GameHistory.BrassGameHistory history;public int Saves;
        public ReleaseOutput(IRouter router,BrassApplication.GameHistory.BrassGameHistory history){_router=router;this.history=history;}
        public void SaveGame(BrassApplicationGame game,ISaveGameDelegate callback){history.Repository.Save(game);Saves++;callback.SaveGameComplete(true);}
        public void StartHumanTurn(){}public void StartAITurn(PlayerMetadata p){}public void StartSellingIndustries(){throw new Exception("unexpected human debt in complete-game smoke");}
        public void StartGameReview(PlayerMetadata p){}public void StartNetworkHumanTurn(){throw new Exception("network unsupported");}public void StartNetworkSpectatorTurn(PlayerMetadata p){throw new Exception("network unsupported");}
    }
}
