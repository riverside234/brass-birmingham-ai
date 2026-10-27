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
using BrassApplication.AI;
using BrassApplication.Game;
using BrassApplication.UseCases.Game.PlayAiTurn;
using BrassApplication.UseCases.Game.PlayGame;
using BoardGameRules.Entities.Common;
using BoardGameRules.Entities.Game;
using InterfaceAdapters.Game.PlayAiTurn;
using InterfaceAdapters.Routing;
using Frameworks.View.Game;
using Newtonsoft.Json.Linq;
using UnityEngine;
using UnityEngine.SceneManagement;

public sealed partial class LocalAiUnityDriver
{
    IEnumerator NativeCases()
    {
        while(Get<int>(menu,"_numberOfPlayers")<4)menu.IncreaseNumberOfPlayers();
        menu.SetIntroductoryGame(false);yield return null;
        Check(NativePackage.Enabled,"native package present in isolated game");
        var player=Setup.PlayerMetadata[1];var row=Rows()[1].GetComponent<AiSetupRow>();row.OpenPicker();
        var panel=AiSelectionPanel.Active;panel.SwitchFamily(AiFamily.AlphaTest);
        Check(panel.CurrentPage==PlayerSelectionPage.AlphaTest && panel.ApplyButton.interactable,"Alpha Test is available");
        Check(!panel.GetComponentsInChildren<UnityEngine.UI.Button>().Any(b=>b.name=="Page V4"),"NN roster omits V4 seat choices");
        panel.DifficultyButtons[0].onClick.Invoke();Capture("l13-alpha-panel.png");panel.ApplyButton.onClick.Invoke();yield return null;
        Check(Setup.PlayerMetadata.Where(p=>p.AiType!=AIType.Human).All(p=>LocalMenu.Choice(p).Family==AiFamily.AlphaTest),"Alpha selection switches whole table");
        foreach(var pick in new[]{new[]{2,1},new[]{3,3}}) {
            Rows()[pick[0]].GetComponent<AiSetupRow>().OpenPicker();panel=AiSelectionPanel.Active;
            panel.DifficultyButtons[pick[1]].onClick.Invoke();panel.ApplyButton.onClick.Invoke();yield return null;
        }
        var mixedTimes=Setup.PlayerMetadata.Skip(1).Select(LocalMenu.Difficulty).ToArray();
        Check(mixedTimes.SequenceEqual(new[]{"easy","mid","max"}),"4s, 8s and 12s NN players coexist in one table");
        Check(LocalMenu.CanStart(menu),"standard NN table starts");
        Check(!LocalMenu.EditChoice(player,new AiChoice(AiFamily.V4,"balanced")),"seat-only V4 injection rejected");
        menu.DecreaseNumberOfPlayers();yield return null;
        Check(!LocalMenu.CanStart(menu),"unsupported three-seat NN table cannot start");
        Rows()[1].GetComponent<AiSetupRow>().OpenPicker();panel=AiSelectionPanel.Active;
        Check(!panel.GetComponentsInChildren<UnityEngine.UI.Button>().Any(b=>b.name=="Alpha Test table"),"unsupported NN option omitted");
        panel.Close();yield return null;menu.IncreaseNumberOfPlayers();menu.SetIntroductoryGame(true);yield return null;
        Check(!LocalMenu.CanStart(menu),"introductory NN table rejected");menu.SetIntroductoryGame(false);yield return null;
        // Table rollback is explicit, and affects every AI seat together.
        Rows()[1].GetComponent<AiSetupRow>().OpenPicker();panel=AiSelectionPanel.Active;panel.SwitchFamily(AiFamily.V4);panel.ApplyButton.onClick.Invoke();yield return null;
        Check(Setup.PlayerMetadata.Where(p=>p.AiType!=AIType.Human).All(p=>LocalMenu.Choice(p).Family==AiFamily.V4),"whole-table V4 rollback");
        Check(Setup.PlayerMetadata.Skip(1).Select(LocalMenu.Difficulty).SequenceEqual(mixedTimes),"family switch preserves other seats' independent difficulties");
        Rows()[1].GetComponent<AiSetupRow>().OpenPicker();panel=AiSelectionPanel.Active;panel.SwitchFamily(AiFamily.AlphaTest);panel.ApplyButton.onClick.Invoke();yield return null;
        Capture("l13-alpha-table.png");
        var app=menu.StartGame(92770);Check(app!=null,"new NN game created through actual menu");
        var saved=LocalGameState.Read(app.Metadata);Check(NativeGameState.IsNative(saved) && (string)saved["family"]=="nn","NN table persisted");
        Check(saved["assignments"].Where(r=>(string)r["kind"]=="native").Select(r=>(int)r["execution"]["search_ms"]).SequenceEqual(new[]{4000,8000,12000}),"mixed NN allowances persist");
        history.Repository.Save(app);var loaded=history.Repository.Load(app.Metadata.GameId);
        Check(JToken.DeepEquals(LocalGameState.Read(loaded.Metadata),saved),"NN save-load retains identities");
        var bad=(JObject)saved.DeepClone();bad["assignments"][1]["kind"]="legacy";SaveExtensions.Set(loaded.Metadata,bad);
        bool rejected=false;try{LocalGameState.Read(loaded.Metadata);}catch(InvalidDataException){rejected=true;}
        Check(rejected && JToken.DeepEquals(SaveExtensions.Get(loaded.Metadata),bad),"mixed save rejected without rewriting payload");
        selectionCasesPassed=true;
        router.HideView<InterfaceAdapters.Common.CreateGame.ICreateOfflineGameView>();yield return SceneManager.LoadSceneAsync("GameScene");yield return null;router.ReloadPrefabs();
        var aiView=(PlayAiTurnView)router.GetView<IPlayAiTurnView>();
        var turn=new PlayAiTurnUseCase(new PlayAITurnPresenter(router),history);
        router.UseCases[typeof(PlayAiTurnUseCase)]=turn;aiView.Controller=new PlayAiTurnController(turn);
        var output=new OutputAdapter(history);var play=new PlayGameUseCase(output,history,null,new Analytics());
        router.UseCases[typeof(PlayGameUseCase)]=play;
        var measured=new JArray();
        foreach(string difficulty in Environment.GetCommandLineArgs().Contains("--brass-native-short-test")?new[]{"easy"}:DifficultyCatalog.Keys) {
            var config=new GameStartConfiguration((NumberOfPlayers)4){Seed=92771,IsIntroductoryGame=false};
            var meta=new GameMetadata(config,GameType.Local);
            foreach(var start in config.Players)meta.JoinGame(new PlayerMetadata{PlayerStartConfiguration=start,AiType=AIType.Hard,NickName="Alpha Test",PlayerNetworkId="local-"+start.Color});
            var state=NativeGameState.Create(meta,p=>difficulty);SaveExtensions.Set(meta,state);
            app=history.Repository.Create(meta);app.Game.StartGame();state["game_id"]=app.Metadata.GameId;state["ready"]=true;SaveExtensions.Set(app.Metadata,state);
            // An ordinary two-action turn, reached using confirmed game events.
            while(app.Game.GameProgressInformation.CurrentRoundNumber==1) {
                var actorPlayer=app.Game.GameProgressInformation.CurrentPlayer;
                app.Game.TurnFlow.StartPassAction().Pass(actorPlayer.Hand[0],actorPlayer);app.Game.TurnFlow.ConfirmCurrentAction();
                app.Game.GameFlow.EndTurn();if(app.Game.GameFlow.IsLastTurnOver())app.Game.GameFlow.EndRound();
            }
            history.Repository.Save(app);history.SetGame(app);int revision=app.Game.Revision,actor=(int)app.Game.GameProgressInformation.CurrentPlayer.Color;
            var events=LocalGameState.History(app,history.Repository.Advanced.GameStorage.LoadEvents(app.Metadata.GameId,0,revision));
            var input=NativeInput.Capture(app,events,actor,new JArray());
            // Shadow uses the exact production adapter and never commits.
            long generation=(long)typeof(LocalTurnBridge).GetField("generation",System.Reflection.BindingFlags.Static|System.Reflection.BindingFlags.NonPublic).GetValue(null)+1;
            var watch=new System.Diagnostics.Stopwatch();var shadow=new NativeJob(NativeInput.Request(input,"shadow-"+difficulty,app.Metadata.GameId,generation,DifficultyCatalog.Milliseconds(difficulty)),watch,DifficultyCatalog.Milliseconds(difficulty));
            var job=AiExecutionScheduler.Submit(shadow);int frames=0;float until=Time.realtimeSinceStartup+75;
            while(!job.Completion.IsCompleted && Time.realtimeSinceStartup<until){frames++;yield return null;}
            Check(job.Completion.IsCompleted && job.Failure==null,"real native shadow completes: "+job.Failure);
            Check(frames>2 && app.Game.Revision==revision,"shadow leaves live state unchanged and Unity responsive");
            var actions=NativeInput.Actions(shadow.Result);Host.Capture(app,events).ValidateLiveAction(actions[0],Host.Composition((JObject)state["configuration"]));
            measured.Add(new JObject{["difficulty"]=difficulty,["shadow_ms"]=watch.Elapsed.TotalMilliseconds,["frames"]=frames,["actions"]=actions.Length,["completed"]=shadow.Result["completed"]});
            // New generation after shadow; both actions use the same live clock.
            history.SetGame(app);router.ShowView<IPlayAiTurnView>();turn.PlayAITurn(false);
            if(difficulty=="easy") {
                yield return null;router.HideView<IPlayAiTurnView>();yield return null;
                Check(LocalTurnBridge.IsThinking,"hiding the thinking panel preserves native work");
                router.ShowView<IPlayAiTurnView>();
            }
            int transitions=0,last=revision;until=Time.realtimeSinceStartup+40;
            while(LocalTurnBridge.IsThinking && Time.realtimeSinceStartup<until) {
                if(app.Game.Revision!=last){transitions++;last=app.Game.Revision;
                    if(app.Game.TurnFlow.GetRemainingActionsCount()==1) {
                        var journal=LocalGameState.Read(app.Metadata)["records"].Last;
                        Check((int)journal["remaining_ms"]>=0 && (int)journal["remaining_ms"]<DifficultyCatalog.Milliseconds(difficulty),"action two retains reduced allowance");
                        var midway=history.Repository.Load(app.Metadata.GameId);
                        Check(midway.Game.Revision==revision,"action one does not overwrite the pre-AI checkpoint");
                    }
                }
                yield return null;
            }
            Check(!LocalTurnBridge.IsThinking && app.Game.TurnFlow.GetRemainingActionsCount()==0,"Alpha Test completes both live actions "+difficulty);
            Check(app.Game.Revision>revision && transitions>=1,"native complete actions committed");
            loaded=history.Repository.Load(app.Metadata.GameId);
            Check(loaded.Game.Revision==revision && NativeGameState.IsNative(LocalGameState.Read(loaded.Metadata)),"completed AI actions leave the pre-AI checkpoint intact");
            Check(!AiExecutionResources.IsComputing && AiExecutionScheduler.IsIdle,"resources retire after both native owners");
            // Cancel in-flight work through the actual history lifecycle.
            var replay=history.Repository.Load(app.Metadata.GameId);history.SetGame(replay);
            revision=replay.Game.Revision;
            router.ShowView<IPlayAiTurnView>();turn.PlayAITurn(false);yield return null;yield return null;
            history.SetGame(replay);until=Time.realtimeSinceStartup+20;
            while(!AiExecutionScheduler.IsIdle && Time.realtimeSinceStartup<until)yield return null;
            Check(AiExecutionScheduler.IsIdle && replay.Game.Revision==revision && !LocalTurnBridge.IsThinking,"replacement cancels without late application");
        }
        var validationCases=NativeValidationLifecycleCases(turn);while(validationCases.MoveNext())yield return validationCases.Current;
        // History/menu exits retire the same backend lease. A retry of the
        // same live turn keeps its original clock.
        app=history.CurrentGame;int before=app.Game.Revision;
        int seat=(int)app.Game.GameProgressInformation.CurrentPlayer.Color;
        var originalClock=NativeTurnClock.For(app,seat,false).Clock;
        foreach(bool menuExit in new[]{false,true}) {
            router.ShowView<IPlayAiTurnView>();turn.PlayAITurn(false);yield return null;yield return null;
            Check(System.Object.ReferenceEquals(originalClock,NativeTurnClock.For(app,seat,false).Clock),"retry retains the ordinary-turn clock");
            if(menuExit)LocalTurnBridge.OnExit(null);else LocalTurnBridge.OnMoveHistory(history,before);
            float deadline=Time.realtimeSinceStartup+20;
            while(!AiExecutionScheduler.IsIdle && Time.realtimeSinceStartup<deadline)yield return null;
            Check(AiExecutionScheduler.IsIdle && app.Game.Revision==before && !LocalTurnBridge.IsThinking,"menu/history retires without live mutation");
        }
        // Kill only the owned test worker during a live request. Failure must
        // preserve the NN table and release resources without a substitute move.
        expectedError=true;
        router.ShowView<IPlayAiTurnView>();turn.PlayAITurn(false);yield return null;yield return null;
        var shared=typeof(NativeProcess).GetField("shared",System.Reflection.BindingFlags.Static|System.Reflection.BindingFlags.NonPublic).GetValue(null);
        var child=Get<System.Diagnostics.Process>(shared,"process");child.Kill();
        float retired=Time.realtimeSinceStartup+20;
        while((LocalTurnBridge.IsThinking || !AiExecutionScheduler.IsIdle) && Time.realtimeSinceStartup<retired)yield return null;
        expectedError=false;
        Check(!LocalTurnBridge.IsThinking && AiExecutionScheduler.IsIdle && !AiExecutionResources.IsComputing,"worker failure retires all live resources");
        Check(app.Game.Revision==before && NativeGameState.IsNative(LocalGameState.Read(app.Metadata)),"worker failure preserves board and NN family");
        try {NativeProcess.Shared();} catch(IOException) { }

        // Two debtors with multiple choices; the debt prompt must agree with
        // the qualified native public-state actor even with another turn order.
        app=ValidationGame(93813);
        while(app.Game.GameProgressInformation.CurrentTurnNumber<4 || app.Game.TurnFlow.GetRemainingActionsCount()>0) {
            var p=app.Game.GameProgressInformation.CurrentPlayer;
            while(app.Game.TurnFlow.GetRemainingActionsCount()>0){app.Game.TurnFlow.StartPassAction().Pass(p.Hand[0],p);app.Game.TurnFlow.ConfirmCurrentAction();}
            if(app.Game.GameProgressInformation.CurrentTurnNumber<4)app.Game.GameFlow.EndTurn();
        }
        var debtors=app.Game.PlayersTrack.Players.Take(2).ToArray();
        foreach(var p in debtors) {
            foreach(var type in new[]{BoardGameRules.Entities.Board.BuildingSlots.BuildingType.Pottery,BoardGameRules.Entities.Board.BuildingSlots.BuildingType.CottonMill}) {
                var slot=app.Game.Board.Regions.SelectMany(r=>r.BuildingSlots).First(s=>s.BuildedIndustry==null && s.CanBuildIndustryType(type));
                app.Game.ApplyEvent(new BoardGameRules.Entities.Actions.Events.IndustryBuilt(p.Color,type,slot.UniqueId));
            }
            app.Game.ApplyEvent(new BoardGameRules.Entities.NumericalResources.Events.MoneySpent(p.Color,p.Money));
            app.Game.ApplyEvent(new BoardGameRules.Entities.NumericalResources.Events.IncomeMarkerPositionChanged(p.Color,-p.IncomeMarkerPosition,""));
        }
        app.Game.SellIndustryFlow.StartIndustrySelling();history.Repository.Save(app);history.SetGame(app);
        Check(debtors.All(p=>!app.Game.SellIndustryFlow.CanIndustrySellingBeCompleted(p.Color)),"authored debt fixture has two active debtors: "+String.Join(",",debtors.Select(p=>p.Color+":"+p.Money+":"+p.IncomeMarkerPosition+":"+app.Game.SellIndustryFlow.RemainingDebtToPay(p.Color))));
        var debtOutput=new NativeDebtOutput(router,history);
        var debtPlay=new PlayGameUseCase(debtOutput,history,null,new Analytics());
        router.UseCases[typeof(PlayGameUseCase)]=debtPlay;
        int debtRevision=app.Game.Revision;debtPlay.PlayGame();
        Check((int)app.Game.GameProgressInformation.CurrentPlayer.Color==debtors.Where(p=>!app.Game.SellIndustryFlow.CanIndustrySellingBeCompleted(p.Color)).Min(p=>(int)p.Color),"debt prompt and canonical native actor agree");
        float debtDeadline=Time.realtimeSinceStartup+30;
        while(LocalTurnBridge.IsThinking && Time.realtimeSinceStartup<debtDeadline)yield return null;
        Check(app.Game.Revision>debtRevision && !BrassPatch.Liquidation.LiquidationFixes.IsPending(app.Game.SellIndustryFlow),"multiple native debtors finish without intermediate saves");
        Check(debtOutput.Saves==0 && history.Repository.Load(app.Metadata.GameId).Game.Revision==debtRevision,"AI liquidation and round completion do not write a checkpoint");
        Check(AiExecutionScheduler.IsIdle && !AiExecutionResources.IsComputing,"debt owners retire without a save callback");
        File.WriteAllText(Output+".native-turns.json",measured.ToString());
        Check(errors.Count==0,"no unexpected Unity errors: "+String.Join("\n",errors));
    }
    sealed class NativeDebtOutput : IPlayGameOutput
    {
        readonly IRouter _router;readonly BrassApplication.GameHistory.BrassGameHistory history;public int Saves;
        public NativeDebtOutput(IRouter router,BrassApplication.GameHistory.BrassGameHistory history){_router=router;this.history=history;}
        public void SaveGame(BrassApplicationGame game,BrassApplication.UseCases.Common.SaveGame.ISaveGameDelegate callback){history.Repository.Save(game);Saves++;}
        public void StartHumanTurn(){} public void StartAITurn(PlayerMetadata p){} public void StartSellingIndustries(){throw new Exception("unexpected human debt");}
        public void StartGameReview(PlayerMetadata p){} public void StartNetworkHumanTurn(){} public void StartNetworkSpectatorTurn(PlayerMetadata p){}
    }
}
