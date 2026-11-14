using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using BoardGameRules.Entities.Actions.Events;
using BoardGameRules.Entities.Common;
using BoardGameRules.Entities.Game;
using BrassApplication.AI;
using BrassApplication.Game;
using BrassApplication.UseCases.Game.PlayAiTurn;
using BrassLegacyAI;
using BrassLegacyAI.Execution;
using BrassLegacyAI.Native;
using BrassPatch;
using BrassPatch.Persistence;
using InterfaceAdapters.Game.PlayAiTurn;
using Newtonsoft.Json.Linq;
using TMPro;
using UnityEngine;
using UnityEngine.SceneManagement;

public sealed partial class LocalAiUnityDriver
{
    IEnumerator OpeningTemperatureCases()
    {
        if(Environment.GetEnvironmentVariable("BRASS_OPENING_TEMPERATURE_TEST")=="live") {
            var live=OpeningTemperatureLive();while(live.MoveNext())yield return live.Current;yield break;
        }
        while(Get<int>(menu,"_numberOfPlayers")<4)menu.IncreaseNumberOfPlayers();
        menu.SetIntroductoryGame(false);yield return null;
        Rows()[1].GetComponent<AiSetupRow>().OpenPicker();var panel=AiSelectionPanel.Active;
        panel.SelectPage(PlayerSelectionPage.AlphaTest);
        Check(panel.DraftTemperature==1.0,"opening temperature defaults to 1.0");
        var betaPlus=panel.Tiles.Single(b=>b.name=="beta-2").GetComponentsInChildren<TextMeshProUGUI>();
        Check(betaPlus.Any(t=>t.name=="Name" && t.text=="Beta-Plus") && betaPlus.Any(t=>t.name=="Description" && t.text=="Re-start g1190 · Neutral context."),
            "Beta-Plus displays the new name and description on the existing model card");
        Check(panel.TemperatureButtons.Select(b=>b.GetComponentInChildren<TextMeshProUGUI>().text).SequenceEqual(new[]{"1.0","1.5","2.0"}),"opening options in displayed order");
        panel.TemperatureButtons[2].onClick.Invoke();
        Check(panel.DraftTemperature==2.0 && LocalMenu.TableTemperature(Setup.PlayerMetadata[1])==1.0,"draft applies only on confirmation");
        Check(panel.TemperatureButtons.Select(b=>b.GetComponent<AiControlSurface>().Selected).SequenceEqual(new[]{false,false,true}),"exactly one opening option is selected");
        panel.ApplyButton.onClick.Invoke();yield return null;
        MixedPick(2,"beta-2",0,16,1.5);yield return null;
        MixedPick(3,"il");yield return null;
        Call(menu,"CreateNewGameConfiguration");yield return null;
        Check(Setup.PlayerMetadata.Skip(1).Select(LocalMenu.TableTemperature).SequenceEqual(new[]{2.0,1.5,1.0}),"independent opening choices persist in menu preferences");
        var preference=JObject.Parse(MixedPreferences()["BrassLocalAI.Setup.v2"]);
        preference["seats"][1]["nn_temperature"]=1.3;
        var decoded=typeof(LocalMenu).Assembly.GetType("BrassLegacyAI.LocalSetupPreferences")
            .GetMethod("Decode",ReflectionAccess.Flags).Invoke(null,new object[]{preference.ToString()});
        var seats=Get<Array>(decoded,"Seats");
        Check(Get<double>(seats.GetValue(1),"Temperature")==1.0 && Get<double>(seats.GetValue(2),"Temperature")==1.5,
            "old menu preferences reset only the retired T1.3 option");

        foreach(var size in new[]{new[]{1920,1080},new[]{1280,720}}) {
            router.Resize(size[0],size[1]);yield return null;
            Rows()[1].GetComponent<AiSetupRow>().OpenPicker();panel=AiSelectionPanel.Active;
            if(size[0]==1280) {panel.ChangeTextSize(1);panel.ChangeTextSize(1);}
            yield return null;Canvas.ForceUpdateCanvases();
            var help=panel.Window.Find("NN opening temperature help").GetComponent<TextMeshProUGUI>();
            Check(help.text.Contains("First 3 moves") && help.text.Contains("then 1.0"),"help describes the exact cutoff");
            foreach(var button in panel.TemperatureButtons)AssertInsideTarget((RectTransform)button.transform,"opening option "+size[0]);
            AssertInsideTarget(help.rectTransform,"opening help "+size[0]);
            Check(help.GetPreferredValues(help.text,help.rectTransform.rect.width,0).y<=help.rectTransform.rect.height,"opening help fits "+size[0]);
            Capture("opening-temperature-"+size[0]+"x"+size[1]+".png");
            panel.TemperatureButtons[1].onClick.Invoke();panel.SelectPage(PlayerSelectionPage.V4);
            Check(panel.TemperatureButtons.Length==0,"opening options are only shown for NN");
            panel.Close();yield return null;
            Check(LocalMenu.TableTemperature(Setup.PlayerMetadata[1])==2.0,"cancel retains the confirmed opening temperature");
        }
        var app=menu.StartGame(92771);Check(app!=null,"game starts with three opening selections");
        var state=LocalGameState.Read(app.Metadata);
        history.Repository.Save(app);var loaded=history.Repository.Load(app.Metadata.GameId);
        var restored=LocalGameState.Read(loaded.Metadata);
        Check(JToken.DeepEquals(state,restored),"opening choices survive a real disk save/load");
        Check(loaded.Metadata.PlayerMetadata.Skip(1).Select(p=>NativeGameState.Temperature(restored,(int)p.PlayerStartConfiguration.Color))
            .SequenceEqual(new[]{2.0,1.5,1.0}),"saved settings stay attached to their seats");
        Check(NativeGameState.Model(restored,(int)loaded.Metadata.PlayerMetadata[2].PlayerStartConfiguration.Color)==NativeModel.Beta2,
            "Beta-Plus keeps the g1190 model in the save");
        int oldColor=(int)loaded.Metadata.PlayerMetadata[1].PlayerStartConfiguration.Color;
        var legacy=MixedGameState.Create(loaded.Metadata,
            p=>(int)p.PlayerStartConfiguration.Color==oldColor?new AiChoice(AiFamily.AlphaTest,"alpha-test"):new AiChoice(AiFamily.V4,"balanced"),
            p=>"easy",p=>16,p=>(int)p.PlayerStartConfiguration.Color==oldColor?1.3:1.0);
        legacy["ready"]=true;
        SaveExtensions.Set(loaded.Metadata,legacy);
        history.Repository.Save(loaded);
        var oldGame=history.Repository.Load(loaded.Metadata.GameId);
        var oldState=LocalGameState.Read(oldGame.Metadata);
        Check(NativeGameState.Temperature(oldState,oldColor)==1.3,"legacy game saves retain T1.3");
    }

    IEnumerator OpeningTemperatureLive()
    {
        yield return SceneManager.LoadSceneAsync("GameScene");yield return null;router.ReloadPrefabs();
        var view=(Frameworks.View.Game.PlayAiTurnView)router.GetView<IPlayAiTurnView>();
        var turn=new PlayAiTurnUseCase(new PlayAITurnPresenter(router),history);
        router.UseCases[typeof(PlayAiTurnUseCase)]=turn;view.Controller=new PlayAiTurnController(turn);
        var config=new GameStartConfiguration((NumberOfPlayers)4){Seed=92771,IsIntroductoryGame=false};
        var metadata=new GameMetadata(config,GameType.Local);
        foreach(var start in config.Players)metadata.JoinGame(new PlayerMetadata{PlayerStartConfiguration=start,
            AiType=AIType.Hard,NickName="Opening test",PlayerNetworkId="local-"+start.Color});
        var app=history.Repository.Create(metadata);app.Game.StartGame();metadata=app.Metadata;
        var color=app.Game.GameProgressInformation.CurrentPlayer.Color;
        var state=MixedGameState.Create(metadata,p=>new AiChoice(AiFamily.AlphaTest,"beta-2"),p=>"easy",p=>16,
            p=>p.PlayerStartConfiguration.Color==color?2.0:1.0);
        state["ready"]=true;SaveExtensions.Set(metadata,state);
        Process worker=null;
        int completedMoves=0;
        for(int round=1;round<=3;round++) {
            Check(app.Game.GameProgressInformation.CurrentRoundNumber==round,"expected confirmed Canal round "+round);
            while(app.Game.GameProgressInformation.CurrentPlayer.Color!=color) {
                var other=app.Game.GameProgressInformation.CurrentPlayer;
                while(app.Game.TurnFlow.GetRemainingActionsCount()>0) {
                    app.Game.TurnFlow.StartPassAction().Pass(other.Hand[0],other);app.Game.TurnFlow.ConfirmCurrentAction();
                }
                app.Game.GameFlow.EndTurn();
            }
            Check(app.Game.GameProgressInformation.CurrentRoundNumber==round,"same seat returns once in each Canal round");
            history.Repository.Save(app);history.SetGame(app);
            int revision=app.Game.Revision, actions=app.Game.TurnFlow.GetRemainingActionsCount();
            Check(actions==(round==1?1:2),"real rule engine gives the expected number of actions");
            double expected=round<=2?2.0:1.0;
            var jobs=new List<NativeJob>();var clock=Stopwatch.StartNew();
            router.ShowView<IPlayAiTurnView>();turn.PlayAITurn(false);
            while((LocalTurnBridge.IsThinking || !AiExecutionScheduler.IsIdle) && clock.Elapsed.TotalSeconds<14) {
                var bridge=typeof(LocalTurnBridge).GetField("active",ReflectionAccess.Flags).GetValue(null);
                var job=bridge==null?null:Get<NativeJob>(bridge,"nativeBackend");
                if(job!=null && !jobs.Contains(job))jobs.Add(job);
                yield return null;
            }
            Check(AiExecutionScheduler.IsIdle && !LocalTurnBridge.IsThinking && app.Game.TurnFlow.GetRemainingActionsCount()==0,"NN completes live round "+round);
            Check(jobs.Count>0 && jobs.All(j=>j.Result!=null && (double)j.Result["policy_temperature"]==expected),"Rust applies effective temperature "+expected+" in round "+round);
            foreach(var job in jobs) {
                var process=Get<Process>(Get<NativeProcess>(job,"worker"),"process");
                if(worker==null)worker=process;
                Check(ReferenceEquals(worker,process),"temperature changes reuse the loaded model");
            }
            int confirmed=app.Game.GetUncommittedEvents().Count(e=>e is ActionCountDecreased && e.Metadata.Revision>revision);
            Check(confirmed==actions,"temperature boundaries count confirmed game moves");
            completedMoves+=confirmed;
            turns.Add(new {round,temperature=expected,confirmed,completedMoves});
            app.Game.GameFlow.EndTurn();
            while(!app.Game.GameFlow.IsLastTurnOver()) {
                var other=app.Game.GameProgressInformation.CurrentPlayer;
                while(app.Game.TurnFlow.GetRemainingActionsCount()>0) {
                    app.Game.TurnFlow.StartPassAction().Pass(other.Hand[0],other);app.Game.TurnFlow.ConfirmCurrentAction();
                }
                app.Game.GameFlow.EndTurn();
            }
            app.Game.GameFlow.EndRound();
            history.Repository.Save(app);app=history.Repository.Load(metadata.GameId);
        }
        state=LocalGameState.Read(app.Metadata);
        var context=new JObject{["phase"]="action",["era"]="RailEra",["round"]=1};
        Check(NativeGameState.SearchTemperature(state,(int)color,context)==1.0,"Rail does not restart the opening");
        context["phase"]="debt";context["era"]="CanalEra";
        Check(NativeGameState.SearchTemperature(state,(int)color,context)==1.0,"debt uses regular temperature");
    }
}
