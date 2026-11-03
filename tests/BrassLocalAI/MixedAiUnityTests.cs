using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using BrassApplication.AI;
using BrassApplication.Game;
using BrassApplication.UseCases.Game.PlayAiTurn;
using BrassApplication.UseCases.Game.PlayGame;
using BoardGameRules.Entities.Game;
using BoardGameRules.Entities.EventSourcing.Events;
using BoardGameRules.Entities.GamePhases.Events;
using BrassHost;
using BrassLegacyAI;
using BrassLegacyAI.Native;
using BrassLegacyAI.Execution;
using BrassPatch;
using BrassPatch.Persistence;
using Frameworks.Utils;
using InterfaceAdapters.Game.PlayAiTurn;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using UnityEngine;
using UnityEngine.SceneManagement;

public sealed partial class LocalAiUnityDriver
{
    Dictionary<string,string> MixedPreferences()
    { return (Dictionary<string,string>)typeof(LocalMenu).GetField("TestPreferences",ReflectionAccess.Flags).GetValue(null); }
    void MixedPick(int seat,string key,int difficulty=0,int k=16,double temperature=1.0)
    {
        Rows()[seat].GetComponent<AiSetupRow>().OpenPicker();var panel=AiSelectionPanel.Active;
        if(key=="human")panel.SelectPage(PlayerSelectionPage.Human);
        else {
            bool native=NativeModel.ValidChoice(key);panel.SelectPage(native?PlayerSelectionPage.AlphaTest:PlayerSelectionPage.V4);
            panel.Tiles.Single(b=>b.name==key).onClick.Invoke();
            panel.DifficultyButtons[difficulty].onClick.Invoke();
            if(native){panel.SearchKButtons[k==8?0:1].onClick.Invoke();panel.TemperatureButtons[temperature==1.0?0:1].onClick.Invoke();}
        }
        Check(panel.ApplyButton.interactable,"per-seat selection available: "+key);panel.ApplyButton.onClick.Invoke();
    }
    void MixedRestored()
    {
        var p=Setup.PlayerMetadata;
        Check(p[0].AiType==AIType.Human,"human seat persists");
        Check(LocalMenu.Choice(p[1]).Equals(new AiChoice(AiFamily.V4,"balanced")),"V4 seat persists beside NN");
        Check(LocalMenu.Choice(p[2]).EntryId=="alpha-test" && LocalMenu.Choice(p[3]).EntryId=="il","distinct NN seats persist");
        Check(p.Skip(1).Select(LocalMenu.Difficulty).SequenceEqual(new[]{"easy","mid","max"}),"per-seat 4/8/12s persist");
        Check(LocalMenu.TableSearchK(p[2])==8 && LocalMenu.TableSearchK(p[3])==16 &&
            LocalMenu.TableTemperature(p[2])==1.5 && LocalMenu.TableTemperature(p[3])==1.0,"K and temperature stay per seat");
        Check(LocalMenu.CanStart(menu),"mixed table can start");
    }
    IEnumerator MixedAiCases()
    {
        Check(UIProperties.DefaultAnimationSpeed==2f && UIProperties.CurrentAnimationSpeed==2f,"startup animation default is 2x");
        if(Environment.GetEnvironmentVariable("BRASS_MIXED_CASE")=="live") {
            var live=MixedLive();while(live.MoveNext())yield return live.Current;yield break;
        }
        bool read=Environment.GetEnvironmentVariable("BRASS_MIXED_CASE")=="read";
        string pref=Environment.GetEnvironmentVariable("BRASS_MIXED_PREFERENCES");MixedPreferences().Clear();
        if(read)foreach(var pair in JsonConvert.DeserializeObject<Dictionary<string,string>>(File.ReadAllText(pref)))MixedPreferences().Add(pair.Key,pair.Value);
        Call(menu,"CreateNewGameConfiguration");yield return null;
        if(!read) {
            while(Get<int>(menu,"_numberOfPlayers")<4)menu.IncreaseNumberOfPlayers();menu.SetIntroductoryGame(false);yield return null;
            MixedPick(1,"balanced");yield return null;MixedPick(2,"alpha-test",1,8,1.5);yield return null;MixedPick(3,"il",3);yield return null;MixedRestored();
            string before=JsonConvert.SerializeObject(MixedPreferences());
            Rows()[2].GetComponent<AiSetupRow>().OpenPicker();var panel=AiSelectionPanel.Active;
            Check(!panel.GetComponentsInChildren<TMPro.TextMeshProUGUI>().Any(t=>t.text.Contains("whole table")),"whole-table UI removed");
            Check(panel.Tiles.Where(b=>b.gameObject.activeSelf).Select(b=>b.name).OrderBy(x=>x).SequenceEqual(NativeModel.All.Select(m=>m.Key).OrderBy(x=>x)),"NN roster comes from model registry");
            panel.Tiles.Single(b=>b.name=="beta").onClick.Invoke();panel.DifficultyButtons[2].onClick.Invoke();panel.Close();yield return null;
            Check(before==JsonConvert.SerializeObject(MixedPreferences()),"Cancel does not change any persisted seat");MixedRestored();
            MixedPick(3,"beta",2);yield return null;
            Check(LocalMenu.Choice(Setup.PlayerMetadata[1]).Family==AiFamily.V4 && LocalMenu.Choice(Setup.PlayerMetadata[2]).EntryId=="alpha-test" &&
                LocalMenu.Choice(Setup.PlayerMetadata[3]).EntryId=="beta" && LocalMenu.CanStart(menu),"V4 / Alpha / Beta share a table");
            MixedPick(1,"il");yield return null;Check(LocalMenu.CanStart(menu),"Alpha / Beta / IL share a table");
            MixedPick(1,"balanced");yield return null;MixedPick(3,"il",3);yield return null;
            MixedPick(0,"beta");yield return null;Check(LocalMenu.CanStart(menu),"V4, Alpha, Beta and IL may fill all four seats");
            MixedPick(0,"human");yield return null;MixedRestored();
            foreach(var size in new[]{new[]{1920,1080},new[]{1920,1200},new[]{1280,720}}) {
                router.Resize(size[0],size[1]);yield return null;Canvas.ForceUpdateCanvases();router.Camera.Render();yield return null;
                Rows()[2].GetComponent<AiSetupRow>().OpenPicker();panel=AiSelectionPanel.Active;
                panel.ChangeTextSize(1);panel.ChangeTextSize(1);yield return null;yield return null;Canvas.ForceUpdateCanvases();
                AssertInsideTarget(panel.Window,"mixed chooser "+size[0]);
                AssertInsideTarget((RectTransform)panel.ApplyButton.transform,"mixed apply "+size[0]);
                foreach(var button in panel.DifficultyButtons)AssertInsideTarget((RectTransform)button.transform,"mixed difficulty "+size[0]);
                foreach(var tile in panel.Tiles)Check(InsideViewport((RectTransform)tile.transform,panel.GridScroll.viewport),"NN model card fully visible at "+size[0]);
                var footer=panel.GetComponentsInChildren<TMPro.TextMeshProUGUI>().Single(t=>t.name=="Current AI").rectTransform;
                foreach(var page in panel.GetComponentsInChildren<UnityEngine.UI.Button>().Where(b=>b.name.StartsWith("Page "))) {
                    var a=new Vector3[4];var b=new Vector3[4];((RectTransform)page.transform).GetWorldCorners(a);footer.GetWorldCorners(b);
                    Check(a.Min(p=>p.y)>b.Max(p=>p.y),"navigation above footer at "+size[0]);
                }
                Capture("mixed-ai-"+size[0]+"x"+size[1]+".png");panel.Close();yield return null;
            }
            File.WriteAllText(pref,JsonConvert.SerializeObject(MixedPreferences()));
            Call(menu,"CreateNewGameConfiguration");yield return null;MixedRestored();
            // An old whole-table preference expands into independent seat defaults.
            var key=MixedPreferences().Keys.Single(k=>k=="BrassLocalAI.Setup.v2");var old=JObject.Parse(MixedPreferences()[key]);
            old["version"]=2;old["family"]="nn";old["nn_model"]="beta";old["nn_search_k"]=16;old["nn_temperature"]=1.5;
            MixedPreferences()[key]=old.ToString();Call(menu,"CreateNewGameConfiguration");yield return null;
            Check(Setup.PlayerMetadata.All(p=>LocalMenu.Choice(p).EntryId=="beta"),"old whole-table preferences migrate without losing model");
            MixedPreferences().Clear();foreach(var pair in JsonConvert.DeserializeObject<Dictionary<string,string>>(File.ReadAllText(pref)))MixedPreferences().Add(pair.Key,pair.Value);
            Call(menu,"CreateNewGameConfiguration");yield return null;
        }
        MixedRestored();
        var app=menu.StartGame(93251);Check(app!=null,"real menu creates mixed game");
        var state=LocalGameState.Read(app.Metadata);Check(MixedGameState.IsMixed(state),"new per-seat save envelope");
        history.Repository.Save(app);var loaded=history.Repository.Load(app.Metadata.GameId);
        Check(JToken.DeepEquals(state,LocalGameState.Read(loaded.Metadata)),"mixed identities survive disk save/load");
        LocalMenu.ResumeLabels(new InterfaceAdapters.Common.CreateGame.CreateOfflineGamePresenter(router,new Localization()),loaded);
        foreach(var p in loaded.Metadata.PlayerMetadata.Where(p=>p.AiType!=AIType.Human))
            Check(LocalMenu.RowLabel(p).Contains((string)LocalGameState.Assignment(state,(int)p.PlayerStartConfiguration.Color)["kind"]=="legacy"?"V4":NativeGameState.Model(state,(int)p.PlayerStartConfiguration.Color).Label),"resume shows correct per-seat model");
        foreach(string field in new[]{"backend","model_id","nn_search_k","nn_temperature","kind"}) {
            var bad=(JObject)state.DeepClone();var row=(JObject)bad["assignments"].First(r=>(string)r["kind"]=="native");row[field]=field=="nn_search_k"?(JToken)99:field=="nn_temperature"?(JToken)9.0:"changed";
            SaveExtensions.Set(loaded.Metadata,bad);bool rejected=false;try{LocalGameState.Read(loaded.Metadata);}catch(Exception){rejected=true;}
            Check(rejected,"changed saved identity rejected: "+field);
        }
        foreach(var model in NativeModel.All) {
            var old=NativeGameState.Create(loaded.Metadata,p=>"easy",16,1.0,model);old["ready"]=true;SaveExtensions.Set(loaded.Metadata,old);
            Check(NativeGameState.Model(LocalGameState.Read(loaded.Metadata))==model,"old homogeneous save remains readable: "+model.Label);
        }
        var v4=LocalGameState.Create(loaded.Metadata,p=>"balanced",p=>"easy");v4["ready"]=true;SaveExtensions.Set(loaded.Metadata,v4);
        Check(!NativeGameState.IsNative(LocalGameState.Read(loaded.Metadata)),"old V4 save remains readable");
    }
    IEnumerator MixedLive()
    {
        yield return SceneManager.LoadSceneAsync("GameScene");yield return null;router.ReloadPrefabs();
        var replayView=(Frameworks.View.Game.Replay.ReplayView)router.GetView<InterfaceAdapters.Game.Replay.IReplayView>();
        UIProperties.CurrentAnimationSpeed=1f;replayView.RestoreDefaultAnimationSpeed();
        Check(UIProperties.CurrentAnimationSpeed==2f,"reset after an action restores 2x");replayView.FastForwardReplay();
        Check(UIProperties.CurrentAnimationSpeed==2f,"fast-forward preserves 2x playback");
        var view=(Frameworks.View.Game.PlayAiTurnView)router.GetView<IPlayAiTurnView>();
        var turn=new PlayAiTurnUseCase(new PlayAITurnPresenter(router),history);router.UseCases[typeof(PlayAiTurnUseCase)]=turn;view.Controller=new PlayAiTurnController(turn);
        var serializer=new Frameworks.Persistence.Serialization.JsonSerializer();List<IEvent> events;
        using(var stream=File.OpenRead(Environment.GetEnvironmentVariable("BRASS_MIXED_FIXTURE")))events=serializer.Deserialize<List<IEvent>>(stream);
        var game=new BoardGameRulesGame(((GameStarted)events[0]).GameStartConfiguration);foreach(var e in events)game.ApplyEvent(e);
        var metadata=new GameMetadata(game.GameStartConfiguration,GameType.Local){GameId="mixed-"+Guid.NewGuid().ToString("N")};
        foreach(var start in game.GameStartConfiguration.Players)metadata.JoinGame(new PlayerMetadata{PlayerStartConfiguration=start,AiType=AIType.Hard,NickName="Mixed test",PlayerNetworkId="local-"+start.Color});
        var choices=new Dictionary<int,AiChoice>();int index=0;
        foreach(var p in metadata.PlayerMetadata)choices[(int)p.PlayerStartConfiguration.Color]=index++==0?new AiChoice(AiFamily.V4,"balanced"):new AiChoice(AiFamily.AlphaTest,NativeModel.All[index-2].Key);
        var state=MixedGameState.Create(metadata,p=>choices[(int)p.PlayerStartConfiguration.Color],p=>"easy",p=>16,p=>1.0);state["ready"]=true;
        var app=new BrassApplicationGame(game,metadata);SaveExtensions.Set(metadata,state);history.Repository.Save(app);history.SetGame(app);
        var seen=new HashSet<string>();var output=new ReleaseOutput(router,history);var play=new PlayGameUseCase(output,history,null,new Analytics());
        router.UseCases[typeof(PlayGameUseCase)]=play;
        for(int i=0;i<7 && seen.Count<4;i++) {
            int actor=(int)app.Game.GameProgressInformation.CurrentPlayer.Color;var choice=choices[actor];int revision=app.Game.Revision;
            state=LocalGameState.Read(app.Metadata);var jobs=new List<NativeJob>();var timer=Stopwatch.StartNew();
            router.ShowView<IPlayAiTurnView>();turn.PlayAITurn(false);
            while((LocalTurnBridge.IsThinking || !AiExecutionScheduler.IsIdle) && timer.Elapsed.TotalSeconds<12) {
                var bridge=typeof(LocalTurnBridge).GetField("active",ReflectionAccess.Flags).GetValue(null);
                if(bridge!=null){Check(Get<bool>(bridge,"native")== (choice.Family==AiFamily.AlphaTest),"actor dispatch matches saved family");var job=Get<NativeJob>(bridge,"nativeBackend");if(job!=null&&!jobs.Contains(job))jobs.Add(job);}
                yield return null;
            }
            Check(AiExecutionScheduler.IsIdle && !LocalTurnBridge.IsThinking && app.Game.TurnFlow.GetRemainingActionsCount()==0,"mixed live turn completes: "+choice);
            Check(history.Repository.Load(metadata.GameId).Game.Revision==revision,"no save mid-action");
            if(choice.Family==AiFamily.AlphaTest) {
                var model=NativeModel.FromChoice(choice.EntryId);Check(jobs.Count>0 && jobs.All(j=>j.Result!=null),"NN completed replies");
                Check(jobs.All(j=>(string)j.Result["model_id"]==model.Model && (string)j.Result["backend"]==model.Backend && (int)j.Result["sessions"]==2 && (int)j.Result["threads_per_session"]==1),"correct NN package and two one-thread sessions");
            }else Check(jobs.Count==0,"V4 does not invoke native NN");
            seen.Add(choice.EntryId);int prior=output.Saves;play.EndTurn();Check(output.Saves==prior+1,"one save after completed AI turn");
            var loaded=history.Repository.Load(metadata.GameId);Check(JToken.DeepEquals(LocalGameState.Read(app.Metadata),LocalGameState.Read(loaded.Metadata)),"mixed journals survive turn save/load");
            app=loaded;history.SetGame(app);turns.Add(new{actor,model=choice.ToString(),milliseconds=timer.Elapsed.TotalMilliseconds,revision=app.Game.Revision});
        }
        Check(seen.SetEquals(new[]{"balanced","alpha-test","beta","il"}),"all four backends ran at one table");
        state=LocalGameState.Read(app.Metadata);Check(state["records"].Any(r=>r["intent"]!=null) && state["records"].Any(r=>r["recall"]!=null),"V4 intent and NN clock journals coexist");
        // A history replacement retires an in-flight mixed controller before any commit.
        int before=app.Game.Revision;router.ShowView<IPlayAiTurnView>();turn.PlayAITurn(false);LocalTurnBridge.CancelActive();
        var wait=Stopwatch.StartNew();while(!AiExecutionScheduler.IsIdle && wait.Elapsed.TotalSeconds<5)yield return null;
        history.SetGame(history.Repository.Load(metadata.GameId));yield return null;
        Check(!LocalTurnBridge.IsThinking && AiExecutionScheduler.IsIdle && app.Game.Revision==before,"cancellation and reload reject stale mixed turn");
        Check(errors.Count==0,"no mixed gameplay errors");
    }
}
