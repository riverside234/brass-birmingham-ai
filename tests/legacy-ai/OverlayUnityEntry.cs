using System;
using System.IO;
using System.Linq;
using System.Collections.Generic;
using Newtonsoft.Json.Linq;
using UnityEngine;
using BrassHost;
using BrassLegacyAI;
using BoardGameRules.Entities.Common;
using BoardGameRules.Entities.Game;
using BoardGameRules.Entities.Players;
using BoardGameRules.Entities.EventSourcing.Events;
using BrassApplication.AI;
using BrassApplication.AI.AiBehaviors;
using BrassApplication.Game;
using BrassApplication.GameHistory;
using BrassApplication.Persistence.Repository;
using BrassApplication.UseCases.Game.PlayAiTurn;
using Frameworks.Persistence.GameFactory;
using Frameworks.Persistence.Repository;
using Frameworks.Persistence.RepositoryBackends.InMemory;
using InterfaceAdapters.Common;
using InterfaceAdapters.Routing;

public static class UnityEntry
{
    public static void Run(){new GameObject("Legacy overlay test driver").AddComponent<OverlayDriver>();}
}
public sealed class OverlayDriver:MonoBehaviour
{
    sealed class History:IGameHistory
    {
        public IRepository Repository{get;set;}
        public BrassApplicationGame CurrentGame{get;private set;}
        public void SetGame(BrassApplicationGame g){CurrentGame=g;}
        public void MoveGameToRevision(int r){CurrentGame=Repository.Load(CurrentGame.Metadata.GameId,r);}
        public void MoveToLatestGame(){CurrentGame=Repository.Load(CurrentGame.Metadata.GameId);}
        public IReadOnlyCollection<IEvent> GetUnseenEvents(PlayerColor c){return CurrentGame.Game.GetUncommittedEvents();}
        public bool IsUndoActionAvailable(){return false;} public bool IsCancelActionAvailable(){return false;}
        public void UndoAction(){throw new NotSupportedException();}public void CancelAction(){throw new NotSupportedException();}
    }
    sealed class Router:IRouter
    {
        public bool Visible=true;
        public IUseCaseBuilder UseCaseBuilder{get;set;}
        public IAnimation ShowView<T>() where T:class,IBaseView{return null;}
        public IAnimation HideView<T>() where T:class,IBaseView{return null;}
        public void MoveOnTop<T>() where T:class,IBaseView{}
        public bool IsViewVisible<T>() where T:class,IBaseView{return Visible;}
        public bool IsViewCached<T>() where T:class,IBaseView{return false;}
        public T GetView<T>() where T:class,IBaseView{return null;}
        public void ProcessInput(){}public void RegisterKey(KeyEnum k,IBaseView v){}public void UnregisterKey(KeyEnum k,IBaseView v){}public void UnregisterAllKeys(IBaseView v){}
    }
    sealed class Presenter:IPlayAITurnOutput
    {
        readonly IRouter _router;readonly History history;
        public int Ends;
        public Presenter(IRouter router,History history){_router=router;this.history=history;}
        public void EndTurn(){Ends++;history.CurrentGame.Game.GameFlow.EndTurn();}
    }
    string root;History history;Router router;Presenter presenter;PlayAiTurnUseCase owner;
    int phase,frames,thinkingFrames,startRevision,cancelFrame;float started;
    History NewGame()
    {
        var config=new GameStartConfiguration(NumberOfPlayers.Four){Seed=7654};var metadata=new GameMetadata(config,GameType.Local);
        foreach(var start in config.Players){var p=new PlayerMetadata{PlayerStartConfiguration=start,PlayerNetworkId="overlay-"+start.Color,NickName="Overlay",AiType=AIType.Hard};p.AIBehaviour=AiBehaviorFactory.CreateAiBehavior(p);metadata.JoinGame(p);}
        var repository=new BaseRepository(new InMemoryGameStorage(),new InMemorySnapshotStorage(),new BrassBirminghamFactory());
        var app=repository.Create(metadata);app.Game.StartGame();repository.Save(app);var h=new History{Repository=repository};h.SetGame(app);return h;
    }
    void Begin()
    {
        history=NewGame();router=new Router();presenter=new Presenter(router,history);owner=new PlayAiTurnUseCase(presenter,history);
        startRevision=history.CurrentGame.Game.Revision;owner.PlayAITurn();owner.PlayAITurn();
    }
    void Start()
    {
        root=Path.GetDirectoryName(Application.dataPath);started=Time.realtimeSinceStartup;
        try
        {
            File.WriteAllText(Path.Combine(root,"configs/legacy-ai/live.json"),"{\"enabled\":true,\"profiles\":[\"cotton-v2\",\"cotton-v2\",\"cotton-v2\",\"cotton-v2\"]}");
            Begin();phase=1;
        }
        catch(Exception e){Finish(false,e.ToString());}
    }
    void Update()
    {
        if(phase==0)return;frames++;
        try
        {
            if(Time.realtimeSinceStartup-started>30)throw new Exception("overlay_watchdog");
            if(phase==1)
            {
                if(presenter.Ends==0){thinkingFrames++;return;}
                if(presenter.Ends!=1 || thinkingFrames<3 || history.CurrentGame.Game.GameProgressInformation.CurrentTurnNumber!=2)throw new Exception("async_or_duplicate_turn");
                string path=Path.Combine(root,"artifacts/legacy-ai/sidecars",Wire.Hash(history.CurrentGame.Metadata.GameId)+"-intents.json");
                var saved=JObject.Parse(File.ReadAllText(path));
                if(saved["records"].Count()!=1 || (string)saved["records"][0]["intent"]["profile"]!="legacy-hard-cotton-v2")throw new Exception("overlay_intent_not_persisted");
                phase=2;cancelFrame=frames;return;
            }
            if(phase==2 && frames>cancelFrame+2){Begin();router.Visible=false;cancelFrame=frames;phase=3;return;}
            if(phase==3 && frames>cancelFrame+20)
            {
                if(presenter.Ends!=0 || history.CurrentGame.Game.Revision!=startRevision)throw new Exception("menu_cancel_applied_action");
                Finish(true,null);
            }
        }
        catch(Exception e){Finish(false,e.ToString());}
    }
    void Finish(bool success,string error)
    {
        phase=0;File.WriteAllText(Path.Combine(root,"overlay-ui-result.json"),Wire.Json(new{passed=success,error,unity=Application.unityVersion,frames,thinking_frames=thinkingFrames,duplicate_turn_rejected=success,menu_cancel_unchanged=success,intent_persisted=success,scope="real Unity Update and woven PlayAITurn with mock view router; not rendered human acceptance"}));
        File.WriteAllText(Path.Combine(root,"configs/legacy-ai/live.json"),"{\"enabled\":false}");Application.Quit(success?0:1);
    }
}
