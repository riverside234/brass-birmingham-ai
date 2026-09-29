// Optional Unity adapter. Built separately; not loaded by the training worker.
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using Newtonsoft.Json.Linq;
using UnityEngine;
using BrassHost;
using BrassApplication.AI;
using BrassApplication.AI.AiBehaviors;
using BrassApplication.Game;
using BrassApplication.GameHistory;
using BrassApplication.UseCases.Game.PlayAiTurn;
using InterfaceAdapters.Game.PlayAiTurn;
using InterfaceAdapters.Routing;
namespace BrassLegacyAI
{
    public sealed class UnityBridge:MonoBehaviour
    {
        static UnityBridge active;
        IGameHistory history;
        IPlayAITurnOutput presenter;
        IRouter router;
        BrassApplicationGame app;
        JObject composition;
        Profile profile;
        Ticket ticket;
        TurnCoordinator coordinator;
        IntentJournal intents;
        IntentState selectedIntent;
        Queue<string[]> continuation=new Queue<string[]>();
        string error;
        bool endTurn,playNow;
        int initialRevision;
        readonly System.Diagnostics.Stopwatch turnClock=System.Diagnostics.Stopwatch.StartNew();
        double turnBudget;
        static T Field<T>(object o,string n){return (T)o.GetType().GetField(n,BindingFlags.NonPublic|BindingFlags.Instance).GetValue(o);}
        static string Root {get{return Path.GetDirectoryName(Application.dataPath);}}
        public static IAiBehavior Factory(IAiBehavior original,PlayerMetadata metadata)
        {
            // Preserve the vendor type in saves. Both new and loaded games resolve
            // profile sidecars at the asynchronous turn hook, once player count exists.
            return original;
        }
        public static bool TryStart(PlayAiTurnUseCase owner,bool endTurn)
        {
            string configPath=Path.Combine(Root,"configs/legacy-ai/live.json");
            if(!File.Exists(configPath))return false;
            var config=JObject.Parse(File.ReadAllText(configPath));
            if((bool?)config["enabled"]!=true)return false;
            var history=Field<IGameHistory>(owner,"_gameHistory");var app=history.CurrentGame;
            if(app==null || (int)app.Metadata.GameStartConfiguration.NumberOfPlayers!=4)return false;
            int seat=(int)app.Game.GameProgressInformation.CurrentPlayer.Color;
            if(app.Metadata.GetPlayerMetadataByColor(app.Game.GameProgressInformation.CurrentPlayer.Color).AiType!=AIType.Hard)return false;
            if(active!=null)
            {
                if(ReferenceEquals(active.app,app) && active.initialRevision==app.Game.Revision)return true;
                Destroy(active.gameObject);active=null;
            }
            var go=new GameObject("Brass legacy AI turn");active=go.AddComponent<UnityBridge>();
            active.history=history;active.app=app;active.endTurn=endTurn;active.initialRevision=app.Game.Revision;
            active.presenter=Field<IPlayAITurnOutput>(owner,"_presenter");
            active.router=Field<IRouter>(active.presenter,"_router");
            try
            {
                string name=(string)config["profiles"][seat];
                if(string.IsNullOrEmpty(name) || name.Any(c=>!(char.IsLetterOrDigit(c) || c=='-')))throw new Exception("unsupported_live_profile");
                var data=JObject.Parse(File.ReadAllText(Path.Combine(Root,"configs/legacy-ai/"+name+".json")));
                active.profile=Profile.Read(data);
                active.turnBudget=active.profile.milliseconds;
                active.composition=JObject.Parse(File.ReadAllText(Path.Combine(Root,"configs/contracts/v0.2.0/card-composition.json")));
                active.CheckSidecar(config["profiles"]);
                active.StartRequest();
            }
            catch(Exception e){active.error=e.ToString();Debug.LogError(active.error);}
            return true;
        }
        void CheckSidecar(JToken names)
        {
            string game=app.Metadata.GameId;
            if(string.IsNullOrEmpty(game))throw new Exception("A persistent game ID is required for the legacy sidecar");
            string directory=Path.Combine(Root,"artifacts/legacy-ai/sidecars");Directory.CreateDirectory(directory);
            string path=Path.Combine(directory,Wire.Hash(game)+".json");
            var profiles=new JArray(names.Select(n=>new JObject{["name"]=n.DeepClone(),["sha256"]=Wire.Hash(JObject.Parse(File.ReadAllText(Path.Combine(Root,"configs/legacy-ai/"+(string)n+".json"))))}));
            bool hasIntent=names.Any(n=>(bool?)JObject.Parse(File.ReadAllText(Path.Combine(Root,"configs/legacy-ai/"+(string)n+".json")))["intent_enabled"]==true);
            var value=new JObject{["format"]="legacy-sidecar/0.1.0",["game_id"]=game,["rules"]=Wire.Rules,["profiles"]=profiles,["decision_seed"]=1,["intent_version"]=hasIntent?"legacy-intent/1.0.0":"H2-stateless"};
            if(File.Exists(path))
            {
                if(!JToken.DeepEquals(value,JObject.Parse(File.ReadAllText(path))))throw new Exception("Legacy profile sidecar mismatch; restore the saved profile versions before resuming.");
            }
            else File.WriteAllText(path,Wire.Json(value));
            intents=new IntentJournal(Path.Combine(directory,Wire.Hash(game)+"-intents.json"),game,profiles);
        }
        void StartRequest()
        {
            var host=Capture();var view=(JObject)Wire.Token(host.Observe(host.Actor));
            var cfg=Profile.Read(JObject.FromObject(profile));cfg.milliseconds=Math.Max(0,turnBudget-turnClock.Elapsed.TotalMilliseconds);
            ticket=Ticket.Create(host.GameId,app.Game.Revision,host.Actor,host.Phase,cfg);
            coordinator=new TurnCoordinator(ticket,view,host.Choices(),cfg,composition,intents.Read(host.Actor,app.Game.Revision,cfg.id));
        }
        Host Capture()
        {
            // Save/load clears GetUncommittedEvents. Load the persisted public action
            // history too, bounded at the current revision (including after an undo).
            var events=history.Repository.Advanced.GameStorage.LoadEvents(app.Metadata.GameId,0,app.Game.Revision);
            return Host.Capture(app,events);
        }
        void Update()
        {
            if(error!=null)return;
            if(!ReferenceEquals(history.CurrentGame,app) || !router.IsViewVisible<IPlayAiTurnView>())
            {Destroy(gameObject);return;}
            if(app.Game.Revision!=initialRevision || (int)app.Game.GameProgressInformation.CurrentPlayer.Color!=ticket.actor)
            {Destroy(gameObject);return;}
            try
            {
                if(continuation.Count==0)
                {
                    var now=JObject.FromObject(ticket).ToObject<Ticket>();now.revision=app.Game.Revision;now.actor=(int)app.Game.GameProgressInformation.CurrentPlayer.Color;
                    var plan=coordinator.Take(now,playNow);
                    if(plan==null)
                    {
                        if(coordinator.Finished && !coordinator.Ready && coordinator.Failure!=null)throw new Exception(coordinator.Failure);
                        if(app.Game.Revision!=ticket.revision)Destroy(gameObject);
                        return;
                    }
                    foreach(var a in plan.actions)continuation.Enqueue(a);
                    selectedIntent=plan.intent;
                }
                var path=continuation.Dequeue();var live=Capture();
                new LegacyBehavior(app,()=>live.LiveCommit(path,composition)).GetNextAIMove(app)();
                initialRevision=app.Game.Revision;
                intents.Commit(initialRevision,selectedIntent);
                if(app.Game.TurnFlow.GetRemainingActionsCount()==0)
                {
                    if(ReferenceEquals(active,this))active=null;
                    if(endTurn)presenter.EndTurn();Destroy(gameObject);return;
                }
                if(continuation.Count==0)
                {
                    // S0 or a resumed action: carry the SAME turn deadline forward.
                    coordinator.Dispose();StartRequest();
                }
                // A queued S1 continuation is revalidated against a fresh live view on
                // the next frame, and commits without another thinking allowance.
            }
            catch(Exception e){error=e.ToString();coordinator?.Dispose();Debug.LogError(error);}
        }
        void OnGUI()
        {
            if(error!=null)
            {
                GUI.Box(new Rect(20,20,650,105),"Legacy AI stopped safely. No further actions will be applied.\n"+error.Split('\n')[0]+"\nReturn to the menu and reload the last safe save.");
                return;
            }
            if(coordinator!=null && coordinator.Ready && GUI.Button(new Rect(20,20,140,42),"Play now"))playNow=true;
        }
        void OnDestroy(){coordinator?.Dispose();if(ReferenceEquals(active,this))active=null;}
    }
}
