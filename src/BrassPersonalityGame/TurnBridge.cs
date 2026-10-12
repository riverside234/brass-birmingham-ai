using System;
using System.Collections.Generic;
using System.Linq;
using BrassHost;
using BrassPatch;
using BrassApplication.Game;
using BrassApplication.GameHistory;
using BrassApplication.UseCases.Game.PlayAiTurn;
using BrassApplication.UseCases.Game.PlayGame;
using InterfaceAdapters.Game.PlayAiTurn;
using InterfaceAdapters.Routing;
using BoardGameRules.Entities.EventSourcing.Events;
using BrassLegacyAI.Execution;
using Newtonsoft.Json.Linq;
using UnityEngine;
using UnityEngine.UI;
using TMPro;

namespace BrassLegacyAI
{
    public sealed class PersonalityTurnBridge:MonoBehaviour
    {
        static PersonalityTurnBridge active;
        IGameHistory history;BrassApplicationGame app;IPlayAITurnOutput presenter;IRouter router;
        PlayGameUseCase debtOwner;
        AiExecutionScope resources;AiExecutionJob execution;PersonalityJob backend;
        JObject saved,composition,reply;JArray paths;int actor,revision;bool endTurn,stopped;
        AiTurnControls controls;Button play;TextMeshProUGUI intent;
        public static bool IsThinking {get{return active!=null && !active.stopped;}}
        public static void CancelActive() {if(!ReferenceEquals(active,null))active.Stop();}
        public static void PlayNow() {if(active!=null && active.backend!=null)active.backend.PlayNow();}
        public static bool PlayDebt(PlayGameUseCase owner) {
            var history=ReflectionAccess.Get<IGameHistory>(owner,"_gameHistory");var app=history.CurrentGame;
            if(app==null)return false;
            int actor=(int)app.Game.GameProgressInformation.CurrentPlayer.Color;
            var assignment=PersonalityState.Assignment(app.Metadata,actor);if(assignment==null)return false;
            StartDebt(owner,history,app,actor);return true;
        }
        [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
        static void StartDebt(PlayGameUseCase owner,IGameHistory history,BrassApplicationGame app,int actor) {
            if(active!=null && ReferenceEquals(active.app,app) && active.revision==app.Game.Revision)return;
            LocalTurnBridge.CancelActive();
            var bridge=active=new GameObject("Brass neural mandatory debt").AddComponent<PersonalityTurnBridge>();
            bridge.history=history;bridge.app=app;bridge.actor=actor;bridge.revision=app.Game.Revision;bridge.debtOwner=owner;
            bridge.router=ReflectionAccess.Get<IRouter>(ReflectionAccess.Get<IPlayGameOutput>(owner,"_presenter"),"_router");
            try {
                bridge.controls=AiTurnControls.Attach(bridge.router,"Gamma · mandatory liquidation");bridge.MakeControls();
                bridge.composition=Host.Composition((JObject)LocalGameState.Read(app.Metadata)["configuration"]);
                bridge.saved=PersonalityState.Snapshot(app.Metadata,app.Game.Revision,bridge.Events(),out bridge.paths);
                bridge.paths=null;bridge.Launch();
            }catch(Exception e){bridge.Fail(e);}
        }
        public static bool Dispatch(PlayAiTurnUseCase owner,bool endTurn) {
            var history=ReflectionAccess.Get<IGameHistory>(owner,"_gameHistory");var app=history.CurrentGame;
            if(app==null)return false;
            int actor=(int)app.Game.GameProgressInformation.CurrentPlayer.Color;
            var assignment=PersonalityState.Assignment(app.Metadata,actor);if(assignment==null)return false;
            if(active!=null && ReferenceEquals(active.app,app) && active.revision==app.Game.Revision)return true;
            LocalTurnBridge.CancelActive();
            var bridge=active=new GameObject("Brass neural personality turn").AddComponent<PersonalityTurnBridge>();
            bridge.history=history;bridge.app=app;bridge.actor=actor;bridge.revision=app.Game.Revision;bridge.endTurn=endTurn;
            bridge.presenter=ReflectionAccess.Get<IPlayAITurnOutput>(owner,"_presenter");
            bridge.router=ReflectionAccess.Get<IRouter>(bridge.presenter,"_router");
            try {
                var group=PersonalityPackage.Entries().Single(e=>e.Choice.EntryId==(string)assignment["archetype"]);
                bridge.controls=AiTurnControls.Attach(bridge.router,"Gamma · "+group.Name+" · "+(string)assignment["tier"]);
                bridge.MakeControls();bridge.composition=Host.Composition((JObject)LocalGameState.Read(app.Metadata)["configuration"]);
                bridge.saved=PersonalityState.Snapshot(app.Metadata,app.Game.Revision,bridge.Events(),out bridge.paths);
                if(bridge.paths==null || bridge.paths.Count==0)bridge.Launch();
            }catch(Exception e){bridge.Fail(e);}
            return true;
        }
        List<IEvent> Events(){return LocalGameState.History(app,history.Repository.Advanced.GameStorage.LoadEvents(app.Metadata.GameId,0,app.Game.Revision));}
        void MakeControls() {
            if(controls==null)return;
            var template=controls.GetComponentInChildren<TextMeshProUGUI>();
            play=AiWidgets.Button(controls.transform,template,"Play Now","Play Now",new Vector2(310,36),new Vector2(120,32),19);
            play.onClick.AddListener(PlayNow);
            intent=AiWidgets.Text(controls.transform,template,"Public intent","",new Vector2(0,70),new Vector2(440,30),17);
            intent.richText=false;
        }
        void Launch() {
            int[] debts;var host=Host.CapturePersonality(app,Events(),out debts);
            if(host.Actor!=actor || host.Phase!=(debtOwner!=null?"debt":"action") || debtOwner==null && app.Game.TurnFlow.CurrentAction!=null)
                throw new InvalidOperationException("Neural action/debt boundary required");
            var body=new JObject{["protocol"]=PersonalityPackage.Protocol,["request_id"]="pending",["view"]=host.ObservePersonality(actor),
                ["choices"]=Wire.Token(host.Choices()),["save"]=saved,["insolvencies"]=new JArray(debts)};
            backend=new PersonalityJob(body);resources=resources??AiExecutionResources.BeginComputation();
            execution=AiExecutionScheduler.Submit(backend);resources.Observe(execution);
            paths=new JArray();reply=null;
        }
        void Update() {
            if(stopped)return;
            if(!ReferenceEquals(history.CurrentGame,app) || app.Game.Revision!=revision || (int)app.Game.GameProgressInformation.CurrentPlayer.Color!=actor){Stop();return;}
            try {
                if(paths.Count==0) {
                    if(execution==null || !execution.Completion.IsCompleted)return;
                    if(execution.Failure!=null)throw new InvalidOperationException(execution.Failure);
                    reply=backend.Result;
                    if(reply==null || (string)reply["status"]!="ok" || (string)reply["request_id"]!=backend.Id)throw new InvalidOperationException("Neural plan cancelled or unavailable");
                    saved=(JObject)reply["save"];paths=(JArray)reply["paths"];
                    if(paths.Count==0 || paths.Count>2)throw new InvalidOperationException("Invalid complete neural plan");
                }
                // Completion includes both workers and native inference requests.
                // Presentation/resource restoration is identical to the v4 path.
                resources?.Presenting();if(play!=null)play.interactable=false;controls?.SetApplying(true);
                int[] debts;var live=Host.CapturePersonality(app,Events(),out debts);var path=paths[0].ToObject<string[]>();
                live.ValidatePersonalityAction(path,composition); // before ANY live mutation
                if(debtOwner!=null)live.CommitPersonalityAction(path,composition);
                else new LegacyBehavior(app,()=>live.CommitPersonalityAction(path,composition)).GetNextAIMove(app)();
                paths.RemoveAt(0);revision=app.Game.Revision;
                PersonalityState.Commit(app.Metadata,revision,Events(),saved,paths);
                if(debtOwner!=null) {
                    var owner=debtOwner;Stop();
                    typeof(BrassPatch.Liquidation.LiquidationFixes).GetMethod("Save",System.Reflection.BindingFlags.NonPublic|System.Reflection.BindingFlags.Static)
                        .Invoke(null,new object[]{owner});return;
                }
                if(intent!=null && reply?["public"]!=null)intent.text=(string)reply["public"]["text"]??"";
                if(app.Game.TurnFlow.GetRemainingActionsCount()==0){bool advance=endTurn;var output=presenter;Stop();if(advance)output.EndTurn();return;}
                // Scout can reveal new own cards. The saved remaining allowance,
                // not a fresh 2/5/12 seconds, accompanies that continuation.
                if(paths.Count==0){execution=null;backend=null;Launch();if(play!=null)play.interactable=true;}
            }catch(Exception e){Fail(e);}
        }
        public static string PublicFailure(Exception error){return "The neural turn stopped. The save is preserved; no different personality will be substituted. Return to the menu to reload the last saved turn. ("+error.GetType().Name+")";}
        void Fail(Exception error){var r=router;Stop();if(r!=null)LocalMenu.ShowError(r,PublicFailure(error));}
        void Stop(){if(stopped)return;stopped=true;execution?.Cancel();resources?.Dispose();resources=null;if(ReferenceEquals(active,this))active=null;if(controls!=null)Destroy(controls.gameObject);Destroy(gameObject);}
        void OnDestroy(){Stop();app=null;history=null;presenter=null;router=null;}
    }
}
