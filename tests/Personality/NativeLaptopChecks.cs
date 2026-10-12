using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using BrassHost;
using BrassLegacyAI;
using BrassPatch.Persistence;
using BrassApplication.Game;
using BrassApplication.AI;
using BoardGameRules.Entities.EventSourcing.Events;
using BoardGameRules.Entities.Actions.Events;
using BoardGameRules.Entities.NumericalResources.Events;
using BoardGameRules.Entities.Industries;
using BoardGameRules.Entities.Common;
using Newtonsoft.Json.Linq;

public static class NativeLaptopChecks
{
    static int checks;
    static void Check(bool condition,string name){checks++;if(!condition)throw new Exception(name);}
    static BrassApplicationGame App(Host h) {
        var meta=new GameMetadata(h.Game.GameStartConfiguration,GameType.Local){GameId="p6-native-fixture"};
        foreach(var p in h.Game.GameStartConfiguration.Players)meta.JoinGame(new PlayerMetadata{PlayerStartConfiguration=p,AiType=AIType.Hard,PlayerNetworkId="test-"+p.Color});
        return new BrassApplicationGame(h.Game,meta);
    }
    public static int Main(string[] args) {
        try {
            var records=new List<object>();
            foreach(string requested in new[]{"pass","loan","build","develop","network","scout"}) {
                var h=new Host(327);var app=App(h);bool taken=false;string action=null;
                for(int step=0;step<120 && !taken;step++) {
                    var legal=h.Choices();var c=h.Phase=="action" ? legal.FirstOrDefault(x=>x.kind==requested)??legal.First(x=>x.kind=="pass") : legal.FirstOrDefault(x=>x.kind=="finish")??legal[0];
                    if(h.Phase=="action")action=c.kind;
                    h.Apply(h.Revision,c.id);
                    if(c.kind=="finish") {
                        int[] debts;var live=Host.CapturePersonality(app,null,out debts);
                        var expected=(JObject)Wire.Token(h.Observe(h.Actor));var actual=(JObject)Wire.Token(live.Observe(live.Actor));
                        Check(JToken.DeepEquals(expected["history"],actual["history"]),"exact_public_history_"+action+"\n"+expected["history"]+"\n"+actual["history"]);
                        Check(debts.All(x=>x==0),"no_spurious_insolvency");
                        var events=h.Game.GetUncommittedEvents().ToArray();h.Game.ClearUncommittedEvents();
                        // Keep complete history on later turns, as repository saves do.
                        // This fixture ends when its requested action is reached.
                        if(action==requested){
                            var saved=Host.CapturePersonality(app,events,out debts);
                            Check(JToken.DeepEquals(actual,Wire.Token(saved.Observe(saved.Actor))),"saved_projection_parity");taken=true;
                            var neural=live.ObservePersonality(live.Actor);
                            Check(neural["configuration"]==null && neural["context"]["player_count"]==null && (string)neural["protocol"]=="0.2.0","explicit_standard_four_player_wire");
                            records.Add(new {kind=requested,view=neural,choices=live.Choices(),insolvencies=debts});
                        } else throw new Exception("fixture_requested_action_unavailable_"+requested);
                    }
                }
                Check(taken,"covered_"+requested);
            }
            var baseline=new Host(55);var legacyApp=App(baseline);var state=LocalGameState.Create(legacyApp.Metadata,p=>"balanced");
            state["game_id"]=legacyApp.Metadata.GameId;state["ready"]=true;SaveExtensions.Set(legacyApp.Metadata,state);
            Check(JToken.DeepEquals(state,LocalGameState.Read(legacyApp.Metadata)),"legacy_save_unchanged");
            var legacyHistory=new BrassApplication.GameHistory.BrassGameHistory();
            legacyHistory.SetGame(legacyApp);
            Check(!PersonalityTurnBridge.PlayDebt(new BrassApplication.UseCases.Game.PlayGame.PlayGameUseCase(null,legacyHistory,null,null)),
                "legacy_liquidation_keeps_existing_dispatch");
            var composition=Host.Composition(Host.Configuration(baseline.Game.GameStartConfiguration));var before=Wire.Hash(baseline.Canonical());
            bool rejected=false;try{baseline.ValidateLiveAction(new[]{"invalid"},composition);}catch{rejected=true;}
            Check(rejected && before==Wire.Hash(baseline.Canonical()),"illegal_plan_no_live_mutation");
            var dh=new Host(638);var da=App(dh);var debtor=dh.Seats.First();
            foreach(var type in new[]{Wire.Type("Pottery"),Wire.Type("CottonMill")}) {
                var slot=dh.Slots.First(x=>x.Value.BuildedIndustry==null && x.Value.CanBuildIndustryType(type)).Value;
                dh.Game.ApplyEvent(new IndustryBuilt(debtor.Color,type,slot.UniqueId));
            }
            dh.Game.ApplyEvent(new MoneySpent(debtor.Color,debtor.Money));
            dh.Game.ApplyEvent(new IncomeMarkerPositionChanged(debtor.Color,-debtor.IncomeMarkerPosition,""));
            dh.Game.SellIndustryFlow.StartIndustrySelling();int[] debtCounts;
            var debtLive=Host.CapturePersonality(da,null,out debtCounts);var debtView=debtLive.ObservePersonality(debtLive.Actor);
            Check(debtLive.Phase=="debt" && debtCounts[(int)debtor.Color]==1,"public_debt_event_once");
            records.Add(new {kind="debt",view=debtView,choices=debtLive.Choices(),insolvencies=debtCounts});
            var debtPath=new[]{debtLive.Choices()[0].id};int debtRevision=dh.Game.Revision;
            debtLive.ValidatePersonalityAction(debtPath,composition);
            Check(dh.Game.Revision==debtRevision,"debt_validation_read_only");
            rejected=false;try{debtLive.ValidatePersonalityAction(new[]{"liquidate|foreign"},composition);}catch{rejected=true;}
            Check(rejected && dh.Game.Revision==debtRevision,"illegal_debt_no_mutation");
            debtLive.CommitPersonalityAction(debtPath,composition);
            Check(dh.Game.Revision>debtRevision && dh.Game.SellIndustryFlow.GetSoldIndustriesInLastRound()[debtor.Color].Count==1,"debt_commits_once");
            if(args.Length>2 && File.Exists(args[2])) {
                var unknown=new JObject{["format"]="brass-personality-game-save/0.9.0"};
                rejected=false;try{PersonalityState.Unwrap(unknown);}catch(InvalidDataException){rejected=true;}
                Check(rejected,"unknown_neural_save_never_falls_back_to_legacy");
                Check(!PersonalityTurnBridge.PublicFailure(new InvalidOperationException("own_hand: SECRET; sampled opponent SECRET")).Contains("SECRET"),"native_error_hides_private_input");
                // Explicit in-memory fixture injection only. Production package
                // loading has no test override or fixture release certificate.
                var initial=JObject.Parse(File.ReadAllText(args[2]));
                var flags=System.Reflection.BindingFlags.NonPublic|System.Reflection.BindingFlags.Static;
                typeof(PersonalityPackage).GetField("loaded",flags).SetValue(null,true);
                typeof(PersonalityPackage).GetField("manifest",flags).SetValue(null,new JObject{["id"]=initial["package_id"]});
                var outer=new JObject{["format"]=PersonalityState.Format,["package_id"]=initial["package_id"],["legacy"]=state,
                    ["initial"]=initial,["journal"]=new JArray()};outer["integrity"]=Wire.Hash(outer);SaveExtensions.Set(legacyApp.Metadata,outer);
                var events=baseline.Game.GetUncommittedEvents().ToList();JArray pending;
                var restored=PersonalityState.Snapshot(legacyApp.Metadata,baseline.Game.Revision,events,out pending);
                Check(pending==null && JToken.DeepEquals(initial,restored),"initial_neural_state_no_reroll");
                var plan=new JArray(new JArray("pass|fixture","finish|||||"));
                PersonalityState.Commit(legacyApp.Metadata,baseline.Game.Revision,events,restored,plan);
                restored=PersonalityState.Snapshot(legacyApp.Metadata,baseline.Game.Revision,events,out pending);
                Check(JToken.DeepEquals(initial,restored) && JToken.DeepEquals(plan,pending),"pending_action_rng_intent_retained");
                PersonalityState.SetLegacy(legacyApp.Metadata,state);
                Check(JToken.DeepEquals(state,LocalGameState.Read(legacyApp.Metadata)),"mixed_legacy_update_keeps_wrapper");
                var persisted=JObject.Parse(SaveExtensions.Get(legacyApp.Metadata).ToString());SaveExtensions.Set(legacyApp.Metadata,persisted);
                restored=PersonalityState.Snapshot(legacyApp.Metadata,baseline.Game.Revision,events,out pending);
                Check(JToken.DeepEquals(initial,restored) && JToken.DeepEquals(plan,pending),"serialized_neural_restore");
                restored=PersonalityState.Snapshot(legacyApp.Metadata,0,events,out pending);
                Check(JToken.DeepEquals(initial,restored) && pending==null,"rewind_does_not_reuse_future_plan");
            }
            File.WriteAllText(args[0],Wire.Json(new {passed=true,assertions=checks,records,fixture=true,installed=false}));
            Console.WriteLine("P6 native checks passed: "+checks);return 0;
        }catch(Exception error){Console.Error.WriteLine(error);return 1;}
    }
}
