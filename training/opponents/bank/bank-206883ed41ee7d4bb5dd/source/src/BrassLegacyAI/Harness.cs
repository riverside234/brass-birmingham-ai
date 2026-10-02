using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Reflection;
using Newtonsoft.Json.Linq;
using BrassHost;
using BoardGameRules.Entities.Game;
using BoardGameRules.Entities.Common;
using BrassApplication.Game;
using BrassApplication.AI;
using BrassApplication.AI.AiBehaviors;
using BrassApplication.AI.RuleBasedSystem.RuleSelectors;
using BrassApplication.AI.RuleBasedSystem.Rules;
using BrassApplication.AI.RuleBasedSystem;
using BoardGameRules.Entities.SellIndustry.Events;
namespace BrassLegacyAI
{
    public sealed class RecordingSelector:RuleSelector
    {
        readonly RuleSelector inner;
        public string selected;
        public int calls;
        public RecordingSelector(RuleSelector inner):base(null){this.inner=inner;}
        public override IRule Select(BrassApplicationGame app)
        {calls++;var r=inner.Select(app);selected=r==null?null:r.GetType().FullName+":"+r.Id;return r;}
    }
    public static class Harness
    {
        public static object Run(JObject request,JObject composition)
        {
            int seed=(int)request["seed"];int candidateSeat=(int?)request["candidate_seat"]??-1;
            if((int?)request["player_count"]!=4)throw new Exception("four_players_required");
            var profile=Profile.Read((JObject)request["profile"]);bool fixedWork=(bool?)request["fixed_work"]??false;
            var table=request["table_profiles"]?.Select(p=>p.Type==JTokenType.Null?null:Profile.Read((JObject)p)).ToArray();
            if(table!=null && table.Length!=4)throw new Exception("table_requires_four_seats");
            var intents=new IntentState[4];
            bool diagnosis=(bool?)request["diagnosis"]??false;bool record=(bool?)request["record_selectors"]??true;
            var h=new Host(seed);var planner=new Planner(composition);var config=new GameStartConfiguration(NumberOfPlayers.Four){Seed=seed};
            var metadata=new GameMetadata(config,GameType.Local);var selectors=new Dictionary<int,RecordingSelector>();
            foreach(var start in config.Players)
            {
                var p=new PlayerMetadata{PlayerStartConfiguration=start,PlayerNetworkId="legacy-"+start.Color,NickName="Legacy "+start.Color,AiType=AIType.Hard};
                p.AIBehaviour=AiBehaviorFactory.CreateAiBehavior(p);
                var f=typeof(HardAiBehavior).GetField("_actionRuleSelector",BindingFlags.NonPublic|BindingFlags.Instance);
                var r=new RecordingSelector((RuleSelector)f.GetValue(p.AIBehaviour));if(record)f.SetValue(p.AIBehaviour,r);selectors[(int)start.Color]=r;metadata.JoinGame(p);
            }
            var trace=new List<object>();var turnReports=new List<object>();var timer=Stopwatch.StartNew();string failure=null;int actions=0,turns=0;
            try
            {
                while(!h.Terminal)
                {
                    if(actions>160 || turns>80)throw new Exception("progress_limit");
                    int actor=h.Actor;var startContext=(JObject)Wire.Token(h.PublicContext());
                    if((bool?)request["check_live_observer"]==true)
                    {
                        var captured=Host.Capture(new BrassApplicationGame(h.Game,metadata));
                        var snapshot=(JObject)Wire.Token(captured.Observe(captured.Actor));
                        var sampled=new Host(1,Belief.Setup(snapshot,composition,1));
                        if(!sampled.Choices().Select(x=>x.id).SequenceEqual(h.Choices().Select(x=>x.id)))throw new Exception("live_observer_legal_parity");
                    }
                    bool replacement=table!=null?table[actor]!=null:candidateSeat==actor || candidateSeat==4;
                    var actorProfile=table!=null && replacement?table[actor]:profile;
                    if(h.Phase=="debt")
                    {
                        var view=(JObject)Wire.Token(h.Observe(actor));Choice c;
                        if(replacement)c=h.Choices().OrderByDescending(x=>Evaluator.Proposal(view,x)).ThenBy(x=>x.id,StringComparer.Ordinal).First();
                        else
                        {
                            // Run the untouched stock liquidation rule on an isolated
                            // trusted copy. Apply only this stock seat's next selection.
                            var copy=new Host(1,h.TrustedSetup());int revision=copy.Game.Revision;
                            var copyApp=new BrassApplicationGame(copy.Game,metadata);
                            var selector=RuleSelectorFactory.CreatePrioritySelector(RuleSetFactory.CreateSellIndustryRuleSet());
                            selector.Select(copyApp).MakeAction(copyApp).Invoke();
                            var sold=copy.Game.GetUncommittedEvents().OfType<IndustrySold>().First(e=>e.Metadata.Revision>revision && (int)e.PlayerColor==actor);
                            string slot=copy.Slots.Single(s=>s.Value.UniqueId==sold.SlotUniqueId).Key;
                            c=h.Choices().Single(x=>x.target==slot);
                        }
                        h.Apply(h.Revision,c.id);trace.Add(new{seat=actor,kind="debt",choices=new[]{c.id}});continue;
                    }
                    var turnTimer=Stopwatch.StartNew();turns++;var cached=new Queue<string[]>();int nodes=0;
                    int ordinary=h.Remaining;
                    for(int a=0;a<ordinary;a++)
                    {
                        var view=(JObject)Wire.Token(h.Observe(actor));var before=Wire.Hash(h.Canonical());
                        Decision decision=null;var actionTimer=Stopwatch.StartNew();
                        if(replacement || diagnosis)
                        {
                            if(cached.Count==0)
                            {
                                var cfg=Profile.Read((JObject)Wire.Token(actorProfile));
                                // Replacement actions share a turn budget. Historical
                                // diagnostics get an independent allowance per action.
                                cfg.milliseconds=replacement?Math.Max(0,actorProfile.milliseconds-turnTimer.Elapsed.TotalMilliseconds):actorProfile.milliseconds;
                                cfg.work_limit=replacement?Math.Max(1,actorProfile.work_limit-nodes):actorProfile.work_limit;
                                decision=planner.Decide(view,h.Choices(),cfg,fixedWork,1,intent:intents[actor]);nodes+=decision.nodes;
                                foreach(var path in decision.selected.actions)cached.Enqueue(path);
                            }
                            if(before!=Wire.Hash(h.Canonical()))throw new Exception("diagnosis_mutated_live_game");
                        }
                        if(replacement)
                        {
                            var plan=cached.Dequeue();h.CommitPlan(plan);actions++;
                            if(decision!=null)intents[actor]=decision.selected.intent;
                            trace.Add(new{seat=actor,context=startContext,source=actorProfile.id,choices=plan,decision,milliseconds=actionTimer.Elapsed.TotalMilliseconds});
                        }
                        else
                        {
                            var app=new BrassApplicationGame(h.Game,metadata);int calls=selectors[actor].calls;
                            var stockTimer=Stopwatch.StartNew();
                            var stock=h.StockStep(app,()=>metadata.GetPlayerMetadataByColor(h.Player.Color).AIBehaviour.GetNextAIMove(app));
                            stockTimer.Stop();
                            if(record && selectors[actor].calls!=calls+1)throw new Exception("stock_selector_call_count");
                            actions++;cached.Clear();
                            trace.Add(new{seat=actor,context=startContext,source="stock-hard-guarded-1.1.0",selected_rule=selectors[actor].selected,stock,decision,stock_action_ms=stockTimer.Elapsed.TotalMilliseconds,diagnosis_ms=decision==null?0:decision.elapsed_ms,milliseconds=actionTimer.Elapsed.TotalMilliseconds});
                        }
                        // Semantic public trace supports repeatability independent of vendor GUIDs.
                        trace.Add(new{after=Wire.Hash(h.Observe(actor)),context=h.PublicContext()});
                    }
                    turnReports.Add(new{seat=actor,source=replacement?actorProfile.id:"stock-hard-guarded-1.1.0",context=startContext,whole_turn_ms=turnTimer.Elapsed.TotalMilliseconds});
                }
            }
            catch(Exception ex){failure=ex.ToString();}
            return new{format="legacy-match/0.1.0",seed,player_count=4,candidate_seat=candidateSeat,profile=profile.id,table_profiles=table?.Select(p=>p?.id??"stock-hard-guarded-1.1.0").ToArray(),fixed_work=fixedWork,diagnosis,completed=h.Terminal,failure,failure_observation=failure==null?null:h.Observe(Math.Max(0,h.Actor)),actions,turns,elapsed_ms=timer.Elapsed.TotalMilliseconds,result=h.Result(),trace,turn_reports=turnReports};
        }
    }
}
