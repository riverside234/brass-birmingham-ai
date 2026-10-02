// Authored Brass adaptation of OpenSpiel's Apache-2.0 ISMCTS mechanics.
// Copyright 2019 DeepMind (reference mechanics). See NOTICE.md and LICENSE-OpenSpiel.
using System;
using System.Linq;
using System.Collections.Generic;
using System.Threading;
using Newtonsoft.Json.Linq;
using BrassHost;
namespace BrassLegacyAI
{
    public sealed class SearchEdge
    {
        public int visits,available;
        public double[] returns=new double[4];
    }
    public sealed class CompleteEvidence
    {
        public string signature,first;
        public int visits;
        public double[] returns=new double[4];
        public Dictionary<string,double> scenarios=new Dictionary<string,double>();
        public Plan plan;
    }
    public sealed class MaskedTelemetry
    {
        public string format="masked-ismcts/1.1.0",horizon="settled-reply-prefix",value_units="bounded-relative-heuristic",evaluator_kind="heuristic",q_transform;
        public int revisited_root_edges,revisited_plans,incumbent_visits,normalization_uses,unseen_competitions,legal_count,eligible_count;
        public bool fallback=true;
        public int workers=1;
        public Dictionary<string,int> cpu_batching;
        public string incumbent_signature;
        public double raw_q_lower=1,raw_q_upper=0,selection_q_lower=1,selection_q_upper=0,exploration_max;
        public Dictionary<string,double> timing_ms=new Dictionary<string,double>();
        public Dictionary<string,CompleteEvidence> plans=new Dictionary<string,CompleteEvidence>();
        public int completed,interrupted,opponent_choices,opponent_nodes=0,maximum_intervening_turns,outside_k,selected_outside_k;
        public double deferred_prior_mass;
        public double? conversion_floor;
        public Dictionary<string,SearchEdge> root=new Dictionary<string,SearchEdge>();
        public List<string> expansion=new List<string>();
    }
    public sealed partial class Planner
    {
        sealed class SearchNode
        {
            public int visits;
            public string[] legal;
            public double[] priors;
            public int[] ranked;
            public int[] protected_indices;
            public HashSet<int> admitted=new HashSet<int>();
            public double[] returns=new double[4];
            public Dictionary<string,SearchEdge> edges=new Dictionary<string,SearchEdge>();
        }
        sealed class Backup
        {
            public SearchNode node;
            public SearchEdge edge;
        }
        sealed class Outcome
        {
            public Plan plan;
            public string first;
            public CompleteEvidence evidence;
        }
        static uint MixSeed(uint seed,uint index)
        {uint x=seed+0x9e3779b9u*(index+1);x^=x<<13;x^=x>>17;x^=x<<5;return x==0?1:x;}
        static string Turn(JObject v)
        {var c=v["context"];return c["era"]+"/"+c["round"]+"/"+c["turn"]+"/"+c["turn_owner"];}
        // Only a permitted observation and semantic legality enter this identity.
        // Neither a world seed nor a sampled opponent hand is an input.
        public static string InformationKey(JObject permitted,List<Choice> choices)
        {
            var v=(JObject)permitted.DeepClone();var c=(JObject)v["context"];
            c.Remove("revision");c.Remove("game_id");
            return Wire.Hash(new JObject{["view"]=v,["legal"]=new JArray(choices.Select(x=>x.id))});
        }
        static double[] Priors(JObject v,List<Choice> legal,Profile p,IntentState intent)
        {
            var proposal=Evaluator.ProposalValues(v,legal,p.conversion_policy==1);
            var scores=legal.Select((c,i)=>proposal[i]+Tactics.Bias(p,intent,c,v)).ToArray();
            double max=scores.Max();var weights=scores.Select(x=>Math.Exp(Math.Max(-60,(x-max)/4))).ToArray();
            double sum=weights.Sum();return weights.Select(x=>.95*x/sum+.05/weights.Length).ToArray();
        }
        public static Choice MaskedRollout(JObject view,List<Choice> legal,string style)
        {
            if(legal.Count==0)throw new Exception("empty_nonterminal_legal_set");
            var p=new Profile{id="masked-rollout/"+style,tactic=style,intent_enabled=true,cash_policy=1,conversion_policy=1};
            var intent=Tactics.Refresh(view,p,null);
            var proposal=Evaluator.ProposalValues(view,legal,true);
            var selected=legal.Select((c,i)=>new{c,score=c.kind=="beer"?Conversion.BeerPriority(view,c,0):proposal[i]+Tactics.Bias(p,intent,c,view)}).OrderByDescending(x=>x.score).ThenBy(x=>x.c.id,StringComparer.Ordinal).First().c;
            if((string)view["context"]["phase"]=="action" && CashManagement.Project(view,(int)view["seat"]).shortfall>0)
                selected=legal.Where(c=>c.kind=="loan").OrderByDescending(c=>Evaluator.Proposal(view,c,true)).ThenBy(c=>c.id,StringComparer.Ordinal).FirstOrDefault()??selected;
            if(selected.kind=="develop" && legal.Any(c=>c.kind=="finish"))
            {
                var tile=Conversion.Player(view,(int)view["seat"])["mat"].Single(m=>(string)m["type"]==selected.industry)["tiles"].First();
                if(InvestmentOpportunity.Development(view,selected.industry)<=0)selected=legal.Single(c=>c.kind=="finish");
            }
            return selected;
        }
        static double[] LeafValues(JObject permitted,bool terminal)
        {
            var batch=HeuristicBatchQueue.Current;
            return batch==null?LeafValuesCore(permitted,terminal):batch.Evaluate("leaf/"+terminal+"/"+HeuristicBatchQueue.Key(permitted),()=>LeafValuesCore(permitted,terminal));
        }
        static double[] LeafValuesCore(JObject permitted,bool terminal)
        {
            var values=new double[4];var seats=permitted["players"].Select(x=>(int)x["seat"]).ToArray();
            if(terminal){foreach(int seat in seats)values[seat]=Evaluator.Evaluate(permitted,seat,true).win_share.Value;return values;}
            var ledgers=Evaluator.EvaluateAll(permitted,true);
            var scores=seats.Select(seat=>ledgers[seat].score-CashManagement.Project(permitted,seat).penalty).ToArray();
            double max=scores.Max();var weights=scores.Select(x=>Math.Exp(Math.Max(-60,(x-max)/20))).ToArray();
            for(int i=0;i<seats.Length;i++)values[seats[i]]=weights[i]/weights.Sum();return values;
        }
        static string Family(Choice c)
        {return c.kind+(c.kind=="build" || c.kind=="develop"?"/"+c.industry:c.kind=="beer"?"/"+c.target:"");}
        static List<int> Expanded(SearchNode n,List<Choice> legal,Profile p,bool root,Random rng,MaskedTelemetry log)
        {
            if(n.ranked==null)
            {
                n.ranked=Enumerable.Range(0,legal.Count).OrderByDescending(i=>n.priors[i]).ThenBy(i=>legal[i].id,StringComparer.Ordinal).ToArray();
                // Retain exact cards/routes/resources. Equivalent-looking choices are
                // deferred, not declared interchangeable or removed from legality.
                n.protected_indices=n.ranked.GroupBy(i=>Family(legal[i])).Select(g=>g.First()).ToArray();
            }
            int leading=n.edges.Count==0?0:n.edges.Values.Max(e=>e.visits);
            int width=Math.Min(legal.Count,(root?p.root_top_k:p.interior_top_k)+leading/p.reopen_support);
            var result=new HashSet<int>(n.ranked.Take(width));result.UnionWith(n.protected_indices);result.UnionWith(n.admitted);
            foreach(int i in n.ranked)if(n.edges.ContainsKey(legal[i].id))result.Add(i);
            var tail=n.ranked.Where(i=>!result.Contains(i)).ToArray();
            if(tail.Length>0 && leading>=p.reopen_support && rng.NextDouble()<p.tail_admission)
            {
                double pick=rng.NextDouble()*tail.Sum(i=>.9*n.priors[i]+.1/legal.Count);int admitted=tail.Last();
                foreach(int i in tail){pick-=.9*n.priors[i]+.1/legal.Count;if(pick<=0){admitted=i;break;}}
                result.Add(admitted);n.admitted.Add(admitted);if(root)log.outside_k++;
            }
            if(root){log.deferred_prior_mass=n.ranked.Where(i=>!result.Contains(i)).Sum(i=>n.priors[i]);log.eligible_count=result.Count;}
            return result.OrderBy(i=>i).ToList();
        }
        static int Select(SearchNode n,List<Choice> choices,List<int> active,Profile p,int seat,MaskedTelemetry log)
        {
            double parent=n.visits>0?n.returns[seat]/n.visits:1.0/p.player_count;
            double mass=Enumerable.Range(0,choices.Count).Where(i=>n.edges.ContainsKey(choices[i].id) && n.edges[choices[i].id].visits>0).Sum(i=>n.priors[i]);
            double fpu=Math.Max(0,Math.Min(1,parent-p.fpu_reduction*Math.Sqrt(mass)));
            var supported=n.edges.Values.Where(e=>e.visits>=p.minimum_plan_visits).Select(e=>e.returns[seat]/e.visits).ToArray();
            bool normalize=p.q_transform=="regularized_sibling" && supported.Length>=2;
            double lower=supported.Concat(new[]{parent}).Min(),upper=supported.Concat(new[]{parent}).Max();
            if(active.Any(i=>!n.edges.ContainsKey(choices[i].id)) && active.Any(i=>n.edges.ContainsKey(choices[i].id)))log.unseen_competitions++;
            if(p.selection_policy=="unseen_first")
            {var unseen=active.Where(i=>!n.edges.ContainsKey(choices[i].id)).ToArray();if(unseen.Length>0)return unseen.OrderByDescending(i=>n.priors[i]).First();}
            double priorMass=active.Sum(i=>n.priors[i]);
            double best=double.NegativeInfinity;int chosen=active[0];
            foreach(int i in active)
            {
                SearchEdge e;n.edges.TryGetValue(choices[i].id,out e);double raw=SearchSelection.Mean(e,seat,fpu);
                double q=normalize?SearchSelection.Transform(raw,lower,upper,p.q_min_span):raw;
                double u=p.exploration*(n.priors[i]/priorMass)*Math.Sqrt(Math.Max(1,n.visits))/(1+(e==null?0:e.visits));
                log.raw_q_lower=Math.Min(log.raw_q_lower,raw);log.raw_q_upper=Math.Max(log.raw_q_upper,raw);
                log.selection_q_lower=Math.Min(log.selection_q_lower,q);log.selection_q_upper=Math.Max(log.selection_q_upper,q);log.exploration_max=Math.Max(log.exploration_max,u);
                if(q+u>best){best=q+u;chosen=i;}
            }
            if(normalize)log.normalization_uses++;return chosen;
        }
        static List<Outcome> SafeOutcomes(List<Outcome> candidates,Profile p)
        {
            if(p.cash_policy==1)
            {
                int shortfall=candidates.Min(c=>c.plan.cash==null?0:c.plan.cash.shortfall);
                candidates=candidates.Where(c=>(c.plan.cash==null?0:c.plan.cash.shortfall)==shortfall).ToList();
                if(p.conversion_policy==1 && candidates.Any(c=>c.plan.cash==null || !c.plan.cash.future_credit_blocked))
                    candidates=candidates.Where(c=>c.plan.cash==null || !c.plan.cash.future_credit_blocked).ToList();
            }
            return candidates;
        }
        static void SaleIncumbents(Session s,Host root,int target,List<Plan> finals)
        {
            if(s.profile.conversion_policy!=1 || root.Phase!="action")return;
            // A bounded conversion fallback, independent of search expansion.
            // Compare a batched sale and all three beer-source preferences before
            // timing out with an avoidably unflipped tile. No beam is constructed.
            var sales=root.Choices().Where(c=>c.kind=="sell").OrderByDescending(c=>Evaluator.Proposal(s.root,c,true))
                .ThenBy(c=>c.id,StringComparer.Ordinal).GroupBy(c=>c.target+"/"+c.merchant).Select(g=>g.First()).Take(1);
            foreach(var sale in sales)for(int beer=0;beer<3;beer++)
            {
                var p=Score(s,new List<string[]>{GreedyFinish(s,new List<string>(),sale.id,true,beer)});
                while(p.actions.Count<target)p=Complete(s,p,target);
                if(!finals.Any(x=>x.signature==p.signature))finals.Add(p);
            }
        }
        static void Timing(MaskedTelemetry log,string key,System.Diagnostics.Stopwatch timer)
        {double old;log.timing_ms.TryGetValue(key,out old);log.timing_ms[key]=old+timer.Elapsed.TotalMilliseconds;}
        static Plan FastPlan(Session s,int target,string first=null,int beerPreference=0,bool recovery=false,string secondKind=null)
        {
            var world=s.World(new string[0]);var actions=new List<string[]>();var partial=new List<string>();
            for(int step=0;step<96;step++)
            {
                var legal=world.Choices();Choice chosen;
                if(step==0 && first!=null)chosen=legal.Single(c=>c.id==first);
                else if(legal.Count==1)chosen=legal[0];
                else
                {
                    var view=s.View(world);var values=Evaluator.ProposalValues(view,legal,s.profile.conversion_policy==1);
                    chosen=legal.Select((c,i)=>new{c,score=c.kind=="beer"?Conversion.BeerPriority(view,c,beerPreference):values[i]+Tactics.Bias(s.profile,s.intent,c,view)})
                        .OrderByDescending(x=>x.score).ThenBy(x=>x.c.id,StringComparer.Ordinal).First().c;
                    if(secondKind!=null && actions.Count==1 && partial.Count==0 && world.Phase=="action")
                        chosen=legal.Select((c,i)=>new{c,score=values[i]}).Where(x=>x.c.kind==secondKind)
                            .OrderByDescending(x=>x.score).ThenBy(x=>x.c.id,StringComparer.Ordinal).Select(x=>x.c).FirstOrDefault()??chosen;
                    if(world.Phase=="action" && s.profile.cash_policy==1 && CashManagement.Project(view,s.seat).shortfall>0)
                        chosen=legal.FirstOrDefault(c=>c.kind=="loan")??chosen;
                    if(recovery && actions.Count>0 && world.Phase=="action")
                    {
                        var loan=CashManagement.Project(view,s.seat).shortfall>0?legal.FirstOrDefault(c=>c.kind=="loan"):null;
                        chosen=loan??legal.FirstOrDefault(c=>c.kind=="pass")??chosen;
                    }
                    if(chosen.kind=="develop" && legal.Any(c=>c.kind=="finish"))
                    {var tile=Conversion.Player(view,s.seat)["mat"].Single(m=>(string)m["type"]==chosen.industry)["tiles"].First();if(s.profile.conversion_policy==1?InvestmentOpportunity.Development(view,chosen.industry)<=0:(int)tile["level"]>1)chosen=legal.Single(c=>c.kind=="finish");}
                }
                world.SearchApply(chosen.id);s.nodes++;partial.Add(chosen.id);
                if(world.ActionComplete){actions.Add(partial.ToArray());partial.Clear();if(actions.Count>=target || world.Remaining==0)return ScoreWorld(s,actions,world);}
            }
            throw new InvalidOperationException("fast_plan_completion_cap");
        }
        static List<Plan> FastIncumbents(Session s,Host root,int target)
        {
            var quick=FastPlan(s,target);var result=new List<Plan>{quick};
            // Rail investments compete as complete plans, including their sale
            // or resource completion. Protected policy families alone cannot
            // guarantee that a one-second search actually finishes a build plan.
            if(target>=2 && !s.Stop)
            {
                var legal=root.Choices();var values=Evaluator.ProposalValues(s.root,legal,true);
                var families=legal.Select((c,i)=>new{c,score=values[i]}).Where(x=>x.c.kind=="build")
                    .OrderByDescending(x=>x.score).GroupBy(x=>x.c.industry).Select(g=>g.First().c).ToArray();
                // Each private owner explores different industry families. Four
                // workers must not spend the whole short clock reproducing the
                // same two coal/iron plans while never comparing a scoring build.
                var builds=families.Skip(s.profile.partition_index).Where((c,i)=>i%s.profile.partition_count==0).Take(2);
                foreach(var build in builds)
                {
                    if(s.Stop || (!s.fixedWork && s.clock.Elapsed.TotalMilliseconds>s.profile.milliseconds*.35))break;
                    result.Add(FastPlan(s,target,build.id));
                    if(!s.Stop && Conversion.Sellable(Conversion.Player(s.root,s.seat)["mat"].Single(m=>(string)m["type"]==build.industry)["tiles"][0]) &&
                        s.root["slots"].Any(x=>x["tile"]!=null && x["tile"].Type!=JTokenType.Null && (int)x["tile"]["seat"]==s.seat && Conversion.Sellable(x["tile"]) && !(bool)x["tile"]["flipped"]))
                        result.Add(FastPlan(s,target,build.id,secondKind:"sell"));
                }
            }
            if(s.profile.cash_policy==1 && quick.cash!=null && (quick.cash.shortfall>0 || quick.cash.future_credit_blocked))
            {
                var rescue=root.Choices().Where(c=>c.kind=="loan" || c.kind=="sell" || c.kind=="pass")
                    .OrderByDescending(c=>c.kind=="loan").ThenByDescending(c=>Evaluator.Proposal(s.root,c,true)).GroupBy(c=>c.kind+"/"+c.target+"/"+c.merchant).Select(g=>g.First()).Take(5);
                foreach(var first in rescue)
                {
                    int variants=first.kind=="sell"?3:1;bool rescued=false;
                    for(int beer=0;beer<variants;beer++)
                    {
                        var plan=FastPlan(s,target,first.id,beer,true);result.Add(plan);
                        if(plan.cash.shortfall==0 && !plan.cash.future_credit_blocked){rescued=true;break;}
                    }
                    if(rescued)break;
                }
            }
            if(!s.Stop && target>=2 && (s.fixedWork || s.clock.Elapsed.TotalMilliseconds<s.profile.milliseconds*.45))
            {
                var choices=root.Choices();var values=Evaluator.ProposalValues(s.root,choices,s.profile.conversion_policy==1);
                var setup=choices.Select((c,i)=>new{c,value=values[i]}).Where(x=>new[]{"loan","develop","network"}.Contains(x.c.kind))
                    .OrderByDescending(x=>x.c.kind=="develop" && x.value>0).ThenByDescending(x=>x.value).GroupBy(x=>x.c.kind).Select(g=>g.First().c).Take(2);
                foreach(var first in setup)
                {
                    if(s.Stop || (!s.fixedWork && s.clock.Elapsed.TotalMilliseconds>s.profile.milliseconds*.45))break;
                    result.Add(FastPlan(s,target,first.id));
                    if(first.kind=="loan" && !s.Stop && choices.Any(c=>c.kind=="develop" && Evaluator.Proposal(s.root,c,true)>0))
                        result.Add(FastPlan(s,target,first.id,secondKind:"develop"));
                }
            }
            if(s.profile.conversion_policy==1 && (!result.Any(p=>p.actions.Any(a=>a[0].StartsWith("sell|"))) ||
                Conversion.Actions(s.root,s.seat)<=2 || s.root["slots"].Any(x=>x["tile"]!=null && (int)x["tile"]["seat"]==s.seat && (string)x["tile"]["type"]=="Brewery" && (int)x["tile"]["resource"]>0)))
            {
                var sale=root.Choices().Where(c=>c.kind=="sell").OrderByDescending(c=>Evaluator.Proposal(s.root,c,true)).FirstOrDefault();
                if(sale!=null)for(int beer=0;beer<3;beer++)
                {
                    bool critical=Conversion.Actions(s.root,s.seat)<=2;
                    if(!critical && (s.Stop || (!s.fixedWork && s.clock.Elapsed.TotalMilliseconds>s.profile.milliseconds*.4)))break;
                    result.Add(FastPlan(s,target,sale.id,beer));
                }
            }
            return result;
        }
        public Decision Decide(JObject observation,List<Choice> legal,Profile profile,bool fixedWork=false,uint seed=1,CancellationToken cancel=default(CancellationToken),Action<Plan> incumbent=null,IntentState intent=null)
        {
            Profile.ValidateSearch(profile);
            var s=new Session{root=(JObject)observation.DeepClone(),profile=profile,fixedWork=fixedWork,cancel=cancel,seat=(int)observation["seat"]};
            var log=new MaskedTelemetry{q_transform=profile.q_transform,horizon=profile.leaf_policy+"/"+profile.opponent_turn_limit,legal_count=legal.Count};
            if((int)observation["context"]["actor"]!=s.seat || observation["players"].Count()!=profile.player_count)throw new Exception("actor_or_player_count");
            var timer=System.Diagnostics.Stopwatch.StartNew();s.intent=Tactics.Refresh(s.root,profile,intent);s.setup=Belief.Setup(s.root,composition,seed);
            var root=s.World(new string[0]);var rootIds=legal.Select(c=>c.id).OrderBy(x=>x,StringComparer.Ordinal).ToArray();
            if(!root.Choices().Select(c=>c.id).SequenceEqual(rootIds))throw new Exception("legal_contract_mismatch");
            int target=root.Phase=="debt"?1:Math.Min(2,root.Remaining);List<Plan> finals;
            if(root.Phase=="action")finals=FastIncumbents(s,root,target);
            else
            {
                var first=MaskedRollout(s.root,legal,profile.tactic);var action=root.Phase=="debt"?new[]{first.id}:GreedyFinish(s,new List<string>(),first.id,true);
                var plan=Score(s,new List<string[]>{action});while(plan.actions.Count<target)plan=Complete(s,plan,target);finals=new List<Plan>{plan};
            }
            var baseline=Choose(finals,profile);var best=baseline;log.incumbent_signature=baseline.signature;incumbent?.Invoke(best);Timing(log,"incumbent",timer);
            var tree=new Dictionary<string,SearchNode>();string firstKey=InformationKey(s.root,legal),initialTurn=Turn(s.root);
            if(profile.conversion_policy==1 && Conversion.Actions(s.root,s.seat)<=2 && root.Choices().Any(c=>c.kind=="sell"))log.conversion_floor=best.score-profile.credible_window;
            var outcomes=new Dictionary<string,Outcome>();Plan challenger=null;
            for(uint iteration=0;root.Phase!="debt" && !profile.incumbent_only && iteration<profile.simulations && !s.Stop;iteration++)
            {
                var path=new List<Backup>();var actions=new List<string[]>();var partial=new List<string>();Plan plan=null;string first=null;
                // Each block confirms incumbent/challenger on two matched worlds.
                // Replica warmups differ, while incumbent scenario IDs are common.
                uint scenario=(iteration/4)*2+(iteration%4)/2;uint worldSeed=MixSeed(seed,scenario);string scenarioId=worldSeed.ToString();
                var rng=new Random((int)(MixSeed(worldSeed,(uint)profile.partition_index)&0x7fffffff));
                var opponentRng=new Random((int)(worldSeed&0x7fffffff));var styles=Enumerable.Range(0,4).Select(_=>Tactics.Styles[opponentRng.Next(Tactics.Styles.Length)]).ToArray();
                var forced=iteration%4==0 || iteration%4==2?baseline:iteration%4==3?challenger:null;
                var forcedIds=forced==null?null:forced.actions.SelectMany(a=>a).ToArray();int forcedIndex=0;
                try
                {
                    timer=System.Diagnostics.Stopwatch.StartNew();var world=new Host((int)(worldSeed&0x7fffffff),Belief.Setup(s.root,composition,worldSeed));s.nodes++;Timing(log,"world_setup",timer);
                    if(!world.Choices().Select(c=>c.id).SequenceEqual(rootIds))throw new Exception("root_private_world_invariance");
                    double[] values=null;string lastOpponent=null;int opponentTurns=0;bool finished=false;
                    for(int step=0;step<profile.max_choices;step++)
                    {
                        if(s.Stop)throw new OperationCanceledException();
                        if(world.Terminal){timer=System.Diagnostics.Stopwatch.StartNew();values=LeafValues((JObject)Wire.Token(world.Observe(s.seat)),true);Timing(log,"leaf",timer);break;}
                        int actor=world.Actor;timer=System.Diagnostics.Stopwatch.StartNew();var choices=world.Choices();
                        bool deterministic=choices.Count==1 && path.Count>0 && !finished;
                        var permitted=deterministic?null:(JObject)Wire.Token(world.Observe(actor));Timing(log,"legal_observation",timer);
                        if(choices.Count==0)throw new Exception("empty_nonterminal_legal_set");
                        if(finished && world.Phase=="action" && (actor==s.seat || (profile.leaf_policy=="settled_reply_prefix" && opponentTurns>=profile.opponent_turn_limit)))
                        {timer=System.Diagnostics.Stopwatch.StartNew();values=LeafValues((JObject)Wire.Token(world.Observe(s.seat)),false);Timing(log,"leaf",timer);break;}
                        Choice choice;
                        if(deterministic)
                        {
                            choice=choices[0];
                            if(forcedIds!=null){if(forcedIndex>=forcedIds.Length || forcedIds[forcedIndex++]!=choice.id)throw new InvalidOperationException("forced_deterministic_mismatch");}
                            partial.Add(choice.id);
                        }
                        else if(!finished && actor==s.seat && Turn(permitted)==initialTurn)
                        {
                            timer=System.Diagnostics.Stopwatch.StartNew();string key=step==0?firstKey:InformationKey(permitted,choices);Timing(log,"information_key",timer);
                            SearchNode node;if(!tree.TryGetValue(key,out node))
                            {
                                if(tree.Count>=16384)throw new OperationCanceledException();timer=System.Diagnostics.Stopwatch.StartNew();
                                node=new SearchNode{legal=choices.Select(c=>c.id).ToArray(),priors=Priors(permitted,choices,profile,s.intent)};tree.Add(key,node);Timing(log,"priors",timer);
                            }
                            if(!node.legal.SequenceEqual(choices.Select(c=>c.id)))throw new Exception("information_node_legal_mismatch");
                            var active=Expanded(node,choices,profile,key==firstKey,rng,log);int index;
                            if(forcedIds!=null)
                            {
                                if(forcedIndex>=forcedIds.Length)throw new InvalidOperationException("incomplete_forced_turn");
                                string forcedIdentity=forcedIds[forcedIndex++];index=choices.FindIndex(c=>c.id==forcedIdentity);if(index<0)throw new InvalidOperationException("forced_turn_changed_information");
                            }
                            else if(iteration%4==1 && key==firstKey && challenger==null)
                            {
                                var distinct=node.ranked.GroupBy(i=>choices[i].kind+"/"+choices[i].industry+"/"+choices[i].target+"/"+choices[i].merchant).Select(g=>g.First()).Take(profile.root_top_k).ToArray();
                                index=distinct[profile.partition_index%distinct.Length];
                                if(choices[index].id==baseline.actions[0][0] && distinct.Length>1)index=distinct[(profile.partition_index+1)%distinct.Length];
                            }
                            else index=Select(node,choices,active,profile,s.seat,log);
                            choice=choices[index];SearchEdge edge;if(!node.edges.TryGetValue(choice.id,out edge)){edge=new SearchEdge();node.edges.Add(choice.id,edge);}
                            path.Add(new Backup{node=node,edge=edge});if(first==null)first=choice.id;partial.Add(choice.id);
                        }
                        else
                        {
                            timer=System.Diagnostics.Stopwatch.StartNew();string turn=Turn(permitted);
                            if(world.Phase=="action" && turn!=lastOpponent){opponentTurns++;lastOpponent=turn;}
                            choice=MaskedRollout(permitted,choices,styles[actor]);log.opponent_choices++;Timing(log,"opponent_policy",timer);
                        }
                        world.SearchApply(choice.id);s.nodes++;
                        if(!finished && actor==s.seat && partial.Count>0 && world.ActionComplete){actions.Add(partial.ToArray());partial.Clear();}
                        if(world.ActionComplete && world.Remaining==0)
                        {
                            if(!finished && actor==s.seat)
                            {
                                timer=System.Diagnostics.Stopwatch.StartNew();string signature=string.Join(";",actions.Select(a=>string.Join(",",a)));
                                Outcome prior;if(outcomes.TryGetValue(signature,out prior))plan=prior.plan;else plan=ScoreWorld(s,actions,world);
                                finished=true;Timing(log,"root_completion",timer);if(s.Stop)throw new OperationCanceledException();
                            }
                            if(!world.Terminal)world.Advance();
                        }
                    }
                    if(values==null || plan==null || path.Count==0 || s.Stop)throw new OperationCanceledException();
                    if(!world.Terminal && plan.ledger.sale_tempo<0)
                    {
                        // Same action-opportunity cost as the complete incumbent.
                        // Reweight raw heuristic utilities before any selection-Q
                        // transform. Official terminal win shares stay exact.
                        values[s.seat]*=Math.Exp(plan.ledger.sale_tempo/20);
                        double total=values.Sum();for(int seat=0;seat<4;seat++)values[seat]/=total;
                    }
                    if(values.Length!=4 || values.Any(x=>double.IsNaN(x) || double.IsInfinity(x) || x<0 || x>1))throw new Exception("four_seat_value_contract");
                    foreach(var entry in path){entry.node.visits++;foreach(var edge in entry.node.edges.Values)edge.available++;entry.edge.visits++;for(int seat=0;seat<4;seat++){entry.edge.returns[seat]+=values[seat];entry.node.returns[seat]+=values[seat];}}
                    log.completed++;log.maximum_intervening_turns=Math.Max(log.maximum_intervening_turns,opponentTurns);
                    Outcome outcome;if(!outcomes.TryGetValue(plan.signature,out outcome))
                    {outcome=new Outcome{plan=plan,first=first,evidence=new CompleteEvidence{signature=plan.signature,first=first,plan=plan}};outcomes.Add(plan.signature,outcome);finals.Add(plan);log.plans.Add(plan.signature,outcome.evidence);}
                    var evidence=outcome.evidence;evidence.visits++;for(int seat=0;seat<4;seat++)evidence.returns[seat]+=values[seat];evidence.scenarios[scenarioId]=values[s.seat];
                    if(iteration%4==1)challenger=plan;
                    Outcome control;if(outcomes.TryGetValue(baseline.signature,out control) && control.evidence.scenarios.Count>=profile.minimum_plan_visits)
                    {
                        var safe=SafeOutcomes(outcomes.Values.ToList(),profile).Where(x=>x.evidence.visits>=profile.minimum_plan_visits &&
                            x.evidence.scenarios.Keys.Intersect(control.evidence.scenarios.Keys).Count()>=profile.minimum_plan_visits).ToList();
                        if(profile.cash_policy==1 && baseline.cash!=null)safe=safe.Where(x=>x.plan.cash.shortfall<=baseline.cash.shortfall && (baseline.cash.future_credit_blocked || !x.plan.cash.future_credit_blocked)).ToList();
                        if(log.conversion_floor.HasValue)safe=safe.Where(x=>x.plan.score>=log.conversion_floor.Value).ToList();
                        if(baseline.ledger.win_share.HasValue)safe=safe.Where(x=>x.plan.ledger.win_share.HasValue && x.plan.ledger.win_share>=baseline.ledger.win_share && x.plan.ledger.rank<=baseline.ledger.rank).ToList();
                        Func<Outcome,double> advantage=x=>x.evidence.scenarios.Where(kv=>control.evidence.scenarios.ContainsKey(kv.Key)).Average(kv=>kv.Value-control.evidence.scenarios[kv.Key]);
                        var selected=safe.OrderByDescending(advantage).ThenByDescending(x=>x.plan.score).ThenByDescending(x=>Math.Min(.02,Math.Max(-.02,x.plan.preference/100))).ThenBy(x=>x.plan.signature,StringComparer.Ordinal).FirstOrDefault();
                        if(selected!=null && selected.plan.signature!=best.signature && advantage(selected)>=0){best=selected.plan;log.fallback=false;incumbent?.Invoke(best);}
                    }
                }
                catch(OperationCanceledException){log.interrupted++;break;}
                catch(InvalidOperationException){s.failures++;log.interrupted++;}
            }
            SearchNode rootNode;if(tree.TryGetValue(firstKey,out rootNode)){log.root=rootNode.edges;log.expansion=rootNode.edges.Keys.OrderBy(x=>x,StringComparer.Ordinal).ToList();log.revisited_root_edges=rootNode.edges.Values.Count(e=>e.visits>=2);}
            log.revisited_plans=outcomes.Values.Count(x=>x.evidence.visits>=2);Outcome baseEvidence;if(outcomes.TryGetValue(baseline.signature,out baseEvidence))log.incumbent_visits=baseEvidence.evidence.visits;
            if(rootNode!=null && !rootNode.ranked.Take(profile.root_top_k).Any(i=>rootNode.legal[i]==best.actions[0][0]))log.selected_outside_k=1;
            return new Decision{profile=profile.id,mode="masked_ismcts",observation_hash=Wire.Hash(observation),status=cancel.IsCancellationRequested?"cancelled_with_incumbent":s.Stop?"budget":"complete",nodes=s.nodes,failures=s.failures,elapsed_ms=s.clock.Elapsed.TotalMilliseconds,overrun=!fixedWork && s.clock.Elapsed.TotalMilliseconds>profile.milliseconds,selected=best,alternatives=finals.GroupBy(p=>p.signature).Select(g=>g.First()).Take(16).ToArray(),search=log};
        }
    }
}
