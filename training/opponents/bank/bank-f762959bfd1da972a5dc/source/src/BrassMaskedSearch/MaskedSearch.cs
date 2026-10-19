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
            if(target>=1 && !s.Stop)
            {
                var legal=root.Choices();var values=Evaluator.ProposalValues(s.root,legal,true);
                var families=legal.Select((c,i)=>new{c,score=values[i]+Tactics.Bias(s.profile,s.intent,c,s.root)}).Where(x=>x.c.kind=="build")
                    .OrderByDescending(x=>x.score).GroupBy(x=>x.c.industry).Select(g=>g.First().c).ToArray();
                // Each private owner explores different industry families. Four
                // workers must not spend the whole short clock reproducing the
                // same two coal/iron plans while never comparing a scoring build.
                var builds=families.Skip(s.profile.partition_index).Where((c,i)=>i%s.profile.partition_count==0).Take(2);
                foreach(var build in builds)
                {
                    if(s.Stop || (!s.fixedWork && s.clock.Elapsed.TotalMilliseconds>s.profile.milliseconds*.35))break;
                    result.Add(FastPlan(s,target,build.id));
                    if(target>=2 && !s.Stop && Conversion.Sellable(Conversion.Player(s.root,s.seat)["mat"].Single(m=>(string)m["type"]==build.industry)["tiles"][0]) &&
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
                var setup=choices.Select((c,i)=>new{c,value=values[i]+Tactics.Bias(s.profile,s.intent,c,s.root)}).Where(x=>new[]{"loan","develop","network"}.Contains(x.c.kind))
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
        // Official IL/game v4 strategy (bank-3dc65253a610191d66a5).
        // Network/Scout candidates are deliberately not selected.
        static List<Plan> SearchIncumbents(Session s,Host root,int target,MaskedTelemetry log)
        {return FastIncumbents(s,root,target);}
        static double SearchOpportunity(Plan plan)
        {return plan.ledger.sale_tempo;}
    }
}
