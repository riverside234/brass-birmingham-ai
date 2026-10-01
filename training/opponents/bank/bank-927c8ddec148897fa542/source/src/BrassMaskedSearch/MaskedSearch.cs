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
    public sealed class MaskedTelemetry
    {
        public string format="masked-ismcts/1.0.0",horizon="actual-next-ordinary-root-turn",value_units="bounded-relative-heuristic";
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
            public int visits;
            public double sum;
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
            var scores=legal.Select(c=>Evaluator.Proposal(v,c,p.conversion_policy==1)+Tactics.Bias(p,intent,c)).ToArray();
            double max=scores.Max();var weights=scores.Select(x=>Math.Exp(Math.Max(-60,(x-max)/4))).ToArray();
            double sum=weights.Sum();return weights.Select(x=>.95*x/sum+.05/weights.Length).ToArray();
        }
        public static Choice MaskedRollout(JObject view,List<Choice> legal,string style)
        {
            if(legal.Count==0)throw new Exception("empty_nonterminal_legal_set");
            var p=new Profile{id="masked-rollout/"+style,tactic=style,intent_enabled=true,cash_policy=1,conversion_policy=1};
            var intent=Tactics.Refresh(view,p,null);
            var selected=legal.OrderByDescending(c=>c.kind=="beer"?Conversion.BeerPriority(view,c,0):Evaluator.Proposal(view,c,true)+Tactics.Bias(p,intent,c)).ThenBy(c=>c.id,StringComparer.Ordinal).First();
            if((string)view["context"]["phase"]=="action" && CashManagement.Project(view,(int)view["seat"]).shortfall>0)
                selected=legal.Where(c=>c.kind=="loan").OrderByDescending(c=>Evaluator.Proposal(view,c,true)).ThenBy(c=>c.id,StringComparer.Ordinal).FirstOrDefault()??selected;
            if(selected.kind=="develop" && legal.Any(c=>c.kind=="finish"))
            {
                var tile=Conversion.Player(view,(int)view["seat"])["mat"].Single(m=>(string)m["type"]==selected.industry)["tiles"].First();
                if((int)tile["level"]>1)selected=legal.Single(c=>c.kind=="finish");
            }
            return selected;
        }
        static double[] LeafValues(JObject permitted,bool terminal)
        {
            var values=new double[4];var seats=permitted["players"].Select(x=>(int)x["seat"]).ToArray();
            if(terminal){foreach(int seat in seats)values[seat]=Evaluator.Evaluate(permitted,seat,true).win_share.Value;return values;}
            var scores=seats.Select(seat=>Evaluator.Evaluate(permitted,seat,true).score-CashManagement.Project(permitted,seat).penalty).ToArray();
            double max=scores.Max();var weights=scores.Select(x=>Math.Exp(Math.Max(-60,(x-max)/20))).ToArray();
            for(int i=0;i<seats.Length;i++)values[seats[i]]=weights[i]/weights.Sum();return values;
        }
        static List<int> Expanded(SearchNode n,List<Choice> legal,Profile p,bool root,Random rng,MaskedTelemetry log)
        {
            int width=Math.Min(legal.Count,(root?p.root_top_k:p.interior_top_k)+(int)(2*Math.Sqrt(n.visits)));
            var ranked=Enumerable.Range(0,legal.Count).OrderByDescending(i=>n.priors[i]).ThenBy(i=>legal[i].id,StringComparer.Ordinal).ToArray();
            var result=new HashSet<int>(ranked.Take(width));
            // Setup, recovery, all industries, selling and beer sources survive a poor prior.
            foreach(var group in ranked.GroupBy(i=>legal[i].kind+(legal[i].kind=="build" || legal[i].kind=="develop"?"/"+legal[i].industry:legal[i].kind=="beer"?"/"+legal[i].merchant:"")))result.Add(group.First());
            foreach(int i in ranked)if(n.edges.ContainsKey(legal[i].id))result.Add(i);
            var tail=ranked.Where(i=>!result.Contains(i)).ToArray();
            if(tail.Length>0 && (n.edges.Count==0 || n.edges.Count%4==0))
            {
                double sum=tail.Sum(i=>.9*n.priors[i]+.1/legal.Count),pick=rng.NextDouble()*sum;
                int admitted=tail[tail.Length-1];foreach(int i in tail){pick-=.9*n.priors[i]+.1/legal.Count;if(pick<=0){admitted=i;break;}}
                result.Add(admitted);if(root)log.outside_k++;
            }
            if(root)log.deferred_prior_mass=ranked.Where(i=>!result.Contains(i)).Sum(i=>n.priors[i]);
            return result.OrderBy(i=>i).ToList();
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
        public Decision Decide(JObject observation,List<Choice> legal,Profile profile,bool fixedWork=false,uint seed=1,CancellationToken cancel=default(CancellationToken),Action<Plan> incumbent=null,IntentState intent=null)
        {
            var s=new Session{root=(JObject)observation.DeepClone(),profile=profile,fixedWork=fixedWork,cancel=cancel,seat=(int)observation["seat"]};
            if((int)observation["context"]["actor"]!=s.seat || observation["players"].Count()!=profile.player_count)throw new Exception("actor_or_player_count");
            s.intent=Tactics.Refresh(s.root,profile,intent);s.setup=Belief.Setup(s.root,composition,seed);
            var root=s.World(new string[0]);var rootIds=legal.Select(c=>c.id).OrderBy(x=>x,StringComparer.Ordinal).ToArray();
            if(!root.Choices().Select(c=>c.id).SequenceEqual(rootIds))throw new Exception("legal_contract_mismatch");
            int target=root.Phase=="debt"?1:Math.Min(2,root.Remaining);
            List<Plan> finals;
            if(root.Phase=="action")finals=CashIncumbents(s,root,target,Complete(s,Actions(s,new List<string[]>(),1)[0],target));
            else
            {
                var first=MaskedRollout(s.root,legal,profile.tactic);
                var action=root.Phase=="debt"?new[]{first.id}:GreedyFinish(s,new List<string>(),first.id,true);
                var plan=Score(s,new List<string[]>{action});while(plan.actions.Count<target)plan=Complete(s,plan,target);
                finals=new List<Plan>{plan};
            }
            SaleIncumbents(s,root,target,finals);
            var best=Choose(finals,profile);incumbent?.Invoke(best);
            var log=new MaskedTelemetry();var tree=new Dictionary<string,SearchNode>();
            if(profile.conversion_policy==1 && Conversion.Actions(s.root,s.seat)<=2 && root.Choices().Any(c=>c.kind=="sell"))
                log.conversion_floor=best.score-profile.credible_window;
            if(root.Phase=="debt")return new Decision{profile=profile.id,mode="masked_ismcts",observation_hash=Wire.Hash(observation),status="forced_debt_incumbent",selected=best,alternatives=finals.ToArray(),nodes=s.nodes,elapsed_ms=s.clock.Elapsed.TotalMilliseconds,search=log};
            var outcomes=new Dictionary<string,Outcome>();string firstKey=null;string initialTurn=Turn(s.root);
            for(uint iteration=0;iteration<profile.simulations && !s.Stop;iteration++)
            {
                var path=new List<Backup>();var actions=new List<string[]>();var partial=new List<string>();Plan plan=null;string first=null;
                uint worldSeed=MixSeed(seed,(uint)(iteration+profile.partition_index*1000003));var rng=new Random((int)(worldSeed&0x7fffffff));
                var styles=Enumerable.Range(0,4).Select(_=>Tactics.Styles[rng.Next(Tactics.Styles.Length)]).ToArray();
                try
                {
                    var world=new Host((int)(worldSeed&0x7fffffff),Belief.Setup(s.root,composition,worldSeed));s.nodes++;
                    if(!world.Choices().Select(c=>c.id).SequenceEqual(rootIds) || InformationKey((JObject)Wire.Token(world.Observe(s.seat)),world.Choices())!=InformationKey(s.root,root.Choices()))throw new Exception("root_private_world_invariance");
                    double[] values=null;string lastOpponent=null;int opponentTurns=0;
                    for(int step=0;step<profile.max_choices;step++)
                    {
                        if(s.Stop)throw new OperationCanceledException();
                        if(world.Terminal){values=LeafValues((JObject)Wire.Token(world.Observe(s.seat)),true);break;}
                        int actor=world.Actor;var permitted=(JObject)Wire.Token(world.Observe(actor));
                        if(actor==s.seat && world.Phase=="action" && Turn(permitted)!=initialTurn)
                        {values=LeafValues(permitted,false);break;}
                        var choices=world.Choices();if(choices.Count==0)throw new Exception("empty_nonterminal_legal_set");
                        Choice choice;
                        if(actor==s.seat && Turn(permitted)==initialTurn)
                        {
                            string key=InformationKey(permitted,choices);if(firstKey==null)firstKey=key;
                            SearchNode node;if(!tree.TryGetValue(key,out node))
                            {
                                if(tree.Count>=16384)throw new OperationCanceledException();
                                node=new SearchNode{legal=choices.Select(c=>c.id).ToArray(),priors=Priors(permitted,choices,profile,s.intent)};tree.Add(key,node);
                            }
                            if(!node.legal.SequenceEqual(choices.Select(c=>c.id)))throw new Exception("information_node_legal_mismatch");
                            var active=Expanded(node,choices,profile,key==firstKey,rng,log);
                            var unseen=active.Where(i=>!node.edges.ContainsKey(choices[i].id)).ToArray();int index;
                            if(unseen.Length>0)
                            {
                                var tail=unseen.Where(i=>!Enumerable.Range(0,choices.Count).OrderByDescending(j=>node.priors[j]).Take(key==firstKey?profile.root_top_k:profile.interior_top_k).Contains(i)).ToArray();
                                var pool=tail.Length>0 && (node.edges.Count==0 || node.edges.Count%4==0)?tail:unseen;
                                double pick=rng.NextDouble()*pool.Sum(i=>.9*node.priors[i]+.1/choices.Count);index=pool[pool.Length-1];
                                foreach(int i in pool){pick-=.9*node.priors[i]+.1/choices.Count;if(pick<=0){index=i;break;}}
                                node.edges.Add(choices[index].id,new SearchEdge());
                            }
                            else index=active.OrderByDescending(i=>{var e=node.edges[choices[i].id];return (e.visits==0?1.0/profile.player_count:e.returns[s.seat]/e.visits)+profile.exploration*node.priors[i]*Math.Sqrt(Math.Max(1,node.visits))/(1+e.visits);}).ThenBy(i=>i).First();
                            choice=choices[index];path.Add(new Backup{node=node,edge=node.edges[choice.id]});
                            if(first==null)first=choice.id;partial.Add(choice.id);
                        }
                        else
                        {
                            // Opponents receive only their own view and legal choices.
                            // No root-private-conditioned optimized opponent node exists.
                            string turn=Turn(permitted);if(world.Phase=="action" && turn!=lastOpponent){opponentTurns++;lastOpponent=turn;}
                            choice=MaskedRollout(permitted,choices,styles[actor]);log.opponent_choices++;
                        }
                        world.SearchApply(choice.id);s.nodes++;
                        if(actor==s.seat && partial.Count>0 && world.ActionComplete){actions.Add(partial.ToArray());partial.Clear();}
                        if(world.ActionComplete && world.Remaining==0)
                        {
                            if(actor==s.seat && Turn(permitted)==initialTurn && plan==null)
                            {plan=Score(s,actions);if(s.Stop)throw new OperationCanceledException();}
                            world.Advance();
                        }
                    }
                    if(values==null || plan==null || path.Count==0)throw new OperationCanceledException();
                    if(s.Stop)throw new OperationCanceledException();
                    if(values.Length!=4 || values.Any(x=>double.IsNaN(x) || double.IsInfinity(x) || x<0 || x>1))throw new Exception("four_seat_value_contract");
                    // Transactional backup: an interrupted path contributes nothing.
                    foreach(var entry in path){entry.node.visits++;foreach(var edge in entry.node.edges.Values)edge.available++;entry.edge.visits++;for(int seat=0;seat<4;seat++)entry.edge.returns[seat]+=values[seat];}
                    log.completed++;log.maximum_intervening_turns=Math.Max(log.maximum_intervening_turns,opponentTurns);
                    Outcome outcome;if(!outcomes.TryGetValue(plan.signature,out outcome)){outcome=new Outcome{plan=plan,first=first};outcomes.Add(plan.signature,outcome);finals.Add(plan);}
                    outcome.visits++;outcome.sum+=values[s.seat];
                    var safe=SafeOutcomes(outcomes.Values.ToList(),profile);
                    // Keep the published rescue incumbent if all searched turns are less safe.
                    if(profile.cash_policy==1 && best.cash!=null)safe=safe.Where(x=>x.plan.cash.shortfall<=best.cash.shortfall && (best.cash.future_credit_blocked || !x.plan.cash.future_credit_blocked)).ToList();
                    if(log.conversion_floor.HasValue)safe=safe.Where(x=>x.plan.score>=log.conversion_floor.Value).ToList();
                    if(best.ledger.win_share.HasValue)safe=safe.Where(x=>x.plan.ledger.win_share.HasValue && x.plan.ledger.win_share>=best.ledger.win_share && x.plan.ledger.rank<=best.ledger.rank).ToList();
                    if(safe.Count>0)
                    {
                        var selected=safe.OrderByDescending(x=>tree[firstKey].edges[x.first].visits).ThenByDescending(x=>x.sum/x.visits+Math.Min(.02,Math.Max(-.02,x.plan.preference/100))).ThenBy(x=>x.plan.signature,StringComparer.Ordinal).First().plan;
                        if(selected.signature!=best.signature){best=selected;incumbent?.Invoke(best);}
                    }
                }
                catch(OperationCanceledException){log.interrupted++;break;}
                catch(InvalidOperationException){s.failures++;log.interrupted++;}
            }
            if(firstKey!=null){log.root=tree[firstKey].edges;log.expansion=tree[firstKey].edges.Keys.OrderBy(x=>x,StringComparer.Ordinal).ToList();var n=tree[firstKey];if(!n.legal.Select((x,i)=>new{x,i}).OrderByDescending(x=>n.priors[x.i]).Take(profile.root_top_k).Any(x=>x.x==best.actions[0][0]))log.selected_outside_k=1;}
            return new Decision{profile=profile.id,mode="masked_ismcts",observation_hash=Wire.Hash(observation),status=cancel.IsCancellationRequested?"cancelled_with_incumbent":s.Stop?"budget":"complete",nodes=s.nodes,failures=s.failures,elapsed_ms=s.clock.Elapsed.TotalMilliseconds,overrun=!fixedWork && s.clock.Elapsed.TotalMilliseconds>profile.milliseconds,selected=best,alternatives=finals.GroupBy(p=>p.signature).Select(g=>g.First()).Take(16).ToArray(),search=log};
        }
    }
}
