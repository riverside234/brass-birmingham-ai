// Authored Brass adaptation of OpenSpiel's Apache-2.0 ISMCTS mechanics.
// Copyright 2019 DeepMind (reference mechanics). See NOTICE.md and LICENSE-OpenSpiel.
using System;
using System.Linq;
using System.Collections.Generic;
using System.Threading;
using Newtonsoft.Json.Linq;
using BrassHost;
// Shared execution for game and versioned IL/self-play search runtimes.
namespace BrassLegacyAI
{
    public sealed partial class Profile { public bool deadline_admission=true; }
    public sealed class MaskedTelemetry
    {
        public string format="masked-ismcts/1.1.0",horizon="settled-reply-prefix",value_units="bounded-relative-heuristic",evaluator_kind="heuristic",q_transform;
        public int revisited_root_edges,revisited_plans,incumbent_visits,normalization_uses,unseen_competitions,legal_count,eligible_count;
        public bool fallback=true;
        public int workers=1;
        public int network_candidates,network_completed,network_rejected;
        public int deadline_stops;
        public int skipped_leaf_observations,reused_root_choices;
        public double deadline_remaining_ms,simulation_estimate_ms;
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
            if(root.Phase=="action")finals=SearchIncumbents(s,root,target,log);
            else
            {
                var first=MaskedRollout(s.root,legal,profile.tactic);var action=root.Phase=="debt"?new[]{first.id}:GreedyFinish(s,new List<string>(),first.id,true);
                var plan=Score(s,new List<string[]>{action});while(plan.actions.Count<target)plan=Complete(s,plan,target);finals=new List<Plan>{plan};
            }
            var baseline=Choose(finals,profile);var best=baseline;log.incumbent_signature=baseline.signature;incumbent?.Invoke(best);Timing(log,"incumbent",timer);
            var tree=new Dictionary<string,SearchNode>();string firstKey=InformationKey(s.root,legal),initialTurn=Turn(s.root);
            if(profile.conversion_policy==1 && Conversion.Actions(s.root,s.seat)<=2 && root.Choices().Any(c=>c.kind=="sell"))log.conversion_floor=best.score-profile.credible_window;
            var outcomes=new Dictionary<string,Outcome>();Plan challenger=null;var admission=new DeadlineAdmission();
            for(uint iteration=0;root.Phase!="debt" && !profile.incumbent_only && iteration<profile.simulations && !s.Stop;iteration++)
            {
                if(!fixedWork && profile.deadline_admission && !admission.Admit(profile.milliseconds-s.clock.Elapsed.TotalMilliseconds))
                {log.deadline_stops++;log.deadline_remaining_ms=Math.Max(0,profile.milliseconds-s.clock.Elapsed.TotalMilliseconds);break;}
                var simulationClock=System.Diagnostics.Stopwatch.StartNew();
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
                    var worldRootChoices=world.Choices();
                    if(!worldRootChoices.Select(c=>c.id).SequenceEqual(rootIds))throw new Exception("root_private_world_invariance");
                    double[] values=null;string lastOpponent=null;int opponentTurns=0;bool finished=false;
                    for(int step=0;step<profile.max_choices;step++)
                    {
                        if(s.Stop)throw new OperationCanceledException();
                        if(world.Terminal){timer=System.Diagnostics.Stopwatch.StartNew();values=LeafValues((JObject)Wire.Token(world.Observe(s.seat)),true);Timing(log,"leaf",timer);break;}
                        int actor=world.Actor;
                        // At the SAME settled leaf boundary, no next-actor
                        // choices or private observation are consumed. Avoid
                        // constructing them before the root observer's value.
                        if(finished && world.Phase=="action" && (actor==s.seat || (profile.leaf_policy=="settled_reply_prefix" && opponentTurns>=profile.opponent_turn_limit)))
                        {log.skipped_leaf_observations++;timer=System.Diagnostics.Stopwatch.StartNew();values=LeafValues((JObject)Wire.Token(world.Observe(s.seat)),false);Timing(log,"leaf",timer);break;}
                        timer=System.Diagnostics.Stopwatch.StartNew();var choices=step==0?worldRootChoices:world.Choices();
                        if(step==0)log.reused_root_choices++;
                        bool deterministic=choices.Count==1 && path.Count>0 && !finished;
                        var permitted=deterministic?null:(JObject)Wire.Token(world.Observe(actor));Timing(log,"legal_observation",timer);
                        if(choices.Count==0)throw new Exception("empty_nonterminal_legal_set");
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
                    if(!world.Terminal && SearchOpportunity(plan)<0)
                    {
                        // Same action-opportunity cost as the complete incumbent.
                        // Reweight raw heuristic utilities before any selection-Q
                        // transform. Official terminal win shares stay exact.
                        values[s.seat]*=Math.Exp((SearchOpportunity(plan))/20);
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
                    admission.Observe(simulationClock.Elapsed.TotalMilliseconds);
                    Timing(log,"completed_simulation",simulationClock);
                }
                catch(OperationCanceledException){log.interrupted++;break;}
                catch(InvalidOperationException){s.failures++;log.interrupted++;}
            }
            log.simulation_estimate_ms=admission.Estimate;
            SearchNode rootNode;if(tree.TryGetValue(firstKey,out rootNode)){log.root=rootNode.edges;log.expansion=rootNode.edges.Keys.OrderBy(x=>x,StringComparer.Ordinal).ToList();log.revisited_root_edges=rootNode.edges.Values.Count(e=>e.visits>=2);}
            log.revisited_plans=outcomes.Values.Count(x=>x.evidence.visits>=2);Outcome baseEvidence;if(outcomes.TryGetValue(baseline.signature,out baseEvidence))log.incumbent_visits=baseEvidence.evidence.visits;
            if(rootNode!=null && !rootNode.ranked.Take(profile.root_top_k).Any(i=>rootNode.legal[i]==best.actions[0][0]))log.selected_outside_k=1;
            return new Decision{profile=profile.id,mode="masked_ismcts",observation_hash=Wire.Hash(observation),status=cancel.IsCancellationRequested?"cancelled_with_incumbent":s.Stop || log.deadline_stops>0?"budget":"complete",nodes=s.nodes,failures=s.failures,elapsed_ms=s.clock.Elapsed.TotalMilliseconds,overrun=!fixedWork && s.clock.Elapsed.TotalMilliseconds>profile.milliseconds,selected=best,alternatives=finals.GroupBy(p=>p.signature).Select(g=>g.First()).Take(16).ToArray(),search=log};
        }
    }
}
