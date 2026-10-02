using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Newtonsoft.Json.Linq;
using BrassHost;

namespace BrassLegacyAI
{
    public static class RootForest
    {
        public static Decision Decide(JObject composition,JObject view,List<Choice> choices,Profile profile,int workers,uint seed=1,
            CancellationToken cancellation=default(CancellationToken),Action<Plan> publish=null,IntentState intent=null)
        {
            if(workers<1 || workers>4)throw new Exception("root_worker_allowance");
            if(workers==1)return new Planner(composition).Decide(view,choices,profile,false,seed,cancellation,publish,intent);
            var clock=Stopwatch.StartNew();var queue=new HeuristicBatchQueue();var sync=new object();Plan incumbent=null;
            Action<Plan> update=plan=>{lock(sync){if(incumbent==null || Planner.Choose(new List<Plan>{incumbent,plan},profile).signature!=incumbent.signature)
                {incumbent=JObject.FromObject(plan).ToObject<Plan>();publish?.Invoke(incumbent);}}};
            var jobs=Enumerable.Range(0,workers).Select(index=>Task.Factory.StartNew(()=>{
                var cfg=Profile.Read(JObject.FromObject(profile));cfg.partition_index=index;cfg.partition_count=workers;
                cfg.milliseconds=Math.Max(0,profile.milliseconds-clock.Elapsed.TotalMilliseconds);
                using(queue.Bind())return new Planner(composition).Decide((JObject)view.DeepClone(),JArray.FromObject(choices).ToObject<List<Choice>>(),
                    cfg,false,seed,cancellation,update,intent==null?null:JObject.FromObject(intent).ToObject<IntentState>());
            },CancellationToken.None,TaskCreationOptions.LongRunning,TaskScheduler.Default)).ToArray();
            // Cancellation/deadline is not retirement: await every owner before
            // releasing snapshots, reply channels, or presentation resources.
            try{Task.WaitAll(jobs);}catch{foreach(var job in jobs)try{job.Wait();}catch{}throw;}
            var parts=jobs.Select(j=>j.Result).ToArray();var log=new MaskedTelemetry{workers=workers,q_transform=profile.q_transform};
            var plans=parts.SelectMany(d=>d.alternatives.Concat(new[]{d.selected})).GroupBy(p=>p.signature).ToDictionary(g=>g.Key,g=>g.First());
            var baseline=Planner.Choose(parts.Select(d=>plans[d.search.incumbent_signature]).ToList(),profile);
            var scenarios=new Dictionary<string,Dictionary<string,List<double>>>();
            foreach(var part in parts)
            {
                var s=part.search;if(s.opponent_nodes!=0)throw new Exception("opponent_tree_merge");
                log.completed+=s.completed;log.interrupted+=s.interrupted;log.opponent_choices+=s.opponent_choices;
                log.network_candidates+=s.network_candidates;log.network_completed+=s.network_completed;log.network_rejected+=s.network_rejected;
                foreach(var timing in s.timing_ms){double value;log.timing_ms.TryGetValue(timing.Key,out value);log.timing_ms[timing.Key]=value+timing.Value;}
                log.maximum_intervening_turns=Math.Max(log.maximum_intervening_turns,s.maximum_intervening_turns);
                foreach(var item in s.root)
                {
                    SearchEdge e;if(!log.root.TryGetValue(item.Key,out e)){e=new SearchEdge();log.root.Add(item.Key,e);}
                    e.visits+=item.Value.visits;e.available+=item.Value.available;for(int seat=0;seat<4;seat++)e.returns[seat]+=item.Value.returns[seat];
                }
                foreach(var item in s.plans)
                {
                    var source=item.Value;CompleteEvidence e;
                    if(source.visits<=0 || source.returns.Length!=4 || source.plan.signature!=item.Key)throw new Exception("incomplete_forest_evidence");
                    plans[item.Key]=source.plan;
                    if(!log.plans.TryGetValue(item.Key,out e))
                    {e=new CompleteEvidence{signature=item.Key,first=source.first,plan=source.plan};log.plans.Add(item.Key,e);scenarios.Add(item.Key,new Dictionary<string,List<double>>());}
                    e.visits+=source.visits;for(int seat=0;seat<4;seat++)e.returns[seat]+=source.returns[seat];
                    foreach(var scenario in source.scenarios)
                    {List<double> values;if(!scenarios[item.Key].TryGetValue(scenario.Key,out values)){values=new List<double>();scenarios[item.Key].Add(scenario.Key,values);}values.Add(scenario.Value);}
                }
            }
            foreach(var e in log.plans)foreach(var sample in scenarios[e.Key])e.Value.scenarios.Add(sample.Key,sample.Value.Average());
            log.incumbent_signature=baseline.signature;log.incumbent_visits=log.plans.ContainsKey(baseline.signature)?log.plans[baseline.signature].visits:0;
            CompleteEvidence control;Plan selected=baseline;
            if(log.plans.TryGetValue(baseline.signature,out control) && control.visits>=profile.minimum_plan_visits)
            {
                double? floor=parts.Where(d=>d.search.conversion_floor.HasValue).Select(d=>d.search.conversion_floor).DefaultIfEmpty(null).Max();
                var credible=log.plans.Values.Where(e=>e.visits>=profile.minimum_plan_visits &&
                    e.scenarios.Keys.Intersect(control.scenarios.Keys).Count()>=profile.minimum_plan_visits &&
                    e.plan.cash.shortfall<=baseline.cash.shortfall && (baseline.cash.future_credit_blocked || !e.plan.cash.future_credit_blocked) &&
                    (!floor.HasValue || e.plan.score>=floor.Value) &&
                    (!baseline.ledger.win_share.HasValue || (e.plan.ledger.win_share>=baseline.ledger.win_share && e.plan.ledger.rank<=baseline.ledger.rank)));
                Func<CompleteEvidence,double> advantage=e=>e.scenarios.Where(s=>control.scenarios.ContainsKey(s.Key)).Average(s=>s.Value-control.scenarios[s.Key]);
                var best=credible.OrderByDescending(advantage).ThenByDescending(e=>e.plan.score).ThenByDescending(e=>e.plan.preference).ThenBy(e=>e.signature,StringComparer.Ordinal).FirstOrDefault();
                if(best!=null && advantage(best)>=0)selected=best.plan;
                log.conversion_floor=floor;
            }
            log.fallback=selected.signature==baseline.signature;log.revisited_root_edges=log.root.Values.Count(e=>e.visits>=2);
            log.revisited_plans=log.plans.Values.Count(e=>e.visits>=2);log.expansion=log.root.Keys.OrderBy(x=>x,StringComparer.Ordinal).ToList();
            log.cpu_batching=queue.Report();publish?.Invoke(selected);
            return new Decision{profile=profile.id,mode="masked_ismcts",observation_hash=Wire.Hash(view),status=cancellation.IsCancellationRequested?"cancelled_with_incumbent":"forest_complete",
                nodes=parts.Sum(d=>d.nodes),failures=parts.Sum(d=>d.failures),elapsed_ms=clock.Elapsed.TotalMilliseconds,overrun=clock.Elapsed.TotalMilliseconds>profile.milliseconds,
                selected=selected,alternatives=plans.Values.OrderByDescending(p=>p.score).Take(16).ToArray(),search=log};
        }
    }
}
