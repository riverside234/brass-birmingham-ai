using System;
using System.Linq;
using System.Collections.Generic;
using System.Diagnostics;
using System.Threading;
using Newtonsoft.Json.Linq;
using BrassHost;
namespace BrassLegacyAI
{
    public sealed class ReplyEstimate
    {
        public string signature;
        public int samples;
        public double mean;
        public double? variance;
        public double[] seats;
        public bool terminal;
    }
    public sealed class ReplyTelemetry
    {
        public string format="legacy-replies/1.0.0",status="s1_incumbent",mixture="masked-tactical-mixture/1.0.0",horizon="next ordinary root turn or terminal";
        public int worlds,choices,pilot_batches,fresh_batches,discarded_batches,failures;
        public List<string> errors=new List<string>();
        public List<int> intervening_turns=new List<int>();
        public List<string> leaf_eras=new List<string>();
        public List<uint> pilot_seeds=new List<uint>(),fresh_seeds=new List<uint>();
        public ReplyEstimate[] estimates=new ReplyEstimate[0];
    }
    public static class ReplySearch
    {
        // This rollout interface cannot receive a host, root hand or opponent intent.
        public static Choice RolloutChoice(JObject view,List<Choice> legal,string style)
        {
            var p=new Profile{id="rollout-"+style,tactic=style,intent_enabled=true,style_weight=1};
            var intent=Tactics.Refresh(view,p,null);
            var ranked=legal.OrderByDescending(c=>Evaluator.Proposal(view,c)+Tactics.Bias(p,intent,c)).ThenBy(c=>c.id,StringComparer.Ordinal).ToArray();
            var best=ranked.First();
            if(best.kind=="develop" && legal.Any(c=>c.kind=="finish"))
            {
                var tile=view["players"][(int)view["seat"]]["mat"].Single(m=>(string)m["type"]==best.industry)["tiles"].First();
                if((int)tile["level"]>1)return legal.Single(c=>c.kind=="finish");
            }
            return best;
        }
        public static double[] Values(JObject view,bool terminal)
        {
            return Enumerable.Range(0,4).Select(s=>{
                var d=Evaluator.Evaluate(view,s);
                return terminal?d.win_share.Value+0.05*(4-d.rank.Value):d.score;
            }).ToArray();
        }
        static uint Mix(uint seed,uint index){uint x=seed+0x9e3779b9u*(index+1);x^=x<<13;x^=x>>17;x^=x<<5;return x==0?1:x;}
        public static double[] Scenario(JObject view,JObject composition,Plan plan,uint seed,ReplyTelemetry log,Func<bool> stop)
        {
            if(stop())throw new OperationCanceledException();
            int root=(int)view["seat"];
            var world=new Host((int)(seed&0x7fffffff),Belief.Setup(view,composition,seed));log.worlds++;
            // Exactly the same current turn in every scenario; never optimize action
            // two using a sampled hand or a future draw.
            foreach(var action in plan.actions)foreach(var id in action){world.SearchApply(id);log.choices++;}
            world.Advance();
            var styles=Enumerable.Range(0,4).Select(s=>Tactics.Styles[Mix(seed,(uint)s)%Tactics.Styles.Length]).ToArray();
            int turns=0;string last=null;
            for(int step=0;step<512;step++)
            {
                if(world.Terminal || (world.Actor==root && world.Phase=="action"))
                {
                    log.intervening_turns.Add(turns);
                    log.leaf_eras.Add(world.Game.GameProgressInformation.CurrentGameEra.ToString());
                    return Values((JObject)Wire.Token(world.Observe(root)),world.Terminal);
                }
                if(stop())throw new OperationCanceledException();
                int actor=world.Actor;var permitted=(JObject)Wire.Token(world.Observe(actor));
                var context=permitted["context"];
                string key=(string)context["era"]+":"+context["round"]+":"+context["turn"]+":"+actor;
                if(world.Phase=="action" && key!=last){turns++;last=key;if(turns>6)throw new Exception("reply_horizon_exceeded");}
                var choice=RolloutChoice(permitted,world.Choices(),styles[actor]);
                world.SearchApply(choice.id);log.choices++;
                if(world.ActionComplete && world.Remaining==0)world.Advance();
            }
            throw new Exception("reply_progress_limit");
        }
        static ReplyEstimate Estimate(Plan plan,List<double[]> values,bool terminal,int seat)
        {
            var v=values.Select(x=>x[seat]).ToArray();double mean=v.Average();
            return new ReplyEstimate{signature=plan.signature,samples=v.Length,mean=mean,variance=v.Length>1?(double?)(v.Sum(x=>(x-mean)*(x-mean))/(v.Length-1)):null,seats=Enumerable.Range(0,4).Select(s=>values.Average(x=>x[s])).ToArray(),terminal=terminal};
        }
        public static Decision Decide(Planner planner,JObject view,List<Choice> legal,Profile profile,JObject composition,bool fixedWork,uint seed,CancellationToken cancel,Action<Plan> publish,IntentState intent)
        {
            var clock=Stopwatch.StartNew();var cfg=Profile.Read(JObject.FromObject(profile));cfg.mode="S1";
            cfg.work_limit=Math.Max(1,(int)(profile.work_limit*profile.proposal_fraction));cfg.milliseconds=profile.milliseconds*profile.proposal_fraction;
            var d=planner.Decide(view,legal,cfg,fixedWork,seed,cancel,publish,intent);d.mode="S2";var log=new ReplyTelemetry();d.replies=log;
            if((string)view["context"]["phase"]!="action")return d;
            var candidates=new[]{d.selected}.Concat(d.alternatives??new Plan[0]).Concat(d.family_best==null?new Plan[0]:d.family_best.Values.ToArray()).GroupBy(p=>p.signature).Select(g=>g.First()).OrderByDescending(p=>p.score).Take(profile.reply_candidates).ToList();
            if(!candidates.Any(p=>p.signature==d.selected.signature)){candidates[candidates.Count-1]=d.selected;}
            bool terminal=(string)view["context"]["era"]=="RailEra" && (int)view["context"]["round"]==8;
            Func<bool> stop=()=>cancel.IsCancellationRequested || log.failures>0 || log.choices>=profile.reply_work || (!fixedWork && clock.Elapsed.TotalMilliseconds>=profile.milliseconds);
            var pilot=candidates.ToDictionary(p=>p.signature,p=>new List<double[]>());
            Func<List<Plan>,uint,Dictionary<string,List<double[]>>,bool> batch=(plans,sample,values)=>{
                var pending=new Dictionary<string,double[]>();
                try{foreach(var p in plans)pending[p.signature]=Scenario(view,composition,p,sample,log,stop);}
                catch(OperationCanceledException){log.discarded_batches++;return false;}
                catch(Exception ex){log.discarded_batches++;log.failures++;log.errors.Add(ex.ToString());log.status="reply_failure_s1_incumbent";return false;}
                foreach(var p in plans)values[p.signature].Add(pending[p.signature]);return true;
            };
            for(uint i=0;i<profile.reply_scenarios;i++)
            {
                uint sample=Mix(seed,i);if(!batch(candidates,sample,pilot))break;
                log.pilot_batches++;log.pilot_seeds.Add(sample);
            }
            if(log.pilot_batches>0)
            {
                int seat=(int)view["seat"];
                var finalists=candidates.OrderByDescending(p=>pilot[p.signature].Average(v=>v[seat])).ThenBy(p=>p.signature,StringComparer.Ordinal).Take(Math.Min(2,candidates.Count)).ToList();
                var fresh=finalists.ToDictionary(p=>p.signature,p=>new List<double[]>());
                for(uint i=0;i<profile.reply_confirmation;i++)
                {uint sample=Mix(seed,65536+i);if(!batch(finalists,sample,fresh))break;log.fresh_batches++;log.fresh_seeds.Add(sample);}
                if(log.fresh_batches>0)
                {
                    log.estimates=finalists.Select(p=>Estimate(p,fresh[p.signature],terminal,seat)).ToArray();
                    double best=log.estimates.Max(e=>e.mean);double window=terminal?0:profile.credible_window;
                    var eligible=finalists.Where(p=>log.estimates.Single(e=>e.signature==p.signature).mean>=best-window);
                    d.selected=eligible.OrderByDescending(p=>log.estimates.Single(e=>e.signature==p.signature).mean+(terminal?0:p.preference)).ThenBy(p=>p.signature,StringComparer.Ordinal).First();
                    log.status="confirmed";publish?.Invoke(d.selected);
                }
                else log.status="no_fresh_batch_s1_incumbent";
            }
            d.elapsed_ms=clock.Elapsed.TotalMilliseconds;d.overrun=!fixedWork && d.elapsed_ms>profile.milliseconds;
            d.status=cancel.IsCancellationRequested?"cancelled_with_incumbent":log.status;return d;
        }
    }
}
