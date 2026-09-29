using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading;
using Newtonsoft.Json.Linq;
using BrassHost;
namespace BrassLegacyAI
{
    public sealed class Profile
    {
        public string id,mode="S1";
        public int player_count=4,work_limit=160,first_width=24,beam=10,second_width=12;
        public double milliseconds=1000,credible_window=1.5,rail_preference=0;
        public string tactic="balanced";
        public bool intent_enabled=false;
        public double style_weight=1,proposal_fraction=0.40;
        public int reply_candidates=4,reply_scenarios=2,reply_confirmation=2,reply_work=2048;
        public static Profile Read(JObject j)
        {
            var p=j.ToObject<Profile>();
            if(new[]{p.credible_window,p.rail_preference,p.style_weight,p.proposal_fraction}.Any(x=>double.IsNaN(x) || double.IsInfinity(x)))throw new Exception("nonfinite_profile");
            if(p.player_count!=4 || (p.mode!="S0" && p.mode!="S1" && p.mode!="S2") || p.work_limit<1 || p.milliseconds<0 || double.IsNaN(p.milliseconds) || double.IsInfinity(p.milliseconds) || p.first_width<1 || p.beam<1 || p.second_width<1 || string.IsNullOrEmpty(p.id) || !Tactics.Styles.Contains(p.tactic) || p.reply_candidates<1 || p.reply_scenarios<1 || p.reply_confirmation<1 || p.reply_work<1 || p.proposal_fraction<=0 || p.proposal_fraction>=1 || p.credible_window<0 || p.style_weight<0)throw new Exception("profile_contract");
            return p;
        }
    }
    public sealed class Plan
    {
        public List<string[]> actions=new List<string[]>();
        public double score,preference;
        public Ledger ledger;
        public IntentState intent;
        public string signature {get{return string.Join(";",actions.Select(a=>string.Join(",",a)));}}
    }
    public sealed class Decision
    {
        public string format="legacy-decision/0.1.0",profile,mode,observation_hash,status;
        public int nodes,failures;
        public double elapsed_ms;
        public bool overrun;
        public Plan selected;
        public Plan[] alternatives;
        public Dictionary<string,Plan> family_best;
        public ReplyTelemetry replies;
    }
    // Each request is independent. Policies receive only observation + semantic legal IDs.
    // There is no access to a real Host, BoardHelper, historical stock rule, or RNG.
    public sealed class Planner
    {
        readonly JObject composition;
        public Planner(JObject composition){this.composition=(JObject)composition.DeepClone();}
        sealed class Session
        {
            public JObject root,setup;
            public Profile profile;
            public int seat,nodes,failures;
            public bool fixedWork;
            public Stopwatch clock=Stopwatch.StartNew();
            public CancellationToken cancel;
            public IntentState intent;
            public bool Stop {get{return cancel.IsCancellationRequested || nodes>=profile.work_limit || (!fixedWork && clock.Elapsed.TotalMilliseconds>=profile.milliseconds);}}
            public Host World(IEnumerable<string> path)
            {
                nodes++;var h=new Host(1,setup);
                foreach(var id in path)h.SearchApply(id);return h;
            }
            public JObject View(Host h){return (JObject)Wire.Token(h.Observe(seat));}
        }
        static IEnumerable<Choice> Proposals(JObject view,List<Choice> legal,int limit,Profile profile,IntentState intent)
        {
            // Retain semantic alternatives; do not let eight interchangeable discard
            // cards consume the beam. Two discard options protect enabling continuations.
            var ordered=legal.OrderByDescending(c=>Evaluator.Proposal(view,c)+Tactics.Bias(profile,intent,c)).ThenBy(c=>c.id,StringComparer.Ordinal).ToArray();
            if(limit==1)return ordered.Take(1);
            var distinct=ordered.GroupBy(c=>c.kind+"|"+c.target+"|"+c.industry+"|"+c.merchant+(c.kind=="scout"?"|"+c.id:"")).SelectMany(g=>g.Take(cappedCards(g.Key))).ToArray();
            var result=new List<Choice>();
            result.AddRange(distinct.Where(c=>c.kind=="sell").Take(1));
            result.AddRange(distinct.Where(c=>c.kind=="build").GroupBy(c=>c.industry).Select(g=>g.First()));
            foreach(string kind in new[]{"loan","network","develop","scout","pass"})
                result.AddRange(distinct.Where(c=>c.kind==kind).Take(kind=="develop"?2:1));
            foreach(var c in distinct)if(result.Count<limit && !result.Contains(c))result.Add(c);
            return result.Take(Math.Max(limit,11));
        }
        static int cappedCards(string key){return key.StartsWith("loan|") || key.StartsWith("develop|") || key.StartsWith("network|")?2:1;}
        static string[] GreedyFinish(Session s,List<string> prefix,string first,bool extend)
        {
            var path=new List<string>(prefix);path.Add(first);
            var h=s.World(path);int start=prefix.Count;
            for(int i=0;i<40;i++)
            {
                if(h.ActionComplete)return path.Skip(start).ToArray();
                var view=s.View(h);var choices=h.Choices();Choice c;
                if(!extend && choices.Any(x=>x.kind=="finish"))c=choices.Single(x=>x.kind=="finish");
                else
                {
                    c=choices.OrderByDescending(x=>Evaluator.Proposal(view,x)).ThenBy(x=>x.id,StringComparer.Ordinal).First();
                    // Finishing is an independent option. Extra development must clear
                    // a useful low tile, not discard a valuable high tile by habit.
                    if(c.kind=="develop" && choices.Any(x=>x.kind=="finish"))
                    {
                        var tile=view["players"][s.seat]["mat"].Single(m=>(string)m["type"]==c.industry)["tiles"].First();
                        if((int)tile["level"]>1)c=choices.Single(x=>x.kind=="finish");
                    }
                }
                path.Add(c.id);h.SearchApply(c.id);
            }
            throw new Exception("completion_limit");
        }
        static Plan Score(Session s,List<string[]> actions)
        {
            var h=s.World(actions.SelectMany(a=>a));var view=s.View(h);
            if(h.Remaining==0 && (string)view["context"]["era"]=="RailEra" && (int)view["context"]["round"]==8 && (int)view["context"]["turn"]==4)
                view=Evaluator.KnownFinalScore(view);
            var ledger=Evaluator.Evaluate(view,s.seat);var p=new Plan{actions=actions,ledger=ledger,score=ledger.score};
            p.intent=s.intent;p.preference=Tactics.Preference(s.profile,s.intent,actions);return p;
        }
        static List<Plan> Actions(Session s,List<string[]> before,int limit,bool firstStage=false)
        {
            var prefix=before.SelectMany(a=>a).ToList();var root=s.World(prefix);var view=s.View(root);var results=new List<Plan>();
            foreach(var first in Proposals(view,root.Choices(),limit,s.profile,s.intent))
            {
                if(s.Stop && results.Count>0)break;
                if(firstStage && results.Count>=8 && (s.nodes>=s.profile.work_limit*0.45 || (!s.fixedWork && s.clock.Elapsed.TotalMilliseconds>=s.profile.milliseconds*0.45)))break;
                foreach(bool extend in new[]{false,true})
                {
                    if(extend && (s.Stop || (first.kind!="sell" && first.kind!="network" && first.kind!="develop")))continue;
                    try
                    {
                        var action=GreedyFinish(s,prefix,first.id,extend);
                        var all=new List<string[]>(before){action};var plan=Score(s,all);
                        if(!results.Any(p=>p.signature==plan.signature))results.Add(plan);
                    }
                    catch(InvalidOperationException){s.failures++;}
                }
            }
            if(results.Count==0)throw new Exception("no_complete_candidate");
            return results.OrderByDescending(p=>p.score).ThenBy(p=>p.signature,StringComparer.Ordinal).ToList();
        }
        static Plan Complete(Session s,Plan p,int target)
        {
            if(p.actions.Count>=target)return p;
            // A valid full-turn incumbent is constructed even if the search budget has
            // expired. Its bounded validation overrun is reported, never hidden.
            return Actions(s,p.actions,1)[0];
        }
        public Decision Decide(JObject observation,List<Choice> legal,Profile profile,bool fixedWork=false,uint seed=1,CancellationToken cancel=default(CancellationToken),Action<Plan> incumbent=null,IntentState intent=null)
        {
            if(profile.mode=="S2")return ReplySearch.Decide(this,observation,legal,profile,composition,fixedWork,seed,cancel,incumbent,intent);
            var s=new Session{root=(JObject)observation.DeepClone(),profile=profile,fixedWork=fixedWork,cancel=cancel,seat=(int)observation["seat"]};
            if((int)observation["context"]["actor"]!=s.seat || observation["players"].Count()!=4)throw new Exception("actor_or_player_count");
            s.intent=Tactics.Refresh(s.root,profile,intent);
            s.setup=Belief.Setup(s.root,composition,seed);
            // Keep the public events needed for wildcard accounting. The evaluator uses
            // the current public board, not a repeated serialization of every old move.
            s.setup["position"]["history"]=new JArray(s.root["history"].Where(e=>(string)e["choice"]["kind"]=="scout" || ((string)e["choice"]["card"]??"").StartsWith("wild:")).Select(e=>e.DeepClone()));
            var root=s.World(new string[0]);
            if(!root.Choices().Select(c=>c.id).SequenceEqual(legal.OrderBy(c=>c.id,StringComparer.Ordinal).Select(c=>c.id)))throw new Exception("legal_contract_mismatch");
            if(root.Phase!="action")
            {
                var ordered=legal.OrderByDescending(c=>Evaluator.Proposal(s.root,c)).ThenBy(c=>c.id,StringComparer.Ordinal).ToArray();
                var selected=new Plan{intent=s.intent,actions=new List<string[]>{root.Phase=="debt"?new[]{ordered[0].id}:GreedyFinish(s,new List<string>(),ordered[0].id,true)}};
                return new Decision{profile=profile.id,mode=profile.mode,observation_hash=Wire.Hash(observation),status="forced_or_resumed",selected=selected,alternatives=new[]{selected},nodes=s.nodes,elapsed_ms=s.clock.Elapsed.TotalMilliseconds};
            }
            int target=profile.mode=="S1"?Math.Min(2,root.Remaining):1;
            // Seed a legal, useful incumbent before widening. No stock helper is called.
            var quick=Actions(s,new List<string[]>(),1);var best=Complete(s,quick[0],target);incumbent?.Invoke(best);
            var finals=new List<Plan>{best};
            if(!s.Stop)
            {
                var first=Actions(s,new List<string[]>(),profile.first_width,profile.mode=="S1");
                if(target==1)finals.AddRange(first);
                // Preserve setup families even when their one-action value is low.
                var beam=new List<Plan>();
                beam.AddRange(first.Take(3));
                foreach(string kind in new[]{"loan|","develop|","scout|","network|"})
                {var p=first.FirstOrDefault(x=>x.actions[0][0].StartsWith(kind));if(p!=null && !beam.Contains(p))beam.Add(p);}
                foreach(var p in first)if(beam.Count<profile.beam && !beam.Contains(p))beam.Add(p);
                foreach(var p in beam)
                {
                    if(s.Stop)break;
                    var candidates=target==1?new List<Plan>{p}:Actions(s,p.actions,profile.second_width);
                    finals.AddRange(candidates);
                    var next=Choose(finals,profile);
                    if(next.signature!=best.signature){best=next;incumbent?.Invoke(best);}
                }
            }
            best=Choose(finals,profile);
            return new Decision{profile=profile.id,mode=profile.mode,observation_hash=Wire.Hash(observation),status=cancel.IsCancellationRequested?"cancelled_with_incumbent":s.Stop?"budget":"complete",nodes=s.nodes,failures=s.failures,elapsed_ms=s.clock.Elapsed.TotalMilliseconds,overrun=!fixedWork && s.clock.Elapsed.TotalMilliseconds>profile.milliseconds,selected=best,alternatives=finals.GroupBy(p=>p.signature).Select(g=>g.First()).OrderByDescending(p=>p.score).Take(8).ToArray(),family_best=finals.GroupBy(p=>p.actions[0][0].Split('|')[0]).ToDictionary(g=>g.Key,g=>g.OrderByDescending(p=>p.score).ThenBy(p=>p.signature,StringComparer.Ordinal).First())};
        }
        static Plan Choose(List<Plan> plans,Profile profile)
        {
            // Terminal win-share units cannot use the nonterminal VP style window.
            // Preserve outcome/rank before personality; equal outcomes can still vary.
            if(plans.All(p=>p.ledger!=null && p.ledger.win_share.HasValue))
                return plans.OrderByDescending(p=>p.ledger.win_share.Value).ThenBy(p=>p.ledger.rank.Value).ThenByDescending(p=>p.preference).ThenByDescending(p=>p.ledger.banked).ThenBy(p=>p.signature,StringComparer.Ordinal).First();
            double max=plans.Max(p=>p.score);
            return plans.Where(p=>p.score>=max-profile.credible_window).OrderByDescending(p=>p.score+p.preference).ThenBy(p=>p.signature,StringComparer.Ordinal).First();
        }
    }
}
