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
        public string id,mode="masked_ismcts";
        public int simulations=100000,root_top_k=16,interior_top_k=8,max_choices=512;
        public double exploration=1.4;
        public int player_count=4,work_limit=160,first_width=24,beam=10,second_width=12;
        // Lab replicas partition root branches; each still produces a complete
        // cash-safe incumbent. The installed/frozen profiles default to serial.
        public int partition_index=0,partition_count=1;
        public double milliseconds=1000,credible_window=1.5,rail_preference=0;
        public string tactic="balanced";
        public bool intent_enabled=false;
        public int conversion_policy=0; // 1: deadline-aware industries and complete beer-source comparisons.
        public int cash_policy=0; // 0: archived v2 behavior; 1: v3 round-payment protection.
        public double style_weight=1,proposal_fraction=0.40;
        public int reply_candidates=4,reply_scenarios=2,reply_confirmation=2,reply_work=2048;
        public static Profile Read(JObject j)
        {
            var p=j.ToObject<Profile>();
            if(p.conversion_policy<0 || p.conversion_policy>1)throw new Exception("conversion_policy_contract");
            if(p.cash_policy<0 || p.cash_policy>1)throw new Exception("cash_policy_contract");
            if(p.partition_count<1 || p.partition_count>32 || p.partition_index<0 || p.partition_index>=p.partition_count)throw new Exception("partition_contract");
            if(new[]{p.credible_window,p.rail_preference,p.style_weight,p.proposal_fraction}.Any(x=>double.IsNaN(x) || double.IsInfinity(x)))throw new Exception("nonfinite_profile");
            if(p.player_count!=4 || p.mode!="masked_ismcts" || p.work_limit<1 || p.milliseconds<0 || double.IsNaN(p.milliseconds) || double.IsInfinity(p.milliseconds) || p.first_width<1 || p.beam<1 || p.second_width<1 || string.IsNullOrEmpty(p.id) || !Tactics.Styles.Contains(p.tactic) || p.reply_candidates<1 || p.reply_scenarios<1 || p.reply_confirmation<1 || p.reply_work<1 || p.proposal_fraction<=0 || p.proposal_fraction>=1 || p.credible_window<0 || p.style_weight<0)throw new Exception("profile_contract");
            if(p.simulations<1 || p.root_top_k<1 || p.interior_top_k<1 || p.max_choices<32 || p.max_choices>4096 || double.IsNaN(p.exploration) || double.IsInfinity(p.exploration) || p.exploration<=0)throw new Exception("masked_search_contract");
            return p;
        }
    }
    public sealed class Plan
    {
        public List<string[]> actions=new List<string[]>();
        public double score,preference;
        public Ledger ledger;
        public CashProjection cash;
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
        public MaskedTelemetry search;
    }
    // Each request is independent. Policies receive only observation + semantic legal IDs.
    // There is no access to a real Host, BoardHelper, historical stock rule, or RNG.
    public sealed partial class Planner
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
            // Used only for a fast complete legal incumbent, never a beam search.
            return legal.OrderByDescending(c=>Evaluator.Proposal(view,c,profile.conversion_policy==1)+Tactics.Bias(profile,intent,c)).ThenBy(c=>c.id,StringComparer.Ordinal).Take(1);
        }
        static string[] GreedyFinish(Session s,List<string> prefix,string first,bool extend,int beerPreference=0)
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
                    c=choices.OrderByDescending(x=>s.profile.conversion_policy==1 && x.kind=="beer"?Conversion.BeerPriority(view,x,beerPreference):Evaluator.Proposal(view,x,s.profile.conversion_policy==1)).ThenBy(x=>x.id,StringComparer.Ordinal).First();
                    // Finishing is an independent option. Extra development must clear
                    // a useful low tile, not discard a valuable high tile by habit.
                    if(c.kind=="develop" && choices.Any(x=>x.kind=="finish"))
                    {
                        var tile=Conversion.Player(view,s.seat)["mat"].Single(m=>(string)m["type"]==c.industry)["tiles"].First();
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
            if(h.Remaining==0 && ((string)view["context"]["era"]=="RailEra" || Conversion.Intro(view)) &&
                (int)view["context"]["round"]==Conversion.RoundLimit(view) &&
                (int)view["context"]["turn"]==s.profile.player_count)
            {
                // Only final public outcomes are consumed. No future sampled cards enter a decision.
                h.Advance();
                // Introductory Canal scoring can pause for mandatory liquidation.
                // Finish that bounded public phase before comparing win-share units;
                // a pending-debt heuristic must never compete with terminal utility.
                int remainingLiquidations=view["slots"].Count(x=>x["tile"]!=null && x["tile"].Type!=JTokenType.Null);
                while(h.Phase=="debt")
                {
                    if(remainingLiquidations--<=0)throw new InvalidOperationException("final_settlement_did_not_progress");
                    var publicState=s.View(h);
                    var selected=h.Choices().OrderBy(c=>{
                        var tile=publicState["slots"].Single(x=>(string)x["id"]==c.target)["tile"];
                        // Match the live AI's public liquidation valuation. Human
                        // choices remain a model; semantic ties never use setup IDs.
                        return (((string)publicState["context"]["era"]=="CanalEra" && (int)tile["level"]==2)?2:1)*
                            ((int)tile["vp"]+(int)tile["link_vp"]);
                    }).ThenBy(c=>c.target,StringComparer.Ordinal).First();
                    h.SearchApply(selected.id);
                }
                if(!h.Terminal)throw new InvalidOperationException("final_settlement_not_terminal");
                view=s.View(h);
            }
            var ledger=Evaluator.Evaluate(view,s.seat,s.profile.conversion_policy==1);var p=new Plan{actions=actions,ledger=ledger,score=ledger.score};
            if(s.profile.cash_policy==1)
            {
                p.cash=CashManagement.Project(view,s.seat);
                if(!ledger.win_share.HasValue)
                {
                    // Partial setup actions may still be funded by action two.
                    ledger.liquidity=-p.cash.penalty*(h.Remaining==0?1:0.25);
                    ledger.score+=ledger.liquidity;p.score=ledger.score;
                }
            }
            p.intent=s.intent;p.preference=Tactics.Preference(s.profile,s.intent,actions);return p;
        }
        static List<Plan> Actions(Session s,List<string[]> before,int limit,bool firstStage=false)
        {
            var prefix=before.SelectMany(a=>a).ToList();var root=s.World(prefix);var view=s.View(root);var results=new List<Plan>();
            var proposals=Proposals(view,root.Choices(),limit,s.profile,s.intent);

            foreach(var first in proposals)
            {
                if(s.Stop && results.Count>0)break;
                if(firstStage && results.Count>=8 && (s.nodes>=s.profile.work_limit*0.45 || (!s.fixedWork && s.clock.Elapsed.TotalMilliseconds>=s.profile.milliseconds*0.45)))break;
                foreach(bool extend in new[]{false,true})
                {
                    if(extend && s.profile.conversion_policy==1 && limit==1 && first.kind=="sell")continue;
                    if(extend && (s.Stop || (first.kind!="sell" && first.kind!="network" && first.kind!="develop")))continue;
                    try
                    {
                        int variants=s.profile.conversion_policy==1 && (first.kind=="sell" || first.kind=="network")?3:1;
                        for(int beerPreference=0;beerPreference<variants;beerPreference++)
                        {
                            if(beerPreference>0 && s.Stop)break;
                            bool batch=extend || (s.profile.conversion_policy==1 && limit==1 && first.kind=="sell");
                            var action=GreedyFinish(s,prefix,first.id,batch,beerPreference);
                            var all=new List<string[]>(before){action};var plan=Score(s,all);
                            if(!results.Any(p=>p.signature==plan.signature))results.Add(plan);
                        }
                    }
                    catch(InvalidOperationException){s.failures++;}
                }
            }
            if(results.Count==0 && !(before.Count==0 && limit>1 && s.profile.partition_count>1))throw new Exception("no_complete_candidate");
            return results.OrderByDescending(p=>p.score).ThenBy(p=>p.signature,StringComparer.Ordinal).ToList();
        }
        static Plan Complete(Session s,Plan p,int target)
        {
            if(p.actions.Count>=target)return p;
            // A valid full-turn incumbent is constructed even if the search budget has
            // expired. Its bounded validation overrun is reported, never hidden.
            return Actions(s,p.actions,1)[0];
        }
        static List<Plan> CashIncumbents(Session s,Host root,int target,Plan quick)
        {
            var result=new List<Plan>{quick};
            if(s.profile.cash_policy!=1 || quick.cash==null || quick.ledger.win_share.HasValue ||
                (quick.cash.shortfall==0 && (s.profile.conversion_policy!=1 || !quick.cash.future_credit_blocked)))return result;
            // Bounded mandatory completion, just like the original greedy seed.
            // Publish no unsafe timed-out incumbent before trying financing,
            // immediate sales and conserving cash. Never consult another hand.
            var legal=root.Choices();
            bool conversion=s.profile.conversion_policy==1;
            var sales=legal.Where(c=>c.kind=="sell")
                .OrderByDescending(c=>(int)s.root["slots"].Single(x=>(string)x["id"]==c.target)["tile"]["income"])
                // Merchant money or income can cover the current bill. Preserve
                // these destinations instead of collapsing every sale of a tile.
                .ThenByDescending(c=>conversion && s.root["merchants"].Any(m=>(string)m["id"]==c.merchant && (int)m["beer"]>0)
                    ? (Evaluator.Region(c.merchant)=="Warrington"?5:Evaluator.Region(c.merchant)=="Oxford"?2:0):0)
                .ThenBy(c=>c.id,StringComparer.Ordinal)
                .GroupBy(c=>c.target+(conversion?"|"+c.merchant:""))
                .Select(g=>g.First()).Take(conversion?8:4);
            var rescue=legal.Where(c=>c.kind=="loan" || c.kind=="pass")
                .GroupBy(c=>c.kind).Select(g=>g.OrderByDescending(c=>Evaluator.Proposal(s.root,c,s.profile.conversion_policy==1)).ThenBy(c=>c.id,StringComparer.Ordinal).First())
                .Concat(sales);
            foreach(var first in rescue)
            {
                // The bounded emergency path runs even at zero search budget.
                // Test own-brewery completion as well as merchant bonuses before
                // publishing a plan that would cause avoidable liquidation.
                int variants=conversion && first.kind=="sell"?3:1;
                for(int beerPreference=0;beerPreference<variants;beerPreference++)
                {
                    var actions=new List<string[]>{GreedyFinish(s,new List<string>(),first.id,true,beerPreference)};
                    while(actions.Count<target)
                    {
                        var prefix=actions.SelectMany(a=>a).ToList();var h=s.World(prefix);var view=s.View(h);
                        var choices=h.Choices();
                        // A sale can lift income above the borrowing floor. Use the
                        // newly legal loan if action two still needs to cover debt.
                        var recovery=CashManagement.Project(view,s.seat).shortfall>0
                            ? choices.Where(c=>c.kind=="loan").OrderByDescending(c=>Evaluator.Proposal(view,c,s.profile.conversion_policy==1)).ThenBy(c=>c.id,StringComparer.Ordinal).FirstOrDefault():null;
                        recovery=recovery ?? choices.Where(c=>c.kind=="pass").OrderByDescending(c=>Evaluator.Proposal(view,c,s.profile.conversion_policy==1)).ThenBy(c=>c.id,StringComparer.Ordinal).First();
                        actions.Add(GreedyFinish(s,prefix,recovery.id,false));
                    }
                    var plan=Score(s,actions);result.Add(plan);
                    if(conversion && plan.cash.shortfall==0 && !plan.cash.future_credit_blocked)return result;
                }
            }
            return result;
        }
        public static Plan Choose(List<Plan> plans,Profile profile)
        {
            if(plans.Count==0)throw new Exception("no_complete_candidate");
            // Terminal win-share units cannot use the nonterminal VP style window.
            // Preserve outcome/rank before personality; equal outcomes can still vary.
            if(profile.conversion_policy==1 && plans.All(p=>p.ledger!=null && p.ledger.win_share.HasValue))
                return plans.OrderByDescending(p=>p.ledger.win_share.Value).ThenBy(p=>p.ledger.rank.Value).ThenByDescending(p=>p.ledger.banked).ThenByDescending(p=>p.ledger.final_income).ThenByDescending(p=>p.ledger.final_cash).ThenByDescending(p=>p.preference).ThenBy(p=>p.signature,StringComparer.Ordinal).First();
            if(plans.All(p=>p.ledger!=null && p.ledger.win_share.HasValue))
                return plans.OrderByDescending(p=>p.ledger.win_share.Value).ThenBy(p=>p.ledger.rank.Value).ThenByDescending(p=>p.preference).ThenByDescending(p=>p.ledger.banked).ThenBy(p=>p.signature,StringComparer.Ordinal).First();
            if(profile.cash_policy==1)
            {
                // A personality bonus cannot spend money reserved for a known
                // income bill. When distress is unavoidable, minimize its size.
                int shortfall=plans.Min(p=>p.cash.shortfall);
                plans=plans.Where(p=>p.cash.shortfall==shortfall).ToList();
                // Preserve the option to finance a later completion. Otherwise
                // a short search can stack loans, exhaust its cash on unflipped
                // industries, and spend the rest of the game below the borrowing
                // floor. Emergency borrowing still wins over a payment shortfall;
                // same-turn income recovery and final-round spending remain valid.
                if(profile.conversion_policy==1 && plans.Any(p=>!p.cash.future_credit_blocked))
                    plans=plans.Where(p=>!p.cash.future_credit_blocked).ToList();
            }
            double max=plans.Max(p=>p.score);
            return plans.Where(p=>p.score>=max-profile.credible_window).OrderByDescending(p=>p.score+p.preference).ThenBy(p=>p.signature,StringComparer.Ordinal).First();
        }
    }
}
