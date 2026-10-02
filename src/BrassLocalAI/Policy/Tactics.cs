using System;
using System.Linq;
using System.Collections.Generic;
using Newtonsoft.Json.Linq;
using BrassHost;
namespace BrassLegacyAI
{
    public sealed class IntentState
    {
        public string format="legacy-intent/1.0.0",profile,focus="balanced",target="",milestone="evaluate",era,reason;
        public int seat,revision,age,remaining_actions,cash_budget,goal_level;
        public string book_family,progress_key;
        public string[] required_cards=new string[0];
        public int coal,iron,beer;
    }
    public static class Tactics
    {
        public static readonly string[] Styles={"balanced","rail-tempo","cotton","goods-beer","supplier","pottery-flex"};
        public static string[] Industries(string style)
        {
            switch(style){case "cotton":return new[]{"CottonMill","Brewery"};case "goods-beer":return new[]{"Manufacturer","Brewery"};case "supplier":return new[]{"IronWorks","Brewery","Manufacturer"};case "pottery-flex":return new[]{"Pottery","IronWorks","Brewery"};case "rail-tempo":return new[]{"Pottery","Brewery","CoalMine"};default:return new string[0];}
        }
        static bool CardSupports(string card,string industry,string region)
        {return card.StartsWith("wild:") || card=="location:"+region || (card.StartsWith("industry:") && card.Contains(industry));}
        public static IntentState Refresh(JObject v,Profile p,IntentState previous)
        {
            if(!p.intent_enabled)return null;
            int seat=(int)v["seat"],revision=(int)v["context"]["revision"];
            if(previous!=null && (previous.format!="legacy-intent/1.0.0" || previous.seat!=seat || previous.profile!=p.id || previous.revision>revision))throw new Exception("intent_contract");
            var me=v["players"][seat];string era=(string)v["context"]["era"];
            var next=new IntentState{profile=p.id,seat=seat,revision=revision,era=era,remaining_actions=Evaluator.Rounds(v)*2+Evaluator.I(v["context"],"actions_remaining"),cash_budget=Evaluator.I(me,"cash"),reason="no feasible preferred opportunity; balanced recovery"};
            var strategic=StrategicOpenings.Intent(v,p);
            var types=Industries(p.tactic);
            // A pivot must still finish existing investments of any industry.
            var own=v["slots"].Where(s=>s["tile"]!=null && s["tile"].Type!=JTokenType.Null && (int)s["tile"]["seat"]==seat && !(bool)s["tile"]["flipped"]).ToArray();
            var sellable=own.Where(s=>new[]{"CottonMill","Manufacturer","Pottery"}.Contains((string)s["tile"]["type"]) && Evaluator.Completion(v,s,seat)>=0.5).OrderByDescending(s=>(int)s["tile"]["vp"]).FirstOrDefault();
            if(sellable!=null)
            {
                next.focus=(string)sellable["tile"]["type"];next.target=(string)sellable["id"];next.milestone="complete_sale";next.beer=Evaluator.I(sellable["tile"],"beer");next.reason="finish an existing feasible investment";
                if(Evaluator.Completion(v,sellable,seat)<0.9)
                {
                    var brewery=me["mat"].Single(m=>(string)m["type"]=="Brewery")["tiles"].FirstOrDefault();
                    var site=v["slots"].FirstOrDefault(s=>s["tile"]==null && s["types"].Any(t=>(string)t=="Brewery") && v["own_hand"].Any(c=>CardSupports((string)c,"Brewery",Evaluator.Region((string)s["id"]))));
                    if(brewery!=null && site!=null && next.remaining_actions>=2 && next.cash_budget>=Evaluator.I(brewery,"cost")+4)
                    {next.focus="Brewery";next.target=(string)site["id"];next.milestone="secure_own_beer";next.reason="contested beer unavailable; affordable own-brewery contingency";next.required_cards=v["own_hand"].Select(c=>(string)c).Where(c=>CardSupports(c,"Brewery",Evaluator.Region(next.target))).ToArray();}
                    else {next.focus="balanced";next.target="";next.milestone="recover";next.reason="sale beer unavailable and no affordable own-beer contingency";}
                }
            }
            else if(p.tactic=="rail-tempo" && era=="RailEra" && (strategic==null || next.remaining_actions<=4) && Evaluator.I(me,"link_tiles")>0 && next.cash_budget>=Math.Max(next.remaining_actions<=4?5:10,(int)EraEconomy.InvestmentReserve(v,seat)))
            {next.focus="network";next.milestone="useful_link";next.reason="rail access and link value remain available";}
            else if(strategic!=null)next=strategic;
            // Revised profiles fall back to the unbiased economic evaluator
            // when the quoted book fails. The old unpriced template must not
            // resurrect the same unaffordable opening under another milestone.
            else foreach(var type in p.conversion_policy==1?new string[0]:types)
            {
                var tile=me["mat"].Single(m=>(string)m["type"]==type)["tiles"].FirstOrDefault();
                if(tile==null || next.remaining_actions<2)continue;
                // Capacity for financing is conditional on an affordable legal loan.
                int finance=Evaluator.I(me,"income_marker")>=3?30:0;
                if(next.cash_budget+finance<Evaluator.I(tile,"cost"))continue;
                var slots=v["slots"].Where(s=>s["tile"]==null && s["types"].Any(t=>(string)t==type) && v["own_hand"].Any(c=>CardSupports((string)c,type,Evaluator.Region((string)s["id"]))));
                foreach(var slot in slots.OrderByDescending(s=>previous!=null && previous.era==era && previous.focus==type && previous.target==(string)s["id"]).ThenBy(s=>(string)s["id"],StringComparer.Ordinal))
                {
                    var imagined=(JObject)slot.DeepClone();imagined["tile"]=tile.DeepClone();
                    double completion=Evaluator.Completion(v,imagined,seat);
                    if(completion<(p.tactic=="supplier"?0.45:0.50))continue;
                    next.focus=type;next.target=(string)slot["id"];next.milestone="build_then_complete";next.coal=Evaluator.I(tile,"coal");next.iron=Evaluator.I(tile,"iron");next.beer=Evaluator.I(tile,"beer");
                    next.required_cards=v["own_hand"].Select(c=>(string)c).Where(c=>CardSupports(c,type,Evaluator.Region(next.target))).ToArray();next.reason="cards, site, funding and completion opportunity available";break;
                }
                if(next.focus!="balanced")break;
            }
            if(previous!=null && previous.focus!="balanced")
            {
                bool retained=previous.era==era && previous.focus==next.focus && previous.target==next.target;
                next.progress_key=Progress(v,next);
                bool progressed=retained && next.progress_key!=previous.progress_key;
                next.age=retained && !progressed?previous.age+(revision==previous.revision?0:1):0;
                if(retained && next.age<4)next.reason="retained feasible intent: "+next.milestone;
                else if(retained){next.focus="balanced";next.target="";next.milestone="recover";next.reason="intent stalled for four decisions; balanced recovery";next.age=0;}
                else next.reason="abandoned "+previous.focus+" at "+previous.target+": opportunity changed; "+next.reason;
            }
            else next.progress_key=Progress(v,next);
            return next;
        }
        static string Progress(JObject v,IntentState intent)
        {
            if(intent.focus=="balanced")return null;
            int seat=(int)v["seat"];var me=Conversion.Player(v,seat);
            var mat=me["mat"].FirstOrDefault(m=>(string)m["type"]==intent.focus);
            int owned=v["slots"].Count(s=>s["tile"]!=null && s["tile"].Type!=JTokenType.Null &&
                (int)s["tile"]["seat"]==seat && (string)s["tile"]["type"]==intent.focus);
            int route=0;
            if(mat!=null && !string.IsNullOrEmpty(intent.target))route=Conversion.MerchantDistance(v,Evaluator.Region(intent.target),intent.focus);
            return (mat==null?0:mat["tiles"].Count())+"/"+owned+"/"+route+"/"+((int)me["cash"]>=intent.cash_budget?"funded":"unfunded");
        }
        public static double Bias(Profile p,IntentState intent,Choice c,JObject view=null)
        {
            if(!p.intent_enabled || intent==null || intent.focus=="balanced")return 0;
            double cardPenalty=c.card!=null && intent.required_cards.Contains(c.card) && c.kind!="build" && c.kind!="sell"?-1.5*p.style_weight:0;
            bool industry=c.industry==intent.focus;
            bool needsDevelopment=intent.milestone=="develop_high_tier";
            if(view!=null && intent.goal_level>0)
            {
                var tile=Conversion.Player(view,(int)view["seat"])["mat"].Single(m=>(string)m["type"]==intent.focus)["tiles"].FirstOrDefault();
                needsDevelopment=tile!=null && (int)tile["level"]<intent.goal_level;
            }
            if(c.kind=="loan" && intent.milestone=="finance_sequence")
                return (view==null || (int)Conversion.Player(view,(int)view["seat"])["cash"]<intent.cash_budget)?p.style_weight:0;
            if(c.kind=="build" && industry)return p.style_weight*(needsDevelopment?-.75:c.target==intent.target?1:.75);
            if(c.kind=="sell" && c.target==intent.target)return 2*p.style_weight;
            // A merchant removal is ranked by its actual portfolio improvement.
            // Reusing the opening's paid-development bias here can reject a
            // clearly better free Pottery unlock just to maintain a Cotton goal.
            if(c.kind=="bonus_develop")return 0;
            // The named tier ends the development incentive. Search may still
            // choose any legal upgrade if the resulting complete plan is better.
            if(c.kind=="develop" && industry)return p.style_weight*(needsDevelopment?1:intent.goal_level>0?0:.25)+cardPenalty;
            if(c.kind=="network" && intent.focus=="network")return p.style_weight*(intent.target==c.target?2:1)+cardPenalty;
            return cardPenalty;
        }
        public static double Preference(Profile p,IntentState intent,List<string[]> actions)
        {
            double railWeight=1; // Prices and feasible capital, not an era quota, govern rails.
            double value=actions.SelectMany(a=>a).Count(x=>x.StartsWith("network|"))*p.rail_preference*railWeight;
            if(intent==null)return value;
            foreach(var id in actions.SelectMany(a=>a))
            {
                var parts=id.Split('|');if(parts.Length<3)continue;
                if(parts[0]=="build" && parts[2]==intent.focus)value+=p.style_weight;
                if(parts[0]=="sell" && parts[1]==intent.target)value+=p.style_weight;
                if(parts[0]=="network" && intent.focus=="network")value+=p.style_weight;
            }
            return Math.Min(4,value);
        }
    }
}
