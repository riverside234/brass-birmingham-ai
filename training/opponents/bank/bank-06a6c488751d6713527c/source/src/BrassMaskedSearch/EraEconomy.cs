using System;
using System.Collections.Generic;
using System.Linq;
using Newtonsoft.Json.Linq;
using BoardGameRules.Entities.Markets;
using BrassHost;

namespace BrassLegacyAI
{
    // Original heuristic policy over permitted observations. Quotes call the
    // declared rules engine; detached simulations remain the payment authority.
    public static class EraEconomy
    {
        sealed class CoalQuote:CoalMarket {public CoalQuote(int stock){Amount=stock;}}
        sealed class IronQuote:IronMarket {public IronQuote(int stock){Amount=stock;}}
        public static bool Rail(JObject v){return (string)v["context"]["era"]=="RailEra";}
        static bool Present(JToken s){return s["tile"]!=null && s["tile"].Type!=JTokenType.Null;}
        public static string Strategy(JObject v){return Rail(v)?"industry-completion-and-funded-rail":"income-conversion-and-carryover";}
        public static int CoalPrice(JObject v,int count){return count<=0?0:new CoalQuote((int)v["coal_market"]).CoalPrice(count);}
        public static int IronPrice(JObject v,int count){return count<=0?0:new IronQuote((int)v["iron_market"]).IronPrice(count);}
        static HashSet<string> Reach(JObject v,IEnumerable<string> starts,string added=null)
        {
            var seen=new HashSet<string>(starts);bool changed=true;
            while(changed){changed=false;foreach(var l in v["links"])
                if((l["owner"]!=null && l["owner"].Type!=JTokenType.Null) || (string)l["id"]==added)
                    if(l["regions"].Any(r=>seen.Contains((string)r)))foreach(var r in l["regions"])changed=seen.Add((string)r)||changed;}
            return seen;
        }
        static int BoardCoal(JObject v,HashSet<string> reach)
        {return v["slots"].Where(Present).Where(s=>(string)s["tile"]["type"]=="CoalMine" && reach.Contains(Evaluator.Region((string)s["id"]))).Sum(s=>(int)s["tile"]["resource"]);}
        public static int BuildCost(JObject v,JToken tile,string region)
        {
            var reach=Reach(v,new[]{region});int coal=Math.Max(0,(int)tile["coal"]-BoardCoal(v,reach));
            int iron=Math.Max(0,(int)tile["iron"]-v["slots"].Where(Present).Where(s=>(string)s["tile"]["type"]=="IronWorks").Sum(s=>(int)s["tile"]["resource"]));
            if(coal>0 && !v["merchants"].Any(m=>reach.Contains(Evaluator.Region((string)m["id"]))))return 100000;
            return (int)tile["cost"]+CoalPrice(v,coal)+IronPrice(v,iron);
        }
        public static int NetworkCost(JObject v,Choice c)
        {
            if(!Rail(v))return 3;
            var link=v["links"].Single(l=>(string)l["id"]==c.target);
            var reach=Reach(v,link["regions"].Select(r=>(string)r),c.target);
            int coal=BoardCoal(v,reach)>0?0:CoalPrice(v,1);
            bool second=(string)v["pending"]["kind"]=="network" && (int)v["pending"]["parts"]>0;
            return (second?10:5)+coal;
        }
        public static double BeerOpportunity(JObject v,Choice c)
        {
            if(!Rail(v) || (string)v["pending"]["kind"]!="network" || (int)v["pending"]["parts"]==0)return 0;
            int seat=(int)v["seat"];
            bool ready=v["slots"].Where(Present).Any(s=>(int)s["tile"]["seat"]==seat && Conversion.Sellable(s["tile"]) && !(bool)s["tile"]["flipped"] && (int)s["tile"]["beer"]>0);
            return ready?2.5:0.75;
        }
        public static double LinkProbability(JObject v,JToken slot,int owner,double forecast)
        {
            if((bool)slot["tile"]["flipped"])return 1;
            if(!Rail(v))return forecast;
            int actor=(int)slot["tile"]["seat"],actions=Conversion.Actions(v,actor);
            double reliability=actor==owner?0.85:0.55;
            return forecast*reliability*(actions<=2?0.65:1);
        }
        public static double NetworkProposal(JObject v,Choice c)
        {
            var l=v["links"].Single(x=>(string)x["id"]==c.target);var ends=l["regions"].Select(r=>(string)r).ToArray();
            int seat=(int)v["seat"];
            double points=v["slots"].Where(Present).Where(s=>ends.Contains(Evaluator.Region((string)s["id"])))
                .Sum(s=>(int)s["tile"]["link_vp"]*LinkProbability(v,s,seat,Conversion.Completion(v,s,(int)s["tile"]["seat"])));
            points+=2*v["merchants"].Select(m=>Evaluator.Region((string)m["id"])).Distinct().Count(ends.Contains);
            return points-NetworkCost(v,c)*0.25-BeerOpportunity(v,c);
        }
        public static double InvestmentReserve(JObject v,int seat)
        {
            if(!Rail(v) || Conversion.Actions(v,seat)<2 || (bool)v["context"]["terminal"])return 0;
            var p=Conversion.Player(v,seat);double best=0,budget=0;
            var network=new HashSet<string>(v["slots"].Where(Present).Where(s=>(int)s["tile"]["seat"]==seat).Select(s=>Evaluator.Region((string)s["id"])));
            foreach(var link in v["links"].Where(l=>(int?)l["owner"]==seat))foreach(var region in link["regions"])network.Add((string)region);
            foreach(var mat in p["mat"])
            {
                var t=mat["tiles"].FirstOrDefault();if(t==null || (bool)t["canal_only"])continue;
                string type=(string)mat["type"];
                // Own cards are private to this observer; opponents use public
                // opportunities only, never this observer's hand or sampled cards.
                var sites=v["slots"].Where(s=>!Present(s) && s["types"].Any(k=>(string)k==type));
                if(seat==(int)v["seat"])sites=sites.Where(s=>v["own_hand"].Any(c=>{
                    string card=(string)c,region=Evaluator.Region((string)s["id"]);
                    return card=="wild:location" || card=="location:"+region ||
                        ((card=="wild:industry" || (card.StartsWith("industry:") && card.Contains(type))) && (network.Count==0 || network.Contains(region)));}));
                foreach(var site in sites.Take(3))
                {
                    int cost=BuildCost(v,t,Evaluator.Region((string)site["id"]));if(cost>=100000)continue;
                    var imagined=(JObject)site.DeepClone();imagined["tile"]=t.DeepClone();
                    double completion=Conversion.Completion(v,imagined,seat);
                    if(completion<0.45)continue;
                    double gain=(int)t["vp"]*completion+(int)t["income"]*0.35-cost*0.12;
                    if(gain>best){best=gain;budget=cost;}
                }
            }
            return Math.Min(40,budget);
        }
        public static double Finance(JObject v,int seat,double projected,double capacity,double reserve)
        {
            if(projected<0)return projected*0.55;
            if(!Rail(v))return capacity*0.32*(1-Math.Exp(-projected/capacity));
            double cash=Math.Max(0,(int)Conversion.Player(v,seat)["cash"]);
            double working=Math.Min(cash,reserve),rest=Math.Max(0,projected-working);
            return working*0.30+capacity*0.24*(1-Math.Exp(-rest/capacity));
        }
        public static int ConnectionBudget(JObject v,int links)
        {
            if(links>=99)return 100000;
            // Conservative separate-action quote. Completed candidate paths pay
            // exact actual costs, including shared coal, double-rail beer and fees.
            return Rail(v)?links*5+CoalPrice(v,links):links*3;
        }
    }
}
