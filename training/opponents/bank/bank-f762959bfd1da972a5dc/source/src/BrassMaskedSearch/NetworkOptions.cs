using System;
using System.Collections.Generic;
using System.Linq;
using Newtonsoft.Json.Linq;
using BrassHost;

namespace BrassLegacyAI
{
    // Small permitted-state proposal cues. No Scout, wild-hand experiments,
    // brewery-demand forecast or extra VP for hypothetical future buildings.
    public static class NetworkOptions
    {
        public const int EarlyCanalLastRound=3;
        public const double MaximumRouteBias=.75,EarlyCanalCost=.75;
        static bool Present(JToken s){return s["tile"]!=null && s["tile"].Type!=JTokenType.Null;}
        public static bool EarlyCanal(JObject v)
        {return !EraEconomy.Rail(v) && (int)v["context"]["round"]<=EarlyCanalLastRound && !(bool)v["context"]["terminal"];}
        public static string[] Preferred(string style,IntentState intent=null)
        {
            if(intent!=null && new[]{"CottonMill","Manufacturer","Pottery","IronWorks","CoalMine"}.Contains(intent.focus))return new[]{intent.focus};
            switch(style){case "cotton":return new[]{"CottonMill"};case "goods-beer":return new[]{"CottonMill","IronWorks"};
                case "rail-tempo":return new[]{"Manufacturer","IronWorks"};case "supplier":return new[]{"IronWorks","Manufacturer","CottonMill"};
                case "pottery-flex":return new[]{"Pottery","IronWorks"};default:return new[]{"CottonMill","Manufacturer","Pottery","IronWorks"};}
        }
        public static double[] BiasValues(JObject v,List<Choice> choices,Profile p,IntentState intent)
        {
            var result=new double[choices.Count];
            if(p.conversion_policy!=1 || !choices.Any(c=>c.kind=="network"))return result;
            bool networkFocus=intent!=null && intent.focus=="network";
            var types=Preferred(p.tactic,intent);int seat=(int)v["seat"];
            var slots=v["slots"].GroupBy(s=>Evaluator.Region((string)s["id"]))
                .ToDictionary(g=>g.Key,g=>g.ToArray());
            var occupied=new HashSet<string>(v["slots"].Where(Present).Where(s=>(int)s["tile"]["seat"]==seat).Select(s=>Evaluator.Region((string)s["id"])));
            var hand=v["own_hand"].Select(c=>(string)c).ToArray();
            var links=v["links"].ToDictionary(l=>(string)l["id"]);
            // Components are reused across all candidate/card variants. The
            // prospective link joins its two existing public components.
            var reach=new Dictionary<string,HashSet<string>>();
            var compatible=v["merchants"].Where(m=>m["types"]!=null && m["types"].Any(t=>types.Contains((string)t)))
                .Select(m=>Evaluator.Region((string)m["id"])).Distinct().ToArray();
            var quoted=new Dictionary<string,double>();
            for(int i=0;i<choices.Count;i++)
            {
                var choice=choices[i];if(choice.kind!="network")continue;
                double bias;if(!quoted.TryGetValue(choice.target,out bias))
                {
                    var ends=links[choice.target]["regions"].Select(r=>(string)r).ToArray();
                    int capacity=0,breweries=0,useful=0;bool buyer=false;
                    foreach(string region in ends)
                    {
                        JToken[] sites;if(slots.TryGetValue(region,out sites))
                        {
                            capacity+=sites.Length;breweries+=sites.Count(s=>s["types"].Any(t=>(string)t=="Brewery"));
                            if(EraEconomy.Rail(v) || !occupied.Contains(region))
                                useful+=sites.Count(s=>!Present(s) && s["types"].Any(t=>types.Contains((string)t)) &&
                                    hand.Any(card=>card=="wild:location" || card=="location:"+region || card=="wild:industry" ||
                                        (card.StartsWith("industry:") && s["types"].Any(t=>types.Contains((string)t) && card.Contains((string)t)))));
                        }
                        HashSet<string> component;
                        if(!reach.TryGetValue(region,out component)){component=Evaluator.Reach(v,region);foreach(string member in component)reach[member]=component;}
                        buyer=buyer || compatible.Any(component.Contains);
                    }
                    bias=Math.Min(MaximumRouteBias,(networkFocus?Math.Min(.60,capacity*.045+breweries*.10):Math.Min(.60,useful*.15))+(buyer?.15:0));
                    quoted.Add(choice.target,bias);
                }
                result[i]=bias;
            }
            return result;
        }
        static bool UsefulSale(JObject v,string link)
        {
            int seat=(int)v["seat"];
            foreach(var site in v["slots"].Where(Present).Where(s=>(int)s["tile"]["seat"]==seat && !(bool)s["tile"]["flipped"] && Conversion.Sellable(s["tile"])))
            {
                string region=Evaluator.Region((string)site["id"]),type=(string)site["tile"]["type"];
                int before=Conversion.MerchantDistance(v,region,type),after=Conversion.MerchantDistance(v,region,type,link);
                if(after<before && after<=1)return true;
            }
            return false;
        }
        static HashSet<string> OwnNetwork(JObject v,int seat)
        {
            var regions=new HashSet<string>(v["slots"].Where(Present).Where(s=>(int)s["tile"]["seat"]==seat).Select(s=>Evaluator.Region((string)s["id"])));
            foreach(var l in v["links"].Where(l=>(int?)l["owner"]==seat))foreach(var r in l["regions"])regions.Add((string)r);
            return regions;
        }
        public static double CanalPenalty(JObject root,List<string[]> actions)
        {
            if(!EarlyCanal(root) || !actions.Any(a=>a.Any(id=>id.StartsWith("network|"))))return 0;
            int seat=(int)root["seat"];var working=(JObject)root.DeepClone();double penalty=0;
            bool first=!root["links"].Any(l=>(string)l["era"]=="Canal" && (int?)l["owner"]==seat);
            for(int a=0;a<actions.Count;a++)foreach(string id in actions[a].Where(x=>x.StartsWith("network|")))
            {
                string target=id.Split('|')[1];var link=working["links"].Single(l=>(string)l["id"]==target);
                bool useful=first || UsefulSale(working,target);var own=OwnNetwork(working,seat);
                var after=(JObject)working.DeepClone();after["links"].Single(l=>(string)l["id"]==target)["owner"]=seat;
                foreach(string build in actions.Skip(a+1).SelectMany(x=>x).Where(x=>x.StartsWith("build|")))
                {
                    var parts=build.Split('|');string region=Evaluator.Region(parts[1]),type=parts[2],card=parts[3];
                    var tile=Conversion.Player(working,seat)["mat"].Single(m=>(string)m["type"]==type)["tiles"].FirstOrDefault();
                    bool industryCard=card=="wild:industry" || card.StartsWith("industry:");
                    if(industryCard && own.Count>0 && !own.Contains(region) && link["regions"].Any(r=>(string)r==region))useful=true;
                    if(tile!=null && EraEconomy.BuildCost(working,tile,region)>=100000 && EraEconomy.BuildCost(after,tile,region)<100000)useful=true;
                }
                if(!useful)penalty+=EarlyCanalCost;
                working=after;first=false;
            }
            return Math.Min(EarlyCanalCost*2,penalty);
        }
        // Reuse the complete afterstate's existing liquidity/reserve quote.
        // No hypothetical future income or loan is available to pay the rails.
        public static bool RetainsCapital(Plan plan)
        {return plan.cash!=null && plan.cash.shortfall==0 && !plan.cash.future_credit_blocked && plan.cash.financing_actions==0;}
    }
}
