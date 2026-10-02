using System;
using System.Linq;
using System.Collections.Generic;
using Newtonsoft.Json.Linq;
using BrassHost;

namespace BrassLegacyAI
{
    // Bounded proposal preferences and explicit action costs, not extra official
    // VP. Every proposed pair/Scout is replayed through fresh host choices.
    public static class NetworkAndHand
    {
        static bool Present(JToken s){return s["tile"]!=null && s["tile"].Type!=JTokenType.Null;}
        public static string[] Preferred(string style)
        {
            switch(style){case "cotton":return new[]{"CottonMill"};case "goods-beer":return new[]{"CottonMill","IronWorks"};
                case "rail-tempo":return new[]{"Manufacturer","IronWorks"};case "supplier":return new[]{"IronWorks","CoalMine","Manufacturer","CottonMill"};
                case "pottery-flex":return new[]{"Pottery","IronWorks","CottonMill"};default:return new[]{"CottonMill","Manufacturer","Pottery","IronWorks"};}
        }
        public static double RouteBias(JObject v,Choice c,string style,bool networkFocus)
        {
            var link=v["links"].Single(l=>(string)l["id"]==c.target);var ends=link["regions"].Select(r=>(string)r).ToArray();
            var types=Preferred(style);int seat=(int)v["seat"];
            var sites=v["slots"].Where(s=>ends.Contains(Evaluator.Region((string)s["id"]))).ToArray();
            int useful=sites.Count(s=>!Present(s) && s["types"].Any(t=>types.Contains((string)t)) &&
                (EraEconomy.Rail(v) || !v["slots"].Where(Present).Any(x=>(int)x["tile"]["seat"]==seat && Evaluator.Region((string)x["id"])==Evaluator.Region((string)s["id"]))) &&
                v["own_hand"].Any(card=>((string)card).StartsWith("wild:") || (string)card=="location:"+Evaluator.Region((string)s["id"]) ||
                    s["types"].Any(t=>types.Contains((string)t) && ((string)card).Contains((string)t))));
            double access=ends.Any(r=>types.Any(t=>Conversion.MerchantDistance(v,r,t)<=1))?.25:0;
            // City size/empty brewery slots are tie preferences only. Actual
            // brewery link VP comes from tile.link_vp and its flip forecast.
            double capacity=networkFocus?Math.Min(.65,sites.Length*.05+sites.Count(s=>s["types"].Any(t=>(string)t=="Brewery"))*.12):Math.Min(.65,useful*.18);
            return capacity+access;
        }
        public static bool RailBeer(JObject v,Choice c)
        {
            int seat=(int)v["seat"];if(CompetitiveEconomy.OwnBeer(v,seat)>CompetitiveEconomy.CommittedBeer(v,seat))return true;
            var after=(JObject)v.DeepClone();var link=after["links"].Single(l=>(string)l["id"]==c.target);link["owner"]=seat;
            var reach=Evaluator.Reach(after,(string)link["regions"][0]);
            return after["slots"].Where(Present).Any(s=>(string)s["tile"]["type"]=="Brewery" && (int)s["tile"]["seat"]!=seat &&
                (int)s["tile"]["resource"]>0 && reach.Contains(Evaluator.Region((string)s["id"])));
        }
        public static double RailDemand(JObject v,int seat)
        {
            var batch=HeuristicBatchQueue.Current;
            return batch==null?RailDemandCore(v,seat):batch.Evaluate("rail-demand/"+seat+"/"+HeuristicBatchQueue.Key(v),()=>new[]{RailDemandCore(v,seat)})[0];
        }
        static double RailDemandCore(JObject v,int seat)
        {
            if(!EraEconomy.Rail(v) || Conversion.Actions(v,seat)<2)return 0;
            var me=Conversion.Player(v,seat);if((int)me["link_tiles"]<2 || (int)me["cash"]<15+EraEconomy.CoalPrice(v,2))return 0;
            var network=new HashSet<string>(v["slots"].Where(Present).Where(s=>(int)s["tile"]["seat"]==seat).Select(s=>Evaluator.Region((string)s["id"])));
            foreach(var l in v["links"].Where(l=>(int?)l["owner"]==seat))foreach(var r in l["regions"])network.Add((string)r);
            var points=v["links"].Where(l=>(string)l["era"]=="Rail" && (l["owner"]==null || l["owner"].Type==JTokenType.Null) && l["regions"].Any(r=>network.Contains((string)r)))
                .Select(l=>{var ends=l["regions"].Select(r=>(string)r).ToArray();
                    bool coal=v["slots"].Where(Present).Any(s=>(string)s["tile"]["type"]=="CoalMine" && (int)s["tile"]["resource"]>0 && ends.Any(r=>Evaluator.Reach(v,r).Contains(Evaluator.Region((string)s["id"]))));
                    bool market=v["merchants"].Any(m=>ends.Any(r=>Evaluator.Reach(v,r).Contains(Evaluator.Region((string)m["id"]))));
                    if(!coal && !market)return 0.0;
                    return v["slots"].Where(Present).Where(s=>ends.Contains(Evaluator.Region((string)s["id"])) && (bool)s["tile"]["flipped"]).Sum(s=>(int)s["tile"]["link_vp"])+
                        2*v["merchants"].Select(m=>Evaluator.Region((string)m["id"])).Distinct().Count(ends.Contains);
                }).OrderByDescending(x=>x).Take(2).ToArray();
            return points.Length==2 && points.Sum()>=8?.65:0;
        }
        public static double ScoutBase(JObject v)
        {
            int seat=(int)v["seat"],actions=Conversion.Actions(v,seat);if(actions<3 || v["own_hand"].Any(c=>((string)c).StartsWith("wild:")))return -8;
            var wild=(JObject)v.DeepClone();((JArray)wild["own_hand"]).Add("wild:location");((JArray)wild["own_hand"]).Add("wild:industry");
            Func<JObject,int,double> opportunity=(view,extra)=>{
                var me=Conversion.Player(view,seat);double best=0;
                foreach(var mat in me["mat"])
                {
                    var tiles=mat["tiles"].ToArray();int removed=EraEconomy.Rail(view)?tiles.TakeWhile(t=>(bool)t["canal_only"]).Count():0;
                    if(removed>=tiles.Length || removed>0 && !tiles.Take(removed).All(t=>(bool)t["developable"]))continue;
                    int paid;string site;var tile=tiles[removed];double gain=InvestmentOpportunity.Prospect(view,tile,out paid,out site);
                    int dev=(removed+1)/2,iron=EraEconomy.IronPrice(view,removed),borrow=paid+iron>(int)me["cash"]?1:0;
                    if(actions<extra+dev+1+(Conversion.Sellable(tile)?1:0)+borrow || paid+iron>(int)me["cash"]+(borrow>0 && (int)me["income"]>=-7?30:0))continue;
                    best=Math.Max(best,gain-dev*1.5-iron*.25);
                }
                return best;
            };
            return Math.Min(8,(opportunity(wild,1)-opportunity(v,0))*.75)-2;
        }
        public static double CanalPenalty(JObject root,List<string[]> actions)
        {
            if(EraEconomy.Rail(root))return 0;
            int seat=(int)root["seat"];var working=(JObject)root.DeepClone();double penalty=0;
            bool opening=!root["links"].Any(l=>(int?)l["owner"]==seat);
            foreach(var action in actions)foreach(var id in action.Where(x=>x.StartsWith("network|")))
            {
                string target=id.Split('|')[1];var link=working["links"].Single(l=>(string)l["id"]==target);bool needed=false;
                foreach(var tile in working["slots"].Where(Present).Where(s=>(int)s["tile"]["seat"]==seat && Conversion.Sellable(s["tile"]) && !(bool)s["tile"]["flipped"]))
                {
                    string region=Evaluator.Region((string)tile["id"]),type=(string)tile["tile"]["type"];
                    if(Conversion.MerchantDistance(working,region,type,target)<Conversion.MerchantDistance(working,region,type))needed=true;
                }
                // A same-turn build into the newly enabled area is another
                // concrete purpose; don't penalize its prerequisite connection.
                if(actions.SelectMany(a=>a).Any(x=>x.StartsWith("build|") && link["regions"].Any(r=>(string)r==Evaluator.Region(x.Split('|')[1]))))needed=true;
                if(!opening && !needed && Conversion.Actions(root,seat)>4)penalty+=(int)Conversion.Player(root,seat)["income"]<5?2:1;
                link["owner"]=seat;opening=false;
            }
            return Math.Min(4,penalty);
        }
        public static double RailPreference(List<string[]> actions)
        {return Math.Min(1,actions.Count(a=>a.Count(x=>x.StartsWith("network|"))==2)*.5);}
    }
}
