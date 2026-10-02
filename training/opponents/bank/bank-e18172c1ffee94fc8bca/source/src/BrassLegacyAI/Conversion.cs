using System;
using System.Linq;
using System.Collections.Generic;
using Newtonsoft.Json.Linq;
using BrassHost;

namespace BrassLegacyAI
{
    // Public-state forecasts only. These are conservative heuristic estimates,
    // never a promise that another player will consume our resources.
    public static class Conversion
    {
        public static JToken Player(JObject v,int seat){return v["players"].Single(p=>(int)p["seat"]==seat);}
        public static int[] Seats(JObject v){return v["players"].Select(p=>(int)p["seat"]).ToArray();}
        public static int RoundLimit(JObject v){return (int?)v["context"]["round_limit"]??8;}
        public static bool Intro(JObject v){return (bool?)v["context"]["introductory"]??false;}
        public static int Actions(JObject v,int seat)
        {
            if((bool)v["context"]["terminal"])return 0;
            int rounds=Math.Max(0,RoundLimit(v)-(int)v["context"]["round"]);
            int current=(int)v["context"]["turn"]-1;
            int index=Array.IndexOf(v["order"].Select(x=>(int)x).ToArray(),seat);
            int now=index==current?(int)v["context"]["actions_remaining"]:index>current?2:0;
            if(index>current && (string)v["context"]["era"]=="CanalEra" && (int)v["context"]["round"]==1)now=1;
            return rounds*2+now;
        }
        public static bool Sellable(JToken tile){return new[]{"CottonMill","Manufacturer","Pottery"}.Contains((string)tile["type"]);}
        static bool Present(JToken slot){return slot["tile"]!=null && slot["tile"].Type!=JTokenType.Null;}
        public static int MerchantDistance(JObject v,string region,string type,string addedLink=null)
        {
            // Count missing links, not geographical hops. Only the current era's
            // unbuilt links are available; existing public connections cost zero.
            var distance=new Dictionary<string,int>{{region,0}};
            string era=(string)v["context"]["era"]=="CanalEra"?"Canal":"Rail";
            bool changed=true;
            while(changed)
            {
                changed=false;
                foreach(var link in v["links"])
                {
                    bool built=(link["owner"]!=null && link["owner"].Type!=JTokenType.Null) || (string)link["id"]==addedLink;
                    if(!built && (string)link["era"]!=era)continue;
                    var ends=link["regions"].Select(x=>(string)x).ToArray();
                    foreach(var from in ends)foreach(var to in ends)
                    {
                        int d;if(!distance.TryGetValue(from,out d))continue;
                        int cost=d+(built?0:1),old;
                        if(!distance.TryGetValue(to,out old) || cost<old){distance[to]=cost;changed=true;}
                    }
                }
            }
            return v["merchants"].Where(m=>m["types"]!=null && m["types"].Type!=JTokenType.Null && m["types"].Any(x=>(string)x==type))
                .Select(m=>distance.ContainsKey(Evaluator.Region((string)m["id"]))?distance[Evaluator.Region((string)m["id"])]:99).DefaultIfEmpty(99).Min();
        }
        public static double Completion(JObject v,JToken slot,int seat)
        {
            var t=slot["tile"];if((bool)t["flipped"])return 1;
            int own=Actions(v,seat);string type=(string)t["type"];
            var reach=Evaluator.Reach(v,Evaluator.Region((string)slot["id"]));
            if(Sellable(t))
            {
                if(own==0)return 0; // No other player can sell this tile for us.
                var merchants=v["merchants"].Where(m=>m["types"]!=null && m["types"].Type!=JTokenType.Null && m["types"].Any(x=>(string)x==type) && reach.Contains(Evaluator.Region((string)m["id"])));
                int beer=v["slots"].Where(Present).Where(s=>(string)s["tile"]["type"]=="Brewery" && ((int)s["tile"]["seat"]==seat || reach.Contains(Evaluator.Region((string)s["id"])))).Sum(s=>(int)s["tile"]["resource"]);
                bool ready=merchants.Any(m=>beer+(int)m["beer"]>=(int)t["beer"]);
                // A sale still costs an action, and beer/merchants can be contested.
                if(ready)return own<=2?0.68:0.82;
                if(own<2)return 0;
                if(merchants.Any())return 0.38;
                int links=MerchantDistance(v,Evaluator.Region((string)slot["id"]),type);
                // A nearby merchant is a concrete completion route. Allow for
                // connecting, selling, and a beer/financing contingency; do not
                // credit an inaccessible far-away tile like a one-link setup.
                int cost=EraEconomy.ConnectionBudget(v,links);
                var player=Player(v,seat);
                int financing=(int)player["cash"]<cost?1:0;
                bool canFinance=financing==0 || (int)player["income"]>=-7;
                if(links<=2 && own>=links+2+financing && canFinance)
                    return links==1?0.60:0.36;
                return own>=links+2 && canFinance?0.15:0;
            }
            int resource=(int)t["resource"];if(resource==0)return 1;
            double demand=0,speculativeBeer=0;
            foreach(int actor in Seats(v))
            {
                int actions=Actions(v,actor);if(actions==0)continue;
                // An opponent needs a connection to use our brewery. Our own
                // beer can be used anywhere; iron has no connectivity restriction.
                double factor=actor==seat?1:0.55;
                if(type=="Brewery")
                {
                    double sales=v["slots"].Where(Present).Where(s=>Sellable(s["tile"]) && !(bool)s["tile"]["flipped"] && (int)s["tile"]["seat"]==actor &&
                        (actor==seat || reach.Contains(Evaluator.Region((string)s["id"])))).Sum(s=>{
                            var route=Evaluator.Reach(v,Evaluator.Region((string)s["id"]));
                            var buyers=v["merchants"].Where(m=>m["types"]!=null && m["types"].Type!=JTokenType.Null && m["types"].Any(x=>(string)x==(string)s["tile"]["type"]) && route.Contains(Evaluator.Region((string)m["id"]))).ToArray();
                            if(buyers.Length==0)return 0;
                            int need=(int)s["tile"]["beer"];
                            // Merchant barrels compete with brewery demand; they
                            // are useful bonuses, not guaranteed brewery buyers.
                            return need-0.5*Math.Min(need,buyers.Max(m=>(int)m["beer"]));
                        });
                    // Public, currently reachable sales dominate. Speculative
                    // future builds carry very little weight near the deadline.
                    demand+=factor*Math.Min(sales,actions*2)*0.65;
                    speculativeBeer+=factor*Math.Max(0,actions-2)*0.015;
                    if((string)v["context"]["era"]=="RailEra" && (actor==seat || reach.Count>1))speculativeBeer+=factor*Math.Min(2,actions)*0.06;
                }
                else demand+=factor*actions*(type=="IronWorks"?0.26:(string)v["context"]["era"]=="RailEra"?0.42:0.18);
            }
            // A long era does not make multiple unsupported breweries likely
            // to flip. Cap total speculative beer demand before dividing among
            // competing barrels. Actual consumption still scores exactly.
            if(type=="Brewery")demand+=Math.Min(0.25,speculativeBeer);
            if(type=="CoalMine" && reach.Count==1)demand*=0.20;
            double supply=v["slots"].Where(Present).Where(s=>(string)s["tile"]["type"]==type &&
                (type=="IronWorks" || reach.Contains(Evaluator.Region((string)s["id"])) || (type=="Brewery" && (int)s["tile"]["seat"]==seat))).Sum(s=>(int)s["tile"]["resource"]);
            return Math.Min(0.85,demand/Math.Max(resource,supply));
        }
        public static Dictionary<string,double> Forecast(JObject v)
        {
            var tiles=v["slots"].Where(Present).ToArray();
            var result=tiles.ToDictionary(s=>(string)s["id"],s=>Completion(v,s,(int)s["tile"]["seat"]));
            // A board barrel is promised once across the public conversion forecast.
            // Ready sales can share one sale action; do not charge one action per tile.
            var barrels=tiles.Where(s=>(string)s["tile"]["type"]=="Brewery")
                .ToDictionary(s=>(string)s["id"],s=>(int)s["tile"]["resource"]);
            var merchants=v["merchants"].ToDictionary(m=>(string)m["id"],m=>(int)m["beer"]);
            var routes=new Dictionary<string,HashSet<string>>();
            foreach(var slot in tiles.Where(s=>Sellable(s["tile"]) && !(bool)s["tile"]["flipped"])
                .OrderByDescending(s=>(int)s["tile"]["vp"]+(int)s["tile"]["income"]*.4).ThenBy(s=>(string)s["id"],StringComparer.Ordinal))
            {
                int seat=(int)slot["tile"]["seat"],actions=Actions(v,seat),need=(int)slot["tile"]["beer"];
                if(actions==0 || need==0)continue;
                string region=Evaluator.Region((string)slot["id"]);HashSet<string> reach;
                if(!routes.TryGetValue(region,out reach)){reach=Evaluator.Reach(v,region);routes.Add(region,reach);}
                var buyers=v["merchants"].Where(m=>m["types"]!=null && m["types"].Type!=JTokenType.Null &&
                    m["types"].Any(x=>(string)x==(string)slot["tile"]["type"]) && reach.Contains(Evaluator.Region((string)m["id"])))
                    .OrderByDescending(m=>merchants[(string)m["id"]]).ThenBy(m=>(string)m["id"],StringComparer.Ordinal).ToArray();
                if(buyers.Length==0)continue; // The independent setup forecast remains conservative.
                var supply=tiles.Where(s=>(string)s["tile"]["type"]=="Brewery" &&
                    ((int)s["tile"]["seat"]==seat || reach.Contains(Evaluator.Region((string)s["id"]))))
                    .OrderByDescending(s=>(int)s["tile"]["seat"]==seat).ThenBy(s=>(string)s["id"],StringComparer.Ordinal).ToArray();
                var buyer=buyers.First();string merchant=(string)buyer["id"];
                if(supply.Sum(s=>barrels[(string)s["id"]])+merchants[merchant]<need)
                {result[(string)slot["id"]]=Math.Min(result[(string)slot["id"]],actions<=1?0:.25);continue;}
                int used=Math.Min(need,merchants[merchant]);merchants[merchant]-=used;need-=used;
                foreach(var brewery in supply){string id=(string)brewery["id"];used=Math.Min(need,barrels[id]);barrels[id]-=used;need-=used;}
            }
            return result;
        }
        public static double IndustryValue(JObject v,JToken slot,int seat,double? probability=null)
        {
            var t=slot["tile"];double p=probability??Completion(v,slot,seat);
            // Level 2+ survives Canal. Its next-era value is separate from the
            // expiring Canal award, so a viable carry-over investment stays legal.
            double later=(string)v["context"]["era"]=="CanalEra" && !Intro(v) && (int)t["level"]>1?
                ((bool)t["flipped"]?0.80:0.45*(Sellable(t)?1:Math.Max(0.20,p))):0;
            return (int)t["vp"]*(p+later);
        }
        public static double MerchantBonus(JObject v,JToken merchant)
        {
            string bonus=(string)merchant["bonus"];
            if(bonus=="VP")return Evaluator.Region((string)merchant["id"])=="Shrewsbury"?4:3;
            if(bonus=="Money")return 5*0.32;
            if(bonus=="Income")return Math.Min(4,Actions(v,(int)v["seat"])*0.4);
            if(bonus=="Develop")return Player(v,(int)v["seat"])["mat"].Any(m=>m["tiles"].Any() && (bool)m["tiles"][0]["developable"] && (int)m["tiles"][0]["level"]==1)?2:0;
            return 0;
        }
        public static double BonusDevelopment(JObject v,Choice c)
        {
            if(c.kind=="bonus_skip")return 0;
            var tiles=Player(v,(int)v["seat"])["mat"].Single(m=>(string)m["type"]==c.industry)["tiles"].ToArray();
            if(tiles.Length<2 || Actions(v,(int)v["seat"])==0)return -1;
            // Spend a free development on an actual upgrade. Discarding the
            // last useful tile, or removing tiles with no time to rebuild, loses
            // to the legal skip. No automatic brewery/cotton preference.
            return 1+Math.Max(0,(int)tiles[1]["vp"]-(int)tiles[0]["vp"])*0.3+
                ((int)tiles[0]["level"]==1 && (int)tiles[1]["level"]>1?2:0);
        }
        // Search compares complete afterstates for these three beer-source
        // preferences. No bonus is added to the final score a second time.
        public static double BeerPriority(JObject v,Choice c,int preference)
        {
            if(c.kind!="beer")return Evaluator.Proposal(v,c,true);
            if(c.target.StartsWith("merchant:"))
            {
                var merchant=v["merchants"].Single(m=>(string)m["id"]==c.target.Substring(9));
                return (preference==1?0:20)+MerchantBonus(v,merchant);
            }
            var slot=v["slots"].Single(s=>(string)s["id"]==c.target);var tile=slot["tile"];
            bool own=(int)tile["seat"]==(int)v["seat"];
            double flip=(int)tile["resource"]==1?(int)tile["vp"]*(1-Completion(v,slot,(int)tile["seat"])):0;
            return own?(preference==1?20:preference==2?-10:1)+flip:-2-flip*0.18;
        }
    }
}
