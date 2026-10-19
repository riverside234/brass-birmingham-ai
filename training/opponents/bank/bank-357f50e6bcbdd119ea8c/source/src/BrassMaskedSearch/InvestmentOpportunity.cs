using System;
using System.Collections.Generic;
using System.Linq;
using Newtonsoft.Json.Linq;

namespace BrassLegacyAI
{
    // Prospects rank proposals; hypothetical complete plans still pay through
    // the rules host and earn value only from their resulting public state.
    public static class InvestmentOpportunity
    {
        static bool Present(JToken s){return s["tile"]!=null && s["tile"].Type!=JTokenType.Null;}
        static HashSet<string> Network(JObject v,int seat)
        {
            var network=new HashSet<string>(v["slots"].Where(Present).Where(s=>(int)s["tile"]["seat"]==seat).Select(s=>Evaluator.Region((string)s["id"])));
            foreach(var l in v["links"].Where(l=>(int?)l["owner"]==seat))foreach(var r in l["regions"])network.Add((string)r);
            return network;
        }
        static bool Card(JObject v,JToken slot,string type,HashSet<string> network)
        {
            string region=Evaluator.Region((string)slot["id"]);
            return v["own_hand"].Any(c=>{string card=(string)c;return card=="wild:location" || card=="location:"+region ||
                ((card=="wild:industry" || (card.StartsWith("industry:") && card.Contains(type))) && (network.Count==0 || network.Contains(region)));});
        }
        public static bool FundingSite(JObject v,JToken tile,out int cost,out string site)
        {
            cost=100000;site=null;
            if(tile==null || (EraEconomy.Rail(v) && (bool)tile["canal_only"]) || (!EraEconomy.Rail(v) && (bool)tile["rail_only"]))return false;
            int seat=(int)v["seat"];string type=(string)tile["type"];var network=Network(v,seat);double best=double.NegativeInfinity;
            foreach(var slot in v["slots"].Where(s=>!Present(s) && s["types"].Any(t=>(string)t==type) && Card(v,s,type,network)))
            {
                string region=Evaluator.Region((string)slot["id"]);
                if(!EraEconomy.Rail(v) && v["slots"].Where(Present).Any(s=>(int)s["tile"]["seat"]==seat && Evaluator.Region((string)s["id"])==region))continue;
                int price=EraEconomy.BuildCost(v,tile,region);if(price>=100000)continue;
                double preference=-price+(type=="Brewery"?CompetitiveEconomy.BreweryPrivacy(v,region):0);
                if(preference>best){best=preference;cost=price;site=(string)slot["id"];}
            }
            // A necessary supply building can be correctly priced before its
            // future customer/access exists. Its own projected VP is irrelevant
            // to whether this funded setup is possible; the host still replays it.
            return site!=null;
        }
        public static double Prospect(JObject v,JToken tile,out int cost)
        {string site;return Prospect(v,tile,out cost,out site);}
        public static double Prospect(JObject v,JToken tile,out int cost,out string site)
        {return Prospect(v,tile,out cost,out site,null);}
        public static double Prospect(JObject v,JToken tile,out int cost,out string site,string preferredRegion)
        {
            cost=100000;site=null;int seat=(int)v["seat"],actions=Conversion.Actions(v,seat);
            if(tile==null || (EraEconomy.Rail(v) && (bool)tile["canal_only"]) || (!EraEconomy.Rail(v) && (bool)tile["rail_only"]) || actions<1)return 0;
            string type=(string)tile["type"];double best=0;var network=Network(v,seat);
            foreach(var slot in v["slots"].Where(s=>!Present(s) && s["types"].Any(t=>(string)t==type) && Card(v,s,type,network)))
            {
                // Canal permits one owned industry per city. A vacant second
                // space is not a replacement opportunity; it previously
                // encouraged upgrades that had no legal follow-up build.
                // Actual overbuild choices remain in the full rules mask.
                if(!EraEconomy.Rail(v) && v["slots"].Where(Present).Any(s=>(int)s["tile"]["seat"]==seat &&
                    Evaluator.Region((string)s["id"])==Evaluator.Region((string)slot["id"])))continue;
                int paid=EraEconomy.BuildCost(v,tile,Evaluator.Region((string)slot["id"]));if(paid>=100000)continue;
                var imagined=(JObject)slot.DeepClone();imagined["tile"]=tile.DeepClone();
                imagined["tile"]["seat"]=seat;
                if(type=="Brewery")imagined["tile"]["resource"]=EraEconomy.Rail(v)?2:1;
                int receipts;int sold=CompetitiveEconomy.MarketSale(v,tile,Evaluator.Region((string)slot["id"]),out receipts);
                if(sold>0){imagined["tile"]["resource"]=(int)tile["resource"]-sold;imagined["tile"]["flipped"]=sold==(int)tile["resource"];}
                double probability=Conversion.Completion(v,imagined,seat);
                if(Conversion.Sellable(tile) && actions<2)probability=0;
                double gain=Conversion.IndustryValue(v,imagined,seat,probability)+(int)tile["income"]*probability*.4-paid*.16+receipts*.16;
                if(preferredRegion!=null && Evaluator.Region((string)slot["id"])==preferredRegion)gain+=1;
                if(gain>best){best=gain;cost=paid;site=(string)slot["id"];}
            }
            return best;
        }
        public static double Development(JObject v,string type)
        {
            int seat=(int)v["seat"];
            var tiles=Conversion.Player(v,seat)["mat"].Single(m=>(string)m["type"]==type)["tiles"].ToArray();
            if(tiles.Length<2 || Conversion.Actions(v,seat)<2)return -6;
            var current=FundedDevelopment.Quote(v,tiles[0],0);
            double before=current==null?0:current.prospect,best=-5;
            for(int removed=1;removed<=FundedDevelopment.MaximumRemovals && removed<tiles.Length;removed++)
            {
                if(!(bool)tiles[removed-1]["developable"])break;
                bool expiring=EraEconomy.Rail(v) && (bool)tiles[0]["canal_only"];
                if(!expiring && (int)tiles[removed]["level"]<=(int)tiles[0]["level"])continue;
                var plan=FundedDevelopment.Quote(v,tiles[removed],removed);if(plan==null)continue;
                // A loan can fund a later goal, but cannot pay today's iron.
                if(FundedDevelopment.IronBudget(v,removed)>(int)Conversion.Player(v,seat)["cash"])continue;
                double value=Math.Min(12,(plan.prospect-before)*.95+(expiring?2:0))-plan.iron_cost*.35-.25-
                    1.0*(plan.develop_actions-1)-plan.connections*.5-plan.breweries*.6-plan.bridge*.25-(plan.borrow?.75:0);
                if(value>0 && FundedDevelopment.CheapFunded(v,plan))value*=1.20;
                best=Math.Max(best,value);
            }
            return best;
        }
        public static double Loan(JObject v)
        {
            var p=Conversion.Player(v,(int)v["seat"]);int cash=(int)p["cash"],actions=Conversion.Actions(v,(int)v["seat"]);
            if(CashManagement.Project(v,(int)v["seat"]).shortfall>0)return 12;
            if(actions<2)return -4;
            double best=0;
            foreach(var mat in p["mat"])
            {
                var tile=mat["tiles"].FirstOrDefault();int cost;double prospect=Prospect(v,tile,out cost);
                if(cost>cash && cost<=cash+30 && prospect>0 && (!Conversion.Sellable(tile) || actions>=3))best=Math.Max(best,prospect);
            }
            // Borrow for an identified investment, not a fixed cash threshold.
            return best>0?Math.Min(9,2+best*.5):(cash<8?1:-4);
        }
        public static double MatValue(JObject v,JToken tile)
        {
            var current=FundedDevelopment.Quote(v,tile,0);
            double value=current==null?0:Math.Min(12,current.prospect*.65)-current.connections*.5-current.breweries*.6-current.bridge*.25;
            var player=Conversion.Player(v,(int)v["seat"]);
            var tiles=player["mat"].Single(m=>(string)m["type"]==(string)tile["type"])["tiles"].ToArray();
            // An unusable Canal-only head in Rail must not hide a reachable
            // upgrade. Keep a small bounded option portfolio, never a tile-count
            // reward; the last action still cannot earn development option value.
            if(tiles.Length>0 && JToken.DeepEquals(tile,tiles[0]))
            {
                var levels=new HashSet<int>();
                for(int removed=1;removed<=FundedDevelopment.MaximumRemovals && removed<tiles.Length;removed++)
                {
                    if(!(bool)tiles[removed-1]["developable"])break;
                    var next=tiles[removed];int level=(int)next["level"];
                    if(level<=(int)tile["level"] || !levels.Add(level))continue;
                    var plan=FundedDevelopment.Quote(v,next,removed);if(plan==null)continue;
                    value=Math.Max(value,Math.Min(12,plan.prospect*.65)-plan.develop_actions-plan.iron_cost*.35-
                        plan.connections*.5-plan.breweries*.6-plan.bridge*.25-(plan.borrow?.75:0));
                }
            }
            return Math.Max(0,value);
        }
        public static double Portfolio(IEnumerable<double> options)
        {
            var best=options.OrderByDescending(x=>x).Take(2).ToArray();
            return best.Length==0?0:best[0]+(best.Length>1?best[1]*.35:0);
        }
        public static Dictionary<string,double> FreeDevelopments(JObject v)
        {
            var player=Conversion.Player(v,(int)v["seat"]);
            var options=player["mat"].ToDictionary(m=>(string)m["type"],m=>m["tiles"].Any()?MatValue(v,m["tiles"][0]):0);
            double before=Portfolio(options.Values);var result=new Dictionary<string,double>();
            // The sale in progress still consumes one action. A merchant's
            // free bonus has no iron/action charge of its own, but the unlocked
            // tile needs time, a buyer, a permitted card and actual funding.
            int future=Math.Max(0,Conversion.Actions(v,(int)v["seat"])-1);
            foreach(var mat in player["mat"])
            {
                string type=(string)mat["type"];result[type]=-1;
                var tiles=mat["tiles"].ToArray();if(tiles.Length<2 || !(bool)tiles[0]["developable"])continue;
                int cost;double prospect=Prospect(v,tiles[1],out cost);
                int needed=Conversion.Sellable(tiles[1])?2:1;
                if(cost>(int)player["cash"])
                {if((int)player["income"]<-7 || cost>(int)player["cash"]+30)continue;needed++;}
                if(future<needed || prospect<=0)continue;
                double next=MatValue(v,tiles[1]);
                result[type]=Portfolio(options.Where(kv=>kv.Key!=type).Select(kv=>kv.Value).Concat(new[]{next}))-before;
            }
            // Rank the change in the existing afterstate mat value. No bonus
            // is added again to the completed plan's score.
            return result;
        }
        public static double FreeDevelopment(JObject v,string type){return FreeDevelopments(v)[type];}
    }
}
