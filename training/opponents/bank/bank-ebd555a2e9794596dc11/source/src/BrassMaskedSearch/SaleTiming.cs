using System;
using System.Linq;
using System.Collections.Generic;
using Newtonsoft.Json.Linq;
using BrassHost;

namespace BrassLegacyAI
{
    // Only permitted public resources and our own cards enter sale timing.
    // All grouped sales are still executed and paid by the strict rules host.
    public static class SaleTiming
    {
        static bool Present(JToken s){return s["tile"]!=null && s["tile"].Type!=JTokenType.Null;}
        static bool Unsold(JToken s,int seat){return Present(s) && (int)s["tile"]["seat"]==seat && Conversion.Sellable(s["tile"]) && !(bool)s["tile"]["flipped"];}
        static JToken[] Buyers(JObject v,JToken slot)
        {
            var reach=Evaluator.Reach(v,Evaluator.Region((string)slot["id"]));
            return v["merchants"].Where(m=>m["types"]!=null && m["types"].Type!=JTokenType.Null &&
                m["types"].Any(t=>(string)t==(string)slot["tile"]["type"]) && reach.Contains(Evaluator.Region((string)m["id"]))).ToArray();
        }
        static JToken[] Barrels(JObject v,JToken slot)
        {
            int seat=(int)v["seat"];var reach=Evaluator.Reach(v,Evaluator.Region((string)slot["id"]));
            // Spend local/external supply before globally usable own barrels.
            return v["slots"].Where(Present).Where(s=>(string)s["tile"]["type"]=="Brewery" &&
                ((int)s["tile"]["seat"]==seat || reach.Contains(Evaluator.Region((string)s["id"]))))
                .OrderBy(s=>(int)s["tile"]["seat"]==seat).ThenBy(s=>(string)s["id"],StringComparer.Ordinal).ToArray();
        }
        public static bool Ready(JObject v,JToken slot)
        {
            if(!Unsold(slot,(int)v["seat"]) || Conversion.Actions(v,(int)v["seat"])==0)return false;
            int supply=Barrels(v,slot).Sum(s=>(int)s["tile"]["resource"]);
            return Buyers(v,slot).Any(m=>supply+(int)m["beer"]>=(int)slot["tile"]["beer"]);
        }
        static bool Reserve(JObject v,JToken slot,string buyer,Dictionary<string,int> barrels,Dictionary<string,int> merchants)
        {
            var sources=Barrels(v,slot);int need=(int)slot["tile"]["beer"];
            if(sources.Sum(s=>barrels[(string)s["id"]])+merchants[buyer]<need)return false;
            int take=Math.Min(need,merchants[buyer]);need-=take;merchants[buyer]-=take;
            foreach(var source in sources)
            {string id=(string)source["id"];take=Math.Min(need,barrels[id]);barrels[id]-=take;need-=take;}
            return true;
        }
        public static int BatchCount(JObject v,string target,string merchant)
        {
            int seat=(int)v["seat"];var first=v["slots"].FirstOrDefault(s=>(string)s["id"]==target);
            if(first==null || !Ready(v,first) || !Buyers(v,first).Any(m=>(string)m["id"]==merchant))return 0;
            var barrels=v["slots"].Where(Present).Where(s=>(string)s["tile"]["type"]=="Brewery")
                .ToDictionary(s=>(string)s["id"],s=>(int)s["tile"]["resource"]);
            var merchants=v["merchants"].ToDictionary(m=>(string)m["id"],m=>(int)m["beer"]);
            if(!Reserve(v,first,merchant,barrels,merchants))return 0;
            int count=1;
            foreach(var slot in v["slots"].Where(s=>Unsold(s,seat) && (string)s["id"]!=target)
                .OrderByDescending(s=>(int)s["tile"]["vp"]).ThenBy(s=>(string)s["id"],StringComparer.Ordinal))
                foreach(var buyer in Buyers(v,slot).OrderByDescending(m=>merchants[(string)m["id"]]).ThenBy(m=>(string)m["id"],StringComparer.Ordinal))
                    if(Reserve(v,slot,(string)buyer["id"],barrels,merchants)){count++;break;}
            // A conservative feasible allocation, not an exhaustive optimal sale.
            // The real legal mask still executes and pays every continuation.
            return count;
        }
        public static bool WaitForBatch(JObject v,string target,string merchant,bool merchantBeerUsed=false)
        {
            int seat=(int)v["seat"],actions=Conversion.Actions(v,seat);
            if(actions<=4 || (bool)v["context"]["terminal"] || (string)v["pending"]["kind"]=="sell")return false;
            var slot=v["slots"].FirstOrDefault(s=>(string)s["id"]==target);
            if(slot==null || !Ready(v,slot) || BatchCount(v,target,merchant)>=2)return false;
            var tile=slot["tile"];int need=(int)tile["beer"];
            var buyer=v["merchants"].FirstOrDefault(m=>(string)m["id"]==merchant);
            if(merchantBeerUsed || (buyer!=null && need>0 && (int)buyer["beer"]>0))return false;
            var player=Conversion.Player(v,seat);
            // Income conversion can avert forced liquidation or credit lockout.
            if(CashManagement.Project(v,seat).shortfall>0 || (int)player["income"]<0)return false;
            int ownBeer=v["slots"].Where(Present).Where(s=>(int)s["tile"]["seat"]==seat && (string)s["tile"]["type"]=="Brewery").Sum(s=>(int)s["tile"]["resource"]);
            var ready=v["slots"].Where(s=>Unsold(s,seat) && Ready(v,s)).ToArray();
            // Opponent beer and insufficient own barrels make delay dangerous.
            if(need>0 && ownBeer<Math.Max(need,ready.Sum(s=>(int)s["tile"]["beer"])))return false;
            if(ready.Length>=2)return false;
            if(v["own_hand"].Count()<2)return false;
            // Own beer is not private when another unsold industry can reach it.
            if(v["slots"].Where(Present).Any(s=>(int)s["tile"]["seat"]!=seat && Conversion.Sellable(s["tile"]) &&
                !(bool)s["tile"]["flipped"] && (int)s["tile"]["beer"]>0 &&
                v["slots"].Where(Present).Any(b=>(int)b["tile"]["seat"]==seat && (string)b["tile"]["type"]=="Brewery" &&
                    (int)b["tile"]["resource"]>0 && Evaluator.Reach(v,Evaluator.Region((string)s["id"])).Contains(Evaluator.Region((string)b["id"])))))return false;
            // Wait only when another funded, reachable sellable investment can
            // join the sale. Do not postpone a lone tile with no follow-up.
            foreach(var mat in player["mat"])
            {
                var next=mat["tiles"].FirstOrDefault();if(next==null || !Conversion.Sellable(next))continue;
                var plan=FundedDevelopment.Quote(v,next,0);
                if(plan!=null && !plan.borrow && plan.connections==0 && plan.breweries==0 && plan.actions==2 &&
                    ownBeer>=need+(int)next["beer"])return true;
            }
            return false;
        }
        public static double Proposal(JObject v,Choice c)
        {
            var tile=v["slots"].Single(s=>(string)s["id"]==c.target)["tile"];
            // Continuations keep selling all feasible tiles in this SAME action.
            if((string)v["pending"]["kind"]=="sell")return 40+(int)tile["vp"];
            int batch=BatchCount(v,c.target,c.merchant);bool group=batch>=2;
            bool wait=!group && WaitForBatch(v,c.target,c.merchant);
            var buyer=v["merchants"].Single(m=>(string)m["id"]==c.merchant);
            return (wait?2:34)+(int)tile["vp"]*(wait?.35:1)+Math.Min(9,Math.Max(0,batch-1)*3)+
                ((int)tile["beer"]>0 && (int)buyer["beer"]>0?Conversion.MerchantBonus(v,buyer):0);
        }
        public static double SinglePenalty(JObject root,List<string[]> actions)
        {
            double penalty=0;
            foreach(var action in actions)
            {
                var sales=action.Where(id=>id.StartsWith("sell|",StringComparison.Ordinal)).ToArray();
                if(sales.Length!=1)continue;
                var fields=sales[0].Split('|');
                bool merchantBeer=action.Any(id=>id.StartsWith("beer|merchant:",StringComparison.Ordinal));
                if(fields.Length>=5 && WaitForBatch(root,fields[1],fields[4],merchantBeer))penalty+=3;
            }
            return Math.Min(4,penalty);
        }
    }
}
