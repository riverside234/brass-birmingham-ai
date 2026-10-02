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
        public static bool WaitForBatch(JObject v,string target,string merchant,bool merchantBeerUsed=false)
        {
            int seat=(int)v["seat"],actions=Conversion.Actions(v,seat);
            if(actions<=4 || (bool)v["context"]["terminal"])return false;
            var slot=v["slots"].FirstOrDefault(s=>(string)s["id"]==target);
            if(slot==null || !Present(slot))return false;
            var tile=slot["tile"];int need=(int)tile["beer"];
            var buyer=v["merchants"].FirstOrDefault(m=>(string)m["id"]==merchant);
            if(merchantBeerUsed || (buyer!=null && need>0 && (int)buyer["beer"]>0))return false;
            var player=Conversion.Player(v,seat);
            // Income conversion can avert forced liquidation or credit lockout.
            if(CashManagement.Project(v,seat).shortfall>0 || (int)player["income"]<0)return false;
            int ownBeer=v["slots"].Where(Present).Where(s=>(int)s["tile"]["seat"]==seat && (string)s["tile"]["type"]=="Brewery").Sum(s=>(int)s["tile"]["resource"]);
            var ready=v["slots"].Where(s=>Unsold(s,seat) && Conversion.Completion(v,s,seat)>=.50).ToArray();
            // Opponent beer and insufficient own barrels make delay dangerous.
            if(need>0 && ownBeer<Math.Max(need,ready.Sum(s=>(int)s["tile"]["beer"])))return false;
            if(ready.Length>=2)return true;
            if(v["own_hand"].Count()<2)return false;
            // Wait only when another funded, reachable sellable investment can
            // join the sale. Do not postpone a lone tile with no follow-up.
            foreach(var mat in player["mat"])
            {
                var next=mat["tiles"].FirstOrDefault();if(next==null || !Conversion.Sellable(next))continue;
                int cost;double prospect=InvestmentOpportunity.Prospect(v,next,out cost);
                if(prospect>0 && cost<=(int)player["cash"] && ownBeer>=need+(int)next["beer"])return true;
            }
            return false;
        }
        public static double Proposal(JObject v,Choice c)
        {
            var tile=v["slots"].Single(s=>(string)s["id"]==c.target)["tile"];
            // Continuations keep selling all feasible tiles in this SAME action.
            if((string)v["pending"]["kind"]=="sell")return 30+(int)tile["vp"];
            int seat=(int)v["seat"];
            bool group=v["slots"].Count(s=>Unsold(s,seat) && Conversion.Completion(v,s,seat)>=.68)>=2;
            bool wait=!group && WaitForBatch(v,c.target,c.merchant);
            var buyer=v["merchants"].Single(m=>(string)m["id"]==c.merchant);
            return (wait?2:30)+(int)tile["vp"]*(wait?.35:1)+
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
