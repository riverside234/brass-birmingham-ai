using System;
using System.Linq;
using Newtonsoft.Json.Linq;
using BoardGameRules.Entities.Markets;
using BrassHost;

namespace BrassLegacyAI
{
    // Public board, public spending and the observer's own commitments only.
    // These are ranking estimates; the host supplies every legal overbuild,
    // resource payment and official score. Nothing here changes the rules mask.
    public static class CompetitiveEconomy
    {
        sealed class CoalQuote:CoalMarket {public CoalQuote(int stock){Amount=stock;}}
        sealed class IronQuote:IronMarket {public IronQuote(int stock){Amount=stock;}}
        static bool Present(JToken s){return s["tile"]!=null && s["tile"].Type!=JTokenType.Null;}
        static bool Unsold(JToken s,int seat){return Present(s) && (int)s["tile"]["seat"]==seat && Conversion.Sellable(s["tile"]) && !(bool)s["tile"]["flipped"];}
        public static int[] NextOrder(JObject v)
        {
            // LINQ's stable ordering matches the engine's existing-order tie break.
            return v["order"].Select(x=>(int)x).OrderBy(s=>(int)Conversion.Player(v,s)["spent"]).ToArray();
        }
        public static double RivalWeight(JObject v)
        {
            if(!EraEconomy.Rail(v))return .18;
            double progress=Math.Max(0,Math.Min(1,((int)v["context"]["round"]-1.0)/Math.Max(1,Conversion.RoundLimit(v)-1)));
            return .35+.45*progress;
        }
        public static int OwnBeer(JObject v,int seat)
        {return v["slots"].Where(Present).Where(s=>(int)s["tile"]["seat"]==seat && (string)s["tile"]["type"]=="Brewery").Sum(s=>(int)s["tile"]["resource"]);}
        public static int CommittedBeer(JObject v,int seat)
        {return v["slots"].Where(s=>Unsold(s,seat)).Sum(s=>(int)s["tile"]["beer"]);}
        public static double ResourceProposal(JObject v,JToken tile,string region)
        {
            string type=(string)tile["type"];if(type!="CoalMine" && type!="IronWorks")return 0;
            BaseMarket market=type=="CoalMine"?(BaseMarket)new CoalQuote((int)v["coal_market"]):new IronQuote((int)v["iron_market"]);
            bool access=type=="IronWorks" || v["merchants"].Any(m=>Evaluator.Reach(v,region).Contains(Evaluator.Region((string)m["id"])));
            int sold=access?Math.Min((int)tile["resource"],market.AmountOfResourcePossibleToSell()):0;
            double receipts=market.GetSellingPrice(sold);
            int income=(int)Conversion.Player(v,(int)v["seat"])["income"];
            double price=type=="CoalMine"?EraEconomy.CoalPrice(v,1):EraEconomy.IronPrice(v,1);
            // Resource industries are useful income/market plays, not the default
            // scoring engine. Cheap, oversupplied coal is especially expensive in
            // action opportunity once income is healthy. Full immediate flips,
            // shortage profits and low-income recovery remain competitive.
            double penalty=type=="CoalMine"?(income>=5?6:income>=2?3:0):(price<=2?3:0);
            if(sold==(int)tile["resource"])penalty*=.25;
            if(price>=4)penalty=0;
            return receipts*.30-penalty;
        }
        public static double SellableProposal(JObject v,JToken tile,string region)
        {
            if(!Conversion.Sellable(tile))return 0;
            int seat=(int)v["seat"],need=(int)tile["beer"];
            // Existing unsold tiles already own a claim on our barrels. Prefer a
            // higher-value, finishable investment over piling up low-level stock.
            int spare=Math.Max(0,OwnBeer(v,seat)-CommittedBeer(v,seat));
            bool merchant=v["merchants"].Any(m=>m["types"]!=null && m["types"].Type!=JTokenType.Null &&
                m["types"].Any(t=>(string)t==(string)tile["type"]) && (int)m["beer"]>=need && Evaluator.Reach(v,region).Contains(Evaluator.Region((string)m["id"])));
            return need>spare && !merchant?-Math.Min(5,2+CommittedBeer(v,seat)*.5):0;
        }
        public static double InventoryRisk(JObject v,int seat,System.Collections.Generic.Dictionary<string,double> forecast)
        {
            if(Conversion.Actions(v,seat)==0)return 0; // Exact final scoring handles unsold tiles.
            return Math.Min(10,v["slots"].Where(s=>Unsold(s,seat)).Sum(s=>
                Math.Max(0,.55-forecast[(string)s["id"]])*(2+(int)s["tile"]["vp"]*.35)));
        }
        public static double Tempo(JObject v,int seat)
        {
            if((bool)v["context"]["terminal"] || (int)v["context"]["round"]>=Conversion.RoundLimit(v))return 0;
            var ready=v["slots"].Where(s=>Unsold(s,seat) && Conversion.Completion(v,s,seat)>=.5).ToArray();
            int ownBeer=OwnBeer(v,seat);double opportunity=ready.Length>=2?Math.Min(4,ready.Sum(s=>(int)s["tile"]["vp"])*.2):0;
            if(ownBeer>0)
            {
                double threatened=0;
                foreach(var brewery in v["slots"].Where(Present).Where(s=>(int)s["tile"]["seat"]==seat && (string)s["tile"]["type"]=="Brewery" && (int)s["tile"]["resource"]>0))
                {
                    string region=Evaluator.Region((string)brewery["id"]);
                    threatened+=v["slots"].Where(Present).Where(s=>(int)s["tile"]["seat"]!=seat && Conversion.Sellable(s["tile"]) && !(bool)s["tile"]["flipped"] &&
                        Evaluator.Reach(v,Evaluator.Region((string)s["id"])).Contains(region)).Sum(s=>(int)s["tile"]["beer"])*.5;
                }
                opportunity+=Math.Min(2,threatened)*Math.Min(1,CommittedBeer(v,seat)/(double)ownBeer);
                // Beer can fund a double rail, but only with legal public network
                // access, cash and spare link tiles; no reward for first place alone.
                var me=Conversion.Player(v,seat);
                if(EraEconomy.Rail(v) && (int)me["cash"]>=15+EraEconomy.CoalPrice(v,2) && (int)me["link_tiles"]>=2 &&
                    ownBeer>CommittedBeer(v,seat))opportunity+=1;
            }
            int index=Array.IndexOf(NextOrder(v),seat);
            // Opponents still to play may change spending. This is a bounded
            // forecast, not a guaranteed next-round turn or fixed action bonus.
            return Math.Min(3,opportunity*.55)*(v["players"].Count()-1-index)/Math.Max(1.0,v["players"].Count()-1);
        }
        public static double BreweryPrivacy(JObject v,string region)
        {
            int seat=(int)v["seat"];if(CommittedBeer(v,seat)<=OwnBeer(v,seat))return 0;
            bool exposed=v["slots"].Where(Present).Any(s=>(int)s["tile"]["seat"]!=seat && Conversion.Sellable(s["tile"]) && !(bool)s["tile"]["flipped"] &&
                (int)s["tile"]["beer"]>0 && Evaluator.Reach(v,Evaluator.Region((string)s["id"])).Contains(region));
            return exposed?-2:2; // An isolated own brewery remains usable anywhere by its owner.
        }
    }
}
