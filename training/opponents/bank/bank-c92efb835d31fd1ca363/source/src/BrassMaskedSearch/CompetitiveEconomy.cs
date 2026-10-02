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
        public static int MarketSale(JObject v,JToken tile,string region,out int receipts)
        {
            string type=(string)tile["type"];receipts=0;
            if(type!="CoalMine" && type!="IronWorks")return 0;
            BaseMarket market=type=="CoalMine"?(BaseMarket)new CoalQuote((int)v["coal_market"]):new IronQuote((int)v["iron_market"]);
            bool access=type=="IronWorks" || v["merchants"].Any(m=>Evaluator.Reach(v,region).Contains(Evaluator.Region((string)m["id"])));
            int sold=access?Math.Min((int)tile["resource"],market.AmountOfResourcePossibleToSell()):0;
            receipts=market.GetSellingPrice(sold);return sold;
        }
        public static int AvailableBeer(JObject v,string region,string type,int setupActions=0)
        {
            int seat=(int)v["seat"];var reach=Evaluator.Reach(v,region);
            int external=v["slots"].Where(Present).Where(s=>(string)s["tile"]["type"]=="Brewery" &&
                (int)s["tile"]["seat"]!=seat && reach.Contains(Evaluator.Region((string)s["id"]))).Sum(s=>(int)s["tile"]["resource"]);
            // The immediate legal mask decides the real payment. A longer
            // setup discounts contested public barrels instead of inventing
            // future opponents' breweries or reserving their beer exclusively.
            if(setupActions>2 && external>0)external=Math.Max(1,external/2);
            int merchant=v["merchants"].Where(m=>m["types"]!=null && m["types"].Type!=JTokenType.Null &&
                m["types"].Any(t=>(string)t==type) && reach.Contains(Evaluator.Region((string)m["id"])))
                .Select(m=>(int)m["beer"]).DefaultIfEmpty(0).Max();
            return Math.Max(0,OwnBeer(v,seat)+external+merchant-CommittedBeer(v,seat));
        }
        public static double ResourceProposal(JObject v,JToken tile,string region)
        {
            string type=(string)tile["type"];if(type!="CoalMine" && type!="IronWorks")return 0;
            bool access=type=="IronWorks" || v["merchants"].Any(m=>Evaluator.Reach(v,region).Contains(Evaluator.Region((string)m["id"])));
            int receipts;int sold=MarketSale(v,tile,region,out receipts);
            int income=(int)Conversion.Player(v,(int)v["seat"])["income"];
            double price=type=="CoalMine"?EraEconomy.CoalPrice(v,1):EraEconomy.IronPrice(v,1);
            // Resource industries are useful income/market plays, not the default
            // scoring engine. Cheap, oversupplied coal is especially expensive in
            // action opportunity once income is healthy. Full immediate flips,
            // shortage profits and low-income recovery remain competitive.
            double penalty=type=="CoalMine"?(income>=5?6:income>=2?3:0):(price<=2?3:0);
            if(sold==(int)tile["resource"])penalty*=.25;
            if(price>=4)penalty=0;
            if(type=="CoalMine" && price<4 && sold<(int)tile["resource"])
            {
                // Low income is a reason to recover it, not to accumulate coal
                // that has not produced income. Existing unsold stock and an
                // inaccessible new mine make another identical recovery slower.
                int waiting=v["slots"].Where(Present).Where(s=>(int)s["tile"]["seat"]==(int)v["seat"] &&
                    (string)s["tile"]["type"]=="CoalMine" && (int)s["tile"]["resource"]>0).Sum(s=>(int)s["tile"]["resource"]);
                penalty+=Math.Min(8,waiting*.8)+(access?0:2);
            }
            return receipts*.30-penalty;
        }
        public static double SellableProposal(JObject v,JToken tile,string region)
        {
            if(!Conversion.Sellable(tile))return 0;
            int seat=(int)v["seat"],need=(int)tile["beer"];
            // Existing unsold tiles already own a claim on our barrels. Prefer a
            // higher-value, finishable investment over piling up low-level stock.
            int supply=AvailableBeer(v,region,(string)tile["type"],1);
            return need>supply?-Math.Min(5,2+CommittedBeer(v,seat)*.5):0;
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
        public static double NetworkAccess(JObject v,BrassHost.Choice c)
        {
            if((string)v["context"]["phase"]!="action" || Conversion.Actions(v,(int)v["seat"])<2)return 0;
            int seat=(int)v["seat"];var link=v["links"].Single(l=>(string)l["id"]==c.target);
            var ends=link["regions"].Select(r=>(string)r).ToArray();
            var after=(JObject)v.DeepClone();after["links"].Single(l=>(string)l["id"]==c.target)["owner"]=seat;
            var me=Conversion.Player(v,seat);double opportunity=0;
            int fee=EraEconomy.NetworkCost(v,c),cash=(int)me["cash"];
            foreach(var mat in me["mat"])
            {
                var tiles=mat["tiles"].ToArray();int removed=EraEconomy.Rail(v)?tiles.TakeWhile(t=>(bool)t["canal_only"]).Count():0;
                if(removed>=tiles.Length || removed>0 && !tiles.Take(removed).All(t=>(bool)t["developable"]))continue;
                var tile=tiles[removed];string type=(string)tile["type"];
                int development=(removed+1)/2;
                if(Conversion.Actions(v,seat)<2+development+(Conversion.Sellable(tile)?1:0))continue;
                // A first canal can be anywhere, but once laid it constrains
                // industry-card builds to that network. Preserve actual cards,
                // and compare coal/market/buyer access for the next investment.
                foreach(var site in after["slots"].Where(s=>!Present(s) && ends.Contains(Evaluator.Region((string)s["id"])) && s["types"].Any(t=>(string)t==type)))
                {
                    string region=Evaluator.Region((string)site["id"]);
                    if(!v["own_hand"].Any(card=>((string)card).StartsWith("wild:") || (string)card=="location:"+region || ((string)card).Contains(type)))continue;
                    if(!EraEconomy.Rail(v) && v["slots"].Where(Present).Any(s=>(int)s["tile"]["seat"]==seat && Evaluator.Region((string)s["id"])==region))continue;
                    int paid=EraEconomy.BuildCost(after,tile,region);
                    int iron=EraEconomy.IronPrice(v,removed); // Conservative prerequisite quote; the host pays actual iron.
                    if(paid>=100000 || fee+paid+iron>cash+((int)me["income"]>=-7?30:0))continue;
                    int before=EraEconomy.BuildCost(v,tile,region);
                    int buyerBefore=Conversion.Sellable(tile)?Conversion.MerchantDistance(v,region,type):0;
                    int buyerAfter=Conversion.Sellable(tile)?Conversion.MerchantDistance(after,region,type):0;
                    int receipts,sold=MarketSale(after,tile,region,out receipts);
                    if(Conversion.Sellable(tile) && buyerAfter>1)continue;
                    double gain=(before>=100000?2.5:Math.Max(0,before-paid)*.25)+(buyerAfter<buyerBefore?2:0);
                    // Opening access has value even before there is a tile to
                    // score beside it. Later expansion needs an actual gain.
                    if((int)v["context"]["round"]<=2)gain+=Math.Max(0,(int)tile["vp"]*.12+receipts*.1-fee*.15);
                    if(type=="CoalMine" && sold==0 && (int)me["income"]>=2)gain*=.35;
                    opportunity=Math.Max(opportunity,gain);
                }
            }
            return Math.Min(4,opportunity);
        }
    }
}
