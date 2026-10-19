using System;
using System.Linq;
using Newtonsoft.Json.Linq;

namespace BrassLegacyAI
{
    // V5 quotes one complete investment, not a reward for discarding tiles.
    // This file is compiled by both the game and the lab shared-search runtime.
    public sealed class DevelopmentPlan
    {
        public string site;
        public int removed,develop_actions,connections,connection_cost,breweries,iron_cost,cost,actions,bridge,reserve;
        public bool borrow;
        public double prospect;
        public int Budget {get{return cost+bridge+reserve;}}
        public double Quality {get{return (prospect-iron_cost*.30-connection_cost*.20-bridge*.25)/
            (1+develop_actions*.55+connections*.5+breweries*.6+(borrow?.5:0));}}
    }

    public static class FundedDevelopment
    {
        public const string Version="funded-industry-v5/1.0.0";
        public const int MaximumRemovals=6;
        static bool Present(JToken s){return s["tile"]!=null && s["tile"].Type!=JTokenType.Null;}
        public static int BoardIron(JObject v)
        {return v["slots"].Where(Present).Where(s=>(string)s["tile"]["type"]=="IronWorks").Sum(s=>(int)s["tile"]["resource"]);}
        public static int IronBudget(JObject v,int count)
        {return EraEconomy.IronPrice(v,Math.Max(0,count-BoardIron(v)));}
        public static int DevelopActions(JObject v,int removed)
        {
            if(removed==0)return 0;
            // At the optional second removal the current action is still open.
            return (string)v["pending"]["kind"]=="develop"?1+removed/2:(removed+1)/2;
        }
        public static DevelopmentPlan Quote(JObject v,JToken tile,int removed,string region=null)
        {
            int seat=(int)v["seat"],available=Conversion.Actions(v,seat);
            if(tile==null || removed<0 || removed>MaximumRemovals || available<1)return null;
            var me=Conversion.Player(v,seat);string type=(string)tile["type"];
            var tiles=me["mat"].Single(m=>(string)m["type"]==type)["tiles"].ToArray();
            if(removed>0 && (removed>=tiles.Length || !tiles.Take(removed).All(t=>(bool)t["developable"]) ||
                !JToken.DeepEquals(tile,tiles[removed])))return null;
            int paid;string site;
            double prospect=InvestmentOpportunity.Prospect(v,tile,out paid,out site,region);
            if(prospect<=0)return null;
            int dev=DevelopActions(v,removed),links=Conversion.Sellable(tile)?Conversion.MerchantDistance(v,Evaluator.Region(site),type):0;
            if(links>2)return null;
            int total=paid+EraEconomy.ConnectionBudget(v,links),buildIron=(int)tile["iron"];
            int originallyQuotedIron=IronBudget(v,buildIron),allIron=removed+buildIron;
            int barrels=EraEconomy.Rail(v)?2:1,beer=Conversion.Sellable(tile)?(int)tile["beer"]:0;
            int supply=CompetitiveEconomy.AvailableBeer(v,Evaluator.Region(site),type,dev+links+1);
            int breweries=Math.Max(0,(beer-supply+barrels-1)/barrels);
            if(breweries>0)
            {
                var brewTiles=me["mat"].Single(m=>(string)m["type"]=="Brewery")["tiles"].ToArray();
                int skip=EraEconomy.Rail(v)?brewTiles.TakeWhile(t=>(bool)t["canal_only"]).Count():0;
                if(!brewTiles.Take(skip).All(t=>(bool)t["developable"]) || skip+breweries>brewTiles.Length)return null;
                dev+=(skip+1)/2;allIron+=skip;
                var reserved=(JObject)v.DeepClone();
                var industry=reserved["slots"].Single(s=>(string)s["id"]==site);
                industry["tile"]=tile.DeepClone();industry["tile"]["seat"]=seat;
                foreach(var brew in brewTiles.Skip(skip).Take(breweries))
                {
                    int price;string brewSite;
                    if(!InvestmentOpportunity.FundingSite(reserved,brew,out price,out brewSite))return null;
                    total+=price;allIron+=(int)brew["iron"];originallyQuotedIron+=IronBudget(v,(int)brew["iron"]);
                    var slot=reserved["slots"].Single(s=>(string)s["id"]==brewSite);
                    slot["tile"]=brew.DeepClone();slot["tile"]["seat"]=seat;slot["tile"]["resource"]=barrels;
                }
            }
            // Development and ALL subsequent builds share the same public iron
            // stock and price ladder. Never spend one free cube more than once.
            total+=IronBudget(v,allIron)-originallyQuotedIron;
            int cash=(int)me["cash"],income=(int)me["income"];
            int reserve=removed>0?4:0;
            int needed=dev+1+(Conversion.Sellable(tile)?1:0)+links+breweries;
            bool borrow=false;int bridge=0;
            for(int attempt=0;attempt<2;attempt++)
            {
                int actions=needed+(borrow?1:0);
                // Include the next settlement even when this is the last action
                // of the turn. Do not borrow against forecast flips/market sales.
                int current=Math.Max(1,(int)v["context"]["actions_remaining"]);
                int payments=Math.Max(1,1+Math.Max(0,actions-current+1)/2);
                bridge=Math.Max(0,-income+(borrow?3:0))*payments;
                if(total+bridge+reserve<=cash+(borrow?30:0))
                {
                    if(actions>available || (borrow && income<-7))return null;
                    return new DevelopmentPlan{site=site,removed=removed,develop_actions=dev,connections=links,
                        connection_cost=EraEconomy.ConnectionBudget(v,links),breweries=breweries,
                        iron_cost=IronBudget(v,allIron)-IronBudget(v,allIron-removed),cost=total,actions=actions,
                        bridge=bridge,reserve=reserve,borrow=borrow,prospect=prospect};
                }
                borrow=true;
            }
            return null;
        }
        public static bool CheapFunded(JObject v,DevelopmentPlan plan)
        {return !plan.borrow && plan.removed>0 && plan.iron_cost<=plan.removed*2;}
    }
}
