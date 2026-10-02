using System;
using System.Collections.Generic;
using System.Linq;
using Newtonsoft.Json.Linq;

namespace BrassLegacyAI
{
    // Authored conditional templates, not transcribed champion moves. See
    // ai-research/TOURNAMENT_STYLE_V3.md for evidence and attribution limits.
    public static class StrategicOpenings
    {
        public static string Label(string key)
        {
            switch(key){case "cotton":return "Cotton Exporter";case "goods-beer":return "Goods Magnate";
                case "rail-tempo":return "Rail Financier";case "supplier":return "Market Opportunist";
                case "pottery-flex":return "Pottery Investor";default:return "Adaptive Industrialist";}
        }
        static string[] Candidates(JObject v,string style)
        {
            bool rail=EraEconomy.Rail(v);var me=Conversion.Player(v,(int)v["seat"]);
            bool iron=EraEconomy.IronPrice(v,1)>=4,coal=(int)me["income"]<2 || EraEconomy.CoalPrice(v,1)>=5;
            switch(style)
            {
                case "cotton":return new[]{"CottonMill","Brewery","Pottery","Manufacturer"};
                case "goods-beer":return new[]{"Manufacturer","Brewery","CottonMill","Pottery"};
                case "pottery-flex":return new[]{"Pottery","Brewery","CottonMill","Manufacturer"};
                case "supplier":return (iron?new[]{"IronWorks"}:new string[0]).Concat(coal?new[]{"CoalMine"}:new string[0]).Concat(new[]{"CottonMill","Manufacturer","Pottery","Brewery"}).ToArray();
                case "rail-tempo":return new[]{"Brewery","IronWorks","CottonMill","Pottery","Manufacturer"};
                default:return new[]{"CottonMill","Manufacturer","Pottery","Brewery","IronWorks"};
            }
        }
        public static IntentState Intent(JObject v,Profile p)
        {
            if(!p.intent_enabled || p.conversion_policy!=1)return null;
            int seat=(int)v["seat"],actions=Conversion.Actions(v,seat);var me=Conversion.Player(v,seat);
            if(actions<3)return null;
            bool rail=EraEconomy.Rail(v);int cash=(int)me["cash"];
            string[] types=Candidates(v,p.tactic);double best=0;string focus=null,milestone=null;
            int goalLevel=0,goalCost=0;
            foreach(string type in types)
            {
                var mat=me["mat"].Single(m=>(string)m["type"]==type);var tiles=mat["tiles"].ToArray();
                int freeIron=v["slots"].Where(s=>s["tile"]!=null && s["tile"].Type!=JTokenType.Null && (string)s["tile"]["type"]=="IronWorks").Sum(s=>(int)s["tile"]["resource"]);
                // Three developments can reach Cotton III/Goods V-VI. Compare
                // the entire funded setup, including beer and a sale deadline.
                var levels=new HashSet<int>();
                for(int removed=0;removed<=6 && removed<tiles.Length;removed++)
                {
                    if(removed>0 && !tiles.Take(removed).All(t=>(bool)t["developable"]))break;
                    var tile=tiles[removed];int paid;double prospect=InvestmentOpportunity.Prospect(v,tile,out paid);
                    if(!levels.Add((int)tile["level"]))continue;
                    if(prospect<=0)continue;
                    int develops=(removed+1)/2,iron=EraEconomy.IronPrice(v,Math.Max(0,removed-freeIron));
                    int beer=Conversion.Sellable(tile)?(int)tile["beer"]:0;
                    int spare=Math.Max(0,CompetitiveEconomy.OwnBeer(v,seat)-CompetitiveEconomy.CommittedBeer(v,seat));
                    bool merchant=v["merchants"].Any(m=>m["types"]!=null && m["types"].Type!=JTokenType.Null && m["types"].Any(t=>(string)t==type) && (int)m["beer"]>=beer);
                    bool brewery=beer>spare && !merchant;
                    var brew=me["mat"].Single(m=>(string)m["type"]=="Brewery")["tiles"].FirstOrDefault();
                    int brewCost=0;
                    if(brewery && (brew==null || InvestmentOpportunity.Prospect(v,brew,out brewCost)<=0))continue;
                    bool loan=paid+iron+brewCost>cash;
                    if(loan && ((int)me["income"]<-7 || paid+iron+brewCost>cash+30))continue;
                    int needed=develops+1+(Conversion.Sellable(tile)?1:0)+(brewery?1:0)+(loan?1:0);
                    if(actions<needed)continue;
                    double value=(prospect-iron*.25)/(1+develops*.45+(brewery?.5:0)+(loan?.25:0));
                    if(!rail && (int)tile["level"]==1 && Conversion.Sellable(tile))value*=.55;
                    if(type==types[0] && p.tactic!="balanced")value*=1.25;
                    if(type=="IronWorks" && EraEconomy.IronPrice(v,1)<=2)value*=.45;
                    if(type=="CoalMine" && (int)me["income"]>=5 && EraEconomy.CoalPrice(v,1)<=3)value*=.4;
                    if(value<=best)continue;
                    best=value;focus=type;goalLevel=(int)tile["level"];goalCost=paid+iron+brewCost;
                    milestone=loan?"finance_sequence":removed>0?"develop_high_tier":brewery?"secure_own_beer":"build_high_tier";
                }
            }
            if(focus==null)return null;
            if(milestone=="secure_own_beer")focus="Brewery";
            return new IntentState{profile=p.id,seat=seat,revision=(int)v["context"]["revision"],era=(string)v["context"]["era"],
                focus=focus,milestone=milestone,target="",remaining_actions=actions,cash_budget=goalCost,
                reason=Label(p.tactic)+": "+(rail?"Rail":"Canal")+" funded level "+goalLevel+" sequence; adapt to cards, buyers, beer, market and remaining actions"};
        }
    }
}
