using System;
using System.Linq;
using Newtonsoft.Json.Linq;

namespace BrassLegacyAI
{
    // Conditional adaptations of the creator's published descriptions, not
    // unobserved video moves. Attribution and source limits are recorded in
    // ai-research/TOURNAMENT_STYLE_V3.md. All actions still pass the host mask.
    public static class TournamentOpenings
    {
        public const string Version="bilibili-opening-families/1.0.0";
        sealed class Target
        {
            public string type;public int level,quantity;
            public Target(string t,int l,int n=1){type=t;level=l;quantity=n;}
        }
        static Target[] Targets(JObject v,Profile p,out string family)
        {
            bool rail=EraEconomy.Rail(v);var me=Conversion.Player(v,(int)v["seat"]);
            switch(p.tactic)
            {
                case "cotton":family="three-cotton";return new[]{new Target("CottonMill",rail?4:3,3),new Target("Brewery",2,2),new Target("CoalMine",2)};
                case "goods-beer":family="goods-and-beer";return new[]{new Target("Manufacturer",rail?6:2,3),new Target("Brewery",rail?3:2,3),new Target("CoalMine",2)};
                case "rail-tempo":family="rail-and-pottery";return new[]{new Target("Pottery",rail?5:3,2),new Target("Brewery",rail?3:2,3),new Target("CoalMine",2)};
                case "pottery-flex":family="three-flip-pottery";return new[]{new Target("Pottery",rail?5:3,2),new Target("IronWorks",rail?3:2,2),new Target("Brewery",rail?3:2,3)};
                case "supplier":family="standard-iron-and-beer";return new[]{new Target("IronWorks",rail?3:2,2),new Target("Brewery",rail?3:2,2),new Target("Manufacturer",2,2)};
                default:
                    // The zero-loan guide depends on repeated market receipts.
                    // It is an opportunity mode, never a prohibition on rescue.
                    bool market=EraEconomy.IronPrice(v,1)>=4 || EraEconomy.CoalPrice(v,1)>=4;
                    family=market?"zero-loan-market":v["own_hand"].Any(c=>(string)c=="location:Worcester")?"worcester-recovery":"adaptive-opening-book";
                    return new[]{new Target("IronWorks",2,2),new Target("CoalMine",2,2),new Target("Manufacturer",rail?6:2,2),new Target("CottonMill",rail?4:3,3),new Target("Pottery",rail?5:3,2),new Target("Brewery",2,2)};
            }
        }
        static bool Present(JToken s){return s["tile"]!=null && s["tile"].Type!=JTokenType.Null;}
        public static IntentState Intent(JObject v,Profile p)
        {
            if(!p.intent_enabled || p.conversion_policy!=1)return null;
            int seat=(int)v["seat"],actions=Conversion.Actions(v,seat);if(actions<3)return null;
            var me=Conversion.Player(v,seat);int cash=(int)me["cash"],income=(int)me["income"];
            string family;var targets=Targets(v,p,out family);IntentState best=null;double quality=0;
            int boardIron=v["slots"].Where(Present).Where(s=>(string)s["tile"]["type"]=="IronWorks").Sum(s=>(int)s["tile"]["resource"]);
            foreach(var target in targets)
            {
                var own=v["slots"].Where(Present).Where(s=>(int)s["tile"]["seat"]==seat && (string)s["tile"]["type"]==target.type).ToArray();
                if(own.Length>=target.quantity)continue;
                // Supplier/recovery exceptions use actual shortage or low income.
                if(target.type=="CoalMine" && income>=5 && EraEconomy.CoalPrice(v,1)<4)continue;
                if(target.type=="IronWorks" && EraEconomy.IronPrice(v,1)<=2 && income>=2)continue;
                var tiles=me["mat"].Single(m=>(string)m["type"]==target.type)["tiles"].ToArray();
                if(tiles.Length==0)continue;
                int removed=0;
                // Stop at the named tier. Non-developable Pottery must be built;
                // never keep removing III/V merely because a generic goal persists.
                while(removed<tiles.Length-1 && (int)tiles[removed]["level"]<target.level && (bool)tiles[removed]["developable"])removed++;
                var tile=tiles[removed];int paid;string site;
                double prospect=InvestmentOpportunity.Prospect(v,tile,out paid,out site);if(prospect<=0)continue;
                int develop=(removed+1)/2,iron=EraEconomy.IronPrice(v,Math.Max(0,removed-boardIron));
                int connections=Conversion.Sellable(tile)?Conversion.MerchantDistance(v,Evaluator.Region(site),target.type):0;
                if(connections>2)continue;
                int connectionCost=EraEconomy.ConnectionBudget(v,connections);
                var reach=Evaluator.Reach(v,Evaluator.Region(site));
                int merchant=v["merchants"].Where(m=>m["types"]!=null && m["types"].Type!=JTokenType.Null && m["types"].Any(t=>(string)t==target.type) && reach.Contains(Evaluator.Region((string)m["id"])))
                    .Select(m=>(int)m["beer"]).DefaultIfEmpty(0).Max();
                int beer=Conversion.Sellable(tile)?(int)tile["beer"]:0;
                int spare=Math.Max(0,CompetitiveEconomy.OwnBeer(v,seat)-CompetitiveEconomy.CommittedBeer(v,seat));
                bool brewery=beer>spare+merchant;int brewCost=0,brewCount=0;
                if(brewery)
                {
                    int barrels=railBarrels(v);brewCount=(beer-spare-merchant+barrels-1)/barrels;
                    var brews=me["mat"].Single(m=>(string)m["type"]=="Brewery")["tiles"].Take(brewCount).ToArray();
                    if(brews.Length!=brewCount)continue;
                    var supply=(JObject)v.DeepClone();bool feasible=true;
                    // Reserve the proposed industry site before finding supply:
                    // a Canal brewery there would block the industry itself.
                    var investment=supply["slots"].Single(s=>(string)s["id"]==site);
                    investment["tile"]=tile.DeepClone();investment["tile"]["seat"]=seat;
                    foreach(var brew in brews)
                    {
                        int price;string brewSite;
                        if(InvestmentOpportunity.Prospect(supply,brew,out price,out brewSite)<=0){feasible=false;break;}
                        brewCost+=price;var slot=supply["slots"].Single(s=>(string)s["id"]==brewSite);
                        slot["tile"]=brew.DeepClone();slot["tile"]["seat"]=seat;slot["tile"]["resource"]=barrels;
                    }
                    if(!feasible)continue;
                }
                int total=paid+iron+brewCost+connectionCost;
                bool borrow=total>cash;
                int needed=develop+1+(Conversion.Sellable(tile)?1:0)+connections+brewCount+(borrow?1:0);
                // Reserve income payments until the proposed conversion, not just
                // principal for a build. A four-loan source sequence is not blindly
                // affordable in the actual hand/board situation.
                int bridge=Math.Max(0,-income+(borrow?3:0))*Math.Max(0,(needed-1)/2);
                if(total+bridge>cash && !borrow){borrow=true;needed++;bridge=Math.Max(0,3-income)*Math.Max(0,(needed-1)/2);}
                if(actions<needed || (borrow && (income<-7 || total+bridge>cash+30)))continue;
                double score=(prospect-iron*.3-connectionCost*.2-bridge*.25)/(1+develop*.8+connections*.5+brewCount*.6+(borrow?.5:0));
                if(target.type==targets[0].type && p.tactic!="balanced")score*=1.20;
                if(target.type=="CoalMine" || target.type=="IronWorks")score+=CompetitiveEconomy.ResourceProposal(v,tile,Evaluator.Region(site))*.5;
                if(borrow && family=="zero-loan-market")score*=.7;
                if(family=="worcester-recovery" && Evaluator.Region(site)=="Worcester" && income<3)score*=1.15;
                if(score<=quality)continue;
                quality=score;
                best=new IntentState{profile=p.id,seat=seat,revision=(int)v["context"]["revision"],era=(string)v["context"]["era"],
                    focus=brewery?"Brewery":target.type,target=brewery?"":site,goal_level=brewery?0:(int)tile["level"],
                    book_family=!brewery && CompetitiveEconomy.OwnBeer(v,seat)==0 && beer>0 && p.tactic=="cotton"?"no-own-brewery-three-cotton":
                        !brewery && CompetitiveEconomy.OwnBeer(v,seat)==0 && beer>0 && p.tactic=="pottery-flex"?"no-own-brewery-pottery":family,
                    milestone=borrow?"finance_sequence":removed>0?"develop_high_tier":brewery?"secure_own_beer":"build_high_tier",
                    remaining_actions=actions,cash_budget=total+bridge,
                    reason=StrategicOpenings.Label(p.tactic)+": "+family+"; funded tier "+(int)tile["level"]+", buyer access, beer and income runway; reassess after each action"};
            }
            return best;
        }
        static int railBarrels(JObject v){return EraEconomy.Rail(v)?2:1;}
    }
}
