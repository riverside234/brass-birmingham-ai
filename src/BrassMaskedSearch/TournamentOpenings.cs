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
        public const string Version="bilibili-advanced-openings/2.0.0";
        sealed class Target
        {
            public string type,region;public int level,quantity;
            public Target(string t,int l,int n=1,string r=null){type=t;level=l;quantity=n;region=r;}
        }
        static Target[] Targets(JObject v,Profile p,out string family)
        {
            bool rail=EraEconomy.Rail(v);var me=Conversion.Player(v,(int)v["seat"]);
            bool worcester=!rail && v["own_hand"].Any(c=>(string)c=="location:Worcester");
            bool mature=(int)me["income"]>=5 && Conversion.Actions(v,(int)v["seat"])>=7;
            switch(p.tactic)
            {
                // No-own-brewery is a preferred supply route, not a hard ban
                // on an emergency brewery or a claim on opponents' future beer.
                case "cotton":family="external-beer-three-cotton";return new[]{new Target("CottonMill",rail?4:3,3),new Target("IronWorks",2),new Target("CoalMine",2)};
                case "goods-beer":family=worcester?"worcester-iron-cotton":"external-beer-iron-cotton";return new[]{new Target("CottonMill",rail?4:2,3,worcester?"Worcester":null),new Target("IronWorks",rail?3:2,2)};
                case "rail-tempo":family="coventry-goods";return new[]{new Target("Manufacturer",rail || mature?6:2,2,!rail?"Coventry":null),new Target("IronWorks",rail?3:2,2)};
                case "pottery-flex":family="advanced-pottery-variants";return new[]{new Target("Pottery",rail?5:3,2),new Target("IronWorks",rail?3:2,2),new Target("CottonMill",rail?4:2)};
                case "supplier":family="zero-loan-industry";return new[]{new Target("IronWorks",rail?3:2,3),new Target("CoalMine",2,2),new Target("Manufacturer",rail || mature?6:2,2),new Target("CottonMill",rail?4:3)};
                default:
                    // The zero-loan guide depends on repeated market receipts.
                    // It is an opportunity mode, never a prohibition on rescue.
                    bool market=EraEconomy.IronPrice(v,1)>=4 || EraEconomy.CoalPrice(v,1)>=4;
                    family=market?"zero-loan-industry":worcester?"worcester-iron-cotton":"adaptive-advanced-book";
                    return new[]{new Target("CottonMill",rail?4:3,3,worcester?"Worcester":null),new Target("Pottery",rail?5:3,2),new Target("IronWorks",rail?3:2,2),new Target("CoalMine",2,2),new Target("Manufacturer",rail || mature?6:2,2)};
            }
        }
        static bool Present(JToken s){return s["tile"]!=null && s["tile"].Type!=JTokenType.Null;}
        public static IntentState Intent(JObject v,Profile p)
        {
            if(!p.intent_enabled || p.conversion_policy!=1)return null;
            int seat=(int)v["seat"],actions=Conversion.Actions(v,seat);if(actions<3)return null;
            var me=Conversion.Player(v,seat);int cash=(int)me["cash"],income=(int)me["income"];
            string family;var targets=Targets(v,p,out family);IntentState best=null;double quality=0;
            var access=OpeningAccess(v,p,family);if(access!=null)return access;
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
                double prospect=InvestmentOpportunity.Prospect(v,tile,out paid,out site,target.region);if(prospect<=0)continue;
                int receipts,sold=CompetitiveEconomy.MarketSale(v,tile,Evaluator.Region(site),out receipts);
                if(family=="zero-loan-industry" && (target.type=="IronWorks" || target.type=="CoalMine") &&
                    sold==0 && income>=0)continue; // The source's cash cycle requires a real market buyer.
                int develop=(removed+1)/2,iron=EraEconomy.IronPrice(v,Math.Max(0,removed-boardIron));
                int connections=Conversion.Sellable(tile)?Conversion.MerchantDistance(v,Evaluator.Region(site),target.type):0;
                if(connections>2)continue;
                int connectionCost=EraEconomy.ConnectionBudget(v,connections);
                int beer=Conversion.Sellable(tile)?(int)tile["beer"]:0;
                int supplyBeer=CompetitiveEconomy.AvailableBeer(v,Evaluator.Region(site),target.type,develop+connections+1);
                bool brewery=beer>supplyBeer;int brewCost=0,brewCount=0;
                if(brewery)
                {
                    int barrels=railBarrels(v);brewCount=(beer-supplyBeer+barrels-1)/barrels;
                    var breweryTiles=me["mat"].Single(m=>(string)m["type"]=="Brewery")["tiles"].ToArray();
                    int brewRemoved=EraEconomy.Rail(v)?breweryTiles.TakeWhile(t=>(bool)t["canal_only"]).Count():0;
                    if(brewRemoved>0)
                    {
                        if(!breweryTiles.Take(brewRemoved).All(t=>(bool)t["developable"]))continue;
                        develop+=(brewRemoved+1)/2;iron=EraEconomy.IronPrice(v,Math.Max(0,removed+brewRemoved-boardIron));
                    }
                    var brews=breweryTiles.Skip(brewRemoved).Take(brewCount).ToArray();
                    if(brews.Length!=brewCount)continue;
                    var supply=(JObject)v.DeepClone();bool feasible=true;
                    // Reserve the proposed industry site before finding supply:
                    // a Canal brewery there would block the industry itself.
                    var investment=supply["slots"].Single(s=>(string)s["id"]==site);
                    investment["tile"]=tile.DeepClone();investment["tile"]["seat"]=seat;
                    foreach(var brew in brews)
                    {
                        int price;string brewSite;
                        if(!InvestmentOpportunity.FundingSite(supply,brew,out price,out brewSite)){feasible=false;break;}
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
                if(target.type==targets[0].type && p.tactic!="balanced")score*=1.35;
                if(target.type=="CoalMine" || target.type=="IronWorks")score+=CompetitiveEconomy.ResourceProposal(v,tile,Evaluator.Region(site))*.5;
                if(borrow && family=="zero-loan-industry")score*=.65;
                if(!brewery && beer>0 && CompetitiveEconomy.OwnBeer(v,seat)==0 &&
                    (p.tactic=="cotton" || p.tactic=="goods-beer" || p.tactic=="balanced"))score*=1.15;
                if(family=="worcester-iron-cotton" && Evaluator.Region(site)=="Worcester" && income<3)score*=1.15;
                if(score<=quality)continue;
                quality=score;
                bool supplyFirst=brewery && own.Any(s=>!(bool)s["tile"]["flipped"]);
                // The link/pottery source builds the first pottery before its
                // supporting breweries. Do not demand brewery customers first
                // and thereby reject the opening that creates those customers.
                best=new IntentState{profile=p.id,seat=seat,revision=(int)v["context"]["revision"],era=(string)v["context"]["era"],
                    focus=supplyFirst?"Brewery":target.type,target=supplyFirst?"":site,goal_level=supplyFirst?0:(int)tile["level"],
                    book_family=family,
                    milestone=borrow?"finance_sequence":removed>0?"develop_high_tier":supplyFirst?"secure_own_beer":"build_high_tier",
                    remaining_actions=actions,cash_budget=total+bridge,
                    required_cards=v["own_hand"].Select(c=>(string)c).Where(c=>c.StartsWith("wild:") || c=="location:"+Evaluator.Region(site) || c.Contains(target.type)).ToArray(),
                    reason=StrategicOpenings.Label(p.tactic)+": "+family+"; funded tier "+(int)tile["level"]+", buyer access, beer and income runway; reassess after each action"};
            }
            return best;
        }
        static IntentState OpeningAccess(JObject v,Profile p,string family)
        {
            int seat=(int)v["seat"],round=(int)v["context"]["round"];
            if(EraEconomy.Rail(v) || round!=1 || p.tactic!="rail-tempo" ||
                Array.IndexOf(v["order"].Select(x=>(int)x).ToArray(),seat)!=v["players"].Count()-1 ||
                v["slots"].Where(Present).Any(s=>(int)s["tile"]["seat"]==seat) || v["links"].Any(l=>(int?)l["owner"]==seat))return null;
            var cards=v["own_hand"].Select(c=>(string)c).Where(c=>c=="location:Coventry" || c=="wild:location" || c=="wild:industry" || c.Contains("Manufacturer")).ToArray();
            if(cards.Length==0 || v["own_hand"].Count()<2)return null;
            var link=v["links"].FirstOrDefault(l=>(string)l["era"]=="Canal" && (l["owner"]==null || l["owner"].Type==JTokenType.Null) && l["regions"].Any(r=>(string)r=="Coventry"));
            if(link==null)return null;
            var after=(JObject)v.DeepClone();after["links"].Single(l=>(string)l["id"]==(string)link["id"])["owner"]=seat;
            var tile=Conversion.Player(after,seat)["mat"].Single(m=>(string)m["type"]=="Manufacturer")["tiles"].FirstOrDefault();
            int paid;string site;double value=InvestmentOpportunity.Prospect(after,tile,out paid,out site,"Coventry");
            if(value<=0 || Evaluator.Region(site)!="Coventry" || paid+3>(int)Conversion.Player(v,seat)["cash"] ||
                Conversion.MerchantDistance(after,"Coventry","Manufacturer")>1)return null;
            return new IntentState{profile=p.id,seat=seat,revision=(int)v["context"]["revision"],era="CanalEra",focus="network",
                target=(string)link["id"],book_family=family,milestone="funded_opening_access",cash_budget=paid+3,
                remaining_actions=Conversion.Actions(v,seat),required_cards=cards,
                reason="Guide 16: fourth-seat Coventry link opening, only with a supported funded factory and buyer route; the legal mask remains authoritative"};
        }
        static int railBarrels(JObject v){return EraEconomy.Rail(v)?2:1;}
    }
}
