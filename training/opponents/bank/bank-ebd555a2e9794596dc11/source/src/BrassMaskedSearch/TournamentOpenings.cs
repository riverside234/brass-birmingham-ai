using System;
using System.Linq;
using System.Collections.Generic;
using Newtonsoft.Json.Linq;

namespace BrassLegacyAI
{
    // Conditional adaptations of the creator's published descriptions, not
    // unobserved video moves. Attribution and source limits are recorded in
    // ai-research/TOURNAMENT_STYLE_V3.md. All actions still pass the host mask.
    public static class TournamentOpenings
    {
        public const string Version="funded-industry-v5/1.1.0";
        sealed class Target
        {
            public string type,region;public int level,quantity;
            public Target(string t,int l,int n=1,string r=null){type=t;level=l;quantity=n;region=r;}
        }
        static Target[] Targets(JObject v,Profile p,out string family)
        {
            bool rail=EraEconomy.Rail(v);var me=Conversion.Player(v,(int)v["seat"]);
            bool worcester=!rail && v["own_hand"].Any(c=>(string)c=="location:Worcester");
            bool scoring=ScoringEconomy.Established(v,(int)v["seat"]);
            int goods=ScoringEconomy.PreferredLevel(v,"Manufacturer");
            switch(p.tactic)
            {
                // No-own-brewery is a preferred supply route, not a hard ban
                // on an emergency brewery or a claim on opponents' future beer.
                case "cotton":family="external-beer-three-cotton";return new[]{new Target("CottonMill",4,3),new Target("IronWorks",3),new Target("CoalMine",2)};
                case "goods-beer":family=worcester?"worcester-iron-cotton":"external-beer-iron-cotton";return new[]{new Target("CottonMill",4,3,worcester?"Worcester":null),new Target("Manufacturer",goods,2),new Target("IronWorks",3,2)};
                case "rail-tempo":family="coventry-goods";return new[]{new Target("Manufacturer",goods,2,!rail?"Coventry":null),new Target("IronWorks",3,2)};
                case "pottery-flex":family="advanced-pottery-variants";return new[]{new Target("Pottery",rail?5:3,2),new Target("IronWorks",3,2),new Target("CottonMill",4)};
                case "supplier":
                    family=scoring?"funded-scoring-industry":"resource-income-to-scoring";
                    return scoring?new[]{new Target("Manufacturer",goods,2),new Target("CottonMill",4,3),new Target("Pottery",rail?5:3,2),new Target("IronWorks",3,3),new Target("CoalMine",2)}:
                        new[]{new Target("IronWorks",3,3),new Target("Manufacturer",goods,2),new Target("CottonMill",4),new Target("CoalMine",2)};
                default:
                    // The zero-loan guide depends on repeated market receipts.
                    // It is an opportunity mode, never a prohibition on rescue.
                    bool market=EraEconomy.IronPrice(v,1)>=4 || EraEconomy.CoalPrice(v,1)>=4;
                    family=scoring?"funded-scoring-industry":market?"resource-income-to-scoring":worcester?"worcester-iron-cotton":"adaptive-advanced-book";
                    return new[]{new Target("CottonMill",4,3,worcester?"Worcester":null),new Target("Pottery",rail?5:3,2),new Target("Manufacturer",goods,2),new Target("IronWorks",3,2),new Target("CoalMine",2)};
            }
        }
        static bool Present(JToken s){return s["tile"]!=null && s["tile"].Type!=JTokenType.Null;}
        public static IntentState Intent(JObject v,Profile p)
        {
            if(!p.intent_enabled || p.conversion_policy!=1)return null;
            int seat=(int)v["seat"],actions=Conversion.Actions(v,seat);if(actions<3)return null;
            var me=Conversion.Player(v,seat);int income=(int)me["income"];
            string family;var targets=Targets(v,p,out family);IntentState best=null;double quality=0;
            var access=OpeningAccess(v,p,family);if(access!=null)return access;
            foreach(var target in targets)
            {
                var own=v["slots"].Where(Present).Where(s=>(int)s["tile"]["seat"]==seat && (string)s["tile"]["type"]==target.type).ToArray();
                // Completed low-tier stock must not permanently block an
                // upgrade. Unfinished commitments still consume this capacity.
                if(own.Count(s=>!Conversion.Sellable(s["tile"]) || !(bool)s["tile"]["flipped"] || (int)s["tile"]["level"]>=target.level)>=target.quantity)continue;
                if(target.type=="CoalMine" && income>=5 && EraEconomy.CoalPrice(v,1)<4)continue;
                if(target.type=="IronWorks" && EraEconomy.IronPrice(v,1)<=2 && income>=2)continue;
                var tiles=me["mat"].Single(m=>(string)m["type"]==target.type)["tiles"].ToArray();
                // Compare achievable tiers instead of an all-or-nothing distant
                // target. A non-developable pottery must be built first; duplicate
                // levels are not a reason to discard productive tiles.
                var seen=new HashSet<int>();
                for(int removed=0;removed<tiles.Length && removed<=FundedDevelopment.MaximumRemovals;removed++)
                {
                    if(removed>0 && !(bool)tiles[removed-1]["developable"])break;
                    var tile=tiles[removed];int level=(int)tile["level"];
                    if(removed>0 && level>target.level)break;
                    if(!seen.Add(level))continue;
                    var plan=FundedDevelopment.Quote(v,tile,removed,target.region);if(plan==null)continue;
                    int receipts,sold=CompetitiveEconomy.MarketSale(v,tile,Evaluator.Region(plan.site),out receipts);
                    if(family=="resource-income-to-scoring" && (target.type=="IronWorks" || target.type=="CoalMine") && sold==0 && income>=0)continue;
                    double score=plan.Quality;
                    // Higher tiers are a preference only after the complete cash,
                    // iron, beer, card/site and deadline quote has succeeded.
                    score*=ScoringEconomy.TierPriority(v,tile);
                    if(FundedDevelopment.CheapFunded(v,plan))score*=1.20;
                    if(target.type==targets[0].type && p.tactic!="balanced")score*=1.35;
                    if(target.type=="CoalMine" || target.type=="IronWorks")score+=CompetitiveEconomy.ResourceProposal(v,tile,Evaluator.Region(plan.site))*.5;
                    if(target.type=="CoalMine" && score>0)score*=.70;
                    bool scoring=ScoringEconomy.Established(v,seat);
                    if(scoring && Conversion.Sellable(tile) && score>0)score*=1.20;
                    if(plan.borrow && family=="resource-income-to-scoring")score*=.65;
                    if(plan.breweries==0 && Conversion.Sellable(tile) && (int)tile["beer"]>0 && CompetitiveEconomy.OwnBeer(v,seat)==0 &&
                        (p.tactic=="cotton" || p.tactic=="goods-beer" || p.tactic=="balanced"))score*=1.15;
                    if(family=="worcester-iron-cotton" && Evaluator.Region(plan.site)=="Worcester" && income<3)score*=1.15;
                    if(score<=quality)continue;
                    quality=score;
                    bool supplyFirst=plan.breweries>0 && own.Any(s=>!(bool)s["tile"]["flipped"]);
                    best=new IntentState{profile=p.id,seat=seat,revision=(int)v["context"]["revision"],era=(string)v["context"]["era"],
                        focus=supplyFirst?"Brewery":target.type,target=supplyFirst?"":plan.site,goal_level=supplyFirst?0:level,
                        book_family="v5/"+family,
                        milestone=plan.borrow?"finance_sequence":removed>0?"develop_high_tier":supplyFirst?"secure_own_beer":"build_high_tier",
                        remaining_actions=actions,cash_budget=plan.Budget,iron=removed,
                        required_cards=v["own_hand"].Select(c=>(string)c).Where(c=>c.StartsWith("wild:") || c=="location:"+Evaluator.Region(plan.site) || c.Contains(target.type)).ToArray(),
                        reason="v5 funded tier "+level+" (preferred "+target.level+"); removals="+removed+", iron cost="+plan.iron_cost+
                            ", sequence actions="+plan.actions+", budget="+plan.Budget+"; reassess cash, iron, buyer and beer after each action"};
                }
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
    }
}
