using System;
using System.Linq;
using System.Collections.Generic;
using Newtonsoft.Json.Linq;
using BoardGameRules.Entities.NumericalResources;
using BrassHost;
namespace BrassLegacyAI
{
    public sealed class Ledger
    {
        public double banked,industries,links,secured_links,forecast_links,industry_reserve,finance,development,rival,liquidity,sale_tempo,inventory_risk,tempo,score;
        public string era_strategy;
        public double? win_share;
        public int? rank;
        public int final_income,final_cash;
        public string units="heuristic VP equivalent, not a win probability";
    }
    public static class Evaluator
    {
        public static int I(JToken t,string k){return t[k]==null?0:(int)t[k];}
        public static string S(JToken t,string k){return (string)t[k]??"";}
        public static bool B(JToken t,string k){return t[k]!=null && (bool)t[k];}
        public static string Region(string id){return id.Split(':')[0];}
        public static int Rounds(JObject v){return Math.Max(0,Conversion.RoundLimit(v)-I(v["context"],"round"));}
        public static HashSet<string> Reach(JObject v,string start)
        {
            var seen=new HashSet<string>{start};bool changed=true;
            while(changed){changed=false;foreach(var l in v["links"].Where(l=>l["owner"]!=null && l["owner"].Type!=JTokenType.Null))if(l["regions"].Any(r=>seen.Contains((string)r)))foreach(var r in l["regions"])changed=seen.Add((string)r)||changed;}
            return seen;
        }
        public static double Completion(JObject v,JToken slot,int seat,bool conversion=false)
        {
            if(conversion)return Conversion.Completion(v,slot,seat);
            var t=slot["tile"];if(B(t,"flipped"))return 1;
            int rounds=Rounds(v);int acts=rounds*2+(I(v["context"],"actor")==seat?I(v["context"],"actions_remaining"):0);
            string type=S(t,"type");int resource=I(t,"resource");
            if(type=="CoalMine" || type=="IronWorks" || type=="Brewery")
            {
                var connected=Reach(v,Region(S(slot,"id")));
                double supply=v["slots"].Where(s=>s["tile"]!=null && S(s["tile"],"type")==type && (type=="IronWorks" || connected.Contains(Region(S(s,"id"))))).Sum(s=>I(s["tile"],"resource"));
                double demand=(rounds+0.5)*(type=="IronWorks"?2.5:type=="CoalMine"?((string)v["context"]["era"]=="RailEra"?3.5:1.4):2.2);
                if(type=="CoalMine" && connected.Count==1)demand*=0.42;
                double p=resource==0?1:Math.Min(0.93,demand/Math.Max(1,supply)*0.65);
                return Math.Max(0,Math.Min(p,rounds==0?0.12:p));
            }
            var reach=Reach(v,Region(S(slot,"id")));
            var merchants=v["merchants"].Where(m=>m["types"]!=null && m["types"].Any(x=>(string)x==type) && reach.Contains(Region(S(m,"id")))).ToArray();
            int beer=v["slots"].Where(s=>s["tile"]!=null && S(s["tile"],"type")=="Brewery" && (I(s["tile"],"seat")==seat || reach.Contains(Region(S(s,"id"))))).Sum(s=>I(s["tile"],"resource"));
            bool ready=merchants.Any(m=>beer+I(m,"beer")>=I(t,"beer"));
            if(acts==0)return 0;
            if(ready)return 0.97;
            if(acts<2)return 0.06;
            return merchants.Length>0?0.50:0.34;
        }
        public static Ledger Seat(JObject v,int seat,bool conversion=false)
        {return Seat(v,seat,conversion,conversion?Forecast(v):null);}
        static Dictionary<string,double> Forecast(JObject v)
        {
            // One forecast per board tile per evaluation, shared by industry
            // and link scoring for all seats. This cache never outlives a view.
            return Conversion.Forecast(v);
        }
        static Ledger Seat(JObject v,int seat,bool conversion,Dictionary<string,double> forecast)
        {
            var p=Conversion.Player(v,seat);bool canal=(string)v["context"]["era"]=="CanalEra";
            int rounds=Rounds(v),remaining=rounds*2+(I(v["context"],"actor")==seat?I(v["context"],"actions_remaining"):0);
            if(conversion)remaining=Conversion.Actions(v,seat);
            bool fullGame=!Conversion.Intro(v); bool final= (!canal || !fullGame) && remaining==0;
            // Merchant VP is already credited in vp. bonus_vp is the era-summary
            // counter, not an additional pending award.
            var d=new Ledger{banked=I(p,"vp"),era_strategy=EraEconomy.Strategy(v)};
            double pendingIncome=0;
            foreach(var s in v["slots"].Where(s=>s["tile"]!=null && I(s["tile"],"seat")==seat))
            {
                var t=s["tile"];double prob=conversion?forecast[(string)s["id"]]:Completion(v,s,seat);
                double eras=canal && fullGame && I(t,"level")>1?1.80:1;
                d.industries+=conversion?Conversion.IndustryValue(v,s,seat,prob):I(t,"vp")*prob*eras;
                if(!B(t,"flipped"))pendingIncome+=I(t,"income")*prob*0.65;
            }
            foreach(var l in v["links"].Where(l=>l["owner"]!=null && (int)l["owner"]==seat))
            {
                var regions=l["regions"].Select(x=>(string)x).ToArray();
                foreach(var slot in v["slots"].Where(s=>s["tile"]!=null && regions.Contains(Region(S(s,"id")))))
                {
                    double prob=conversion?forecast[(string)slot["id"]]:Completion(v,slot,I(slot["tile"],"seat"));
                    double points=I(slot["tile"],"link_vp")*(conversion?EraEconomy.LinkProbability(v,slot,seat,prob):prob);
                    if(B(slot["tile"],"flipped"))d.secured_links+=points;else d.forecast_links+=points;
                }
                // A border REGION scores two, even with empty/incompatible merchant
                // tiles. Merchant-slot count and beer availability are irrelevant.
                d.secured_links+=2*v["merchants"].Select(m=>Region(S(m,"id"))).Distinct().Count(regions.Contains);
            }
            d.links=d.secured_links+d.forecast_links;
            if(!final)
            {
                // Exactly one resource forecast. No separate loan/cash/income bonuses.
                // Include this round's collection except final Rail; the marker mapping
                // is the declared engine contract, including negative income bands.
                int collections=rounds+(canal && fullGame?Conversion.RoundLimit(v):0);
                double income=IncomeTrack.GetCurrentIncomeLevel(I(p,"income_marker"));
                double expected=IncomeTrack.GetCurrentIncomeLevel(Math.Min(99,I(p,"income_marker")+(int)pendingIncome));
                double discounted=Enumerable.Range(0,collections).Sum(i=>Math.Pow(0.82,i));
                double money=I(p,"cash")+income*discounted+(expected-income)*Math.Max(0,discounted-1)*0.65;
                // Capital far beyond a few affordable investments has little marginal
                // value. Otherwise a short search can hoard loans instead of using them.
                double capacity=Math.Max(5,Math.Min(38,(remaining+(canal && fullGame?4:0))*7.0));
                d.industry_reserve=conversion?EraEconomy.InvestmentReserve(v,seat):0;
                d.finance=conversion?EraEconomy.Finance(v,seat,money,capacity,d.industry_reserve):money<0?money*0.55:capacity*0.32*(1-Math.Exp(-money/capacity));
                // Value development through reachable replacement investments, never a
                // reward per discarded tile. Cost/action savings are counted here once.
                var options=new List<double>();
                foreach(var m in p["mat"])
                {
                    var next=m["tiles"].FirstOrDefault();if(next==null || remaining<2)continue;
                    string type=S(m,"type");
                    bool card=seat==(int)v["seat"] && v["own_hand"].Any(c=>((string)c).StartsWith("wild:") || ((string)c).Contains(type) || ((string)c).StartsWith("location:"));
                    if(!card)continue;
                    bool slot=v["slots"].Any(s=>s["tile"]==null && s["types"].Any(t=>(string)t==type));
                    if(conversion && seat==(int)v["seat"])options.Add(InvestmentOpportunity.MatValue(v,next));
                    else if(slot && I(next,"level")>1)d.development+=Math.Min(3,I(next,"vp")*0.20)*(canal?1.3:0.65);
                }
                // Cards, money and actions cannot finance six simultaneous
                // industry plans. A strongest investment plus a discounted
                // alternative prevents farming unrelated mat-upgrade values.
                if(conversion && seat==(int)v["seat"])d.development=InvestmentOpportunity.Portfolio(options);
            }
            if(conversion && !final){d.inventory_risk=-CompetitiveEconomy.InventoryRisk(v,seat,forecast);d.tempo=CompetitiveEconomy.Tempo(v,seat);}
            d.score=d.banked+d.industries+d.links+d.finance+d.development+d.inventory_risk+d.tempo;return d;
        }
        public static Ledger Evaluate(JObject v,int seat,bool conversion=false)
        {
            if(B(v["context"],"terminal"))
            {
                var ordered=v["players"].OrderByDescending(p=>I(p,"vp")).ThenByDescending(p=>I(p,"income")).ThenByDescending(p=>I(p,"cash")).ToArray();
                var player=Conversion.Player(v,seat);Func<JToken,JToken,bool> tie=(a,b)=>I(a,"vp")==I(b,"vp") && I(a,"income")==I(b,"income") && I(a,"cash")==I(b,"cash");
                double share=tie(player,ordered[0])?1.0/ordered.Count(x=>tie(x,ordered[0])):0;
                return new Ledger{banked=I(player,"vp"),final_income=I(player,"income"),final_cash=I(player,"cash"),score=share,win_share=share,rank=Array.FindIndex(ordered,x=>tie(x,player))+1,units="official terminal win share; VP/income/cash tie break"};
            }
            return EvaluateAll(v,conversion)[seat];
        }
        public static Dictionary<int,Ledger> EvaluateAll(JObject v,bool conversion=false)
        {
            var seats=Conversion.Seats(v);
            if(B(v["context"],"terminal"))return seats.ToDictionary(x=>x,x=>Evaluate(v,x,conversion));
            var forecast=conversion?Forecast(v):null;
            var scores=seats.ToDictionary(x=>x,x=>Seat(v,x,conversion,forecast));
            var raw=scores.ToDictionary(x=>x.Key,x=>x.Value.score);
            double weight=conversion?CompetitiveEconomy.RivalWeight(v):.18;
            foreach(int seat in seats){scores[seat].rival=seats.Where(x=>x!=seat).Max(x=>raw[x])*weight;scores[seat].score-=scores[seat].rival;}
            return scores;
        }
        public static double[] ProposalValues(JObject v,List<Choice> choices,bool conversion)
        {
            var batch=HeuristicBatchQueue.Current;
            return batch==null?ComputeProposals(v,choices,conversion):batch.Evaluate("proposal/"+conversion+"/"+HeuristicBatchQueue.Key(v)+"/"+Wire.Hash(choices),()=>ComputeProposals(v,choices,conversion));
        }
        static double[] ComputeProposals(JObject v,List<Choice> choices,bool conversion)
        {
            var cache=new Dictionary<string,double>();var values=new double[choices.Count];
            var bonuses=conversion && choices.Any(c=>c.kind=="bonus_develop")?InvestmentOpportunity.FreeDevelopments(v):null;
            for(int i=0;i<choices.Count;i++)
            {
                var c=choices[i];bool cardDependent=new[]{"build","network","develop","loan","pass","sell"}.Contains(c.kind);
                string key=c.kind+"/"+c.target+"/"+c.industry+"/"+c.merchant;
                double value;
                if(!cardDependent)values[i]=bonuses!=null && c.kind=="bonus_develop"?bonuses[c.industry]:Proposal(v,c,conversion);
                else
                {
                    if(!cache.TryGetValue(key,out value)){value=Proposal(v,c,conversion)+CardCost(v,c.card)*.15;cache.Add(key,value);}
                    values[i]=value-CardCost(v,c.card)*.15;
                }
            }
            return values;
        }
        public static JObject KnownFinalScore(JObject view)
        {
            // Last seat, last Rail turn, both actions completed. Project only public
            // scoring; do not call EndTurn and reveal a hypothetical replenished hand.
            var v=(JObject)view.DeepClone();
            foreach(var player in v["players"])
            {
                int seat=I(player,"seat"),points=I(player,"vp");
                points+=v["slots"].Where(s=>s["tile"]!=null && I(s["tile"],"seat")==seat && B(s["tile"],"flipped")).Sum(s=>I(s["tile"],"vp"));
                foreach(var link in v["links"].Where(l=>l["owner"]!=null && (int)l["owner"]==seat))
                {
                    var regions=link["regions"].Select(x=>(string)x).ToArray();
                    points+=v["slots"].Where(s=>s["tile"]!=null && B(s["tile"],"flipped") && regions.Contains(Region(S(s,"id")))).Sum(s=>I(s["tile"],"link_vp"));
                    points+=2*v["merchants"].Select(m=>Region(S(m,"id"))).Distinct().Count(regions.Contains);
                }
                player["vp"]=points;player["bonus_vp"]=0;
            }
            v["context"]["terminal"]=true;return v;
        }
        public static double CardCost(JObject v,string card)
        {
            if(string.IsNullOrEmpty(card))return 0;
            if(card.StartsWith("wild:"))return 5;
            if(card.StartsWith("industry:"))return 1.2;
            var region=card.Substring(9);return v["slots"].Count(s=>Region(S(s,"id"))==region && s["tile"]==null)*0.7;
        }
        public static double Proposal(JObject v,BrassHost.Choice c,bool conversion=false)
        {
            int seat=(int)v["seat"];var p=Conversion.Player(v,seat);double score=-CardCost(v,c.card)*0.15;
            if(c.kind=="finish")return 0;
            if(c.kind=="sell")
            {
                if(conversion)return score+SaleTiming.Proposal(v,c);
                var tile=v["slots"].Single(s=>S(s,"id")==c.target)["tile"];
                var merchant=v["merchants"].Single(m=>S(m,"id")==c.merchant);
                return 30+I(tile,"vp")+(conversion && I(tile,"beer")>0 && I(merchant,"beer")>0?Conversion.MerchantBonus(v,merchant):0);
            }
            if(c.kind=="build")
            {
                var t=p["mat"].Single(m=>S(m,"type")==c.industry)["tiles"].First();
                int paid=conversion && EraEconomy.Rail(v)?EraEconomy.BuildCost(v,t,Region(c.target)):I(t,"cost");
                score+=I(t,"vp")+I(t,"income")*(conversion?ScoringEconomy.IncomeWeight(v,seat):.4)-paid*0.25;
                if(conversion && Conversion.Actions(v,seat)<=1 && Conversion.Sellable(t))
                    score-=I(t,"vp")*2+10; // A build cannot also be a sale action.
                if(c.industry=="IronWorks")score+=Math.Max(0,8-I(v,"iron_market"))*0.5;
                if(conversion)score+=CompetitiveEconomy.ResourceProposal(v,t,Region(c.target))+CompetitiveEconomy.SellableProposal(v,t,Region(c.target));
                if(conversion)
                {
                    // The rules mask has already authorized this build. Keep
                    // a legal opponent resource overbuild visible to a short
                    // clock: its denial value is otherwise lost in a location tie.
                    var old=v["slots"].Single(x=>S(x,"id")==c.target)["tile"];
                    if(old!=null && old.Type!=JTokenType.Null && I(old,"seat")!=seat)
                        score+=CompetitiveEconomy.RivalWeight(v)*I(old,"vp")*(B(old,"flipped")?1:.5);
                }
                if(c.industry=="Brewery")
                {
                    if(!conversion)score+=2;
                    else
                    {
                        // Proposal order must agree with afterstate scoring:
                        // unsupported beer should not monopolize short searches.
                        var imagined=(JObject)v["slots"].Single(s=>S(s,"id")==c.target).DeepClone();
                        imagined["tile"]=t.DeepClone();imagined["tile"]["resource"]=(string)v["context"]["era"]=="CanalEra"?1:2;
                        score-=I(t,"vp")*(1-Conversion.Completion(v,imagined,seat));
                        score+=CompetitiveEconomy.BreweryPrivacy(v,Region(c.target));
                    }
                }
                if(c.industry=="Pottery" || c.industry=="CottonMill" || c.industry=="Manufacturer")
                {
                    var reach=Reach(v,Region(c.target));
                    bool sale=v["merchants"].Any(m=>m["types"]!=null && m["types"].Any(x=>(string)x==c.industry) && reach.Contains(Region(S(m,"id"))));
                    if(!sale)
                    {
                        int distance=conversion?Conversion.MerchantDistance(v,Region(c.target),c.industry):99;
                        score-=distance==1?1:distance==2?3:I(t,"vp")*0.55+2;
                    }
                }
                return score;
            }
            if(c.kind=="network")
            {
                var l=v["links"].Single(x=>S(x,"id")==c.target);var regions=l["regions"].Select(x=>(string)x).ToArray();
                if(conversion)
                    foreach(var slot in v["slots"].Where(s=>s["tile"]!=null && I(s["tile"],"seat")==seat && !B(s["tile"],"flipped") && Conversion.Sellable(s["tile"])))
                    {
                        string region=Region(S(slot,"id")),type=S(slot["tile"],"type");
                        int before=Conversion.MerchantDistance(v,region,type),after=Conversion.MerchantDistance(v,region,type,c.target);
                        if(after<before)score+=(after==0?6:2)+I(slot["tile"],"income")*ScoringEconomy.IncomeWeight(v,seat);
                    }
                if(conversion)score+=CompetitiveEconomy.NetworkAccess(v,c);
                if(conversion && EraEconomy.Rail(v))return score+EraEconomy.NetworkProposal(v,c)+1;
                return score+v["slots"].Where(s=>s["tile"]!=null && regions.Contains(Region(S(s,"id")))).Sum(s=>I(s["tile"],"link_vp")*(B(s["tile"],"flipped")?1:0.55))+1;
            }
            if(c.kind=="loan")return score+(conversion?InvestmentOpportunity.Loan(v):I(p,"cash")<15?9:-4);
            if(conversion && (c.kind=="bonus_develop" || c.kind=="bonus_skip"))return Conversion.BonusDevelopment(v,c);
            if(c.kind=="develop" || c.kind=="bonus_develop")return score+(conversion?InvestmentOpportunity.Development(v,c.industry):c.industry=="Brewery" || c.industry=="CottonMill"?3:1);
            if(c.kind=="scout")return -5-c.cards.Sum(x=>CardCost(v,x))*0.15;
            if(c.kind=="pass")return (conversion?-1:-100)+score;
            if(c.kind=="bonus_skip")return -10;
            if(c.kind=="liquidate"){var t=v["slots"].Single(s=>S(s,"id")==c.target)["tile"];return I(t,"cost")*0.3-I(t,"vp")*(B(t,"flipped")?2:1);}
            if(c.target=="market")return -4;
            if(c.target!=null && c.target.StartsWith("merchant:"))return conversion?Conversion.BeerPriority(v,c,0):5;
            if(c.target!=null){var t=v["slots"].Single(s=>S(s,"id")==c.target)["tile"];return (I(t,"seat")==seat?3:-1)+(I(t,"resource")==1?(I(t,"seat")==seat?5:-4):0);}
            return score;
        }
    }
}
