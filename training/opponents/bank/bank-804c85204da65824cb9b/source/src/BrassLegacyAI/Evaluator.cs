using System;
using System.Linq;
using System.Collections.Generic;
using Newtonsoft.Json.Linq;
using BoardGameRules.Entities.NumericalResources;
namespace BrassLegacyAI
{
    public sealed class Ledger
    {
        public double banked,industries,links,finance,development,rival,score;
        public double? win_share;
        public int? rank;
        public string units="heuristic VP equivalent, not a win probability";
    }
    public static class Evaluator
    {
        public static int I(JToken t,string k){return t[k]==null?0:(int)t[k];}
        public static string S(JToken t,string k){return (string)t[k]??"";}
        public static bool B(JToken t,string k){return t[k]!=null && (bool)t[k];}
        public static string Region(string id){return id.Split(':')[0];}
        public static int Rounds(JObject v){return Math.Max(0,8-I(v["context"],"round"));}
        public static HashSet<string> Reach(JObject v,string start)
        {
            var seen=new HashSet<string>{start};bool changed=true;
            while(changed){changed=false;foreach(var l in v["links"].Where(l=>l["owner"]!=null && l["owner"].Type!=JTokenType.Null))if(l["regions"].Any(r=>seen.Contains((string)r)))foreach(var r in l["regions"])changed=seen.Add((string)r)||changed;}
            return seen;
        }
        public static double Completion(JObject v,JToken slot,int seat)
        {
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
        public static Ledger Seat(JObject v,int seat)
        {
            var p=v["players"][seat];bool canal=(string)v["context"]["era"]=="CanalEra";
            int rounds=Rounds(v),remaining=rounds*2+(I(v["context"],"actor")==seat?I(v["context"],"actions_remaining"):0);
            bool final= !canal && remaining==0;
            // Merchant VP is already credited in vp. bonus_vp is the era-summary
            // counter, not an additional pending award.
            var d=new Ledger{banked=I(p,"vp")};
            double pendingIncome=0;
            foreach(var s in v["slots"].Where(s=>s["tile"]!=null && I(s["tile"],"seat")==seat))
            {
                var t=s["tile"];double prob=Completion(v,s,seat);
                double eras=canal && I(t,"level")>1?1.80:1;
                d.industries+=I(t,"vp")*prob*eras;
                if(!B(t,"flipped"))pendingIncome+=I(t,"income")*prob*0.65;
            }
            foreach(var l in v["links"].Where(l=>l["owner"]!=null && (int)l["owner"]==seat))
            {
                var regions=l["regions"].Select(x=>(string)x).ToArray();
                d.links+=v["slots"].Where(s=>s["tile"]!=null && regions.Contains(Region(S(s,"id")))).Sum(s=>I(s["tile"],"link_vp")*Completion(v,s,I(s["tile"],"seat")));
                // A border REGION scores two, even with empty/incompatible merchant
                // tiles. Merchant-slot count and beer availability are irrelevant.
                d.links+=2*v["merchants"].Select(m=>Region(S(m,"id"))).Distinct().Count(regions.Contains);
            }
            if(!final)
            {
                // Exactly one resource forecast. No separate loan/cash/income bonuses.
                // Include this round's collection except final Rail; the marker mapping
                // is the declared engine contract, including negative income bands.
                int collections=rounds+(canal?8:0);
                double income=IncomeTrack.GetCurrentIncomeLevel(I(p,"income_marker"));
                double expected=IncomeTrack.GetCurrentIncomeLevel(Math.Min(99,I(p,"income_marker")+(int)pendingIncome));
                double discounted=Enumerable.Range(0,collections).Sum(i=>Math.Pow(0.82,i));
                double money=I(p,"cash")+income*discounted+(expected-income)*Math.Max(0,discounted-1)*0.65;
                // Capital far beyond a few affordable investments has little marginal
                // value. Otherwise a short search can hoard loans instead of using them.
                double capacity=Math.Max(5,Math.Min(38,(remaining+(canal?4:0))*7.0));
                d.finance=money<0?money*0.55:capacity*0.32*(1-Math.Exp(-money/capacity));
                // Value development through reachable replacement investments, never a
                // reward per discarded tile. Cost/action savings are counted here once.
                foreach(var m in p["mat"])
                {
                    var next=m["tiles"].FirstOrDefault();if(next==null || remaining<2)continue;
                    string type=S(m,"type");
                    bool card=seat==(int)v["seat"] && v["own_hand"].Any(c=>((string)c).StartsWith("wild:") || ((string)c).Contains(type) || ((string)c).StartsWith("location:"));
                    if(!card)continue;
                    bool slot=v["slots"].Any(s=>s["tile"]==null && s["types"].Any(t=>(string)t==type));
                    if(slot && I(next,"level")>1)d.development+=Math.Min(3,I(next,"vp")*0.20)*(canal?1.3:0.65);
                }
            }
            d.score=d.banked+d.industries+d.links+d.finance+d.development;return d;
        }
        public static Ledger Evaluate(JObject v,int seat)
        {
            if(B(v["context"],"terminal"))
            {
                var ordered=v["players"].OrderByDescending(p=>I(p,"vp")).ThenByDescending(p=>I(p,"income")).ThenByDescending(p=>I(p,"cash")).ToArray();
                var player=v["players"][seat];Func<JToken,JToken,bool> tie=(a,b)=>I(a,"vp")==I(b,"vp") && I(a,"income")==I(b,"income") && I(a,"cash")==I(b,"cash");
                double share=tie(player,ordered[0])?1.0/ordered.Count(x=>tie(x,ordered[0])):0;
                return new Ledger{banked=I(player,"vp"),score=share,win_share=share,rank=Array.FindIndex(ordered,x=>tie(x,player))+1,units="official terminal win share; VP/income/cash tie break"};
            }
            var d=Seat(v,seat);d.rival=Enumerable.Range(0,4).Where(s=>s!=seat).Max(s=>Seat(v,s).score)*0.18;d.score-=d.rival;return d;
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
        public static double Proposal(JObject v,BrassHost.Choice c)
        {
            int seat=(int)v["seat"];var p=v["players"][seat];double score=-CardCost(v,c.card)*0.15;
            if(c.kind=="finish")return 0;
            if(c.kind=="sell")return 30+I(v["slots"].Single(s=>S(s,"id")==c.target)["tile"],"vp");
            if(c.kind=="build")
            {
                var t=p["mat"].Single(m=>S(m,"type")==c.industry)["tiles"].First();
                score+=I(t,"vp")+I(t,"income")*0.4-I(t,"cost")*0.25;
                if(c.industry=="IronWorks")score+=Math.Max(0,8-I(v,"iron_market"))*0.5;
                if(c.industry=="Brewery")score+=2;
                if(c.industry=="Pottery" || c.industry=="CottonMill" || c.industry=="Manufacturer")
                {
                    var reach=Reach(v,Region(c.target));
                    bool sale=v["merchants"].Any(m=>m["types"]!=null && m["types"].Any(x=>(string)x==c.industry) && reach.Contains(Region(S(m,"id"))));
                    if(!sale)score-=I(t,"vp")*0.55+2;
                }
                return score;
            }
            if(c.kind=="network")
            {
                var l=v["links"].Single(x=>S(x,"id")==c.target);var regions=l["regions"].Select(x=>(string)x).ToArray();
                return score+v["slots"].Where(s=>s["tile"]!=null && regions.Contains(Region(S(s,"id")))).Sum(s=>I(s["tile"],"link_vp")*(B(s["tile"],"flipped")?1:0.55))+1;
            }
            if(c.kind=="loan")return score+(I(p,"cash")<15?9:-4);
            if(c.kind=="develop" || c.kind=="bonus_develop")return score+(c.industry=="Brewery" || c.industry=="CottonMill"?3:1);
            if(c.kind=="scout")return -5-c.cards.Sum(x=>CardCost(v,x))*0.15;
            if(c.kind=="pass")return -100+score;
            if(c.kind=="bonus_skip")return -10;
            if(c.kind=="liquidate"){var t=v["slots"].Single(s=>S(s,"id")==c.target)["tile"];return I(t,"cost")*0.3-I(t,"vp")*(B(t,"flipped")?2:1);}
            if(c.target=="market")return -4;
            if(c.target!=null && c.target.StartsWith("merchant:"))return 5;
            if(c.target!=null){var t=v["slots"].Single(s=>S(s,"id")==c.target)["tile"];return (I(t,"seat")==seat?3:-1)+(I(t,"resource")==1?(I(t,"seat")==seat?5:-4):0);}
            return score;
        }
    }
}
