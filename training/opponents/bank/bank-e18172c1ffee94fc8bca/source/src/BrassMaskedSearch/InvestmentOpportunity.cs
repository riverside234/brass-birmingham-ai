using System;
using System.Collections.Generic;
using System.Linq;
using Newtonsoft.Json.Linq;

namespace BrassLegacyAI
{
    // Prospects rank proposals; hypothetical complete plans still pay through
    // the rules host and earn value only from their resulting public state.
    public static class InvestmentOpportunity
    {
        static bool Present(JToken s){return s["tile"]!=null && s["tile"].Type!=JTokenType.Null;}
        static HashSet<string> Network(JObject v,int seat)
        {
            var network=new HashSet<string>(v["slots"].Where(Present).Where(s=>(int)s["tile"]["seat"]==seat).Select(s=>Evaluator.Region((string)s["id"])));
            foreach(var l in v["links"].Where(l=>(int?)l["owner"]==seat))foreach(var r in l["regions"])network.Add((string)r);
            return network;
        }
        static bool Card(JObject v,JToken slot,string type,HashSet<string> network)
        {
            string region=Evaluator.Region((string)slot["id"]);
            return v["own_hand"].Any(c=>{string card=(string)c;return card=="wild:location" || card=="location:"+region ||
                ((card=="wild:industry" || (card.StartsWith("industry:") && card.Contains(type))) && (network.Count==0 || network.Contains(region)));});
        }
        public static double Prospect(JObject v,JToken tile,out int cost)
        {
            cost=100000;int seat=(int)v["seat"],actions=Conversion.Actions(v,seat);
            if(tile==null || (EraEconomy.Rail(v) && (bool)tile["canal_only"]) || (!EraEconomy.Rail(v) && (bool)tile["rail_only"]) || actions<1)return 0;
            string type=(string)tile["type"];double best=0;var network=Network(v,seat);
            foreach(var slot in v["slots"].Where(s=>!Present(s) && s["types"].Any(t=>(string)t==type) && Card(v,s,type,network)))
            {
                int paid=EraEconomy.BuildCost(v,tile,Evaluator.Region((string)slot["id"]));if(paid>=100000)continue;
                var imagined=(JObject)slot.DeepClone();imagined["tile"]=tile.DeepClone();
                if(type=="Brewery")imagined["tile"]["resource"]=EraEconomy.Rail(v)?2:1;
                double probability=Conversion.Completion(v,imagined,seat);
                if(Conversion.Sellable(tile) && actions<2)probability=0;
                double gain=Conversion.IndustryValue(v,imagined,seat,probability)+(int)tile["income"]*probability*.4-paid*.16;
                if(gain>best){best=gain;cost=paid;}
            }
            return best;
        }
        public static double Development(JObject v,string type)
        {
            int seat=(int)v["seat"],actions=Conversion.Actions(v,seat);
            var tiles=Conversion.Player(v,seat)["mat"].Single(m=>(string)m["type"]==type)["tiles"].ToArray();
            if(tiles.Length<2 || actions<2)return -6;
            int costBefore;double before=Prospect(v,tiles[0],out costBefore),best=-5;
            int cash=(int)Conversion.Player(v,seat)["cash"],maximum=(string)v["pending"]["kind"]=="develop"?1:2;
            int freeIron=v["slots"].Where(Present).Where(s=>(string)s["tile"]["type"]=="IronWorks").Sum(s=>(int)s["tile"]["resource"]);
            for(int removed=1;removed<=maximum && removed<tiles.Length;removed++)
            {
                if(!tiles.Take(removed).All(t=>(bool)t["developable"]))continue;
                int costAfter;double after=Prospect(v,tiles[removed],out costAfter);
                bool canFund=costAfter<=cash || ((int)Conversion.Player(v,seat)["income"]>=-7 && costAfter<=cash+30 && actions>=3);
                if(after<=0 || !canFund || (Conversion.Sellable(tiles[removed]) && actions<3))continue;
                bool expiring=EraEconomy.Rail(v) && (bool)tiles[0]["canal_only"];
                if(!expiring && (int)tiles[removed]["level"]<=(int)tiles[0]["level"])continue;
                double inventory=0;for(int i=1;i<removed;i++){int ignored;inventory+=Prospect(v,tiles[i],out ignored)*.35;}
                double value=Math.Min(7,(after-before-inventory)*.55+(expiring?2:0))-EraEconomy.IronPrice(v,Math.Max(0,removed-freeIron))*.25-.5;
                best=Math.Max(best,value);
            }
            return best;
        }
        public static double Loan(JObject v)
        {
            var p=Conversion.Player(v,(int)v["seat"]);int cash=(int)p["cash"],actions=Conversion.Actions(v,(int)v["seat"]);
            if(CashManagement.Project(v,(int)v["seat"]).shortfall>0)return 12;
            if(actions<2)return -4;
            double best=0;
            foreach(var mat in p["mat"])
            {
                var tile=mat["tiles"].FirstOrDefault();int cost;double prospect=Prospect(v,tile,out cost);
                if(cost>cash && cost<=cash+30 && prospect>0 && (!Conversion.Sellable(tile) || actions>=3))best=Math.Max(best,prospect);
            }
            // Borrow for an identified investment, not a fixed cash threshold.
            return best>0?Math.Min(9,2+best*.5):(cash<8?1:-4);
        }
        public static double MatValue(JObject v,JToken tile)
        {
            int cost;double prospect=Prospect(v,tile,out cost);
            if(prospect<=0 || cost> (int)Conversion.Player(v,(int)v["seat"])["cash"]+30)return 0;
            return Math.Min(2,prospect*.12);
        }
    }
}
