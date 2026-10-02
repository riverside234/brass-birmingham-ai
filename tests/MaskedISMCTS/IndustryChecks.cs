using System;
using System.IO;
using System.Linq;
using System.Collections.Generic;
using Newtonsoft.Json.Linq;
using BrassHost;
using BrassLegacyAI;

public static class IndustryChecks
{
    static int checks;
    static void Check(bool ok,string label){checks++;if(!ok)throw new Exception(label);}
    public static int Main(string[] args)
    {
        try
        {
            var host=new Host(843100,null);
            var view=(JObject)Wire.Token(host.Observe(host.Actor));int seat=(int)view["seat"];
            view["context"]["round"]=2;view["context"]["actions_remaining"]=2;
            var player=Conversion.Player(view,seat);player["cash"]=70;player["income"]=4;player["income_marker"]=20;
            view["own_hand"]=new JArray("wild:location","wild:industry","industry:CottonMill");
            foreach(var link in view["links"])link["owner"]=seat;
            foreach(var merchant in view["merchants"]){merchant["types"]=new JArray("CottonMill","Pottery","Manufacturer");merchant["beer"]=0;}
            var cotton=player["mat"].Single(m=>(string)m["type"]=="CottonMill")["tiles"];
            // A common opening has already used two level-one Cotton tiles.
            cotton[0].Remove();cotton[0].Remove();
            var brewery=player["mat"].Single(m=>(string)m["type"]=="Brewery")["tiles"][0];
            foreach(var site in view["slots"].Where(s=>s["types"].Any(t=>(string)t=="Brewery")).Take(2))
            {site["tile"]=brewery.DeepClone();site["tile"]["seat"]=seat;site["tile"]["resource"]=1;site["tile"]["flipped"]=false;}
            view["iron_market"]=8;
            double cheap=InvestmentOpportunity.Development(view,"CottonMill");
            var expensive=(JObject)view.DeepClone();expensive["iron_market"]=0;
            Check(cheap>0,"cheap Canal upgrade is worth considering");
            Check(cheap>InvestmentOpportunity.Development(expensive,"CottonMill"),"higher real iron cost reduces upgrade value");
            var occupied=(JObject)view.DeepClone();
            foreach(var region in occupied["slots"].Where(s=>s["types"].Any(t=>(string)t=="CottonMill")).Select(s=>Evaluator.Region((string)s["id"])).Distinct())
            {
                var site=occupied["slots"].First(s=>Evaluator.Region((string)s["id"])==region);
                site["tile"]=Conversion.Player(occupied,seat)["mat"].Single(m=>(string)m["type"]=="CottonMill")["tiles"][0].DeepClone();site["tile"]["seat"]=seat;
            }
            int ignoredCost;
            Check(InvestmentOpportunity.Prospect(occupied,Conversion.Player(occupied,seat)["mat"].Single(m=>(string)m["type"]=="CottonMill")["tiles"][0],out ignoredCost)==0,"vacant spaces in occupied own Canal cities are not build prospects");
            var mat=player["mat"].Single(m=>(string)m["type"]=="CottonMill");
            Check(InvestmentOpportunity.MatValue(view,mat["tiles"].First(t=>(int)t["level"]==2))>InvestmentOpportunity.MatValue(view,mat["tiles"][0]),"useful Canal level two is distinguishable from level one");
            var rail=(JObject)view.DeepClone();rail["context"]["era"]="RailEra";
            var railTiles=Conversion.Player(rail,seat)["mat"].Single(m=>(string)m["type"]=="CottonMill")["tiles"];
            while((int)railTiles[0]["level"]<2)railTiles[0].Remove();
            Check(InvestmentOpportunity.Development(rail,"CottonMill")>0,"cheap Rail level-three upgrade is worth considering");
            var late=(JObject)view.DeepClone();late["context"]["round"]=8;late["context"]["actions_remaining"]=1;
            Check(InvestmentOpportunity.Development(late,"CottonMill")<0,"never develop beyond a sale deadline");
            var poor=(JObject)view.DeepClone();Conversion.Player(poor,seat)["cash"]=0;Conversion.Player(poor,seat)["income"]=-10;
            Check(InvestmentOpportunity.Development(poor,"CottonMill")<0,"cheap iron cannot bypass funding and credit");
            var slots=view["slots"].Where(s=>s["types"].Any(t=>(string)t=="CottonMill")).Take(2).ToArray();
            string target=(string)slots[0]["id"],merchantId=(string)view["merchants"][0]["id"];
            var tile=(JObject)mat["tiles"][0].DeepClone();tile["seat"]=seat;tile["flipped"]=false;
            slots[0]["tile"]=tile;
            Check(SaleTiming.WaitForBatch(view,target,merchantId),"safe lone sale waits for a funded companion");
            var single=new List<string[]>{new[]{new Choice("sell",target,card:"industry:CottonMill",merchant:merchantId).id,new Choice("finish").id}};
            Check(SaleTiming.SinglePenalty(view,single)>0,"single sale pays one bounded action-opportunity cost");
            var group=new List<string[]>{new[]{single[0][0],new Choice("sell",(string)slots[1]["id"],merchant:merchantId).id,new Choice("finish").id}};
            Check(SaleTiming.SinglePenalty(view,group)==0,"grouped sale pays no single-sale penalty");
            var deadline=(JObject)view.DeepClone();deadline["context"]["round"]=7;
            Check(!SaleTiming.WaitForBatch(deadline,target,merchantId),"last two turns convert before the deadline");
            slots[0]["tile"]["beer"]=1;
            foreach(var s in view["slots"].Where(s=>s["tile"]!=null && s["tile"].Type!=JTokenType.Null && (string)s["tile"]["type"]=="Brewery"))s["tile"]["resource"]=0;
            Check(!SaleTiming.WaitForBatch(view,target,merchantId),"do not wait when own beer is insufficient");
            view["merchants"][0]["beer"]=1;
            Check(!SaleTiming.WaitForBatch(view,target,merchantId),"merchant beer justifies an immediate sale");
            view["merchants"][0]["beer"]=0;slots[0]["tile"]["beer"]=0;player["income"]=-4;
            Check(!SaleTiming.WaitForBatch(view,target,merchantId),"income conversion protects distressed finances");
            player["income"]=4;view["pending"]["kind"]="sell";
            Check(SaleTiming.Proposal(view,new Choice("sell",target,merchant:merchantId))>0,"continuation keeps grouping legal sales");
            Check(ProfileCatalog.Snapshot("balanced",4,false)["id"].ToString().StartsWith(ProfileCatalog.Runtime+"/"),"new games use the new immutable strategy identity");
            Check(ProfileCatalog.SourceHash("balanced",ProfileCatalog.EraRuntime)==ProfileCatalog.SourceHash("balanced","brass-local-ai/1.4.15"),"old era profile bytes remain available");
            File.WriteAllText(args[0],Wire.Json(new{passed=true,checks,cheap_development=cheap}));return 0;
        }
        catch(Exception e){File.WriteAllText(args[0],Wire.Json(new{passed=false,checks,error=e.ToString()}));return 1;}
    }
}
