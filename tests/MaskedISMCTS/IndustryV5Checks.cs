using System;
using System.IO;
using System.Linq;
using Newtonsoft.Json.Linq;
using BrassHost;
using BrassLegacyAI;

public static class IndustryV5Checks
{
    static int checks;
    static void Check(bool ok,string message){checks++;if(!ok)throw new Exception(message);}
    static JObject Fixture()
    {
        var host=new Host(845500,null);var v=(JObject)Wire.Token(host.Observe(host.Actor));
        v["context"]["round"]=2;v["context"]["actions_remaining"]=2;
        var me=Conversion.Player(v,(int)v["seat"]);me["cash"]=100;me["income"]=8;me["income_marker"]=35;
        v["own_hand"]=new JArray("wild:location","wild:industry","industry:CottonMill");
        v["iron_market"]=8;v["coal_market"]=14;
        foreach(var l in v["links"])l["owner"]=(int)v["seat"];
        foreach(var m in v["merchants"]){m["types"]=new JArray("CottonMill","Manufacturer","Pottery");m["beer"]=3;}
        var mat=me["mat"].Single(m=>(string)m["type"]=="CottonMill")["tiles"];
        mat[0].Remove();mat[0].Remove();return v;
    }
    static JToken[] Tiles(JObject v,string type)
    {return Conversion.Player(v,(int)v["seat"])["mat"].Single(m=>(string)m["type"]==type)["tiles"].ToArray();}
    public static int Main(string[] args)
    {
        try
        {
            var v=Fixture();int seat=(int)v["seat"];var cotton=Tiles(v,"CottonMill");
            int removed=Array.FindIndex(cotton,t=>(int)t["level"]==4);
            var snapshot=Wire.Json(v);var high=FundedDevelopment.Quote(v,cotton[removed],removed);
            Check(high!=null && high.removed==removed && high.reserve==4 && !high.borrow,"cash-rich quote reaches cotton IV with cash reserve");
            Check(Wire.Json(v)==snapshot,"quotes cannot mutate real iron, mat or board");
            var profile=Profile.Read(JObject.Parse(File.ReadAllText(Path.Combine(args[1],"configs/legacy-ai/cotton-v5-masked-industry.json"))));
            var intent=TournamentOpenings.Intent(v,profile);
            Check(intent!=null && intent.focus=="CottonMill" && intent.goal_level>=3,"funded cotton selects a higher tier");
            Check(intent.book_family.StartsWith("v5/") && intent.reason.Contains("iron cost="),"v5 goal reports the actual funded sequence");
            double cheap=InvestmentOpportunity.Development(v,"CottonMill");
            var scarce=(JObject)v.DeepClone();scarce["iron_market"]=0;
            var costly=FundedDevelopment.Quote(scarce,Tiles(scarce,"CottonMill")[removed],removed);
            Check(costly!=null && costly.cost>high.cost && costly.iron_cost>high.iron_cost,"iron scarcity reprices the complete development sequence");
            Check(cheap>0 && cheap>InvestmentOpportunity.Development(scarce,"CottonMill"),"cheap iron increases useful development priority");
            var poor=(JObject)v.DeepClone();var p=Conversion.Player(poor,seat);p["cash"]=0;p["income"]=-10;
            Check(FundedDevelopment.Quote(poor,Tiles(poor,"CottonMill")[removed],removed)==null,"unfunded upgrade rejected including lost credit");
            Check(InvestmentOpportunity.Development(poor,"CottonMill")<0,"no cheap-iron excuse for an unaffordable build");
            var late=(JObject)v.DeepClone();late["context"]["round"]=8;late["context"]["actions_remaining"]=1;
            Check(FundedDevelopment.Quote(late,Tiles(late,"CottonMill")[removed],removed)==null,"final action cannot develop then build then sell");
            Check(InvestmentOpportunity.Development(late,"CottonMill")<0,"no terminal development bonus");
            var absent=(JObject)v.DeepClone();absent["own_hand"]=new JArray("industry:Pottery");
            Check(FundedDevelopment.Quote(absent,Tiles(absent,"CottonMill")[removed],removed)==null,"absent building card invalidates the goal");
            var noBuyer=(JObject)v.DeepClone();foreach(var m in noBuyer["merchants"])m["types"]=new JArray("Pottery");
            Check(FundedDevelopment.Quote(noBuyer,Tiles(noBuyer,"CottonMill")[removed],removed)==null,"no cotton buyer invalidates the goal");
            var pottery=Tiles(v,"Pottery");
            Check(!(bool)pottery[0]["developable"] && FundedDevelopment.Quote(v,pottery[1],1)==null,"non-developable pottery cannot be skipped");
            var iron=(JObject)v.DeepClone();var works=Tiles(iron,"IronWorks")[0];
            var ironSite=iron["slots"].First(s=>s["tile"]==null && s["types"].Any(t=>(string)t=="IronWorks"));
            ironSite["tile"]=works.DeepClone();ironSite["tile"]["resource"]=1;ironSite["tile"]["seat"]=(seat+1)%4;
            int combined=FundedDevelopment.IronBudget(iron,3),separate=FundedDevelopment.IronBudget(iron,2)+FundedDevelopment.IronBudget(iron,1);
            Check(combined>separate,"shared free iron cannot be counted twice");
            var shared=FundedDevelopment.Quote(iron,Tiles(iron,"CottonMill")[removed],removed);
            var sharedTile=Tiles(iron,"CottonMill")[removed];int baseBuild=EraEconomy.BuildCost(iron,sharedTile,Evaluator.Region(shared.site));
            Check(shared.cost==baseBuild+FundedDevelopment.IronBudget(iron,removed+(int)sharedTile["iron"])-FundedDevelopment.IronBudget(iron,(int)sharedTile["iron"]),
                "complete plan uses the cumulative iron quote, including build iron");
            Check(shared.cost<high.cost,"actual public board iron reduces the funded quote");
            Check(FundedDevelopment.DevelopActions(v,5)==3,"up to three paid develop actions are considered");
            var continuing=(JObject)v.DeepClone();continuing["pending"]["kind"]="develop";
            Check(FundedDevelopment.DevelopActions(continuing,1)==1 && FundedDevelopment.DevelopActions(continuing,2)==2,"optional second removal is costed in the open action");
            var rail=(JObject)v.DeepClone();rail["context"]["era"]="RailEra";
            Check(InvestmentOpportunity.MatValue(rail,Tiles(rail,"CottonMill")[0])>0,"Canal-only head does not hide a funded Rail replacement");
            var reached=(JObject)v.DeepClone();var mat=Conversion.Player(reached,seat)["mat"].Single(m=>(string)m["type"]=="CottonMill")["tiles"];
            while((int)mat[0]["level"]<intent.goal_level)mat[0].Remove();
            Check(Tactics.Bias(profile,intent,new Choice("develop",industry:"CottonMill"),reached)==0,"goal ends development bias when tier is reached");
            var end=(JObject)v.DeepClone();end["context"]["terminal"]=true;
            Check(InvestmentOpportunity.MatValue(end,Tiles(end,"CottonMill")[0])==0,"no unearned mat option at terminal");
            File.WriteAllText(args[0],Wire.Json(new{passed=true,checks,intent,cheap_development=cheap,high,costly,combined_iron=combined}));return 0;
        }
        catch(Exception e){File.WriteAllText(args[0],Wire.Json(new{passed=false,checks,error=e.ToString()}));return 1;}
    }
}
