using System;
using System.IO;
using System.Linq;
using Newtonsoft.Json.Linq;
using BrassHost;
using BrassLegacyAI;

public static class EconomyChecks
{
    static int checks;
    static void Check(bool ok,string label){checks++;if(!ok)throw new Exception(label);}
    public static int Main(string[] args)
    {
        try
        {
            var host=new Host(843200,null);var v=(JObject)Wire.Token(host.Observe(host.Actor));int seat=(int)v["seat"];
            v["context"]["era"]="RailEra";v["context"]["round"]=2;v["context"]["actions_remaining"]=2;
            var me=Conversion.Player(v,seat);me["cash"]=70;me["income"]=12;me["income_marker"]=50;
            foreach(var link in v["links"])link["owner"]=seat;
            foreach(var m in v["merchants"]){m["types"]=new JArray("CottonMill","Manufacturer","Pottery");m["beer"]=0;}
            v["own_hand"]=new JArray("wild:location","wild:industry");v["coal_market"]=14;v["iron_market"]=10;
            var coal=me["mat"].Single(m=>(string)m["type"]=="CoalMine")["tiles"][1];
            var iron=me["mat"].Single(m=>(string)m["type"]=="IronWorks")["tiles"][1];
            double plentifulCoal=CompetitiveEconomy.ResourceProposal(v,coal,"Birmingham"),plentifulIron=CompetitiveEconomy.ResourceProposal(v,iron,"Birmingham");
            var scarce=(JObject)v.DeepClone();scarce["coal_market"]=0;scarce["iron_market"]=0;
            Check(CompetitiveEconomy.ResourceProposal(scarce,coal,"Birmingham")>plentifulCoal,"shortage market cash and immediate flip improve coal priority");
            Check(CompetitiveEconomy.ResourceProposal(scarce,iron,"Birmingham")>plentifulIron,"expensive iron is preferred to flooded iron");
            var poor=(JObject)v.DeepClone();Conversion.Player(poor,seat)["income"]=0;
            Check(CompetitiveEconomy.ResourceProposal(poor,coal,"Birmingham")>plentifulCoal,"coal remains an income recovery option");
            var waiting=(JObject)poor.DeepClone();var mine=waiting["slots"].First(s=>s["tile"]==null && s["types"].Any(t=>(string)t=="CoalMine"));
            mine["tile"]=coal.DeepClone();mine["tile"]["seat"]=seat;mine["tile"]["resource"]=3;
            Check(CompetitiveEconomy.ResourceProposal(waiting,coal,"Birmingham")<CompetitiveEconomy.ResourceProposal(poor,coal,"Birmingham"),"unconsumed coal is not repeated income recovery");
            var cotton=me["mat"].Single(m=>(string)m["type"]=="CottonMill")["tiles"].First(t=>(int)t["level"]==3);
            var sites=v["slots"].Where(s=>s["types"].Any(t=>(string)t=="CottonMill")).Take(2).ToArray();
            foreach(var s in sites){s["tile"]=cotton.DeepClone();s["tile"]["seat"]=seat;s["tile"]["flipped"]=false;}
            Check(CompetitiveEconomy.SellableProposal(v,cotton,"Birmingham")<0,"existing commitments prevent reusing absent beer for another build");
            var forecast=Conversion.Forecast(v);Check(CompetitiveEconomy.InventoryRisk(v,seat,forecast)>0,"unfunded beer-starved inventory carries risk");
            var brewSite=v["slots"].First(s=>s["tile"]==null && s["types"].Any(t=>(string)t=="Brewery"));
            brewSite["tile"]=me["mat"].Single(m=>(string)m["type"]=="Brewery")["tiles"][2].DeepClone();brewSite["tile"]["seat"]=seat;brewSite["tile"]["resource"]=3;
            var external=(JObject)v.DeepClone();external["slots"].Single(s=>(string)s["id"]==(string)brewSite["id"])["tile"]["seat"]=(seat+1)%4;
            Check(CompetitiveEconomy.SellableProposal(external,cotton,"Birmingham")==0,"reachable opponents' beer supports a no-own-brewery investment");
            foreach(var link in external["links"])link["owner"]=null;
            Check(CompetitiveEconomy.AvailableBeer(external,"isolated-test-region","CottonMill")==0,"unreachable external beer is never assigned to our plan");
            Check(CompetitiveEconomy.SellableProposal(v,cotton,"Birmingham")==0,"spare own beer enables another funded scoring investment");
            Check(CompetitiveEconomy.InventoryRisk(v,seat,Conversion.Forecast(v))==0,"a finishable batch is not penalized");
            foreach(var p in v["players"])p["spent"]=10;me["spent"]=0;
            Check(CompetitiveEconomy.NextOrder(v)[0]==seat,"low public spend predicts next first turn");
            double early=CompetitiveEconomy.Tempo(v,seat);me["spent"]=100;
            Check(early>CompetitiveEconomy.Tempo(v,seat),"turn order has value only with an actual batch or funded beer plan");
            foreach(var s in sites)s["tile"]["flipped"]=true;brewSite["tile"]["resource"]=0;
            Check(CompetitiveEconomy.Tempo(v,seat)==0,"no reward for a cheap turn without a follow-up opportunity");
            var late=(JObject)v.DeepClone();late["context"]["round"]=8;
            Check(CompetitiveEconomy.Tempo(late,seat)==0,"no phantom turn-order reward after the final round");
            Check(CompetitiveEconomy.RivalWeight(late)>CompetitiveEconomy.RivalWeight(v),"late Rail plans prioritize relative score more");
            Check(ProfileCatalog.Keys.Select(ProfileCatalog.Label).Distinct().Count()==6,"six new strategy choices");
            Check(ProfileCatalog.Label("balanced").Contains("Adaptive Advanced Book"),"one adaptive source-based balanced strategy");
            Check(ProfileCatalog.Label("supplier").Contains("Zero-Loan") && ProfileCatalog.Label("rail-tempo").Contains("Coventry"),"advanced market and seat-conditioned books replace traditional menu families");
            Check(ProfileCatalog.SourceHash("balanced",ProfileCatalog.IndustryRuntime)==ProfileCatalog.SourceHash("balanced","brass-local-ai/1.4.18"),"saved industry strategy bytes retained");
            // Reachable beer makes a scoring opening viable in this synthetic
            // flooded-market fixture; no strategy must invent unavailable beer.
            foreach(var merchant in v["merchants"])merchant["beer"]=3;
            int quoted;string supplySite;var supply=me["mat"].Single(m=>(string)m["type"]=="Brewery")["tiles"].First(t=>(int)t["level"]==2);
            Check(InvestmentOpportunity.FundingSite(v,supply,out quoted,out supplySite) && quoted<100000 && supplySite!=null,"necessary future brewery can be quoted without existing customers");
            foreach(string key in ProfileCatalog.Keys)
            {
                var intent=StrategicOpenings.Intent(v,ProfileCatalog.Create(key,4));
                Check(intent==null || (intent.remaining_actions>=3 && !string.IsNullOrEmpty(intent.reason) && !string.IsNullOrEmpty(intent.book_family)),"funded strategy or explicit fallback: "+key);
            }
            var cfg=ProfileCatalog.Create("cotton",4);
            var tier=new IntentState{focus="CottonMill",milestone="develop_high_tier",goal_level=3,target="Birmingham|0"};
            var develop=new Choice("develop",industry:"CottonMill");
            var before=Tactics.Bias(cfg,tier,develop,v);
            var reached=(JObject)v.DeepClone();var mat=Conversion.Player(reached,seat)["mat"].Single(m=>(string)m["type"]=="CottonMill");
            while((int)mat["tiles"][0]["level"]<3)((JArray)mat["tiles"]).RemoveAt(0);
            Check(before>0 && Tactics.Bias(cfg,tier,develop,reached)==0,"reaching named tier stops stale development bias");
            tier.milestone="finance_sequence";tier.cash_budget=20;
            Check(Tactics.Bias(cfg,tier,new Choice("loan"),reached)==0,"funded goal does not encourage another loan");
            tier.cash_budget=100;
            Check(Tactics.Bias(cfg,tier,new Choice("loan"),reached)>0,"unfunded viable goal retains a soft financing prior");
            Check(Tactics.Bias(cfg,tier,new Choice("bonus_develop",industry:"CottonMill"),reached)==0,"merchant free development remains portfolio-ranked");
            var progressing=(JObject)v.DeepClone();var goal=Tactics.Refresh(progressing,cfg,null);
            Check(goal.focus=="CottonMill" && goal.goal_level==4,"funded high-cotton fixture produces a named development goal");
            goal.age=2;progressing["context"]["revision"]=(int)progressing["context"]["revision"]+1;
            var idle=Tactics.Refresh(progressing,cfg,goal);
            Check(idle.age==3,"unchanged goal still advances its stall clock");
            var progressingMat=Conversion.Player(progressing,seat)["mat"].Single(m=>(string)m["type"]=="CottonMill");
            ((JArray)progressingMat["tiles"]).RemoveAt(0);((JArray)progressingMat["tiles"]).RemoveAt(0);
            progressing["context"]["revision"]=(int)progressing["context"]["revision"]+1;
            var advanced=Tactics.Refresh(progressing,cfg,idle);
            Check(advanced.focus=="CottonMill" && advanced.target==goal.target && advanced.age==0,"productive development resets the stall clock without abandoning the opening");
            var final=(JObject)v.DeepClone();final["context"]["round"]=8;final["context"]["actions_remaining"]=2;
            Check(StrategicOpenings.Intent(final,cfg)==null,"opening book cannot launch beyond conversion deadline");
            var expansion=(JObject)v.DeepClone();foreach(var l in expansion["links"])l["owner"]=null;
            foreach(var s in expansion["slots"])s["tile"]=null;
            var access=expansion["links"].First(l=>(string)l["era"]=="Rail" && l["regions"].Any(r=>expansion["merchants"].Any(m=>Evaluator.Region((string)m["id"])==(string)r)));
            string snapshot=Wire.Json(expansion);
            Check(CompetitiveEconomy.NetworkAccess(expansion,new Choice("network",target:(string)access["id"]))>0,"early market access is ranked for a funded follow-up investment");
            Check(Wire.Json(expansion)==snapshot,"network quotes preserve the real board and market");
            Check(ProfileCatalog.Keys.All(k=>ProfileCatalog.Label(k).EndsWith(" v4")),"the selected previous strategy is presented as v4");
            File.WriteAllText(args[0],Wire.Json(new{passed=true,checks,styles=ProfileCatalog.Keys.Select(ProfileCatalog.Label).ToArray()}));return 0;
        }
        catch(Exception e){File.WriteAllText(args[0],Wire.Json(new{passed=false,checks,error=e.ToString()}));return 1;}
    }
}
