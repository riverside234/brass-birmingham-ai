using System;
using System.IO;
using System.Linq;
using System.Collections.Generic;
using Newtonsoft.Json.Linq;
using BrassHost;
using BrassLegacyAI;

public static class NetworkOptionsChecks
{
    static int checks;
    static void Check(bool ok,string message){checks++;if(!ok)throw new Exception(message);}
    static bool Present(JToken s){return s["tile"]!=null && s["tile"].Type!=JTokenType.Null;}
    static JObject RailPosition()
    {
        var h=new Host(844000);
        for(int i=0;i<160;i++){
            var v=(JObject)Wire.Token(h.Observe(h.Actor));
            if(EraEconomy.Rail(v) && h.Remaining==2 && h.Phase=="action")return v;
            h.SearchApply(h.Choices().First(c=>c.kind=="pass").id);h.SearchApply(h.Choices().Single(c=>c.kind=="finish").id);h.Advance();
        }
        throw new Exception("rail fixture boundary");
    }
    public static int Main(string[] args)
    {
        try{
            string root=args[1];var composition=JObject.Parse(File.ReadAllText(Path.Combine(root,"configs/contracts/v0.2.0/card-composition.json")));
            var v=(JObject)Wire.Token(new Host(844000).Observe(0));int seat=(int)v["seat"];
            var p=Profile.Read(JObject.Parse(File.ReadAllText(Path.Combine(root,"configs/legacy-ai/balanced-v3-masked-economy.json"))));
            var links=v["links"].Where(l=>(string)l["era"]=="Canal").ToArray();
            var link=links.First(l=>!l["regions"].Any(r=>v["merchants"].Any(m=>Evaluator.Region((string)m["id"])==(string)r)));
            var choices=new List<Choice>{new Choice("network",target:(string)link["id"],card:"industry:CottonMill"),new Choice("pass",card:"industry:CottonMill")};
            v["own_hand"]=new JArray("industry:CoalMine");
            var network=NetworkOptions.BiasValues(v,choices,p,new IntentState{focus="network"});
            var noCard=NetworkOptions.BiasValues(v,choices,p,new IntentState{focus="CottonMill"});
            Check(network[0]>noCard[0],"network intent favors city/brewery capacity without manufacturing cards");
            Check(network.All(x=>x>=0 && x<=NetworkOptions.MaximumRouteBias) && network[1]==0,"route cues are bounded and do not bias unrelated actions");
            var supported=(JObject)v.DeepClone();supported["own_hand"]=new JArray("wild:industry");
            var candidates=links.Select(l=>new Choice("network",target:(string)l["id"])).ToList();
            var absent=NetworkOptions.BiasValues(v,candidates,p,new IntentState{focus="CottonMill"});
            var present=NetworkOptions.BiasValues(supported,candidates,p,new IntentState{focus="CottonMill"});
            Check(present.Where((x,i)=>x>absent[i]).Any(),"industry expansion uses vacant slots and supporting hand cards");
            var merchantLink=links.First(l=>l["regions"].Any(r=>v["merchants"].Any(m=>Evaluator.Region((string)m["id"])==(string)r)));
            var merchantChoices=new List<Choice>{new Choice("network",target:(string)merchantLink["id"])};
            var buyer=(JObject)v.DeepClone();foreach(var m in buyer["merchants"])m["types"]=new JArray("CottonMill");
            var wrongBuyer=(JObject)buyer.DeepClone();foreach(var m in wrongBuyer["merchants"])m["types"]=new JArray("Pottery");
            Check(NetworkOptions.BiasValues(buyer,merchantChoices,p,new IntentState{focus="CottonMill"})[0]>
                NetworkOptions.BiasValues(wrongBuyer,merchantChoices,p,new IntentState{focus="CottonMill"})[0],"merchant access must match the intended industry");
            string original=Wire.Json(v);var all=new List<string[]>{new[]{choices[0].id,"finish|||||"}};
            Check(NetworkOptions.CanalPenalty(v,all)==0,"first canal is exempt");
            v["links"].Single(l=>(string)l["id"]==(string)links.Last()["id"])["owner"]=seat;v["context"]["round"]=3;
            Check(NetworkOptions.CanalPenalty(v,all)==NetworkOptions.EarlyCanalCost,"unsupported additional canal carries a small early cost");
            v["context"]["round"]=4;Check(NetworkOptions.CanalPenalty(v,all)==0,"explicit early cutoff ends after round three");
            v["context"]["round"]=8;Check(NetworkOptions.CanalPenalty(v,all)==0,"late scoring/access canals have no additional penalty");
            // A connection that extends our network to an actual later
            // industry-card build is exempt. A build BEFORE it is not enabling.
            var enable=(JObject)v.DeepClone();enable["context"]["round"]=2;
            string addedRegion=(string)link["regions"][0];
            var buildSite=enable["slots"].First(s=>Evaluator.Region((string)s["id"])==addedRegion);
            string buildType=(string)buildSite["types"][0];
            foreach(var l in enable["links"])l["owner"]=null;
            var anchor=enable["links"].First(l=>(string)l["id"]!=(string)link["id"] && !l["regions"].Any(r=>(string)r==addedRegion));anchor["owner"]=seat;
            var buildId=new Choice("build",target:(string)buildSite["id"],industry:buildType,card:"industry:"+buildType).id;
            Check(NetworkOptions.CanalPenalty(enable,new List<string[]>{all[0],new[]{buildId}})==0,"actual later industry access exempts an enabling early canal");
            Check(NetworkOptions.CanalPenalty(enable,new List<string[]>{new[]{buildId},all[0]})>0,"an earlier build does not retroactively justify a later canal");
            v["context"]["era"]="RailEra";v["context"]["round"]=1;Check(NetworkOptions.CanalPenalty(v,all)==0,"rail never inherits the canal opportunity penalty");
            var empty=(JObject)buyer.DeepClone();empty["links"].Single(l=>(string)l["id"]==(string)link["id"])["owner"]=seat;
            var final=Evaluator.KnownFinalScore(empty);
            Check((int)Conversion.Player(final,seat)["vp"]==(int)Conversion.Player(empty,seat)["vp"],"empty city/brewery capacity is not earned link VP");
            var quoteBefore=Wire.Json(empty);NetworkOptions.BiasValues(empty,candidates,p,new IntentState{focus="network"});
            Check(Wire.Json(empty)==quoteBefore,"route preparation leaves the permitted observation unchanged");
            // Actual rules replay, not a sum of independent approximate rails.
            var rail=RailPosition();seat=(int)rail["seat"];var me=Conversion.Player(rail,seat);me["cash"]=100;me["income"]=20;me["income_marker"]=80;me["link_tiles"]=2;
            foreach(var l in rail["links"])l["owner"]=(string)l["era"]=="Rail"?(JToken)seat:null;
            var free=rail["links"].Where(l=>(string)l["era"]=="Rail").Take(2).ToArray();foreach(var l in free)l["owner"]=null;
            var brewery=rail["slots"].First(s=>s["types"].Any(t=>(string)t=="Brewery"));
            var breweryMat=(JArray)me["mat"].Single(m=>(string)m["type"]=="Brewery")["tiles"];
            var breweryTile=breweryMat.First(t=>(int)t["level"]==2);int breweryIndex=breweryMat.IndexOf(breweryTile);
            brewery["tile"]=breweryTile.DeepClone();for(int i=0;i<=breweryIndex;i++)breweryMat.RemoveAt(0);
            brewery["tile"]["seat"]=seat;brewery["tile"]["resource"]=2;rail["coal_market"]=14;
            var cottonMat=(JArray)me["mat"].Single(m=>(string)m["type"]=="CottonMill")["tiles"];
            var cottonTile=cottonMat.First(t=>(int)t["level"]==2);int cottonIndex=cottonMat.IndexOf(cottonTile);
            var cottonSite=rail["slots"].First(s=>!Present(s) && s["types"].Any(t=>(string)t=="CottonMill"));
            cottonSite["tile"]=cottonTile.DeepClone();cottonSite["tile"]["seat"]=seat;
            for(int i=0;i<=cottonIndex;i++)cottonMat.RemoveAt(0);
            foreach(var m in rail["merchants"])m["types"]=new JArray("CottonMill","Manufacturer","Pottery");
            var h=new Host(1,Belief.Setup(rail,composition,7));rail=(JObject)Wire.Token(h.Observe(seat));
            p.simulations=1;p.work_limit=1000000;
            var d=new Planner(composition).Decide(rail,h.Choices(),p,true);
            var pair=d.alternatives.FirstOrDefault(plan=>plan.actions[0].Count(id=>id.StartsWith("network|"))==2);
            Check(pair!=null,"complete double-rail alternatives are evaluated alongside ordinary plans");
            int beforeCash=(int)Conversion.Player(rail,seat)["cash"],beforeBeer=(int)rail["slots"].Single(s=>(string)s["id"]==(string)brewery["id"])["tile"]["resource"];
            foreach(string id in pair.actions[0]){Check(h.Choices().Any(c=>c.id==id),"every pair/payment continuation is legal");h.SearchApply(id);}
            var paid=(JObject)Wire.Token(h.Observe(seat));int delta=beforeCash-(int)Conversion.Player(paid,seat)["cash"];
            Check(delta==17,"double rail pays both fees (15) and both market coal cubes (2)");
            Check((int)paid["coal_market"]==12,"two actual coal payments change the market");
            Check(beforeBeer-(int)paid["slots"].Single(s=>(string)s["id"]==(string)brewery["id"])["tile"]["resource"]==1,"double rail pays one actual brewery barrel");
            Check(d.alternatives.Any(plan=>plan.actions.Any(a=>a[0].StartsWith("build|"))),"double rail does not remove industry comparisons");
            Check(d.alternatives.Any(plan=>plan.actions.Any(a=>a[0].StartsWith("build|")) && plan.actions.Any(a=>a[0].StartsWith("sell|"))),"feasible industry-and-sale turns compete with double rails");
            Check(d.search.network_candidates<=1,"additional pair construction is bounded to one per private owner");
            Check(!NetworkOptions.RetainsCapital(new Plan{cash=new CashProjection{financing_actions=1}}),"explicit pair cannot discard necessary working capital");
            Check(!NetworkOptions.RetainsCapital(new Plan{cash=new CashProjection{shortfall=1}}),"explicit pair cannot borrow hypothetical income to pay a bill");
            var dry=(JObject)rail.DeepClone();dry["slots"].Single(s=>(string)s["id"]==(string)brewery["id"])["tile"]["resource"]=0;
            var dryHost=new Host(1,Belief.Setup(dry,composition,7));dryHost.SearchApply(dryHost.Choices().First(c=>c.kind=="network").id);
            while(dryHost.Phase=="payment")dryHost.SearchApply(dryHost.Choices().First().id);
            Check(!dryHost.Choices().Any(c=>c.kind=="network"),"the second rail is masked when no brewery beer is available");
            p.milliseconds=0;var stopped=new Planner(composition).Decide(rail,new Host(1,Belief.Setup(rail,composition,7)).Choices(),p);
            Check(stopped.search.network_candidates==0,"expired clocks skip additional rail construction and retain a legal incumbent");
            File.WriteAllText(args[0],Wire.Json(new{passed=true,checks,early_canal_last_round=NetworkOptions.EarlyCanalLastRound,
                route_bias_cap=NetworkOptions.MaximumRouteBias,pair_cash=delta,pair=d.search.network_candidates}));
            Console.WriteLine("Network options PASS "+checks);return 0;
        }catch(Exception e){File.WriteAllText(args[0],Wire.Json(new{passed=false,checks,error=e.ToString()}));Console.Error.WriteLine(e);return 1;}
    }
}
