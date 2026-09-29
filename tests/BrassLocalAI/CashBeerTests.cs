using System;
using System.IO;
using System.Linq;
using System.Collections.Generic;
using Newtonsoft.Json.Linq;
using BrassHost;
using BrassLegacyAI;

public static class CashBeerTests
{
    static int checks;static readonly List<object> cases=new List<object>();
    static void Check(bool ok,string message){checks++;if(!ok)throw new Exception(message);}
    static JToken Put(JObject v,string id,string type)
    {
        var mat=(JArray)Conversion.Player(v,(int)v["seat"])["mat"].Single(m=>(string)m["type"]==type)["tiles"];
        var tile=mat[0].DeepClone();mat.RemoveAt(0);tile["flipped"]=false;
        if(type=="Brewery")tile["resource"]=1;
        v["slots"].Single(s=>(string)s["id"]==id)["tile"]=tile;return tile;
    }
    public static int Main(string[] args)
    {
        try
        {
            var composition=JObject.Parse(File.ReadAllText(Path.Combine(args[1],"configs/contracts/v0.2.0/card-composition.json")));
            foreach(bool brewery in new[]{true,false})foreach(string key in new[]{"balanced","rail-tempo","cotton","goods-beer","supplier","pottery-flex"})
            {
                var source=new Host(1301);var v=(JObject)Wire.Token(source.Observe(source.Actor));int seat=source.Actor;
                v["context"]["round"]=7;v["context"]["actions_remaining"]=1;
                var me=Conversion.Player(v,seat);me["cash"]=0;me["income_marker"]=2;
                Put(v,"Worcester:0","CottonMill");if(brewery)Put(v,"BurtonOnTrent:0","Brewery");
                int n=0;foreach(var link in v["links"].Where(l=>((string)l["id"]).StartsWith("Canal")))link["owner"]=n++%4;
                foreach(var merchant in v["merchants"])
                {
                    string region=Evaluator.Region((string)merchant["id"]);
                    bool enabled=region=="Shrewsbury" || (!brewery && region=="Warrington");
                    merchant["types"]=enabled?new JArray("CottonMill"):null;merchant["beer"]=enabled?1:0;
                }
                var h=new Host(1,Belief.Setup(v,composition,7));v=(JObject)Wire.Token(h.Observe(seat));
                Check(!h.Choices().Any(c=>c.kind=="loan"),"fixture is below loan floor");
                // Independently prove there is a legal payment-safe sale.
                var proof=new Host(1,Belief.Setup(v,composition,7));
                var sale=proof.Choices().First(c=>c.kind=="sell" && c.merchant.StartsWith(brewery?"Shrewsbury:":"Warrington:"));proof.SearchApply(sale.id);
                proof.SearchApply(proof.Choices().First(c=>c.kind=="beer" && (brewery?c.target=="BurtonOnTrent:0":c.target.StartsWith("merchant:Warrington:"))).id);
                proof.SearchApply(proof.Choices().Single(c=>c.kind=="finish").id);
                Check(CashManagement.Project((JObject)Wire.Token(proof.Observe(seat)),seat).shortfall==0,"safe alternative exists in actual rules");
                var profile=Profile.Read(JObject.Parse(File.ReadAllText(Path.Combine(args[1],"configs/legacy-ai/"+key+"-v3.json"))));profile.work_limit=1;profile.milliseconds=0;
                var published=new List<Plan>();var d=new Planner(composition).Decide(v,h.Choices(),profile,false,1,default(System.Threading.CancellationToken),p=>published.Add(p));
                foreach(var plan in published.Concat(new[]{d.selected}))
                {
                    var replay=new Host(1,Belief.Setup(v,composition,7));foreach(var action in plan.actions)foreach(string id in action)replay.SearchApply(id);
                    Check(replay.Remaining==0 && CashManagement.Project((JObject)Wire.Token(replay.Observe(seat)),seat).shortfall==0,
                        "exhausted-budget cash/beer plan: "+key+" brewery="+brewery+" "+plan.signature);
                }
                Check(d.nodes<160,"bounded emergency alternatives");cases.Add(new{key,brewery,d.selected.signature,d.nodes});
            }
            File.WriteAllText(args[0],Wire.Json(new{passed=true,checks,cases}));Console.WriteLine("Cash/beer PASS "+checks);return 0;
        }
        catch(Exception e){File.WriteAllText(args[0],Wire.Json(new{passed=false,checks,cases,error=e.ToString()}));Console.Error.WriteLine(e);return 1;}
    }
}
