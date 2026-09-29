using System;
using System.IO;
using System.Linq;
using Newtonsoft.Json.Linq;
using BrassHost;
using BrassLegacyAI;

public static class DemandTests
{
    static int checks;
    static void Check(bool ok,string message){checks++;if(!ok)throw new Exception(message);}
    public static int Main(string[] args)
    {
        try
        {
            // Public opening from the complete-game regression. Reconstruct it
            // through legal actions; no opponent hand is supplied to the planner.
            var h=new Host(69001);
            foreach(string id in new[]{"build|Nuneaton:0|Brewery|location:Nuneaton||","iron|market||||","finish|||||",
                "build|BurtonOnTrent:1|Brewery|industry:Brewery||","iron|market||||","finish|||||",
                "loan|||industry:Brewery||","finish|||||","loan|||industry:Brewery||","finish|||||"})
            {Check(h.Choices().Any(c=>c.id==id),"legal regression opening");h.SearchApply(id);if(h.ActionComplete)h.Advance();}
            Check(h.Actor==0 && h.Remaining==2,"regression seat/round");
            var v=(JObject)Wire.Token(h.Observe(0));
            foreach(var brewery in v["slots"].Where(s=>s["tile"]!=null && s["tile"].Type!=JTokenType.Null))
                Check(Conversion.Completion(v,brewery,(int)brewery["tile"]["seat"])<=0.25,"unsupported beer has only small speculative value");
            var composition=JObject.Parse(File.ReadAllText(Path.Combine(args[1],"configs/contracts/v0.2.0/card-composition.json")));
            var cases=new JArray();string signature=null;
            foreach(int work in new[]{64,240})
            {
                var copy=new Host(1,h.TrustedSetup());string before=Wire.Hash(copy.Canonical());
                var profile=Profile.Read(JObject.Parse(File.ReadAllText(Path.Combine(args[1],"configs/legacy-ai/balanced-v3.json"))));profile.work_limit=work;
                var d=new Planner(composition).Decide(v,copy.Choices(),profile,true);
                Check(before==Wire.Hash(copy.Canonical()),"short search leaves the live game unchanged");
                foreach(var action in d.selected.actions)foreach(string id in action)copy.SearchApply(id);
                var after=(JObject)Wire.Token(copy.Observe(0));
                var owned=after["slots"].Where(s=>s["tile"]!=null && s["tile"].Type!=JTokenType.Null && (int)s["tile"]["seat"]==0).ToArray();
                // Preserve the original 240-node regression. The short-search
                // extension targets repeated unsupported investment: one early
                // brewery can prepare a later sale. Complete-game runs provide
                // separate evidence about financing and sustained progress.
                int breweries=owned.Count(s=>(string)s["tile"]["type"]=="Brewery");
                Check(breweries<=(work==64?1:0) || owned.Any(s=>Conversion.Sellable(s["tile"])),
                    "avoid unsupported brewery accumulation at work "+work+": "+d.selected.signature);
                Check(d.failures==0 && copy.Remaining==0,"complete legal demand-aware turn");
                signature=d.selected.signature;cases.Add(JObject.FromObject(new{work,signature}));
            }
            File.WriteAllText(args[0],Wire.Json(new{passed=true,checks,signature,cases,unsupportedBeerLimit=0.25}));
            Console.WriteLine("Demand PASS "+checks+" "+signature);return 0;
        }
        catch(Exception e){File.WriteAllText(args[0],Wire.Json(new{passed=false,checks,error=e.ToString()}));Console.Error.WriteLine(e);return 1;}
    }
}
