using System;
using System.IO;
using System.Linq;
using System.Collections.Generic;
using Newtonsoft.Json.Linq;
using BrassHost;
using BrassLegacyAI;

public static class RouteTests
{
    static int checks;static readonly List<object> cases=new List<object>();
    static void Check(bool ok,string message){checks++;if(!ok)throw new Exception(message);}
    public static int Main(string[] args)
    {
        try
        {
            // Reproduce the public opening of the Goods/Beer economic stall.
            // The old planner preferred repeated breweries/development despite
            // affordable production one canal away from a compatible merchant.
            var h=new Host(82002);
            foreach(var action in new[]{
                new[]{"loan|||location:Cannock||","finish|||||"},
                new[]{"loan|||industry:CottonMill+Manufacturer||","finish|||||"},
                new[]{"loan|||industry:Brewery||","finish|||||"},
                new[]{"loan|||industry:CoalMine||","finish|||||"},
                new[]{"loan|||location:Kidderminster||","finish|||||"},
                new[]{"build|Derby:0|Brewery|location:Derby||","iron|market||||","finish|||||"}})
            {foreach(string id in action){Check(h.Choices().Any(c=>c.id==id),"legal regression prefix");h.SearchApply(id);}h.Advance();}
            Check(h.Actor==1 && h.Remaining==2,"regression turn");
            var view=(JObject)Wire.Token(h.Observe(1));
            var composition=JObject.Parse(File.ReadAllText(Path.Combine(args[1],"configs/contracts/v0.2.0/card-composition.json")));
            foreach(int work in new[]{64,240})
            {
                var p=Profile.Read(JObject.Parse(File.ReadAllText(Path.Combine(args[1],"configs/legacy-ai/goods-beer-v3.json"))));p.work_limit=work;
                string before=Wire.Hash(h.Canonical());
                var d=new Planner(composition).Decide(view,h.Choices(),p,true);
                Check(Wire.Hash(h.Canonical())==before,"planning is isolated");
                var replay=new Host(1,h.TrustedSetup());
                foreach(var action in d.selected.actions)foreach(string id in action)replay.SearchApply(id);
                var after=(JObject)Wire.Token(replay.Observe(1));
                var production=after["slots"].Where(s=>s["tile"]!=null && s["tile"].Type!=JTokenType.Null && (int)s["tile"]["seat"]==1 && Conversion.Sellable(s["tile"])).ToArray();
                Check(replay.Remaining==0 && d.failures==0,"complete legal turn");
                Check(production.Any(s=>Conversion.Completion(after,s,1)>=0.68),"invest in reachable production instead of an unsupported supply loop");
                Check(!d.selected.cash.future_credit_blocked && d.selected.cash.shortfall==0,"preserve finance for completion");
                // With no actions before the deadline, even that promising route
                // cannot score an unfinished tile. This is not a blanket build bonus.
                after["context"]["round"]=Conversion.RoundLimit(after);after["context"]["actions_remaining"]=0;
                foreach(var slot in production)Check(Conversion.Completion(after,slot,1)==0,"deadline overrides route potential");
                cases.Add(new{work,d.selected.signature,d.nodes});
            }
            File.WriteAllText(args[0],Wire.Json(new{passed=true,checks,cases}));Console.WriteLine("Route PASS "+checks);return 0;
        }
        catch(Exception e){File.WriteAllText(args[0],Wire.Json(new{passed=false,checks,cases,error=e.ToString()}));Console.Error.WriteLine(e);return 1;}
    }
}
