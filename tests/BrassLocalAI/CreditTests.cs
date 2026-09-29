using System;
using System.IO;
using System.Linq;
using System.Collections.Generic;
using Newtonsoft.Json.Linq;
using BoardGameRules.Entities.NumericalResources;
using BrassHost;
using BrassLegacyAI;

public static class CreditTests
{
    static int checks;static readonly List<object> cases=new List<object>();
    static void Check(bool ok,string message){checks++;if(!ok)throw new Exception(message);}
    public static int Main(string[] args)
    {
        try
        {
            var composition=JObject.Parse(File.ReadAllText(Path.Combine(args[1],"configs/contracts/v0.2.0/card-composition.json")));
            foreach(int cash in new[]{12,0})foreach(string key in new[]{"balanced","rail-tempo","cotton","goods-beer","supplier","pottery-flex"})
            {
                var source=new Host(1901);var v=(JObject)Wire.Token(source.Observe(source.Actor));int seat=source.Actor;
                v["context"]["round"]=2;v["context"]["actions_remaining"]=2;
                var me=Conversion.Player(v,seat);me["cash"]=cash;
                me["income_marker"]=Enumerable.Range(0,100).First(x=>IncomeTrack.GetCurrentIncomeLevel(x)==-6);
                var h=new Host(1,Belief.Setup(v,composition,7));v=(JObject)Wire.Token(h.Observe(seat));
                Check(h.Choices().Any(c=>c.kind=="loan"),"fixture can still borrow");
                var profile=Profile.Read(JObject.Parse(File.ReadAllText(Path.Combine(args[1],"configs/legacy-ai/"+key+"-v3.json"))));
                profile.work_limit=1;profile.milliseconds=0;
                var published=new List<Plan>();var d=new Planner(composition).Decide(v,h.Choices(),profile,false,1,default(System.Threading.CancellationToken),p=>published.Add(p));
                foreach(var plan in published.Concat(new[]{d.selected}))
                {
                    var replay=new Host(1,Belief.Setup(v,composition,7));foreach(var action in plan.actions)foreach(string id in action)replay.SearchApply(id);
                    var after=(JObject)Wire.Token(replay.Observe(seat));var money=CashManagement.Project(after,seat);
                    Check(replay.Remaining==0 && money.shortfall==0,"complete payment-safe incumbent");
                    if(cash>0)Check(!money.future_credit_blocked,"retain borrowing access when passing is payment-safe");
                    else Check(plan.signature.Contains("loan|"),"emergency loan remains available");
                }
                Check(d.nodes<240,"bounded recovery search");
                cases.Add(new{key,cash,d.selected.signature,d.nodes});
            }
            File.WriteAllText(args[0],Wire.Json(new{passed=true,checks,cases}));Console.WriteLine("Credit PASS "+checks);return 0;
        }
        catch(Exception e){File.WriteAllText(args[0],Wire.Json(new{passed=false,checks,cases,error=e.ToString()}));Console.Error.WriteLine(e);return 1;}
    }
}
