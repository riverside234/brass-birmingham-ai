// Additive diagnostic only, linked against the hash-verified frozen C# runtime.
using System;
using System.Linq;
using BrassHost;
using Newtonsoft.Json.Linq;
using BoardGameRules.Entities.Players;

public static class RulesExporter
{
    public static JObject Export(Host host)
    {
        var state=StateExporter.Export(host);
        bool pending=host.Game.GameFlow.IsSellIndustryNecessary();
        state["settlement_debts"]=new JArray(Enumerable.Range(0,4).Select(i=>pending?host.Game.SellIndustryFlow._moneyToPayBySelling[(PlayerColor)i]:0));
        return new JObject { ["ok"]=true,["state"]=state,["choices"]=host.ChancePending?new JArray():Wire.Token(host.Choices()),["chance"]=host.ChancePending?Wire.Token(host.ChanceOutcomes()):JValue.CreateNull() };
    }
    public static int Main()
    {
        Host host=null;string line;
        while((line=Console.ReadLine())!=null)
        {
            try
            {
                var request=JObject.Parse(line);
                if(request["checkpoint"]!=null)
                {
                    var cp=(JObject)request["checkpoint"];
                    if(request["setup"]!=null)
                    {
                        host=new Host((int)cp["seed"],(JObject)request["setup"]);host.ExplicitChance=true;
                        foreach(var command in cp["commands"]) { if(command["chance"]!=null)host.ResolveChance((string)command["chance"]);else host.Apply(host.Revision,(string)command["choice"]); }
                    }
                    else host=Host.Restore(cp);
                }
                else if(request["seed"]!=null) host=new Host((int)request["seed"],(JObject)request["setup"]);
                if(host==null)throw new InvalidOperationException("no_host");host.ExplicitChance=true;
                if(request["choice"]!=null)host.Apply((int?)request["revision"]??host.Revision,(string)request["choice"]);
                if(request["card"]!=null)host.ResolveChance((string)request["card"]);
                var reply=Export(host);
                if((bool?)request["projection"]==true)reply["snapshot"]=Wire.Token(host.SearchSnapshot());
                Console.WriteLine(Wire.Json(reply));
            }
            catch(Exception error){Console.WriteLine(Wire.Json(new{ok=false,message=error.ToString()}));}
        }
        return 0;
    }
}
