// Rebuild explicitly corrected authored fixtures with the frozen C# engine.
// Original checkpoints and projections remain immutable.
using System;
using System.Linq;
using BrassHost;
using Newtonsoft.Json.Linq;
using BoardGameRules.Entities.Players;

public static class FixtureRepairExporter
{
    public static int Main()
    {
        string line;
        while((line=Console.ReadLine())!=null)
        {
            try
            {
                var request=JObject.Parse(line);var checkpoint=(JObject)request["checkpoint"];
                Host host=new Host((int)checkpoint["seed"],(JObject)request["setup"]);
                host.ExplicitChance=true;
                foreach(JObject command in checkpoint["commands"])
                {
                    if(command["chance"]!=null)host.ResolveChance((string)command["chance"]);
                    else host.Apply(host.Revision,(string)command["choice"]);
                }
                var state=StateExporter.Export(host);bool pending=host.Game.GameFlow.IsSellIndustryNecessary();
                state["settlement_debts"]=new JArray(Enumerable.Range(0,4).Select(i=>pending?host.Game.SellIndustryFlow._moneyToPayBySelling[(PlayerColor)i]:0));
                Console.WriteLine(Wire.Json(new {ok=true,state,snapshot=host.SearchSnapshot()}));
            }
            catch(Exception error) {Console.WriteLine(Wire.Json(new {ok=false,message=error.ToString()}));}
        }
        return 0;
    }
}
