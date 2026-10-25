// Additional flow state which is intentionally absent from public observations.
// Compiled beside StateExporter against the exact frozen reference runtime.
using System;
using System.Linq;
using BrassHost;
using Newtonsoft.Json.Linq;
using BoardGameRules.Entities.Players;

public static class FlowExporter
{
    public static int Main()
    {
        string line;
        while((line=Console.ReadLine())!=null)
        {
            try
            {
                var request=JObject.Parse(line);
                Host host=Host.Restore((JObject)request["checkpoint"]);
                host.ExplicitChance=true;
                if(request["picks"]!=null)foreach(var pick in request["picks"])host.ResolveChance((string)pick);
                var state=StateExporter.Export(host);
                bool pending=host.Game.GameFlow.IsSellIndustryNecessary();
                state["settlement_debts"]=new JArray(Enumerable.Range(0,4).Select(i=>pending?host.Game.SellIndustryFlow._moneyToPayBySelling[(PlayerColor)i]:0));
                Console.WriteLine(Wire.Json(new {ok=true,state,snapshot=host.SearchSnapshot()}));
            }
            catch(Exception error) {Console.WriteLine(Wire.Json(new {ok=false,message=error.ToString()}));}
        }
        return 0;
    }
}
