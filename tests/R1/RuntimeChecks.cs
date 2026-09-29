using System;
using System.IO;
using System.Linq;
using System.Collections;
using System.Reflection;
using System.Security.Cryptography;
using Newtonsoft.Json.Linq;
using BrassHost;
using BoardGameRules.Entities.EventSourcing.Aggregates;
using BoardGameRules.Entities.EventSourcing.Events;
using BoardGameRules.Entities.NumericalResources.Events;

public static class RuntimeChecks
{
    static string Hash(string file){using(var s=File.OpenRead(file))using(var h=SHA256.Create())return BitConverter.ToString(h.ComputeHash(s)).Replace("-","").ToLowerInvariant();}
    public static object Run(string traces, bool unity)
    {
        var transaction=new Host(19);
        string before=Wire.Hash(transaction.Canonical());
        var routes=(IDictionary)typeof(AggregateRoot).GetField("_applyEventMethods",BindingFlags.NonPublic|BindingFlags.Instance).GetValue(transaction.Game);
        var registration=(IEventRegistration)routes[typeof(MoneyLoaned)];
        var original=registration.GetType().GetMethod("ApplyEvent");
        transaction.Game.UnregisterApplyEventRegistration(registration);
        transaction.Game.RegisterApplyEvent<MoneyLoaned>(e=>{original.Invoke(registration,new object[]{e});throw new InvalidOperationException("injected_after_mutation");});
        var loan=transaction.Choices().First(c=>c.kind=="loan");
        bool rejected=false;
        try{transaction.Apply(transaction.Revision,loan.id);}catch(InvalidOperationException e){rejected=e.Message=="injected_after_mutation";}
        if(!rejected || Wire.Hash(transaction.Canonical())!=before)throw new Exception("Transactional rollback failed");
        transaction.Apply(transaction.Revision,loan.id); // Reconstructed engine has clean handlers.
        var input=JArray.Parse(File.ReadAllText(traces));
        int decisions=0;
        foreach(JObject row in input)
        {
            var cp=(JObject)row["checkpoint"];
            var game=Host.Restore(cp);
            if(Wire.Hash(game.Canonical())!=(string)cp["hash"])throw new Exception("Trace differs: "+(string)row["label"]);
            decisions+=cp["commands"].Count();
        }
        if(CoreRegression.Main(unity?new[]{"unity"}:new string[0])!=0)throw new Exception("Core rules regressions failed");
        var resources=ResourcesRegression.Run();
        int liquidation=BrassPatch.Tests.Liquidation.LiquidationTests.Run();
        int scout=BrassPatch.Tests.Scout.ScoutTests.Run();
        if(StateRegression.Main(new string[0])!=0)throw new Exception("State regressions failed");
        return new{passed=true,runtime=Environment.Version.ToString(),unity,host_sha256=Hash(typeof(Host).Assembly.Location),trace_sha256=Hash(traces),restore_points=input.Count,replayed_decisions=decisions,transaction_rollback="passed",resources,liquidation,scout,core="passed",state="passed"};
    }
    public static int Main(string[] args)
    {
        try {var result=Run(args[0],false);File.WriteAllText(args[1],Wire.Json(result));Console.WriteLine(Wire.Json(result));return 0;}
        catch(Exception e){Console.Error.WriteLine(e);return 1;}
    }
}
