using System;
using System.IO;
using Newtonsoft.Json.Linq;

public static class L14ContractChecks
{
    public static int Main(string[] args)
    {
        int validation=NativeValidationContractChecks.Main(new[]{args[0],Path.Combine(args[1],"validation-events.json")});
        Exception failure=null;int checkpoints=0;
        try {checkpoints=BrassPatch.Tests.Liquidation.LiquidationTests.Run();}
        catch(Exception error){failure=error;Console.Error.WriteLine(error);}
        File.WriteAllText(Path.Combine(args[1],"checkpoint-tests.json"),new JObject{["passed"]=failure==null,["checks"]=checkpoints,["error"]=failure?.ToString()}.ToString());
        return validation==0 && failure==null?0:1;
    }
}
