using System;
using System.Linq;
using Newtonsoft.Json.Linq;
namespace BrassHost
{
    public static class SearchProgram
    {
        public static int Main()
        {
            Host host=null;string line;
            while((line=Console.ReadLine())!=null)
            {
                string id=null;
                try
                {
                    var r=JObject.Parse(line);id=(string)r["id"];
                    if((string)r["protocol"]!=Wire.Version)throw new InvalidOperationException("protocol_mismatch");
                    string op=(string)r["op"];object data;
                    if(op=="reset"){host=new Host((int)r["seed"],r["setup"] as JObject);host.ExplicitChance=true;data=host.SearchContext();}
                    else if(op=="restore"){host=Host.Restore((JObject)r["checkpoint"]);host.InvalidateRequestsAfterRestore();data=host.SearchContext();}
                    else
                    {
                        if(host==null)throw new InvalidOperationException("reset_required");
                        if(op=="apply" || op=="chance_apply")
                            if((string)r["game_id"]!=host.GameId || (int)r["revision"]!=host.Revision)throw new InvalidOperationException("stale_request");
                        switch(op)
                        {
                            case "context":data=host.SearchContext();break;
                            case "chance":data=host.ChanceOutcomes();break;
                            case "chance_apply":host.ResolveChance((string)r["card"]);data=host.SearchContext();break;
                            case "observe":if(host.ChancePending)throw new InvalidOperationException("unresolved_private_chance");data=host.Observe((int)r["seat"]);break;
                            case "choices":if(host.ChancePending)throw new InvalidOperationException("chance_required");data=host.Choices();break;
                            case "apply":if(host.ChancePending)throw new InvalidOperationException("chance_required");host.Apply((int)r["revision"],(string)r["choice"]);data=host.SearchContext();break;
                            case "checkpoint":data=host.Checkpoint();break;
                            case "canonical_hash":data=host.FullHash();break;
                            case "result":data=host.Result();break;
                            default:throw new InvalidOperationException("unknown_operation");
                        }
                    }
                    Console.WriteLine(Wire.Json(new{id,ok=true,data}));
                }
                catch(Exception e){Console.WriteLine(Wire.Json(new{id,ok=false,message=e.Message}));}
            }
            return 0;
        }
    }
}
