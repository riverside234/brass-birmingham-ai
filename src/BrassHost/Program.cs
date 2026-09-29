using System;
using System.IO;
using System.Linq;
using Newtonsoft.Json.Linq;

namespace BrassHost
{
    public static class Program
    {
        public static int Main(string[] args)
        {
            Host host=null;
            string line;
            while((line=Console.ReadLine())!=null)
            {
                string id=null;
                try
                {
                    var req=JObject.Parse(line);id=(string)req["id"];
                    if((string)req["protocol"]!=Wire.Version) throw new InvalidOperationException("protocol_mismatch");
                    string op=(string)req["op"]; object data;
                    if(op=="reset"){host=new Host((int)req["seed"],req["setup"] as JObject);data=host.Context();}
                    else if(op=="restore"){host=Host.Restore((JObject)req["checkpoint"]);host.InvalidateRequestsAfterRestore();data=host.Context();}
                    else
                    {
                        if(host==null) throw new InvalidOperationException("reset_required");
                        if((op=="apply" || op=="apply_action") && (string)req["game_id"]!=host.GameId) throw new InvalidOperationException("stale_game");
                        switch(op)
                        {
                            case "context":data=host.Context();break;
                            case "observe":data=host.Observe((int)req["seat"]);break;
                            case "choices":data=host.Choices();break;
                            case "apply":host.Apply((int)req["revision"],(string)req["choice"]);data=host.Context();break;
                            case "apply_action":host.ApplyAction((int)req["revision"],req["choices"].Select(x=>(string)x).ToArray());data=host.Context();break;
                            case "advance":host.Advance();data=host.Context();break;
                            case "checkpoint":data=host.Checkpoint();break;
                            case "canonical_hash":data=Wire.Hash(host.Canonical());break;
                            case "result":data=host.Result();break;
                            default:throw new InvalidOperationException("unknown_operation");
                        }
                    }
                    Console.WriteLine(Wire.Json(new{id,ok=true,data}));
                }
                catch(Exception e){Console.WriteLine(Wire.Json(new{id,ok=false,error=e.GetType().Name,message=e.Message}));}
            }
            return 0;
        }
    }
}
