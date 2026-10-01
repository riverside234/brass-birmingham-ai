using System;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using Newtonsoft.Json.Linq;
using BrassHost;
namespace BrassLegacyAI
{
    public static class Program
    {
        public const string LabRoot="/data/not_backed_up/yxu209/ai-training/";
        static void VerifyRuntime()
        {
            string dir=AppDomain.CurrentDomain.BaseDirectory;
            var manifest=JObject.Parse(File.ReadAllText(Path.Combine(dir,"runtime.json")));
            foreach(var item in manifest["files"])
            {
                using(var stream=File.OpenRead(Path.Combine(dir,(string)item["name"])))using(var hash=SHA256.Create())
                if(BitConverter.ToString(hash.ComputeHash(stream)).Replace("-","").ToLowerInvariant()!=(string)item["sha256"])throw new Exception("runtime_fingerprint:"+item["name"]);
            }
        }
        public static int Main(string[] args)
        {
            string root=args.Length>0?args[0]:LabRoot;VerifyRuntime();
            var composition=JObject.Parse(File.ReadAllText(Path.Combine(root,"configs/contracts/v0.2.0/card-composition.json")));
            var planner=new Planner(composition);Host host=null;string line;
            while((line=Console.ReadLine())!=null)
            {
                string id=null;
                try
                {
                    var r=JObject.Parse(line);id=(string)r["id"];
                    if((string)r["protocol"]!="legacy/0.1.0")throw new Exception("protocol_mismatch");
                    string op=(string)r["op"];object data;
                    switch(op)
                    {
                        case "reset":host=new Host((int)r["seed"],r["setup"] as JObject);data=host.Context();break;
                        case "observe":data=host.Observe((int)r["seat"]);break;
                        case "choices":data=host.Choices();break;
                        case "context":data=host.Context();break;
                        case "hash":data=Wire.Hash(host.Canonical());break;
                        case "checkpoint":data=host.Checkpoint();break;
                        case "restore":host=Host.Restore((JObject)r["checkpoint"]);host.InvalidateRequestsAfterRestore();data=host.Context();break;
                        case "apply":
                            if(host.GameId!=(string)r["game_id"])throw new Exception("stale_game");
                            host.Apply((int)r["revision"],(string)r["choice"]);data=host.Context();break;
                        case "commit":
                            if(host.GameId!=(string)r["game_id"] || host.Revision!=(int)r["revision"])throw new Exception("stale_request");
                            host.CommitPlan(r["choices"].Select(x=>(string)x).ToArray());data=host.Context();break;
                        case "plan":data=planner.Decide((JObject)r["observation"],r["choices"].ToObject<System.Collections.Generic.List<Choice>>(),Profile.Read((JObject)r["profile"]),(bool?)r["fixed_work"]??false,(uint?)r["sampling_seed"]??1,intent:r["intent"]?.ToObject<IntentState>());break;
                        case "intent":data=Tactics.Refresh((JObject)r["observation"],Profile.Read((JObject)r["profile"]),r["intent"]?.ToObject<IntentState>());break;
                        case "select_plans":data=Planner.Choose(r["plans"].ToObject<System.Collections.Generic.List<Plan>>(),Profile.Read((JObject)r["profile"]));break;
                        case "rollout_choice":data=Planner.MaskedRollout((JObject)r["observation"],r["choices"].ToObject<System.Collections.Generic.List<Choice>>(),(string)r["style"]??"balanced");break;
                        case "evaluate":data=Evaluator.Evaluate((JObject)r["observation"],(int)r["seat"]);break;
                        case "run":data=Harness.Run(r,composition);break;
                        default:throw new Exception("unknown_operation");
                    }
                    Console.WriteLine(Wire.Json(new{id,ok=true,data}));
                }
                catch(Exception ex){Console.WriteLine(Wire.Json(new{id,ok=false,error=ex.GetType().Name,message=ex.Message,detail=ex.StackTrace}));}
            }
            return 0;
        }
    }
}
