using System;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using BrassHost;
using BrassLegacyAI.Native;
using Newtonsoft.Json.Linq;

// Additional non-timed checks: compare actual emitted event payloads as well as
// canonical state. Event metadata contains wall-clock/instance stamps, so keep
// its revision explicitly and compare every other payload field and event type.
public static class NativeValidationContractChecks
{
    static int checks;
    static void Check(bool value,string reason){checks++;if(!value)throw new Exception(reason);}
    static JObject View(Host h){return (JObject)Wire.Token(h.Observe(h.Actor));}
    static JObject Composition(Host h){return Host.Composition(Host.Configuration(h.Game.GameStartConfiguration));}
    static JArray Events(Host h,int after){return new JArray(h.Game.GetUncommittedEvents().Where(e=>e.Metadata.Revision>after).Select(e=>{
        var payload=JObject.FromObject(e);payload.Remove("Metadata");
        return new JObject{["type"]=e.GetType().FullName,["revision"]=e.Metadata.Revision,["payload"]=payload};
    }));}
    static string Bound(string[] values,JObject reply){return NativeValidation.Binding(values[0],values[1],values[2],Int64.Parse(values[3]),Int64.Parse(values[4]),Int32.Parse(values[5]),Int32.Parse(values[6]),Int32.Parse(values[7]),values[8],reply);}
    public static int Main(string[] args)
    {
        Exception failure=null;var evidence=new JArray();
        try {
            foreach(JObject fixture in JArray.Parse(File.ReadAllText(args[0]))) {
                var baseline=ValidationBench.Root((JObject)fixture["input"]);var candidate=ValidationBench.Root((JObject)fixture["input"]);
                int action=0;
                foreach(JArray ids in fixture["actions"]) {
                    var path=ids.Values<string>().ToArray();int oldRevision=baseline.Game.Revision;
                    baseline.ValidatePersonalityAction(path,Composition(baseline));baseline.CommitPersonalityAction(path,Composition(baseline));
                    var validated=new NativeValidation(View(candidate),Composition(candidate),path,"events-"+action);
                    Task.Run(()=>validated.ExecuteAsync(CancellationToken.None)).GetAwaiter().GetResult();
                    validated.Commit(candidate,Composition(candidate),path,"events-"+action,false);
                    var expected=Events(baseline,oldRevision);var actual=Events(candidate,oldRevision);
                    Check(expected.Count>0 && JToken.DeepEquals(expected,actual),"emitted event type/order/revision/payload parity");
                    Check(Wire.Json(baseline.Canonical())==Wire.Json(candidate.Canonical()),"canonical state parity");
                    evidence.Add(new JObject{["fixture"]=fixture["name"],["action"]=action++,["events"]=actual});
                }
            }
            var first=(JObject)JArray.Parse(File.ReadAllText(args[0]))[0];
            var h=ValidationBench.Root((JObject)first["input"]);var legal=((JArray)first["actions"][0]).Values<string>().ToArray();
            var values=new[]{"lifetime","instance","game","11","12","0",h.Game.Revision.ToString(),h.Actor.ToString(),h.Phase};
            var reply=new JObject{["ticket"]=new JObject{["source_sha256"]="source",["root_sha256"]="root"},["backend"]="backend",["model_id"]="model",["export_sha256"]="export"};
            string binding=Bound(values,reply);var receipt=new NativeValidation(View(h),Composition(h),legal,binding);
            Task.Run(()=>receipt.ExecuteAsync(CancellationToken.None)).GetAwaiter().GetResult();
            string before=Wire.Json(h.Canonical());
            Action<string> reject=changed=>{
                bool refused=false;try{receipt.Commit(h,Composition(h),legal,changed,false);}catch(InvalidOperationException){refused=true;}
                Check(refused && before==Wire.Json(h.Canonical()),"changed full identity rejects without mutation");
            };
            for(int i=0;i<values.Length;i++) {
                var changed=(string[])values.Clone();changed[i]=(i>=3 && i<=7)?(Int64.Parse(changed[i])+1).ToString():changed[i]+"-changed";
                reject(Bound(changed,reply));
            }
            foreach(string key in new[]{"backend","model_id","export_sha256"}) {var changed=(JObject)reply.DeepClone();changed[key]="changed";reject(Bound(values,changed));}
            foreach(string key in new[]{"source_sha256","root_sha256"}) {var changed=(JObject)reply.DeepClone();changed["ticket"][key]="changed";reject(Bound(values,changed));}
            bool wrongThread=false;
            try {Task.Run(()=>receipt.Commit(h,Composition(h),legal,binding,false)).GetAwaiter().GetResult();}catch(InvalidOperationException){wrongThread=true;}
            Check(wrongThread && before==Wire.Json(h.Canonical()),"a validated receipt cannot commit from a worker thread");
        } catch(Exception error){failure=error;}
        File.WriteAllText(args[1],new JObject{["passed"]=failure==null,["checks"]=checks,["error"]=failure?.ToString(),["events"]=evidence,
            ["event_metadata_comparison"]="Revision retained; wall-clock/instance metadata excluded. Every semantic payload field and event type/order compared."}.ToString());
        if(failure!=null)Console.Error.WriteLine(failure);return failure==null?0:1;
    }
}
