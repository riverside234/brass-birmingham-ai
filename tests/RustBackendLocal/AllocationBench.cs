using System;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;
using BrassHost;
using BrassLegacyAI;
using BrassLegacyAI.Native;
using BrassLegacyAI.Execution;
using Newtonsoft.Json.Linq;

public static class AllocationBench
{
    [DllImport("allocation-host.exe",CallingConvention=CallingConvention.Cdecl)]static extern long brass_bytes();
    [DllImport("allocation-host.exe",CallingConvention=CallingConvention.Cdecl)]static extern long brass_objects();
    static long Count(Action work){long before=brass_bytes();work();return brass_bytes()-before;}
    static void SelfTest(){long n=brass_objects(),b=brass_bytes();var a=new byte[100][];for(int i=0;i<100;i++)a[i]=new byte[1024];
        if(brass_objects()-n<101 || brass_bytes()-b<102400)throw new Exception("allocation self-test failed");GC.KeepAlive(a);}
    public static int Main(string[] args){try{
        SelfTest();Task.Run((Action)SelfTest).GetAwaiter().GetResult();var rows=new JArray();
        foreach(JObject f in JArray.Parse(File.ReadAllText(args[0])))for(int r=-1;r<2;r++) {
            var states=new string[2];int armIndex=0;
            foreach(string arm in r%2==0?new[]{"baseline","candidate"}:new[]{"candidate","baseline"}) {
                var h=ValidationBench.Root((JObject)f["input"]);var cp=Host.Composition(Host.Configuration(h.Game.GameStartConfiguration));int ordinal=0;
                foreach(JArray action in f["actions"]) {
                    string[] path=action.Values<string>().ToArray();var counts=new JObject();JObject view=null;NativeValidation validation=null;
                    if(arm=="baseline") {
                        counts["full_validation"]=Count(()=>h.ValidatePersonalityAction(path,cp));
                        counts["commit_with_duplicate_validations"]=Count(()=>h.CommitPersonalityAction(path,cp));
                    }else {
                        counts["snapshot"]=Count(()=>{view=(JObject)Wire.Token(h.Observe(h.Actor));validation=new NativeValidation(view,cp,path,"allocation");});
                        long background=0;var job=AiExecutionScheduler.Submit(new DelegateAiExecutionBackend("allocation diagnostic",token=>{
                            background=Count(()=>validation.ExecuteAsync(token).GetAwaiter().GetResult());return Task.FromResult(0);}));
                        job.Completion.GetAwaiter().GetResult();if(job.Failure!=null || !job.ResourcesReleased)throw new Exception(job.Failure??"not retired");
                        counts["background_validation"]=background;
                        counts["guarded_commit"]=Count(()=>validation.Commit(h,cp,path,"allocation",false));
                    }
                    if(r>=0)rows.Add(new JObject{["fixture"]=f["name"],["era"]=f["input"]["view"]["context"]["era"],["repeat"]=r,["action"]=ordinal,["arm"]=arm,["bytes"]=counts});ordinal++;
                }
                states[armIndex++]=Wire.Json(h.Canonical());
            }
            if(states[0]!=states[1])throw new Exception("allocation arms changed final state");
        }
        File.WriteAllText(args[1],new JObject{["passed"]=true,["counter_self_test_main_and_worker"]=true,["rows"]=rows,
            ["scope"]="Exact shipped Mono and candidate managed assemblies in isolated embedding host; actual thread-local object bytes. Excludes Unity presentation, scheduler overhead and ORT/native allocations."}.ToString());return 0;
    }catch(Exception e){Console.Error.WriteLine(e);return 1;}}
}
