// L14 fixed-reply probe: the same public snapshots/actions on the sealed L13
// validator. Instrumentation is outside the baseline assembly.
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading;
using BrassHost;
using BrassLegacyAI;
using BrassApplication.Game;
using Newtonsoft.Json.Linq;

public static class ValidationBench
{
    static readonly Func<long> Allocated = AllocationCounter();
    static Func<long> AllocationCounter() {
        var m=typeof(GC).GetMethod("GetAllocatedBytesForCurrentThread",Type.EmptyTypes);
        return m==null ? (Func<long>)null : (Func<long>)Delegate.CreateDelegate(typeof(Func<long>),m);
    }
    static double Measure(Action action,out long? allocated) {
        long before=Allocated==null?0:Allocated();var t=Stopwatch.StartNew();action();t.Stop();
        allocated=Allocated==null?(long?)null:Allocated()-before;return t.Elapsed.TotalMilliseconds;
    }
    static double Median(IEnumerable<double> x) {var a=x.OrderBy(v=>v).ToArray();return a[a.Length/2];}
    static double P95(IEnumerable<double> x) {var a=x.OrderBy(v=>v).ToArray();return a[(int)Math.Ceiling(a.Length*.95)-1];}
    public static Host Root(JObject input) {
        var v=(JObject)input["view"].DeepClone();var config=Host.Configuration(new BoardGameRules.Entities.Game.GameStartConfiguration((BoardGameRules.Entities.Common.NumberOfPlayers)4){Seed=1,IsIntroductoryGame=false});
        v["protocol"]=v["context"]["protocol"]=Wire.Version;v["rules"]=v["context"]["rules"]=Wire.Rules;
        v["configuration"]=config;v["context"]["player_count"]=4;v["context"]["introductory"]=false;v["context"]["round_limit"]=8;
        return new Host(1,Belief.Setup(v,Host.Composition(config),1));
    }
    public static void Run(string fixtures,string output,int repetitions) {
        var rows=new JArray();Exception failure=null;
        var cpu=Process.GetCurrentProcess().TotalProcessorTime;
        try {
            foreach(JObject f in JArray.Parse(File.ReadAllText(fixtures))) {
                for(int r=-2;r<repetitions;r++) {
                    var h=Root((JObject)f["input"]);var composition=Host.Composition(Host.Configuration(h.Game.GameStartConfiguration));
                    int actionIndex=0;
                    foreach(JArray a in f["actions"]) {
                        string[] path=a.Values<string>().ToArray();var stages=new JObject();var allocations=new JObject();long? bytes;
                        JObject view=null;Host trial=null;int thread=Thread.CurrentThread.ManagedThreadId;
                        stages["observation_ms"]=Measure(()=>view=(JObject)Wire.Token(h.Observe(h.Actor)),out bytes);allocations["observation"]=bytes;
                        stages["detached_copy_ms"]=Measure(()=>view=(JObject)view.DeepClone(),out bytes);allocations["detached_copy"]=bytes;
                        stages["trial_reconstruction_ms"]=Measure(()=>trial=new Host(1,Belief.Setup(view,composition,1)),out bytes);allocations["trial_reconstruction"]=bytes;
                        stages["semantic_replay_ms"]=Measure(()=>{for(int i=0;i<path.Length;i++){trial.SearchApply(path[i]);if(trial.ActionComplete&&i<path.Length-1)throw new Exception("multiple_actions");}if(!trial.ActionComplete)throw new Exception("incomplete_action");},out bytes);allocations["semantic_replay"]=bytes;
                        var full=new JArray();
                        for(int i=0;i<3;i++)full.Add(Measure(()=>h.ValidatePersonalityAction(path,composition),out bytes));
                        stages["three_full_validations_ms"]=full;stages["full_validation_total_ms"]=full.Values<double>().Sum();
                        // Exercise the actual existing commit call chain, including its
                        // two nested full validations. Separate full timings above.
                        stages["baseline_commit_with_two_validations_ms"]=Measure(()=>h.CommitPersonalityAction(path,composition),out bytes);allocations["baseline_commit"]=bytes;
                        if(r>=0)rows.Add(new JObject{["fixture"]=f["name"],["partition"]=f["partition"],["era"]=f["input"]["view"]["context"]["era"],
                            ["repeat"]=r,["action"]=actionIndex,["thread"]=thread,["stages"]=stages,["allocated_bytes"]=allocations});
                        actionIndex++;
                    }
                }
            }
        }catch(Exception e){failure=e;}
        var result=new JObject{["passed"]=failure==null,["error"]=failure?.ToString(),["runtime"]=Environment.Version.ToString(),
            ["allocations_available"]=Allocated!=null,["cpu_ms"]=(Process.GetCurrentProcess().TotalProcessorTime-cpu).TotalMilliseconds,["rows"]=rows};
        if(rows.Count>0)result["summary"]=new JObject{["actions"]=rows.Count,["validation_median_ms"]=Median(rows.Select(r=>(double)r["stages"]["full_validation_total_ms"])),
            ["validation_p95_ms"]=P95(rows.Select(r=>(double)r["stages"]["full_validation_total_ms"]))};
        File.WriteAllText(output,result.ToString());if(failure!=null)throw failure;
    }
    public static int Main(string[] args) {try{Run(args[0],args[1],Int32.Parse(args[2]));return 0;}catch(Exception e){Console.Error.WriteLine(e);return 1;}}
}
#if UNITY_PROBE && !VALIDATION_CANDIDATE
public static class LocalAiUnityTests {
    public static void Begin(string output) {
        try {ValidationBench.Run(Environment.GetEnvironmentVariable("BRASS_L14_FIXTURES"),Environment.GetEnvironmentVariable("BRASS_L14_RESULT"),Int32.Parse(Environment.GetEnvironmentVariable("BRASS_L14_REPETITIONS")));}
        catch(Exception e){UnityEngine.Debug.LogException(e);}
        UnityEngine.Application.Quit();
    }
}
#endif
