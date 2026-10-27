using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using BrassHost;
using BrassLegacyAI;
using BrassLegacyAI.Native;
using BrassLegacyAI.Execution;
using Newtonsoft.Json.Linq;
using UnityEngine;

// Runs in the actual Unity Mono player; baseline methods are unchanged in the
// candidate assembly. Both arms use identical complete action paths and worlds.
public sealed class ValidationCandidateDriver : MonoBehaviour
{
    readonly JArray rows=new JArray(),checks=new JArray();
    string output;int count;
    static double Ms(long ticks) {return ticks*1000.0/Stopwatch.Frequency;}
    void Check(bool okay,string name) {count++;if(!okay)throw new Exception(name);checks.Add(name);}
    static string State(Host host) {return Wire.Json(host.Canonical());}
    static JObject View(Host host) {return (JObject)Wire.Token(host.Observe(host.Actor));}
    static JObject Composition(Host host) {return Host.Composition(Host.Configuration(host.Game.GameStartConfiguration));}
    IEnumerator Start() {
        output=Environment.GetEnvironmentVariable("BRASS_L14_RESULT");
        QualitySettings.vSyncCount=0;Application.targetFrameRate=60;
        var work=Cases();Exception error=null;var cpu=Process.GetCurrentProcess().TotalProcessorTime;
        while(true) {
            bool more=false;object value=null;
            try {more=work.MoveNext();if(more)value=work.Current;}catch(Exception e){error=e;}
            if(error!=null || !more)break;yield return value;
        }
        File.WriteAllText(output,new JObject{["passed"]=error==null,["error"]=error?.ToString(),["checks"]=count,["cases"]=checks,
            ["cpu_ms"]=(Process.GetCurrentProcess().TotalProcessorTime-cpu).TotalMilliseconds,["rows"]=rows,
            ["allocation_measurement"]="Unity Mono lacks GetAllocatedBytesForCurrentThread; separate instrumented profile required"}.ToString());
        UnityEngine.Application.Quit(error==null?0:1);
    }
    IEnumerator Cases() {
        var fixtures=JArray.Parse(File.ReadAllText(Environment.GetEnvironmentVariable("BRASS_L14_FIXTURES")));
        int reps=Int32.Parse(Environment.GetEnvironmentVariable("BRASS_L14_REPETITIONS"));
        foreach(JObject f in fixtures) {
            for(int r=-2;r<reps;r++) {
                var results=new Dictionary<string,List<string>>();
                foreach(string arm in r%2==0?new[]{"baseline","candidate"}:new[]{"candidate","baseline"}) {
                    var h=ValidationBench.Root((JObject)f["input"]);var composition=Composition(h);results[arm]=new List<string>();int ordinal=0;
                    foreach(JArray a in f["actions"]) {
                        yield return null;
                        string[] path=a.Values<string>().ToArray();long started=Stopwatch.GetTimestamp();
                        var cpu=Process.GetCurrentProcess().TotalProcessorTime;double snapshot=0,commit=0,queue=0,execution=0,maxFrame=0;int frames=0;
                        if(arm=="baseline") {
                            var time=Stopwatch.StartNew();h.ValidatePersonalityAction(path,composition);h.CommitPersonalityAction(path,composition);commit=time.Elapsed.TotalMilliseconds;
                        }else {
                            var time=Stopwatch.StartNew();var validation=new NativeValidation(View(h),composition,path,"fixture-"+f["name"]+"-"+ordinal);
                            snapshot=time.Elapsed.TotalMilliseconds;
                            var job=AiExecutionScheduler.Submit(validation);long previous=Stopwatch.GetTimestamp();
                            while(!job.Completion.IsCompleted) {yield return null;long now=Stopwatch.GetTimestamp();maxFrame=Math.Max(maxFrame,Ms(now-previous));previous=now;frames++;}
                            Check(job.Failure==null && validation.Validated,"detached validation completed");
                            queue=job.QueueMilliseconds;execution=job.ExecutionMilliseconds;
                            time.Restart();validation.Commit(h,composition,path,"fixture-"+f["name"]+"-"+ordinal,false);commit=time.Elapsed.TotalMilliseconds;
                            Check(job.ResourcesReleased,"validation resources retired before commit");
                        }
                        double elapsed=Ms(Stopwatch.GetTimestamp()-started);
                        if(r>=0)rows.Add(new JObject{["fixture"]=f["name"],["partition"]=f["partition"],["era"]=f["input"]["view"]["context"]["era"],
                            ["repeat"]=r,["action"]=ordinal,["arm"]=arm,["snapshot_ms"]=snapshot,["queue_ms"]=queue,["background_ms"]=execution,
                            ["main_commit_ms"]=commit,["total_ms"]=elapsed,["frames"]=frames,["max_wait_frame_ms"]=maxFrame,
                            ["cpu_ms"]=(Process.GetCurrentProcess().TotalProcessorTime-cpu).TotalMilliseconds});
                        results[arm].Add(State(h));ordinal++;
                    }
                }
                Check(results["baseline"].SequenceEqual(results["candidate"]),"identical committed state/events after each action");
            }
            File.WriteAllText(output+".progress.json",new JObject{["fixture"]=f["name"],["checks"]=count,["rows"]=rows.Count}.ToString());
        }
        // Negative paths are compared with the production input gate plus the
        // unchanged baseline validator, not a weaker substitute validator.
        foreach(JObject f in fixtures) {
            var legal=((JArray)f["actions"][0]).Values<string>().ToArray();
            foreach(var path in new[]{legal,legal.Take(legal.Length-1).ToArray(),new[]{"invalid"},legal.Concat(legal).ToArray(),new string[0],null,new[]{(string)null}}) {
                var root=ValidationBench.Root((JObject)f["input"]);var composition=Composition(root);string before=State(root);bool original=true,candidate=true;
                try{NativeInput.Actions(new JObject{["actions"]=new JArray(path==null?JValue.CreateNull():(JToken)new JArray(path))});root.ValidatePersonalityAction(path,composition);}catch{original=false;}
                AiExecutionJob checking=null;
                try{var v=new NativeValidation(View(root),composition,path,"identity");checking=AiExecutionScheduler.Submit(v);}catch{candidate=false;}
                if(checking!=null){while(!checking.Completion.IsCompleted)yield return null;candidate=checking.Failure==null;}
                Check(original==candidate && State(root)==before,"baseline acceptance/rejection parity without mutation");
            }
            foreach(string scenario in new[]{"identity","path","composition","revision","actor","state","boundary","cancelled","reuse","load"}) {
                var root=ValidationBench.Root((JObject)f["input"]);var composition=Composition(root);string before=State(root);
                var v=new NativeValidation(View(root),composition,legal,"identity");var j=AiExecutionScheduler.Submit(v);while(!j.Completion.IsCompleted)yield return null;
                Check(j.Failure==null,"negative fixture validation succeeds");
                var current=root;var cp=composition;var path=legal;string identity="identity";bool cancelled=false;
                if(scenario=="identity" || scenario=="load")identity="new-generation";
                if(scenario=="path")path=new[]{"invalid"};
                if(scenario=="composition"){cp=(JObject)cp.DeepClone();cp[cp.Properties().First().Name]=(int)cp.Properties().First().Value+1;}
                if(scenario=="revision" || scenario=="state" || scenario=="boundary")root.SearchApply(legal[0]);
                if(scenario=="actor")root.Game.GameProgressInformation.CurrentPlayer=root.Seats.First(p=>(int)p.Color!=root.Actor);
                if(scenario=="cancelled")cancelled=true;
                if(scenario=="reuse")v.Commit(root,composition,legal,"identity",false);
                before=State(root);bool rejected=false;
                try{v.Commit(current,cp,path,identity,cancelled);}catch(InvalidOperationException){rejected=true;}
                Check(rejected && State(root)==before,"stale/cancelled receipt rejects: "+scenario);
            }
        }
        // A queued cancellation does not run the validator. The scheduler keeps
        // both queued and running work owned until terminal retirement.
        var first=(JObject)fixtures[0];var host=ValidationBench.Root((JObject)first["input"]);var valid=((JArray)first["actions"][0]).Values<string>().ToArray();
        var blocker=new TaskCompletionSource<int>();
        var blocked=AiExecutionScheduler.Submit(new DelegateAiExecutionBackend("test barrier",t=>blocker.Task));
        var pending=new NativeValidation(View(host),Composition(host),valid,"queue");var queued=AiExecutionScheduler.Submit(pending);queued.Cancel();queued.Cancel();
        Check(!queued.Completion.IsCompleted && !AiExecutionScheduler.IsIdle,"cancelled queued work retains scheduler ownership");
        blocker.SetResult(0);while(!queued.Completion.IsCompleted)yield return null;
        Check(queued.CancellationRequested && queued.ResourcesReleased && !pending.Validated,"queued cancellation executes no validation");
        using(var source=new CancellationTokenSource()) {
            var v=new NativeValidation(View(host),Composition(host),valid,"cancel");
            v.ExecuteAsync(source.Token);source.Cancel();string before=State(host);bool rejected=false;
            try{v.Commit(host,Composition(host),valid,"cancel",false);}catch(InvalidOperationException){rejected=true;}
            Check(rejected && State(host)==before,"cancellation after validation before commit");
        }
        Check(AiExecutionScheduler.IsIdle,"all validation work retired");
    }
}
#if VALIDATION_CANDIDATE
public static class LocalAiUnityTests {
    public static void Begin(string output) {UnityEngine.Object.DontDestroyOnLoad(new GameObject("L14 validation comparison").AddComponent<ValidationCandidateDriver>().gameObject);}
}
#endif
