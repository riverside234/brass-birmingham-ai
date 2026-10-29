// Bounded oracle: exhaustive selected-choice membership on recorded roots,
// staged paths, unchanged observations/events, rejected IDs and clock journals.
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Threading;
using System.Threading.Tasks;
using BrassHost;
using BrassLegacyAI;
using BrassLegacyAI.Native;
using BrassLegacyAI.Execution;
using BrassPatch.Persistence;
using BrassApplication.AI;
using BrassApplication.Game;
using Newtonsoft.Json.Linq;

public static class HostDeliveryChecks
{
    static int checks;
    static void Check(bool okay,string message){checks++;if(!okay)throw new Exception(message);}
    static readonly MethodInfo resolver=typeof(Host).GetMethod("ResolveChoice");
    static readonly MethodInfo observation=typeof(Host).GetMethod("ObserveToken");
    static readonly MethodInfo clockView=typeof(NativeInput).GetMethod("ClockView");
    static Choice Resolve(Host h,string id){return resolver==null?h.Choices().Single(c=>c.id==id):(Choice)resolver.Invoke(h,new object[]{id});}
    static JObject View(Host h){return observation==null?(JObject)Wire.Token(h.Observe(h.Actor)):(JObject)observation.Invoke(h,new object[]{h.Actor});}
    static JArray Events(Host h,int revision){return new JArray(h.Game.GetUncommittedEvents().Where(e=>e.Metadata.Revision>revision).Select(e=>{
        var payload=JObject.FromObject(e);payload.Remove("Metadata");return new JObject{["type"]=e.GetType().FullName,["revision"]=e.Metadata.Revision,["payload"]=payload};}));}
    static void Oracle(Host h,JArray roots,string name)
    {
        var legal=h.Choices();string before=Wire.Json(h.Canonical());
        foreach(var choice in legal)Check(Wire.Json(resolver==null?choice:Resolve(h,choice.id))==Wire.Json(choice),"selected membership: "+choice.kind);
        foreach(string invalid in new[]{"", "not-an-id", "finish|||||bad", "loan|||location:Nottingham|illegal|", "build|unknown:999|IronWorks|wild:location||", "iron|unknown||||", "bonus_skip|||industry:CoalMine||"}) {
            bool canonical=legal.Any(c=>c.id==invalid),accepted=canonical;
            if(resolver!=null){accepted=true;try{Resolve(h,invalid);}catch{accepted=false;}}
            Check(accepted==canonical,"invalid selected choice agrees with complete enumeration");
        }
        var direct=View(h);var original=(JObject)Wire.Token(h.Observe(h.Actor));
        Check(JToken.DeepEquals(original,direct) && Wire.Json(original)==Wire.Json(direct),"direct observation preserves exact token/wire contents");
        Check(before==Wire.Json(h.Canonical()),"membership/observation checks never mutate state");
        Check(h.Links.Select(l=>l.LinkId).SequenceEqual(h.Game.Board.Regions.SelectMany(r=>r.Links).Distinct().OrderBy(l=>l.LinkId,StringComparer.Ordinal).Select(l=>l.LinkId)),"board index equals current topology");
        roots.Add(new JObject{["name"]=name,["choices"]=Wire.Token(legal),["observation"]=direct});
    }
    static void Journal(Host h,JArray rows)
    {
        var metadata=new GameMetadata(h.Game.GameStartConfiguration,GameType.Local){GameId="clock-oracle"};
        foreach(var start in h.Game.GameStartConfiguration.Players)metadata.JoinGame(new PlayerMetadata{PlayerStartConfiguration=start,AiType=AIType.Hard,NickName="Alpha Test",PlayerNetworkId="local-"+start.Color});
        var app=new BrassApplicationGame(h.Game,metadata);var state=NativeGameState.Create(metadata,p=>"easy");state["ready"]=true;
        var full=h.ObservePersonality(h.Actor);JObject small=clockView==null?full:(JObject)clockView.Invoke(null,new object[]{app,h.Actor});
        // A detached fixture's semantic revision can differ from vendor events.
        // ClockView deliberately binds the current vendor event revision.
        full["context"]["revision"]=app.Game.Revision;
        var recall=new JArray(new JObject{["revision"]=app.Game.Revision,["hand"]=full["own_hand"].DeepClone()});
        SaveExtensions.Set(metadata,(JObject)state.DeepClone());NativeGameState.Record(metadata,full,"history",1234,recall);
        var expected=SaveExtensions.Get(metadata).DeepClone();
        SaveExtensions.Set(metadata,(JObject)state.DeepClone());NativeGameState.Record(metadata,small,"history",1234,recall);
        Check(JToken.DeepEquals(expected,SaveExtensions.Get(metadata)),"minimal journal equals full snapshot journal");
        rows.Add(expected);
    }
    public static int Main(string[] args)
    {
        Exception error=null;var roots=new JArray();var results=new JArray();var timings=new JArray();var journals=new JArray();
        try {
            foreach(JObject fixture in JArray.Parse(File.ReadAllText(args[0]))) {
                var h=ValidationBench.Root((JObject)fixture["input"]);int ordinal=0;
                Oracle(h,roots,(string)fixture["name"]+"-root");
                foreach(JArray ids in fixture["actions"]) {
                    var path=ids.Values<string>().ToArray();var composition=Host.Composition(Host.Configuration(h.Game.GameStartConfiguration));
                    var validation=new NativeValidation(View(h),composition,path,"oracle");
                    validation.ExecuteAsync(CancellationToken.None).GetAwaiter().GetResult();
                    int revision=h.Game.Revision;validation.Commit(h,composition,path,"oracle",false);
                    results.Add(new JObject{["fixture"]=fixture["name"],["action"]=ordinal++,["state"]=Wire.Token(h.Canonical()),["events"]=Events(h,revision)});
                    Journal(h,journals);
                    if(h.Remaining>0)Oracle(h,roots,(string)fixture["name"]+"-second");
                }
                var staged=ValidationBench.Root((JObject)fixture["input"]);int step=0;
                foreach(JArray ids in fixture["actions"])foreach(string id in ids.Values<string>()) {
                    if(!staged.ActionComplete)Oracle(staged,roots,(string)fixture["name"]+"-stage-"+step);
                    staged.SearchApply(id);step++;
                }
                Check(Wire.Json(staged.Canonical())==Wire.Json(h.Canonical()),"staged oracle equals guarded committed path");
                // Warm once, then two identical fixed-reply passes. Root setup,
                // assertions and reporting are outside the measured action spans.
                for(int repeat=-1;repeat<2;repeat++) {
                    var timed=ValidationBench.Root((JObject)fixture["input"]);int action=0;
                    var cp=Host.Composition(Host.Configuration(timed.Game.GameStartConfiguration));
                    foreach(JArray ids in fixture["actions"]) {
                        string[] path=ids.Values<string>().ToArray();var timer=Stopwatch.StartNew();
                        var validation=new NativeValidation(View(timed),cp,path,"fixed");double snapshot=timer.Elapsed.TotalMilliseconds;
                        timer.Restart();var job=AiExecutionScheduler.Submit(validation);job.Completion.GetAwaiter().GetResult();
                        double background=timer.Elapsed.TotalMilliseconds;Check(job.Failure==null,"fixed validation succeeds");
                        timer.Restart();validation.Commit(timed,cp,path,"fixed",false);double commit=timer.Elapsed.TotalMilliseconds;
                        if(repeat>=0)timings.Add(new JObject{["fixture"]=fixture["name"],["repeat"]=repeat,["action"]=action,["snapshot_ms"]=snapshot,["background_ms"]=background,["commit_ms"]=commit,["total_ms"]=snapshot+background+commit});
                        action++;
                    }
                }
            }
        }catch(Exception e){error=e;}
        File.WriteAllText(args[1],new JObject{["passed"]=error==null,["checks"]=checks,["error"]=error?.ToString(),
            ["optimized_resolver"]=resolver!=null,["roots"]=roots,["results"]=results,["journals"]=journals,["timings"]=timings}.ToString());
        if(error!=null)Console.Error.WriteLine(error);return error==null?0:1;
    }
}
