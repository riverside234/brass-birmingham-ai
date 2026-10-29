// Independent C# rules replay of both native actions and the new boundary guard.
using System;
using System.IO;
using System.Linq;
using System.Diagnostics;
using BrassHost;
using BrassLegacyAI.Native;
using Newtonsoft.Json.Linq;

public static class SearchV2ValidationTests
{
    static int checks;
    static void Check(bool condition,string text){checks++;if(!condition)throw new Exception(text);}
    public static int Main(string[] args)
    {
        var rows=new JArray();
        try {
            foreach(JObject fixture in JArray.Parse(File.ReadAllText(args[0]))) {
                var input=(JObject)fixture["input"];var response=(JObject)fixture["response"];
                var host=ValidationBench.Root(input);int actor=host.Actor;
                var composition=Host.Composition(Host.Configuration(host.Game.GameStartConfiguration));
                var paths=NativeInput.Actions(response);var clock=Stopwatch.StartNew();
                int ordinal=0;
                foreach(var path in paths) {
                    host.ValidatePersonalityAction(path,composition);
                    host.CommitPersonalityAction(path,composition);ordinal++;
                    if(ordinal==1 && paths.Length==2 && (bool?)response["search_revision"]?["focus"]?["switched"]==true) {
                        var actual=(JObject)input.DeepClone();actual["view"]=host.ObservePersonality(actor);
                        // SearchApply confirms Finish without journaling it. The
                        // live CapturePersonality event adapter does journal it
                        // (P6RecordAction). Supply that confirmed marker only in
                        // this imported-state probe; the Unity test uses events.
                        Check(path.Last()=="finish|||||" && host.ActionComplete,"confirmed boundary");
                        ((JArray)actual["view"]["history"]).Add(new JObject{["seat"]=actor,
                            ["era"]=input["view"]["context"]["era"].DeepClone(),
                            ["choice"]=new JObject{["id"]="finish|||||",["kind"]="finish"}});
                        var recall=(JArray)input["recall"].DeepClone();
                        if(recall.Count==0)recall.Add(new JObject{["revision"]=input["view"]["context"]["revision"],["hand"]=input["view"]["own_hand"].DeepClone()});
                        bool matches=NativeInput.ContinuationMatches(response,actual,recall);
                        if(!matches)File.WriteAllText(args[1]+".boundary.json",new JObject{["actual"]=actual,["recall"]=recall,["expected"]=response["search_revision"]["focus"]["permitted_successor"]}.ToString());
                        Check(matches,"projected/actual C# successor: "+fixture["name"]);
                        foreach(string mutation in new[]{"cash","hand","actor","revision","history","ledger","recall"}) {
                            var bad=(JObject)actual.DeepClone();var badRecall=(JArray)recall.DeepClone();
                            if(mutation=="cash")bad["view"]["players"][actor]["cash"]=(int)bad["view"]["players"][actor]["cash"]+1;
                            if(mutation=="hand")((JArray)bad["view"]["own_hand"]).RemoveAt(0);
                            if(mutation=="actor")bad["view"]["context"]["actor"]=(actor+1)%4;
                            if(mutation=="revision")bad["view"]["context"]["revision"]=response["ticket"]["revision"].DeepClone();
                            if(mutation=="history")((JArray)bad["view"]["history"]).RemoveAt(0);
                            if(mutation=="ledger")bad["ledger"]["spent"][actor]=.075;
                            if(mutation=="recall")badRecall[0]["revision"]=(int)badRecall[0]["revision"]+1;
                            Check(!NativeInput.ContinuationMatches(response,bad,badRecall),"reject changed "+mutation);
                        }
                    }
                }
                Check(paths.Length<2 || host.Remaining==0,"two complete actions exhaust turn");
                var outcome=host.ObservePersonality(actor);var player=outcome["players"][actor];
                rows.Add(new JObject{["name"]=fixture["name"],["actions"]=paths.Length,["elapsed_ms"]=clock.Elapsed.TotalMilliseconds,
                    ["cash"]=player["cash"].DeepClone(),["income"]=player["income"].DeepClone(),["vp"]=player["vp"].DeepClone(),
                    ["income_marker"]=player["income_marker"].DeepClone(),["link_tiles"]=player["link_tiles"].DeepClone(),
                    ["remaining_hand"]=outcome["own_hand"].DeepClone(),["mat"]=player["mat"].DeepClone(),
                    ["own_tiles"]=new JArray(outcome["slots"].Where(s=>(int?)s["tile"]?["seat"]==actor).Select(s=>s.DeepClone()))});
            }
            File.WriteAllText(args[1],new JObject{["passed"]=true,["checks"]=checks,["rows"]=rows}.ToString());return 0;
        }catch(Exception e){File.WriteAllText(args[1],new JObject{["passed"]=false,["checks"]=checks,["error"]=e.ToString(),["rows"]=rows}.ToString());Console.Error.WriteLine(e);return 1;}
    }
}
