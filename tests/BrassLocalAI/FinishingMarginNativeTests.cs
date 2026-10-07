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
using Newtonsoft.Json.Linq;
using UnityEngine;

// Loaded only by the existing isolated --brass-local-ai-test entry point.
public static class LocalAiUnityTests
{
    public static void Begin(string output)
    { var go=new GameObject("Game search deadline checks");UnityEngine.Object.DontDestroyOnLoad(go);go.AddComponent<FinishingMarginDriver>().Output=output; }
}
public sealed class FinishingMarginDriver:MonoBehaviour
{
    public string Output;
    readonly List<string> errors=new List<string>();
    IEnumerator Start()
    {
        QualitySettings.vSyncCount=0;Application.targetFrameRate=30;
        Application.logMessageReceived+=(message,stack,type)=>{if(type==LogType.Error || type==LogType.Exception)lock(errors)errors.Add(message+" "+stack);};
        var job=Task.Run(()=>FinishingMarginMeasurements.Run(Path.GetDirectoryName(Output)));
        while(!job.IsCompleted)yield return null;
        JObject report;
        try{report=job.Result;}catch(Exception e){report=new JObject{["passed"]=false,["error"]=e.ToString()};}
        lock(errors){report["native_errors"]=JArray.FromObject(errors);if(errors.Count>0)report["passed"]=false;}
        File.WriteAllText(Output,report.ToString());Application.Quit((bool)report["passed"]?0:1);
    }
}
public static class FinishingMarginMeasurements
{
    static int checks;
    static void Check(bool value,string reason){checks++;if(!value)throw new Exception(reason);}
    static void Validate(Plan plan,JObject view,JObject cards)
    {
        Check(plan!=null && plan.actions.Count==Math.Min(2,(int)view["context"]["actions_remaining"]),"complete whole-turn plan");
        var h=new Host(1,Belief.Setup(view,cards,17));
        foreach(var action in plan.actions){foreach(var id in action){Check(h.Choices().Any(c=>c.id==id),"fallback/current plan is legal");h.SearchApply(id);}Check(h.ActionComplete,"action complete");}
        Check(h.Remaining==0,"both actions and subchoices complete");
    }
    public static JObject Run(string directory)
    {
#if FINISHING_MARGIN
        Check(TurnCoordinator.SearchWorkers==2,"two game workers unchanged");
        foreach(var setting in new[]{new[]{2000.0,250.0},new[]{5000.0,300.0},new[]{12000.0,400.0}})
            Check(TurnFinishingBudget.MarginMilliseconds(setting[0])==setting[1],"measured setting-specific margin");
        Check(TurnFinishingBudget.MarginMilliseconds(0)==0 && TurnFinishingBudget.MarginMilliseconds(100)==12.5,"no extra time on exhausted/replan budget");
        Check(TurnFinishingBudget.MarginMilliseconds(10000)>300 && TurnFinishingBudget.MarginMilliseconds(10000)<400,"old saves retain bounded margin");
        Check(TurnFinishingBudget.MarginMilliseconds(60000)==400,"long allowances do not inflate margin");
        Check(TurnFinishingBudget.MayPublish(1999,2000,true) && !TurnFinishingBudget.MayPublish(2000,2000,true) && !TurnFinishingBudget.MayPublish(2001,2000,true),"late merge cannot replace completed fallback");
        Check(TurnFinishingBudget.MayPublish(2001,2000,false),"mandatory first legal incumbent survives exceptional overrun");
        foreach(double invalid in new[]{-1.0,double.NaN,double.PositiveInfinity}){bool rejected=false;try{TurnFinishingBudget.MarginMilliseconds(invalid);}catch(ArgumentOutOfRangeException){rejected=true;}Check(rejected,"invalid clock rejected");}
#endif
        var input=JObject.Parse(File.ReadAllText(Path.Combine(directory,"deadline-fixtures.json")));
        var cards=(JObject)input["composition"];var rows=new JArray();
        if((bool?)input["native"]!=true)
        {
            var h=new Host(842100);var fixtures=new JArray();
            for(int step=0;step<1600 && !h.Terminal;step++)
            {
                var v=(JObject)Wire.Token(h.Observe(h.Actor));var c=v["context"];var legal=h.Choices();
                string name=(string)c["era"]=="CanalEra" && (int)c["round"]==4?"canal-middle":
                    (string)c["era"]=="RailEra" && (int)c["round"]==2?"rail-early":
                    (string)c["era"]=="RailEra" && (int)c["round"]==7?"rail-late":null;
                if(name!=null && (string)c["phase"]=="action" && (int)c["actions_remaining"]==2 && !fixtures.Any(f=>(string)f["name"]==name))
                    fixtures.Add(new JObject{["name"]=name,["style"]=fixtures.Count==0?"balanced":fixtures.Count==1?"cotton":"pottery-flex",["view"]=v,["choices"]=JArray.FromObject(legal)});
                if(fixtures.Count==3)break;
                h.Apply(h.Revision,Planner.MaskedRollout(v,legal,ProfileCatalog.Keys[h.Actor]).id);
            }
            Check(fixtures.Count==3,"native four-player Canal and Rail fixtures");
            input["fixtures"]=fixtures;input["native"]=true;File.WriteAllText(Path.Combine(directory,"deadline-fixtures.json"),input.ToString());
        }
        foreach(var fixture in input["fixtures"])
        foreach(int budget in new[]{2000,5000,12000})
        {
            var view=(JObject)fixture["view"];var choices=fixture["choices"].ToObject<List<Choice>>();
            var p=ProfileCatalog.Create((string)fixture["style"],4);ProfileCatalog.ConfigureClock(p,budget);p.milliseconds=budget;p.work_limit=LocalTurnBridge.SearchWorkLimit;
            var before=Wire.Hash(view);var published=new List<Plan>();var publications=new List<double>();
            var clock=Stopwatch.StartNew();double publishMs=0;
#if FINISHING_MARGIN
            var ticket=Ticket.Create((string)view["context"]["game_id"],(int)view["context"]["revision"],(int)view["seat"],"action",p);
            TurnSearchTiming t;Plan final;
            using(var coordinator=new TurnCoordinator(ticket,view,choices,p,cards))
            {
                Check(coordinator.Worker.Wait(30000),"bounded worker retirement");t=coordinator.Timing;
                Check(coordinator.Finished && coordinator.Execution.ResourcesReleased,"worker ownership and resources retired");
                final=coordinator.Take(ticket);Check(coordinator.Take(ticket)==null,"one publication consumed once");
            }
            double elapsed=clock.Elapsed.TotalMilliseconds;Validate(final,view,cards);
            Check(t.workers==2 && t.failures==0 && !t.cancelled,"two completed private roots");
            Check(t.search_cutoff_ms==budget-t.margin_ms && ticket.deadline_ms==budget,"same whole-turn deadline, margin not multiplied per root/action");
            Check(t.elapsed_ms<budget && t.final_publish_ms<budget && t.late_publications_ignored==0,"both roots and merged publication before deadline");
            Check(Wire.Hash(view)==before,"observation immutable");
            rows.Add(new JObject{["fixture"]=fixture["name"],["style"]=p.tactic,["budget_ms"]=budget,["elapsed_ms"]=elapsed,
                ["finalization_after_cutoff_ms"]=t.finalization_ms,["publish_copy_ms"]=t.publish_copy_ms,
                ["final_publish_ms"]=t.final_publish_ms,["completed"]=t.completed,["interrupted"]=t.interrupted,["nodes"]=t.nodes,["failures"]=t.failures,["timing"]=JObject.FromObject(t)});
#else
            Action<Plan> publish=plan=>{var began=clock.Elapsed.TotalMilliseconds;var copy=JObject.FromObject(plan).ToObject<Plan>();lock(published){published.Add(copy);publications.Add(clock.Elapsed.TotalMilliseconds);}publishMs+=clock.Elapsed.TotalMilliseconds-began;};
            var d=RootForest.Decide(cards,view,choices,p,2,1,CancellationToken.None,publish);
            // Include the coordinator's existing final detached publication.
            publish(d.selected);double elapsed=clock.Elapsed.TotalMilliseconds;
            Validate(d.selected,view,cards);foreach(var plan in published)Validate(plan,view,cards);
            Check(Wire.Hash(view)==before,"observation immutable");Check(d.search.workers==2 && d.search.cpu_batching["pending"]==0,"both owners retired and queue drained");
            rows.Add(new JObject{["fixture"]=fixture["name"],["style"]=p.tactic,["budget_ms"]=budget,["elapsed_ms"]=elapsed,
                ["finalization_after_cutoff_ms"]=Math.Max(0,elapsed-budget),["publish_copy_ms"]=publishMs,
                ["first_publish_ms"]=publications.First(),["final_publish_ms"]=publications.Last(),
                ["completed"]=d.search.completed,["interrupted"]=d.search.interrupted,["nodes"]=d.nodes,["failures"]=d.failures});
#endif
        }
#if FINISHING_MARGIN
        // Cancel a real running forest after it has a complete legal fallback.
        var first=input["fixtures"].First();var firstView=(JObject)first["view"];var firstChoices=first["choices"].ToObject<List<Choice>>();
        var cancellationProfile=ProfileCatalog.Create("balanced",4);cancellationProfile.work_limit=LocalTurnBridge.SearchWorkLimit;cancellationProfile.milliseconds=12000;
        var cancelTicket=Ticket.Create((string)firstView["context"]["game_id"],(int)firstView["context"]["revision"],(int)firstView["seat"],"action",cancellationProfile);
        using(var coordinator=new TurnCoordinator(cancelTicket,firstView,firstChoices,cancellationProfile,cards))
        {
            var ready=Stopwatch.StartNew();while(!coordinator.Ready && !coordinator.Finished && ready.ElapsedMilliseconds<8000)Thread.Sleep(10);
            Check(coordinator.Ready && !coordinator.Finished,"cancellation exercised with active simulations and complete fallback");
            coordinator.Dispose();Check(coordinator.Worker.Wait(5000),"cancelled roots actually retire");
            Check(coordinator.Take(cancelTicket)==null && !coordinator.Ready && coordinator.Execution.ResourcesReleased,"cancelled/stale plan cannot publish and resources restored");
        }
        Check(BrassLegacyAI.Execution.AiExecutionScheduler.Statistics().running==0,"no detached execution remains");
#endif
        return new JObject{["passed"]=true,["checks"]=checks,["runtime"]="native Unity Mono; 30 FPS; two workers",["rows"]=rows};
    }
}
