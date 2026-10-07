using System;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Threading.Tasks;
using BrassHost;
using BrassLegacyAI;
using BrassLegacyAI.Execution;
using Newtonsoft.Json.Linq;

public static class FinishingMarginChecks
{
    static int checks;
    static void Check(bool value,string message){checks++;if(!value)throw new Exception(message);}
    public static int Main(string[] args)
    {
        try
        {
            var input=JObject.Parse(File.ReadAllText(args[0]));var fixture=input["fixtures"].First();
            var view=(JObject)fixture["view"];var cards=(JObject)input["composition"];var choices=fixture["choices"].ToObject<System.Collections.Generic.List<Choice>>();
            var profile=ProfileCatalog.Create("balanced",4);profile.milliseconds=0;
            var ticket=Ticket.Create((string)view["context"]["game_id"],(int)view["context"]["revision"],(int)view["seat"],"action",profile);
            using(var coordinator=new TurnCoordinator(ticket,view,choices,profile,cards))
            {
                Check(coordinator.Worker.Wait(10000),"exhausted request still retires both workers");
                Check(coordinator.Ready && coordinator.Failure==null,"exhausted request retains a complete fallback");
                Check(coordinator.Timing.search_cutoff_ms==0 && coordinator.Timing.margin_ms==0,"exhausted request receives no new time");
                Check(coordinator.Timing.late_publications_ignored>0,"late merge discarded instead of replacing fallback");
                var field=typeof(TurnCoordinator).GetField("incumbent",BindingFlags.NonPublic|BindingFlags.Instance);
                var fallback=(Plan)field.GetValue(coordinator);
                typeof(TurnCoordinator).GetMethod("Publish",BindingFlags.NonPublic|BindingFlags.Instance).Invoke(coordinator,new object[]{new Plan()});
                Check(ReferenceEquals(fallback,field.GetValue(coordinator)),"late publication cannot replace the legal incumbent");
                var plan=coordinator.Take(ticket);Check(ReferenceEquals(plan,fallback),"deadline take returns the retained plan");
                var host=new Host(1,Belief.Setup(view,cards,17));
                foreach(var action in plan.actions){foreach(var id in action){Check(host.Choices().Any(c=>c.id==id),"fallback legality");host.SearchApply(id);}Check(host.ActionComplete,"complete fallback action");}
                Check(host.Remaining==0 && plan.actions.Count==2,"fallback finishes both actions");
                Check(coordinator.Take(ticket)==null && coordinator.Execution.ResourcesReleased,"single take and complete resource retirement");
            }
            var hold=new TaskCompletionSource<int>();var blocker=AiExecutionScheduler.Submit(new DelegateAiExecutionBackend("queued-cancellation-fixture",token=>hold.Task));
            using(var coordinator=new TurnCoordinator(ticket,view,choices,profile,cards))
            {
                coordinator.Dispose();hold.SetResult(0);
                Check(coordinator.Worker.Wait(10000),"cancel before admission retires without waiting worker threads");
                Check(coordinator.Timing==null && !coordinator.Ready && coordinator.Take(ticket)==null && coordinator.Execution.ResourcesReleased,"cancelled job never launches or publishes");
            }
            Check(AiExecutionScheduler.IsIdle && AiExecutionScheduler.Statistics().running==0,"no detached work");
            File.WriteAllText(args[1],new JObject{["passed"]=true,["checks"]=checks}.ToString());return 0;
        }
        catch(Exception error){File.WriteAllText(args[1],new JObject{["passed"]=false,["checks"]=checks,["error"]=error.ToString()}.ToString());return 1;}
    }
}
