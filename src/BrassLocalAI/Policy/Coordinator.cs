using System;
using System.Diagnostics;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Newtonsoft.Json.Linq;
using BrassHost;
using BrassLegacyAI.Execution;
namespace BrassLegacyAI
{
    public sealed class Ticket
    {
        public string protocol="legacy/0.1.0",rules=Wire.Rules,schema=Wire.Version,profile,game_id,request_id,phase;
        public int revision,actor;
        public long generation;
        public string profile_fingerprint;
        public double deadline_ms;
        public static Ticket Create(string game,int revision,int actor,string phase,Profile profile)
        {return new Ticket{game_id=game,revision=revision,actor=actor,phase=phase,profile=profile.id,request_id=Guid.NewGuid().ToString("N"),deadline_ms=profile.milliseconds};}
        public Ticket Copy()
        {return new Ticket{protocol=protocol,rules=rules,schema=schema,profile=profile,game_id=game_id,request_id=request_id,phase=phase,revision=revision,actor=actor,generation=generation,profile_fingerprint=profile_fingerprint,deadline_ms=deadline_ms};}
    }
    // Own one coordinator per active turn, on the game thread. It accepts immutable
    // copies only; neither its worker nor its callback receives a live engine object.
    public sealed class TurnCoordinator:IDisposable,IAiExecutionBackend,IAiTurnResources
    {
        readonly object sync=new object();
        readonly Stopwatch clock=Stopwatch.StartNew();
        readonly Ticket ticket;
        readonly AiExecutionJob execution;
        JObject view,cards;
        System.Collections.Generic.List<Choice> legal;
        Profile cfg;
        IntentState ownIntent;
        Plan incumbent;
        bool taken,disposed;
        string failure;
        public string Name {get{return "S1 local search";}}
        public AiExecutionJob Execution {get{return execution;}}
        public Task Worker {get{return execution.Completion;}}
        public bool Ready {get{lock(sync)return incumbent!=null && !taken && !disposed;}}
        public bool Finished {get{return Worker.IsCompleted;}}
        public string Failure {get{lock(sync)return failure??execution.Failure;}}
        public double RemainingMilliseconds {get{return Math.Max(0,ticket.deadline_ms-clock.Elapsed.TotalMilliseconds);}}
        public TurnCoordinator(Ticket request,JObject observation,System.Collections.Generic.List<Choice> choices,Profile profile,JObject composition,IntentState intent=null)
        {
            ticket=request.Copy();
            if(ticket.protocol!="legacy/0.1.0" || ticket.schema!=Wire.Version || ticket.rules!=Wire.Rules || ticket.profile!=profile.id || ticket.actor!=(int)observation["seat"])throw new Exception("ticket_contract");
            view=(JObject)observation.DeepClone();legal=JArray.FromObject(choices).ToObject<System.Collections.Generic.List<Choice>>();
            cfg=Profile.Read(JObject.FromObject(profile));cards=(JObject)composition.DeepClone();
            ownIntent=intent==null?null:JObject.FromObject(intent).ToObject<IntentState>();
            execution=AiExecutionScheduler.Submit(this);
        }
        public Task ExecuteAsync(CancellationToken cancellation)
        {
            try
            {
                if(cancellation.IsCancellationRequested)return Task.FromResult(0);
                cfg.milliseconds=RemainingMilliseconds;
                var d=new Planner(cards).Decide(view,legal,cfg,false,1,cancellation,Publish,ownIntent);
                Publish(d.selected);
            }
            catch(Exception e){lock(sync)failure=e.GetType().Name+": "+e.Message;}
            return Task.FromResult(0);
        }
        public void ReleaseTurnResources() {view=null;legal=null;cfg=null;cards=null;ownIntent=null;}
        void Publish(Plan plan)
        {
            lock(sync)if(taken || disposed)return;
            // The detached copy is expensive; the game thread need not wait for
            // JSON serialization while it checks readiness/identity.
            var copy=JObject.FromObject(plan).ToObject<Plan>();
            lock(sync)if(!taken && !disposed)incumbent=copy;
        }
        public Plan Take(Ticket current,bool playNow=false)
        {return Take(current,current.revision,current.actor,current.generation,playNow);}
        public Plan Take(Ticket current,int revision,int actor,long generation,bool playNow=false)
        {
            Plan result=null;
            lock(sync)
            {
                if(disposed || taken || incumbent==null)return null;
                if(generation!=ticket.generation || current.profile_fingerprint!=ticket.profile_fingerprint || current.game_id!=ticket.game_id || current.request_id!=ticket.request_id || revision!=ticket.revision || actor!=ticket.actor || current.phase!=ticket.phase || current.profile!=ticket.profile || current.protocol!=ticket.protocol || current.rules!=ticket.rules || current.schema!=ticket.schema)
                {disposed=true;incumbent=null;}
                else
                {
                    if(!playNow && !Finished && RemainingMilliseconds>0)return null;
                    taken=true;result=incumbent;incumbent=null;
                }
            }
            execution.Cancel();return result;
        }
        public void Dispose(){lock(sync){disposed=true;incumbent=null;}execution.Cancel();}
    }
}
