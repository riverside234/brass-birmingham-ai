using System;
using System.Diagnostics;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Newtonsoft.Json.Linq;
using BrassHost;
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
    }
    // Own one coordinator per active turn, on the game thread. It accepts immutable
    // copies only; neither its worker nor its callback receives a live engine object.
    public sealed class TurnCoordinator:IDisposable
    {
        readonly object sync=new object();
        readonly CancellationTokenSource cancellation=new CancellationTokenSource();
        readonly Stopwatch clock=Stopwatch.StartNew();
        readonly Ticket ticket;
        Plan incumbent;
        bool taken,disposed,finished;
        string failure;
        public Task Worker {get;private set;}
        public bool Ready {get{lock(sync)return incumbent!=null && !taken && !disposed;}}
        public bool Finished {get{lock(sync)return finished;}}
        public string Failure {get{lock(sync)return failure;}}
        public double RemainingMilliseconds {get{return Math.Max(0,ticket.deadline_ms-clock.Elapsed.TotalMilliseconds);}}
        public TurnCoordinator(Ticket request,JObject observation,System.Collections.Generic.List<Choice> choices,Profile profile,JObject composition,IntentState intent=null)
        {
            ticket=JObject.FromObject(request).ToObject<Ticket>();
            if(ticket.protocol!="legacy/0.1.0" || ticket.schema!=Wire.Version || ticket.rules!=Wire.Rules || ticket.profile!=profile.id || ticket.actor!=(int)observation["seat"])throw new Exception("ticket_contract");
            var view=(JObject)observation.DeepClone();var legal=JArray.FromObject(choices).ToObject<System.Collections.Generic.List<Choice>>();
            var cfg=Profile.Read(JObject.FromObject(profile));var cards=(JObject)composition.DeepClone();
            var ownIntent=intent==null?null:JObject.FromObject(intent).ToObject<IntentState>();
            Worker=Task.Run(()=>{
                try
                {
                    cfg.milliseconds=RemainingMilliseconds;
                    var d=new Planner(cards).Decide(view,legal,cfg,false,1,cancellation.Token,Publish,ownIntent);
                    Publish(d.selected);
                }
                catch(Exception e){lock(sync)failure=e.GetType().Name+": "+e.Message;}
                finally{lock(sync)finished=true;}
            });
        }
        void Publish(Plan plan)
        {
            lock(sync)if(!taken && !disposed)incumbent=JObject.FromObject(plan).ToObject<Plan>();
        }
        public Plan Take(Ticket current,bool playNow=false)
        {
            lock(sync)
            {
                if(disposed || taken || incumbent==null)return null;
                if(current.generation!=ticket.generation || current.profile_fingerprint!=ticket.profile_fingerprint || current.game_id!=ticket.game_id || current.request_id!=ticket.request_id || current.revision!=ticket.revision || current.actor!=ticket.actor || current.phase!=ticket.phase || current.profile!=ticket.profile || current.protocol!=ticket.protocol || current.rules!=ticket.rules || current.schema!=ticket.schema)
                {disposed=true;cancellation.Cancel();return null;}
                if(!playNow && !finished && RemainingMilliseconds>0)return null;
                taken=true;cancellation.Cancel();return JObject.FromObject(incumbent).ToObject<Plan>();
            }
        }
        public void Dispose(){lock(sync){disposed=true;cancellation.Cancel();}}
    }
}
