using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Threading;
using Newtonsoft.Json.Linq;
using BrassHost;

namespace BrassLegacyAI
{
    // A turn-local dynamic CPU queue. Private trees never share mutable nodes.
    // One caller drains batches outside the lock; identical permitted inputs
    // coalesce. No extra evaluation thread competes with simulator workers.
    public sealed class HeuristicBatchQueue
    {
        [ThreadStatic] public static HeuristicBatchQueue Current;
        [ThreadStatic] static HeuristicBatchQueue computing;
        sealed class Request
        {public string key;public Func<double[]> compute;public double[] value;public Exception error;public bool done;}
        readonly object sync=new object();
        readonly Queue<Request> queue=new Queue<Request>();
        readonly Dictionary<string,Request> pending=new Dictionary<string,Request>();
        readonly Dictionary<string,double[]> cache=new Dictionary<string,double[]>();
        readonly Queue<string> order=new Queue<string>();
        bool draining;
        public int requests,cache_hits,coalesced,batches,max_batch;
        public static string Key(JObject v)
        {
            var context=new Dictionary<string,JToken>();foreach(var p in ((JObject)v["context"]).Properties())if(p.Name!="game_id" && p.Name!="revision")context.Add(p.Name,p.Value);
            var state=new Dictionary<string,JToken>();foreach(var p in v.Properties())if(p.Name!="context")state.Add(p.Name,p.Value);
            return Wire.Hash(new{context,state});
        }
        public IDisposable Bind(){return new Binding(this);}
        sealed class Binding:IDisposable
        {readonly HeuristicBatchQueue previous;public Binding(HeuristicBatchQueue owner){previous=Current;Current=owner;}public void Dispose(){Current=previous;}}
        public double[] Evaluate(string key,Func<double[]> compute)
        {
            // A leaf/proposal may request a small cached sub-evaluation. The
            // drain owner must execute it inline: waiting on its own queue
            // would deadlock all forest workers. No computation holds sync.
            if(computing==this)
            {
                lock(sync){requests++;double[] found;if(cache.TryGetValue(key,out found)){cache_hits++;return (double[])found.Clone();}}
                var nested=compute();
                if(nested==null || Array.Exists(nested,x=>double.IsNaN(x)||double.IsInfinity(x)))throw new Exception("nonfinite_heuristic_batch");
                lock(sync){if(!cache.ContainsKey(key)){cache[key]=(double[])nested.Clone();order.Enqueue(key);while(order.Count>512)cache.Remove(order.Dequeue());}}
                return (double[])nested.Clone();
            }
            Request request;bool lead=false;
            lock(sync)
            {
                requests++;double[] found;
                if(cache.TryGetValue(key,out found)){cache_hits++;return (double[])found.Clone();}
                if(pending.TryGetValue(key,out request))coalesced++;
                else{request=new Request{key=key,compute=compute};pending.Add(key,request);queue.Enqueue(request);}
                if(!draining){draining=true;lead=true;}
            }
            if(lead)Drain();
            lock(sync){while(!request.done)Monitor.Wait(sync);if(request.error!=null)throw new InvalidOperationException("heuristic_batch_failed",request.error);return (double[])request.value.Clone();}
        }
        void Drain()
        {
            // Short adaptive flush; complete clocks and queued latency are measured
            // by the enclosing turn. Cache/pending keys include observer and hand.
            var flush=Stopwatch.StartNew();while(flush.Elapsed.TotalMilliseconds<0.15)Thread.Yield();
            while(true)
            {
                var batch=new List<Request>();
                lock(sync)
                {
                    while(queue.Count>0 && batch.Count<16)batch.Add(queue.Dequeue());
                    if(batch.Count==0){draining=false;Monitor.PulseAll(sync);return;}
                    batches++;max_batch=Math.Max(max_batch,batch.Count);
                }
                foreach(var request in batch)
                {
                    double[] value=null;Exception failure=null;
                    var previous=computing;
                    try{computing=this;value=request.compute();if(value==null || Array.Exists(value,x=>double.IsNaN(x)||double.IsInfinity(x)))throw new Exception("nonfinite_heuristic_batch");}
                    catch(Exception error){failure=error;}
                    finally{computing=previous;}
                    lock(sync)
                    {
                        request.value=value;request.error=failure;request.done=true;pending.Remove(request.key);
                        if(failure==null){cache[request.key]=(double[])value.Clone();order.Enqueue(request.key);while(order.Count>512)cache.Remove(order.Dequeue());}
                        Monitor.PulseAll(sync);
                    }
                }
            }
        }
        public Dictionary<string,int> Report()
        {lock(sync)return new Dictionary<string,int>{{"requests",requests},{"cache_hits",cache_hits},{"coalesced",coalesced},{"batches",batches},{"max_batch",max_batch},{"pending",pending.Count}};}
    }
}
