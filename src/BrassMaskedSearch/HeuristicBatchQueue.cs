using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Threading;
using Newtonsoft.Json.Linq;
using BrassHost;

namespace BrassLegacyAI
{
    // Turn-local single-flight cache. Unrelated evaluations run concurrently
    // on their requesting search threads; only identical permitted inputs wait.
    // Private trees and mutable simulation worlds are never shared.
    public sealed class HeuristicBatchQueue
    {
        [ThreadStatic] public static HeuristicBatchQueue Current;
        sealed class Request
        {public int owner;public double[] value;public Exception error;public bool done;}
        readonly object sync=new object();
        readonly Dictionary<string,Request> pending=new Dictionary<string,Request>();
        readonly Dictionary<string,double[]> cache=new Dictionary<string,double[]>();
        readonly Queue<string> order=new Queue<string>();
        int active,max_concurrent;
        long compute_ticks,wait_ticks;
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
            Request request;bool lead=false;long waiting=0;
            lock(sync)
            {
                requests++;double[] found;
                if(cache.TryGetValue(key,out found)){cache_hits++;return (double[])found.Clone();}
                if(pending.TryGetValue(key,out request))
                {
                    // Recursive evaluation of the SAME key is a programming
                    // error, not a reason to deadlock its own owner.
                    if(request.owner==Thread.CurrentThread.ManagedThreadId)throw new InvalidOperationException("recursive_heuristic_key");
                    coalesced++;waiting=Stopwatch.GetTimestamp();
                }
                else
                {
                    request=new Request{owner=Thread.CurrentThread.ManagedThreadId};pending.Add(key,request);lead=true;
                    active++;max_concurrent=Math.Max(max_concurrent,active);batches++;max_batch=1;
                }
            }
            if(lead)
            {
                double[] value=null;Exception failure=null;long began=Stopwatch.GetTimestamp();
                try
                {
                    value=compute();
                    if(value==null || Array.Exists(value,x=>double.IsNaN(x)||double.IsInfinity(x)))throw new Exception("nonfinite_heuristic_batch");
                    value=(double[])value.Clone();
                }
                catch(Exception error){failure=error;}
                lock(sync)
                {
                    compute_ticks+=Stopwatch.GetTimestamp()-began;active--;
                    request.value=value;request.error=failure;request.done=true;pending.Remove(key);
                    if(failure==null){cache[key]=value;order.Enqueue(key);while(order.Count>512)cache.Remove(order.Dequeue());}
                    Monitor.PulseAll(sync);
                }
            }
            lock(sync)
            {
                while(!request.done)Monitor.Wait(sync);
                if(!lead)wait_ticks+=Stopwatch.GetTimestamp()-waiting;
                if(request.error!=null)throw new InvalidOperationException("heuristic_batch_failed",request.error);
                return (double[])request.value.Clone();
            }
        }
        public Dictionary<string,int> Report()
        {lock(sync)return new Dictionary<string,int>{{"requests",requests},{"cache_hits",cache_hits},{"coalesced",coalesced},{"batches",batches},{"max_batch",max_batch},{"pending",pending.Count},
            {"max_concurrent",max_concurrent},{"active",active},{"cache_entries",cache.Count},
            {"compute_ms",(int)(compute_ticks*1000.0/Stopwatch.Frequency)},{"coalesced_wait_ms",(int)(wait_ticks*1000.0/Stopwatch.Frequency)}};}
    }
}
