// DIAGNOSTIC BUILD ONLY. Never linked into the deployable candidate.
using System;
using System.Collections.Concurrent;
using System.Diagnostics;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace BrassLegacyAI.Native
{
    public static class NativeStageTrace
    {
        static readonly ConcurrentQueue<object> samples=new ConcurrentQueue<object>();
        static readonly AsyncLocal<string> identity=new AsyncLocal<string>();
        static readonly AsyncLocal<long> parent=new AsyncLocal<long>();
        static long sequence;
        static readonly Func<long> allocated=Counter();
        static string allocationSource;
        [DllImport("brass_l14_alloc",CallingConvention=CallingConvention.Cdecl)]static extern int brass_alloc_start();
        [DllImport("brass_l14_alloc",CallingConvention=CallingConvention.Cdecl)]static extern long brass_alloc_bytes();
        [DllImport("brass_l14_alloc",CallingConvention=CallingConvention.Cdecl)]static extern long brass_alloc_objects();
        static Func<long> Counter(){
            var m=typeof(GC).GetMethod("GetAllocatedBytesForCurrentThread",Type.EmptyTypes);
            if(m!=null){allocationSource="GC.GetAllocatedBytesForCurrentThread";return (Func<long>)Delegate.CreateDelegate(typeof(Func<long>),m);}
            allocationSource="unavailable";
            if(Environment.GetEnvironmentVariable("BRASS_L14_MONO_ALLOC")!="1")return null;
            try {
                if(brass_alloc_start()!=1){allocationSource="Mono profiler exports unavailable";return null;}
                long before=brass_alloc_bytes(),count=brass_alloc_objects();
                var retained=new byte[100][];for(int i=0;i<100;i++)retained[i]=new byte[1024];
                long delta=brass_alloc_bytes()-before,objects=brass_alloc_objects()-count;GC.KeepAlive(retained);
                samples.Enqueue(new{stage="allocation.self_test",minimum_bytes=102400,minimum_objects=100,bytes=delta,objects});
                if(delta<102400 || objects<100){allocationSource="Mono callback self-test failed; allocation instrumentation is unavailable";return null;}
                allocationSource="Mono profiler gc_allocation / mono_object_get_size (thread local)";return brass_alloc_bytes;
            } catch(Exception e){allocationSource=e.GetType().Name+": "+e.Message;return null;}
        }
        [DllImport("kernel32.dll")]static extern IntPtr GetCurrentThread();
        [DllImport("kernel32.dll")]static extern uint GetCurrentThreadId();
        [DllImport("kernel32.dll")]static extern bool GetThreadTimes(IntPtr thread,out long creation,out long exit,out long kernel,out long user);
        static long Cpu(){long c,e,k,u;return GetThreadTimes(GetCurrentThread(),out c,out e,out k,out u)?k+u:0;}
        public static IDisposable Measure(string stage,string request=null){return new Span(stage,request);}
        sealed class Span:IDisposable
        {
            readonly string stage,request,oldIdentity;readonly long id,previous,started,cpu,bytes,heap;readonly int thread,gc0,gc1,gc2;
            public Span(string stage,string request){this.stage=stage;oldIdentity=identity.Value;if(request!=null)identity.Value=request;this.request=identity.Value;
                id=Interlocked.Increment(ref sequence);previous=parent.Value;parent.Value=id;thread=(int)GetCurrentThreadId();
                bytes=allocated==null?0:allocated();heap=GC.GetTotalMemory(false);gc0=GC.CollectionCount(0);gc1=GC.CollectionCount(1);gc2=GC.CollectionCount(2);cpu=Cpu();started=Stopwatch.GetTimestamp();}
            public void Dispose(){long end=Stopwatch.GetTimestamp();samples.Enqueue(new{id,parent=previous,stage,request,thread,started,end,
                wall_ms=(end-started)*1000.0/Stopwatch.Frequency,cpu_ms=(Cpu()-cpu)/10000.0,
                allocated_bytes=allocated==null?(long?)null:allocated()-bytes,heap_delta_bytes=GC.GetTotalMemory(false)-heap,
                gc=new[]{GC.CollectionCount(0)-gc0,GC.CollectionCount(1)-gc1,GC.CollectionCount(2)-gc2}});parent.Value=previous;identity.Value=oldIdentity;}
        }
        public static void Point(string stage,long nativeId,int bytes=0){samples.Enqueue(new{stage,bytes,native_id=nativeId,request=identity.Value,parent=parent.Value,thread=GetCurrentThreadId(),at=Stopwatch.GetTimestamp()});}
        public static byte[] Encode(JObject value){using(Measure("ipc.serialize")){var b=Encoding.UTF8.GetBytes(value.ToString(Formatting.None));Point("ipc.request_bytes",(long)value["id"],b.Length);return b;}}
        public static JObject Parse(string value){using(Measure("ipc.deserialize")){
            var received=Stopwatch.GetTimestamp();var result=JObject.Parse(value);
            samples.Enqueue(new{stage="ipc.reply_received",native_id=(long)result["id"],at=received,bytes=Encoding.UTF8.GetByteCount(value),request=identity.Value,parent=parent.Value,thread=GetCurrentThreadId()});
            if(result["data"]?["response_ready_ns"]!=null)samples.Enqueue(new{stage="native.response",request=identity.Value,parent=parent.Value,
                native_id=(long)result["id"],response_ready_ns=(long)result["data"]["response_ready_ns"],deadline_ns=(long)result["data"]["deadline_ns"],
                ticket=result["data"]["ticket"],completed=result["data"]["completed"]});
            return result;
        }}
        public static void Save(string path){File.WriteAllText(path,JsonConvert.SerializeObject(new{format="brass-l14-stage-trace/1.0.0",frequency=Stopwatch.Frequency,
            allocation_counter_available=allocated!=null,allocation_source=allocationSource,heap_delta_is_not_allocation_count=true,samples=samples.ToArray()},Formatting.Indented));}
    }
}
