using System;
using System.Diagnostics;
using System.IO;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using BrassLegacyAI.Execution;

namespace BrassLegacyAI.Native
{
    // Only the existing scheduler submits jobs. A persistent process retains the
    // model; reads and control writes never hold Unity's main thread.
    public sealed class NativeProcess : IDisposable
    {
        static NativeProcess shared;
        readonly object writes=new object();
        readonly Process process;
        long sequence;
        public readonly NativePackage Package;
        public readonly JObject Readiness;
        public static NativeProcess Shared() { return Shared(16); }
        public static NativeProcess Shared(int searchK)
        { return Shared(searchK,1.0); }
        public static NativeProcess Shared(int searchK,double temperature)
        {
            if(!NativePackage.Enabled) throw new InvalidOperationException("Alpha Test is disabled. This NN table must stop or be replaced as a whole.");
            // Only the scheduler's exclusive compute lease selects a worker.
            // Retire the old table's process before admitting another K.
            if(shared!=null && (shared.Package.SearchK!=searchK || shared.Package.Temperature!=temperature)){shared.Dispose();shared=null;}
            if(shared==null) shared=new NativeProcess(new NativePackage(NativePackage.Location,searchK,temperature));
            if(shared.process.HasExited) {shared.Dispose();shared=null;throw new IOException("Native worker exited; reload the saved game to recover");}
            return shared;
        }
        public NativeProcess(NativePackage package)
        {
            Package=package;
            process=new Process {StartInfo=new ProcessStartInfo {
                FileName=Path.Combine(package.Directory,"brass-ai-local.exe"),
                Arguments="\""+package.PolicyPath+"\" "+package.PolicyHash,
                WorkingDirectory=package.Directory,UseShellExecute=false,CreateNoWindow=true,
                RedirectStandardInput=true,RedirectStandardOutput=true,RedirectStandardError=true }};
            process.StartInfo.EnvironmentVariables["PYTHONPATH"]="";
            try {
                process.Start();
                // Drain diagnostics continuously. Never let a full stderr pipe
                // retain a compute owner or include private input in UI errors.
                process.ErrorDataReceived+=(s,e)=>{};process.BeginErrorReadLine();
                var initial=Task.Run(()=>Read());
                if(!initial.Wait(60000))throw new TimeoutException("Native startup timeout");
                var hello=initial.Result;
                if((long)hello["id"]!=0 || (string)hello["status"]!="ready")throw new IOException("Native readiness failed");
                Readiness=(JObject)hello["data"];
                NativePackage.ValidateAcceptance((JObject)Readiness["utility_acceptance"],Package.ModelManifest);
                if((string)Readiness["backend"]!=NativePackage.Backend || (string)Readiness["build"]["source_sha256"]!=package.Source ||
                    (string)Readiness["model_id"]!=NativePackage.Model || (string)Readiness["export_sha256"]!=NativePackage.Export ||
                    (int)Readiness["sessions"]!=2 || (int)Readiness["threads_per_session"]!=1 ||
                    (string)Readiness["provider"]!="CPUExecutionProvider" || (bool?)Readiness["measured_readiness"]["warmup_passed"]!=true)
                    throw new IOException("Native startup identity mismatch");
                if(!JToken.DeepEquals(Readiness["k_by_allowance_ms"],package.Policy["k_by_allowance_ms"]))
                    throw new IOException("Native startup search K mismatch");
                if((double?)Readiness["policy_temperature"]!=package.Temperature || (string)Readiness["temperature_scope"]!="all_nn_policy_evaluations")
                    throw new IOException("Native startup temperature mismatch");
            } catch {Dispose();throw;}
        }
        JObject Read()
        {
            var stream=process.StandardOutput.BaseStream;var header=Exact(stream,4);int length=BitConverter.ToInt32(header,0);
            if(length<1 || length>1024*1024)throw new IOException("Invalid native frame size");
            var result=JObject.Parse(new UTF8Encoding(false,true).GetString(Exact(stream,length)));
            if((string)result["protocol"]!=NativePackage.Protocol)throw new IOException("Native protocol mismatch");
            return result;
        }
        static byte[] Exact(Stream stream,int length)
        {
            var bytes=new byte[length];int offset=0;
            while(offset<length){int read=stream.Read(bytes,offset,length-offset);if(read==0)throw new EndOfStreamException("Native worker retired transport");offset+=read;}
            return bytes;
        }
        long Write(JObject operation)
        {
            lock(writes) {
                long id=++sequence;
                var request=new JObject{["protocol"]=NativePackage.Protocol,["id"]=id,["request"]=operation};
                var bytes=Encoding.UTF8.GetBytes(request.ToString(Formatting.None));
                if(bytes.Length>1024*1024)throw new InvalidDataException("Native request exceeds bound");
                var stream=process.StandardInput.BaseStream;var header=BitConverter.GetBytes(bytes.Length);
                stream.Write(header,0,header.Length);stream.Write(bytes,0,bytes.Length);stream.Flush();return id;
            }
        }
        public JObject Search(JObject request,CancellationToken cancellation)
        {
            cancellation.ThrowIfCancellationRequested();
            long id=Write(request);long cancelId=0;bool terminal=false;var gate=new object();
            // Cancellation only sends a control command. It never marks work as
            // complete. The response reader drains the acknowledgement AND the
            // terminal search reply before releasing the scheduler lease.
            using(cancellation.Register(()=>{
                try {lock(gate) {if(!terminal)cancelId=Write(new JObject{["op"]="cancel",["target"]=id});}}
                catch {try{process.Kill();}catch{} }
            })) {
                JObject result=null;long acknowledged=0;
                var read=Task.Run(()=>{
                    while(true){var reply=Read();long replyId=(long)reply["id"];
                        lock(gate) {
                            if(replyId==id) {result=reply;terminal=true;}
                            else if(replyId==cancelId && cancelId>0) acknowledged=replyId;
                            else throw new IOException("Stale native reply");
                            if(result!=null && (cancelId==0 || acknowledged==cancelId))break;
                        }
                    }
                });
                try {if(!read.Wait(20000))throw new TimeoutException("Native worker failed to retire");read.GetAwaiter().GetResult();}
                catch {Terminate();try {read.Wait();}catch{} throw;}
                if(cancellation.IsCancellationRequested)throw new OperationCanceledException(cancellation);
                if((string)result["status"]!="ok" || (bool?)result["data"]["retired"]!=true)throw new IOException("Native search stopped: "+(string)result["data"]["error"]);
                return (JObject)result["data"];
            }
        }
        void Terminate() {try {if(!process.HasExited)process.Kill();process.WaitForExit();}catch{} }
        public void Dispose() {Terminate();process.Dispose();}
    }

    public sealed class NativeJob : IAiExecutionBackend
    {
        readonly JObject request;
        readonly Stopwatch clock;
        readonly int allowance;
        readonly int searchK;
        readonly double temperature;
        NativeProcess worker;
        public string FailureBundle {get;private set;}
        public JObject Result {get;private set;}
        public string Name {get{return NativePackage.Backend;}}
        public NativeJob(JObject request,Stopwatch clock,int allowance) : this(request,clock,allowance,16) { }
        public NativeJob(JObject request,Stopwatch clock,int allowance,int searchK)
            : this(request,clock,allowance,searchK,1.0) { }
        public NativeJob(JObject request,Stopwatch clock,int allowance,int searchK,double temperature)
        {
            if(searchK!=8 && searchK!=16)throw new ArgumentOutOfRangeException("searchK");
            if(!NativePackage.ValidTemperature(temperature))throw new ArgumentOutOfRangeException("temperature");
            this.request=(JObject)request.DeepClone();this.clock=clock;this.allowance=allowance;this.searchK=searchK;
            this.temperature=temperature;
        }
        public Task ExecuteAsync(CancellationToken cancellation)
        {
            try { return Execute(cancellation); }
            catch(OperationCanceledException) { throw; }
            catch(Exception error) { RecordFailure("native-search",error.ToString(),null);throw; }
        }
        public void RecordFailure(string stage,string error,string[] path)
        {
            if(FailureBundle!=null)return;
            FailureBundle=NativeFailureBundle.Write(NativeFailureBundle.Location,request,Result,worker?.Package.Policy,
                worker?.Package.ModelManifest,worker?.Readiness,stage,error,clock.Elapsed.TotalMilliseconds,path);
        }
        Task Execute(CancellationToken cancellation)
        {
            // ExecuteAsync is invoked on the scheduler's pool task.
            cancellation.ThrowIfCancellationRequested();worker=NativeProcess.Shared(searchK,temperature);cancellation.ThrowIfCancellationRequested();
            if(!clock.IsRunning)clock.Start();
            request["remaining_ms"]=Math.Max(0,allowance-(long)Math.Ceiling(clock.Elapsed.TotalMilliseconds));
            request["ticket"]["source_sha256"]=worker.Package.Source;
            Result=worker.Search(request,cancellation);
            if(clock.Elapsed.TotalMilliseconds>=allowance)throw new TimeoutException("Native response exceeded the original turn allowance");
            if(!JToken.DeepEquals(Result["ticket"],request["ticket"]) || (string)Result["model_id"]!=NativePackage.Model ||
                (string)Result["export_sha256"]!=NativePackage.Export || (string)Result["backend"]!=Name)
                throw new IOException("Native response belongs to a different game/request/model");
            if((int?)Result["requested_k"]!=searchK)throw new IOException("Native response used a different search K");
            if((double?)Result["policy_temperature"]!=temperature || (string)Result["temperature_scope"]!="all_nn_policy_evaluations")
                throw new IOException("Native response used a different NN temperature");
            return Task.FromResult(0);
        }
    }
}
