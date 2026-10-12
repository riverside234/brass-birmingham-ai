using System;
using System.Diagnostics;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using BrassLegacyAI.Execution;

namespace BrassLegacyAI
{
    public static class PersonalityRpc
    {
        static Process process;static readonly object gate=new object(),write=new object();
        static string Quote(string text) {if(String.IsNullOrEmpty(text)||text.IndexOfAny(new[]{'"','\n','\r'})>=0)throw new InvalidDataException("Invalid Python/package path");return "\""+text.TrimEnd('\\')+"\"";}
        static void Start() {
            if(process!=null && !process.HasExited)return;
            var pointer=PersonalityPackage.Pointer;if(pointer==null)throw new InvalidDataException("Qualified package missing");
            var info=new ProcessStartInfo((string)pointer["python"],"-m brass_ai.personality.laptop.cli serve --package "+Quote(PersonalityPackage.DirectoryPath)+" --backend "+(string)pointer["backend"]+(PersonalityPackage.ReviewMode?" --fixture":""))
                {UseShellExecute=false,CreateNoWindow=true,RedirectStandardInput=true,RedirectStandardOutput=true,RedirectStandardError=true,
                 WorkingDirectory=PersonalityPackage.DirectoryPath};
            info.EnvironmentVariables["PYTHONPATH"]=Path.Combine(PersonalityPackage.DirectoryPath,"runtime/src");
            info.EnvironmentVariables["PYTHONDONTWRITEBYTECODE"]="1";
            process=Process.Start(info);process.ErrorDataReceived+=(s,e)=>{};process.BeginErrorReadLine();
        }
        static void Send(string id,string op,JObject body) {
            lock(write) {
                if(process==null || process.HasExited)return;
                process.StandardInput.WriteLine(new JObject{["protocol"]=PersonalityPackage.Protocol,["id"]=id,["op"]=op,["body"]=body}.ToString(Formatting.None));process.StandardInput.Flush();
            }
        }
        public static void Interrupt(string id,bool cancel) {Send(id,cancel?"cancel":"play-now",null);}
        public static JObject Call(string op,JObject body,string id=null,CancellationToken cancellation=default(CancellationToken),Action sent=null) {
            lock(gate) {
                Start();id=id??Guid.NewGuid().ToString("N");Send(id,op,body);sent?.Invoke();
                // Callback only writes to the idle control reader, never waits for
                // search. ExecuteAsync returns only after all workers/device retire.
                using(cancellation.Register(()=>Interrupt(id,true))) {
                    string line=process.StandardOutput.ReadLine();if(line==null)throw new IOException("Neural runtime exited");
                    var r=JObject.Parse(line);
                    if((string)r["protocol"]!=PersonalityPackage.Protocol || (string)r["id"]!=id)throw new IOException("Neural response correlation differs");
                    if((bool?)r["ok"]!=true)throw new InvalidOperationException((string)r["error"]);
                    return (JObject)r["data"];
                }
            }
        }
        public static void Shutdown() {
            lock(gate) {if(process==null)return;try{process.StandardInput.Close();process.WaitForExit();}finally{process.Dispose();process=null;}}
        }
    }
    public sealed class PersonalityJob:IAiExecutionBackend
    {
        public string Name {get{return "personality-onnx";}}
        public readonly string Id=Guid.NewGuid().ToString("N");readonly JObject body;
        readonly object sync=new object();bool started,playRequested;
        public JObject Result {get;private set;}
        public PersonalityJob(JObject body){this.body=body;body["request_id"]=Id;}
        public Task ExecuteAsync(CancellationToken token) {return Task.Run(()=>{token.ThrowIfCancellationRequested();Result=PersonalityRpc.Call("plan",body,Id,token,()=>{lock(sync){started=true;if(playRequested)PersonalityRpc.Interrupt(Id,false);}});});}
        public void PlayNow(){lock(sync){playRequested=true;if(started)PersonalityRpc.Interrupt(Id,false);}}
    }
}
