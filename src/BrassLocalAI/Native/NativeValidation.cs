using System;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using BrassHost;
using BrassLegacyAI.Execution;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace BrassLegacyAI.Native
{
    // Immutable, private validator inputs. No application, Unity object, event,
    // live Host or rules object is retained by background work. The deterministic
    // validator has exactly the same inputs as Host.ValidateLiveAction:
    // boundary + Observe(actor) + composition + complete semantic path.
    public sealed class NativeValidation : IAiExecutionBackend
    {
        // The user retained the measured responsiveness gain on 2026-10-26.
        // Keep a compile-time baseline for controlled comparisons; correctness
        // checks and the remaining whole-backend qualification still apply.
#if BRASS_L14_BASELINE_VALIDATION
        public const bool Enabled = false;
#else
        public const bool Enabled = true;
#endif
        readonly JObject view, composition;
        readonly string binding;
        readonly string[] path;
        readonly int capturedThread;
        int validated, consumed;
        CancellationToken executionCancellation;
        public string Name { get { return "Alpha Test detached action validation"; } }
        public bool Validated { get { return Volatile.Read(ref validated)==1; } }
        public NativeValidation(JObject view,JObject composition,string[] path,string binding)
        {
            if(view==null || composition==null || String.IsNullOrEmpty(binding))throw new ArgumentException("Missing validation identity");
            CheckPath(path);
            if((string)view["context"]?["phase"]!="action")throw new InvalidOperationException("Native validation requires an ordinary action boundary");
            this.view=(JObject)view.DeepClone();this.composition=(JObject)composition.DeepClone();
            this.path=(string[])path.Clone();this.binding=binding;capturedThread=Thread.CurrentThread.ManagedThreadId;
        }
        static void CheckPath(string[] value)
        {
            if(value==null || value.Length<1 || value.Length>256 || value.Any(s=>s==null || s.Length>512))
                throw new InvalidOperationException("Malformed native action");
        }
        public Task ExecuteAsync(CancellationToken cancellation)
        {
            executionCancellation=cancellation;cancellation.ThrowIfCancellationRequested();
            var trial=new Host(1,Belief.Setup(view,composition,1));
            for(int i=0;i<path.Length;i++) {
                cancellation.ThrowIfCancellationRequested();trial.SearchApply(path[i]);
                if(trial.ActionComplete && i<path.Length-1)throw new InvalidOperationException("multiple_actions");
            }
            if(!trial.ActionComplete)throw new InvalidOperationException("incomplete_live_action");
            cancellation.ThrowIfCancellationRequested();Volatile.Write(ref validated,1);
            return Task.FromResult(0);
        }
        // Only the capturing (Unity main) thread may consume a receipt, once.
        // Equality, not a hash alone, proves the deterministic full validator
        // would see the exact same inputs. No chance draw or Advance occurs in
        // SearchApply: hidden opponent cards/deck cannot affect a current action.
        // If this equivalence ever changes, retain a full final validation here.
        public void Commit(Host live,JObject currentComposition,string[] currentPath,string currentBinding,bool cancelled)
        {
            if(Thread.CurrentThread.ManagedThreadId!=capturedThread)throw new InvalidOperationException("Native commit must run on its capturing thread");
            if(cancelled || executionCancellation.IsCancellationRequested || !Validated || Volatile.Read(ref consumed)!=0)
                throw new InvalidOperationException("Cancelled, unfinished or consumed native validation");
            CheckPath(currentPath);
            if(binding!=currentBinding || !path.SequenceEqual(currentPath) || !JToken.DeepEquals(composition,currentComposition) ||
                live.Phase!="action" || live.Game.TurnFlow.CurrentAction!=null ||
                !JToken.DeepEquals(view,Wire.Token(live.Observe(live.Actor))))
                throw new InvalidOperationException("Stale native validation");
            if(Interlocked.Exchange(ref consumed,1)!=0)throw new InvalidOperationException("Native validation was already consumed");
            // All mutation and event confirmation remain on the main thread.
            // SearchApply also resolves each semantic ID against the current
            // legal choices. No coroutine/await is permitted inside this loop.
            foreach(string choice in path)live.SearchApply(choice);
        }
        public static string Binding(string lifetime,string instance,string gameId,long generation,long request,int action,
            int revision,int actor,string phase,JObject reply)
        {
            return new JObject{["lifetime"]=lifetime,["instance"]=instance,["game_id"]=gameId,["generation"]=generation,
                ["request"]=request,["action"]=action,["revision"]=revision,["actor"]=actor,["phase"]=phase,
                ["model"]=NativePackage.Model,["export"]=NativePackage.Export,["profile"]=NativePackage.Profile,
                ["ticket"]=reply?["ticket"]?.DeepClone(),["backend"]=reply?["backend"],
                ["reply_model"]=reply?["model_id"],["reply_export"]=reply?["export_sha256"]}.ToString(Formatting.None);
        }
    }
}
