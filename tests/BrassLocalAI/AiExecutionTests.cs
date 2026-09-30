using System;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using BrassLegacyAI.Execution;
using Newtonsoft.Json.Linq;

public static class AiExecutionTests
{
    static int checks;
    static void Check(bool value, string reason) { checks++; if (!value) throw new Exception(reason); }
    public static int Main(string[] args)
    {
        try
        {
            var started = new ManualResetEventSlim();
            var deviceCompletion = new TaskCompletionSource<int>();
            int buffersReturned = 0;
            var first = AiExecutionScheduler.Submit(new DelegateAiExecutionBackend("fake asynchronous GPU", token => { started.Set(); return deviceCompletion.Task; }, () => Interlocked.Increment(ref buffersReturned)));
            Check(started.Wait(3000), "asynchronous device job starts");
            first.Cancel(); first.Dispose();
            int secondStarted = 0;
            var second = AiExecutionScheduler.Submit(new DelegateAiExecutionBackend("next model", token => { Interlocked.Increment(ref secondStarted); return Task.FromResult(0); }));
            Check(!second.Completion.Wait(100) && secondStarted == 0, "cancellation does not release the next job before device completion");
            Check(!first.ResourcesReleased && buffersReturned == 0 && !AiExecutionScheduler.IsIdle, "in-flight job retains its cancellation resources and borrowed buffers");
            deviceCompletion.SetResult(0);
            Check(second.Completion.Wait(3000), "next model starts after full device completion");
            Check(first.ResourcesReleased && second.ResourcesReleased && secondStarted == 1 && buffersReturned == 1, "completed jobs release resources and return buffers once");
            first.Cancel(); first.Dispose(); second.Dispose();

            var blockerStarted = new ManualResetEventSlim(); var blockerCompletion = new TaskCompletionSource<int>();
            var blocker = AiExecutionScheduler.Submit(new DelegateAiExecutionBackend("queue blocker", token => { blockerStarted.Set(); return blockerCompletion.Task; }));
            Check(blockerStarted.Wait(3000), "queue blocker starts");
            int abandonedCalls = 0, abandonedBuffersReturned = 0;
            var abandoned = AiExecutionScheduler.Submit(new DelegateAiExecutionBackend("cancelled before dispatch", token => { Interlocked.Increment(ref abandonedCalls); return Task.FromResult(0); }, () => Interlocked.Increment(ref abandonedBuffersReturned)));
            abandoned.Dispose(); blockerCompletion.SetResult(0);
            Check(abandoned.Completion.Wait(3000) && abandonedCalls == 0 && abandoned.ResourcesReleased && abandonedBuffersReturned == 1, "cancelled queued backend never executes and releases borrowed resources");
            var failure = AiExecutionScheduler.Submit(new DelegateAiExecutionBackend("failing backend", token => { throw new InvalidOperationException("backend fixture failure"); }));
            Check(failure.Completion.Wait(3000) && failure.Failure.Contains("backend fixture failure") && failure.ResourcesReleased, "failure is observed and retires the job");
            var recovery = AiExecutionScheduler.Submit(new DelegateAiExecutionBackend("recovery backend", token => Task.FromResult(0)));
            Check(recovery.Completion.Wait(3000), "backend failure does not poison subsequent scheduling");
            var stats = AiExecutionScheduler.Statistics();
            Check(stats.running == 0 && stats.maximumRunning == 1 && stats.submitted == stats.completed && stats.cancelled >= 2, "single active job and bounded completed-job bookkeeping");
            File.WriteAllText(args[0], JObject.FromObject(new { passed = true, checks, statistics = stats }).ToString());
            return 0;
        }
        catch (Exception error) { File.WriteAllText(args[0], JObject.FromObject(new { passed = false, checks, error = error.ToString() }).ToString()); return 1; }
    }
}
