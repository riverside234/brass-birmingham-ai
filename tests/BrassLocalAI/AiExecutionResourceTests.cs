using System;
using System.Collections;
using System.Diagnostics;
using System.Threading.Tasks;
using BrassLegacyAI;
using BrassLegacyAI.Execution;
using UnityEngine;

public static class AiExecutionResourceTests
{
    sealed class Restore : IDisposable
    {
        Action restore;
        public Restore(Action restore) { this.restore = restore; }
        public void Dispose() { var action = restore; restore = null; action?.Invoke(); }
    }
    sealed class OptionalWork : IAiOptionalWork
    {
        public bool Paused;
        public int Pauses, Restores;
        public Action OnSuspend;
        public IDisposable Suspend()
        {
            bool previous = Paused; Paused = true; Pauses++;
            OnSuspend?.Invoke();
            return new Restore(() => { Paused = previous; Restores++; });
        }
    }
    public static object Measurements;
    public static int Checks;
    public static bool Passed;
    public static IEnumerator Run(Action<bool, string> check)
    {
        Checks = 0; Passed = false;
        var assertion = check; check = (value, reason) => { Checks++; assertion(value, reason); };
        int originalFps = Application.targetFrameRate, originalVSync = QualitySettings.vSyncCount;
        float originalTime = Time.timeScale;
        bool originalPhysics = Physics.autoSimulation, originalPhysics2D = Physics2D.autoSimulation;
        AiExecutionResources.Configure(new AiResourcePolicy());
        Application.targetFrameRate = -1; QualitySettings.vSyncCount = 1;
        var effect = new OptionalWork(); var registration = AiExecutionResources.Register(effect);
        var first = AiExecutionResources.BeginComputation();
        check(Application.targetFrameRate == 30 && QualitySettings.vSyncCount == 0 && effect.Paused, "computation cap overrides VSync and suspends only registered work");
        check(Time.timeScale == originalTime && Physics.autoSimulation == originalPhysics && Physics2D.autoSimulation == originalPhysics2D, "resource policy preserves global timing and physics");
        var nested = AiExecutionResources.BeginComputation(); first.Presenting();
        check(AiExecutionResources.IsComputing && effect.Pauses == 1 && effect.Restores == 0, "overlapping scopes suspend registered work once");
        nested.Dispose(); nested.Dispose();
        check(Application.targetFrameRate == -1 && QualitySettings.vSyncCount == 1 && !effect.Paused && effect.Restores == 1, "last computing scope restores exact previous rendering/effect state");
        first.WaitingForExecution(); AiExecutionResources.Configure(new AiResourcePolicy(20));
        check(Application.targetFrameRate == 20 && effect.Pauses == 2, "policy changes apply without restarting or double-suspending computation");
        AiExecutionResources.Configure(new AiResourcePolicy(0, false));
        check(Application.targetFrameRate == -1 && QualitySettings.vSyncCount == 1 && !effect.Paused, "disabled policy immediately restores rendering and optional work");
        first.Dispose(); registration.Dispose();
        AiExecutionResources.Configure(new AiResourcePolicy());
        var unregistering = new OptionalWork(); IDisposable unregisteringRegistration = null;
        unregisteringRegistration = AiExecutionResources.Register(unregistering);
        unregistering.OnSuspend = () => unregisteringRegistration.Dispose();
        using (var teardown = AiExecutionResources.BeginComputation())
            check(!unregistering.Paused && unregistering.Restores == 1 && AiExecutionResources.ParticipantCount == 0,
                "unregistering during suspension restores the returned state token exactly once");
        Application.targetFrameRate = 18; QualitySettings.vSyncCount = 0;
        AiExecutionResources.Configure(new AiResourcePolicy());
        using (var lower = AiExecutionResources.BeginComputation()) check(Application.targetFrameRate == 18, "existing lower cap is respected");
        check(Application.targetFrameRate == 18, "existing lower cap is restored");

        var go = new GameObject("Explicitly optional resource fixture"); go.SetActive(false);
        var playing = new GameObject("playing particles").AddComponent<ParticleSystem>(); playing.transform.SetParent(go.transform);
        var paused = new GameObject("already paused particles").AddComponent<ParticleSystem>(); paused.transform.SetParent(go.transform);
        var stopped = new GameObject("stopped particles").AddComponent<ParticleSystem>(); stopped.transform.SetParent(go.transform);
        var enabledScript = go.AddComponent<AiOptionalTestScript>();
        var disabledScript = go.AddComponent<AiOptionalTestScript>(); disabledScript.enabled = false;
        var marker = go.AddComponent<AiOptionalSimulation>();
        marker.Particles = new[] { playing, paused, stopped }; marker.Scripts = new Behaviour[] { enabledScript, disabledScript };
        go.SetActive(true); playing.Play(false); paused.Play(false); paused.Pause(false); stopped.Stop(false, ParticleSystemStopBehavior.StopEmittingAndClear);
        var particleScope = AiExecutionResources.BeginComputation();
        check(playing.isPaused && paused.isPaused && !stopped.isPlaying && !enabledScript.enabled && !disabledScript.enabled, "optional particles/scripts suspend while preserving pre-paused and disabled objects");
        particleScope.Presenting();
        check(playing.isPlaying && paused.isPaused && !stopped.isPlaying && enabledScript.enabled && !disabledScript.enabled, "presentation restores only previously running participants");
        particleScope.Dispose(); UnityEngine.Object.Destroy(go); yield return null;
        check(AiExecutionResources.ParticipantCount == 0, "destroyed optional participant unregisters without retained scene references");

        Application.targetFrameRate = 60; QualitySettings.vSyncCount = 0;
        var device = new TaskCompletionSource<int>(); var deviceStarted = new TaskCompletionSource<int>();
        var job = AiExecutionScheduler.Submit(new DelegateAiExecutionBackend("future GPU inference fixture", token => { deviceStarted.SetResult(0); return device.Task; }));
        var scope = AiExecutionResources.BeginComputation(); scope.Observe(job);
        float dispatchDeadline = Time.realtimeSinceStartup + 3;
        while (!deviceStarted.Task.IsCompleted && Time.realtimeSinceStartup < dispatchDeadline) yield return null;
        check(deviceStarted.Task.IsCompleted, "future inference backend dispatches before cancellation");
        job.Cancel(); yield return null;
        check(!job.Completion.IsCompleted && AiExecutionResources.IsComputing && Application.targetFrameRate == 30, "GPU-style outstanding work remains constrained after cancellation request");
        device.SetResult(0);
        float deadline = Time.realtimeSinceStartup + 3;
        while (!job.Completion.IsCompleted && Time.realtimeSinceStartup < deadline) yield return null;
        yield return null;
        check(job.Completion.IsCompleted && job.ResourcesReleased && !AiExecutionResources.IsComputing && Application.targetFrameRate == 60, "actual device completion automatically restores resources on main thread");
        scope.Dispose(); scope.Dispose();

        // Keep this source usable against the frozen baseline as well.
        var delivery=typeof(AiExecutionScope).GetMethod("ObserveDelivery");
        if(delivery!=null) {
            var releaseDelivery=new TaskCompletionSource<int>();
            var deliveryJob=AiExecutionScheduler.Submit(new DelegateAiExecutionBackend("short detached delivery",token=>releaseDelivery.Task));
            var deliveryScope=AiExecutionResources.BeginComputation();delivery.Invoke(deliveryScope,new object[]{deliveryJob});
            check(Application.targetFrameRate==60 && QualitySettings.vSyncCount==0 && AiExecutionResources.IsComputing,"delivery uses 60 Hz handoffs with optional work still constrained");
            var searchScope=AiExecutionResources.BeginComputation();
            check(Application.targetFrameRate==30,"overlapping search preserves the search frame cap");
            searchScope.Dispose();check(Application.targetFrameRate==60,"retiring overlapping search restores the delivery cap");
            deliveryJob.Cancel();check(!deliveryJob.Completion.IsCompleted && AiExecutionResources.IsComputing,"delivery cancellation still owns retiring work");
            releaseDelivery.SetResult(0);deadline=Time.realtimeSinceStartup+3;
            while(!deliveryJob.Completion.IsCompleted && Time.realtimeSinceStartup<deadline)yield return null;
            yield return null;deliveryScope.Dispose();
            check(Application.targetFrameRate==60 && AiExecutionResources.ScopeCount==0,"delivery retirement restores the previous cap");
            Application.targetFrameRate=18;
            var lowerJob=AiExecutionScheduler.Submit(new DelegateAiExecutionBackend("lower cap delivery",token=>Task.FromResult(0)));
            var lowerScope=AiExecutionResources.BeginComputation();delivery.Invoke(lowerScope,new object[]{lowerJob});
            check(Application.targetFrameRate==18,"delivery respects an existing lower software cap");
            while(!lowerJob.Completion.IsCompleted)yield return null;
            lowerScope.Dispose();Application.targetFrameRate=60;
        }

        var replacementDevice = new TaskCompletionSource<int>(); var replacementStarted = new TaskCompletionSource<int>();
        var replacementJob = AiExecutionScheduler.Submit(new DelegateAiExecutionBackend("independent future model", token => { replacementStarted.SetResult(0); return replacementDevice.Task; }));
        var replacementScope = AiExecutionResources.BeginComputation(); replacementScope.Observe(replacementJob);
        deadline = Time.realtimeSinceStartup + 3;
        while (!replacementStarted.Task.IsCompleted && Time.realtimeSinceStartup < deadline) yield return null;
        check(replacementStarted.Task.IsCompleted, "independent model starts without an S1 bridge");
        LocalTurnBridge.CancelActive();
        check(replacementJob.CancellationRequested && !replacementJob.ResourcesReleased && AiExecutionResources.ScopeCount == 0 && Application.targetFrameRate == 60,
            "game/menu lifecycle cancels any model and restores rendering before outstanding device work drains");
        replacementScope.Dispose(); replacementDevice.SetResult(0);
        deadline = Time.realtimeSinceStartup + 3;
        while (!replacementJob.Completion.IsCompleted && Time.realtimeSinceStartup < deadline) yield return null;
        check(replacementJob.Completion.IsCompleted && replacementJob.ResourcesReleased, "independent cancelled model retires only after device completion");

        // Report foreground-frame pacing and process CPU, without asserting a
        // universal GPU/power improvement from a synthetic idle fixture.
        var process = Process.GetCurrentProcess();
        double normalCpuStart = process.TotalProcessorTime.TotalMilliseconds;
        int normalFrames = 0; float normalStart = Time.realtimeSinceStartup;
        while (Time.realtimeSinceStartup - normalStart < 1) { normalFrames++; yield return null; }
        double normalSeconds = Time.realtimeSinceStartup - normalStart;
        process.Refresh(); double normalCpu = process.TotalProcessorTime.TotalMilliseconds - normalCpuStart;
        double cappedCpuStart = process.TotalProcessorTime.TotalMilliseconds;
        var measured = AiExecutionResources.BeginComputation();
        int cappedFrames = 0; float cappedStart = Time.realtimeSinceStartup;
        while (Time.realtimeSinceStartup - cappedStart < 1) { cappedFrames++; yield return null; }
        double cappedSeconds = Time.realtimeSinceStartup - cappedStart;
        process.Refresh(); double cappedCpu = process.TotalProcessorTime.TotalMilliseconds - cappedCpuStart;
        measured.Dispose(); process.Dispose();
        Measurements = new { normalFrames, normalSeconds, normalFps = normalFrames / normalSeconds, normalCpuMilliseconds = normalCpu,
            cappedFrames, cappedSeconds, cappedFps = cappedFrames / cappedSeconds, cappedCpuMilliseconds = cappedCpu,
            scope = "Synthetic menu frame-pacing fixture; process CPU includes engine work. No GPU power or thermal measurement." };
        check(Application.targetFrameRate == 60 && QualitySettings.vSyncCount == 0, "measurement cleanup restores foreground settings");
        var reset = AiExecutionResources.BeginComputation(); AiExecutionResources.Reset(); reset.Dispose();
        check(AiExecutionResources.ScopeCount == 0 && Application.targetFrameRate == 60, "global teardown invalidates leases and restores settings synchronously");
        Application.targetFrameRate = originalFps; QualitySettings.vSyncCount = originalVSync;
        Passed = true;
    }
}
public sealed class AiOptionalTestScript : MonoBehaviour { }
