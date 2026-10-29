using System;
using System.Collections.Generic;
using System.Threading;
using BrassLegacyAI.Execution;
using UnityEngine;

namespace BrassLegacyAI
{
    public sealed class AiResourcePolicy
    {
        // Zero disables the cap. Optional work is explicitly registered, never
        // discovered by blanket scene scans or global physics/time switches.
        public readonly int FramesPerSecond;
        public readonly bool SuspendOptionalWork;
        public AiResourcePolicy(int framesPerSecond = 30, bool suspendOptionalWork = true)
        {
            if (framesPerSecond < 0 || framesPerSecond > 1000) throw new ArgumentOutOfRangeException(nameof(framesPerSecond));
            FramesPerSecond = framesPerSecond; SuspendOptionalWork = suspendOptionalWork;
        }
    }

    public interface IAiOptionalWork
    {
        // Return a token that restores this participant's PREVIOUS state.
        IDisposable Suspend();
    }

    public sealed class AiExecutionScope : IDisposable
    {
        internal bool Computing = true, Released;
        internal bool Delivering;
        internal AiExecutionJob Job;
        public void Observe(AiExecutionJob job)
        { Observe(job,false); }
        public void ObserveDelivery(AiExecutionJob job)
        { Observe(job,true); }
        void Observe(AiExecutionJob job,bool delivery)
        {
            AiExecutionResources.AssertMainThread();
            if (Released) throw new ObjectDisposedException(nameof(AiExecutionScope));
            Job = job ?? throw new ArgumentNullException(nameof(job)); Computing = true; Delivering=delivery; AiExecutionResources.Refresh();
        }
        public void WaitingForExecution()
        {
            AiExecutionResources.AssertMainThread();
            if (Released) throw new ObjectDisposedException(nameof(AiExecutionScope));
            Job?.Cancel(); Job = null; Computing = true; Delivering=false; AiExecutionResources.Refresh();
        }
        public void Presenting()
        {
            AiExecutionResources.AssertMainThread();
            if (Released) return;
            // Retain ownership of a retiring job until this scope is disposed;
            // presentation can resume while its cancellation tail drains.
            Computing = false; Delivering=false; AiExecutionResources.Refresh();
        }
        public void Dispose() { AiExecutionResources.Release(this); }
    }

    public static class AiExecutionResources
    {
        sealed class Registration : IDisposable
        {
            internal IAiOptionalWork Work;
            internal IDisposable Restoration;
            public void Dispose()
            {
                AssertMainThread(); Restore(); participants.Remove(this); Work = null;
            }
            internal void Pause()
            {
                if (Work == null || Restoration != null) return;
                try
                {
                    var token = Work.Suspend();
                    // Disabling an optional script may unregister/destroy its
                    // owner before Suspend returns. Do not strand its token.
                    if (Work == null) token?.Dispose(); else Restoration = token;
                }
                catch (Exception error) { Debug.LogWarning("AI optional work could not suspend: " + error.Message); }
            }
            internal void Restore()
            {
                var token = Restoration; Restoration = null;
                if (token == null) return;
                try { token.Dispose(); }
                catch (Exception error) { Debug.LogWarning("AI optional work could not restore: " + error.Message); }
            }
        }
        static readonly List<AiExecutionScope> scopes = new List<AiExecutionScope>(4);
        static readonly List<Registration> participants = new List<Registration>(16);
        static AiResourcePolicy policy = new AiResourcePolicy();
        static AiResourceDriver driver;
        static int mainThread, previousFrameRate, previousVSync;
        static bool renderingConstrained, optionalSuspended;
        public static bool IsComputing { get { for (int i = 0; i < scopes.Count; i++) if (scopes[i].Computing) return true; return false; } }
        public static int ScopeCount { get { return scopes.Count; } }
        public static int ParticipantCount { get { return participants.Count; } }
        public static AiResourcePolicy Policy { get { return policy; } }
        internal static void AssertMainThread()
        {
            int current = Thread.CurrentThread.ManagedThreadId;
            if (mainThread != 0 && current != mainThread) throw new InvalidOperationException("AI game resources must be changed on the Unity main thread.");
        }
        static void Initialize()
        {
            AssertMainThread();
            if (driver != null) return;
            mainThread = Thread.CurrentThread.ManagedThreadId;
            driver = new GameObject("Brass AI execution resources").AddComponent<AiResourceDriver>();
            UnityEngine.Object.DontDestroyOnLoad(driver.gameObject);
        }
        public static void Configure(AiResourcePolicy value)
        {
            Initialize(); policy = value ?? throw new ArgumentNullException(nameof(value)); Refresh();
        }
        public static AiExecutionScope BeginComputation()
        {
            Initialize(); var scope = new AiExecutionScope(); scopes.Add(scope); Refresh(); return scope;
        }
        public static IDisposable Register(IAiOptionalWork work)
        {
            if (work == null) throw new ArgumentNullException(nameof(work));
            Initialize(); var registration = new Registration { Work = work }; participants.Add(registration);
            if (optionalSuspended) registration.Pause(); return registration;
        }
        internal static void Release(AiExecutionScope scope)
        {
            AssertMainThread(); if (scope.Released) return;
            scope.Released = true; scope.Job?.Cancel(); scope.Job = null; scopes.Remove(scope); Refresh();
        }
        internal static void Refresh()
        {
            AssertMainThread(); bool busy = IsComputing;
            if (busy && policy.FramesPerSecond > 0)
            {
                if (!renderingConstrained)
                {
                    previousFrameRate = Application.targetFrameRate; previousVSync = QualitySettings.vSyncCount; renderingConstrained = true;
                }
                // Preserve an existing lower software cap. VSync otherwise takes
                // precedence over targetFrameRate in the shipped Windows player.
                QualitySettings.vSyncCount = 0;
                // Search retains its CPU-saving cap. Only short validation jobs
                // use 60 Hz handoffs; concurrent search always keeps priority.
                bool deliveryOnly=true;
                foreach(var scope in scopes)if(scope.Computing && !scope.Delivering){deliveryOnly=false;break;}
                int cap=deliveryOnly?Math.Max(60,policy.FramesPerSecond):policy.FramesPerSecond;
                Application.targetFrameRate = previousFrameRate > 0 ? Math.Min(previousFrameRate,cap) : cap;
            }
            else if (renderingConstrained)
            {
                Application.targetFrameRate = previousFrameRate; QualitySettings.vSyncCount = previousVSync; renderingConstrained = false;
            }
            bool pause = busy && policy.SuspendOptionalWork;
            if (pause == optionalSuspended) return;
            optionalSuspended = pause;
            // Lifecycle callbacks may register/unregister other participants.
            // Snapshot once at the boundary; the per-frame Tick stays allocation-free.
            if (participants.Count == 0) return;
            foreach (var participant in participants.ToArray()) { if (pause) participant.Pause(); else participant.Restore(); }
        }
        internal static void Tick()
        {
            // No allocations, serialization, or scene enumeration in this loop.
            bool changed = false;
            for (int i = 0; i < scopes.Count; i++)
            {
                var scope = scopes[i];
                if (scope.Computing && scope.Job != null && scope.Job.Completion.IsCompleted)
                { scope.Computing = false; scope.Job = null; changed = true; }
            }
            if (changed) Refresh();
        }
        public static void Reset()
        {
            AssertMainThread();
            for (int i = 0; i < scopes.Count; i++) { scopes[i].Released = true; scopes[i].Job?.Cancel(); scopes[i].Job = null; }
            scopes.Clear(); Refresh();
        }
        internal static void DriverDestroyed() { Reset(); driver = null; }
    }

    public sealed class AiResourceDriver : MonoBehaviour
    {
        void LateUpdate() { AiExecutionResources.Tick(); }
        void OnApplicationQuit() { AiExecutionResources.Reset(); }
        void OnDestroy() { AiExecutionResources.DriverDestroyed(); }
    }
}
