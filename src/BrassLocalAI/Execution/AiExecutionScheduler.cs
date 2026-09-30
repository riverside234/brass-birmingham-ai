using System;
using System.Diagnostics;
using System.Threading;
using System.Threading.Tasks;

namespace BrassLegacyAI.Execution
{
    // A backend's task must represent ALL of its turn work, including queued GPU
    // commands. Loaded model weights belong to the backend, not to a turn job.
    public interface IAiExecutionBackend
    {
        string Name { get; }
        Task ExecuteAsync(CancellationToken cancellation);
    }

    public interface IAiTurnResources
    {
        // Called after device/CPU completion, including cancellation before
        // dispatch. Release borrowed turn buffers; retain shared model weights.
        void ReleaseTurnResources();
    }

    public sealed class DelegateAiExecutionBackend : IAiExecutionBackend, IAiTurnResources
    {
        readonly Func<CancellationToken, Task> execute;
        readonly Action release;
        public string Name { get; private set; }
        public DelegateAiExecutionBackend(string name, Func<CancellationToken, Task> execute, Action releaseTurnResources = null)
        {
            if (string.IsNullOrEmpty(name)) throw new ArgumentException("A backend name is required.", nameof(name));
            this.execute = execute ?? throw new ArgumentNullException(nameof(execute)); release = releaseTurnResources; Name = name;
        }
        public Task ExecuteAsync(CancellationToken cancellation) { return execute(cancellation); }
        public void ReleaseTurnResources() { release?.Invoke(); }
    }

    public sealed class AiExecutionJob : IDisposable
    {
        readonly object sync = new object();
        readonly CancellationTokenSource cancellation = new CancellationTokenSource();
        readonly Stopwatch queued = Stopwatch.StartNew();
        IAiExecutionBackend backend;
        bool cancellationRequested, resourcesReleased;
        string failure;
        double queueMilliseconds, executionMilliseconds;
        public string BackendName { get; private set; }
        public Task Completion { get; private set; }
        public bool CancellationRequested { get { lock (sync) return cancellationRequested; } }
        public bool ResourcesReleased { get { lock (sync) return resourcesReleased; } }
        public string Failure { get { lock (sync) return failure; } }
        public double QueueMilliseconds { get { lock (sync) return queueMilliseconds; } }
        public double ExecutionMilliseconds { get { lock (sync) return executionMilliseconds; } }

        internal AiExecutionJob(IAiExecutionBackend backend) { this.backend = backend; BackendName = backend.Name; }
        internal void Start(Task predecessor) { Completion = Task.Run(() => Execute(predecessor)); }
        async Task Execute(Task predecessor)
        {
            bool running = false;
            var elapsed = new Stopwatch();
            try
            {
                // Awaiting a predecessor releases the pool thread. No main-thread
                // Wait/Join, extra dedicated thread, or thread-pool reset is needed.
                await predecessor.ConfigureAwait(false);
                CancellationToken token;
                lock (sync) { queueMilliseconds = queued.Elapsed.TotalMilliseconds; token = cancellation.Token; }
                if (token.IsCancellationRequested) return;
                running = true; AiExecutionScheduler.OnRunning(); elapsed.Start();
                await backend.ExecuteAsync(token).ConfigureAwait(false);
            }
            catch (OperationCanceledException) when (CancellationRequested) { }
            catch (Exception error) { lock (sync) failure = error.GetType().Name + ": " + error.Message; }
            finally
            {
                elapsed.Stop();
                try { (backend as IAiTurnResources)?.ReleaseTurnResources(); }
                catch (Exception error) { lock (sync) failure = (failure == null ? "" : failure + "; ") + "Turn cleanup: " + error.Message; }
                // No backend operation may use the source after ExecuteAsync ends.
                // Cancel and Dispose are serialized so repeated teardown is safe.
                lock (sync)
                {
                    executionMilliseconds = elapsed.Elapsed.TotalMilliseconds;
                    backend = null; cancellation.Dispose(); resourcesReleased = true;
                }
                AiExecutionScheduler.OnCompleted(running, CancellationRequested);
            }
        }
        public void Cancel()
        {
            lock (sync)
            {
                if (cancellationRequested || resourcesReleased) return;
                cancellationRequested = true;
                try { cancellation.Cancel(); }
                catch (AggregateException error) { failure = "Cancellation callback: " + error.Message; }
            }
        }
        public void Dispose() { Cancel(); }
    }

    public sealed class AiExecutionStatistics
    {
        public long submitted, completed, cancelled;
        public int running, maximumRunning;
    }

    // One executing turn job; a backend can use its own internal threads/batches.
    // This scheduler has no knowledge of strategies, difficulty, or time limits.
    public static class AiExecutionScheduler
    {
        static readonly object sync = new object();
        static Task tail = Task.FromResult(0);
        static long submitted, completed, cancelled;
        static int running, maximumRunning;
        public static bool IsIdle { get { lock (sync) return tail.IsCompleted; } }
        public static AiExecutionJob Submit(IAiExecutionBackend backend)
        {
            if (backend == null) throw new ArgumentNullException(nameof(backend));
            lock (sync)
            {
                var job = new AiExecutionJob(backend);
                Interlocked.Increment(ref submitted);
                job.Start(tail); tail = job.Completion; return job;
            }
        }
        internal static void OnRunning()
        {
            int count = Interlocked.Increment(ref running), previous;
            do { previous = maximumRunning; if (previous >= count) break; }
            while (Interlocked.CompareExchange(ref maximumRunning, count, previous) != previous);
        }
        internal static void OnCompleted(bool wasRunning, bool wasCancelled)
        {
            if (wasRunning) Interlocked.Decrement(ref running);
            if (wasCancelled) Interlocked.Increment(ref cancelled);
            Interlocked.Increment(ref completed);
        }
        public static AiExecutionStatistics Statistics()
        {
            return new AiExecutionStatistics { submitted = Interlocked.Read(ref submitted), completed = Interlocked.Read(ref completed),
                cancelled = Interlocked.Read(ref cancelled), running = Volatile.Read(ref running), maximumRunning = Volatile.Read(ref maximumRunning) };
        }
    }
}
