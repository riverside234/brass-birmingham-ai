using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using BrassPatch.Statistics;
using UnityEngine;

namespace BrassPatch.History
{
    public static class HistoryRecording
    {
        static GameHistoryStore production, test;
        sealed class Publication { public Task Task;public string Key; }
        static readonly List<Publication> pending = new List<Publication>();
        static HistoryWriteHost host;
        static string failedKey;
        public static string LastFailure {get;private set;}
        public static void UseTestStore(GameHistoryStore store)
        { if (!StartupFixes.IsSelfTest) throw new InvalidOperationException("History test stores require the isolated self-test."); test = store; }
        public static GameHistoryStore Store
        {
            get {
                if (StartupFixes.IsSelfTest) {
                    if (test == null) test = new GameHistoryStore(Path.Combine(Path.GetDirectoryName(Application.dataPath), "artifacts", "history-test-" + Guid.NewGuid().ToString("N").Substring(0, 8)));
                    return test;
                }
                return production ?? (production = new GameHistoryStore(Path.Combine(Application.persistentDataPath, "GameHistoryStatistics")));
            }
        }
        public static void Queue(CompletedMatchStatistics report)
        {
            // Ordinary native suites collect matches but do not opt into writes.
            if (StartupFixes.IsSelfTest && test == null) return;
            var store = Store;
            if (host == null) { host = new GameObject("History report publication").AddComponent<HistoryWriteHost>(); UnityEngine.Object.DontDestroyOnLoad(host.gameObject); }
            string key=GameHistoryStatisticsCodec.Key(report.GameId);
            var task = Task.Run(() => store.Record(report)); lock (pending) pending.Add(new Publication{Task=task,Key=key});
        }
        public static bool IsWriting { get { lock (pending) return pending.Any(t => !t.Task.IsCompleted); } }
        public static void Drain()
        {
            Publication[] finished; lock (pending) { finished = pending.Where(t => t.Task.IsCompleted).ToArray(); pending.RemoveAll(t => t.Task.IsCompleted); }
            foreach (var item in finished) {
                if(item.Task.IsFaulted){failedKey=item.Key;LastFailure="The latest statistics report was not recorded. Earlier recorded games remain available.";Debug.LogWarning("Game History was not recorded: "+item.Task.Exception.GetBaseException().Message);}
                else {failedKey=LastFailure=null;}
            }
        }
        public static void ApplyFailure(GameHistoryList list)
        {
            if(LastFailure==null)return;
            if(list.Games.Any(g=>g.Key==failedKey&&g.Unavailable==null)){failedKey=LastFailure=null;return;}
            if(list.Error==null)list.Error=LastFailure;
        }
        public static void Flush()
        {
            Task[] tasks; lock (pending) tasks = pending.Select(t=>t.Task).ToArray();
            try { Task.WaitAll(tasks, 3000); } catch (AggregateException) { } Drain();
        }
    }
    public sealed class HistoryWriteHost : MonoBehaviour
    {
        void Update() { HistoryRecording.Drain(); }
        void OnApplicationQuit() { HistoryRecording.Flush(); }
    }
}
