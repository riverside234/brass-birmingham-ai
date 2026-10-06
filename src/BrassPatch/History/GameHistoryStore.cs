using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading;
using BrassPatch.Statistics;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace BrassPatch.History
{
    // One atomic publication contains both admission and retention. No old
    // report files, saved games or permanent report backups are pruning targets.
    public sealed class GameHistoryStore
    {
        public const int Limit = 10;
        const int MaxBytes = 64 * 1024 * 1024;
        static readonly object io = new object();
        readonly string root, file, pending;
        readonly Func<DateTime> clock;
        public Action<string> Fault { get; set; }
        public string Root { get { return root; } }
        public GameHistoryStore(string directory, Func<DateTime> now = null)
        {
            root = Path.GetFullPath(directory); file = Child("history.json"); pending = Child("history.pending.json"); clock = now ?? (() => DateTime.UtcNow);
        }
        string Child(string name)
        {
            var path = Path.GetFullPath(Path.Combine(root, name));
            if (!path.StartsWith(root.TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar) + Path.DirectorySeparatorChar, StringComparison.OrdinalIgnoreCase)) throw new InvalidDataException("History path escapes its root.");
            return path;
        }
        void CheckRoot()
        {
            Directory.CreateDirectory(root);
            if ((File.GetAttributes(root) & FileAttributes.ReparsePoint) != 0) throw new IOException("History directory must not be a filesystem link.");
            foreach (var path in new[] { file, pending }) if (File.Exists(path) && (File.GetAttributes(path) & FileAttributes.ReparsePoint) != 0) throw new IOException("History file must not be a filesystem link.");
        }
        static JObject Empty()
        { return new JObject { ["format"] = "brass-history/1", ["generation"] = 0L, ["lastSequence"] = 0L, ["seen"] = new JObject(), ["games"] = new JArray() }; }
        static void ValidateBank(JObject bank)
        {
            if (bank == null || bank.Properties().Count() != 5 || (string)bank["format"] != "brass-history/1") throw new InvalidDataException("Unsupported Game History format.");
            var games = bank["games"] as JArray; var seen = bank["seen"] as JObject; long last = (long)bank["lastSequence"];
            if (games == null || games.Count > Limit || seen == null || seen.Count > 500000 || last < 0 || (long)bank["generation"] < 0) throw new InvalidDataException("Invalid Game History index.");
            var sequences = new HashSet<long>();
            foreach (var marker in seen.Properties()) {
                long sequence = (long)marker.Value;
                if (!Regex.IsMatch(marker.Name, "^[0-9a-f]{64}$") || sequence <= 0 || sequence > last || !sequences.Add(sequence)) throw new InvalidDataException("Invalid history deduplication marker.");
            }
            var keys = new HashSet<string>(); long prior = Int64.MaxValue;
            foreach (var item in games) {
                string key = (string)item["key"]; long sequence = (long)item["sequence"];
                if (item is JObject == false || key == null || !keys.Add(key) || seen[key] == null || (long)seen[key] != sequence || sequence >= prior) throw new InvalidDataException("History ordering or identity differs.");
                prior = sequence;
            }
        }
        static JObject Read(string path)
        {
            if (new FileInfo(path).Length > MaxBytes) throw new InvalidDataException("Game History exceeds its size limit.");
            using (var stream = new StreamReader(path, new UTF8Encoding(false, true)))
            using (var reader = new JsonTextReader(stream) { DateParseHandling = DateParseHandling.None, MaxDepth = 40 }) {
                var bank = JObject.Load(reader); if (reader.Read()) throw new InvalidDataException("Unexpected trailing history data."); ValidateBank(bank); return bank;
            }
        }
        JObject Recover()
        {
            CheckRoot(); JObject bank = File.Exists(file) ? Read(file) : Empty();
            if (File.Exists(pending)) {
                JObject candidate = null;
                try {
                    candidate = Read(pending);
                    foreach (var item in (JArray)candidate["games"]) {
                        var retained=((JArray)bank["games"]).OfType<JObject>().FirstOrDefault(g=>(string)g["key"]==(string)item["key"]);
                        // Unavailable older entries remain opaque. Validate every
                        // incoming/changed report, while retaining those rows.
                        if(retained==null||!JToken.DeepEquals(retained,item))GameHistoryStatisticsCodec.Record((JObject)item);
                    }
                }
                catch (JsonException) { File.Delete(pending);candidate=null; }
                catch (InvalidDataException) { File.Delete(pending);candidate=null; }
                if (candidate != null) {
                    if ((long)candidate["generation"] > (long)bank["generation"]) { Publish(); bank = candidate; }
                    else File.Delete(pending);
                }
            }
            return bank;
        }
        void Inject(string stage) { var callback = Fault; if (callback != null) callback(stage); }
        void Publish()
        {
            for(int attempt=0;;attempt++) {
                try { if(File.Exists(file))File.Replace(pending,file,null,true);else File.Move(pending,file);return; }
                catch(IOException error) {
                    int code=error.HResult&0xffff;
                    if(attempt>=5||!File.Exists(pending)||code!=32&&code!=33&&code!=1175&&!error.Message.Contains("1175")&&!error.Message.Contains("Sharing violation"))throw;
                    Thread.Sleep(100*(attempt+1));
                }
            }
        }
        void Commit(JObject bank)
        {
            ValidateBank(bank); byte[] bytes = Encoding.UTF8.GetBytes(bank.ToString(Formatting.None));
            if (bytes.Length > MaxBytes) throw new InvalidDataException("Game History exceeds its size limit.");
            Inject("before-write");
            using (var stream = new FileStream(pending, FileMode.Create, FileAccess.Write, FileShare.None)) { stream.Write(bytes, 0, bytes.Length); stream.Flush(true); }
            Inject("after-flush"); Read(pending); Inject("before-publish"); Publish(); Inject("after-publish");
        }
        public HistoryAdmission Record(CompletedMatchStatistics report, string producer = "1.4.23")
        {
            lock (io) {
                var bank = Recover(); string key = GameHistoryStatisticsCodec.Key(report.GameId);
                var seen = (JObject)bank["seen"]; var games = (JArray)bank["games"]; var existing = games.OfType<JObject>().FirstOrDefault(g => (string)g["key"] == key);
                if (seen[key] != null && existing == null) return HistoryAdmission.AlreadyRetired;
                if (existing != null) {
                    // A sound complete report wins over a weaker recapture. A
                    // corrupt entry can be repaired by a valid completed report.
                    try {
                        var old = GameHistoryStatisticsCodec.Record(existing).Report;
                        if (old.CompleteHistory && !report.CompleteHistory || old.FinalRevision > report.FinalRevision || old.CompleteHistory == report.CompleteHistory && old.FinalRevision == report.FinalRevision)
                            return HistoryAdmission.Unchanged;
                    } catch (InvalidDataException) { }
                    catch (JsonException) { }
                }
                long sequence = existing == null ? checked((long)bank["lastSequence"] + 1) : (long)existing["sequence"];
                DateTime time = existing == null ? clock().ToUniversalTime() : DateTime.ParseExact((string)existing["recordedUtc"], "o", CultureInfo.InvariantCulture, DateTimeStyles.RoundtripKind);
                var record = GameHistoryStatisticsCodec.Envelope(report, sequence, time, producer);
                // Decode our own publication once: missing computed backing data
                // cannot be accepted as an apparently successful capture.
                GameHistoryStatisticsCodec.Record(record);
                if (existing != null) existing.Remove(); games.Add(record);
                bank["games"] = new JArray(games.OfType<JObject>().OrderByDescending(g => (long)g["sequence"]).Take(Limit));
                if (existing == null) { seen[key] = sequence; bank["lastSequence"] = sequence; }
                bank["generation"] = checked((long)bank["generation"] + 1); Commit(bank);
                return existing == null ? HistoryAdmission.Added : HistoryAdmission.Updated;
            }
        }
        public GameHistoryList List()
        {
            lock (io) {
                try { var bank = Recover(); return new GameHistoryList { Games = ((JArray)bank["games"]).OfType<JObject>().Select(g => GameHistoryStatisticsCodec.Summary(g)).ToList().AsReadOnly() }; }
                catch (Exception error) { return new GameHistoryList { Games = new List<GameHistorySummary>().AsReadOnly(), Error = "Game History could not be read. Use Refresh to retry.",Diagnostic=error.Message }; }
            }
        }
        public GameHistoryRecord Load(string key)
        {
            lock (io) {
                if (key == null || !Regex.IsMatch(key, "^[0-9a-f]{64}$")) throw new InvalidDataException("Invalid history selection.");
                var record = ((JArray)Recover()["games"]).OfType<JObject>().SingleOrDefault(g => (string)g["key"] == key);
                if (record == null) throw new InvalidDataException("This game is no longer in the latest ten.");
                return GameHistoryStatisticsCodec.Record(record);
            }
        }
    }
}
