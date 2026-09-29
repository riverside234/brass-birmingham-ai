using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Security.Cryptography;
using System.Text;
using BoardGameRules.Entities.EventSourcing.Events;
using BrassApplication.Game;
using BrassApplication.Persistence.Serialization;
using Frameworks.Persistence.RepositoryBackends.Filesystem;
using Frameworks.Persistence.Repository;
using Frameworks.Persistence.GameFactory;
using BrassApplication.Persistence.Repository;
using Newtonsoft.Json.Linq;

namespace BrassPatch.Persistence
{
    /// <summary>Stock JSON remains authoritative. A bounded redo journal makes its two-file update recoverable.</summary>
    public static class PersistenceFixes
    {
        private const string JournalName = "brasspatch-save-v1.pending";
        private const int Magic = 0x42535031;
        private static readonly ConcurrentDictionary<string, object> Gates = new ConcurrentDictionary<string, object>(StringComparer.OrdinalIgnoreCase);
        private static T Property<T>(FilesystemGameStorage self, string name)
        {
            return (T)typeof(FilesystemGameStorage).GetProperty(name, BindingFlags.Instance | BindingFlags.NonPublic).GetValue(self, null);
        }
        private static string DirectoryFor(FilesystemGameStorage self, string gameId)
        {
            if (String.IsNullOrEmpty(gameId) || gameId == "." || gameId == ".." || gameId.IndexOfAny(Path.GetInvalidFileNameChars()) >= 0 || gameId.IndexOf('/') >= 0 || gameId.IndexOf('\\') >= 0)
                throw new ArgumentException("gameId must be a single directory name.");
            string root = Path.GetFullPath(Property<string>(self, "RootPath"));
            return Path.Combine(root, gameId);
        }
        private static object Gate(string directory) { return Gates.GetOrAdd(directory, key => new object()); }
        private static byte[] Serialize<T>(ISerialize serializer, T value)
        {
            using (MemoryStream stream = new MemoryStream()) { serializer.Serialize(stream, value); return stream.ToArray(); }
        }
        private static T Deserialize<T>(ISerialize serializer, byte[] bytes)
        {
            using (MemoryStream stream = new MemoryStream(bytes, false)) return serializer.Deserialize<T>(stream);
        }
        private static byte[] Hash(byte[] bytes) { using (SHA256 hash = SHA256.Create()) return hash.ComputeHash(bytes); }
        private static bool Equal(byte[] first, byte[] second) { return first.Length == second.Length && first.SequenceEqual(second); }
        private static void WriteAtomic(string path, byte[] bytes, bool retainBackup)
        {
            if (File.Exists(path) && Equal(File.ReadAllBytes(path), bytes)) return;
            string temporary = path + ".brasspatch-" + Guid.NewGuid().ToString("N") + ".tmp";
            try
            {
                using (FileStream stream = new FileStream(temporary, FileMode.CreateNew, FileAccess.Write, FileShare.None))
                { stream.Write(bytes, 0, bytes.Length); stream.Flush(true); }
                if (File.Exists(path)) File.Replace(temporary, path, retainBackup ? path + ".brasspatch-backup" : null);
                else File.Move(temporary, path);
            }
            finally { if (File.Exists(temporary)) File.Delete(temporary); }
        }
        private sealed class Transaction { public byte[] Events; public byte[] Metadata; }
        private static byte[] Journal(Transaction transaction)
        {
            using (MemoryStream stream = new MemoryStream())
            using (BinaryWriter writer = new BinaryWriter(stream, Encoding.UTF8))
            {
                writer.Write(Magic); WritePayload(writer, transaction.Events); WritePayload(writer, transaction.Metadata);
                writer.Flush(); return stream.ToArray();
            }
        }
        private static void WritePayload(BinaryWriter writer, byte[] bytes)
        {
            writer.Write(bytes != null);
            if (bytes == null) return;
            writer.Write(bytes.Length); writer.Write(Hash(bytes)); writer.Write(bytes);
        }
        private static byte[] ReadPayload(BinaryReader reader)
        {
            if (!reader.ReadBoolean()) return null;
            int length = reader.ReadInt32();
            if (length < 0 || length > reader.BaseStream.Length - reader.BaseStream.Position - 32)
                throw new InvalidDataException("Incomplete save recovery journal; original files are preserved.");
            byte[] expected = reader.ReadBytes(32), bytes = reader.ReadBytes(length);
            if (!Equal(expected, Hash(bytes))) throw new InvalidDataException("Save recovery journal checksum failed; original files are preserved.");
            return bytes;
        }
        private static Transaction ReadJournal(string directory)
        {
            using (FileStream stream = new FileStream(Path.Combine(directory, JournalName), FileMode.Open, FileAccess.Read, FileShare.Read))
            using (BinaryReader reader = new BinaryReader(stream, Encoding.UTF8))
            {
                if (reader.ReadInt32() != Magic) throw new InvalidDataException("Unrecognized save recovery journal version.");
                Transaction transaction = new Transaction { Events = ReadPayload(reader), Metadata = ReadPayload(reader) };
                if (stream.Position != stream.Length) throw new InvalidDataException("Unexpected trailing save recovery data.");
                return transaction;
            }
        }
        private static void Recover(string directory)
        {
            string journal = Path.Combine(directory, JournalName);
            if (!File.Exists(journal)) return;
            Transaction transaction = ReadJournal(directory);
            if (transaction.Events != null) WriteAtomic(Path.Combine(directory, "events.storage"), transaction.Events, true);
            if (transaction.Metadata != null) WriteAtomic(Path.Combine(directory, "metadata.storage"), transaction.Metadata, true);
            File.Delete(journal);
        }
        private static FileStream Lock(string directory)
        {
            Directory.CreateDirectory(directory);
            // A second process must retry rather than interleave its history with this save.
            return new FileStream(Path.Combine(directory, "brasspatch-save.lock"), FileMode.OpenOrCreate, FileAccess.ReadWrite, FileShare.None);
        }
        private static List<IEvent> ReadEvents(FilesystemGameStorage self, string directory)
        {
            string path = Path.Combine(directory, "events.storage");
            if (!File.Exists(path)) return new List<IEvent>();
            List<IEvent> events = Deserialize<List<IEvent>>(Property<ISerialize>(self, "Serializer"), File.ReadAllBytes(path));
            if (events == null) throw new InvalidDataException("Stored event history is null.");
            for (int i = 0; i < events.Count; i++)
                if (events[i] == null || events[i].Metadata == null || events[i].Metadata.Revision != i + 1)
                    throw new InvalidDataException("Stored event revisions are not contiguous. Preserve this save for recovery.");
            return events;
        }
        private static byte[] MetadataBytes(FilesystemGameStorage self, string id, GameMetadata metadata)
        {
            if (metadata == null) return null;
            GameMetadata copy = (GameMetadata)metadata.Clone(); copy.GameId = id;
            return SaveExtensions.Merge(metadata, id, DirectoryFor(self, id), Serialize(Property<ISerialize>(self, "Serializer"), copy));
        }
        public static void SaveGame(FilesystemGameStorage self, string gameId, IList<IEvent> incoming, GameMetadata metadata)
        {
            if (incoming == null) throw new ArgumentNullException("incoming");
            string directory = DirectoryFor(self, gameId);
            lock (Gate(directory)) using (Lock(directory))
            {
                Recover(directory);
                ISerialize serializer = Property<ISerialize>(self, "Serializer");
                List<IEvent> events = ReadEvents(self, directory);
                int originalCount = events.Count, lastRevision = -1;
                foreach (IEvent item in incoming)
                {
                    if (item == null || item.Metadata == null) throw new InvalidDataException("Event metadata is required.");
                    int revision = item.Metadata.Revision;
                    if (revision <= 0 || (lastRevision != -1 && revision != lastRevision + 1)) throw new InvalidDataException("Incoming event revisions are not contiguous.");
                    lastRevision = revision;
                    if (revision <= events.Count)
                    {
                        JToken stored = JToken.Parse(Encoding.UTF8.GetString(Serialize<IEvent>(serializer, events[revision - 1])).TrimStart('\uFEFF'));
                        JToken retry = JToken.Parse(Encoding.UTF8.GetString(Serialize<IEvent>(serializer, item)).TrimStart('\uFEFF'));
                        if (!JToken.DeepEquals(stored, retry)) throw new InvalidDataException("A different event already occupies revision " + revision + ".");
                    }
                    else if (revision == events.Count + 1) events.Add(item);
                    else throw new InvalidDataException("Incoming events leave a revision gap.");
                }
                if (incoming.Count > 0 && lastRevision < originalCount) throw new InvalidDataException("This save is older than the persisted history.");
                // Finish every fallible serialization before publishing the redo journal.
                Transaction transaction = new Transaction { Events = Serialize(serializer, events), Metadata = MetadataBytes(self, gameId, metadata) };
                WriteAtomic(Path.Combine(directory, JournalName), Journal(transaction), false);
                Recover(directory);
            }
        }
        public static void SaveGameMetadata(FilesystemGameStorage self, string gameId, GameMetadata metadata)
        {
            string directory = DirectoryFor(self, gameId);
            if (metadata == null) return;
            lock (Gate(directory)) using (Lock(directory))
            {
                Recover(directory);
                WriteAtomic(Path.Combine(directory, "metadata.storage"), MetadataBytes(self, gameId, metadata), true);
            }
        }
        public static List<IEvent> ReadEventsFromFile(FilesystemGameStorage self, string gameId)
        {
            string directory = DirectoryFor(self, gameId);
            if (!Directory.Exists(directory)) return new List<IEvent>();
            lock (Gate(directory)) using (Lock(directory)) { Recover(directory); return ReadEvents(self, directory); }
        }
        public static GameMetadata LoadGameMetadata(FilesystemGameStorage self, string gameId)
        {
            string directory = DirectoryFor(self, gameId);
            if (!Directory.Exists(directory)) return null;
            lock (Gate(directory)) using (Lock(directory))
            {
                Recover(directory);
                string path = Path.Combine(directory, "metadata.storage");
                if (!File.Exists(path)) return null;
                byte[] bytes = File.ReadAllBytes(path);
                if (bytes.Length == 0) throw new InvalidDataException("Metadata is empty. Save history and backups have been preserved.");
                GameMetadata metadata = Deserialize<GameMetadata>(Property<ISerialize>(self, "Serializer"), bytes);
                SaveExtensions.Read(metadata, bytes);
                return metadata;
            }
        }
        public static void SavePlayerMetadata(FilesystemGameStorage self, string gameId, PlayerMetadata playerMetadata)
        {
            if (playerMetadata == null) throw new ArgumentNullException("playerMetadata");
            string directory = DirectoryFor(self, gameId);
            // Replay saves last-seen metadata asynchronously. Hold the same lock over its
            // entire read/modify/write so it cannot publish an older full-game snapshot.
            lock (Gate(directory)) using (Lock(directory))
            {
                Recover(directory);
                string path = Path.Combine(directory, "metadata.storage");
                if (!File.Exists(path)) throw new InvalidDataException("No game metadata exists for this player update.");
                GameMetadata current = Deserialize<GameMetadata>(Property<ISerialize>(self, "Serializer"), File.ReadAllBytes(path));
                if (!String.IsNullOrEmpty(playerMetadata.GameId) && playerMetadata.GameId != gameId)
                    throw new InvalidDataException("Player metadata belongs to a different game.");
                // Every stock offline seat uses the network ID "Player". Resolve its
                // immutable starting seat and color, not the first shared network ID.
                var matches = current.PlayerMetadata.Where(p => p != null && p.PlayerNetworkId == playerMetadata.PlayerNetworkId);
                if (playerMetadata.PlayerStartConfiguration != null)
                    matches = matches.Where(p => p.PlayerStartConfiguration != null && p.PlayerStartConfiguration.Equals(playerMetadata.PlayerStartConfiguration));
                var players = matches.ToArray();
                if (players.Length != 1) throw new InvalidDataException("Player metadata does not identify exactly one seat in this game.");
                PlayerMetadata player = players[0];
                // Lower revisions are valid when replay/undo explicitly moves the last-seen boundary.
                player.LastSeenEventRevision = playerMetadata.LastSeenEventRevision;
                WriteAtomic(path, MetadataBytes(self, gameId, current), true);
            }
        }
        public static IRepository CreateOfflineGameRepository(RepositoryFactory self, string storagePath)
        {
            // The shipped cache publishes appended events before the filesystem save succeeds.
            // Offline histories are small; direct storage keeps reads consistent with journal recovery.
            return new BaseRepository(new FilesystemGameStorage(storagePath, new Frameworks.Persistence.Serialization.JsonSerializer()),
                new FilesystemSnapshotStorage(storagePath, new Frameworks.Persistence.Serialization.JsonSerializer()), new BrassBirminghamFactory());
        }
    }
}
