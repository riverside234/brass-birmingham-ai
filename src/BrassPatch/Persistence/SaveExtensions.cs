using System;
using System.IO;
using System.Runtime.CompilerServices;
using System.Text;
using Newtonsoft.Json.Linq;
using BrassApplication.Game;

namespace BrassPatch.Persistence
{
    // JSON-only extension boundary: this assembly has no dependency on the AI executor.
    // Intent and assignment state share the existing metadata/event redo transaction.
    public static class SaveExtensions
    {
        public const string Key = "brasspatch_ai";
        private static JObject Parse(byte[] bytes)
        {
            using (var text = new StringReader(Encoding.UTF8.GetString(bytes).TrimStart('\uFEFF')))
            using (var reader = new Newtonsoft.Json.JsonTextReader(text) { DateParseHandling = Newtonsoft.Json.DateParseHandling.None })
                return JObject.Load(reader);
        }
        private sealed class Value { public JObject Data; }
        private static readonly ConditionalWeakTable<GameMetadata, Value> Values = new ConditionalWeakTable<GameMetadata, Value>();
        public static JObject Get(GameMetadata metadata)
        {
            if (metadata == null) return null;
            var value = Values.GetValue(metadata, key => new Value());
            lock (value) return value.Data == null ? null : (JObject)value.Data.DeepClone();
        }
        public static void Set(GameMetadata metadata, JObject data)
        {
            if (metadata == null || data == null) throw new ArgumentNullException();
            var value = Values.GetValue(metadata, key => new Value());
            lock (value) value.Data = (JObject)data.DeepClone();
        }
        public static object Clone(GameMetadata metadata)
        {
            var copy = (GameMetadata)ReflectionAccess.Original(metadata, "Clone", Type.EmptyTypes);
            var data = Get(metadata);
            if (data != null) Set(copy, data);
            return copy;
        }
        public static void Read(GameMetadata metadata, byte[] bytes)
        {
            var token = Parse(bytes)[Key];
            if (token == null) return;
            if (!(token is JObject)) throw new InvalidDataException("The saved AI configuration is invalid. The save has been preserved.");
            Set(metadata, (JObject)token);
        }
        public static byte[] Merge(GameMetadata metadata, string gameId, string directory, byte[] bytes)
        {
            var data = Get(metadata);
            string path = Path.Combine(directory, "metadata.storage");
            // Replay may supply a vendor clone with only a last-seen revision update.
            if (data == null && File.Exists(path))
            {
                Read(metadata, File.ReadAllBytes(path));
                data = Get(metadata);
            }
            if (data == null) return bytes;
            string previousGame = (string)data["game_id"];
            if (!String.IsNullOrEmpty(previousGame) && previousGame != gameId) throw new InvalidDataException("AI configuration belongs to another game.");
            data["game_id"] = gameId;
            Set(metadata, data);
            var document = Parse(bytes);
            document[Key] = data;
            return Encoding.UTF8.GetBytes(document.ToString(Newtonsoft.Json.Formatting.Indented));
        }
    }
}
