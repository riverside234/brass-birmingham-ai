using System;
using System.IO;
using Newtonsoft.Json.Linq;

namespace BrassLegacyAI
{
    // Execution settings are independent of immutable research profiles and intent IDs.
    public static class DifficultyCatalog
    {
        public const string Version = "local-ai-execution/1.1.0";
        public const string PreviousVersion = "local-ai-execution/1.0.0";
        public const string Default = "hard";
        public const string Existing = "existing";
        public const int ExistingMilliseconds = 10000;
        public static readonly string[] Keys = { "easy", "mid", "hard" };
        public static bool Contains(string key) { return Array.IndexOf(Keys, key) >= 0; }
        public static int Milliseconds(string key)
        {
            switch (key) {
                case "easy": return 6000;
                case "mid": return 10000;
                case "hard": return 15000;
                case Existing: return ExistingMilliseconds;
                default: throw new InvalidDataException("Unknown AI difficulty. The saved game has been preserved.");
            }
        }
        public static string Label(string key)
        {
            switch (key) {
                case "easy": return "Easy"; case "mid": return "Mid"; case "hard": return "Hard"; case Existing: return "Existing";
                default: throw new InvalidDataException("Unknown AI difficulty.");
            }
        }
        public static string Summary(string key) { return Label(key) + " · " + Milliseconds(key) / 1000 + "s"; }
        public static string Description(string key) { return "Allows up to " + Milliseconds(key) / 1000 + " seconds of search for this AI's turn. It may finish sooner. Preparing the board and playing animations can add time."; }
        public static JObject Snapshot(string key)
        {
            if (!Contains(key)) throw new InvalidDataException("Select Easy, Mid or Hard for a new AI player.");
            return new JObject { ["version"] = Version, ["difficulty"] = key, ["search_ms"] = Milliseconds(key) };
        }
        public static string Read(JToken token)
        {
            var value = token as JObject;
            if (value == null || value.Count != 3 || value["version"]?.Type != JTokenType.String ||
                ((string)value["version"] != Version && (string)value["version"] != PreviousVersion) || value["difficulty"]?.Type != JTokenType.String ||
                !Contains((string)value["difficulty"]) || value["search_ms"]?.Type != JTokenType.Integer ||
                (long)value["search_ms"] != ((string)value["version"] == PreviousVersion ? PreviousMilliseconds((string)value["difficulty"]) : Milliseconds((string)value["difficulty"])))
                throw new InvalidDataException("Saved AI difficulty is incomplete or incompatible. The save has been preserved.");
            return (string)value["difficulty"];
        }
        static int PreviousMilliseconds(string key)
        {
            // Validate old records without rewriting their signed assignments.
            // Named difficulties use the current allowance when play resumes.
            switch (key) { case "easy": return 2000; case "mid": return 5000; case "hard": return 12000; default: throw new InvalidDataException("Unknown AI difficulty."); }
        }
    }
}
