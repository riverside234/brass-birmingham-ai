using System;
using System.Linq;
using Newtonsoft.Json.Linq;

namespace BrassLegacyAI
{
    // Menu defaults only. These never replace a saved game's signed assignments.
    internal sealed class LocalSetupPreferences
    {
        internal const string Key = "BrassLocalAI.Setup.v2";
        internal sealed class Seat
        {
            public bool Human;
            public string Style, Difficulty;
        }
        public AiFamily Family;
        public int PlayerCount;
        public bool Introductory;
        public int SearchK = 16;
        public double Temperature = 1.0;
        public Seat[] Seats;

        public string Encode()
        {
            return new JObject {
                ["version"] = 2, ["family"] = Family == AiFamily.AlphaTest ? "nn" : "v4",
                ["players"] = PlayerCount, ["introductory"] = Introductory,
                ["nn_search_k"] = SearchK,
                ["nn_temperature"] = Temperature,
                ["seats"] = new JArray(Seats.Select(s => new JObject {
                    ["human"] = s.Human, ["style"] = s.Style, ["difficulty"] = s.Difficulty }))
            }.ToString(Newtonsoft.Json.Formatting.None);
        }

        public static LocalSetupPreferences Decode(string text)
        {
            if (String.IsNullOrEmpty(text) || text.Length > 8192) return null;
            try {
                var value = JObject.Parse(text);
                if ((int?)value["version"] != 2 || value["players"]?.Type != JTokenType.Integer ||
                    value["introductory"]?.Type != JTokenType.Boolean) return null;
                string family = (string)value["family"];
                var seats = value["seats"] as JArray;
                int count = (int)value["players"];
                var k = value["nn_search_k"];
                if (k != null && (k.Type != JTokenType.Integer || ((int)k != 8 && (int)k != 16))) return null;
                var temperature = value["nn_temperature"];
                if (temperature != null && ((temperature.Type != JTokenType.Integer && temperature.Type != JTokenType.Float) ||
                    !Native.NativePackage.ValidTemperature((double)temperature))) return null;
                if ((family != "v4" && family != "nn") || count < 2 || count > 4 || seats == null ||
                    seats.Count < count || seats.Count > 4) return null;
                var result = new LocalSetupPreferences {
                    Family = family == "nn" ? AiFamily.AlphaTest : AiFamily.V4,
                    PlayerCount = count, Introductory = (bool)value["introductory"],
                    SearchK = k == null ? 16 : (int)k,
                    Temperature = temperature == null ? 1.0 : (double)temperature,
                    Seats = new Seat[seats.Count]
                };
                for (int i = 0; i < seats.Count; i++) {
                    var seat = seats[i] as JObject;
                    if (seat == null || seat["human"]?.Type != JTokenType.Boolean ||
                        seat["style"]?.Type != JTokenType.String || seat["difficulty"]?.Type != JTokenType.String ||
                        !LocalMenu.Choices.Contains((string)seat["style"]) ||
                        !DifficultyCatalog.Contains((string)seat["difficulty"])) return null;
                    result.Seats[i] = new Seat { Human = (bool)seat["human"], Style = (string)seat["style"], Difficulty = (string)seat["difficulty"] };
                }
                if (result.Family == AiFamily.AlphaTest && (count != 4 || result.Introductory)) return null;
                return result;
            } catch (Exception e) when (e is Newtonsoft.Json.JsonException || e is ArgumentException || e is FormatException || e is InvalidCastException || e is OverflowException) {
                return null;
            }
        }
    }
}
