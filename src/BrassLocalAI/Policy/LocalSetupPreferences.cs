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
            public string NeuralFocus = "neutral";
            public AiFamily Family;
            public string NeuralChoice = "alpha-test";
            public int SearchK = 16;
            public double Temperature = 1.0;
        }
        public AiFamily Family;
        public int PlayerCount;
        public bool Introductory;
        public int SearchK = 16;
        public double Temperature = 1.0;
        public string NeuralChoice = "alpha-test";
        public Seat[] Seats;

        public string Encode()
        {
            return new JObject {
                ["version"] = 3,
                ["players"] = PlayerCount, ["introductory"] = Introductory,
                ["seats"] = new JArray(Seats.Select(s => new JObject {
                    ["human"] = s.Human, ["style"] = s.Style, ["difficulty"] = s.Difficulty, ["nn_focus"] = s.NeuralFocus,
                    ["family"] = s.Family == AiFamily.AlphaTest ? "nn" : "v4", ["nn_model"] = s.NeuralChoice,
                    ["nn_search_k"] = s.SearchK, ["nn_temperature"] = s.Temperature }))
            }.ToString(Newtonsoft.Json.Formatting.None);
        }

        public static LocalSetupPreferences Decode(string text)
        {
            if (String.IsNullOrEmpty(text) || text.Length > 8192) return null;
            try {
                var value = JObject.Parse(text);
                int? version = (int?)value["version"];
                if ((version != 2 && version != 3) || value["players"]?.Type != JTokenType.Integer ||
                    value["introductory"]?.Type != JTokenType.Boolean) return null;
                string family = version == 2 ? (string)value["family"] : "v4";
                var seats = value["seats"] as JArray;
                int count = (int)value["players"];
                var k = value["nn_search_k"];
                if (k != null && (k.Type != JTokenType.Integer || !Native.NativePackage.ValidSearchK((int)k))) return null;
                var temperature = value["nn_temperature"];
                var model = value["nn_model"];
                if(model!=null && (model.Type!=JTokenType.String || !Native.NativeModel.ValidChoice((string)model)))return null;
                if (temperature != null && ((temperature.Type != JTokenType.Integer && temperature.Type != JTokenType.Float) ||
                    !Native.NativePackage.ValidTemperature((double)temperature))) return null;
                if ((family != "v4" && family != "nn") || count < 2 || count > 4 || seats == null ||
                    seats.Count < count || seats.Count > 4) return null;
                var result = new LocalSetupPreferences {
                    Family = family == "nn" ? AiFamily.AlphaTest : AiFamily.V4,
                    PlayerCount = count, Introductory = (bool)value["introductory"],
                    SearchK = k == null || (int)k == 32 ? 16 : (int)k,
                    Temperature = temperature == null ? 1.0 : (double)temperature,
                    NeuralChoice = model == null ? "alpha-test" : (string)model,
                    Seats = new Seat[seats.Count]
                };
                for (int i = 0; i < seats.Count; i++) {
                    var seat = seats[i] as JObject;
                    if (seat == null || seat["human"]?.Type != JTokenType.Boolean ||
                        seat["style"]?.Type != JTokenType.String || seat["difficulty"]?.Type != JTokenType.String ||
                        !LocalMenu.Choices.Contains((string)seat["style"]) ||
                        !DifficultyCatalog.Contains((string)seat["difficulty"])) return null;
                    var focus = seat["nn_focus"];
                    if (focus != null && (focus.Type != JTokenType.String || !Native.NativeFocus.Valid((string)focus))) return null;
                    string seatFamily = version == 2 ? family : (string)seat["family"];
                    string seatModel = version == 2 ? result.NeuralChoice : (string)seat["nn_model"];
                    var seatK = version == 2 ? k : seat["nn_search_k"];
                    var seatT = version == 2 ? temperature : seat["nn_temperature"];
                    if ((seatFamily != "v4" && seatFamily != "nn") || !Native.NativeModel.ValidChoice(seatModel) ||
                        (version == 3 && (seatK == null || seatT == null)) ||
                        (seatK != null && (seatK.Type != JTokenType.Integer || !Native.NativePackage.ValidSearchK((int)seatK))) ||
                        (seatT != null && ((seatT.Type != JTokenType.Float && seatT.Type != JTokenType.Integer) || !Native.NativePackage.ValidTemperature((double)seatT)))) return null;
                    result.Seats[i] = new Seat { Human = (bool)seat["human"], Style = (string)seat["style"], Difficulty = (string)seat["difficulty"],
                        Family = seatFamily == "nn" ? AiFamily.AlphaTest : AiFamily.V4, NeuralChoice = seatModel,
                        SearchK = seatK == null || (int)seatK == 32 ? 16 : (int)seatK, Temperature = seatT == null ? 1.0 : (double)seatT,
                        // Menu defaults now offer only the neutral NN. Retain
                        // the remaining settings when reading a former tag choice.
                        NeuralFocus = "neutral" };
                }
                if (version == 2 && result.Family == AiFamily.AlphaTest && (count != 4 || result.Introductory)) return null;
                return result;
            } catch (Exception e) when (e is Newtonsoft.Json.JsonException || e is ArgumentException || e is FormatException || e is InvalidCastException || e is OverflowException) {
                return null;
            }
        }
    }
}
