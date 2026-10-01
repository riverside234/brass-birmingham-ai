using System;
using System.IO;
using System.Linq;
using System.Text;
using System.Security.Cryptography;
using Newtonsoft.Json.Linq;
using BrassHost;

namespace BrassLegacyAI
{
    public static class ProfileCatalog
    {
        public const string Runtime = "brass-local-ai/1.4.9";
        public const string OriginalRuntime = "brass-local-ai/1.4.1";
        public const string CashRuntime = "brass-local-ai/1.4.0";
        public const string PreviousRuntime = "brass-local-ai/1.3.0";
        public static readonly string[] Keys = { "balanced", "rail-tempo", "cotton", "goods-beer", "supplier", "pottery-flex" };
        public static readonly string[] Labels = { "Balanced", "Rail Tempo", "Cotton", "Goods/Beer", "Supplier", "Pottery Flex" };
        public static readonly string[] Descriptions = {
            "Adapts to the best available opportunities.", "Favors useful rail links and supporting industries.",
            "Develops cotton investments and prepares profitable sales.", "Combines manufactured goods with brewery support.",
            "Builds coal and iron when supply and demand justify them.", "Pursues feasible pottery opportunities and adapts when needed." };
        public static int Index(string key)
        {
            int index = Array.IndexOf(Keys, key);
            if (index < 0) throw new InvalidDataException("Unknown AI strategy. Restore its matching patch before continuing.");
            return index;
        }
        public static string Label(string key) { return Labels[Index(key)] + " v3"; }
        public static string Description(string key) { return Descriptions[Index(key)]; }
        static string Version(string runtime)
        {
            if(runtime==Runtime || runtime==OriginalRuntime || runtime==CashRuntime)return "v3";
            if(runtime==PreviousRuntime)return "v2";
            throw new InvalidDataException("Unsupported AI runtime.");
        }
        static byte[] Source(string key,string runtime)
        {
            Index(key);
            using (var input = typeof(ProfileCatalog).Assembly.GetManifestResourceStream("BrassLegacyAI.Profiles." + key + "-" + Version(runtime) + (runtime==Runtime?"-masked":runtime==CashRuntime?"-1.4.0":"") + ".json"))
            {
                if (input == null) throw new InvalidDataException("The installed AI package is incomplete.");
                using (var output = new MemoryStream()) { input.CopyTo(output); return output.ToArray(); }
            }
        }
        public static string SourceHash(string key,string runtime=Runtime)
        {
            using (var sha = SHA256.Create()) return BitConverter.ToString(sha.ComputeHash(Source(key,runtime))).Replace("-", "").ToLowerInvariant();
        }
        public static JObject Snapshot(string key, int players, bool introductory)
        { return Snapshot(key,players,introductory,Runtime); }
        public static JObject Snapshot(string key, int players, bool introductory,string runtime)
        {
            if (!Wire.ValidPlayerCount(players)) throw new InvalidDataException("The AI supports two to four players.");
            var source = JObject.Parse(Encoding.UTF8.GetString(Source(key,runtime)).TrimStart('\uFEFF'));
            if ((string)source["id"] != "legacy-hard-" + key + "-" + Version(runtime) + (runtime==Runtime?"-masked":"") || (string)source["mode"] != (runtime==Runtime?"masked_ismcts":"S1") || (int)source["player_count"] != 4 ||
                (runtime==Runtime && ((int?)source["cash_policy"]!=1 || (int?)source["conversion_policy"]!=1)))
                throw new InvalidDataException("The packaged AI profile is not the expected versioned strategy.");
            source["id"] = runtime + "/" + key + "/" + players + (introductory ? "/intro" : "/full");
            source["player_count"] = players;
            Profile.Read(source);
            return source;
        }
        public static Profile Create(string key, int players, bool introductory = false) { return Profile.Read(Snapshot(key, players, introductory)); }
        public static Profile ForPlay(JObject savedProfile)
        {
            // Older saves keep their validated assignments and intent identities.
            // Apply the deployment repair in memory, without rewriting that data.
            var profile=Profile.Read(savedProfile);profile.cash_policy=1;profile.conversion_policy=1;
            profile.mode="masked_ismcts";profile.work_limit=1000000;profile.simulations=100000;return profile;
        }
    }
}
