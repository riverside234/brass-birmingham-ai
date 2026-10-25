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
        public const string Runtime = "brass-local-ai/1.4.20";
        public const string IndustryRuntime = "brass-local-ai/1.4.18";
        public const string EraRuntime = "brass-local-ai/1.4.15";
        public const string StrengthRuntime = "brass-local-ai/1.4.11";
        public const string MaskedRuntime = "brass-local-ai/1.4.9";
        public const string OriginalRuntime = "brass-local-ai/1.4.1";
        public const string CashRuntime = "brass-local-ai/1.4.0";
        public const string PreviousRuntime = "brass-local-ai/1.3.0";
        public static readonly string[] Keys = { "balanced", "rail-tempo", "cotton", "goods-beer", "supplier", "pottery-flex" };
        public static readonly string[] Labels = { "Adaptive Advanced Book", "Coventry Goods Opening", "External-Beer Three Cotton", "Worcester Iron-Cotton", "Zero-Loan Industry", "Advanced Pottery Variants" };
        public static readonly string[] Descriptions = {
            "Selects among advanced hand, seat, external-beer and market-funded books.", "Fourth-seat Coventry access when funded; later goods upgrades and useful expansion.",
            "Targets high cotton using public beer; brews only when conversion needs a contingency.", "Combines iron income with cotton; uses a supported Worcester recovery opening.",
            "Recycles real market receipts into industries; borrowing remains a recovery fallback.", "Adapts brewery, iron and cotton-to-pottery variants to seat, market and available beer." };
        public static int Index(string key)
        {
            int index = Array.IndexOf(Keys, key);
            if (index < 0) throw new InvalidDataException("Unknown AI strategy. Restore its matching patch before continuing.");
            return index;
        }
        public static string Label(string key) { return Labels[Index(key)] + " v4"; }
        public static string Description(string key) { return Descriptions[Index(key)]; }
        static string Version(string runtime)
        {
            if(runtime==Runtime || runtime==IndustryRuntime || runtime==EraRuntime || runtime==StrengthRuntime || runtime==MaskedRuntime || runtime==OriginalRuntime || runtime==CashRuntime)return "v3";
            if(runtime==PreviousRuntime)return "v2";
            throw new InvalidDataException("Unsupported AI runtime.");
        }
        static byte[] Source(string key,string runtime)
        {
            Index(key);
            using (var input = typeof(ProfileCatalog).Assembly.GetManifestResourceStream("BrassLegacyAI.Profiles." + key + "-" + Version(runtime) + (runtime==Runtime?"-masked-economy":runtime==IndustryRuntime?"-masked-industry":runtime==EraRuntime?"-masked-era":runtime==StrengthRuntime?"-masked-strength":runtime==MaskedRuntime?"-masked":runtime==CashRuntime?"-1.4.0":"") + ".json"))
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
            if ((string)source["id"] != (runtime==Runtime || runtime==IndustryRuntime || runtime==EraRuntime?"":"legacy-hard-") + key + "-" + Version(runtime) + (runtime==Runtime?"-masked-economy":runtime==IndustryRuntime?"-masked-industry":runtime==EraRuntime?"-masked-era":runtime==StrengthRuntime?"-masked-strength":runtime==MaskedRuntime?"-masked":"") || (string)source["mode"] != (runtime==Runtime || runtime==IndustryRuntime || runtime==EraRuntime || runtime==StrengthRuntime || runtime==MaskedRuntime?"masked_ismcts":"S1") || (int)source["player_count"] != 4 ||
                (runtime==Runtime && ((int?)source["cash_policy"]!=1 || (int?)source["conversion_policy"]!=1)))
                throw new InvalidDataException("The packaged AI profile is not the expected versioned strategy.");
            source["id"] = runtime + "/" + key + "/" + players + (introductory ? "/intro" : "/full");
            source["player_count"] = players;
            Profile.Read(source);
            return source;
        }
        public static Profile Create(string key, int players, bool introductory = false) { return Profile.Read(Snapshot(key, players, introductory)); }
        public static void ConfigureClock(Profile p,double budget)
        {
            // Called with the ORIGINAL turn allowance, never the remaining clock.
            // A late request must not silently switch evaluation units or horizon.
            p.root_top_k=budget<=2000?4:budget<=5000?6:8;
            p.interior_top_k=budget<=2000?2:budget<=5000?3:4;
            p.opponent_turn_limit=budget<=2000?1:2;
        }
        public static void ConfigureClock(Profile p,double budget,string difficulty)
        {
            // Preserve each tier's existing breadth and reply horizon as its
            // wall-clock allowance changes. Unnamed legacy saves keep the old path.
            // Hard and Max share the existing Hard search shape.
            if(difficulty==DifficultyCatalog.Existing){ConfigureClock(p,budget);return;}
            if(!DifficultyCatalog.Contains(difficulty))throw new InvalidDataException("Unknown AI difficulty.");
            ConfigureClock(p,difficulty=="easy"?2000:difficulty=="mid"?5000:12000);
        }
        public static Profile ForPlay(JObject savedProfile)
        {
            // Older saves keep their validated assignments and intent identities.
            // Apply the deployment repair in memory, without rewriting that data.
            var profile=Profile.Read(savedProfile);profile.cash_policy=1;profile.conversion_policy=1;
            profile.mode="masked_ismcts";profile.work_limit=1000000;profile.simulations=100000;
            profile.search_contract="masked-ismcts/1.1.0";profile.evaluator_kind="heuristic";profile.q_transform="regularized_sibling";
            profile.leaf_policy="settled_reply_prefix";profile.exploration=.6;return profile;
        }
    }
}
