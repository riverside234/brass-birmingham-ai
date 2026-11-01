using System;
using System.Collections.Generic;
using System.Linq;
using Newtonsoft.Json.Linq;

namespace BrassLegacyAI.Native
{
    // Keep this order/encoding identical to step5.policy.context_for. These are
    // conditioning inputs to the installed test model, not reward bonuses.
    public static class NativeFocus
    {
        public static readonly IReadOnlyList<string> Keys = Array.AsReadOnly(new[] {
            "neutral", "cotton", "manufacturer", "pottery", "brewery", "iron", "coal" });
        public static readonly IReadOnlyList<string> Labels = Array.AsReadOnly(new[] {
            "Neutral", "Cotton", "Manufacturer", "Pottery", "Brewery", "Iron", "Coal" });
        public static bool Valid(string focus) { return Keys.Contains(focus); }
        public static string ChoiceId(string focus)
        { Check(focus); return focus == "neutral" ? "alpha-test" : "alpha-test-" + focus; }
        public static string FromChoice(string id)
        { return Keys.FirstOrDefault(f => ChoiceId(f) == id); }
        public static string Label(string focus) { Check(focus); return Labels[Keys.ToList().IndexOf(focus)]; }
        public static string Profile(string focus)
        { Check(focus); return "alpha-test-g890-" + focus + "/1.0.0"; }
        public static JArray Style(string focus)
        {
            Check(focus); int i = Keys.ToList().IndexOf(focus);
            return new JArray(i >= 1 && i <= 3 ? 1.0 : 0.0, 0.0, i >= 4 ? 1.0 : 0.0, 0.0);
        }
        public static JArray Intent(string focus)
        { Check(focus); int i = Keys.ToList().IndexOf(focus) - 1; return new JArray(Enumerable.Range(0, 8).Select(j => j == i ? 1.0 : 0.0)); }
        static void Check(string focus)
        { if (!Valid(focus)) throw new ArgumentException("Unsupported NN industry focus", "focus"); }
    }
}
