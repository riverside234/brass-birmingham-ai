using System;
using System.Collections.Generic;
using System.Linq;

namespace BrassLegacyAI
{
    public enum AiFamily { V4, Gamma, AlphaTest }
    public struct AiChoice : IEquatable<AiChoice>
    {
        public readonly AiFamily Family;
        public readonly string EntryId;
        public AiChoice(AiFamily family, string entryId)
        { Family = family; EntryId = entryId ?? ""; }
        public bool Equals(AiChoice other) { return Family == other.Family && String.Equals(EntryId, other.EntryId, StringComparison.Ordinal); }
        public override bool Equals(object other) { return other is AiChoice && Equals((AiChoice)other); }
        public override int GetHashCode() { return ((int)Family * 397) ^ (EntryId ?? "").GetHashCode(); }
        public override string ToString() { return (Family == AiFamily.Gamma ? "gamma" : Family == AiFamily.AlphaTest ? "nn" : "v4") + "/" + EntryId; }
    }
    public sealed class AiCatalogEntry
    {
        public readonly AiChoice Choice;
        public readonly string Name, Description, AvailabilityReason;
        public readonly bool Available;
        public AiCatalogEntry(AiFamily family, string id, string name, string description, bool available, string reason = "")
        { Choice = new AiChoice(family, id); Name = name; Description = description; Available = available; AvailabilityReason = reason; }
    }
    public static class AiSelectionCatalog
    {
        public const int GammaSlots = 16;
        static readonly IReadOnlyList<AiCatalogEntry> v4 = Array.AsReadOnly(ProfileCatalog.Keys.Select((key, index) =>
            new AiCatalogEntry(AiFamily.V4, key, ProfileCatalog.Labels[index], ProfileCatalog.Descriptions[index], true)).ToArray());
        // Intentionally no file discovery or inference backend. UI fixtures and
        // training artifacts cannot enable a neural opponent in this release.
        static readonly IReadOnlyList<AiCatalogEntry> gamma = Array.AsReadOnly(Enumerable.Range(1, GammaSlots).Select(index =>
            new AiCatalogEntry(AiFamily.Gamma, "gamma-" + index.ToString("00"), "Gamma " + index.ToString("00"),
                "Neural AI slot; training pending.", false, "Awaiting model")).ToArray());
        public static IReadOnlyList<AiCatalogEntry> Entries(AiFamily family)
        {
            if (family == AiFamily.V4) return v4;
            if (family == AiFamily.Gamma) return gamma;
            if (family == AiFamily.AlphaTest) return Array.AsReadOnly(new[]{new AiCatalogEntry(family,"alpha-test","Alpha Test",
                "Current test neural network (g890). Experimental local opponent; this is not the final trained model.",Native.NativePackage.Enabled,"Test runtime is not installed")});
            throw new ArgumentOutOfRangeException("family");
        }
        public static AiCatalogEntry Random(AiFamily family)
        {
            int count = Entries(family).Count(e => e.Available);
            return new AiCatalogEntry(family, "random", "Random", family == AiFamily.V4 ?
                "Choose from all six V4 starting styles at game creation. Plans adapt, with balanced fallback when needed. Search does not narrow Random." :
                "Choose from available Gamma models at game creation. Models and the neural game backend are not installed yet. Search does not narrow Random.",
                count > 0, count == 0 ? "Models not installed" : "");
        }
        public static AiCatalogEntry Find(AiChoice choice)
        {
            if (choice.Family != AiFamily.V4 && choice.Family != AiFamily.Gamma && choice.Family != AiFamily.AlphaTest) return null;
            if (choice.Family == AiFamily.AlphaTest && choice.EntryId != "alpha-test") return null;
            return choice.EntryId == "random" ? Random(choice.Family) : Entries(choice.Family).FirstOrDefault(e => e.Choice.Equals(choice));
        }
        public static bool CanChoose(AiChoice choice)
        { var entry = Find(choice); return entry != null && entry.Available; }
    }
}
