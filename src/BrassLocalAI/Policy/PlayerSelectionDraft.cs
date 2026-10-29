namespace BrassLegacyAI
{
    // Navigation includes Human without treating it as an executable AI family.
    public enum PlayerSelectionPage { Human, V4, Gamma, AlphaTest }
    public sealed class PlayerSelectionDraft
    {
        public readonly PlayerSelectionPage Page;
        public readonly AiChoice Choice;
        public readonly string Difficulty;
        public readonly int SearchK;
        public PlayerSelectionDraft(PlayerSelectionPage page, AiChoice choice, string difficulty) : this(page, choice, difficulty, 16) { }
        public PlayerSelectionDraft(PlayerSelectionPage page, AiChoice choice, string difficulty, int searchK)
        { Page = page; Choice = choice; Difficulty = difficulty; SearchK = searchK; }
    }
}
