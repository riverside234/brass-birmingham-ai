namespace BrassLegacyAI
{
    // Navigation includes Human without treating it as an executable AI family.
    public enum PlayerSelectionPage { Human, V4, Gamma, AlphaTest }
    public sealed class PlayerSelectionDraft
    {
        public readonly PlayerSelectionPage Page;
        public readonly AiChoice Choice;
        public readonly string Difficulty;
        public PlayerSelectionDraft(PlayerSelectionPage page, AiChoice choice, string difficulty)
        { Page = page; Choice = choice; Difficulty = difficulty; }
    }
}
