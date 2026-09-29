using System;
using System.Linq;
using System.Runtime.CompilerServices;
using BrassPatch;
using BrassPatch.Persistence;
using BrassApplication.AI;
using BrassApplication.AI.AiBehaviors;
using BrassApplication.ApplicationSettings;
using BrassApplication.Game;
using BrassApplication.GameHistory;
using BrassApplication.UseCases.Menu.OfflineGame.CreateOfflineGame;
using InterfaceAdapters.Common;
using InterfaceAdapters.Common.CreateGame;
using InterfaceAdapters.Routing;
using Newtonsoft.Json.Linq;
using TMPro;
using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.UI;

namespace BrassLegacyAI
{
    public static class LocalMenu
    {
        sealed class Selection { public string Key; }
        sealed class SetupState { public bool ResumeError; }
        static readonly ConditionalWeakTable<PlayerMetadata, Selection> Selections = new ConditionalWeakTable<PlayerMetadata, Selection>();
        static readonly ConditionalWeakTable<CreateOfflineGameUseCase, SetupState> Setups = new ConditionalWeakTable<CreateOfflineGameUseCase, SetupState>();
        static readonly System.Collections.Generic.Dictionary<string, string> TestPreferences = new System.Collections.Generic.Dictionary<string, string>();
        static string ReadPreference(string name) { string value; return StartupFixes.IsSelfTest ? (TestPreferences.TryGetValue(name, out value) ? value : "balanced") : PlayerPrefs.GetString(name, "balanced"); }
        static void WritePreference(string name, string value) { if (StartupFixes.IsSelfTest) TestPreferences[name] = value; else PlayerPrefs.SetString(name, value); }
        static bool IsCustom(CreateOfflineGameUseCase owner) { return owner.GetType() == typeof(CreateOfflineGameUseCase); }
        static T Field<T>(object owner, string name) { return ReflectionAccess.Get<T>(owner, name); }
        static GameMetadata Metadata(CreateOfflineGameUseCase owner) { return Field<GameMetadata>(owner, "_newGameMetadata"); }
        static IRouter Router(CreateOfflineGameUseCase owner) { return Field<IRouter>(Field<ICreateOfflineGameOutput>(owner, "_presenter"), "_router"); }
        public static string Strategy(PlayerMetadata player)
        {
            Selection selection;
            return Selections.TryGetValue(player, out selection) ? selection.Key : "balanced";
        }
        public static void Select(PlayerMetadata player, string key)
        {
            if (key != "random") ProfileCatalog.Index(key);
            Selections.GetValue(player, p => new Selection()).Key = key;
        }
        public static readonly string[] Choices = ProfileCatalog.Keys.Concat(new[] { "random" }).ToArray();
        public static string ChoiceLabel(string key) { return key == "random" ? "Random" : ProfileCatalog.Label(key); }
        public static string ChoiceDescription(string key) { return key == "random" ? "Chooses one of six styles for this game. Plans adapt, with balanced fallback when needed." : ProfileCatalog.Description(key); }
        static string ResolveStrategy(string key)
        {
            if (key != "random") return key;
            // Independent of the game's shuffle RNG. Rejection sampling avoids a
            // modulo bias, and resolving here prevents rerolls on refresh/continue.
            using (var random = System.Security.Cryptography.RandomNumberGenerator.Create())
            {
                var sample = new byte[1]; int limit = 256 - 256 % ProfileCatalog.Keys.Length;
                do { random.GetBytes(sample); } while (sample[0] >= limit);
                return ProfileCatalog.Keys[sample[0] % ProfileCatalog.Keys.Length];
            }
        }
        static void Refresh(CreateOfflineGameUseCase owner)
        {
            typeof(CreateOfflineGameUseCase).GetMethod("PresentGameConfiguration", ReflectionAccess.Flags).Invoke(owner, null);
        }
        public static void CreateConfiguration(CreateOfflineGameUseCase owner)
        {
            LocalTurnBridge.CancelActive();
            Setups.Remove(owner);
            ReflectionAccess.Original(owner, "CreateNewGameConfiguration", Type.EmptyTypes);
            Normalize(owner);
            Refresh(owner);
        }
        public static void Normalize(CreateOfflineGameUseCase owner)
        {
            if (!IsCustom(owner)) return;
            var metadata = Metadata(owner);
            if (metadata == null) return;
            for (int i = 0; i < metadata.PlayerMetadata.Count; i++)
            {
                var player = metadata.PlayerMetadata[i]; Selection selection;
                if (!Selections.TryGetValue(player, out selection))
                {
                    string key = ReadPreference("BrassLocalAI.v1.Strategy." + i);
                    if (!Choices.Contains(key)) key = "balanced";
                    Select(player, key);
                }
                if (player.AiType != AIType.Human) player.AiType = AIType.Hard;
            }
        }
        public static void Next(CreateOfflineGameUseCase owner, PlayerMetadata player) { Cycle(owner, player, 1); }
        public static void Previous(CreateOfflineGameUseCase owner, PlayerMetadata player) { Cycle(owner, player, -1); }
        static void Cycle(CreateOfflineGameUseCase owner, PlayerMetadata player, int direction)
        {
            if (!IsCustom(owner)) { ReflectionAccess.Original(owner, direction > 0 ? "NextAiDifficulty" : "PreviousAiDifficulty", new[] { typeof(PlayerMetadata) }, player); return; }
            if (player.AiType == AIType.Human) return;
            Select(player, Choices[(Array.IndexOf(Choices, Strategy(player)) + direction + Choices.Length) % Choices.Length]);
            player.AiType = AIType.Hard;
            Refresh(owner);
        }
        public static bool CanStart(CreateOfflineGameUseCase owner)
        {
            if (!IsCustom(owner)) return (bool)ReflectionAccess.Original(owner, "CanStartGame", Type.EmptyTypes);
            if (Setups.GetValue(owner, x => new SetupState()).ResumeError) return false;
            int count = Field<int>(owner, "_numberOfPlayers");
            var players = Metadata(owner)?.PlayerMetadata.Take(count).ToArray();
            return players != null && count >= 2 && count <= 4 && players.Length == count && players.Any(p => p.AiType == AIType.Human) &&
                players.All(p => p.PlayerStartConfiguration != null && (p.AiType == AIType.Human || Choices.Contains(Strategy(p))));
        }
        public static BrassApplicationGame StartGame(CreateOfflineGameUseCase owner, int seed)
        {
            if (!IsCustom(owner)) return (BrassApplicationGame)ReflectionAccess.Original(owner, "StartGame", new[] { typeof(int) }, seed);
            try
            {
                LocalTurnBridge.CancelActive();
                Normalize(owner);
                if (!CanStart(owner)) throw new InvalidOperationException("This setup cannot be started. Keep at least one human and resolve any saved-game error first.");
                var history = Field<IGameHistory>(owner, "_gameHistory");
                // Validate existing metadata before any new publication. Superseded saves are retained.
                FindResume(history);
                var metadata = (GameMetadata)Metadata(owner).Clone();
                int count = Field<int>(owner, "_numberOfPlayers");
                metadata.PlayerMetadata = metadata.PlayerMetadata.Take(count).ToList();
                // GameMetadata.Clone shares GameStartConfiguration; use a fresh setup for creation.
                var original = Metadata(owner).GameStartConfiguration;
                var config = new BoardGameRules.Entities.Game.GameStartConfiguration((BoardGameRules.Entities.Common.NumberOfPlayers)count) {
                    Seed = seed >= 0 ? seed : original.Seed, IsIntroductoryGame = original.IsIntroductoryGame,
                    Players = metadata.PlayerMetadata.Select(p => p.PlayerStartConfiguration).ToList() };
                metadata.GameStartConfiguration = config;
                for (int i = 0; i < count; i++) {
                    var player = metadata.PlayerMetadata[i]; string choice = Strategy(Metadata(owner).PlayerMetadata[i]);
                    Select(player, player.AiType == AIType.Human ? choice : ResolveStrategy(choice));
                }
                var state = LocalGameState.Create(metadata, Strategy);
                SaveExtensions.Set(metadata, state);
                var app = history.Repository.Create(metadata);
                foreach (var p in app.Metadata.PlayerMetadata) p.AIBehaviour = AiBehaviorFactory.CreateAiBehavior(p);
                app.Game.StartGame();
                foreach (var p in app.Metadata.PlayerMetadata) p.LastSeenEventRevision = 1;
                state["game_id"] = app.Metadata.GameId; state["ready"] = true;
                SaveExtensions.Set(app.Metadata, state);
                history.Repository.Save(app);
                LocalGameState.Read(app.Metadata);
                var settings = Field<IGameSettings>(owner, "_gameSettings");
                settings.SetDefaultGameSettings(app.Metadata, count);
                for (int i = 0; i < count; i++) WritePreference("BrassLocalAI.v1.Strategy." + i, Strategy(Metadata(owner).PlayerMetadata[i]));
                if (!StartupFixes.IsSelfTest) PlayerPrefs.Save();
                history.SetGame(app);
                Field<ICreateOfflineGameOutput>(owner, "_presenter").LaunchGame(app);
                return app;
            }
            catch (Exception error) { ShowError(Router(owner), "The game could not be started. Previous saves have been kept.\n" + error.Message, error); return null; }
        }
        public static GameMetadata FindResume(IGameHistory history)
        {
            var available = history.Repository.GetGamesMetadata().Where(m => m != null).Select(m => new { Metadata = m, State = LocalGameState.Read(m, true) })
                .Where(x => (x.State == null || (bool?)x.State["ready"] == true) && !x.Metadata.GameProgressInformation.IsGameOver).ToArray();
            return available.OrderByDescending(x => x.State == null ? "" : (string)x.State["created_utc"], StringComparer.Ordinal).Select(x => x.Metadata).FirstOrDefault();
        }
        public static void UpdateResume(CreateOfflineGameUseCase owner)
        {
            if (!IsCustom(owner)) { ReflectionAccess.Original(owner, "UpdateResumeOfflineGameView", Type.EmptyTypes); return; }
            var context = Setups.GetValue(owner, x => new SetupState()); context.ResumeError = false;
            var presenter = Field<ICreateOfflineGameOutput>(owner, "_presenter");
            try
            {
                var history = Field<IGameHistory>(owner, "_gameHistory"); var metadata = FindResume(history);
                bool canContinue = metadata != null && !metadata.GameProgressInformation.IsGameOver;
                ReflectionAccess.Set(owner, "_anyGameToContinue", canContinue);
                if (canContinue) { var game = history.Repository.Load(metadata.GameId); LocalGameState.Read(game.Metadata); presenter.ShowExistingGameContainer(game); }
                else presenter.HideExistingGameContainer();
            }
            catch (Exception error)
            {
                context.ResumeError = true; ReflectionAccess.Set(owner, "_anyGameToContinue", true);
                presenter.HideExistingGameContainer();
                ShowError(Router(owner), "A saved game could not be read. Its files have been preserved; creating a replacement is disabled until it is recovered.\n" + error.Message, error);
            }
            Refresh(owner);
        }
        public static void Continue(CreateOfflineGameUseCase owner)
        {
            if (!IsCustom(owner)) { ReflectionAccess.Original(owner, "ContinueClicked", Type.EmptyTypes); return; }
            try
            {
                LocalTurnBridge.CancelActive(); var history = Field<IGameHistory>(owner, "_gameHistory"); var metadata = FindResume(history);
                if (metadata == null) throw new InvalidOperationException("There is no completed save to continue.");
                var game = history.Repository.Load(metadata.GameId); LocalGameState.Read(game.Metadata);
                history.SetGame(game); Field<ICreateOfflineGameOutput>(owner, "_presenter").GameLoaded(game);
            }
            catch (Exception error) { ShowError(Router(owner), "This game could not be resumed. Its save has been preserved.\n" + error.Message, error); }
        }
        public static void ResumeLabels(CreateOfflineGamePresenter presenter, BrassApplicationGame app)
        {
            var state = LocalGameState.Read(app.Metadata);
            if (state == null) return;
            foreach (var p in app.Metadata.PlayerMetadata.Where(p => p.AiType != AIType.Human)) Select(p, (string)LocalGameState.Assignment(state, (int)p.PlayerStartConfiguration.Color)["strategy"]);
        }
        public static void SetRow(PlayerCreateGameView view, PlayerCreateGameViewModel model)
        {
            // Rendering a resume row must not invoke an editing callback (which is null there).
            if (model.PlayerMetadata != null && model.PlayerNumber <= model.CurrentPlayerCount)
                Field<TMP_InputField>(view, "_nicknameText").SetTextWithoutNotify(model.Nickname);
            ReflectionAccess.Original(view, "SetPlayer", new[] { typeof(PlayerCreateGameViewModel) }, model);
            StyleRow(view, model);
        }
        public static void StyleRow(PlayerCreateGameView view, PlayerCreateGameViewModel model)
        {
            if (model.PlayerMetadata == null || model.PlayerNumber > model.CurrentPlayerCount) return;
            var label = Field<TextMeshProUGUI>(view, "_playerAIDifficultyText");
            Selection selection; bool known = Selections.TryGetValue(model.PlayerMetadata, out selection);
            if (model.AiType != AIType.Human)
            {
                label.text = known ? ChoiceLabel(selection.Key) : "Original AI";
                label.color = new Color(0.95f, 0.90f, 0.78f, 1f);
                label.enableAutoSizing = true; label.fontSizeMin = 15f; label.fontSizeMax = 25f;
                label.enableWordWrapping = false;
                var tooltip = label.GetComponent<StrategyTooltip>() ?? label.gameObject.AddComponent<StrategyTooltip>();
                tooltip.Description = known ? ChoiceDescription(selection.Key) : "This saved game uses its original AI.";
                label.raycastTarget = true;
            }
            foreach (var button in view.GetComponentsInChildren<Button>(true))
                for (int i = 0; i < button.onClick.GetPersistentEventCount(); i++)
                    if (button.onClick.GetPersistentMethodName(i) == "OnNextDifficultyClick" || button.onClick.GetPersistentMethodName(i) == "OnPreviousDifficultyClick")
                        button.interactable = model.Editable && model.AiType != AIType.Human;
        }
        public static void ShowError(IRouter router, string message, Exception error = null)
        {
            if (error != null) Debug.LogException(error);
            router.ShowView<IQuestionPopupView>();
            Action close = () => router.HideView<IQuestionPopupView>();
            router.GetView<IQuestionPopupView>().Setup(message, false, "OK", "Close", close, close);
        }
    }
    public sealed class StrategyTooltip : MonoBehaviour, IPointerEnterHandler, IPointerExitHandler, ISelectHandler, IDeselectHandler
    {
        public string Description;
        static StrategyTooltip active;
        public void OnPointerEnter(PointerEventData e) { Show(); }
        public void OnSelect(BaseEventData e) { Show(); }
        public void OnPointerExit(PointerEventData e) { Hide(); }
        public void OnDeselect(BaseEventData e) { Hide(); }
        void Show()
        {
            var router = Utils.Global.Instance.Router;
            if (router == null || String.IsNullOrEmpty(Description)) return;
            router.ShowView<ITooltipView>(); active = this;
            var p = transform.position;
            router.GetView<ITooltipView>().ShowTooltip(Description, new System.Numerics.Vector3(p.x, p.y, p.z), new System.Numerics.Vector2(0.5f, 0f));
        }
        void Hide() { if (active != this) return; active = null; var router = Utils.Global.Instance.Router; if (router != null) router.HideView<ITooltipView>(); }
        void OnDisable() { Hide(); }
    }
}
