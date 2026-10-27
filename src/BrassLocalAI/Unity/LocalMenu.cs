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
using Frameworks.View.Menu.CreateGame;

namespace BrassLegacyAI
{
    public static class LocalMenu
    {
        sealed class Selection {
            public AiChoice Choice = new AiChoice(AiFamily.V4, "balanced");
            public string Key { get { return Choice.EntryId; } set { Choice = new AiChoice(AiFamily.V4, value); } }
            public string DisplayFamily = "V4";
            public string Difficulty = DifficultyCatalog.Default; public CreateOfflineGameUseCase Owner;
        }
        sealed class SetupState { public bool ResumeError; }
        static readonly ConditionalWeakTable<PlayerMetadata, Selection> Selections = new ConditionalWeakTable<PlayerMetadata, Selection>();
        static readonly ConditionalWeakTable<CreateOfflineGameUseCase, SetupState> Setups = new ConditionalWeakTable<CreateOfflineGameUseCase, SetupState>();
        static readonly System.Collections.Generic.Dictionary<string, string> TestPreferences = new System.Collections.Generic.Dictionary<string, string>();
        static string ReadPreference(string name, string fallback) { string value; return StartupFixes.IsSelfTest ? (TestPreferences.TryGetValue(name, out value) ? value : fallback) : PlayerPrefs.GetString(name, fallback); }
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
        public static string Difficulty(PlayerMetadata player)
        {
            Selection selection;
            return Selections.TryGetValue(player, out selection) ? selection.Difficulty : DifficultyCatalog.Default;
        }
        public static AiChoice Choice(PlayerMetadata player)
        { Selection selection; return Selections.TryGetValue(player, out selection) ? selection.Choice : new AiChoice(AiFamily.V4, "balanced"); }
        public static string RowLabel(PlayerMetadata player)
        { if(Choice(player).Family==AiFamily.AlphaTest)return "Alpha Test"; Selection selection; string family = Selections.TryGetValue(player, out selection) ? selection.DisplayFamily : "V4"; return family + " · " + ChoiceLabel(Strategy(player)); }
        public static bool EditChoice(PlayerMetadata player, AiChoice choice)
        {
            // The menu catalog is a second validation boundary, independent of
            // disabled controls. Gamma has no playable backend in this release.
            if (choice.Family != AiFamily.V4 || TableFamily(player)!=AiFamily.V4 || !AiSelectionCatalog.CanChoose(choice) || !IsEditable(player)) return false;
            Select(player, choice.EntryId); Refresh(Selections.GetValue(player, p => new Selection()).Owner); return true;
        }
        public static void SelectDifficulty(PlayerMetadata player, string key)
        {
            if (!DifficultyCatalog.Contains(key)) throw new ArgumentException("Select Easy, Mid, Hard or Max.", "key");
            Selections.GetValue(player, p => new Selection()).Difficulty = key;
        }
        public static bool IsEditable(PlayerMetadata player)
        {
            Selection selection;
            if (player == null || !Selections.TryGetValue(player, out selection) || selection.Owner == null || player.AiType == AIType.Human) return false;
            var metadata = Metadata(selection.Owner);
            return metadata != null && metadata.PlayerMetadata.Take(Field<int>(selection.Owner, "_numberOfPlayers")).Any(p => ReferenceEquals(p, player));
        }
        public static bool EditStrategy(PlayerMetadata player, string key)
        {
            if (AiSelectionPanel.IsOpen || !IsEditable(player) || TableFamily(player)!=AiFamily.V4) return false;
            Select(player, key); Refresh(Selections.GetValue(player, p => new Selection()).Owner); return true;
        }
        public static bool EditDifficulty(PlayerMetadata player, string key)
        {
            if (AiSelectionPanel.IsOpen || !IsEditable(player)) return false;
            SelectDifficulty(player, key); Refresh(Selections.GetValue(player, p => new Selection()).Owner); return true;
        }
        public static readonly string[] Choices = ProfileCatalog.Keys.Concat(new[] { "random" }).ToArray();
        public static string ChoiceLabel(string key) { return key=="alpha-test"?"Alpha Test":key == "random" ? "Random" : ProfileCatalog.Labels[ProfileCatalog.Index(key)]; }
        public static string ChoiceDescription(string key) { return key=="alpha-test"?"Experimental g890 test neural network. All AI seats use Alpha Test.":key == "random" ? "Chooses one of " + ProfileCatalog.Keys.Length + " starting styles. Plans adapt, with balanced fallback when needed." : ProfileCatalog.Description(key); }
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
            StrategyPicker.CloseActive();
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
                    string key = ReadPreference("BrassLocalAI.v1.Strategy." + i, "balanced");
                    if (!Choices.Contains(key)) key = "balanced";
                    Select(player, key);
                    string difficulty = ReadPreference("BrassLocalAI.v1.Difficulty." + i, DifficultyCatalog.Default);
                    SelectDifficulty(player, DifficultyCatalog.Contains(difficulty) ? difficulty : DifficultyCatalog.Default);
                }
                Selections.GetValue(player, p => new Selection()).Owner = owner;
                if (player.AiType != AIType.Human) player.AiType = AIType.Hard;
            }
        }
        public static void Next(CreateOfflineGameUseCase owner, PlayerMetadata player) { Cycle(owner, player, 1); }
        public static void Previous(CreateOfflineGameUseCase owner, PlayerMetadata player) { Cycle(owner, player, -1); }
        static void Cycle(CreateOfflineGameUseCase owner, PlayerMetadata player, int direction)
        {
            if (!IsCustom(owner)) { ReflectionAccess.Original(owner, direction > 0 ? "NextAiDifficulty" : "PreviousAiDifficulty", new[] { typeof(PlayerMetadata) }, player); return; }
            if (player.AiType == AIType.Human) return;
            if(TableFamily(player)!=AiFamily.V4)return;
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
            if(players==null || count<2 || count>4 || players.Length!=count || !players.Any(p=>p.AiType==AIType.Human))return false;
            var families=players.Where(p=>p.AiType!=AIType.Human).Select(p=>Choice(p).Family).Distinct().ToArray();
            return families.Length<=1 && players.All(p=>p.PlayerStartConfiguration!=null && (p.AiType==AIType.Human ||
                (AiSelectionCatalog.CanChoose(Choice(p)) && DifficultyCatalog.Contains(Difficulty(p))))) &&
                (!families.Contains(AiFamily.AlphaTest) || NativeSupported(players[0]));
        }
        public static BrassApplicationGame StartGame(CreateOfflineGameUseCase owner, int seed)
        {
            if (IsCustom(owner) && AiSelectionPanel.BlocksMenuActivation) return null;
            if (!IsCustom(owner)) return (BrassApplicationGame)ReflectionAccess.Original(owner, "StartGame", new[] { typeof(int) }, seed);
            try
            {
                StrategyPicker.CloseActive();
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
                    if(Choice(Metadata(owner).PlayerMetadata[i]).Family==AiFamily.AlphaTest)
                        Selections.GetValue(player,p=>new Selection()).Choice=new AiChoice(AiFamily.AlphaTest,"alpha-test");
                    else Select(player, player.AiType == AIType.Human ? choice : ResolveStrategy(choice));
                    SelectDifficulty(player, Difficulty(Metadata(owner).PlayerMetadata[i]));
                }
                var state = metadata.PlayerMetadata.Any(p=>p.AiType!=AIType.Human && Choice(p).Family==AiFamily.AlphaTest) ?
                    Native.NativeGameState.Create(metadata,Difficulty) : LocalGameState.Create(metadata, Strategy, Difficulty);
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
                for (int i = 0; i < count; i++) {
                    WritePreference("BrassLocalAI.v1.Strategy." + i, Strategy(Metadata(owner).PlayerMetadata[i]));
                    WritePreference("BrassLocalAI.v1.Difficulty." + i, Difficulty(Metadata(owner).PlayerMetadata[i]));
                }
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
            if (IsCustom(owner) && AiSelectionPanel.BlocksMenuActivation) return;
            if (!IsCustom(owner)) { ReflectionAccess.Original(owner, "ContinueClicked", Type.EmptyTypes); return; }
            try
            {
                StrategyPicker.CloseActive();
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
            foreach (var p in app.Metadata.PlayerMetadata.Where(p => p.AiType != AIType.Human)) {
                var row = LocalGameState.Assignment(state, (int)p.PlayerStartConfiguration.Color);
                if(Native.NativeGameState.IsNative(state))Selections.GetValue(p,x=>new Selection()).Choice=new AiChoice(AiFamily.AlphaTest,"alpha-test");
                else Select(p, (string)row["strategy"]);
                var selection = Selections.GetValue(p, x => new Selection()); selection.Difficulty = LocalGameState.Difficulty(state, row); selection.Owner = null;
                selection.DisplayFamily = (string)state["runtime"] == ProfileCatalog.Runtime ? "V4" :
                    (string)state["runtime"] == ProfileCatalog.PreviousRuntime ? "V2" : "V3";
            }
        }
        public static void Back(CreateOfflineGameUseCase owner)
        {
            if (IsCustom(owner) && StrategyPicker.DismissBack()) return;
            StrategyTooltip.HideActive();
            ReflectionAccess.Original(owner, "BackClicked", Type.EmptyTypes);
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
            // The router configures rows before enabling the cached menu.
            if (!view.GetComponentsInParent<CreateOfflineGameView>(true).Any()) return;
            var label = Field<TextMeshProUGUI>(view, "_playerAIDifficultyText");
            Selection selection = null; bool known = model.PlayerMetadata != null && Selections.TryGetValue(model.PlayerMetadata, out selection);
            if (model.Editable && (!known || selection.Owner == null)) return;
            if (model.AiType != AIType.Human)
            {
                label.text = known ? ChoiceLabel(Strategy(model.PlayerMetadata)) : "Original AI";
                label.color = new Color(0.95f, 0.90f, 0.78f, 1f);
                label.enableAutoSizing = true; label.fontSizeMin = 15f; label.fontSizeMax = 25f;
                label.enableWordWrapping = false;
            }
            foreach (var button in view.GetComponentsInChildren<Button>(true))
                for (int i = 0; i < button.onClick.GetPersistentEventCount(); i++)
                    if (button.onClick.GetPersistentMethodName(i) == "OnNextDifficultyClick" || button.onClick.GetPersistentMethodName(i) == "OnPreviousDifficultyClick")
                    { button.interactable = false; button.gameObject.SetActive(false); }
            var controls = view.GetComponent<AiSetupRow>() ?? view.gameObject.AddComponent<AiSetupRow>();
            controls.Bind(view, model, known);
        }
        public static void ShowError(IRouter router, string message, Exception error = null)
        {
            if (error != null) Debug.LogException(error);
            router.ShowView<IQuestionPopupView>();
            Action close = () => router.HideView<IQuestionPopupView>();
            router.GetView<IQuestionPopupView>().Setup(message, false, "OK", "Close", close, close);
        }
        public static object SetupIdentity(PlayerMetadata player)
        { Selection selection; return player != null && Selections.TryGetValue(player, out selection) ? selection.Owner : null; }
        public static bool CanConfigure(PlayerMetadata player)
        {
            var owner = SetupIdentity(player) as CreateOfflineGameUseCase;
            if (owner == null || !IsCustom(owner)) return false;
            var metadata = Metadata(owner);
            return metadata != null && metadata.PlayerMetadata.Take(Field<int>(owner, "_numberOfPlayers")).Any(p => ReferenceEquals(p, player));
        }
        public static bool NeedsHuman(PlayerMetadata player)
        {
            var owner = SetupIdentity(player) as CreateOfflineGameUseCase;
            return CanConfigure(player) && !Metadata(owner).PlayerMetadata.Take(Field<int>(owner, "_numberOfPlayers")).Any(p => p.AiType == AIType.Human);
        }
        public static AiFamily TableFamily(PlayerMetadata player)
        {
            var owner=SetupIdentity(player) as CreateOfflineGameUseCase;
            return owner!=null && Metadata(owner).PlayerMetadata.Take(Field<int>(owner,"_numberOfPlayers")).Any(p=>p.AiType!=AIType.Human && Choice(p).Family==AiFamily.AlphaTest)?AiFamily.AlphaTest:AiFamily.V4;
        }
        public static bool NativeSupported(PlayerMetadata player)
        {
            var owner=SetupIdentity(player) as CreateOfflineGameUseCase;
            return owner!=null && Field<int>(owner,"_numberOfPlayers")==4 && !Metadata(owner).GameStartConfiguration.IsIntroductoryGame && Native.NativePackage.Enabled;
        }
        public static bool ApplyPlayerSelection(PlayerMetadata player, object setupIdentity, PlayerSelectionDraft draft)
        {
            if (AiSelectionPanel.IsOpen || draft == null || !CanConfigure(player) || !ReferenceEquals(SetupIdentity(player), setupIdentity)) return false;
            bool neural=draft.Choice.Family==AiFamily.AlphaTest;
            if(neural && !NativeSupported(player))return false;
            if (draft.Page != PlayerSelectionPage.Human && ((draft.Page != (neural?PlayerSelectionPage.AlphaTest:PlayerSelectionPage.V4)) ||
                !AiSelectionCatalog.CanChoose(draft.Choice) || !DifficultyCatalog.Contains(draft.Difficulty))) return false;
            var owner=SetupIdentity(player) as CreateOfflineGameUseCase;
            foreach(var seat in Metadata(owner).PlayerMetadata) {
                var other=Selections.GetValue(seat,p=>new Selection());
                if(neural)other.Choice=new AiChoice(AiFamily.AlphaTest,"alpha-test");
                else if(other.Choice.Family==AiFamily.AlphaTest)other.Choice=new AiChoice(AiFamily.V4,"balanced");
            }
            var selection = Selections.GetValue(player, p => new Selection());
            // Unlike the stock cycling control, changing controller does not
            // overwrite the user's nickname. AI behavior is created at Start.
            if (draft.Page == PlayerSelectionPage.Human) player.AiType = AIType.Human;
            else { selection.Choice = draft.Choice; selection.Difficulty = draft.Difficulty; player.AiType = AIType.Hard; }
            Refresh(selection.Owner); return true;
        }
    }
    public sealed class StrategyTooltip : MonoBehaviour, IPointerEnterHandler, IPointerExitHandler, ISelectHandler, IDeselectHandler
    {
        public string Description;
        static StrategyTooltip active;
        GameObject panel;
        public void OnPointerEnter(PointerEventData e) { Show(); }
        public void OnSelect(BaseEventData e) { Show(); }
        public void OnPointerExit(PointerEventData e) { Hide(); }
        public void OnDeselect(BaseEventData e) { Hide(); }
        void Show()
        {
            if (StrategyPicker.IsOpen || String.IsNullOrEmpty(Description)) return;
            var canvas = GetComponentInParent<Canvas>(); var font = GetComponentInChildren<TextMeshProUGUI>(true);
            if (canvas == null || font == null) return;
            HideActive(); active = this;
            var parent = (RectTransform)canvas.rootCanvas.transform;
            var r = AiWidgets.Rect(parent, "AI selection help", Vector2.zero, new Vector2(400, 100)); panel = r.gameObject;
            panel.AddComponent<BrassPatch.UI.LightUiTheme>();
            r.anchorMin = r.anchorMax = r.pivot = new Vector2(.5f, .5f);
            var surface = panel.AddComponent<AiControlSurface>(); surface.Panel = true; surface.raycastTarget = false;
            var text = AiWidgets.Text(r, font, "Explanation", Description, new Vector2(12, -10), new Vector2(376, 100), 19); text.enableWordWrapping = true;
            float height = text.GetPreferredValues(Description, 376, 500).y + 22;
            text.rectTransform.sizeDelta = new Vector2(376, height - 20); r.sizeDelta = new Vector2(400, height);
            var point = parent.InverseTransformPoint(transform.position); var bounds = parent.rect;
            r.anchoredPosition = new Vector2(Mathf.Clamp(point.x, bounds.xMin + 210, bounds.xMax - 210), Mathf.Clamp(point.y + height / 2 + 32, bounds.yMin + height / 2 + 10, bounds.yMax - height / 2 - 10)) - bounds.center;
        }
        public static void HideActive() { if (!ReferenceEquals(active, null)) active.Hide(); }
        void Hide() { if (active != this) return; active = null; if (panel != null) { panel.SetActive(false); Destroy(panel); panel = null; } }
        void OnDisable() { Hide(); }
        void Update()
        {
            if (panel == null) return;
            var view = GetComponentInParent<Frameworks.View.Common.BaseView>();
            if (view == null || !view.IsViewVisible()) Hide();
        }
    }
}

