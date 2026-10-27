using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using BrassHost;
using BoardGameRules.Entities.Common;
using BoardGameRules.Entities.Game;
using BoardGameRules.Entities.Players;
using BrassApplication.AI;
using BrassApplication.ApplicationSettings;
using BrassApplication.Game;
using BrassApplication.GameHistory;
using BrassApplication.Persistence.Repository;
using BrassApplication.UseCases.Common;
using BrassApplication.UseCases.Common.SaveGame;
using BrassApplication.UseCases.Game.PlayAiTurn;
using BrassApplication.UseCases.Game.PlayGame;
using BrassApplication.UseCases.Game.Board;
using BrassApplication.UseCases.Game.HandOfCards;
using BrassApplication.UseCases.Game.PlayerActions;
using BrassApplication.UseCases.Game.Replay;
using BrassApplication.UseCases.Game.PlayHumanTurn;
using InterfaceAdapters.Game.Board;
using BoardGameRules.Entities.Board.Regions;
using BrassApplication.UseCases.Menu.OfflineGame.CreateOfflineGame;
using BrassApplication.Utils;
using BrassLegacyAI;
using BrassPatch;
using BrassPatch.Persistence;
using Frameworks.Persistence.GameFactory;
using Frameworks.Persistence.Repository;
using Frameworks.Persistence.RepositoryBackends.Filesystem;
using Frameworks.Routing.Unity;
using Frameworks.View.Common;
using Frameworks.View.Game;
using Frameworks.View.Menu.CreateGame;
using InterfaceAdapters.Common;
using InterfaceAdapters.Common.CreateGame;
using InterfaceAdapters.Common.LoadScene;
using InterfaceAdapters.Game.PlayAiTurn;
using InterfaceAdapters.Game.PlayGame;
using InterfaceAdapters.Game.PlayHumanTurn;
using InterfaceAdapters.Routing;
using InterfaceAdapters.Utils;
using Newtonsoft.Json.Linq;
using TMPro;
using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.EventSystems;
using UnityEngine.UI;

public static class LocalAiUnityTests
{
    public static void Begin(string output)
    {
        var driver = new GameObject("Local AI integration tests").AddComponent<LocalAiUnityDriver>();
        UnityEngine.Object.DontDestroyOnLoad(driver.gameObject); driver.Output = output;
    }
}

public sealed partial class LocalAiUnityDriver : MonoBehaviour
{
    public string Output;
    int checks; float started; bool done, expectedError;
    readonly List<string> errors = new List<string>();
    readonly List<string> environmentMessages = new List<string>();
    readonly List<string> images = new List<string>();
    readonly List<object> turns = new List<object>();
    readonly List<object> selectionLayouts = new List<object>();
    bool selectionCasesPassed;
    int selectionFixtureChoices, draftChecks, searchAllowanceChecks;
    Router router; BrassGameHistory history; Settings settings;
    CreateOfflineGameUseCase menu; CreateOfflineGameView menuView;
    string saves;
    void Check(bool value, string reason) { checks++; if (!value) throw new Exception(reason); }
    static T Get<T>(object owner, string name) { return ReflectionAccess.Get<T>(owner, name); }
    static object Call(object owner, string method, params object[] args) { return owner.GetType().GetMethod(method, ReflectionAccess.Flags).Invoke(owner, args); }
    GameMetadata Setup { get { return Get<GameMetadata>(menu, "_newGameMetadata"); } }
    PlayerCreateGameView[] Rows(bool resume = false) { return Get<Transform>(menuView, resume ? "_existingGamePlayersContainer" : "_newGamePlayersContainer").GetComponentsInChildren<PlayerCreateGameView>(true); }
    static TextMeshProUGUI Label(PlayerCreateGameView row) { return Get<TextMeshProUGUI>(row, "_playerAIDifficultyText"); }
    void Log(string message, string stack, LogType type)
    {
        if (message.StartsWith("Error saving PlayerPrefs LastGoogleUpdate_I2Languages", StringComparison.Ordinal)) { environmentMessages.Add(message); return; }
        if (!expectedError && (type == LogType.Exception || type == LogType.Error)) errors.Add(message + " " + stack);
    }
    IEnumerator Start()
    {
        QualitySettings.vSyncCount = 0; Application.targetFrameRate = 60;
        started = Time.realtimeSinceStartup;
        Application.logMessageReceived += Log;
        var cases = Cases();
        while (true)
        {
            object current = null; bool more = false; Exception failure = null;
            try { more = cases.MoveNext(); if (more) current = cases.Current; } catch (Exception e) { failure = e; }
            if (failure != null) { Finish(failure); yield break; }
            if (!more) { Finish(null); yield break; }
            yield return current;
        }
    }
    IEnumerator Cases()
    {
        SearchAllowanceContracts();
        yield return SceneManager.LoadSceneAsync("MenuScene"); yield return null;
        var resourceChecks = AiExecutionResourceTests.Run(Check);
        while (resourceChecks.MoveNext()) yield return resourceChecks.Current;
        router = new Router(); Utils.Global.Instance.Router = router;
        router.Initialize();
        // Unity 2019's Mono cannot open the same long paths supported by modern .NET.
        saves = Path.Combine(Path.GetDirectoryName(Application.dataPath), "artifacts", "a-" + Guid.NewGuid().ToString("N").Substring(0, 8));
        var serializer = new Frameworks.Persistence.Serialization.JsonSerializer();
        var repository = new BaseRepository(new FilesystemGameStorage(saves, serializer), new FilesystemSnapshotStorage(saves, serializer), new BrassBirminghamFactory());
        history = new BrassGameHistory { Repository = repository }; settings = new Settings();
        menu = new CreateOfflineGameUseCase(new CreateOfflineGamePresenter(router, new Localization()), new Factory(repository), history, new Analytics(), settings);
        menuView = (CreateOfflineGameView)router.GetView<ICreateOfflineGameView>(); menuView.Controller = new CreateOfflineGameController(menu);
        // Main-menu routing configures the cached view before the show animation
        // enables it. Do not warm it up with ShowView or a user-triggered refresh.
        Check(!menuView.gameObject.activeInHierarchy, "cold custom-game menu starts inactive");
        menuView.Controller.OnViewCreated();
        foreach (var row in Rows().Skip(1))
            Check(row.GetComponent<AiSetupRow>() != null, "cold-menu AI controls bind before the view is enabled");
        router.ShowView<ICreateOfflineGameView>();
        yield return null;
        CheckFirstDisplay("first open"); Capture("local-ai-cold-menu.png");
        if(Environment.GetCommandLineArgs().Any(a=>a.StartsWith("--brass-release-case="))) {
            var release=(IEnumerator)GetType().GetMethod("NativeReleaseCases",ReflectionAccess.Flags).Invoke(this,null);while(release.MoveNext())yield return release.Current;yield break;
        }
        if(Environment.GetCommandLineArgs().Contains("--brass-native-matched-test")) {
            var matched=(IEnumerator)GetType().GetMethod("NativeMatchedCases",ReflectionAccess.Flags).Invoke(this,null);while(matched.MoveNext())yield return matched.Current;yield break;
        }
        if(Environment.GetCommandLineArgs().Contains("--brass-native-local-test")) {
            var native=(IEnumerator)GetType().GetMethod("NativeCases",ReflectionAccess.Flags).Invoke(this,null);while(native.MoveNext())yield return native.Current;yield break;
        }
        router.HideView<ICreateOfflineGameView>(); yield return null;
        router.ShowView<ICreateOfflineGameView>(); yield return null;
        CheckFirstDisplay("cached menu reopened without refresh");
        var rows = Rows();
        Check(Setup.PlayerMetadata.Skip(1).All(p => LocalMenu.Strategy(p) == "balanced"), "new bots default to Adaptive Advanced Book");
        Check(Setup.PlayerMetadata.Skip(1).All(p => LocalMenu.Difficulty(p) == "hard"), "new bots default to Hard " + DifficultyCatalog.Milliseconds("hard") / 1000 + "s");
        foreach (var row in rows.Skip(1))
        {
            var p = Get<PlayerMetadata>(row, "_playerMetadata");
            var controls = row.GetComponent<AiSetupRow>();
            foreach (var key in LocalMenu.Choices)
            {
                controls.StyleButton.onClick.Invoke();
                Check(StrategyPicker.IsOpen && StrategyPicker.Active.Tiles.Count() == 7, "seven direct style choices");
                StrategyPicker.Active.Tiles.Single(b => b.name == key).onClick.Invoke();
                Check(StrategyPicker.IsOpen, "card selection remains a draft");
                StrategyPicker.Active.ApplyButton.onClick.Invoke();
                Check(!StrategyPicker.IsOpen && LocalMenu.Strategy(p) == key && Label(row).text == LocalMenu.ChoiceLabel(key), "actual row directly selects " + key);
                Check(controls.StyleButton.GetComponent<StrategyTooltip>().Description == LocalMenu.ChoiceDescription(key), "strategy tooltip " + key);
                Check(LocalMenu.Difficulty(p) == "hard", "style never changes difficulty");
            }
            foreach (var difficulty in DifficultyCatalog.Keys) {
                controls.OpenPicker();
                Check(StrategyPicker.Active.DifficultyButtons[Array.IndexOf(DifficultyCatalog.Keys, difficulty)].GetComponentInChildren<TextMeshProUGUI>().text == DifficultyCatalog.Summary(difficulty), "difficulty button displays its live allowance");
                StrategyPicker.Active.DifficultyButtons[Array.IndexOf(DifficultyCatalog.Keys, difficulty)].onClick.Invoke();
                StrategyPicker.Active.ApplyButton.onClick.Invoke();
                Check(LocalMenu.Difficulty(p) == difficulty && LocalMenu.Strategy(p) == "random" && p.AiType == AIType.Hard, "difficulty independent of style and original engine enum");
            }
            controls.OpenPicker(); menu.BackClicked(); Check(!StrategyPicker.IsOpen && menuView.gameObject.activeInHierarchy, "Escape/back closes chooser without exiting menu");
            Check(row.GetComponentsInChildren<Button>().All(b => !Enumerable.Range(0, b.onClick.GetPersistentEventCount()).Any(i => b.onClick.GetPersistentMethodName(i).Contains("DifficultyClick"))), "old arrows absent from active UI");
        }
        for (int i = 1; i < 4; i++) { LocalMenu.Select(Setup.PlayerMetadata[i], ProfileCatalog.Keys[i + 2]); LocalMenu.SelectDifficulty(Setup.PlayerMetadata[i], (i == 3 ? "max" : DifficultyCatalog.Keys[i - 1])); }
        Call(menu, "PresentGameConfiguration");
        var selected = Setup.PlayerMetadata[1]; var priorColor = selected.PlayerStartConfiguration.Color;
        menu.NextColorFor(priorColor, priorColor);
        Check(selected.PlayerStartConfiguration.Color != priorColor && LocalMenu.Strategy(selected) == "goods-beer" && LocalMenu.Difficulty(selected) == "easy", "color swap retains player strategy and difficulty");
        menu.DecreaseNumberOfPlayers(); menu.DecreaseNumberOfPlayers(); menu.IncreaseNumberOfPlayers(); menu.IncreaseNumberOfPlayers();
        Check(LocalMenu.Strategy(Setup.PlayerMetadata[3]) == "pottery-flex", "remove/add preserves row strategy");
        rows[1].GetComponent<AiSetupRow>().OpenPicker(); StrategyPicker.Active.SelectPage(PlayerSelectionPage.Human); StrategyPicker.Active.ApplyButton.onClick.Invoke();
        Check(selected.AiType == AIType.Human && rows[1].GetComponent<AiSetupRow>().CanEdit, "switch to Human retains unified selector");
        rows[1].GetComponent<AiSetupRow>().OpenPicker(); StrategyPicker.Active.SelectPage(PlayerSelectionPage.V4); StrategyPicker.Active.ApplyButton.onClick.Invoke();
        Check(LocalMenu.Strategy(selected) == "goods-beer" && Label(rows[1]).text == "Worcester Iron-Cotton" && LocalMenu.Difficulty(selected) == "easy", "switch to bot restores strategy and difficulty");
        LocalMenu.Select(Setup.PlayerMetadata[3], "random"); Call(menu, "PresentGameConfiguration");
        Check(Label(rows[3]).text == "Random" && LocalMenu.CanStart(menu), "Random is selectable for a new bot");
        Capture("local-ai-custom-menu.png");
        var ui = rows[1].GetComponent<AiSetupRow>();
        ui.OpenPicker(); Capture("local-ai-style-picker.png");
        var oldPicker = StrategyPicker.Active; Call(menu, "PresentGameConfiguration");
        Check(!StrategyPicker.IsOpen && !oldPicker.gameObject.activeSelf, "refresh invalidates popup binding and blocker");
        ui.OpenPicker();
        var popup = StrategyPicker.Active; var randomButton = popup.Tiles.First(); randomButton.Select();
        ExecuteEvents.Execute(randomButton.gameObject, new AxisEventData(EventSystem.current) { moveDir = MoveDirection.Down }, ExecuteEvents.moveHandler);
        Check(EventSystem.current.currentSelectedGameObject == popup.Tiles.Single(b => b.name == "balanced").gameObject, "keyboard navigation stays in chooser");
        ExecuteEvents.ExecuteHierarchy(EventSystem.current.currentSelectedGameObject, new BaseEventData(EventSystem.current), ExecuteEvents.cancelHandler);
        Check(!StrategyPicker.IsOpen && EventSystem.current.currentSelectedGameObject == ui.StyleButton.gameObject, "keyboard cancel restores the originating control");
        ui.OpenPicker(); popup = StrategyPicker.Active; yield return null; Canvas.ForceUpdateCanvases();
        var pointer = new PointerEventData(EventSystem.current) { position = RectTransformUtility.WorldToScreenPoint(router.Camera, Get<Button>(menuView, "_startGameButton").transform.position), button = PointerEventData.InputButton.Left };
        var hits = new List<RaycastResult>(); EventSystem.current.RaycastAll(pointer, hits);
        Check(hits.Count > 0 && hits[0].gameObject.transform.IsChildOf(popup.transform), "modal intercepts pointer above the New Game button; hits=" + String.Join(",", hits.Select(h => h.gameObject.name)));
        Check(menu.StartGame(92700) == null && history.CurrentGame == null, "modal rejects underlying keyboard/programmatic Start");
        pointer.position = new Vector2(2, 2); hits.Clear(); EventSystem.current.RaycastAll(pointer, hits);
        Check(hits.Count > 0 && hits[0].gameObject.name == "Dismiss chooser", "outside click lands on input shield");
        ExecuteEvents.Execute(hits[0].gameObject, pointer, ExecuteEvents.pointerClickHandler);
        Check(!StrategyPicker.IsOpen && history.CurrentGame == null, "outside click only dismisses and cannot create a game");
        Check(menu.StartGame(92700) == null, "dismissal frame cannot click through to Start");
        ui.OpenPicker(); router.HideView<ICreateOfflineGameView>(); yield return null;
        Check(!StrategyPicker.IsOpen, "hiding menu closes popup"); router.ShowView<ICreateOfflineGameView>();
        ui.OpenPicker(); var hiding = menuView.CreateHideViewAnimation(); hiding.Kill(); yield return null;
        Check(!StrategyPicker.IsOpen && menuView.gameObject.activeInHierarchy, "production fade visibility closes popup even while objects remain enabled");
        router.ShowView<ICreateOfflineGameView>(); ui.StyleButton.Select();
        ExecuteEvents.Execute(ui.StyleButton.gameObject, new BaseEventData(EventSystem.current), ExecuteEvents.selectHandler);
        Check(GameObject.Find("AI selection help") != null, "focused control shows help before the menu fades");
        hiding = menuView.CreateHideViewAnimation(); hiding.Kill(); yield return null;
        Check(GameObject.Find("AI selection help") == null, "production fade visibility clears detached tooltip");
        router.ShowView<ICreateOfflineGameView>();
        var draftCases = PlayerSelectionCases();
        while (draftCases.MoveNext()) yield return draftCases.Current;
        var panelCases = SelectionPanelCases(rows[1].GetComponent<AiSetupRow>());
        while (panelCases.MoveNext()) yield return panelCases.Current;
        var expanded = Enumerable.Range(1, 16).Select(i => new AiStyleOption("test-" + i, "Style " + i, "Expanded catalog layout fixture " + i)).ToArray();
        string chosen = null;
        var expandedPicker = StrategyPicker.Open(ui, Label(rows[1]), "Expanded catalog preview", "test-16", expanded, key => { chosen = key; return true; });
        Check(expandedPicker.Tiles.Count() == 17 && expandedPicker.Filter != null, "16 styles plus Random and search");
        Capture("local-ai-expanded-picker.png");
        expandedPicker.Filter.text = "no matching choice";
        Check(expandedPicker.Tiles.Count() == 1 && expandedPicker.transform.Find("Style chooser/Styles viewport/No results").gameObject.activeSelf, "empty filter retains Random and shows explanation");
        expandedPicker.transform.Find("Style chooser/Clear filter").GetComponent<Button>().onClick.Invoke();
        Check(expandedPicker.Tiles.Count() == 17, "clear filter restores expanded catalog");
        expandedPicker.Filter.text = "Style 16";
        Check(expandedPicker.Tiles.Count() == 2, "filter selects the sixteenth style by name");
        expandedPicker.Tiles.Single(b => b.name == "test-16").onClick.Invoke(); expandedPicker.ApplyButton.onClick.Invoke(); Check(chosen == "test-16", "last expanded style selectable by stable ID");
        foreach (var size in new[] { new Vector2Int(1920, 1200), new Vector2Int(1920, 1080) }) {
            router.Resize(size.x, size.y); yield return null; ui.OpenPicker();
            AssertInsideTarget(StrategyPicker.Active.transform.Find("Style chooser").GetComponent<RectTransform>(), "picker " + size.x);
            Capture("local-ai-picker-" + size.x + ".png"); StrategyPicker.CloseActive();
        }
        router.Resize(1600, 900); yield return null;
        menu.SetIntroductoryGame(true); var first = menu.StartGame(92701);
        Check(first != null && first.Game.PlayersTrack.Players.Count == 4 && first.Game.GameStartConfiguration.IsIntroductoryGame, "actual StartGame publishes requested setup");
        var state = LocalGameState.Read(first.Metadata);
        foreach (var p in first.Metadata.PlayerMetadata.Where(p => p.AiType != AIType.Human)) {
            var assignment = LocalGameState.Assignment(state, (int)p.PlayerStartConfiguration.Color);
            Check((int)assignment["execution"]["search_ms"] == DifficultyCatalog.Milliseconds(LocalGameState.Difficulty(state, assignment)), "game creation uses selected thinking allowance");
        }
        Check((string)LocalGameState.Assignment(state, (int)selected.PlayerStartConfiguration.Color)["strategy"] == "goods-beer", "saved strategy follows changed color");
        Check(LocalGameState.Difficulty(state, LocalGameState.Assignment(state, (int)selected.PlayerStartConfiguration.Color)) == "easy", "selected difficulty saved with changed color");
        int randomColor = (int)Setup.PlayerMetadata[3].PlayerStartConfiguration.Color;
        string randomStrategy = (string)LocalGameState.Assignment(state, randomColor)["strategy"];
        Check(ProfileCatalog.Keys.Contains(randomStrategy) && (string)LocalGameState.Assignment(state, randomColor)["profile"]["mode"] == "masked_ismcts", "Random resolves to one of the six versioned Masked ISMCTS profiles");
        Check(LocalMenu.Strategy(Setup.PlayerMetadata[3]) == "random", "resolving Random does not change the next-game selection");
        Check((string)LocalGameState.Assignment(LocalGameState.Read(repository.Load(first.Metadata.GameId).Metadata), randomColor)["strategy"] == randomStrategy, "Random remains fixed after load");
        // Recreate a cold menu with a saved game. Both the parent menu and its
        // Continue container are inactive while the controller binds the rows.
        router.DisposeViews(); yield return null;
        menuView = (CreateOfflineGameView)router.GetView<ICreateOfflineGameView>(); menuView.Controller = new CreateOfflineGameController(menu);
        Check(!menuView.gameObject.activeInHierarchy, "cold Continue menu starts inactive");
        menuView.Controller.OnViewCreated();
        foreach (var row in Rows(true).Skip(1)) Check(row.GetComponent<AiSetupRow>() != null, "cold Continue rows bind before activation");
        router.ShowView<ICreateOfflineGameView>(); yield return null;
        CheckFirstDisplay("first open with Continue"); Capture("local-ai-cold-resume-menu.png");
        foreach (var row in Rows(true).Skip(1)) {
            var controls = row.GetComponent<AiSetupRow>();
            Check(!controls.CanEdit && !controls.StyleButton.gameObject.activeInHierarchy && controls.DifficultyButtons.All(b => !b.gameObject.activeInHierarchy), "cold Continue displays read-only controls immediately");
            Check(!Label(row).gameObject.activeInHierarchy && row.GetComponentsInChildren<TextMeshProUGUI>().Any(t => t.name == "Saved AI" && t.text.Contains("s</size>")), "cold Continue shows the saved summary without stock label overlap");
        }
        Check(LocalMenu.Strategy(Setup.PlayerMetadata[1]) == "goods-beer" && LocalMenu.Strategy(Setup.PlayerMetadata[3]) == "random", "saved defaults retain Random");
        Check(Label(Rows(true)[1]).text == "Worcester Iron-Cotton" && LocalMenu.Difficulty(Get<PlayerMetadata>(Rows(true)[1], "_playerMetadata")) == "easy", "resume shows saved style and difficulty");
        Check(Label(Rows(true)[3]).text == LocalMenu.ChoiceLabel(randomStrategy), "resume reveals the resolved Random strategy");
        Check(Rows(true).SelectMany(r => r.GetComponentsInChildren<Button>(true)).Where(b => Enumerable.Range(0, b.onClick.GetPersistentEventCount()).Any(i => b.onClick.GetPersistentMethodName(i).Contains("Difficulty"))).All(b => !b.interactable), "resume arrows disabled");
        Capture("local-ai-resume-menu.png");
        Rows()[3].GetComponent<AiSetupRow>().OpenPicker(); Capture("local-ai-resume-picker.png"); StrategyPicker.CloseActive();
        router.Resize(1920, 1200); yield return null; Capture("local-ai-resume-1920x1200.png");
        foreach (var row in Rows().Where(r => r.GetComponent<AiSetupRow>().CanEdit))
            AssertInsideTarget(row.GetComponent<AiSetupRow>().StyleButton.GetComponent<RectTransform>(), "single split menu control");
        Rows()[3].GetComponent<AiSetupRow>().OpenPicker(); AssertInsideTarget(StrategyPicker.Active.Window, "split picker"); Capture("local-ai-resume-picker-1920x1200.png"); StrategyPicker.CloseActive();
        router.Resize(1600, 900); yield return null;
        Check(Setup.PlayerMetadata.Skip(1).Select(LocalMenu.Difficulty).SequenceEqual(new[]{"easy","mid","max"}), "next-game preferences preserve each difficulty independently of Continue");
        menu.DecreaseNumberOfPlayers(); var second = menu.StartGame(92702);
        Check(second != null && repository.GetGamesMetadata().Count() == 2, "new game keeps previous save");
        Check(LocalMenu.FindResume(history).GameId == second.Metadata.GameId, "newest ready save selected");
        menu.ContinueClicked(); Check(history.CurrentGame.Metadata.GameId == second.Metadata.GameId, "continue loads new game");
        var pending = (GameMetadata)second.Metadata.Clone(); var pendingState = SaveExtensions.Get(pending); pendingState["ready"] = false; SaveExtensions.Set(pending, pendingState);
        // Creation assigns a fresh ID; clear only the extension's old ID before publishing.
        pendingState["game_id"] = null; SaveExtensions.Set(pending, pendingState); var incomplete = repository.Create(pending);
        Check(LocalMenu.FindResume(history).GameId == second.Metadata.GameId, "incomplete creation ignored");
        var metadataPath = Path.Combine(saves, second.Metadata.GameId, "metadata.storage"); var valid = File.ReadAllBytes(metadataPath);
        File.WriteAllText(metadataPath, "{broken"); var beforeFiles = Directory.GetFiles(saves, "*", SearchOption.AllDirectories).Length;
        expectedError = true; menu.UpdateResumeOfflineGameView(); expectedError = false;
        Check(Directory.GetFiles(saves, "*", SearchOption.AllDirectories).Length == beforeFiles && File.ReadAllText(metadataPath) == "{broken", "failed resume preserves every file");
        Check(!LocalMenu.CanStart(menu), "unreadable save disables replacement"); File.WriteAllBytes(metadataPath, valid); router.HideView<IQuestionPopupView>();
        menu.UpdateResumeOfflineGameView(); Check(LocalMenu.CanStart(menu), "recovered save reenables new game");
        // An unextended legacy save still loads through the original bot path and has an honest label.
        var previousPlayer = second.Metadata.PlayerMetadata.First(p => p.AiType != AIType.Human);
        var oldPlayer = new PlayerMetadata { PlayerStartConfiguration = previousPlayer.PlayerStartConfiguration, AiType = AIType.Medium };
        var oldModel = new PlayerCreateGameViewModel { PlayerMetadata = oldPlayer, AiType = AIType.Medium, AiTypeText = "Medium", PlayerNumber = 1, CurrentPlayerCount = 3, Nickname = "Original", PlayerColor = oldPlayer.PlayerStartConfiguration.Color };
        Rows(true)[0].SetPlayer(oldModel); Check(Label(Rows(true)[0]).text == "Original AI", "legacy label preserved without changing saved AI type");
        if (Environment.GetCommandLineArgs().Contains("--brass-local-ai-smoke"))
        { Check(errors.Count == 0, "installed startup/menu has no unexpected errors"); yield break; }
        router.DisposeViews(); yield return null;
        yield return SceneManager.LoadSceneAsync("GameScene"); yield return null;
        router.ReloadPrefabs();
        var aiView = (PlayAiTurnView)router.GetView<IPlayAiTurnView>();
        File.WriteAllText(Output + ".input-layout.json", JArray.FromObject(aiView.GetComponentsInChildren<Transform>(true).Select(t => new {
            path = t.name, components = t.GetComponents<Component>().Where(c => c != null).Select(c => c.GetType().FullName).ToArray(),
            buttons = t.GetComponents<Button>().Select(b => Enumerable.Range(0, b.onClick.GetPersistentEventCount()).Select(i => b.onClick.GetPersistentTarget(i) + ":" + b.onClick.GetPersistentMethodName(i)).ToArray()).ToArray(),
            triggers = t.GetComponents<EventTrigger>().SelectMany(e => e.triggers.Select(entry => new { entry.eventID, calls = Enumerable.Range(0, entry.callback.GetPersistentEventCount()).Select(i => entry.callback.GetPersistentTarget(i) + ":" + entry.callback.GetPersistentMethodName(i)).ToArray() })).ToArray()
        })).ToString());
        var output = new OutputAdapter(history); var play = new PlayGameUseCase(output, history, null, new Analytics());
        var turn = new PlayAiTurnUseCase(new PlayAITurnPresenter(router), history);
        router.UseCases[typeof(PlayGameUseCase)] = play; router.UseCases[typeof(PlayAiTurnUseCase)] = turn;
        aiView.Controller = new PlayAiTurnController(turn);
        foreach (int count in new[] { 2, 3, 4 }) foreach (bool intro in new[] { false, true }) foreach (string key in ProfileCatalog.Keys) foreach (string difficulty in DifficultyCatalog.Keys)
        {
            if(Environment.GetCommandLineArgs().Contains("--brass-masked-quick"))
            {
                bool quickCase=(count==2 && !intro && key=="balanced" && (difficulty=="easy" || difficulty=="mid")) ||
                    (count==4 && !intro && key=="balanced" && (difficulty=="hard" || difficulty=="max")) ||
                    (count==3 && intro && (key=="rail-tempo" || key=="pottery-flex") && difficulty=="easy") ||
                    (count==4 && !intro && (key=="cotton" || key=="goods-beer" || key=="supplier") && difficulty=="easy");
                if(!quickCase)continue;
            }
            bool humanTransition = count == 2 && key == "balanced";
            var config = new GameStartConfiguration((NumberOfPlayers)count) { Seed = 92720 + count, IsIntroductoryGame = intro };
            var meta = new GameMetadata(config, GameType.Local);
            foreach (var start in config.Players) meta.JoinGame(new PlayerMetadata { PlayerStartConfiguration = start, AiType = AIType.Hard, NickName = "Strategy test", PlayerNetworkId = "native-" + start.Color });
            var saved = LocalGameState.Create(meta, p => key, p => difficulty); SaveExtensions.Set(meta, saved);
            var app = repository.Create(meta); app.Game.StartGame(); saved["game_id"] = app.Metadata.GameId; saved["ready"] = true; SaveExtensions.Set(app.Metadata, saved);
            if (key != "balanced" || humanTransition)
            {
                while (app.Game.GameProgressInformation.CurrentRoundNumber == 1)
                {
                    var actor = app.Game.GameProgressInformation.CurrentPlayer;
                    app.Game.TurnFlow.StartPassAction().Pass(actor.Hand[0], actor); app.Game.TurnFlow.ConfirmCurrentAction();
                    app.Game.GameFlow.EndTurn(); if (app.Game.GameFlow.IsLastTurnOver()) app.Game.GameFlow.EndRound();
                }
            }
            if (humanTransition)
            {
                // Use a real mixed game, including the cached board that the old
                // AI scheduling path used to leave permanently disabled.
                var next = app.Game.PlayersTrack.Players.First(p => p.Color != app.Game.GameProgressInformation.CurrentPlayer.Color);
                app.Metadata.GetPlayerMetadataByColor(next.Color).AiType = AIType.Human;
                saved = LocalGameState.Create(app.Metadata, p => key, p => difficulty); saved["ready"] = true; SaveExtensions.Set(app.Metadata, saved);
            }
            repository.Save(app); history.SetGame(app);
            router.ShowView<IPlayAiTurnView>(); aiView.SetCurrentPlayerName("Strategy test");
            int revision = app.Game.Revision, savesBefore = output.Saves, expectedActions = app.Game.TurnFlow.GetRemainingActionsCount();
            var boardOutput = new InputProbeBoard(router, new Localization());
            var boardInput = new BoardUseCase(boardOutput, history, settings);
            router.UseCases[typeof(BoardUseCase)] = boardInput;
            router.GetView<IBoardView>();
            boardInput.RegionSelected(RegionName.Birmingham);
            Check(boardOutput.Builds == 0 && app.Game.Revision == revision, "board click cannot start a human build during an AI turn");
            var actionsInput = new PlayerActionsUseCase(null, history);
            var handInput = new HandOfCardsUseCase(null, history);
            var replayInput = new ReplayUseCase(null, history);
            var inputs = new Action[] { () => boardInput.RegionSelected(RegionName.Birmingham), actionsInput.StartBuildAction,
                actionsInput.StartNetworkAction, actionsInput.StartDevelopAction, actionsInput.StartSellAction, actionsInput.StartLoanAction,
                actionsInput.StartScoutAction, actionsInput.StartPassAction, actionsInput.UndoAction, actionsInput.EndTurn,
                () => handInput.OnCardSelected(app.Game.GameProgressInformation.CurrentPlayer.Hand[0]), replayInput.SkipReplay, LocalTurnBridge.PlayNow };
            foreach (var input in inputs) input();
            Check(app.Game.Revision == revision && boardOutput.Builds == 0, "all stale human/replay inputs ignored before dispatch");
            // Cancel a scheduled turn before it starts, then verify no delayed dispatch survives.
            var schedule = new PlayGamePresenter(router); schedule.StartAITurn(app.Metadata.PlayerMetadata[0]); LocalTurnBridge.CancelActive();
            yield return null; Check(!LocalTurnBridge.IsThinking && app.Game.Revision == revision, "cancel pending dispatch");
            TaskCompletionSource<int> priorDevice = null;
            bool initialQueueDelayTested = count == 2 && !intro && key == "balanced" && difficulty == "easy";
            if (initialQueueDelayTested)
            {
                priorDevice = new TaskCompletionSource<int>();
                BrassLegacyAI.Execution.AiExecutionScheduler.Submit(new BrassLegacyAI.Execution.DelegateAiExecutionBackend("preceding model device tail", token => priorDevice.Task));
            }
            turn.PlayAITurn(); turn.PlayAITurn(); Check(LocalTurnBridge.IsThinking, "new strategy bridge starts");
            Check(AiExecutionResources.IsComputing && Application.targetFrameRate == 30 && QualitySettings.vSyncCount == 0, "AI turn enters general computation resource policy");
            var startTime = Time.realtimeSinceStartup; bool probed = false, replanInjected = false; double injectedElapsed = 0;
            var observedBridge = UnityEngine.Object.FindObjectOfType<LocalTurnBridge>();
            if (initialQueueDelayTested)
            {
                yield return null; yield return null;
                Check(!Get<System.Diagnostics.Stopwatch>(observedBridge, "clock").IsRunning && Get<int>(observedBridge, "planningRequests") == 0,
                    "initial allowance starts after preceding model retires and board preparation completes");
                priorDevice.SetResult(0);
            }
            while (LocalTurnBridge.IsThinking)
            {
                var controls = aiView.GetComponentInChildren<AiTurnControls>();
                var button = controls == null ? null : controls.GetComponentInChildren<Button>();
                Check(button == null, "Play now is absent throughout thinking");
                if (!probed && controls != null)
                {
                    var bridge = UnityEngine.Object.FindObjectOfType<LocalTurnBridge>();
                    var executing = Get<Profile>(bridge, "profile");
                    var bankProfile = ProfileCatalog.Create(key, count, intro);
                    Check(Get<double>(bridge, "budget") == DifficultyCatalog.Milliseconds(difficulty) && executing.work_limit == 1000000 && executing.work_limit == bankProfile.work_limit && executing.mode == "masked_ismcts", "per-player original allowance with Masked ISMCTS safety guard");
                    for (int repeat = 0; repeat < 3; repeat++) foreach (var input in inputs) input();
                    Check(LocalTurnBridge.IsThinking && app.Game.Revision == revision && boardOutput.Builds == 0, "repeated clicks never cancel or skip a thinking turn");
                    // Visual visibility is not ownership of the turn. A popup must
                    // not silently cancel work, unlike opening a different game.
                    router.HideView<IPlayAiTurnView>(); yield return null;
                    Check(LocalTurnBridge.IsThinking, "hiding a panel does not abandon a pending turn");
                    router.ShowView<IPlayAiTurnView>(); probed = true;
                }
                if (count == 2 && !intro && key == "balanced" && difficulty == "mid" && !replanInjected && app.Game.Revision > revision && app.Game.TurnFlow.GetRemainingActionsCount() == 1) {
                    var continuation = Get<Queue<string[]>>(observedBridge, "continuation");
                    if (continuation.Count > 0) { injectedElapsed = Get<System.Diagnostics.Stopwatch>(observedBridge, "clock").Elapsed.TotalMilliseconds; continuation.Clear(); continuation.Enqueue(new[] { "unavailable-choice" }); replanInjected = true; }
                }
                if (count == 2 && !intro && key == "balanced" && difficulty == "mid" && !images.Any(p => p.EndsWith("local-ai-thinking.png")) && Time.realtimeSinceStartup - startTime > .1f) Capture("local-ai-thinking.png");
                if(Time.realtimeSinceStartup-startTime>=35)
                {
                    var bridge=UnityEngine.Object.FindObjectOfType<LocalTurnBridge>();
                    var worker=bridge==null?null:Get<TurnCoordinator>(bridge,"coordinator");
                    Check(false,"native turn watchdog "+count+"/"+intro+"/"+key+
                        "; worker="+(worker==null?"missing":worker.Worker.Status+", ready="+worker.Ready+", finished="+worker.Finished+", failure="+worker.Failure)+
                        "; revision="+app.Game.Revision+"; actions="+app.Game.TurnFlow.GetRemainingActionsCount());
                }
                yield return null;
            }
            Check(output.Saves == savesBefore + 1 && app.Game.Revision > revision, "exactly one end turn/save " + key);
            Check(!AiExecutionResources.IsComputing && AiExecutionResources.ScopeCount == 0 && Application.targetFrameRate == 60 && QualitySettings.vSyncCount == 0, "AI completion restores foreground rendering synchronously");
            var committed = repository.Advanced.GameStorage.LoadEvents(app.Metadata.GameId, revision, app.Game.Revision);
            Check(committed.Count(e => e.Metadata.Revision > revision && e.GetType().Name == "ActionCountDecreased") == expectedActions, "one confirmation per planned action");
            var reloaded = repository.Load(app.Metadata.GameId);
            Check(reloaded.Game.Revision == app.Game.Revision && LocalGameState.Read(reloaded.Metadata)["records"].Any(), "native action and intent reload together");
            Check(!router.IsViewVisible<IQuestionPopupView>(), "native strategy turn has no error popup");
            int requests = Get<int>(observedBridge, "planningRequests"); double remainingBudget = Get<double>(observedBridge, "lastRequestMilliseconds");
            Check(remainingBudget <= DifficultyCatalog.Milliseconds(difficulty), "no request receives a larger allowance than its actor");
            if (count == 2 && !intro && key == "balanced" && difficulty == "mid") Check(replanInjected && requests == 2 && remainingBudget <= Math.Max(0, DifficultyCatalog.Milliseconds(difficulty) - injectedElapsed) + 1, "stale second action replans only within the original turn deadline");
            turns.Add(new { count, intro, key, difficulty, inputGuardTested = probed, humanTransitionTested = humanTransition, replanInjected, initialQueueDelayTested, planningRequests = requests, lastRequestMilliseconds = remainingBudget, thinkingMilliseconds = DifficultyCatalog.Milliseconds(difficulty), workLimit = LocalTurnBridge.SearchWorkLimit, seconds = Time.realtimeSinceStartup - startTime });
            File.WriteAllText(Output + ".progress.json", JObject.FromObject(new { completed = turns.Count, total = Environment.GetCommandLineArgs().Contains("--brass-masked-quick") ? 9 : 144, count, intro, key, difficulty }).ToString());
            if (humanTransition)
            {
                var human = app.Metadata.GetPlayerMetadataByColor(app.Game.GameProgressInformation.CurrentPlayer.Color);
                Check(human.AiType == AIType.Human && app.Game.TurnFlow.GetRemainingActionsCount() == 2, "AI advances to a real human turn");
                new PlayHumanTurnPresenter(router).RestorePlayerActionsView();
                boardInput.RegionSelected(RegionName.Birmingham);
                Check(boardOutput.Builds == 1, "city click starts a human build after the AI turn without resetting board permission");
                human.LastSeenEventRevision = app.Game.Revision;
                var player = app.Game.GameProgressInformation.CurrentPlayer;
                app.Game.TurnFlow.StartPassAction().Pass(player.Hand[0], player); app.Game.TurnFlow.ConfirmCurrentAction();
                int historicalProbes = 0;
                var replay = new ReplayUseCase(new ImmediateReplayOutput(() => {
                    Check(Get<bool>(history, "_movedBackInHistory"), "replay uses a historical game");
                    int oldRevision = history.CurrentGame.Game.Revision;
                    boardInput.RegionSelected(RegionName.Birmingham);
                    actionsInput.StartPassAction(); actionsInput.UndoAction(); actionsInput.EndTurn();
                    Check(boardOutput.Builds == 1 && history.CurrentGame.Game.Revision == oldRevision,
                        "human replay cannot accept board or action input");
                    historicalProbes++;
                }), history);
                replay.PresentEvents(app.Game.GetUncommittedEvents(), human, null);
                Check(historicalProbes > 0 && !replay.GetIsReplayInProgress() && ReferenceEquals(history.CurrentGame, app), "human action replay returns to the live game");
                boardInput.RegionSelected(RegionName.Birmingham);
                Check(boardOutput.Builds == 2 && app.Game.TurnFlow.GetRemainingActionsCount() == 1, "city click works again after human replay");
                // The remaining fixtures exercise cancellation on an AI actor.
                continue;
            }
            // Cancellation after search starts must also prevent late mutation.
            history.SetGame(reloaded); router.ShowView<IPlayAiTurnView>(); revision = reloaded.Game.Revision;
            turn.PlayAITurn(false); history.SetGame(reloaded); yield return null; yield return null;
            Check(reloaded.Game.Revision == revision && !LocalTurnBridge.IsThinking, "cancel running worker prevents late mutation");
            Check(AiExecutionResources.ScopeCount == 0 && Application.targetFrameRate == 60, "game replacement restores computing resources despite worker retirement");
        }
        float retirementDeadline = Time.realtimeSinceStartup + 10;
        while (!BrassLegacyAI.Execution.AiExecutionScheduler.IsIdle && Time.realtimeSinceStartup < retirementDeadline) yield return null;
        Check(BrassLegacyAI.Execution.AiExecutionScheduler.IsIdle && BrassLegacyAI.Execution.AiExecutionScheduler.Statistics().maximumRunning == 1, "all strategy jobs drain with one executing job at a time");
        Check(errors.Count == 0, "no unexpected native errors: " + String.Join("\n", errors));
    }
    void Update() { if (!done && Time.realtimeSinceStartup - started > 1800) Finish(new Exception("native integration watchdog")); }
    void Finish(Exception error)
    {
        var profile=Environment.GetEnvironmentVariable("BRASS_L14_TRACE");
        if(!String.IsNullOrEmpty(profile))typeof(LocalTurnBridge).Assembly.GetType("BrassLegacyAI.Native.NativeStageTrace").GetMethod("Save").Invoke(null,new object[]{profile});
        done = true; LocalTurnBridge.CancelActive(); Application.logMessageReceived -= Log;
        File.WriteAllText(Output, new JObject { ["passed"] = error == null, ["mode"] = Environment.GetCommandLineArgs().Contains("--brass-local-ai-smoke") ? "startup-menu-smoke" : "full", ["checks"] = checks, ["searchAllowanceChecks"] = searchAllowanceChecks, ["error"] = error == null ? null : error.ToString(), ["nativeErrors"] = JArray.FromObject(errors), ["environmentMessages"] = JArray.FromObject(environmentMessages), ["images"] = JArray.FromObject(images), ["turns"] = JArray.FromObject(turns), ["resourceTests"] = JObject.FromObject(new { passed = AiExecutionResourceTests.Passed, checks = AiExecutionResourceTests.Checks }), ["resourceMeasurements"] = AiExecutionResourceTests.Measurements == null ? null : JObject.FromObject(AiExecutionResourceTests.Measurements), ["executionStatistics"] = JObject.FromObject(BrassLegacyAI.Execution.AiExecutionScheduler.Statistics()), ["selectionPanel"] = JObject.FromObject(new { passed = selectionCasesPassed, gammaEnabled = AiSelectionCatalog.Entries(AiFamily.Gamma).Any(e => e.Available), fixtureChoices = selectionFixtureChoices, draftChecks, layouts = selectionLayouts }), ["unity"] = Application.unityVersion }.ToString());
        Application.Quit(error == null ? 0 : 1);
    }
    void SearchAllowanceContracts()
    {
        int first=checks;
        Check(TurnCoordinator.SearchWorkers==2, "V4 retains two search workers");
        Check(LocalTurnBridge.SearchWorkLimit==1000000, "V4 work guard unchanged");
        var expected=new[]{4000,8000,10000,12000};var shape=new[]{2000,5000,12000,12000};
        Check(DifficultyCatalog.Keys.SequenceEqual(new[]{"easy","mid","hard","max"}), "four difficulty IDs in time order");
        Check(DifficultyCatalog.Default=="hard" && DifficultyCatalog.Milliseconds(DifficultyCatalog.Default)==10000, "default remains Hard at 10s");
        for(int tier=0;tier<4;tier++){
            string key=DifficultyCatalog.Keys[tier];
            Check(DifficultyCatalog.Milliseconds(key)==expected[tier], "V4 turn allowance "+key);
            Check(DifficultyCatalog.Summary(key).EndsWith(" · "+expected[tier]/1000+"s"), "summary uses current allowance");
            var saved=DifficultyCatalog.Snapshot(key);
            Check((string)saved["version"]=="local-ai-execution/1.2.0" && DifficultyCatalog.Read(saved)==key, "new execution format includes all four tiers");
            foreach(string version in new[]{"local-ai-execution/1.0.0","local-ai-execution/1.1.0","local-ai-execution/1.2.0","future/9"}){
                var broken=(JObject)saved.DeepClone();broken["version"]=version;broken["search_ms"]=17;
                bool rejected=false;try{DifficultyCatalog.Read(broken);}catch(InvalidDataException){rejected=true;}
                Check(rejected,"invalid execution time rejected for "+version);
            }
            foreach(int count in new[]{2,3,4})foreach(bool intro in new[]{false,true})foreach(string style in ProfileCatalog.Keys){
                var frozen=ProfileCatalog.Snapshot(style,count,intro);var frozenHash=Wire.Hash(frozen);
                var baseline=ProfileCatalog.ForPlay(frozen);ProfileCatalog.ConfigureClock(baseline,shape[tier]);
                var updated=ProfileCatalog.ForPlay(frozen);ProfileCatalog.ConfigureClock(updated,expected[tier],key);
                Check(JToken.DeepEquals(JObject.FromObject(baseline),JObject.FromObject(updated)), "existing search shape preserved; Max shares Hard shape");
                Check(Wire.Hash(frozen)==frozenHash,"frozen strategy and intent identity unchanged");
                var config=new GameStartConfiguration((NumberOfPlayers)count){IsIntroductoryGame=intro};
                var meta=new GameMetadata(config,GameType.Local){GameId="clock-compatibility"};
                foreach(var start in config.Players)meta.JoinGame(new PlayerMetadata{PlayerStartConfiguration=start,AiType=AIType.Hard});
                var state=LocalGameState.Create(meta,p=>style,p=>key);state["ready"]=true;
                var encoded=System.Text.Encoding.UTF8.GetBytes(new JObject{[SaveExtensions.Key]=state}.ToString());SaveExtensions.Read(meta,encoded);var current=LocalGameState.Read(meta);
                Check(Wire.Hash(current)==Wire.Hash(state) && current["assignments"].All(a=>LocalGameState.Difficulty(current,(JObject)a)==key && (int)a["execution"]["search_ms"]==expected[tier]), "all four execution choices survive serialization and read");
                if(tier<3)foreach(string version in new[]{"local-ai-execution/1.0.0","local-ai-execution/1.1.0"}){
                    var old=(JObject)state.DeepClone();int previous=version.EndsWith("1.0.0")?new[]{2000,5000,12000}[tier]:new[]{6000,10000,15000}[tier];
                    foreach(JObject row in old["assignments"])row["execution"]=new JObject{["version"]=version,["difficulty"]=key,["search_ms"]=previous};
                    old["assignment_sha256"]=Wire.Hash(new JObject{["format"]=old["format"],["runtime"]=old["runtime"],["rules"]=old["rules"],["schema"]=old["schema"],["configuration"]=old["configuration"],["assignments"]=old["assignments"]});
                    var before=Wire.Hash(old);SaveExtensions.Read(meta,System.Text.Encoding.UTF8.GetBytes(new JObject{[SaveExtensions.Key]=old}.ToString()));var read=LocalGameState.Read(meta);
                    Check(Wire.Hash(read)==before && read["assignments"].All(a=>LocalGameState.Difficulty(read,(JObject)a)==key), "both prior signed execution versions read without mutation");
                }
            }
        }
        foreach(string version in new[]{"local-ai-execution/1.0.0","local-ai-execution/1.1.0"}){
            bool rejected=false;try{DifficultyCatalog.Read(new JObject{["version"]=version,["difficulty"]="max",["search_ms"]=12000});}catch(InvalidDataException){rejected=true;}
            Check(rejected,"prior versions cannot claim the new Max tier");
        }
        var legacy=ProfileCatalog.Create("balanced",4);ProfileCatalog.ConfigureClock(legacy,10000);
        var compatibility=ProfileCatalog.Create("balanced",4);ProfileCatalog.ConfigureClock(compatibility,10000,DifficultyCatalog.Existing);
        Check(DifficultyCatalog.Milliseconds(DifficultyCatalog.Existing)==10000 && JToken.DeepEquals(JObject.FromObject(legacy),JObject.FromObject(compatibility)), "unnamed legacy saves retain 10s and existing search shape");
        searchAllowanceChecks=checks-first;
    }
    void Capture(string file)
    {
        Canvas.ForceUpdateCanvases(); router.Camera.Render();
        var old = RenderTexture.active; RenderTexture.active = router.Target;
        var texture = new Texture2D(router.Target.width, router.Target.height, TextureFormat.RGB24, false);
        texture.ReadPixels(new Rect(0, 0, texture.width, texture.height), 0, 0); texture.Apply(); RenderTexture.active = old;
        string path = Path.Combine(Path.GetDirectoryName(Output), file); File.WriteAllBytes(path, texture.EncodeToPNG()); images.Add(path); UnityEngine.Object.Destroy(texture);
    }
    IEnumerator PlayerSelectionCases()
    {
        int initialChecks = checks;
        var first = Rows()[0].GetComponent<AiSetupRow>(); var bot = Rows()[1].GetComponent<AiSetupRow>();
        var identity = LocalMenu.SetupIdentity(bot.Player); var original = bot.SelectedChoice; string difficulty = bot.SelectedDifficulty;
        var firstChoice = first.SelectedChoice; string firstDifficulty = first.SelectedDifficulty;
        router.Resize(1920, 1200); yield return null;
        first.OpenPicker(); var panel = AiSelectionPanel.Active;
        Check(panel.CurrentPage == PlayerSelectionPage.Human && panel.Filter == null && !panel.Tiles.Any() && panel.DifficultyButtons.Length == 0, "Human page has no AI controls");
        Check(panel.Window.Find("Page Human").GetSiblingIndex() < panel.Window.Find("Page V4").GetSiblingIndex(), "Human is first in navigation");
        panel.SelectPage(PlayerSelectionPage.V4); panel.Tiles.Last().onClick.Invoke(); panel.DifficultyButtons[0].onClick.Invoke();
        Check(first.Player.AiType == AIType.Human && first.SelectedChoice.Equals(firstChoice) && first.SelectedDifficulty == firstDifficulty, "Human AI edits stay in draft");
        panel.Window.Find("Close panel").GetComponent<Button>().onClick.Invoke();
        Check(first.Player.AiType == AIType.Human && first.SelectedChoice.Equals(firstChoice), "Cancel discards role/style/difficulty");
        first.OpenPicker(); panel = AiSelectionPanel.Active; panel.SelectPage(PlayerSelectionPage.V4); panel.ApplyButton.onClick.Invoke();
        yield return null;
        Check(!LocalMenu.CanStart(menu) && LocalMenu.NeedsHuman(first.Player), "all-AI lineup requires Human");
        Check(Get<Button>(menuView, "_startGameButton").transform.Find("Human required").gameObject.activeInHierarchy, "all-AI explanation visible");
        Capture("player-selection-human-required-1200.png");
        first.OpenPicker(); panel = AiSelectionPanel.Active; panel.SelectPage(PlayerSelectionPage.Human); panel.ApplyButton.onClick.Invoke();
        yield return null;
        Check(LocalMenu.CanStart(menu) && !LocalMenu.NeedsHuman(first.Player), "restoring Human re-enables game creation");
        bot.OpenPicker(); panel = AiSelectionPanel.Active;
        string alternate = original.EntryId == "cotton" ? "supplier" : "cotton";
        panel.Tiles.Single(b => b.name == alternate).onClick.Invoke(); panel.DifficultyButtons[2].onClick.Invoke();
        panel.SelectPage(PlayerSelectionPage.Human); panel.SelectPage(PlayerSelectionPage.V4);
        Check(panel.Tiles.Single(b => b.name == alternate).GetComponent<AiControlSurface>().Selected && panel.DraftDifficulty == "hard", "draft survives Human roundtrip");
        panel.SelectPage(PlayerSelectionPage.Gamma);
        Check(panel.DifficultyButtons.All(b => !b.interactable) && !panel.ApplyButton.interactable, "Gamma difficulty and Apply disabled");
        Check(panel.Tiles.Concat(panel.DifficultyButtons).All(b => !b.GetComponent<AiControlSurface>().Selected), "unavailable Gamma choices do not claim to be selected");
        foreach (var b in panel.DifficultyButtons) b.onClick.Invoke(); panel.ApplyButton.onClick.Invoke();
        Check(AiSelectionPanel.IsOpen && panel.DraftDifficulty == "hard" && bot.SelectedChoice.Equals(original), "forced Gamma callbacks cannot change or apply draft");
        panel.SelectPage(PlayerSelectionPage.Human); panel.ApplyButton.onClick.Invoke();
        Check(bot.Player.AiType == AIType.Human && bot.SelectedChoice.Equals(original) && bot.SelectedDifficulty == difficulty, "Human Apply preserves committed AI settings, discards unapplied AI edits");
        bot.OpenPicker(); panel = AiSelectionPanel.Active; panel.SelectPage(PlayerSelectionPage.V4); panel.ApplyButton.onClick.Invoke();
        Check(bot.Player.AiType == AIType.Hard && bot.SelectedChoice.Equals(original), "switch back restores committed AI");
        bot.OpenPicker(); panel=AiSelectionPanel.Active;panel.DifficultyButtons[3].onClick.Invoke();
        panel.SelectPage(PlayerSelectionPage.Human);panel.SelectPage(PlayerSelectionPage.V4);
        Check(panel.DraftDifficulty=="max" && panel.DifficultyButtons[3].GetComponent<AiControlSurface>().Selected,"Max draft survives page rebuilds");
        Capture("player-selection-max-1200.png");panel.ApplyButton.onClick.Invoke();
        Check(bot.SelectedDifficulty=="max" && Get<TextMeshProUGUI>(bot,"choiceDetail").text=="Max · up to 12s", "Max Apply immediately updates row");
        Capture("player-selection-max-row-1200.png");
        LocalMenu.SelectDifficulty(bot.Player,difficulty);Call(menu,"PresentGameConfiguration");

        bot.OpenPicker(); panel = AiSelectionPanel.Active; panel.DifficultyButtons[1].onClick.Invoke();
        Check(bot.SelectedDifficulty == difficulty, "difficulty-only draft does not commit early");
        panel.ApplyButton.onClick.Invoke(); Check(bot.SelectedDifficulty == "mid" && bot.SelectedChoice.Equals(original), "difficulty-only Apply commits without reselecting style");
        bot.OpenPicker(); panel = AiSelectionPanel.Active; panel.DifficultyButtons[2].onClick.Invoke();
        panel.OnCancel(new BaseEventData(EventSystem.current)); Check(bot.SelectedDifficulty == "mid", "keyboard cancel discards difficulty edit");
        bot.OpenPicker(); panel = AiSelectionPanel.Active; panel.DifficultyButtons[2].onClick.Invoke();
        var staleApply = panel.ApplyButton; Call(menu, "PresentGameConfiguration"); staleApply.onClick.Invoke();
        Check(!AiSelectionPanel.IsOpen && bot.SelectedDifficulty == "mid", "stale Apply after rebind cannot mutate row");
        foreach (var bad in new[] {
            new PlayerSelectionDraft(PlayerSelectionPage.Gamma, new AiChoice(AiFamily.Gamma, "gamma-01"), "hard"),
            new PlayerSelectionDraft(PlayerSelectionPage.V4, original, "unknown"),
            new PlayerSelectionDraft(PlayerSelectionPage.V4, new AiChoice(AiFamily.V4, "missing"), "easy") })
            Check(!LocalMenu.ApplyPlayerSelection(bot.Player, identity, bad) && bot.SelectedDifficulty == "mid" && bot.SelectedChoice.Equals(original), "invalid complete draft rejected atomically");
        Check(!LocalMenu.ApplyPlayerSelection(bot.Player, new object(), new PlayerSelectionDraft(PlayerSelectionPage.Human, original, "hard")), "wrong setup identity rejected");
        var last = Rows()[3].GetComponent<AiSetupRow>(); var removed = last.Player; last.OpenPicker(); staleApply = AiSelectionPanel.Active.ApplyButton;
        menu.DecreaseNumberOfPlayers(); staleApply.onClick.Invoke();
        yield return null; // The removed row is no longer presented; the panel's lifecycle check closes it this frame.
        Check(!last.CanEdit && !AiSelectionPanel.IsOpen && !LocalMenu.CanConfigure(removed), "removed fourth slot loses edit authority: editable=" + last.CanEdit + ", open=" + AiSelectionPanel.IsOpen + ", old reference=" + LocalMenu.CanConfigure(removed) + ", count=" + Get<int>(menu, "_numberOfPlayers"));
        menu.DecreaseNumberOfPlayers(); Check(Rows().Count(r => r.GetComponent<AiSetupRow>().CanEdit) == 2, "two-player setup has two unified controls");
        menu.IncreaseNumberOfPlayers(); Check(Rows().Count(r => r.GetComponent<AiSetupRow>().CanEdit) == 3, "three-player setup has three unified controls");
        menu.IncreaseNumberOfPlayers(); Check(Rows().Count(r => r.GetComponent<AiSetupRow>().CanEdit) == 4, "four-player setup has four unified controls");
        foreach (var row in Rows().Skip(1)) { row.GetComponent<AiSetupRow>().OpenPicker(); panel = AiSelectionPanel.Active; panel.SelectPage(PlayerSelectionPage.Human); panel.ApplyButton.onClick.Invoke(); }
        yield return null; Check(LocalMenu.CanStart(menu) && Setup.PlayerMetadata.All(p => p.AiType == AIType.Human), "all-Human hotseat remains playable");
        Capture("player-selection-hotseat-1200.png");
        foreach (var row in Rows().Skip(1)) { row.GetComponent<AiSetupRow>().OpenPicker(); panel = AiSelectionPanel.Active; panel.SelectPage(PlayerSelectionPage.V4); panel.ApplyButton.onClick.Invoke(); }
        LocalMenu.SelectDifficulty(bot.Player, difficulty); Call(menu, "PresentGameConfiguration");
        string originalName = first.Player.NickName; first.Player.NickName = "<b>Long literal player name & family</b>"; Call(menu, "PresentGameConfiguration");
        first.OpenPicker(); panel = AiSelectionPanel.Active;
        Check(!panel.Window.Find("Title").GetComponent<TextMeshProUGUI>().richText && panel.Window.Find("Title").GetComponent<TextMeshProUGUI>().text.Contains(first.Player.NickName), "nickname renders as literal text");
        Capture("player-selection-literal-name-1200.png"); panel.Close(); first.Player.NickName = originalName; Call(menu, "PresentGameConfiguration");
        draftChecks = checks - initialChecks;
        yield return null;
    }
    IEnumerator SelectionPanelCases(AiSetupRow ui)
    {
        string before = ui.SelectedStyle; string difficulty = ui.SelectedDifficulty;
        Check(!new AiChoice(AiFamily.V4, "random").Equals(new AiChoice(AiFamily.Gamma, "random")), "family is part of choice identity");
        Check(AiSelectionCatalog.Entries(AiFamily.V4).Count == 6 && AiSelectionCatalog.Entries(AiFamily.Gamma).Count == 16, "exact family rosters");
        Check(!AiSelectionCatalog.CanChoose(new AiChoice(AiFamily.Gamma, "random")) && !AiSelectionCatalog.CanChoose(new AiChoice(AiFamily.V4, "unknown")), "empty Gamma Random and unknown identity rejected");
        Check(!LocalMenu.EditChoice(ui.Player, new AiChoice(AiFamily.Gamma, "gamma-01")) && !LocalMenu.EditChoice(ui.Player, new AiChoice(AiFamily.Gamma, "random")), "model and Random unavailable at menu boundary");
        ui.OpenPicker(); var panel = AiSelectionPanel.Active;
        Check(panel.CurrentPage == PlayerSelectionPage.V4 && Math.Abs(panel.TextScale - 1.15f) < .01f, "assigned V4 page and second text preset default");
        panel.SelectPage(AiFamily.Gamma);
        Check(panel.Tiles.Count() == 17 && panel.Tiles.All(b => !b.interactable), "Gamma has 16 unavailable models plus unavailable Random");
        Check(ui.CanEdit && ui.StyleButton.IsInteractable() == false, "modal keeps owner identity editable but underlying controls blocked");
        foreach (var button in panel.Tiles.ToArray()) {
            button.onClick.Invoke();
            Check(AiSelectionPanel.IsOpen && ui.SelectedStyle == before && ui.SelectedDifficulty == difficulty, "forced disabled callback cannot select " + button.name);
        }
        Check(panel.Detail.Contains("Awaiting model"), "unavailable card explains its reason");
        panel.Filter.text = "Gamma 16";
        Check(panel.Tiles.Count() == 2 && panel.Tiles.Last().name == "gamma-16", "Gamma search reaches model 16 by stable ID");
        panel.SelectPage(AiFamily.V4); panel.Filter.text = "Pottery";
        panel.SelectPage(AiFamily.Gamma); Check(panel.Filter.text == "Gamma 16", "Gamma query retained across pages");
        panel.SelectPage(AiFamily.V4); Check(panel.Filter.text == "Pottery", "V4 query retained independently");
        Check(ui.SelectedStyle == before && ui.SelectedDifficulty == difficulty, "page changes and search are read-only");
        panel.Close(); Check(ui.StyleButton.IsInteractable(), "modal close restores menu interaction");
        yield return null;
        foreach (var size in new[] { new Vector2Int(1920, 1200), new Vector2Int(1920, 1080) }) {
            router.Resize(size.x, size.y); yield return null;
            foreach (int preset in new[] { 0, 1, 2 }) {
                ui.OpenPicker(); panel = AiSelectionPanel.Active;
                if (preset == 0) panel.ChangeTextSize(-1); if (preset == 2) panel.ChangeTextSize(1);
                Check(panel.Filter.text == "", "fresh panel clears searches");
                foreach (var family in new[] { PlayerSelectionPage.Human, PlayerSelectionPage.V4, PlayerSelectionPage.Gamma }) {
                    panel.SelectPage(family); Canvas.ForceUpdateCanvases();
                    AssertInsideTarget(panel.Window, "selection window " + size + "/" + preset);
                    foreach (var name in new[] { "Close chooser", "Close panel", "Apply selection", "Page Human", "Page V4", "Page Gamma" })
                        AssertInsideTarget(panel.Window.Find(name).GetComponent<RectTransform>(), "fixed control " + name);
                    Check(panel.Tiles.Count() == (family == PlayerSelectionPage.Human ? 0 : family == PlayerSelectionPage.V4 ? 7 : 17), "page count stable at resolution/text preset");
                    Check(panel.Tiles.All(b => b.interactable == (family == PlayerSelectionPage.V4)), "availability stable after rebuild");
                    if (family != PlayerSelectionPage.Human) {
                    foreach (var name in new[] { "Find AI", "Clear filter", "random", "Difficulty easy", "Difficulty mid", "Difficulty hard", "Difficulty max" })
                        AssertInsideTarget(panel.Window.Find(name).GetComponent<RectTransform>(), "fixed AI control " + name);
                    Check(panel.DifficultyButtons.Length==4, "four search-time buttons on each AI page");
                    for(int tier=0;tier<4;tier++){
                        var button=panel.DifficultyButtons[tier];var text=button.GetComponentInChildren<TextMeshProUGUI>();
                        Check(text.text==DifficultyCatalog.Summary(DifficultyCatalog.Keys[tier]), "all four time labels survive resize/text changes");
                        Check(text.GetPreferredValues(text.text).x<=text.rectTransform.rect.width+1, "search-time label fits its button");
                        if(tier>0){var previousBounds=RectTransformUtility.CalculateRelativeRectTransformBounds(panel.Window,panel.DifficultyButtons[tier-1].transform);var nextBounds=RectTransformUtility.CalculateRelativeRectTransformBounds(panel.Window,button.transform);Check(previousBounds.max.x<nextBounds.min.x,"time buttons do not overlap");}
                    }
                    if (size.y == 1200 && preset == 1 && family == PlayerSelectionPage.Gamma)
                        foreach (var tile in panel.Tiles.Skip(1)) Check(InsideViewport(tile.GetComponent<RectTransform>(), panel.GridScroll.viewport), "all Gamma cards visible at 1200 default: " + tile.name);
                    var closePosition = panel.Window.Find("Close panel").position;
                    panel.GridScroll.verticalNormalizedPosition = 0; Canvas.ForceUpdateCanvases();
                    Check(InsideViewport(panel.Tiles.Last().GetComponent<RectTransform>(), panel.GridScroll.viewport), "last card reachable with scroll");
                    Check(panel.Window.Find("Close panel").position == closePosition, "scroll does not move close control");
                    panel.GridScroll.verticalNormalizedPosition = 1;
                    }
                    Capture("player-selection-" + size.x + "x" + size.y + "-" + family + "-" + preset + ".png");
                    selectionLayouts.Add(new { width = size.x, height = size.y, preset, family = family.ToString(), choices = panel.Tiles.Count(), available = panel.Tiles.Count(b => b.interactable), difficultyButtons = panel.DifficultyButtons.Length, difficultyLabels = panel.DifficultyButtons.Select(b=>b.GetComponentInChildren<TextMeshProUGUI>().text).ToArray() });
                }
                panel.Close(); yield return null;
            }
        }
        Check(ui.SelectedStyle == before && ui.SelectedDifficulty == difficulty && LocalMenu.Choice(ui.Player).Family == AiFamily.V4, "all visual checks preserve actual V4 assignment");
        // Future-ready UI fixtures exercise typed identities and filtered order;
        // they never install models or change a game assignment.
        var fixture = Enumerable.Range(1, 16).Select(i => new AiCatalogEntry(AiFamily.Gamma, "model-" + i,
            i < 3 ? "Duplicate model name" : "Neural model " + i, "Synthetic UI fixture " + i, true)).ToArray();
        foreach (var entry in fixture) {
            AiChoice chosen = default(AiChoice);
            panel = AiSelectionPanel.Open(ui, Label(Rows()[1]), "Synthetic fixture", LocalMenu.Choice(ui.Player), c => { chosen = c; return true; }, AiFamily.Gamma, fixture);
            panel.SelectPage(AiFamily.Gamma); panel.Tiles.Single(b => b.name == entry.Choice.EntryId).onClick.Invoke();
            panel.ApplyButton.onClick.Invoke();
            Check(chosen.Equals(entry.Choice) && !AiSelectionPanel.IsOpen, "typed fixture selects " + entry.Choice);
            selectionFixtureChoices++;
        }
        selectionCasesPassed = true;
        yield return null;
    }
    bool InsideViewport(RectTransform item, RectTransform viewport)
    {
        var corners = new Vector3[4]; item.GetWorldCorners(corners);
        return corners.All(p => { var v = viewport.InverseTransformPoint(p); var r = viewport.rect; return v.x >= r.xMin - .5f && v.x <= r.xMax + .5f && v.y >= r.yMin - .5f && v.y <= r.yMax + .5f; });
    }
    void CheckFirstDisplay(string phase)
    {
        foreach (var row in Rows()) {
            var controls = row.GetComponent<AiSetupRow>();
            Check(controls != null && row.GetComponents<AiSetupRow>().Length == 1 && controls.CanEdit, phase + ": one bound editable component");
            Check(controls.StyleButton.gameObject.activeInHierarchy && controls.DifficultyButtons.Length == 0 && controls.StyleButton.transform.parent.GetComponentsInChildren<Button>().Length == 1, phase + ": one selection button and no external difficulty controls");
            Check(controls.StyleButton.GetComponentInChildren<TextMeshProUGUI>().text == (controls.Player.AiType == AIType.Human ? "Human" : LocalMenu.RowLabel(controls.Player)), phase + ": role/family summary visible immediately");
            Check(controls.StyleButton.GetComponentInChildren<TextMeshProUGUI>().color == BrassPatch.Statistics.StatisticsTheme.Ink &&
                controls.DifficultyButtons.All(b => b.GetComponentInChildren<TextMeshProUGUI>().color == BrassPatch.Statistics.StatisticsTheme.Ink), phase + ": dark ink on light controls during cold binding");
            Check(!Label(row).gameObject.activeInHierarchy && !Get<TextMeshProUGUI>(row, "_playerAITypeText").gameObject.activeInHierarchy, phase + ": stock labels hidden");
            Check(row.GetComponentsInChildren<Button>().All(b => !Enumerable.Range(0, b.onClick.GetPersistentEventCount()).Any(i => b.onClick.GetPersistentMethodName(i).Contains("DifficultyClick"))), phase + ": stock arrows hidden");
            AssertInsideTarget(controls.StyleButton.GetComponent<RectTransform>(), phase + ": style bounds");
            foreach (var button in controls.DifficultyButtons) AssertInsideTarget(button.GetComponent<RectTransform>(), phase + ": difficulty bounds");
        }
    }
    void AssertInsideTarget(RectTransform rect, string name)
    {
        Canvas.ForceUpdateCanvases(); var corners = new Vector3[4]; rect.GetWorldCorners(corners);
        Check(corners.All(p => { var v = router.Camera.WorldToViewportPoint(p); return v.x >= -.002f && v.x <= 1.002f && v.y >= -.002f && v.y <= 1.002f; }), "visible bounds: " + name);
    }
    sealed class Localization : ILocalization
    { public string GetTranslation(string term) { var name = term.Split('/').Last(); return name == "AI_TYPE_HUMAN" ? "Human" : System.Globalization.CultureInfo.InvariantCulture.TextInfo.ToTitleCase(name.Replace('_', ' ').ToLowerInvariant()); } }
    sealed class Settings : IGameSettings
    {
        GameMetadata saved; int count;
        public bool CameFromMenu() { return false; } public void SetMap(bool n) { } public bool GetIsNightMap() { return false; }
        public void SetDefaultGameSettings(GameMetadata m, int n) { saved = m; count = n; }
        public int GetNumberOfPlayers() { return count; } public bool IsSavedDefaultGameSettings() { return saved != null; }
        public PlayerColor GetPlayerIndexSavedColor(int i) { return saved.PlayerMetadata[i].PlayerStartConfiguration.Color; }
        public PlayerAvatar GetPlayerIndexSavedAvatar(int i) { return saved.PlayerMetadata[i].PlayerAvatar; }
        public AIType GetPlayerIndexSavedAIType(int i) { return saved.PlayerMetadata[i].AiType; }
    }
    sealed class Factory : IRepositoryFactory
    {
        IRepository repository; public Factory(IRepository r) { repository = r; }
        public IRepository CreateLocalGameRepository() { return repository; } public IRepository CreateTutorialGameRepository() { return repository; }
        public IRepository CreateTestGamesRepository() { return repository; } public IRepository CreateOnlineGameRepository() { return repository; } public IRepository CreateInMemoryGameRepository() { return repository; }
    }
    sealed class Analytics : IAnalytics
    {
        public void ScreenVisit(string s) { } public void GameStart(GameMetadata m) { } public void GameOver(GameMetadata m) { }
        public void LevelStart(string s) { } public void LevelComplete(string s) { } public void LevelFail(string s) { } public void LevelQuit(string s) { }
        public void TutorialStart(string s) { } public void TutorialStep(string s, int i) { } public void TutorialComplete(string s) { } public void TutorialSkip(string s) { }
        public void StoreOpened() { } public void StoreItemClick(string s) { } public void StoreItemPurchased(string s) { } public void StoreItemsRestored() { } public void LogCustomEvent(string s) { }
    }
    sealed class OutputAdapter : IPlayGameOutput
    {
        BrassGameHistory history; public int Saves; public OutputAdapter(BrassGameHistory h) { history = h; }
        public void SaveGame(BrassApplicationGame game, ISaveGameDelegate callback) { history.Repository.Save(game); Saves++; }
        public void StartHumanTurn() { } public void StartAITurn(PlayerMetadata p) { } public void StartSellingIndustries() { throw new Exception("unexpected opening debt"); }
        public void StartGameReview(PlayerMetadata p) { } public void StartNetworkHumanTurn() { } public void StartNetworkSpectatorTurn(PlayerMetadata p) { }
    }
    sealed class InputProbeBoard : BoardPresenter, IBoardOutput
    {
        public int Builds, Details;
        public InputProbeBoard(IRouter r, ILocalization l) : base(r, l) { }
        void IBoardOutput.StartBuildActionFromBoard(RegionName region) { Builds++; }
        void IBoardOutput.ShowRegionDetailsView(RegionName region, BrassApplicationGame game,
            IDictionary<Tuple<BoardGameRules.Entities.Board.BuildingSlots.BuildingSlot, BoardGameRules.Entities.Board.BuildingSlots.BuildingType>, IList<BoardGameRules.Entities.Actions.CantUseBuildActionReason>> reasons)
        { Details++; }
    }
    sealed class ImmediateReplayOutput : IReplayOutput
    {
        readonly Action probe;
        public ImmediateReplayOutput(Action probe) { this.probe = probe; }
        public void ReplayEvent(BoardGameRules.Entities.EventSourcing.Events.IEvent e, IEventReplayDelegate callback)
        { probe(); callback.EventReplayCompleted(e); }
        public void UpdateCurrentPlayer(string name) { }
        public void UpdateCurrentPlayerAction(BoardGameRules.Entities.EventSourcing.Events.IEvent e) { }
        public void UpdateCurrentPlayerActionText(string text) { }
        public void UpdateGamePresentation(PlayerMetadata player) { }
        public void CancelCurrentReplay() { }
        public void ShowReplaySummary(IList<BoardGameRules.Entities.EventSourcing.Events.IEvent> events, Action dismissed) { dismissed(); }
        public void ShowIndustrySellingSummary(BoardGameRules.Entities.EventSourcing.Events.IEvent e, Action dismissed) { dismissed(); }
        public void UpdateCurrentPlayerActionsView() { }
        public void FinishReplay() { }
    }
    sealed class Animation : IAnimation
    {
        public IAnimation Play() { return this; } public IAnimation Kill() { return this; } public IAnimation OnComplete(Action a) { return this; }
        public IAnimation OnKill(Action a) { return this; } public IAnimation AppendCallback(Action a) { a(); return this; } public bool IsPlaying() { return false; }
    }
    sealed class Router : IRouter, IUseCaseBuilder
    {
        public readonly Dictionary<Type, IUseCase> UseCases = new Dictionary<Type, IUseCase>();
        readonly Dictionary<Type, BaseView> views = new Dictionary<Type, BaseView>(); readonly HashSet<Type> visible = new HashSet<Type>();
        BaseView[] prefabs; GameObject root; public Camera Camera; public RenderTexture Target;
        public IUseCaseBuilder UseCaseBuilder { get; set; }
        public Router() { UseCaseBuilder = this; }
        public void Initialize()
        {
            if (EventSystem.current == null) { var events = new GameObject("Native test events", typeof(EventSystem), typeof(StandaloneInputModule)); UnityEngine.Object.DontDestroyOnLoad(events); }
            root = new GameObject("Native integration canvas", typeof(RectTransform), typeof(Canvas), typeof(GraphicRaycaster), typeof(CanvasScaler)); UnityEngine.Object.DontDestroyOnLoad(root);
            var node = new GameObject("Native test camera", typeof(Camera)); UnityEngine.Object.DontDestroyOnLoad(node); Camera = node.GetComponent<Camera>(); Camera.enabled = false; Camera.orthographic = true;
            Camera.transform.position = new Vector3(0, 0, -100); Camera.farClipPlane = 1000; Camera.clearFlags = CameraClearFlags.SolidColor; Camera.backgroundColor = new Color(.08f, .07f, .06f);
            Target = new RenderTexture(1600, 900, 24, RenderTextureFormat.ARGB32); Target.Create(); Camera.targetTexture = Target;
            var canvas = root.GetComponent<Canvas>(); canvas.renderMode = RenderMode.ScreenSpaceCamera; canvas.worldCamera = Camera; canvas.planeDistance = 100;
            var scaler = root.GetComponent<CanvasScaler>(); scaler.uiScaleMode = CanvasScaler.ScaleMode.ScaleWithScreenSize; scaler.referenceResolution = new Vector2(1920, 1080);
            ReloadPrefabs();
        }
        public void ReloadPrefabs() { prefabs = Resources.FindObjectsOfTypeAll<UnityViewFactory>().SelectMany(f => f.ViewList.Views).OfType<BaseView>().ToArray(); }
        public void Resize(int width, int height)
        {
            Camera.targetTexture = null; Target.Release(); UnityEngine.Object.Destroy(Target);
            Target = new RenderTexture(width, height, 24, RenderTextureFormat.ARGB32); Target.Create(); Camera.targetTexture = Target; Canvas.ForceUpdateCanvases();
        }
        public void DisposeViews() { foreach (var view in views.Values) UnityEngine.Object.Destroy(view.gameObject); views.Clear(); visible.Clear(); }
        public T GetView<T>() where T : class, IBaseView
        {
            BaseView view; if (!views.TryGetValue(typeof(T), out view)) { var prefab = prefabs.First(v => v is T); view = UnityEngine.Object.Instantiate(prefab, root.transform); view.InputRegistration = this; views[typeof(T)] = view; view.gameObject.SetActive(false); }
            return view as T;
        }
        public bool IsViewCached<T>() where T : class, IBaseView { return views.ContainsKey(typeof(T)); }
        public bool IsViewVisible<T>() where T : class, IBaseView { return visible.Contains(typeof(T)); }
        public IAnimation ShowView<T>() where T : class, IBaseView
        {
            if (typeof(T) == typeof(ILoadSceneView)) return new Animation();
            var view = (BaseView)(object)GetView<T>(); visible.Add(typeof(T)); view.CreateShowViewImmediatelyAnimation();
            // Complete this visual transition synchronously without invoking any use-case route.
            var popup = view.transform.Find("SafeArea/Popup"); if (popup != null) { var group = popup.GetComponent<CanvasGroup>(); if (group != null) { group.alpha = 1; group.interactable = true; group.blocksRaycasts = true; } popup.GetComponent<RectTransform>().anchoredPosition = Vector2.zero; popup.localScale = Vector3.one; }
            return new Animation();
        }
        public IAnimation HideView<T>() where T : class, IBaseView { visible.Remove(typeof(T)); BaseView view; if (views.TryGetValue(typeof(T), out view)) view.gameObject.SetActive(false); return new Animation(); }
        public void MoveOnTop<T>() where T : class, IBaseView { ((BaseView)(object)GetView<T>()).transform.SetAsLastSibling(); }
        public void ProcessInput() { } public void RegisterKey(KeyEnum k, IBaseView v) { } public void UnregisterKey(KeyEnum k, IBaseView v) { } public void UnregisterAllKeys(IBaseView v) { }
        public void BuildUseCases() { } public void RestoreUseCases() { } public void RestoreUseCaseOfType(Type t) { } public IController GetController(Type t) { return null; }
        public T GetUseCase<T>() where T : IUseCase { return (T)UseCases[typeof(T)]; }
    }
}


