using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
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

public sealed class LocalAiUnityDriver : MonoBehaviour
{
    public string Output;
    int checks; float started; bool done, expectedError;
    readonly List<string> errors = new List<string>();
    readonly List<string> environmentMessages = new List<string>();
    readonly List<string> images = new List<string>();
    readonly List<object> turns = new List<object>();
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
        yield return SceneManager.LoadSceneAsync("MenuScene"); yield return null;
        router = new Router(); Utils.Global.Instance.Router = router;
        router.Initialize();
        // Unity 2019's Mono cannot open the same long paths supported by modern .NET.
        saves = Path.Combine(Path.GetDirectoryName(Application.dataPath), "artifacts", "a-" + Guid.NewGuid().ToString("N").Substring(0, 8));
        var serializer = new Frameworks.Persistence.Serialization.JsonSerializer();
        var repository = new BaseRepository(new FilesystemGameStorage(saves, serializer), new FilesystemSnapshotStorage(saves, serializer), new BrassBirminghamFactory());
        history = new BrassGameHistory { Repository = repository }; settings = new Settings();
        menu = new CreateOfflineGameUseCase(new CreateOfflineGamePresenter(router, new Localization()), new Factory(repository), history, new Analytics(), settings);
        menuView = (CreateOfflineGameView)router.GetView<ICreateOfflineGameView>(); menuView.Controller = new CreateOfflineGameController(menu);
        router.ShowView<ICreateOfflineGameView>(); menu.CreateNewGameConfiguration(); menu.UpdateResumeOfflineGameView();
        yield return null;
        var rows = Rows();
        Check(Setup.PlayerMetadata.Skip(1).All(p => LocalMenu.Strategy(p) == "balanced"), "new bots default to Balanced");
        foreach (var row in rows.Skip(1))
        {
            var p = Get<PlayerMetadata>(row, "_playerMetadata");
            foreach (var key in LocalMenu.Choices)
            {
                Check(LocalMenu.Strategy(p) == key && Label(row).text == LocalMenu.ChoiceLabel(key), "actual row forward cycles " + key);
                Check(Label(row).GetComponent<StrategyTooltip>().Description == LocalMenu.ChoiceDescription(key), "strategy tooltip " + key);
                row.OnNextDifficultyClick();
            }
            row.OnPreviousDifficultyClick(); Check(Label(row).text == "Random", "previous wraps to Random"); row.OnNextDifficultyClick();
        }
        for (int i = 1; i < 4; i++) LocalMenu.Select(Setup.PlayerMetadata[i], ProfileCatalog.Keys[i + 2]);
        Call(menu, "PresentGameConfiguration");
        var selected = Setup.PlayerMetadata[1]; var priorColor = selected.PlayerStartConfiguration.Color;
        menu.NextColorFor(priorColor, priorColor);
        Check(selected.PlayerStartConfiguration.Color != priorColor && LocalMenu.Strategy(selected) == "goods-beer", "color swap retains player strategy");
        menu.DecreaseNumberOfPlayers(); menu.DecreaseNumberOfPlayers(); menu.IncreaseNumberOfPlayers(); menu.IncreaseNumberOfPlayers();
        Check(LocalMenu.Strategy(Setup.PlayerMetadata[3]) == "pottery-flex", "remove/add preserves row strategy");
        Call(menu, "NextAiType", selected); Check(selected.AiType == AIType.Human && !Label(rows[1]).gameObject.activeSelf, "switch to human hides selector");
        Call(menu, "NextAiType", selected); Check(LocalMenu.Strategy(selected) == "goods-beer" && Label(rows[1]).text == "Goods/Beer v3", "switch to bot restores v3 strategy");
        LocalMenu.Select(Setup.PlayerMetadata[3], "random"); Call(menu, "PresentGameConfiguration");
        Check(Label(rows[3]).text == "Random" && LocalMenu.CanStart(menu), "Random is selectable for a new bot");
        Capture("local-ai-custom-menu.png");
        menu.SetIntroductoryGame(true); var first = menu.StartGame(92701);
        Check(first != null && first.Game.PlayersTrack.Players.Count == 4 && first.Game.GameStartConfiguration.IsIntroductoryGame, "actual StartGame publishes requested setup");
        var state = LocalGameState.Read(first.Metadata);
        Check((string)LocalGameState.Assignment(state, (int)selected.PlayerStartConfiguration.Color)["strategy"] == "goods-beer", "saved strategy follows changed color");
        int randomColor = (int)Setup.PlayerMetadata[3].PlayerStartConfiguration.Color;
        string randomStrategy = (string)LocalGameState.Assignment(state, randomColor)["strategy"];
        Check(ProfileCatalog.Keys.Contains(randomStrategy) && (string)LocalGameState.Assignment(state, randomColor)["profile"]["mode"] == "S1", "Random resolves to one of the six real S1 profiles");
        Check(LocalMenu.Strategy(Setup.PlayerMetadata[3]) == "random", "resolving Random does not change the next-game selection");
        Check((string)LocalGameState.Assignment(LocalGameState.Read(repository.Load(first.Metadata.GameId).Metadata), randomColor)["strategy"] == randomStrategy, "Random remains fixed after load");
        if (Environment.GetCommandLineArgs().Contains("--brass-local-ai-smoke"))
        { Check(errors.Count == 0, "installed startup/menu has no unexpected errors"); yield break; }
        menu.CreateNewGameConfiguration(); Check(LocalMenu.Strategy(Setup.PlayerMetadata[1]) == "goods-beer" && LocalMenu.Strategy(Setup.PlayerMetadata[3]) == "random", "saved defaults retain Random");
        menu.UpdateResumeOfflineGameView();
        Check(Label(Rows(true)[1]).text == "Goods/Beer v3", "resume shows saved strategy with current v3 deployment");
        Check(Label(Rows(true)[3]).text == ProfileCatalog.Label(randomStrategy), "resume reveals the resolved Random strategy");
        Check(Rows(true).SelectMany(r => r.GetComponentsInChildren<Button>(true)).Where(b => Enumerable.Range(0, b.onClick.GetPersistentEventCount()).Any(i => b.onClick.GetPersistentMethodName(i).Contains("Difficulty"))).All(b => !b.interactable), "resume arrows disabled");
        Capture("local-ai-resume-menu.png");
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
        foreach (int count in new[] { 2, 3, 4 }) foreach (bool intro in new[] { false, true }) foreach (string key in ProfileCatalog.Keys)
        {
            bool humanTransition = count == 2 && key == "balanced";
            var config = new GameStartConfiguration((NumberOfPlayers)count) { Seed = 92720 + count, IsIntroductoryGame = intro };
            var meta = new GameMetadata(config, GameType.Local);
            foreach (var start in config.Players) meta.JoinGame(new PlayerMetadata { PlayerStartConfiguration = start, AiType = AIType.Hard, NickName = "Strategy test", PlayerNetworkId = "native-" + start.Color });
            var saved = LocalGameState.Create(meta, p => key); SaveExtensions.Set(meta, saved);
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
                saved = LocalGameState.Create(app.Metadata, p => key); saved["ready"] = true; SaveExtensions.Set(app.Metadata, saved);
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
            turn.PlayAITurn(); turn.PlayAITurn(); Check(LocalTurnBridge.IsThinking, "new strategy bridge starts");
            var startTime = Time.realtimeSinceStartup; bool probed = false;
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
                    Check(Get<double>(bridge, "budget") == 10000 && executing.work_limit == 240 && executing.work_limit == bankProfile.work_limit && executing.reply_work == bankProfile.reply_work && executing.mode == "S1", "10-second allowance with research-bank node and reply limits");
                    for (int repeat = 0; repeat < 3; repeat++) foreach (var input in inputs) input();
                    Check(LocalTurnBridge.IsThinking && app.Game.Revision == revision && boardOutput.Builds == 0, "repeated clicks never cancel or skip a thinking turn");
                    // Visual visibility is not ownership of the turn. A popup must
                    // not silently cancel work, unlike opening a different game.
                    router.HideView<IPlayAiTurnView>(); yield return null;
                    Check(LocalTurnBridge.IsThinking, "hiding a panel does not abandon a pending turn");
                    router.ShowView<IPlayAiTurnView>(); probed = true;
                }
                if (count == 2 && !intro && key == "balanced" && images.Count == 2 && Time.realtimeSinceStartup - startTime > .1f) Capture("local-ai-thinking.png");
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
            var committed = repository.Advanced.GameStorage.LoadEvents(app.Metadata.GameId, revision, app.Game.Revision);
            Check(committed.Count(e => e.Metadata.Revision > revision && e.GetType().Name == "ActionCountDecreased") == expectedActions, "one confirmation per planned action");
            var reloaded = repository.Load(app.Metadata.GameId);
            Check(reloaded.Game.Revision == app.Game.Revision && LocalGameState.Read(reloaded.Metadata)["records"].Any(), "native action and intent reload together");
            Check(!router.IsViewVisible<IQuestionPopupView>(), "native strategy turn has no error popup");
            turns.Add(new { count, intro, key, inputGuardTested = probed, humanTransitionTested = humanTransition, thinkingMilliseconds = LocalTurnBridge.ThinkingMilliseconds, workLimit = LocalTurnBridge.SearchWorkLimit, seconds = Time.realtimeSinceStartup - startTime });
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
        }
        Check(errors.Count == 0, "no unexpected native errors: " + String.Join("\n", errors));
    }
    void Update() { if (!done && Time.realtimeSinceStartup - started > 660) Finish(new Exception("native integration watchdog")); }
    void Finish(Exception error)
    {
        done = true; LocalTurnBridge.CancelActive(); Application.logMessageReceived -= Log;
        File.WriteAllText(Output, new JObject { ["passed"] = error == null, ["mode"] = Environment.GetCommandLineArgs().Contains("--brass-local-ai-smoke") ? "startup-menu-smoke" : "full", ["checks"] = checks, ["error"] = error == null ? null : error.ToString(), ["nativeErrors"] = JArray.FromObject(errors), ["environmentMessages"] = JArray.FromObject(environmentMessages), ["images"] = JArray.FromObject(images), ["turns"] = JArray.FromObject(turns), ["unity"] = Application.unityVersion }.ToString());
        Application.Quit(error == null ? 0 : 1);
    }
    void Capture(string file)
    {
        Canvas.ForceUpdateCanvases(); router.Camera.Render();
        var old = RenderTexture.active; RenderTexture.active = router.Target;
        var texture = new Texture2D(router.Target.width, router.Target.height, TextureFormat.RGB24, false);
        texture.ReadPixels(new Rect(0, 0, texture.width, texture.height), 0, 0); texture.Apply(); RenderTexture.active = old;
        string path = Path.Combine(Path.GetDirectoryName(Output), file); File.WriteAllBytes(path, texture.EncodeToPNG()); images.Add(path); UnityEngine.Object.Destroy(texture);
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
            root = new GameObject("Native integration canvas", typeof(RectTransform), typeof(Canvas), typeof(GraphicRaycaster), typeof(CanvasScaler)); UnityEngine.Object.DontDestroyOnLoad(root);
            var node = new GameObject("Native test camera", typeof(Camera)); UnityEngine.Object.DontDestroyOnLoad(node); Camera = node.GetComponent<Camera>(); Camera.enabled = false; Camera.orthographic = true;
            Camera.transform.position = new Vector3(0, 0, -100); Camera.farClipPlane = 1000; Camera.clearFlags = CameraClearFlags.SolidColor; Camera.backgroundColor = new Color(.08f, .07f, .06f);
            Target = new RenderTexture(1600, 900, 24, RenderTextureFormat.ARGB32); Target.Create(); Camera.targetTexture = Target;
            var canvas = root.GetComponent<Canvas>(); canvas.renderMode = RenderMode.ScreenSpaceCamera; canvas.worldCamera = Camera; canvas.planeDistance = 100;
            var scaler = root.GetComponent<CanvasScaler>(); scaler.uiScaleMode = CanvasScaler.ScaleMode.ScaleWithScreenSize; scaler.referenceResolution = new Vector2(1920, 1080);
            ReloadPrefabs();
        }
        public void ReloadPrefabs() { prefabs = Resources.FindObjectsOfTypeAll<UnityViewFactory>().SelectMany(f => f.ViewList.Views).OfType<BaseView>().ToArray(); }
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
