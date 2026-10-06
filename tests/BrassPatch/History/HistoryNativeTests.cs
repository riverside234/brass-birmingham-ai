using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using BoardGameRules.Entities.Players;
using BrassApplication.Game;
using BrassApplication.GameHistory;
using BrassApplication.UseCases.Common;
using BrassApplication.UseCases.Game.FinishedGameReview;
using BrassApplication.UseCases.Game.Replay;
using BrassApplication.UseCases.Menu.MainMenu;
using BrassPatch;
using BrassPatch.History;
using BrassPatch.Statistics;
using DG.Tweening;
using Frameworks.Routing;
using Frameworks.Routing.Unity;
using Frameworks.View.Menu;
using InterfaceAdapters.Common;
using InterfaceAdapters.Game.FinishedGameReview;
using InterfaceAdapters.Menu.MainMenu;
using InterfaceAdapters.Routing;
using InterfaceAdapters.Utils;
using Newtonsoft.Json;
using TMPro;
using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.SceneManagement;
using UnityEngine.UI;

public static class LocalAiUnityTests
{
    public static void Begin(string output) { var driver = new GameObject("Game History qualification").AddComponent<HistoryNativeDriver>(); UnityEngine.Object.DontDestroyOnLoad(driver.gameObject); driver.Output = output; }
}
public sealed class HistoryNativeDriver : MonoBehaviour
{
    public string Output;
    int checks, matches;
    readonly List<object> cases = new List<object>(), layouts = new List<object>();
    readonly List<string> errors = new List<string>(), images = new List<string>();
    readonly List<HistoryFixture> fixtures = new List<HistoryFixture>();
    Router router; NativeInput input; MainMenuUseCase menu; MainMenuView main;
    GameHistoryStore store; Camera camera; Canvas canvas;
    void Check(bool pass, string reason) { checks++; cases.Add(new { passed = pass, name = reason }); if (!pass) throw new Exception(reason); }
    void Log(string text, string stack, LogType type) { if (text.StartsWith("Error saving PlayerPrefs LastGoogleUpdate_I2Languages")) return; if (type == LogType.Error || type == LogType.Exception) errors.Add(text + stack); }
    IEnumerator Start()
    {
        Application.logMessageReceived += Log; var run = Run(); Exception failure = null;
        while (true) { bool more = false; object current = null; try { more = run.MoveNext(); if (more) current = run.Current; } catch (Exception error) { failure = error; }
            if (failure != null || !more) break; yield return current; }
        HistoryRecording.Flush();
        File.WriteAllText(Output, JsonConvert.SerializeObject(new { passed = failure == null && errors.Count == 0, checks, matches, pages = 10, cases, layouts, images, nativeErrors = errors, error = failure == null ? null : failure.ToString(), historyLimit = 10, reportSchema = 1 }, Formatting.Indented));
        Application.Quit(failure == null && errors.Count == 0 ? 0 : 1);
    }
    IEnumerator Run()
    {
        yield return SceneManager.LoadSceneAsync("MenuScene"); yield return null;
        string root = Path.Combine(Path.GetDirectoryName(Output), "h-" + Guid.NewGuid().ToString("N").Substring(0, 8));
        foreach (int count in new[] { 2, 3, 4 }) foreach (bool intro in new[] { false, true }) {
            var fixture = HistoryFixture.Finish(count, intro, 20261031 + count * 2 + (intro ? 1 : 0)); fixtures.Add(fixture); matches++;
            HistoryCoreChecks.Roundtrip(fixture, Check); yield return null;
        }
        foreach (int count in new[] { 2, 3, 4 }) { var fixture = HistoryFixture.Finish(count, false, 20261050 + count, true); matches++;
            HistoryCoreChecks.Roundtrip(fixture, Check); yield return null; }
        HistoryCoreChecks.Store(Path.Combine(root, "core"), fixtures.Last(), Check);
        store = new GameHistoryStore(Path.Combine(root, "ui")); HistoryRecording.UseTestStore(store);
        var prefabFactory=Resources.FindObjectsOfTypeAll<UnityViewFactory>().First(f=>f.ViewList!=null&&f.ViewList.Views.OfType<MainMenuView>().Any());
        File.WriteAllText(Path.Combine(Path.GetDirectoryName(Output),"history-fonts.json"),JsonConvert.SerializeObject(new{fonts=Resources.FindObjectsOfTypeAll<TMP_FontAsset>().Select(f=>new{f.name,material=f.material==null?null:f.material.name}),
            prefabs=prefabFactory.ViewList.Views.OfType<Frameworks.View.Common.BaseView>().Select(v=>new{type=v.GetType().FullName,fonts=v.GetComponentsInChildren<TextMeshProUGUI>(true).Select(t=>t.font==null?null:t.font.name).Distinct()})},Formatting.Indented));
        var uiRoot=new GameObject("History native UI",typeof(RectTransform),typeof(Canvas),typeof(CanvasScaler),typeof(GraphicRaycaster));
        canvas=uiRoot.GetComponent<Canvas>();uiRoot.GetComponent<CanvasScaler>().uiScaleMode=CanvasScaler.ScaleMode.ScaleWithScreenSize;uiRoot.GetComponent<CanvasScaler>().referenceResolution=new Vector2(1920,1080);
        var nativeFactory=uiRoot.AddComponent<UnityViewFactory>();nativeFactory.ParentTransform=uiRoot.transform;nativeFactory.ViewList=prefabFactory.ViewList;
        input = new NativeInput(); router = new Router(nativeFactory,input); Utils.Global.Instance.Router = router;
        HistoryCoreChecks.Set(Utils.Global.Instance,"VersionReader",new VersionStub());
        menu = new MainMenuUseCase(new MainMenuPresenter(router, new Localization()), null, null); router.UseCaseBuilder = new Builder { Menu = menu };
        main = (MainMenuView)router.GetView<IMainMenuView>();
        File.WriteAllText(Path.Combine(Path.GetDirectoryName(Output), "history-menu-bindings.json"), JsonConvert.SerializeObject(main.GetComponentsInChildren<Button>(true).Select(b => new { b.name, methods = Enumerable.Range(0, b.onClick.GetPersistentEventCount()).Select(b.onClick.GetPersistentMethodName).ToArray(), text = b.GetComponentsInChildren<TextMeshProUGUI>(true).Select(t => t.text).ToArray() }), Formatting.Indented));
        var entry = main.GetComponentsInChildren<Button>(true).Single(b => Enumerable.Range(0, b.onClick.GetPersistentEventCount()).Any(i => b.onClick.GetPersistentMethodName(i) == "OnOnlineGamesClicked"));
        Check(entry.GetComponentInChildren<TextMeshProUGUI>(true).text == "Game History", "H-MENU cold caption changed before first show");
        Check(entry.GetComponentInChildren<TextMeshProUGUI>(true).color == StatisticsTheme.Ink, "H-MENU cold light entry has dark text");
        HistoryEntryPoints.Bind(main); HistoryEntryPoints.Bind(main);
        Check(entry.GetComponents<HistoryMenuCaption>().Length == 1 && entry.GetComponentsInChildren<HistoryControlSurface>(true).Length == 1 && entry.GetComponentsInChildren<HistoryMenuIcon>(true).Length==1, "H-MENU repeated binding is idempotent");
        router.ShowView<IMainMenuView>(); Tick(1); yield return null;
        var factory = ReflectionAccess.Get<UnityViewFactory>(router, "_viewFactory"); canvas = factory.ParentTransform.GetComponentInParent<Canvas>().rootCanvas;
        var cameraNode = new GameObject("History render camera", typeof(Camera)); camera = cameraNode.GetComponent<Camera>(); camera.enabled = false; camera.orthographic = true; camera.transform.position = new Vector3(0, 0, -100); camera.farClipPlane = 1000; camera.clearFlags = CameraClearFlags.SolidColor; camera.backgroundColor = new Color(.07f, .065f, .06f);
        canvas.renderMode = RenderMode.ScreenSpaceCamera; canvas.worldCamera = camera; canvas.planeDistance = 100;
        if (EventSystem.current == null) new GameObject("History EventSystem", typeof(EventSystem), typeof(StandaloneInputModule));
        foreach(int resolution in new[]{1200,1080}){
            var target=new RenderTexture(1920,resolution,24,RenderTextureFormat.ARGB32);target.Create();camera.targetTexture=target;
            yield return null;Capture(target,"history-main-menu-1920x"+resolution);camera.targetTexture=null;target.Release();Destroy(target);
        }
        entry.onClick.Invoke(); Tick(1); yield return null;
        var page = router.GetView<GameHistoryPanel>();
        for (float until = Time.realtimeSinceStartup + 10; Time.realtimeSinceStartup < until && !page.GetComponentsInChildren<TextMeshProUGUI>().Any(t => t.text.StartsWith("No completed games"));) yield return null;
        Check(page.IsOpen && page.Rows.Length == 0 && page.GetComponentsInChildren<TextMeshProUGUI>().Any(t => t.text.StartsWith("No completed games")), "H-MENU empty archive opens directly");
        Check(main.GetComponentsInChildren<Button>(true).All(b => !b.interactable), "H-INPUT underlying main controls disabled");
        input.Down=KeyEnum.Escape;Check(HistoryEntryPoints.ProcessInput(router),"H-INPUT history consumes Escape");
        Check(HistoryEntryPoints.ProcessInput(router),"H-INPUT closing frame prevents Escape reaching main menu");input.Down=null;
        Tick(1); yield return null; Check(router.IsViewVisible<IMainMenuView>() && entry.interactable, "H-INPUT close restores main entry");
        Check(!HistoryEntryPoints.ProcessInput(router),"H-INPUT next frame returns input ownership");
        for (int n = 1; n <= 10; n++) {
            var fixture = fixtures[(n - 1) % fixtures.Count]; var report = GameHistoryStatisticsCodec.Decode(GameHistoryStatisticsCodec.Encode(fixture.Report));
            HistoryCoreChecks.Set(report, "GameId", "ui-" + n);
            if (n == 10) { foreach (var player in report.Players) HistoryCoreChecks.Set(player, "Name", "Eleanor Coade with a particularly long duplicate player name"); }
            store.Record(report);
        }
        entry.onClick.Invoke(); Tick(1); yield return null; page = router.GetView<GameHistoryPanel>();
        for (float until = Time.realtimeSinceStartup + 10; Time.realtimeSinceStartup < until && page.Rows.Length != 10;) yield return null;
        Check(page.Rows.Length == 10 && page.TextScale == 1.15f, "H-UI ten bars and second text preset");
        page.GameScroll.OnScroll(new PointerEventData(EventSystem.current){scrollDelta=new Vector2(0,-5)});
        Check(page.GameScroll.verticalNormalizedPosition<.99f,"H-UI mouse wheel scrolls history list");page.GameScroll.verticalNormalizedPosition=1;
        var start=RectTransformUtility.WorldToScreenPoint(camera,page.GameScroll.viewport.transform.position);
        var drag=new PointerEventData(EventSystem.current){position=start,pressPosition=start,button=PointerEventData.InputButton.Left};
        page.GameScroll.OnInitializePotentialDrag(drag);page.GameScroll.OnBeginDrag(drag);drag.position=start+new Vector2(0,100);drag.delta=new Vector2(0,100);page.GameScroll.OnDrag(drag);page.GameScroll.OnEndDrag(drag);
        Check(page.GameScroll.verticalNormalizedPosition<.99f,"H-UI pointer drag scrolls history list");page.GameScroll.verticalNormalizedPosition=1;
        foreach (int resolution in new[] { 1200, 1080 }) {
            var target = new RenderTexture(1920, resolution, 24, RenderTextureFormat.ARGB32); target.Create(); camera.targetTexture = target;
            yield return null; Canvas.ForceUpdateCanvases();
            for (int preset = 0; preset < 3; preset++) {
                float scale = 1 + preset * .15f; while (page.TextScale > scale + .01f) page.ChangeTextSize(-1); while (page.TextScale < scale - .01f) page.ChangeTextSize(1);
                page.GameScroll.verticalNormalizedPosition = 1; yield return null; Canvas.ForceUpdateCanvases();
                Check(page.Rows.Length == 10 && page.GetComponentsInChildren<TextMeshProUGUI>().Where(t => t.gameObject.activeInHierarchy && !String.IsNullOrEmpty(t.text)).All(t => t.fontSize * canvas.scaleFactor >= 18), "H-UI readable labels " + resolution + "/" + preset);
                Check(page.Rows.SelectMany(r => r.GetComponentsInChildren<TextMeshProUGUI>()).Where(t => t.name.StartsWith("VP ")).All(t => t.GetPreferredValues(t.text).x <= t.rectTransform.rect.width + 1), "H-UI reserved scores fit " + resolution + "/" + preset);
                Check(page.GetComponentsInChildren<Button>().Where(b=>b.name=="Smaller text"||b.name=="Larger text"||b.name=="Close history").Select(b=>b.GetComponentInChildren<TextMeshProUGUI>()).All(t=>t.GetPreferredValues(t.text).x<=t.rectTransform.rect.width+1&&t.GetPreferredValues(t.text).y<=t.rectTransform.rect.height+1),"H-UI header controls fit "+resolution+"/"+preset);
                Capture(target, "history-1920x" + resolution + "-" + preset); layouts.Add(new { width = 1920, height = resolution, preset, games = page.Rows.Length });
                page.HandleKeyDown(KeyEnum.End); yield return null; Canvas.ForceUpdateCanvases();
                Check(EventSystem.current.currentSelectedGameObject == page.Rows.Last().gameObject && page.GameScroll.verticalNormalizedPosition < .02f, "H-UI keyboard reaches tenth row " + resolution + "/" + preset);
                var corners = new Vector3[4]; ((RectTransform)page.Rows.Last().transform).GetWorldCorners(corners);
                Check(corners.Select(c => page.GameScroll.viewport.InverseTransformPoint(c).y).Min() >= page.GameScroll.viewport.rect.yMin - 1, "H-UI tenth row inside viewport " + resolution + "/" + preset);
            }
            camera.targetTexture = null; target.Release(); Destroy(target);
        }
        while (page.TextScale > 1.16f) page.ChangeTextSize(-1); while (page.TextScale < 1.14f) page.ChangeTextSize(1);
        page.GameScroll.verticalNormalizedPosition = .37f; var selectedKey = page.Rows[5].name; page.Rows[5].onClick.Invoke(); yield return null;
        for (float until = Time.realtimeSinceStartup + 10; Time.realtimeSinceStartup < until && !router.IsViewVisible<StatisticsPanel>();) yield return null;
        var viewer = router.GetView<StatisticsPanel>();
        Check(viewer.IsOpen && viewer.OpeningContext == StatisticsOpeningContext.ArchivedHistory && GameHistoryStatisticsCodec.Key(viewer.Report.GameId) == selectedKey, "H-REUSE existing StatisticsPanel displays selected stored game");
        Check(!viewer.GetComponentsInChildren<Button>().Any(b => b.name == "Return to results" || b.name == "Return to board") && viewer.GetComponentsInChildren<Button>().Any(b => b.name == "Back to history"), "H-REUSE archived footer has no live game controls");
        foreach (int resolution in new[] { 1200, 1080 }) {
            var target = new RenderTexture(1920, resolution, 24, RenderTextureFormat.ARGB32); target.Create(); camera.targetTexture = target;
            yield return null;yield return null;Canvas.ForceUpdateCanvases();
            if(resolution==1200){
                File.WriteAllText(Path.Combine(Path.GetDirectoryName(Output),"history-statistics-geometry.json"),JsonConvert.SerializeObject(new{screen=new{Screen.width,Screen.height,safeArea=Screen.safeArea.ToString()},root=((RectTransform)viewer.transform).rect.ToString(),scale=viewer.transform.localScale.ToString(),
                    canvases=viewer.GetComponentsInParent<Canvas>().Select(c=>new{c.name,c.renderMode,root=c.rootCanvas.name,rootMode=c.rootCanvas.renderMode,c.scaleFactor,camera=c.worldCamera==null?null:c.worldCamera.name}),
                    rootCanvas=((RectTransform)canvas.transform).rect.ToString(),pixelRect=camera.pixelRect.ToString(),font=viewer.GetComponentsInChildren<TextMeshProUGUI>().First().font.name},Formatting.Indented));
            }
            for (int p = 0; p < 10; p++) { viewer.SelectPage(p); yield return null; Canvas.ForceUpdateCanvases(); Check(viewer.Report.Players.Count == store.Load(selectedKey).Report.Players.Count && viewer.BodyScroll != null, "H-REUSE stored page " + p + "/" + resolution); Capture(target, "history-statistics-1920x" + resolution + "-" + p); }
            camera.targetTexture = null; target.Release(); Destroy(target);
        }
        var statisticsReview=StatisticsHistoryReviewChecks.Run(viewer,camera,(ok,name,detail)=>Check(ok,name+" "+JsonConvert.SerializeObject(detail)),(target,cam,name)=>Capture(target,name),"history");while(statisticsReview.MoveNext())yield return statisticsReview.Current;
        viewer.Filters.Era = 1; viewer.Filters.Search = "Coal"; viewer.Close(); Tick(1); yield return null;
        Check(page.IsOpen && Math.Abs(page.GameScroll.verticalNormalizedPosition - .37f) < .02f && EventSystem.current.currentSelectedGameObject.name == selectedKey, "H-REUSE return preserves list position and focus");
        for (int index = 0; index < 10; index++) {
            string key = page.Rows[index].name; page.Rows[index].onClick.Invoke();
            for (float until = Time.realtimeSinceStartup + 10; Time.realtimeSinceStartup < until && !router.IsViewVisible<StatisticsPanel>();) yield return null;
            viewer = router.GetView<StatisticsPanel>(); Check(GameHistoryStatisticsCodec.Key(viewer.Report.GameId) == key && viewer.CurrentPage == 0 && viewer.Filters.Era == 0 && viewer.Filters.Search == "", "H-REUSE each row uses correct data and fresh filters " + index);
            viewer.Close(); Tick(.1f); yield return null;
        }
        var stale = page.Rows[0].onClick; page.Close(); Tick(1); yield return null;
        stale.Invoke(); Check(!router.IsViewCached<StatisticsPanel>() && !router.IsViewCached<GameHistoryPanel>(), "H-INPUT stale row cannot reopen disposed history");
        string bankPath=Path.Combine(store.Root,"history.json"),intact=File.ReadAllText(bankPath);Newtonsoft.Json.Linq.JObject changed;
        using(var reader=new JsonTextReader(new StringReader(intact)){DateParseHandling=DateParseHandling.None})changed=Newtonsoft.Json.Linq.JObject.Load(reader);
        changed["games"][0]["schema"]=99;File.WriteAllText(bankPath,changed.ToString(Formatting.None));
        entry.onClick.Invoke();Tick(1);yield return null;page=router.GetView<GameHistoryPanel>();
        for(float until=Time.realtimeSinceStartup+10;Time.realtimeSinceStartup<until&&page.Rows.Length!=10;)yield return null;
        Check(page.Rows.Length==10&&!page.Rows[0].interactable&&page.Rows.Skip(1).All(r=>r.interactable),"H-UI unsupported report disables only its own row");
        Check(!page.Rows[0].GetComponentsInChildren<TextMeshProUGUI>().Any(t=>t.name.StartsWith("VP ")||t.text.Contains("0 players")),"H-UI unavailable report never invents scores or lineup");
        page.Rows[0].onClick.Invoke();yield return null;Check(!router.IsViewCached<StatisticsPanel>(),"H-INPUT unavailable report cannot open zero-filled charts");
        page.Close();Tick(1);yield return null;File.WriteAllText(bankPath,intact);
        var rejected=GameHistoryStatisticsCodec.Decode(GameHistoryStatisticsCodec.Encode(fixtures.Last().Report));HistoryCoreChecks.Set(rejected,"GameId","publication-failed");
        store.Fault=stage=>{if(stage=="before-write")throw new IOException("Injected history storage failure");};HistoryRecording.Queue(rejected);HistoryRecording.Flush();
        Check(HistoryRecording.LastFailure!=null&&File.ReadAllText(bankPath)==intact,"H-CAPTURE failed background write preserves prior reports");store.Fault=null;
        entry.onClick.Invoke();Tick(1);yield return null;page=router.GetView<GameHistoryPanel>();
        for(float until=Time.realtimeSinceStartup+10;Time.realtimeSinceStartup<until&&page.Rows.Length!=10;)yield return null;
        Check(page.Rows.Length==10&&page.GetComponentsInChildren<TextMeshProUGUI>().Any(t=>t.text.Contains("was not recorded")),"H-UI failed capture has a brief message and usable earlier rows");
        page.Close();Tick(1);yield return null;
        var completed = fixtures.Last(); var gameHistory = new BrassGameHistory { Repository = completed.Repository }; gameHistory.SetGame(completed.App);
        var owner = new FinishedGameReviewUseCase(new CapturePresenter(router), gameHistory); owner.StartGameReview(completed.App.Metadata.PlayerMetadata.First());
        for (float until = Time.realtimeSinceStartup + 10; Time.realtimeSinceStartup < until && HistoryRecording.IsWriting;) yield return null; HistoryRecording.Drain();
        Check(store.List().Games.Count == 10 && store.List().Games[0].Key == GameHistoryStatisticsCodec.Key(completed.App.Metadata.GameId) && !router.IsViewCached<StatisticsPanel>(), "H-CAPTURE completion auto-records without opening Statistics");
        owner.StartGameReview(completed.App.Metadata.PlayerMetadata.Last()); HistoryRecording.Flush();
        Check(store.List().Games.Count == 10 && store.List().Games[0].Sequence == 11, "H-CAPTURE repeated player review cannot consume another slot");
        owner.FinishGameReview();
        foreach(var type in new[]{GameType.Tutorial,GameType.Network}){
            var excluded=HistoryFixture.Finish(2,false,20261080+(int)type,true,type);
            var excludedHistory=new BrassGameHistory{Repository=excluded.Repository};excludedHistory.SetGame(excluded.App);
            var excludedOwner=new FinishedGameReviewUseCase(new CapturePresenter(router),excludedHistory);
            StatisticsEntryPoints.Capture(excludedOwner,excluded.App.Metadata.PlayerMetadata.First());HistoryRecording.Flush();
            Check(store.List().Games[0].Sequence==11,"H-CAPTURE excludes "+type);
        }
        var unfinished=HistoryFixture.Start(2,false,20261090);var unfinishedHistory=new BrassGameHistory{Repository=unfinished.Repository};unfinishedHistory.SetGame(unfinished.App);
        StatisticsEntryPoints.Capture(new FinishedGameReviewUseCase(new CapturePresenter(router),unfinishedHistory),unfinished.App.Metadata.PlayerMetadata.First());HistoryRecording.Flush();
        Check(store.List().Games[0].Sequence==11,"H-CAPTURE excludes in-progress match");
        HistoryRecording.UseTestStore(null); Destroy(cameraNode); yield return null;
    }
    void Capture(RenderTexture target, string name)
    {
        Canvas.ForceUpdateCanvases(); camera.Render(); var previous = RenderTexture.active; RenderTexture.active = target;
        var image = new Texture2D(target.width, target.height, TextureFormat.RGB24, false); image.ReadPixels(new Rect(0, 0, target.width, target.height), 0, 0); image.Apply(); RenderTexture.active = previous;
        string path = Path.Combine(Path.GetDirectoryName(Output), name + ".png"); File.WriteAllBytes(path, image.EncodeToPNG()); images.Add(path); Destroy(image);
    }
    static void Tick(float seconds) { for (float t = 0; t < seconds; t += .05f) DOTween.ManualUpdate(.05f, .05f); DOTween.CompleteAll(); }
    sealed class NativeInput : IInputSource {public KeyEnum? Down; public bool GetKeyDown(KeyEnum key) { return Down==key; } public bool GetKey(KeyEnum key) { return false; } public bool GetKeyUp(KeyEnum key) { return false; } }
    sealed class Localization:ILocalization {public string GetTranslation(string term){return term.Split('/').Last();}}
    sealed class VersionStub:BrassApplication.Utils.IVersionReader {public string PublicVersion{get{return "1.4.23";}}public string BuildVersion{get{return "history qualification";}}}
    sealed class Builder : IUseCaseBuilder
    {
        public MainMenuUseCase Menu;
        public void BuildUseCases() { } public void RestoreUseCases() { } public void RestoreUseCaseOfType(Type type) { }
        public IController GetController(Type type) { return typeof(IMainMenuView).IsAssignableFrom(type) ? new MainMenuController(Menu) : null; }
        public T GetUseCase<T>() where T : IUseCase { throw new NotSupportedException(); }
    }
    sealed class CapturePresenter : IFinishedGameReviewOutput
    {
        readonly Router _router; public CapturePresenter(Router router) { _router = router; }
        public void ReplayUnseenActions(PlayerMetadata player, IReplayDelegate callback) { }
        public void ShowSkipToMyNextTurnView() { } public void HidePlayerActionView() { } public void ShowButtonToLaunchGameSummaryTable() { }
        public void ShowGameSummaryTable(List<(BoardGameRules.Entities.Players.PlayerColor, PlayerAvatar, string, int, int, int, int, int, int)> scores) { }
        public void ShowIntroductoryGameSummaryTable(List<(BoardGameRules.Entities.Players.PlayerColor, PlayerAvatar, string, int, int, int, int, int, int, int, int)> scores) { }
    }
}
