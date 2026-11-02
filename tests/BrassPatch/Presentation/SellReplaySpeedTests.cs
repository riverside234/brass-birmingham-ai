using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using BoardGameRules.Entities.Actions.Events;
using BoardGameRules.Entities.Board.BuildingSlots;
using BoardGameRules.Entities.Board.Events;
using BoardGameRules.Entities.Board.Regions;
using BoardGameRules.Entities.EventSourcing.Events;
using BoardGameRules.Entities.GamePhases;
using BoardGameRules.Entities.NumericalResources.Events;
using BrassApplication.UseCases.Game.Replay;
using BrassPatch;
using BrassPatch.Tests.CityLocator;
using DG.Tweening;
using Frameworks.Routing.Unity;
using Frameworks.Utils;
using Frameworks.View.Animation;
using Frameworks.View.Common;
using Frameworks.View.Game;
using Frameworks.View.Game.Replay;
using InterfaceAdapters.Common;
using InterfaceAdapters.Game.Actions.SellAction;
using InterfaceAdapters.Game.Board;
using InterfaceAdapters.Game.HandOfCards;
using InterfaceAdapters.Game.PlayersSummary;
using InterfaceAdapters.Game.Replay;
using InterfaceAdapters.Routing;
using InterfaceAdapters.Utils.MainThreadDispatcher;
using Newtonsoft.Json;
using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.SceneManagement;

public static class LocalAiUnityTests
{
    public static void Begin(string output)
    {
        var driver = new GameObject("Sell replay speed regressions").AddComponent<SellReplaySpeedDriver>();
        UnityEngine.Object.DontDestroyOnLoad(driver.gameObject);
        driver.Output = output;
    }
}

public sealed class SellReplaySpeedDriver : MonoBehaviour
{
    public string Output;
    readonly List<object> cases = new List<object>();
    readonly List<string> errors = new List<string>();
    readonly List<object> timings = new List<object>();
    int checks, failures, reproductions;
    bool baseline;
    void Check(bool ok, string name, object detail = null)
    { checks++; if (!ok) failures++; cases.Add(new { name, passed = ok, detail }); }
    void Log(string message, string stack, LogType type)
    {
        if (message.StartsWith("Error saving PlayerPrefs LastGoogleUpdate_I2Languages")) return;
        if (type == LogType.Error || type == LogType.Exception) errors.Add(message);
    }
    IEnumerator Start()
    {
        yield return SceneManager.LoadSceneAsync("GameScene"); yield return null;
        Application.logMessageReceived += Log;
        baseline = Environment.GetCommandLineArgs().Contains("--brass-sell-speed-baseline");
        Exception failure = null; var run = Run();
        while (true) {
            bool more = false; object current = null;
            try { more = run.MoveNext(); if (more) current = run.Current; } catch (Exception e) { failure = e; }
            if (!more || failure != null) break; yield return current;
        }
        DOTween.KillAll(); UIProperties.CurrentAnimationSpeed = UIProperties.DefaultAnimationSpeed;
        bool passed = failure == null && failures == 0 && errors.Count == 0;
        File.WriteAllText(Output, JsonConvert.SerializeObject(new { passed, baseline, checks, failures, reproductions,
            cases, timings, nativeErrors = errors, error = failure == null ? null : failure.ToString() }, Formatting.Indented));
        Application.Quit(passed ? 0 : 1);
    }
    IEnumerator Run()
    {
        var prefabs = Resources.FindObjectsOfTypeAll<UnityViewFactory>().SelectMany(f => f.ViewList.Views).OfType<BaseView>().ToArray();
        UnityCityLocatorRegression.Prefab = prefabs.OfType<BoardView>().First(v => v.GetType() == typeof(BoardView));
        if (EventSystem.current == null) new GameObject("Replay speed event system", typeof(EventSystem));
        foreach (int count in new[] { 2, 4 })
        foreach (GameEra era in new[] { GameEra.CanalEra, GameEra.RailEra })
        using (var f = new UnityCityLocatorRegression.Fixture(count)) {
            f.Root.GetComponent<Canvas>().sortingOrder = 1000;
            var g = f.App.Game; g.GameProgressInformation.CurrentGameEra = era;
            var player = g.GameProgressInformation.CurrentPlayer;
            var slots = new List<BoardGameRules.Entities.Board.BuildingSlots.BuildingSlot>();
            foreach (var type in new[] { BuildingType.CottonMill, BuildingType.Manufacturer, BuildingType.Pottery, BuildingType.Brewery }) {
                var slot = g.Board.Regions.SelectMany(r => r.BuildingSlots).First(s => s.BuildedIndustry == null && s.BuildingTypes.Contains(type));
                slot.BuildedIndustry = player.Mat.GetFirstBuildingOfType(type); slots.Add(slot);
            }
            f.Update();
            var router = new TestRouter { Board = f.View };
            router.Replay = UnityEngine.Object.Instantiate(prefabs.OfType<ReplayView>().First(), f.Root.transform);
            router.Hand = UnityEngine.Object.Instantiate(prefabs.First(v => v is IHandOfCardsView), f.Root.transform);
            router.Summary = UnityEngine.Object.Instantiate(prefabs.OfType<PlayersSummaryView>().First(), f.Root.transform);
            foreach (var view in new BaseView[] { router.Replay, router.Hand, router.Summary }) view.InputRegistration = router;
            router.Summary.UpdatePlayersSummaryView(g.PlayersTrack.Players.Select(p => new PlayerViewModel { Color = p.Color, Name = p.Color.ToString() }).ToList());
            ReflectionAccess.Set(router.Replay, "_router", router); Utils.Global.Instance.Router = router;
            router.Replay.transform.SetAsLastSibling();
            router.Replay.CreateShowViewImmediatelyAnimation(); Tick(.1f);
            yield return null; Canvas.ForceUpdateCanvases();
            var history = new BrassApplication.GameHistory.BrassGameHistory(); history.SetGame(f.App);
            Utils.Global.Instance.BrassGameHistory = history;
            var dispatcher = new Dispatcher(); var presenter = new ReplayPresenter(router, dispatcher, history, f.Localization);
            var sell = new SellActionPresenter(router, f.Localization);
            var button = ReflectionAccess.Get<GameObject>(router.Replay, "_fastForwardButton");
            var nativeButton = button.GetComponent<UnityEngine.UI.Button>();
            cases.Add(new { name = "native speed input", count, era = era.ToString(), components = button.GetComponents<Component>().Select(c => c.GetType().FullName).ToArray(),
                listeners = Enumerable.Range(0, nativeButton.onClick.GetPersistentEventCount()).Select(i => new { method = nativeButton.onClick.GetPersistentMethodName(i), target = nativeButton.onClick.GetPersistentTarget(i).name }).ToArray() });
            var events = slots.Take(3).Select(s => (IEvent)new IndustryFlipped(s.BuildedIndustry.UniqueId, g.Board.GetRegionContainingGivenIndustry(s.BuildedIndustry).Name)).ToList();
            events.Add(new BeerSupplied(slots[3].BuildedIndustry.UniqueId, 1));
            events.Add(new IncomeMarkerPositionChanged(player.Color, 2, slots[0].BuildedIndustry.UniqueId));
            events.Add(new MerchantBonusReceived(BonusType.Money, player.Color, RegionName.Oxford));
            events.Add(new MerchantBonusReceived(BonusType.Income, player.Color, RegionName.Stafford));
            foreach (IEvent change in events) {
                DOTween.KillAll(); router.Replay.RestoreDefaultAnimationSpeed();
                sell.SetFastForwardButtonActive(false);
                // A menu/highlight animation must not turn the replay blocker back on.
                var unrelated = new DOTweenAnimationSequence(DOTween.Sequence().AppendInterval(2f));
                unrelated.Play(); Check(!button.activeSelf, "selection/view animation leaves resource input accessible"); unrelated.Kill();
                string before = State(f);
                var recorder = new Recorder(); presenter.ReplayEvent(change, recorder);
                // This is the stock sell state-machine epilogue. It runs AFTER
                // constructing a nested sale replay and BEFORE its dispatched Play.
                sell.SetFastForwardButtonActive(false);
                for (int i = 0; i < 100 && dispatcher.Count == 0; i++) yield return null;
                dispatcher.Flush();
                yield return null; // Register the newly enabled native Graphic before raycasting.
                var active = DOTween.TweensById("AnimationFactory");
                Check(active != null && active.Count == 1 && active[0].IsPlaying(), "sale has one playing replay root", change.GetType().Name);
                if (active == null || active.Count != 1) throw new InvalidOperationException("Missing replay sequence");
                var sequence = active[0];
                if (!baseline) {
                    Canvas.ForceUpdateCanvases();
                    var pointer = new PointerEventData(EventSystem.current) { position = new Vector2(Screen.width * .5f, Screen.height * .5f) };
                    var hits = new List<RaycastResult>(); EventSystem.current.RaycastAll(pointer, hits);
                    var first = hits.FirstOrDefault();
                    Check(first.gameObject != null && ExecuteEvents.GetEventHandler<IPointerClickHandler>(first.gameObject) == button,
                        "board screen click reaches native fast-forward button", new {
                            hits = hits.Take(3).Select(h => h.gameObject.transform.parent.name + "/" + h.gameObject.name).ToArray(),
                            rect = ((RectTransform)button.transform).rect.ToString(), position = button.transform.position.ToString(),
                            raycast = button.GetComponent<UnityEngine.UI.Image>().raycastTarget,
                            groups = button.GetComponentsInParent<CanvasGroup>().Select(c => new { c.name, c.alpha, c.blocksRaycasts, c.interactable }).ToArray() });
                }
                Tick(.1f); float elapsed = sequence.Elapsed(false);
                Click(button); Click(button);
                if (baseline) {
                    bool reproduced = !button.activeSelf && UIProperties.CurrentAnimationSpeed == 1f && sequence.timeScale == 1f;
                    if (reproduced) reproductions++;
                    Check(reproduced, "baseline reproduces unavailable sale fast-forward", change.GetType().Name);
                } else {
                    Check(button.activeInHierarchy && UIProperties.CurrentAnimationSpeed == 2f && sequence.timeScale == 2f,
                        "native screen clicks enable exactly 2x sale playback", new { count, era = era.ToString(), change = change.GetType().Name,
                            self = button.activeSelf, hierarchy = button.activeInHierarchy, visible = router.Replay.IsViewVisible(), speed = UIProperties.CurrentAnimationSpeed,
                            sequenceSpeed = sequence.timeScale, id = sequence.id, interactable = nativeButton.IsInteractable() });
                    Tick(.1f);
                    Check(Math.Abs(sequence.Elapsed(false) - elapsed - .2f) < .005f, "sale timeline advances at twice normal speed");
                    router.Replay.PauseAllAnimations(); elapsed = sequence.Elapsed(false); Tick(.1f);
                    Check(Math.Abs(sequence.Elapsed(false) - elapsed) < .001f, "2x sale replay respects pause");
                    router.Replay.ResumeAllAnimations();
                    Check(sequence.timeScale == 2f, "resume retains 2x choice");
                }
                Tick(8f);
                Check(recorder.Count == 1, "sale replay completes exactly once");
                Check(State(f) == before, "speed/input changes leave game state unchanged");
                router.Replay.RestoreDefaultAnimationSpeed();
                Check(UIProperties.CurrentAnimationSpeed == 1f, "next action restores default speed");
                sell.SetFastForwardButtonActive(false);
                Check(!button.activeSelf, "completed replay yields to beer/merchant/next-sale selection");
                if (!baseline && change is IndustryFlipped) {
                    float normal = Measure(router.Replay, change, false), fast = Measure(router.Replay, change, true);
                    timings.Add(new { count, era = era.ToString(), industry = slots[events.IndexOf(change)].BuildedIndustry.BuildingType.ToString(), normal, fast, ratio = normal / fast });
                    Check(normal > 0 && fast > 0 && Math.Abs(normal / fast - 2f) < .04f, "same industry flip completes in half the simulated time");
                }
            }
            // A hidden/cancelled replay cannot reopen a screen-wide input blocker.
            router.Replay.DisableView(); sell.SetFastForwardButtonActive(false);
            new DOTweenAnimationSequence(DOTween.Sequence().AppendInterval(.1f).SetId("AnimationFactory")).Play();
            Check(!button.activeSelf, "hidden replay does not capture clicks after cancellation");
            DOTween.KillAll();
        }
    }
    static string State(UnityCityLocatorRegression.Fixture f)
    {
        var g = f.App.Game;
        return JsonConvert.SerializeObject(new { g.Revision, events = g.GetUncommittedEvents().Count,
            actions = g.TurnFlow.GetRemainingActionsCount(), players = g.PlayersTrack.Players.Select(p => new { p.Color, p.Money, p.IncomeMarkerPosition, p.VP, cards = p.Hand.Count }).ToArray(),
            industries = g.Board.Regions.SelectMany(r => r.BuildingSlots).Where(s => s.BuildedIndustry != null).Select(s => new { s.UniqueId, s.BuildedIndustry.IsFlipped }).ToArray() });
    }
    static void Click(GameObject target)
    {
        var data = new PointerEventData(EventSystem.current) { button = PointerEventData.InputButton.Left, clickCount = 2 };
        ExecuteEvents.Execute(target, data, ExecuteEvents.pointerClickHandler);
    }
    static float Measure(ReplayView view, IEvent change, bool fast)
    {
        view.RestoreDefaultAnimationSpeed();
        var animation = (DOTweenAnimationSequence)view.CreateAnimation(change);
        animation.Sequence.SetAutoKill(false); animation.Play();
        if (fast) { var button = ReflectionAccess.Get<GameObject>(view, "_fastForwardButton"); Click(button); }
        float elapsed = 0;
        while (!animation.Sequence.IsComplete() && elapsed < 10f) { Tick(.01f); elapsed += .01f; }
        animation.Kill(); return elapsed;
    }
    static void Tick(float duration)
    {
        var update = typeof(DOTween).Assembly.GetType("DG.Tweening.Core.TweenManager").GetMethod("Update", ReflectionAccess.Flags);
        for (int i = 0; i < (int)Math.Round(duration * 100); i++) update.Invoke(null, new object[] { UpdateType.Normal, .01f, .01f });
    }
    sealed class Recorder : IEventReplayDelegate
    { public int Count; public void EventReplayCompleted(IEvent change) { Count++; } }
    sealed class Dispatcher : IMainThreadDispatcher
    {
        readonly Queue<Action> actions = new Queue<Action>();
        public int Count { get { lock (actions) return actions.Count; } }
        public bool isMainThread { get { return true; } }
        public void InvokeAsync(Action action) { lock (actions) actions.Enqueue(action); }
        public void Invoke(Action action) { InvokeAsync(action); }
        public void Flush() { lock (actions) while (actions.Count > 0) actions.Dequeue()(); }
    }
    sealed class TestRouter : IRouter
    {
        public BoardView Board; public ReplayView Replay; public BaseView Hand; public PlayersSummaryView Summary;
        public IUseCaseBuilder UseCaseBuilder { get; set; }
        public T GetView<T>() where T : class, IBaseView { return new IBaseView[] { Replay, Board, Hand, Summary }.OfType<T>().FirstOrDefault(); }
        public bool IsViewCached<T>() where T : class, IBaseView { return GetView<T>() != null; }
        public bool IsViewVisible<T>() where T : class, IBaseView { var view = GetView<T>() as BaseView; return view != null && view.IsViewVisible(); }
        public IAnimation ShowView<T>() where T : class, IBaseView { return GetView<T>().CreateShowViewAnimation(); }
        public IAnimation HideView<T>() where T : class, IBaseView { return GetView<T>().CreateHideViewAnimation(); }
        public void MoveOnTop<T>() where T : class, IBaseView { }
        public void ProcessInput() { }
        public void RegisterKey(KeyEnum key, IBaseView view) { }
        public void UnregisterKey(KeyEnum key, IBaseView view) { }
        public void UnregisterAllKeys(IBaseView view) { }
    }
}
