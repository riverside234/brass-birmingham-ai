using System;
using System.Collections;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using BoardGameRules.Entities.Actions.Events;
using BoardGameRules.Entities.Actions;
using BoardGameRules.Entities.Board.BuildingSlots;
using BoardGameRules.Entities.Board.Links;
using BoardGameRules.Entities.Board.Regions;
using BoardGameRules.Entities.EventSourcing.Events;
using BoardGameRules.Entities.GamePhases;
using BoardGameRules.Entities.GamePhases.Events;
using BoardGameRules.Entities.Industries;
using BoardGameRules.Entities.NumericalResources.Events;
using BrassApplication.Game;
using BrassApplication.GameHistory;
using BrassApplication.UseCases.Common;
using BrassApplication.UseCases.Game.Replay;
using BrassApplication.UseCases.Game.Actions;
using BrassApplication.UseCases.Game.Actions.BaseAction;
using BrassApplication.UseCases.Game.Actions.BeerDelivery;
using BrassApplication.UseCases.Game.Actions.CoalDelivery;
using BrassApplication.UseCases.Game.Actions.NetworkAction;
using BrassPatch;
using BrassPatch.Tests.CityLocator;
using DG.Tweening;
using Frameworks.Routing.Unity;
using Frameworks.View.Common;
using Frameworks.View.Game.Replay;
using Frameworks.Persistence.Repository;
using Frameworks.Persistence.RepositoryBackends.InMemory;
using Frameworks.Persistence.GameFactory;
using Frameworks.Utils;
using InterfaceAdapters.Common;
using InterfaceAdapters.Game.Board;
using InterfaceAdapters.Game.Replay;
using InterfaceAdapters.Routing;
using InterfaceAdapters.Utils.MainThreadDispatcher;
using Newtonsoft.Json;
using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.UI;

public static class LocalAiUnityTests
{
    public static void Begin(string output)
    {
        var driver = new GameObject("Link replay regressions").AddComponent<LinkReplayDriver>();
        UnityEngine.Object.DontDestroyOnLoad(driver.gameObject); driver.Output = output;
    }
}
public sealed class LinkReplayDriver : MonoBehaviour
{
    public string Output;
    readonly List<object> cases = new List<object>();
    readonly List<string> errors = new List<string>();
    int checks, failures;
    void Check(bool passed, string name, object detail = null)
    { checks++; if (!passed) failures++; cases.Add(new { name, passed, detail }); }
    void Log(string message, string stack, LogType kind)
    {
        if (message.StartsWith("Error saving PlayerPrefs LastGoogleUpdate_I2Languages")) return;
        if (kind == LogType.Error || kind == LogType.Exception) errors.Add(message);
    }
    IEnumerator Start()
    {
        yield return SceneManager.LoadSceneAsync("GameScene"); yield return null;
        Application.logMessageReceived += Log;
        Exception failure = null; var run = Run();
        while (true) {
            bool more = false; object current = null;
            try { more = run.MoveNext(); if (more) current = run.Current; } catch (Exception e) { failure = e; }
            if (failure != null || !more) break; yield return current;
        }
        File.WriteAllText(Output, JsonConvert.SerializeObject(new { passed = failure == null && failures == 0 && errors.Count == 0,
            checks, failures, cases, nativeErrors = errors, error = failure == null ? null : failure.ToString() }, Formatting.Indented));
        Application.Quit(failure == null && failures == 0 && errors.Count == 0 ? 0 : 1);
    }
    IEnumerator Run()
    {
        var prefabs = Resources.FindObjectsOfTypeAll<UnityViewFactory>().SelectMany(f => f.ViewList.Views).OfType<BaseView>().ToArray();
        UnityCityLocatorRegression.Prefab = prefabs.OfType<BoardView>().First(v => v.GetType() == typeof(BoardView));
        bool staleBaseline = Environment.GetCommandLineArgs().Contains("--brass-network-stale-baseline");
        if (staleBaseline) {
            foreach (int count in new[] { 2, 3, 4 }) {
                var reproduction = Flow(prefabs, count, "canal-opening", false, true);
                while (reproduction.MoveNext()) yield return reproduction.Current;
            }
            yield break;
        }
        bool baseline = Environment.GetCommandLineArgs().Contains("--brass-link-replay-baseline");
        foreach (int count in baseline ? new[] { 2 } : new[] { 2, 3, 4 })
        foreach (string era in new[] { "Canal", "Rail" })
        using (var f = new UnityCityLocatorRegression.Fixture(count))
        {
            var router = new TestRouter { Board = f.View };
            var replay = UnityEngine.Object.Instantiate(prefabs.OfType<ReplayView>().First(), f.Root.transform);
            router.Replay = replay; replay.InputRegistration = router;
            ReflectionAccess.Set(replay, "_router", router); Utils.Global.Instance.Router = router;
            replay.CreateShowViewImmediatelyAnimation(); UnityCityLocatorRegression.Tick(.1f);
            var dispatcher = new Dispatcher(); var history = new BrassGameHistory(); history.SetGame(f.App);
            var presenter = new ReplayPresenter(router, dispatcher, history, f.Localization);
            foreach (var player in baseline ? f.App.Game.PlayersTrack.Players.Take(1) : f.App.Game.PlayersTrack.Players.AsEnumerable())
            foreach (float speed in baseline ? new[] { 1f } : new[] { 1f, 2f })
            foreach (bool interrupt in new[] { false, true })
            {
                UIProperties.CurrentAnimationSpeed = speed;
                var recorder = new Recorder();
                var change = new LinkPlaced(player.Color, era + "BirminghamDudley");
                string before = State(f.App);
                presenter.ReplayEvent(change, recorder);
                for (int i = 0; i < 120; i++) {
                    dispatcher.Flush();
                    var active = DOTween.TweensById("AnimationFactory");
                    if (active != null && active.Any(t => t.IsPlaying())) break;
                    yield return null;
                }
                dispatcher.Flush(); UnityCityLocatorRegression.Tick(.4f / speed);
                var sequence = DOTween.TweensById("AnimationFactory").First();
                var link = f.View.GetComponentsInChildren<LinkView>(true).First(v => v.name == "BirminghamDudley");
                var token = link.transform.Find("LinkToken").GetComponent<Image>();
                if (!baseline) Check(token.color.a == 1f, "link is visible before interruption", new { era, count, speed });
                if (interrupt) sequence.Kill();
                UnityCityLocatorRegression.Tick(3f); yield return null; yield return null;
                Check(recorder.Count == 1, interrupt ? "killed placed-link replay returns control once" : "normal placed-link replay completes once",
                    new { era, count, color = player.Color.ToString(), speed, callbacks = recorder.Count, interrupt });
                if (!baseline) {
                    Check(token.sprite == Resources.Load<Sprite>("Board/Links/Token" + era + "Era" + player.Color)
                        && token.transform.localScale == Vector3.one, "owned token restores its final sprite and scale");
                    Check(!link.transform.Find("LinkToken/Outline").gameObject.activeSelf, "finished placement clears transient outline");
                    Check(State(f.App) == before, "animation recovery leaves rules state unchanged");
                }
                presenter.CancelCurrentReplay(); DOTween.KillAll();
            }
            UIProperties.CurrentAnimationSpeed = 1f;
            var queued = new Recorder(); presenter.ReplayEvent(new LinkPlaced(f.App.Game.GameProgressInformation.CurrentPlayer.Color,
                era + "BirminghamDudley"), queued);
            for (int i = 0; i < 120 && dispatcher.Count == 0; i++) yield return null;
            Check(dispatcher.Count > 0, "link Play is queued before the cancellation test");
            presenter.CancelCurrentReplay(); dispatcher.Flush(); UnityCityLocatorRegression.Tick(4f); yield return null;
            Check(queued.Count == 0, "queued Play cannot resume an intentionally cancelled link replay", new { era, count, callbacks = queued.Count });
            presenter.CancelCurrentReplay(); DOTween.KillAll();
            var highlighted = f.View.GetComponentsInChildren<LinkView>(true).First(v => v.name == "BirminghamDudley");
            var model = new LinkViewModel { IsOutlineVisible = true, IsTokenVisible = false, CurrentEra = GameEra.CanalEra };
            for (int i = 0; i < 20; i++) { highlighted.UpdateLink(model, null); UnityCityLocatorRegression.Tick(.7f); }
            Check(DOTween.TotalPlayingTweens() == 1, "repeated link highlights own exactly one pulse", new { count, era, playing = DOTween.TotalPlayingTweens() });
            highlighted.UpdateLink(new LinkViewModel { CurrentEra = GameEra.CanalEra }, null);
            Check(DOTween.TotalPlayingTweens() == 0, "clearing highlights leaves no orphan pulse", new { count, era, playing = DOTween.TotalPlayingTweens() });
            DOTween.KillAll();
            var scoring = Scoring(f, router, presenter, dispatcher, era, baseline);
            while (scoring.MoveNext()) yield return scoring.Current;
            if (!baseline) {
                var cancellation = Cancellation(f, router, presenter, dispatcher, era);
                while (cancellation.MoveNext()) yield return cancellation.Current;
            }
        }
        if (baseline) yield break;
        foreach (int count in new[] { 2, 3, 4 })
        foreach (string kind in new[] { "canal-opening", "canal-later", "single-rail", "double-rail" })
        foreach (bool interrupt in new[] { false, true }) {
            var flow = Flow(prefabs, count, kind, interrupt);
            while (flow.MoveNext()) yield return flow.Current;
        }
        foreach (int count in new[] { 2, 3, 4 }) {
            var cancelled = CancelledPrompt(prefabs, count);
            while (cancelled.MoveNext()) yield return cancelled.Current;
            SearchContinuation(count);
        }
    }
    static string State(BrassApplicationGame app)
    { var g = app.Game; return JsonConvert.SerializeObject(new { g.Revision, events = g.GetUncommittedEvents().Count,
        actions = g.TurnFlow.GetRemainingActionsCount(), players = g.PlayersTrack.Players.Select(p => new { p.Color, p.Money, p.IncomeMarkerPosition, p.LinkTileCount, p.VP, hand = p.Hand.Count }).ToArray() }); }
    void SearchContinuation(int count)
    {
        var config = new BoardGameRules.Entities.Game.GameStartConfiguration((BoardGameRules.Entities.Common.NumberOfPlayers)count);
        var configuration = BrassHost.Host.Configuration(config);
        var host = new BrassHost.Host(270926, new Newtonsoft.Json.Linq.JObject { ["configuration"] = configuration });
        var g = host.Game; var player = host.Player;
        g.ApplyEvent(new RailEraStarted()); g.ApplyEvent(new MoneyLoaned(player.Color, 100));
        g.ApplyEvent(new LinkPlaced(player.Color, "RailBirminghamOxford"));
        while (player.Mat.GetFirstBuildingOfType(BuildingType.Brewery).Level < 2) g.ApplyEvent(new IndustryDeveloped(player.Color, BuildingType.Brewery));
        var slot = g.Board.GetRegionByName(RegionName.Stafford).BuildingSlots.First(s => s.BuildingTypes.Contains(BuildingType.Brewery));
        g.ApplyEvent(new IndustryBuilt(player.Color, BuildingType.Brewery, slot.UniqueId));
        host.SearchApply(host.Choices().First(c => c.kind == "network" && c.target == "RailBirminghamDudley").id);
        host.SearchApply(host.Choices().First(c => c.kind == "coal").id);
        var restored = new BrassHost.Host(270926, host.TrustedSetup());
        Check(ReflectionAccess.Get<BoardGameRules.Entities.Cards.BaseCard>(restored.Game.TurnFlow.NetworkAction, "_chosenCard") == null &&
            restored.Game.TurnFlow.CurrentAction != null, "N34-SEARCH: fixture restores the real AI partial-action format", new { count });
        restored.SearchApply(restored.Choices().First(c => c.kind == "network").id);
        restored.SearchApply(restored.Choices().First(c => c.kind == "coal").id);
        restored.SearchApply(restored.Choices().First(c => c.kind == "beer").id);
        restored.SearchApply(new BrassHost.Choice("finish").id);
        Check(restored.ActionComplete && restored.Game.TurnFlow.CurrentAction == null && !g.Board.GetLinkById("RailBirminghamWalsall").HasOwner,
            "N34-SEARCH: AI restored double-Rail search pays coal and beer and finishes without touching the live world", new { count });
    }
    static void GuardTick(ReplayView view, float delta)
    {
        var guard = view.GetComponents<Component>().First(c => c.GetType().Name == "LinkReplayGuard");
        var ticket = ReflectionAccess.Get<object>(guard, "Current");
        ticket.GetType().GetMethod("Tick", ReflectionAccess.Flags).Invoke(ticket, new object[] { delta });
    }
    IEnumerator Scoring(UnityCityLocatorRegression.Fixture f, TestRouter router, ReplayPresenter presenter, Dispatcher dispatcher, string era, bool baseline)
    {
        var link = f.View.GetComponentsInChildren<LinkView>(true).First(v => v.name == "BirminghamDudley");
        var token = ReflectionAccess.Get<Image>(link, "_tokenImage");
        var label = ReflectionAccess.Get<TMPro.TextMeshProUGUI>(link, "_victoryPoint");
        var tokenPosition = token.transform.localPosition; var labelPosition = label.transform.localPosition; string text = label.text;
        token.color = Color.white; link.RemoveTokenAnimation(17); UnityCityLocatorRegression.Tick(.2f);
        Check(token.transform.parent != link.transform, "scoring still raises the link token above the board");
        DOTween.KillAll();
        Check(token.transform.parent == link.transform && label.transform.parent == link.transform
            && token.transform.localScale == Vector3.one && label.transform.localScale == Vector3.one && label.text == text,
            "interrupted scoring restores link token and VP label parents and geometry", new { era, baseline });
        var recorder = new Recorder(); Exception missing = null;
        try { presenter.ReplayEvent(new LinkPlaced(f.App.Game.GameProgressInformation.CurrentPlayer.Color, "RailBirminghamDudley"), recorder); }
        catch (Exception e) { missing = e; }
        for (int i = 0; i < 120 && recorder.Count == 0 && missing == null; i++) { dispatcher.Flush(); UnityCityLocatorRegression.Tick(.1f); yield return null; }
        Check(missing == null && recorder.Count == 1, "rail placement after interrupted Canal scoring returns control", new { error = missing == null ? null : missing.GetType().Name, callbacks = recorder.Count });
        presenter.CancelCurrentReplay(); DOTween.KillAll();
        // Repair only the disposable baseline fixture so later checks can continue.
        token.transform.SetParent(link.transform, false); token.transform.localPosition = tokenPosition; token.transform.localScale = Vector3.one;
        label.transform.SetParent(link.transform, false); label.transform.localPosition = labelPosition; label.transform.localScale = Vector3.one; label.text = text; label.alpha = 0f;
        if (baseline) yield break;
        foreach (int points in new[] { 1, 9, 17, 30 }) {
            link.RemoveTokenAnimation(points); UnityCityLocatorRegression.Tick(3f);
            Check(token.transform.parent == link.transform && label.transform.parent == link.transform && label.text == text,
                "normal scoring restores reusable link geometry", new { points });
        }
        link.RemoveTokenAnimation(17); UnityCityLocatorRegression.Tick(.2f); f.Update(); UnityCityLocatorRegression.Tick(3f);
        Check(token.transform.parent == link.transform && label.transform.parent == link.transform && token.transform.localPosition == tokenPosition,
            "a board refresh cancels obsolete scoring before rebinding a link");
        link.RemoveTokenAnimation(17); UnityCityLocatorRegression.Tick(.2f);
        recorder = new Recorder(); presenter.ReplayEvent(new LinkPlaced(f.App.Game.GameProgressInformation.CurrentPlayer.Color, "RailBirminghamDudley"), recorder);
        dispatcher.Flush(); UnityCityLocatorRegression.Tick(4f); yield return null;
        Check(recorder.Count == 1 && token.transform.parent == link.transform && token.color.a == 1f && token.transform.localScale == Vector3.one,
            "new Rail placement cancels pending scoring without a late fade or reparent");
        ReflectionAccess.Set(link, "_tokenImage", null);
        recorder = new Recorder(); presenter.ReplayEvent(new LinkPlaced(f.App.Game.GameProgressInformation.CurrentPlayer.Color, "RailBirminghamDudley"), recorder);
        dispatcher.Flush(); UnityCityLocatorRegression.Tick(.2f); yield return null;
        Check(recorder.Count == 1, "missing visual target cannot stall link replay");
        ReflectionAccess.Set(link, "_tokenImage", token);
        router.ThrowBoard = true;
        recorder = new Recorder(); presenter.ReplayEvent(new LinkPlaced(f.App.Game.GameProgressInformation.CurrentPlayer.Color, "RailBirminghamDudley"), recorder);
        dispatcher.Flush(); UnityCityLocatorRegression.Tick(.2f); yield return null;
        Check(recorder.Count == 1, "animation construction failure uses a finite presentation fallback");
        router.ThrowBoard = false; presenter.CancelCurrentReplay(); DOTween.KillAll();
    }
    IEnumerator Cancellation(UnityCityLocatorRegression.Fixture f, TestRouter router, ReplayPresenter presenter, Dispatcher dispatcher, string era)
    {
        var change = new LinkPlaced(f.App.Game.GameProgressInformation.CurrentPlayer.Color, era + "BirminghamDudley");
        var cancelled = new Recorder(); presenter.ReplayEvent(change, cancelled); presenter.CancelCurrentReplay();
        dispatcher.Flush(); UnityCityLocatorRegression.Tick(4f); yield return null;
        Check(cancelled.Count == 0 && DOTween.TweensById("AnimationFactory") == null, "cancel before queued Play cannot revive old replay");
        var old = new Recorder(); var next = new Recorder();
        presenter.ReplayEvent(change, old); presenter.ReplayEvent(change, next); dispatcher.Flush();
        UnityCityLocatorRegression.Tick(4f); yield return null;
        Check(old.Count == 0 && next.Count == 1, "superseded replay cannot advance its old delegate");
        var paused = new Recorder(); presenter.ReplayEvent(change, paused); dispatcher.Flush();
        router.Replay.PauseAllAnimations(); GuardTick(router.Replay, 20f);
        Check(paused.Count == 0 && DOTween.TweensById("AnimationFactory") != null, "explicit replay pause is respected");
        router.Replay.ResumeAllAnimations(); UnityCityLocatorRegression.Tick(4f); yield return null;
        Check(paused.Count == 1, "resume completes paused placement once");
        var pausedCancel = new Recorder(); presenter.ReplayEvent(change, pausedCancel); dispatcher.Flush();
        DOTween.TweensById("AnimationFactory").First().Pause(); presenter.CancelCurrentReplay();
        GuardTick(router.Replay, 20f); yield return null;
        Check(pausedCancel.Count == 0 && DOTween.TweensById("AnimationFactory") == null, "intentional cancel also kills a paused link sequence");
        var lost = new Recorder(); presenter.ReplayEvent(change, lost); dispatcher.Flush();
        DOTween.TweensById("AnimationFactory").First().OnComplete(null);
        UnityCityLocatorRegression.Tick(4f); yield return null; yield return null;
        Check(lost.Count == 1, "missing completion callback recovers once");
        var stopped = new Recorder(); presenter.ReplayEvent(change, stopped); dispatcher.Flush();
        DOTween.TweensById("AnimationFactory").First().Pause(); GuardTick(router.Replay, 4f); yield return null;
        Check(stopped.Count == 1, "stalled active sequence has a bounded presentation allowance");
        var hidden = new Recorder(); presenter.ReplayEvent(change, hidden); dispatcher.Flush();
        router.Replay.DisableView(); UnityCityLocatorRegression.Tick(4f); yield return null;
        Check(hidden.Count == 0, "leaving the replay view cancels without advancing old gameplay");
        router.Replay.CreateShowViewImmediatelyAnimation(); UnityCityLocatorRegression.Tick(.1f);
        var rebound = new Recorder(); presenter.ReplayEvent(change, rebound); dispatcher.Flush();
        UnityCityLocatorRegression.Tick(.4f); f.Update(); yield return null; yield return null;
        Check(rebound.Count == 1, "board refresh cannot strand placement replay");
        var replaced = new Recorder(); presenter.ReplayEvent(change, replaced); dispatcher.Flush();
        Utils.Global.Instance.Router = f.Router;
        DOTween.TweensById("AnimationFactory").First().Kill(); yield return null; yield return null;
        Check(replaced.Count == 0, "replacing the game router invalidates old replay recovery");
        Utils.Global.Instance.Router = router; presenter.CancelCurrentReplay(); DOTween.KillAll();
    }
    IEnumerator Flow(BaseView[] prefabs, int count, string kind, bool interrupt, bool staleBaseline = false)
    {
        using (var f = new UnityCityLocatorRegression.Fixture(count)) {
            f.App.Metadata.GameId = "link-replay-" + count + "-" + kind;
            var g = f.App.Game; var player = g.GameProgressInformation.CurrentPlayer;
            bool rail = kind.EndsWith("rail", StringComparison.Ordinal);
            if (rail) {
                g.ApplyEvent(new RailEraStarted()); g.ApplyEvent(new MoneyLoaned(player.Color, 100));
                g.ApplyEvent(new LinkPlaced(player.Color, "RailBirminghamOxford"));
                while (player.Mat.GetFirstBuildingOfType(BuildingType.Brewery).Level < 2) g.ApplyEvent(new IndustryDeveloped(player.Color, BuildingType.Brewery));
                var slot = g.Board.GetRegionByName(RegionName.Stafford).BuildingSlots.First(s => s.BuildingTypes.Contains(BuildingType.Brewery));
                g.ApplyEvent(new IndustryBuilt(player.Color, BuildingType.Brewery, slot.UniqueId));
            } else if (kind == "canal-later") g.ApplyEvent(new RoundStarted(2, 10, GameEra.CanalEra));
            if (kind != "canal-opening") g.ApplyEvent(new TurnStarted(player.Color));
            f.Update();
            var repository = new BaseRepository(new InMemoryGameStorage(), new InMemorySnapshotStorage(), new BrassBirminghamFactory());
            repository.Save(f.App);
            var history = new BrassGameHistory { Repository = repository }; history.SetGame(f.App);
            var pm = f.App.Metadata.GetPlayerMetadataByColor(player.Color); pm.LastSeenEventRevision = g.Revision;
            var router = new TestRouter { Board = f.View };
            router.Replay = UnityEngine.Object.Instantiate(prefabs.OfType<ReplayView>().First(), f.Root.transform);
            router.Replay.InputRegistration = router; ReflectionAccess.Set(router.Replay, "_router", router); Utils.Global.Instance.Router = router;
            router.Replay.CreateShowViewImmediatelyAnimation(); UnityCityLocatorRegression.Tick(.1f);
            var dispatcher = new Dispatcher(); var presenter = new ReplayPresenter(router, dispatcher, history, f.Localization);
            var output = new NetworkOutput(f, router, presenter);
            output.Replay = new ReplayUseCase(new ReplayOutput(f, presenter), history);
            var action = new NetworkActionUseCase(output, history);
            int money = player.Money, cards = player.Hand.Count, tiles = player.LinkTileCount, actions = g.TurnFlow.GetRemainingActionsCount();
            int expectedMoney = money - (rail ? 5 : 3);
            action.StartNetworkAction();
            if (!staleBaseline) {
                string selecting = State(f.App); action.LinkSelected(g.Board.GetLinkById((rail ? "Rail" : "Canal") + "BirminghamDudley"));
                Check(State(f.App) == selecting && ReflectionAccess.Get<int>(g.TurnFlow.NetworkAction, "_linkCounter") == 0,
                    "N34-PROMPT: no link callback can run before card selection");
            }
            action.SelectCard(player.Hand[0]);
            var pump = Pump(output.Replay, dispatcher, false); while (pump.MoveNext()) yield return pump.Current;
            Check(output.Links != null && output.Links.Any(l => l.LinkId == (rail ? "Rail" : "Canal") + "BirminghamDudley"), "card discard reaches native link selection", new { count, kind, interrupt });
            // Exercise the actual button listener, including a duplicate Unity click
            // already queued before the board refresh removes the old listener.
            var selected = g.Board.GetLinkById((rail ? "Rail" : "Canal") + "BirminghamDudley");
            var linkView = f.View.GetComponentsInChildren<LinkView>(true).First(v => v.name == "BirminghamDudley");
            var button = ReflectionAccess.Get<Button>(linkView, "_tokenButton");
            if (!staleBaseline) {
                string ready = State(f.App);
                ReflectionAccess.Set(history, "_movedBackInHistory", true); action.LinkSelected(selected);
                ReflectionAccess.Set(history, "_movedBackInHistory", false);
                Check(State(f.App) == ready && !selected.HasOwner, "N34-PROMPT: history browsing cannot consume a live placement");
                action.LinkSelected(null);
                Check(State(f.App) == ready, "N34-PROMPT: null selection is harmless");
            }
            string beforeMat = State(f.App);
            var mat = UnityEngine.Object.Instantiate(prefabs.First(v => v.GetType().Name == "IndustriesDetailsView"), f.Root.transform);
            mat.InputRegistration = router;
            mat.CreateShowViewImmediatelyAnimation(); UnityCityLocatorRegression.Tick(.05f); mat.DisableView();
            Check(State(f.App) == beforeMat, "N34-MAT: opening and closing industry mat preserves the live Network prompt");
            button.onClick.Invoke();
            Check(selected.HasOwner, "N34-CLICK: original button still places the first link", new { count, kind, interrupt });
            string afterFirst = State(f.App); Exception duplicate = null;
            try { button.onClick.Invoke(); } catch (Exception e) { duplicate = e; }
            int counter = ReflectionAccess.Get<int>(g.TurnFlow.NetworkAction, "_linkCounter");
            if (staleBaseline) {
                Check(duplicate is NullReferenceException && counter == 1 && State(f.App) == afterFirst,
                    "N34-BASELINE: completed Canal button poisons the counter before null-player failure", new { count, exception = duplicate == null ? null : duplicate.ToString(), counter });
            } else {
                Check(duplicate == null && State(f.App) == afterFirst && counter == (rail ? 1 : 0),
                    "N34-CLICK: duplicate listener cannot mutate action, payment, money, cards or link count", new { count, kind, interrupt, counter });
                action.LinkSelected(g.Board.GetLinkById((rail ? "Rail" : "Canal") + "BirminghamWalsall"));
                Check(State(f.App) == afterFirst && ReflectionAccess.Get<int>(g.TurnFlow.NetworkAction, "_linkCounter") == counter,
                    "N34-PENDING: queued other-link callback cannot bypass completion or pending coal");
            }
            pump = Pump(output.Replay, dispatcher, interrupt); while (pump.MoveNext()) yield return pump.Current;
            int links = 1;
            if (rail) {
                Check(output.CoalChoices == 1 && output.ActionsShown == 0 && output.EndShown == 0 && g.TurnFlow.CurrentAction != null,
                    "interrupted rail still requires its first coal", new { count, kind, interrupt });
                expectedMoney -= g.Board.CoalMarket.CoalPrice(1); action.SelectCoalSource(g.Board.CoalMarket, 1);
                pump = Pump(output.Replay, dispatcher, false); while (pump.MoveNext()) yield return pump.Current;
                Check(output.AnotherChoices == 1 && output.ActionsShown == 0 && output.EndShown == 0, "coal payment reaches the second-track decision");
                if (kind == "double-rail") {
                    action.SelectSecondLink(); action.LinkSelected(g.Board.GetLinkById("RailBirminghamWalsall")); links++; expectedMoney -= 10;
                    pump = Pump(output.Replay, dispatcher, interrupt); while (pump.MoveNext()) yield return pump.Current;
                    Check(output.CoalChoices == 2 && output.BeerChoices == 0, "second rail still requires coal before beer");
                    expectedMoney -= g.Board.CoalMarket.CoalPrice(1); action.SelectCoalSource(g.Board.CoalMarket, 1);
                    pump = Pump(output.Replay, dispatcher, false); while (pump.MoveNext()) yield return pump.Current;
                    Check(output.BeerChoices == 1 && output.ActionsShown == 0 && output.EndShown == 0, "double rail still requires brewery beer");
                    action.SelectBeerSource(g.Board.GetUnflippedBreweries().First(), 1);
                } else action.EndNetworkAction();
                pump = Pump(output.Replay, dispatcher, false); while (pump.MoveNext()) yield return pump.Current;
            }
            Check(!output.Replay.GetIsReplayInProgress() && g.TurnFlow.CurrentAction == null, "placement replay fully releases the action", new { count, kind, interrupt });
            Check(output.ActionsShown == (actions == 2 ? 1 : 0) && output.EndShown == (actions == 1 ? 1 : 0), "correct action controls or End turn return exactly once");
            Check(g.TurnFlow.GetRemainingActionsCount() == actions - 1 && player.Hand.Count == cards - 1 && player.LinkTileCount == tiles - links,
                "recovery never duplicates action, card or link consumption");
            Check(player.Money == expectedMoney, "link and resource costs are paid exactly once");
            if (kind == "canal-opening") {
                string finished = State(f.App);
                if (!staleBaseline) {
                    action.LinkSelected(selected);
                    Exception unprepared = null;
                    try { g.TurnFlow.NetworkAction.PlaceLink(g.Board.GetLinkById("CanalBirminghamWalsall")); } catch (Exception e) { unprepared = e; }
                    Check(unprepared is InvalidOperationException && State(f.App) == finished && ReflectionAccess.Get<int>(g.TurnFlow.NetworkAction, "_linkCounter") == 0,
                        "N34-DOMAIN: unprepared direct placement is rejected before any mutation");
                }
                g.GameFlow.EndTurn();
                Check(g.GameProgressInformation.CurrentPlayer != player, "N34-HANDOFF: End Turn advances to the next actor");
                if (!staleBaseline) {
                    string nextTurn = State(f.App); action.LinkSelected(g.Board.GetLinkById("CanalBirminghamWalsall"));
                    Check(State(f.App) == nextTurn && ReflectionAccess.Get<int>(g.TurnFlow.NetworkAction, "_linkCounter") == 0,
                        "N34-HANDOFF: old human callback is harmless during the next player's turn");
                }
                var live = BrassHost.Host.Capture(f.App);
                var choice = live.Choices().First(c => c.kind == "network");
                var path = new[] { choice.id, new BrassHost.Choice("finish").id };
                var composition = BrassHost.Host.Composition(BrassHost.Host.Configuration(g.GameStartConfiguration));
                live.ValidateLiveAction(path, composition);
                Exception commit = null;
                try { live.LiveCommit(path, composition); } catch (Exception e) { commit = e; }
                Check(staleBaseline ? commit is InvalidOperationException && commit.Message.Contains("one link per action") : commit == null && g.TurnFlow.CurrentAction == null,
                    staleBaseline ? "N34-BASELINE: next AI live commit reproduces the user's exact Canal error" : "N34-HANDOFF: next AI validated Network action commits successfully",
                    new { count, error = commit == null ? null : commit.ToString() });
            }
            presenter.CancelCurrentReplay(); DOTween.KillAll();
        }
    }
    IEnumerator CancelledPrompt(BaseView[] prefabs, int count)
    {
        using (var f = new UnityCityLocatorRegression.Fixture(count)) {
            f.App.Metadata.GameId = "network-cancel-" + count;
            var repository = new BaseRepository(new InMemoryGameStorage(), new InMemorySnapshotStorage(), new BrassBirminghamFactory());
            repository.Save(f.App);
            var history = new BrassGameHistory { Repository = repository }; history.SetGame(f.App);
            var router = new TestRouter { Board = f.View };
            router.Replay = UnityEngine.Object.Instantiate(prefabs.OfType<ReplayView>().First(), f.Root.transform);
            router.Replay.InputRegistration = router; ReflectionAccess.Set(router.Replay, "_router", router); Utils.Global.Instance.Router = router;
            var dispatcher = new Dispatcher(); var presenter = new ReplayPresenter(router, dispatcher, history, f.Localization);
            var output = new NetworkOutput(f, router, presenter);
            output.Replay = new ReplayUseCase(new ReplayOutput(f, presenter), history);
            var action = new NetworkActionUseCase(output, history); var g = f.App.Game;
            var player = g.GameProgressInformation.CurrentPlayer;
            f.App.Metadata.GetPlayerMetadataByColor(player.Color).LastSeenEventRevision = g.Revision;
            action.StartNetworkAction(); action.SelectCard(player.Hand[0]);
            var pump = Pump(output.Replay, dispatcher, false); while (pump.MoveNext()) yield return pump.Current;
            var link = output.Links.First(); action.CancelAction();
            string current = State(history.CurrentGame); string abandoned = State(f.App);
            action.LinkSelected(link);
            Check(State(history.CurrentGame) == current && State(f.App) == abandoned && !link.HasOwner && ReflectionAccess.Get<int>(g.TurnFlow.NetworkAction, "_linkCounter") == 0,
                "N34-CANCEL: cancelled or replaced game cannot receive an obsolete placement", new { count });
            presenter.CancelCurrentReplay(); DOTween.KillAll();
        }
    }
    IEnumerator Pump(ReplayUseCase replay, Dispatcher dispatcher, bool interrupt)
    {
        bool killed = false;
        for (int i = 0; i < 150 && replay.GetIsReplayInProgress(); i++) {
            dispatcher.Flush(); UnityCityLocatorRegression.Tick(.4f);
            var active = DOTween.TweensById("AnimationFactory");
            if (interrupt && !killed && active != null) { active.First().Kill(); killed = true; }
            yield return null;
        }
        Check(!replay.GetIsReplayInProgress(), "native action replay finishes within the bounded test pump");
        if (interrupt) Check(killed, "fault injected after the link becomes visible");
    }
    sealed class ReplayOutput : IReplayOutput
    {
        readonly UnityCityLocatorRegression.Fixture f; readonly ReplayPresenter presenter;
        public ReplayOutput(UnityCityLocatorRegression.Fixture f, ReplayPresenter presenter) { this.f = f; this.presenter = presenter; }
        public void ReplayEvent(IEvent e, IEventReplayDelegate d) { if (e is LinkPlaced) presenter.ReplayEvent(e, d); else d.EventReplayCompleted(e); }
        public void UpdateGamePresentation(PlayerMetadata p) { f.Update(); }
        public void UpdateCurrentPlayer(string p) { }
        public void UpdateCurrentPlayerAction(IEvent e) { }
        public void UpdateCurrentPlayerActionText(string t) { }
        public void UpdateCurrentPlayerActionsView() { }
        public void CancelCurrentReplay() { presenter.CancelCurrentReplay(); }
        public void FinishReplay() { presenter.FinishReplay(); }
        public void ShowReplaySummary(IList<IEvent> e, Action dismiss) { dismiss(); }
        public void ShowIndustrySellingSummary(IEvent e, Action dismiss) { dismiss(); }
    }
    sealed class NetworkOutput : INetworkActionOutput
    {
        readonly UnityCityLocatorRegression.Fixture f; readonly TestRouter router; readonly ReplayPresenter presenter;
        public ReplayUseCase Replay; public List<Link> Links; public int ActionsShown, EndShown, CoalChoices, BeerChoices, AnotherChoices;
        public NetworkOutput(UnityCityLocatorRegression.Fixture f, TestRouter router, ReplayPresenter presenter) { this.f = f; this.router = router; this.presenter = presenter; }
        public void PresentEvents(ReadOnlyCollection<IEvent> e, PlayerMetadata p, IReplayDelegate d) { router.Replay.CreateShowViewImmediatelyAnimation(); Replay.PresentEvents(e, p, d); }
        public void HighlightValidConnections(BrassApplicationGame g, List<Link> links, ILinkSelectedDelegate d) {
            Links = links; f.Update(); foreach (var m in f.Model.Links.Values) { m.IsOutlineVisible = links.Contains(m.Link); m.IsInteractive = links.Contains(m.Link); } f.View.HighlightLinks(f.Model.Links, d);
        }
        public void ShowPlayerActionsButtons() { ActionsShown++; }
        public void ShowEndTurnButton() { EndShown++; }
        public void ShowAnotherTrackSelectionPanel(List<CantUseLinkActionReason> r) { AnotherChoices++; }
        public void StartCoalDeliveryUseCase(int n, IEnumerable<CoalMine> s, ICoalDeliveryResultDelegate d) { CoalChoices++; }
        public void StartBeerDeliveryUseCase(int n, IEnumerable<Brewery> s, IBeerDeliveryResultDelegate d) { BeerChoices++; }
        public void HideNumberOfTracksToBuildSelectionPanel() { }
        public void ShowCardsToDiscard(ISelectCardDelegate d) { }
        public void HideCards() { }
        public void HideCoalDelivery() { }
        public void HideBeerDelivery() { }
        public void ShowActionInProgressView(IActionInProgressCancelDelegate d, int a, int b) { }
        public void ShowPlayerInfoToSelectCardToDiscard() { }
        public void ShowPlayerInfoToSelectLinks() { }
        public void ShowPlayerInfoToSelectIfAnotherTrackShouldBePlaced() { }
        public void AllowRegionInteractions(bool b) { }
        public void ReplayEvents(ReadOnlyCollection<IEvent> e) { }
        public void SetFastForwardButtonActive(bool b) { }
        public void HideActionInProgressView() { }
        public void UpdateGameStatus(PlayerMetadata m) { }
        public void RestoreCardsView() { }
        public void HideReplayView() { router.Replay.DisableView(); }
        public Task WaitForTweensEnd() { return Task.FromResult(0); }
        public void HideIronDelivery() { }
    }
    sealed class Recorder : IEventReplayDelegate
    { public int Count; public void EventReplayCompleted(IEvent e) { Count++; } }
    sealed class Dispatcher : IMainThreadDispatcher
    {
        readonly Queue<Action> actions = new Queue<Action>();
        public int Count { get { lock (actions) return actions.Count; } }
        public bool isMainThread { get { return true; } }
        public void InvokeAsync(Action action) { lock (actions) actions.Enqueue(action); }
        public void Invoke(Action action) { InvokeAsync(action); }
        public void Flush() { lock (actions) while (actions.Count != 0) actions.Dequeue()(); }
    }
    sealed class TestRouter : IRouter
    {
        public BoardView Board; public ReplayView Replay;
        public bool ThrowBoard;
        public IUseCaseBuilder UseCaseBuilder { get; set; }
        public T GetView<T>() where T : class, IBaseView { if (ThrowBoard && typeof(T) == typeof(IBoardView)) throw new MissingReferenceException("injected absent board"); return typeof(T) == typeof(IReplayView) ? Replay as T : Board as T; }
        public bool IsViewCached<T>() where T : class, IBaseView { return GetView<T>() != null; }
        public bool IsViewVisible<T>() where T : class, IBaseView { var v = GetView<T>() as BaseView; return v != null && v.IsViewVisible(); }
        public IAnimation ShowView<T>() where T : class, IBaseView { return GetView<T>().CreateShowViewAnimation(); }
        public IAnimation HideView<T>() where T : class, IBaseView { return GetView<T>().CreateHideViewAnimation(); }
        public void MoveOnTop<T>() where T : class, IBaseView { }
        public void ProcessInput() { }
        public void RegisterKey(KeyEnum key, IBaseView view) { }
        public void UnregisterKey(KeyEnum key, IBaseView view) { }
        public void UnregisterAllKeys(IBaseView view) { }
    }
}
