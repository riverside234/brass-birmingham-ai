using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Threading.Tasks;
using BoardGameRules.Entities.Cards;
using BoardGameRules.Entities.Common;
using BoardGameRules.Entities.EventSourcing.Events;
using BoardGameRules.Entities.Game;
using BrassApplication.AI;
using BrassApplication.Game;
using BrassApplication.GameHistory;
using BrassApplication.UseCases.Common;
using BrassApplication.UseCases.Game.Actions;
using BrassApplication.UseCases.Game.Actions.ActionInProgress;
using BrassApplication.UseCases.Game.Actions.BaseAction;
using BrassApplication.UseCases.Game.Actions.PassAction;
using BrassApplication.UseCases.Game.Actions.ScoutAction;
using BrassApplication.UseCases.Game.HandOfCards;
using BrassApplication.UseCases.Game.Replay;
using DG.Tweening;
using Frameworks.Persistence.GameFactory;
using Frameworks.Persistence.Repository;
using Frameworks.Persistence.RepositoryBackends.InMemory;
using Frameworks.Routing;
using Frameworks.Routing.Unity;
using Frameworks.View.Common;
using Frameworks.View.Game;
using Frameworks.View.Game.Actions;
using InterfaceAdapters.Common;
using InterfaceAdapters.Game.ActionInProgressView;
using InterfaceAdapters.Game.Actions.ScoutAction;
using InterfaceAdapters.Game.HandOfCards;
using InterfaceAdapters.Game.PlayerActions;
using InterfaceAdapters.Routing;
using InterfaceAdapters.Utils;
using UnityEngine;
using UnityEngine.UI;
using Utils;

// The installed opt-in self-test hook loads this name from its artifact directory.
// The review runner backs up/restores that artifact and the original self-test output.
namespace BrassPatch.Tests.Liquidation
{
    public static class UnityPresentationRegression
    {
        private static MethodInfo TweenUpdate = typeof(DOTween).Assembly.GetType("DG.Tweening.Core.TweenManager").GetMethod("Update", BindingFlags.Static | BindingFlags.NonPublic);
        private static List<string> errors = new List<string>();
        private static void Capture(string text, string stack, LogType type) { if (type == LogType.Error || type == LogType.Exception) errors.Add(type + ": " + text); }
        private sealed class Case
        {
            public BrassGameHistory History; public ScoutActionUseCase Scout; public PassActionUseCase Pass;
            public HandOfCardsUseCase Hand; public ActionInProgressUseCase Progress;
            public Router Router; public Factory Factory; public Proxy Output; public Builder Builder;
        }
        private static Case Create(UnityViewFactory factory)
        {
            var config = new GameStartConfiguration(NumberOfPlayers.Two); config.Seed = 20260927;
            var metadata = new GameMetadata(config, GameType.Local);
            foreach (var p in config.Players) metadata.JoinGame(new PlayerMetadata { PlayerStartConfiguration = p, NickName = "Probe " + p.Color, PlayerNetworkId = "probe-" + p.Color, AiType = AIType.Human });
            var repository = new BaseRepository(new InMemoryGameStorage(), new InMemorySnapshotStorage(), new BrassBirminghamFactory());
            var app = repository.Create(metadata); app.Game.StartGame(); repository.Save(app);
            var c = new Case(); c.History = new BrassGameHistory { Repository = repository }; c.History.SetGame(app);
            c.Factory = new Factory(factory); c.Router = new Router(c.Factory, new NoInput()); Global.Instance.Router = c.Router;
            c.Builder = new Builder(c); c.Router.UseCaseBuilder = c.Builder;
            c.Progress = new ActionInProgressUseCase(new ProgressOutput());
            c.Hand = new HandOfCardsUseCase(new HandOfCardsPresenter(c.Router, new Localization()), c.History);
            c.Output = new Proxy(c); c.Scout = new ScoutActionUseCase(c.Output, c.History); c.Pass = new PassActionUseCase(c.Output, c.History);
            c.Router.ShowView<IHandOfCardsView>(); c.Router.ShowView<IPlayerActionsView>();
            c.Hand.UpdateCardsForPlayer(CurrentMetadata(c)); Tick(c, 1f);
            return c;
        }
        private static PlayerMetadata CurrentMetadata(Case c) { return c.History.CurrentGame.Metadata.GetPlayerMetadataByColor(c.History.CurrentGame.Game.GameProgressInformation.CurrentPlayer.Color); }
        private static void Tick(Case c, float seconds)
        {
            int frames = (int)Math.Ceiling(seconds * 30f);
            for (int i = 0; i < frames; i++)
            {
                TweenUpdate.Invoke(null, new object[] { UpdateType.Normal, 1f / 30f, 1f / 30f });
                c.Factory.EndFrame();
            }
        }
        private static BaseCardView Card(Case c, BaseCard card)
        {
            var info = c.Router.GetView<IHandOfCardsView>().GetCardObjectInfo(card);
            if (info == null) throw new InvalidOperationException("The native card object is missing for " + card.UniqueId);
            return ((GameObject)info.Item1).GetComponent<BaseCardView>();
        }
        private static object Snapshot(Case c)
        {
            var g = c.History.CurrentGame.Game; var p = g.GameProgressInformation.CurrentPlayer;
            return new { revision = g.Revision, remainingActions = g.TurnFlow.GetRemainingActionsCount(), hand = p.Hand.Count,
                wild = p.Hand.Count(x => x is WildIndustryCard || x is WildLocationCard), discard = p.DiscardPile.Count,
                distinctDiscard = p.DiscardPile.Select(x => x.UniqueId).Distinct().Count(), events = g.GetUncommittedEvents().Select(e => e.GetType().Name).ToArray(),
                eventRevisions = g.GetUncommittedEvents().Select(e => e.Metadata.Revision).ToArray() };
        }
        public static int Run()
        {
            string output = Path.Combine(Path.GetDirectoryName(Application.dataPath), "artifacts", "reference-review", "phases-native-results.json");
            var results = new List<object>(); IRouter previous = Global.Instance.Router;
            var factory = GameObject.Find("Utils/UnityViewFactory").GetComponent<UnityViewFactory>();
            DOTween.Init(); Application.logMessageReceived += Capture;
            try
            {
                var resources = Resources.LoadAll<UnityEngine.Object>("");
                var allViews = Resources.FindObjectsOfTypeAll<BaseView>();
                File.WriteAllText(Path.Combine(Path.GetDirectoryName(output), "phases-native-resource-catalog.json"), Newtonsoft.Json.JsonConvert.SerializeObject(new {
                    resources = resources.Select(x => new { name = x.name, type = x.GetType().FullName }).ToArray(),
                    views = allViews.Select(x => new { name = x.name, type = x.GetType().FullName }).ToArray()
                }, Newtonsoft.Json.Formatting.Indented));
                var originalViews = typeof(BaseViewList).GetField("GameViews", BindingFlags.Instance | BindingFlags.NonPublic);
                originalViews.SetValue(factory.ViewList, allViews);
                foreach (string mode in new[] { "normal", "duplicateBeforeTweenUpdate", "cancelAt0.20", "cancelAt2.20", "cancelAt0.20ThenPass" })
                {
                    DOTween.KillAll(); errors.Clear(); Case c = Create(factory);
                    try
                    {
                        c.Scout.StartScoutAction(); Tick(c, 1.5f);
                        Card(c, c.History.CurrentGame.Game.GameProgressInformation.CurrentPlayer.Hand[0]).OnMouseClicked();
                        Tick(c, 3f);
                        var player = c.History.CurrentGame.Game.GameProgressInformation.CurrentPlayer;
                        BaseCard one = player.Hand[0], two = mode == "duplicateBeforeTweenUpdate" ? one : player.Hand[1];
                        BaseCardView first = Card(c, one);
                        first.OnMouseClicked();
                        bool firstLockedImmediately = first.ignoreMouseOver;
                        Card(c, two).OnMouseClicked();
                        var before = Snapshot(c);
                        bool cancelActive = false, cancelInteractable = false, cancelRaycast = false;
                        object afterCancel = null; float passClickedAt = -1;
                        if (mode.StartsWith("cancel"))
                        {
                            float cutoff = mode.StartsWith("cancelAt2") ? 2.2f : 0.2f; Tick(c, cutoff);
                            var progress = (ActionInProgressView)c.Router.GetView<IActionInProgressView>();
                            Transform cancel = progress.transform.Find("SafeArea/Popup/CancelButton");
                            cancelActive = cancel.gameObject.activeInHierarchy; cancelInteractable = cancel.GetComponent<Button>().interactable;
                            cancelRaycast = cancel.GetComponentsInParent<CanvasGroup>().All(x => x.blocksRaycasts && x.interactable);
                            progress.OnCancelClicked(); afterCancel = Snapshot(c);
                            if (mode.EndsWith("ThenPass"))
                            {
                                // Wait for the ordinary native HideCards animation to unlock the
                                // restored hand, then start Pass and wait for its ShowCards animation.
                                bool passStarted = false;
                                for (int frame = 0; frame < 120; frame++)
                                {
                                    Tick(c, 1f / 30f);
                                    var current = c.History.CurrentGame.Game.GameProgressInformation.CurrentPlayer;
                                    BaseCardView card = Card(c, current.Hand[0]);
                                    bool clickable = !card.ignoreMouseOver && card.GetComponent<CanvasGroup>().interactable && card.GetComponent<CanvasGroup>().blocksRaycasts;
                                    if (!passStarted && clickable && c.History.CurrentGame.Game.TurnFlow.GetRemainingActionsCount() > 0)
                                    {
                                        c.Pass.StartPassAction(); passStarted = true; continue;
                                    }
                                    if (passStarted && clickable && c.History.CurrentGame.Game.TurnFlow.CurrentAction != null && passClickedAt < 0)
                                    { card.OnMouseClicked(); passClickedAt = cutoff + (frame + 1) / 30f; }
                                }
                            }
                            else Tick(c, 4f);
                        }
                        else Tick(c, 4f);
                        results.Add(new { mode, firstLockedImmediately, before, cancelActive, cancelInteractable, cancelRaycast, afterCancel,
                            passClickedAt, afterAnimations = Snapshot(c), nativeErrors = errors.ToArray(), callbackCount = c.Output.CallbackCount });
                    }
                    catch (Exception error) { results.Add(new { mode, harnessError = error.ToString(), nativeErrors = errors.ToArray() }); }
                    finally { DOTween.KillAll(); c.Factory.Clear(); }
                }
                File.WriteAllText(output, Newtonsoft.Json.JsonConvert.SerializeObject(new { unity = Application.unityVersion, runtime = Environment.Version.ToString(), cases = results }, Newtonsoft.Json.Formatting.Indented));
                return results.Count;
            }
            finally { Application.logMessageReceived -= Capture; Global.Instance.Router = previous; }
        }
        private sealed class Proxy : IScoutActionOutput, IPassActionOutput
        {
            private readonly Case c; private readonly ScoutActionPresenter native;
            public int CallbackCount;
            public Proxy(Case c) { this.c = c; native = new ScoutActionPresenter(c.Router, new Localization()); }
            public void ShowPlayerInfoToSelectCardToDiscard(IActionInProgressCancelDelegate d, int n, int max) { native.ShowPlayerInfoToSelectCardToDiscard(d, n, max); }
            public void ShowCardsToDiscard(ISelectCardDelegate d) { native.ShowCardsToDiscard(d); }
            public void HideCards() { native.HideCards(); }
            public void ShowExchangeForWildCardsView(BrassApplicationGame game, ISelectCardDelegate d) { native.ShowExchangeForWildCardsView(game, d); }
            public void SetupCardInSlot(int index, BaseCard card, Action done) { native.SetupCardInSlot(index, card, delegate { if (index == 1) CallbackCount++; done(); }); }
            public void HideExchangeForWildCardsView() { native.HideExchangeForWildCardsView(); }
            public void EnableCardsLayoutGroup() { native.EnableCardsLayoutGroup(); }
            public void ReplayEvents(ReadOnlyCollection<IEvent> events) { }
            public void HideActionInProgressView() { native.HideActionInProgressView(); }
            public void RestoreCardsView() { native.RestoreCardsView(); }
            public void UpdateGameStatus(PlayerMetadata p) { c.Hand.UpdateCardsForPlayer(p); }
            public void ShowPlayerActionsButtons() { c.Router.ShowView<IPlayerActionsView>(); }
            public void ShowEndTurnButton() { c.Router.GetView<IPlayerActionsView>().ShowEndTurnButton(); c.Router.ShowView<IPlayerActionsView>(); }
            public void HideReplayView() { }
            public void HideIronDelivery() { }
            public Task WaitForTweensEnd() { return Task.FromResult(true); }
            public void PresentEvents(ReadOnlyCollection<IEvent> events, PlayerMetadata p, IReplayDelegate d)
            {
                // The rules command already applied its events. Establish the same post-replay
                // native hand before selecting trade cards; board/replay visuals are outside scope.
                if (c.History.CurrentGame.Game.TurnFlow.CurrentAction is BoardGameRules.Entities.Actions.ScoutAction && events.Count == 1)
                    c.Hand.UpdateCardsForPlayer(p);
                d.ReplayCompleted();
            }
        }
        private sealed class Builder : IUseCaseBuilder
        {
            private readonly Case c; public Builder(Case c) { this.c = c; }
            public T GetUseCase<T>() where T : IUseCase
            {
                if (typeof(T) == typeof(HandOfCardsUseCase)) return (T)(object)c.Hand;
                if (typeof(T) == typeof(ActionInProgressUseCase)) return (T)(object)c.Progress;
                throw new NotSupportedException(typeof(T).FullName);
            }
            public IController GetController(Type type)
            {
                if (type == typeof(IHandOfCardsView)) return new HandOfCardsController(c.Hand);
                if (type == typeof(IActionInProgressView)) return new ActionInProgressController(c.Progress);
                if (type == typeof(IScoutActionView)) return new ScoutActionController(c.Scout);
                return null;
            }
            public void BuildUseCases() { } public void RestoreUseCases() { } public void RestoreUseCaseOfType(Type type) { }
        }
        private sealed class Factory : IViewFactory
        {
            private readonly UnityViewFactory original; private readonly List<GameObject> all = new List<GameObject>(); private readonly List<GameObject> pending = new List<GameObject>();
            public Factory(UnityViewFactory original) { this.original = original; }
            public T CreateView<T>() where T : class, IBaseView { T view = original.CreateView<T>(); all.Add(((BaseView)(object)view).gameObject); return view; }
            public void DestroyView(IBaseView view) { pending.Add(((BaseView)view).gameObject); original.DestroyView(view); }
            // The startup hook runs synchronously, so flush queued destroys after each simulated
            // tween frame to represent Unity's ordinary end-of-frame destruction.
            public void EndFrame() { foreach (GameObject item in pending) if (item != null) UnityEngine.Object.DestroyImmediate(item); pending.Clear(); }
            public void Clear() { foreach (GameObject item in all) if (item != null) UnityEngine.Object.DestroyImmediate(item); all.Clear(); pending.Clear(); }
        }
        private sealed class NoInput : IInputSource { public bool GetKeyDown(KeyEnum key) { return false; } public bool GetKey(KeyEnum key) { return false; } public bool GetKeyUp(KeyEnum key) { return false; } }
        private sealed class Localization : ILocalization { public string GetTranslation(string text) { return text; } }
        private sealed class ProgressOutput : IActionInProgressOutput { }
    }
}
