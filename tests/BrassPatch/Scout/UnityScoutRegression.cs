using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using BoardGameRules.Entities.Board.BuildingSlots;
using BoardGameRules.Entities.Cards;
using BoardGameRules.Entities.Common;
using DG.Tweening;
using Frameworks.View.Game;
using UnityEngine;
using UnityEngine.UI;

namespace BrassPatch.Tests.Scout
{
    /// <summary>
    /// Native Unity regression, run synchronously on the main thread with FrameworksInit
    /// skipped. Uses the shipped ScoutActionView, IndustryCardView, wild-card components,
    /// and DOTween normal update loop on a minimal in-memory hierarchy. No stores, network
    /// clients, gameplay scene, fabricated production card subclass, or global KillAll.
    /// Serialized gameplay prefab layout/hit testing remains outside this suite.
    /// </summary>
    public static class UnityScoutRegression
    {
        private static int checks;
        private static readonly List<string> nativeProblems = new List<string>();
        private static readonly MethodInfo TweenUpdate = typeof(DOTween).Assembly
            .GetType("DG.Tweening.Core.TweenManager")
            .GetMethod("Update", BindingFlags.Static | BindingFlags.NonPublic);
        private static readonly MethodInfo ShowCompleted = typeof(ScoutActionView)
            .GetMethod("ShowViewCompleted", BindingFlags.Instance | BindingFlags.NonPublic);

        private static void Check(bool value, string description)
        {
            checks++;
            if (!value) throw new InvalidOperationException("Unity Scout regression: " + description);
        }

        private static void Capture(string message, string stack, LogType type)
        {
            if (type == LogType.Warning || type == LogType.Error || type == LogType.Exception)
                nativeProblems.Add(type + ": " + message);
        }

        private static void NoNativeProblems(string scenario)
        {
            Check(nativeProblems.Count == 0, scenario + " has no native warnings/errors: " + string.Join(" | ", nativeProblems.ToArray()));
        }

        private static void Tick(float seconds)
        {
            // Advancing the real normal-update iterator avoids Complete(true)'s special
            // reentrant-despawn behavior and exercises the callbacks used in actual frames.
            for (int frame = 0; frame < (int)Math.Round(seconds * 30f); frame++)
                TweenUpdate.Invoke(null, new object[] { UpdateType.Normal, 1f / 30f, 1f / 30f });
        }

        private static void Stop(ScoutActionView view, bool restoreCards)
        {
            Type fixes = typeof(BrassPatch.StartupFixes).Assembly.GetType("BrassPatch.Scout.ScoutViewFixes", true);
            fixes.GetMethod("StopAnimations", BindingFlags.Static | BindingFlags.Public)
                .Invoke(null, new object[] { view, restoreCards });
        }

        private static void PrepareForShow(ScoutActionView view)
        {
            Type fixes = typeof(BrassPatch.StartupFixes).Assembly.GetType("BrassPatch.Scout.ScoutViewFixes", true);
            fixes.GetMethod("PrepareForShow", BindingFlags.Static | BindingFlags.Public)
                .Invoke(null, new object[] { view });
        }

        private static GameObject Node(string name, Transform parent)
        {
            GameObject result = new GameObject(name, typeof(RectTransform));
            result.transform.SetParent(parent, false);
            return result;
        }

        private static T Card<T>(string name, Transform parent, BaseCard data) where T : BaseCardView
        {
            GameObject node = Node(name, parent);
            GameObject holder = Node("Holder", node.transform);
            Node("CardCover", holder.transform);
            Node("Banner", holder.transform).AddComponent<Image>();
            Node("ImageBackground", holder.transform).AddComponent<Image>();
            T card = node.AddComponent<T>();
            card.BaseCard = data;
            return card;
        }

        private sealed class Fixture : IDisposable
        {
            public readonly GameObject Root;
            public readonly GameObject ScoutObject;
            public readonly ScoutActionView View;
            public readonly Transform Hand;
            public readonly HorizontalLayoutGroup Layout;
            public readonly IndustryCardView First, Second;
            public readonly WildIndustryCardView WildIndustry;
            public readonly WildLocationCardView WildLocation;
            public readonly Transform Sentinel;
            public readonly Tween SentinelTween;
            public readonly Vector3 FirstPosition, SecondPosition;
            public int Accepted, FirstCallbacks, Completions;
            public Action Completion;

            public Fixture()
            {
                Root = Node("ScoutNativeRegression", null);
                Hand = Node("Hand", Root.transform).transform;
                Layout = Hand.gameObject.AddComponent<HorizontalLayoutGroup>();
                First = Card<IndustryCardView>("FirstTradeCard", Hand,
                    new IndustryCard("native-scout-first", NumberOfPlayers.Two, new List<BuildingType> { BuildingType.CoalMine }));
                Second = Card<IndustryCardView>("SecondTradeCard", Hand,
                    new IndustryCard("native-scout-second", NumberOfPlayers.Two, new List<BuildingType> { BuildingType.IronWorks }));
                First.transform.localPosition = new Vector3(-80f, 0f, 0f);
                Second.transform.localPosition = new Vector3(80f, 0f, 0f);
                FirstPosition = First.transform.position;
                SecondPosition = Second.transform.position;
                ScoutObject = Node("Scout", Root.transform);
                Transform safe = Node("SafeArea", ScoutObject.transform).transform;
                GameObject popup = Node("Popup", safe);
                popup.AddComponent<CanvasGroup>();
                Transform panel = Node("Panel", popup.transform).transform;
                Transform slot0 = Node("Slot1", panel).transform;
                Node("Placeholder0", slot0); Node("Placeholder1", slot0);
                Transform slot1 = Node("Slot2", panel).transform;
                Node("Placeholder0", slot1); Node("Placeholder1", slot1);
                Transform industrySlot = Node("WildIndustrySlot", panel).transform;
                Transform locationSlot = Node("WildLocationSlot", panel).transform;
                WildIndustry = Card<WildIndustryCardView>("WildIndustry", industrySlot,
                    new WildIndustryCard("native-scout-wild-industry", NumberOfPlayers.Two));
                WildLocation = Card<WildLocationCardView>("WildLocation", locationSlot,
                    new WildLocationCard("native-scout-wild-location", NumberOfPlayers.Two));
                View = ScoutObject.AddComponent<ScoutActionView>();
                PrepareForShow(View);
                First.OnMouseClickedAction = delegate { Select(First); };
                Second.OnMouseClickedAction = delegate { Select(Second); };
                Sentinel = Node("UnrelatedTween", Root.transform).transform;
                SentinelTween = Sentinel.DOLocalMove(new Vector3(30f, 0f, 0f), 6f).SetEase(Ease.Linear).SetAutoKill(false);
            }

            private void Select(BaseCardView card)
            {
                int slot = Accepted++;
                // The presenter's normal GetCardObjectInfo observes the CURRENT sibling
                // index, including when an earlier card has already left the hand.
                View.AnimateCardToSlot(slot, Tuple.Create((object)card.gameObject, card.transform.GetSiblingIndex()), delegate
                {
                    if (slot == 0) FirstCallbacks++;
                    else { Completions++; if (Completion != null) Completion(); }
                });
                if (slot == 1)
                {
                    // Match ScoutActionPresenter.SetupCardInSlot's hand-wide interaction
                    // lock after its second selected card has been handed to the view.
                    First.ignoreMouseOver = true; Second.ignoreMouseOver = true;
                }
            }

            public void SelectBoth(float betweenSelections)
            {
                First.OnMouseClicked();
                if (betweenSelections > 0f) Tick(betweenSelections);
                Second.OnMouseClicked();
                Check(Accepted == 2, "two distinct native cards are accepted");
            }

            public void CheckRestored(string description)
            {
                Check(Layout.enabled, description + " restores hand layout");
                Check(First != null && Second != null && First.transform.parent == Hand && Second.transform.parent == Hand,
                    description + " restores both selected native cards to the hand");
                Check(First.transform.GetSiblingIndex() == 0 && Second.transform.GetSiblingIndex() == 1,
                    description + " restores original hand ordering after index shifts");
                Check(Vector3.Distance(First.transform.position, FirstPosition) < 0.01f && Vector3.Distance(Second.transform.position, SecondPosition) < 0.01f,
                    description + " restores original card positions");
                Check(!First.ignoreMouseOver && !Second.ignoreMouseOver, description + " restores selected-card input flags");
            }

            public void CheckSentinel(string description)
            {
                Check(SentinelTween.IsActive(), description + " does not kill an unrelated tween");
                Tick(6f);
                Check(Vector3.Distance(Sentinel.localPosition, new Vector3(30f, 0f, 0f)) < 0.01f,
                    description + " leaves the unrelated tween able to finish");
            }

            public void Dispose()
            {
                if (View != null) Stop(View, true);
                SentinelTween.Kill(false);
                if (Root != null) UnityEngine.Object.DestroyImmediate(Root);
            }
        }

        private static void NormalAndDuplicateInput()
        {
            nativeProblems.Clear();
            using (Fixture f = new Fixture())
            {
                f.Completion = delegate
                {
                    // The real completion hides the exchange view immediately. Cleanup
                    // inside a completing tween must be reentrant and preserve wild cards.
                    Stop(f.View, true);
                    f.View.CreateHideViewAnimation().Play();
                };
                f.First.OnMouseClicked();
                Check(f.First.ignoreMouseOver, "first native card is input-locked synchronously before a tween update");
                f.First.OnMouseClicked();
                Check(f.Accepted == 1, "same-frame repeated native click cannot accept the same physical card twice");
                f.Second.OnMouseClicked();
                Check(f.Accepted == 2, "a different native card remains selectable");
                Tick(4f);
                Check(f.Completions == 1, "normal animation invokes completion exactly once");
                Check(f.Layout.enabled, "normal animation restores the hand layout");
                Check(f.WildIndustry.transform.parent == f.Hand && f.WildLocation.transform.parent == f.Hand,
                    "successful exchange transfers both native wild cards into the hand");
                UnityEngine.Object.DestroyImmediate(f.ScoutObject);
                Check(f.WildIndustry != null && f.WildLocation != null,
                    "destroying a completed exchange view preserves its transferred wild cards");
                Tick(1f);
                Check(f.Completions == 1, "extra updates/destruction cannot invoke successful completion again");
                f.CheckSentinel("successful exchange cleanup");
                NoNativeProblems("normal completion and synchronous repeated click");
            }
        }

        private static void Cancel(float cancelAfterSecond, float betweenSelections)
        {
            nativeProblems.Clear();
            using (Fixture f = new Fixture())
            {
                f.SelectBoth(betweenSelections);
                Tick(cancelAfterSecond);
                Check(f.Completions == 0, "cancellation fixture reaches an unfinished exchange");
                Stop(f.View, true);
                f.View.CreateHideViewAnimation().Play();
                f.CheckRestored("cancellation");
                Stop(f.View, true);
                f.CheckRestored("repeated cancellation cleanup");
                Tick(0.4f);
                UnityEngine.Object.DestroyImmediate(f.ScoutObject);
                Tick(4f);
                Check(f.Completions == 0, "cancel/hide/destruction prevents a delayed completion");
                f.CheckRestored("post-destruction cancellation");
                f.CheckSentinel("cancelled exchange cleanup");
                NoNativeProblems("cancelled exchange at " + cancelAfterSecond + " seconds");
            }
        }

        private static void DestroyWithoutExplicitCancel(float destroyAfterSecond)
        {
            nativeProblems.Clear();
            using (Fixture f = new Fixture())
            {
                f.SelectBoth(0f);
                Tick(destroyAfterSecond);
                Check(f.Completions == 0, "destruction fixture reaches an unfinished exchange");
                UnityEngine.Object.DestroyImmediate(f.ScoutObject);
                Tick(4f);
                Check(f.Completions == 0, "native lifetime destruction cancels outstanding completion without an explicit cancel call");
                Check(f.Layout.enabled, "native lifetime destruction releases hand layout");
                f.CheckSentinel("direct Scout destruction");
                NoNativeProblems("direct destruction at " + destroyAfterSecond + " seconds");
            }
        }

        private static void IntroDestruction()
        {
            nativeProblems.Clear();
            using (Fixture f = new Fixture())
            {
                ShowCompleted.Invoke(f.View, null);
                Tick(0.2f);
                UnityEngine.Object.DestroyImmediate(f.ScoutObject);
                Tick(4f);
                Check(f.Completions == 0, "destroying the introductory animation never reports an exchange completion");
                f.CheckSentinel("introductory Scout destruction");
                NoNativeProblems("introductory animation destruction");
            }
        }

        private static void CancelAndReopenCachedView(float cancelAfterSecond)
        {
            nativeProblems.Clear();
            using (Fixture f = new Fixture())
            {
                Transform slot0 = f.ScoutObject.transform.Find("SafeArea/Popup/Panel/Slot1");
                Transform slot1 = f.ScoutObject.transform.Find("SafeArea/Popup/Panel/Slot2");
                Vector3 originalSlot0Scale = slot0.localScale, originalSlot1Scale = slot1.localScale;
                f.SelectBoth(0f);
                Tick(cancelAfterSecond);
                Check(f.Completions == 0, "cached-view reopen starts by cancelling an unfinished Scout");
                Stop(f.View, true);
                InterfaceAdapters.Common.IAnimation oldHide = f.View.CreateHideViewAnimation().Play();
                Tick(0.1f);
                Check(oldHide.IsPlaying(), "old 0.3-second hide is still pending when a new Scout opens");

                // This is the presenter's PrepareForShow followed by Router.ShowView's
                // cancellation of the old hide and reuse of that still-cached view.
                PrepareForShow(f.View);
                oldHide.Kill();
                Check(Vector3.Distance(slot0.localScale, originalSlot0Scale) < 0.001f &&
                    Vector3.Distance(slot1.localScale, originalSlot1Scale) < 0.001f,
                    "cached-view reopen restores the original slot scales");
                f.View.CreateShowViewAnimation().Play();
                f.Accepted = 0;
                f.Completion = delegate { Stop(f.View, true); f.View.CreateHideViewAnimation().Play(); };
                // Select while the new show is pending, before its delayed ShowViewCompleted.
                f.SelectBoth(0.2f);
                Tick(4f);
                Check(f.Completions == 1, "reopened cached Scout accepts a fresh exchange and completes once");
                Check(f.Layout.enabled, "reopened exchange releases layout after successful completion");
                Check(f.WildIndustry.transform.parent == f.Hand && f.WildLocation.transform.parent == f.Hand,
                    "reopened exchange transfers both wild cards to the hand");
                UnityEngine.Object.DestroyImmediate(f.ScoutObject);
                Tick(1f);
                Check(f.Completions == 1, "reopened exchange cannot receive an old completion after destruction");
                Check(f.WildIndustry != null && f.WildLocation != null, "reopened exchange retains successfully transferred wilds");
                f.CheckSentinel("cached Scout reuse");
                NoNativeProblems("cancel/reopen cached view after " + cancelAfterSecond + " seconds");
            }
        }

        public static int Run()
        {
            checks = 0;
            DOTween.Init();
            Check(TweenUpdate != null && ShowCompleted != null, "shipped native animation entry points are available");
            Application.logMessageReceived += Capture;
            try
            {
                Action[] scenarios = {
                    NormalAndDuplicateInput,
                    delegate { Cancel(0.2f, 0f); },
                    delegate { Cancel(2.2f, 0f); },
                    delegate { Cancel(0.2f, 0.2f); },
                    delegate { DestroyWithoutExplicitCancel(0.2f); },
                    delegate { DestroyWithoutExplicitCancel(2.2f); },
                    IntroDestruction,
                    delegate { CancelAndReopenCachedView(0.2f); },
                    delegate { CancelAndReopenCachedView(2.2f); }
                };
                foreach (Action scenario in scenarios)
                {
                    scenario();
                    NoNativeProblems("final fixture disposal");
                }
                return checks;
            }
            finally { Application.logMessageReceived -= Capture; }
        }
    }
}
