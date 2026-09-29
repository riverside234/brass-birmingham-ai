using System;
using System.Reflection;
using BrassApplication.UseCases.Common;
using BrassPatch.Liquidation;
using DG.Tweening;
using Frameworks.Routing;
using Frameworks.Routing.Unity;
using Frameworks.View.Animation;
using Frameworks.View.Common;
using InterfaceAdapters.Common;
using InterfaceAdapters.Routing;
using TMPro;
using UnityEngine;
using UnityEngine.UI;
using Utils;

namespace BrassPatch.Tests.Liquidation
{
    /// <summary>
    /// Optional in-player smoke test. Run synchronously on Unity's main thread after the initial
    /// scene is loaded, with normal FrameworksInit skipped. Does not construct stores or clients.
    /// Uses the shipped popup prefab, actual Router/cache, TMP labels, Button events and DOTween.
    /// </summary>
    public static class UnityPresentationRegression
    {
        private static int checks;
        private static void Check(bool condition, string description)
        {
            checks++;
            if (!condition) throw new InvalidOperationException("Unity liquidation presentation: " + description);
        }
        private static T Read<T>(object owner, string name)
        {
            return (T)owner.GetType().GetField(name, BindingFlags.Instance | BindingFlags.NonPublic).GetValue(owner);
        }
        private static readonly MethodInfo ShowQuestion = typeof(LiquidationFixes).GetMethod("ShowQuestion", BindingFlags.Static | BindingFlags.NonPublic);
        private static void Show(IRouter router, string text, Action yes, Action next)
        {
            ShowQuestion.Invoke(null, new object[] { router, text, "Liquidate £7", "Next industry", yes, next });
        }
        private static QuestionPopupView Current(IRouter router) { return (QuestionPopupView)router.GetView<IQuestionPopupView>(); }
        private static void AdvanceToCompletion(IAnimation animation)
        {
            Frameworks.View.Animation.DOTweenAnimation native = animation as Frameworks.View.Animation.DOTweenAnimation;
            if (native == null) throw new InvalidOperationException("Expected the real DOTween presentation animation.");
            // Use DOTween's actual update loop. Forcing Complete(true) outside that loop can
            // double-despawn a sequence when the next popup kills its completing predecessor.
            // In normal frames the engine defers those kills until its update iteration ends.
            for (int frame = 0; frame < 120 && native.Sequence.IsActive() && !native.Sequence.IsComplete(); frame++)
                DOTween.ManualUpdate(1f / 30f, 1f / 30f);
            if (native.Sequence.IsActive() && !native.Sequence.IsComplete())
                throw new InvalidOperationException("Native popup animation did not finish within four simulated seconds.");
        }

        public static int Run()
        {
            checks = 0;
            GameObject factoryObject = GameObject.Find("Utils/UnityViewFactory");
            Check(factoryObject != null, "initial scene has the shipped UnityViewFactory");
            UnityViewFactory factory = factoryObject.GetComponent<UnityViewFactory>();
            Check(factory != null && factory.ViewList != null && factory.ParentTransform != null, "serialized prefab list and canvas parent are available");
            DOTween.Init();
            IRouter previous = Global.Instance.Router;
            TrackingRouter router = new TrackingRouter(factory);
            Global.Instance.Router = router;
            try
            {
                int nextCalls = 0, confirmCalls = 0;
                const string firstText = "Player Blue must cover £10 of income debt.\nCannock: CoalMine level 2\nReceive £7; debt remaining £3.";
                Show(router, firstText, () => confirmCalls++, delegate
                {
                    nextCalls++;
                    Show(router, "Second owned industry", () => confirmCalls++, () => nextCalls++);
                });
                QuestionPopupView first = Current(router);
                Check(Read<TextMeshProUGUI>(first, "_questionText").text == firstText, "native TMP label receives debt/industry text");
                Check(Read<TextMeshProUGUI>(first, "_confirmButtonText").text == "Liquidate £7", "native confirmation label includes proceeds");
                Check(Read<TextMeshProUGUI>(first, "_cancelButtonText").text == "Next industry", "native alternate label offers another industry");
                Check(first.DestroyWhenClosed, "shipped popup is destroyed rather than retaining callback listeners");
                AdvanceToCompletion(router.LastShow);
                Check(first.IsViewVisible(), "native show animation makes the popup visible");

                first.HandleKeyDown(KeyEnum.Escape);
                Check(nextCalls == 0 && confirmCalls == 0, "Escape waits for the native hide animation before replacing the prompt");
                Check(!first.transform.Find("SafeArea/Popup").GetComponent<CanvasGroup>().interactable, "choice disables native popup interaction immediately");
                AdvanceToCompletion(router.LastHide);
                Check(nextCalls == 1 && confirmCalls == 0, "the alternate callback runs once after native hide completion");
                QuestionPopupView second = Current(router);
                Check(!System.Object.ReferenceEquals(first, second), "Router removes the old cached prefab before constructing the next prompt");
                Check(Read<TextMeshProUGUI>(second, "_questionText").text == "Second owned industry", "next prompt retains its text");
                AdvanceToCompletion(router.LastShow);
                // The old confirm listener can still be invoked until Unity destroys its object at
                // the end of the frame. The production callback's consumed guard must reject it.
                Read<Button>(first, "_confirmButton").onClick.Invoke();
                Check(confirmCalls == 0 && nextCalls == 1, "a stale native button cannot commit another choice");

                second.HandleKeyDown(KeyEnum.Return);
                Check(confirmCalls == 0, "Return defers the mutation until its hide animation finishes");
                AdvanceToCompletion(router.LastHide);
                Check(confirmCalls == 1, "the replacement popup's keyboard callback survives the old popup's listener cleanup");
                Check(!router.IsViewVisible<IQuestionPopupView>(), "completed confirmation removes the modal from the real Router stack");

                Show(router, "Mouse confirmation", () => confirmCalls++, () => nextCalls++);
                QuestionPopupView mouse = Current(router); AdvanceToCompletion(router.LastShow);
                Button button = Read<Button>(mouse, "_confirmButton");
                button.onClick.Invoke(); button.onClick.Invoke();
                Check(confirmCalls == 1, "native double click does not mutate before presentation completes");
                AdvanceToCompletion(router.LastHide);
                Check(confirmCalls == 2 && nextCalls == 1, "native double click dispatches exactly one confirmation");
                Check(!router.IsViewCached<IQuestionPopupView>(), "all closed popup instances leave the native cache");
                return checks;
            }
            finally
            {
                if (router.IsViewVisible<IQuestionPopupView>()) AdvanceToCompletion(router.HideView<IQuestionPopupView>());
                Global.Instance.Router = previous;
            }
        }

        private sealed class TrackingRouter : IRouter
        {
            private readonly Router inner;
            public IAnimation LastShow, LastHide;
            public TrackingRouter(IViewFactory factory) { inner = new Router(factory, new NoInput()); inner.UseCaseBuilder = new NoUseCases(); }
            public IUseCaseBuilder UseCaseBuilder { get { return inner.UseCaseBuilder; } set { inner.UseCaseBuilder = value; } }
            private static IAnimation Manual(IAnimation animation)
            {
                ((Frameworks.View.Animation.DOTweenAnimation)animation).Sequence.SetUpdate(UpdateType.Manual);
                return animation;
            }
            public IAnimation ShowView<T>() where T : class, IBaseView { LastShow = Manual(inner.ShowView<T>()); return LastShow; }
            public IAnimation HideView<T>() where T : class, IBaseView { LastHide = Manual(inner.HideView<T>()); return LastHide; }
            public void MoveOnTop<T>() where T : class, IBaseView { inner.MoveOnTop<T>(); }
            public bool IsViewVisible<T>() where T : class, IBaseView { return inner.IsViewVisible<T>(); }
            public bool IsViewCached<T>() where T : class, IBaseView { return inner.IsViewCached<T>(); }
            public T GetView<T>() where T : class, IBaseView { return inner.GetView<T>(); }
            public void ProcessInput() { inner.ProcessInput(); }
            public void RegisterKey(KeyEnum key, IBaseView view) { inner.RegisterKey(key, view); }
            public void UnregisterKey(KeyEnum key, IBaseView view) { inner.UnregisterKey(key, view); }
            public void UnregisterAllKeys(IBaseView view) { inner.UnregisterAllKeys(view); }
        }
        private sealed class NoInput : IInputSource
        {
            public bool GetKeyDown(KeyEnum key) { return false; }
            public bool GetKey(KeyEnum key) { return false; }
            public bool GetKeyUp(KeyEnum key) { return false; }
        }
        private sealed class NoUseCases : IUseCaseBuilder
        {
            public void BuildUseCases() { }
            public void RestoreUseCases() { }
            public void RestoreUseCaseOfType(Type type) { }
            public IController GetController(Type type) { return null; }
            public TUseCase GetUseCase<TUseCase>() where TUseCase : IUseCase { throw new NotSupportedException("The popup smoke test must not create application use cases."); }
        }
    }
}
