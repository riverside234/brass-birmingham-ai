using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using BrassApplication.Game;
using BrassApplication.GameHistory;
using BrassApplication.UseCases.Game.FinishedGameReview;
using Frameworks.Routing;
using Frameworks.Routing.Unity;
using Frameworks.View.Common;
using Frameworks.View.Game;
using InterfaceAdapters.Common;
using InterfaceAdapters.Game.EndOfGame;
using InterfaceAdapters.Game.FinishedGameReview;
using InterfaceAdapters.Routing;
using TMPro;
using UnityEngine;
using UnityEngine.UI;

namespace BrassPatch.Statistics
{
    public static class StatisticsEntryPoints
    {
        sealed class ReviewSession
        {
            public FinishedGameReviewUseCase Owner;
            public Router Router;
            public CompletedMatchStatistics Report;
            public StatisticsPanel Panel;
            public Button BoardButton;
            public bool Ready,Disposed,FromResults;
            public int ClosedFrame=-1;
        }
        static readonly ConditionalWeakTable<FinishedGameReviewUseCase,ReviewSession> owners=new ConditionalWeakTable<FinishedGameReviewUseCase,ReviewSession>();
        static readonly ConditionalWeakTable<Router,ReviewSession> routers=new ConditionalWeakTable<Router,ReviewSession>();
        static readonly KeyEnum[] keys={KeyEnum.Escape,KeyEnum.Tab,KeyEnum.PageUp,KeyEnum.PageDown,KeyEnum.Home,KeyEnum.End,KeyEnum.UpArrow,KeyEnum.DownArrow};
        public static void Capture(FinishedGameReviewUseCase owner,PlayerMetadata player)
        {
            // Called before ReplayUnseenActions is allowed to rewind presentation.
            try {
                var history=ReflectionAccess.Get<IGameHistory>(owner,"_gameHistory");
                if(history.CurrentGame==null||!history.CurrentGame.Game.GameProgressInformation.IsGameOver||history.CurrentGame.Metadata.GameType==GameType.Network)return;
                var presenter=ReflectionAccess.Get(owner,"_presenter");var router=ReflectionAccess.Get(presenter,"_router") as Router;
                if(router==null)return;
                ReviewSession previous;
                if(owners.TryGetValue(owner,out previous))return;
                if(routers.TryGetValue(router,out previous))Dispose(previous);
                var session=new ReviewSession{Owner=owner,Router=router,Report=StatisticsCollector.Capture(history)};
                owners.Add(owner,session);routers.Add(router,session);
            } catch(MissingMemberException) { /* Non-Unity test presenters have no router. */ }
              catch(Exception error) { Debug.LogWarning("Statistics capture unavailable: "+error.Message); }
        }
        public static void Ready(FinishedGameReviewUseCase owner)
        {
            ReviewSession session;if(!owners.TryGetValue(owner,out session)||session.Disposed||session.Ready)return;
            session.Ready=true;
            var review=session.Router.GetView<IFinishedGameReviewView>() as BaseView;
            if(review==null)return;
            var parent=review.transform.Find("SafeArea")??review.transform;
            var font=Typeface(review.transform);
            float unit=((RectTransform)review.transform).rect.width/Math.Max(1,Screen.width);if(unit<=0)unit=1;
            var button=StatisticsWidgets.Button(parent,font,"Match statistics","Statistics",Vector2.zero,new Vector2(180,48)*unit,20*unit,()=>Open(session,false,session.BoardButton));
            for(int i=0;i<3;i++) {
                var bar=StatisticsWidgets.Rect(button.transform,"Chart icon "+i,new Vector2(12+i*6,-30+i*4)*unit,new Vector2(4,10+i*4)*unit);
                bar.gameObject.AddComponent<Image>().color=StatisticsWidgets.Gold;bar.GetComponent<Image>().raycastTarget=false;
            }
            var rect=(RectTransform)button.transform;rect.anchorMin=rect.anchorMax=Vector2.zero;rect.pivot=Vector2.zero;
            float bottom=20*unit;
            // Stack above the existing review control when it occupies this corner.
            foreach(var existing in parent.GetComponentsInChildren<Button>()) {
                if(existing==button)continue;var er=(RectTransform)existing.transform;var corners=new Vector3[4];er.GetWorldCorners(corners);
                var local=((RectTransform)parent).InverseTransformPoint(corners[1]);
                if(local.x<-( ((RectTransform)parent).rect.width/2)+230*unit)bottom=Mathf.Max(bottom,local.y+((RectTransform)parent).rect.height/2+12*unit);
            }
            rect.anchoredPosition=new Vector2(20*unit,bottom);session.BoardButton=button;
            if(session.Router.IsViewCached<IEndOfGameView>())AttachResults(session.Router.GetView<IEndOfGameView>() as EndOfGameView,null);
        }
        public static void AttachResults(EndOfGameView view,Action exit)
        {
            if(view==null)return;var router=view.InputRegistration as Router;ReviewSession session;
            if(router==null||!routers.TryGetValue(router,out session)||!session.Ready||session.Disposed)return;
            var popup=view.transform.Find("SafeArea/Popup")??view.transform.Find("Popup");if(popup==null||popup.Find("Results statistics")!=null)return;
            float unit=((RectTransform)view.transform).rect.width/Math.Max(1,Screen.width);if(unit<=0)unit=1;
            Button button=null;button=StatisticsWidgets.Button(popup,Typeface(view.transform),"Results statistics","Statistics",Vector2.zero,new Vector2(168,44)*unit,20*unit,()=>Open(session,true,button));
            var rect=(RectTransform)button.transform;rect.anchorMin=rect.anchorMax=new Vector2(.5f,0);rect.pivot=new Vector2(.5f,0);rect.anchoredPosition=new Vector2(0,30*unit);
        }
        static TMP_FontAsset Typeface(Transform parent)
        {
            var label=parent.GetComponentsInChildren<TextMeshProUGUI>(true).FirstOrDefault(t=>t.font!=null);
            return label==null?TMP_Settings.defaultFontAsset:label.font;
        }
        static void Open(ReviewSession session,bool results,Button opener)
        {
            if(session.Disposed||!session.Ready||session.Panel!=null&&session.Panel.IsOpen)return;
            session.FromResults=results;
            if(session.Panel==null) {
                var factory=ReflectionAccess.Get(session.Router,"_viewFactory") as UnityViewFactory;
                if(factory==null||factory.ParentTransform==null)return;
                var root=StatisticsWidgets.Rect(factory.ParentTransform,"Completed match statistics",Vector2.zero,Vector2.zero);
                StatisticsWidgets.Stretch(root,Vector2.zero,Vector2.zero);
                var canvas=root.gameObject.AddComponent<Canvas>();canvas.overrideSorting=true;canvas.sortingOrder=500;
                root.gameObject.AddComponent<GraphicRaycaster>();root.gameObject.AddComponent<CanvasGroup>();
                session.Panel=root.gameObject.AddComponent<StatisticsPanel>();session.Panel.InputRegistration=session.Router;
                session.Panel.Initialize(session.Report,Typeface(factory.ParentTransform),restore=>{
                    if(session.Disposed)return;
                    session.ClosedFrame=Time.frameCount;
                    session.Router.HideView<StatisticsPanel>();
                    if(restore??session.FromResults)session.Owner.ShowGameSummaryTable();
                });
                var cache=ReflectionAccess.Get<Dictionary<Type,IBaseView>>(session.Router,"_viewCache");cache[typeof(StatisticsPanel)]=session.Panel;
            }
            session.Panel.SetOpener(opener==null?null:opener.gameObject);
            if(session.Router.IsViewVisible<IEndOfGameView>())session.Router.HideView<IEndOfGameView>();
            session.Router.ShowView<StatisticsPanel>();
        }
        public static void ProcessInput(Router router)
        {
            ReviewSession session;
            if(routers.TryGetValue(router,out session)&&!session.Disposed) {
                if(session.ClosedFrame==Time.frameCount)return;
                if(session.Panel!=null&&session.Panel.IsOpen) {
                    var source=ReflectionAccess.Get<IInputSource>(router,"_inputSource");
                    foreach(var key in keys) {
                        if(source.GetKeyDown(key))session.Panel.HandleKeyDown(key);
                        if(!session.Panel.IsOpen)return;
                        if(source.GetKey(key))session.Panel.HandleKey(key);
                    }
                    // Return/Space are handled by the panel's EventSystem controls.
                    // Never forward held keys to the board or old result dialog.
                    return;
                }
            }
            BrassPatch.Presentation.PresentationLifecycle.KeepInformationAccessible(router);
            ReflectionAccess.Original(router,"ProcessInput",Type.EmptyTypes);
        }
        public static void Finish(FinishedGameReviewUseCase owner)
        { ReviewSession session;if(owners.TryGetValue(owner,out session))Dispose(session); }
        static void Dispose(ReviewSession session)
        {
            if(session.Disposed)return;session.Disposed=true;owners.Remove(session.Owner);routers.Remove(session.Router);
            if(session.BoardButton!=null){session.BoardButton.gameObject.SetActive(false);UnityEngine.Object.Destroy(session.BoardButton.gameObject);}
            if(session.Panel!=null) {
                session.Panel.gameObject.SetActive(false);
                ReflectionAccess.Get<List<IBaseView>>(session.Router,"_viewStack").Remove(session.Panel);
                ReflectionAccess.Get<Dictionary<Type,IBaseView>>(session.Router,"_viewCache").Remove(typeof(StatisticsPanel));
                session.Router.UnregisterAllKeys(session.Panel);
                var animations=ReflectionAccess.Get<Dictionary<Type,IAnimation>>(session.Router,"_viewAnimations");IAnimation animation;
                if(animations.TryGetValue(typeof(StatisticsPanel),out animation)){animation.Kill();animations.Remove(typeof(StatisticsPanel));}
                UnityEngine.Object.Destroy(session.Panel.gameObject);
            }
        }
    }
}
