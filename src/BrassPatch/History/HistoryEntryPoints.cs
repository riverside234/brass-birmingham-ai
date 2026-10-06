using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;
using BrassApplication.UseCases.Menu.MainMenu;
using BrassPatch.Statistics;
using BrassPatch.UI;
using Frameworks.Routing;
using Frameworks.View.Menu;
using InterfaceAdapters.Menu.MainMenu;
using TMPro;
using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.UI;

namespace BrassPatch.History
{
    public static class HistoryEntryPoints
    {
        sealed class Session
        {
            public Router Router; public GameHistoryPanel Page; public StatisticsPanel Viewer; public GameHistoryStore Store;
            public Task<GameHistoryList> List; public Task<GameHistoryRecord> Record;
            public bool Disposed; public int Generation, ListGeneration, RecordGeneration; public string Key;
            public readonly Dictionary<Selectable, bool> Blocked = new Dictionary<Selectable, bool>();
        }
        sealed class Frame { public int Closed = -1; }
        static readonly ConditionalWeakTable<Router, Session> sessions = new ConditionalWeakTable<Router, Session>();
        static readonly ConditionalWeakTable<Router, Frame> frames = new ConditionalWeakTable<Router, Frame>();
        public static void Bind(MainMenuView view)
        {
            foreach (var button in view.GetComponentsInChildren<Button>(true)) {
                bool entry = Enumerable.Range(0, button.onClick.GetPersistentEventCount()).Any(i => button.onClick.GetPersistentMethodName(i) == "OnOnlineGamesClicked" && ReferenceEquals(button.onClick.GetPersistentTarget(i), view));
                if (!entry) continue;
                var label = button.GetComponentsInChildren<TextMeshProUGUI>(true).FirstOrDefault(); if (label == null) continue;
                foreach (var localizer in label.GetComponents<Behaviour>().Where(c => c.GetType().FullName == "I2.Loc.Localize")) localizer.enabled = false;
                label.font=StatisticsViewHost.ArchiveTypeface(button.transform);label.fontStyle=FontStyles.Normal;
                label.rectTransform.anchorMin=new Vector2(.23f,0);label.rectTransform.anchorMax=Vector2.one;label.rectTransform.offsetMin=Vector2.zero;label.rectTransform.offsetMax=new Vector2(-4,0);
                var theme = button.GetComponent<LightUiTheme>() ?? button.gameObject.AddComponent<LightUiTheme>(); theme.PixelUnit = StatisticsEntryPoints.PixelUnit(button.transform);
                var face=button.transform.Find("Game History light surface");
                if (face == null) {
                    if (button.targetGraphic != null) button.targetGraphic.enabled = false;
                    var rect=StatisticsWidgets.Rect(button.transform,"Game History light surface",Vector2.zero,Vector2.zero);
                    StatisticsWidgets.Stretch(rect,Vector2.zero,Vector2.zero);rect.SetAsFirstSibling();
                    button.targetGraphic=rect.gameObject.AddComponent<HistoryControlSurface>();button.transition=Selectable.Transition.None;
                }
                if(button.transform.Find("Game History chart icon")==null){
                    var rect=StatisticsWidgets.Rect(button.transform,"Game History chart icon",Vector2.zero,Vector2.zero);
                    rect.anchorMin=new Vector2(.08f,.26f);rect.anchorMax=new Vector2(.17f,.74f);rect.offsetMin=rect.offsetMax=Vector2.zero;
                    rect.gameObject.AddComponent<HistoryMenuIcon>().raycastTarget=false;
                }
                var caption = button.GetComponent<HistoryMenuCaption>() ?? button.gameObject.AddComponent<HistoryMenuCaption>(); caption.Label = label;caption.Surface=button.GetComponentInChildren<HistoryControlSurface>(true); caption.Apply();
            }
        }
        public static void Show(MainMenuUseCase owner)
        {
            var presenter = ReflectionAccess.Get(owner, "_presenter"); var router = ReflectionAccess.Get(presenter, "_router") as Router;
            if (router != null) Show(router);
        }
        public static void Show(Router router)
        {
            Session old; if (sessions.TryGetValue(router, out old) && !old.Disposed || frames.GetOrCreateValue(router).Closed == Time.frameCount) return;
            if (!router.IsViewVisible<IMainMenuView>()) return;
            var parent = StatisticsViewHost.Parent(router); if (parent == null) return;
            var session = new Session { Router = router, Store = HistoryRecording.Store };
            foreach (var control in parent.GetComponentsInChildren<Selectable>(true)) { session.Blocked[control] = control.interactable; control.interactable = false; }
            session.Page = StatisticsViewHost.Create<GameHistoryPanel>(router, "Game history");
            session.Page.Initialize(StatisticsViewHost.ArchiveTypeface(parent)); session.Page.OnBack = () => Close(session); session.Page.OnRetry = () => Reload(session);
            session.Page.OnDestroyed=()=>Close(session);
            session.Page.OnOpen = key => Open(session, key); session.Page.Tick = () => Pump(session);
            sessions.Add(router, session); router.HideView<IMainMenuView>(); router.ShowView<GameHistoryPanel>(); Reload(session);
        }
        static bool Current(Session session)
        { Session current; return !session.Disposed && session.Page != null && sessions.TryGetValue(session.Router, out current) && ReferenceEquals(current, session); }
        static void Reload(Session session)
        {
            if (!Current(session)) return; session.Generation++; session.Record = null;
            session.ListGeneration = session.Generation; session.Page.SetStatus("Loading recorded games…"); session.List = Task.Run(() => session.Store.List());
        }
        static void Open(Session session, string key)
        {
            if (!Current(session) || !session.Page.IsOpen) return;
            session.Generation++; session.Key = key; session.Page.SelectRow(key); session.Page.SetStatus("Opening recorded statistics…");
            session.RecordGeneration = session.Generation; session.Record = Task.Run(() => session.Store.Load(key));
        }
        static void Pump(Session session)
        {
            if (!Current(session)) return;
            if (session.List != null && session.List.IsCompleted) {
                var task = session.List; session.List = null;
                if (session.ListGeneration == session.Generation) {
                    var list=task.IsFaulted?new GameHistoryList{Games=new List<GameHistorySummary>().AsReadOnly(),Error="Game History could not be read. Use Refresh to retry."}:task.Result;
                    if(list.Diagnostic!=null)Debug.LogWarning("Game History read unavailable: "+list.Diagnostic);HistoryRecording.ApplyFailure(list);session.Page.SetGames(list);
                }
            }
            if (session.Record != null && session.Record.IsCompleted) {
                var task = session.Record; session.Record = null;
                if (session.RecordGeneration != session.Generation) return;
                if (task.IsFaulted) { session.Page.SetStatus("This statistics report is unavailable. Use Refresh to retry."); return; }
                var record = task.Result; if (record.Key != session.Key) return;
                session.Viewer = StatisticsViewHost.Create<StatisticsPanel>(session.Router, "Archived match statistics");
                session.Viewer.Initialize(record.Report, StatisticsViewHost.ArchiveTypeface(StatisticsViewHost.Parent(session.Router)), _ => Return(session),
                    StatisticsOpeningContext.ArchivedHistory, "Recorded " + record.RecordedUtc.ToLocalTime().ToString("d MMM yyyy, HH:mm"));
                session.Page.SetStatus(""); session.Router.HideView<GameHistoryPanel>(); session.Router.ShowView<StatisticsPanel>();
            }
        }
        static void Return(Session session)
        {
            if (!Current(session)) return; frames.GetOrCreateValue(session.Router).Closed = Time.frameCount;
            session.Router.HideView<StatisticsPanel>(); StatisticsViewHost.Dispose(session.Router, session.Viewer); session.Viewer = null;
            session.Router.ShowView<GameHistoryPanel>();
        }
        static void Close(Session session)
        {
            Session current;if(session.Disposed||!sessions.TryGetValue(session.Router,out current)||!ReferenceEquals(current,session))return;
            session.Disposed = true; session.Generation++; frames.GetOrCreateValue(session.Router).Closed = Time.frameCount;
            sessions.Remove(session.Router); StatisticsViewHost.Dispose(session.Router, session.Viewer); StatisticsViewHost.Dispose(session.Router, session.Page);
            foreach (var control in session.Blocked) if (control.Key != null) control.Key.interactable = control.Value;
            session.Blocked.Clear(); if(StatisticsViewHost.Parent(session.Router)!=null)session.Router.ShowView<IMainMenuView>();
        }
        public static bool ProcessInput(Router router)
        {
            Frame frame; if (frames.TryGetValue(router, out frame) && frame.Closed == Time.frameCount) return true;
            Session session; if (!sessions.TryGetValue(router, out session)) return false;
            if (!Current(session)) { if (!session.Disposed) Close(session); return false; }
            if (session.Viewer != null && session.Viewer.IsOpen) StatisticsViewHost.Keys(router, session.Viewer, () => session.Viewer != null && session.Viewer.IsOpen);
            else if (session.Page.IsOpen) StatisticsViewHost.Keys(router, session.Page, () => session.Page.IsOpen);
            return true;
        }
    }
    public sealed class HistoryMenuCaption : MonoBehaviour,IPointerEnterHandler,IPointerExitHandler,ISelectHandler,IDeselectHandler
    {
        public TextMeshProUGUI Label;
        public HistoryControlSurface Surface;
        public void Apply() { if (Label == null) return; Label.text = "Game History"; Label.richText = false; Label.color = StatisticsTheme.Ink; }
        void OnEnable() { Apply(); }
        void LateUpdate() { if (Label != null && (Label.text != "Game History" || Label.color != StatisticsTheme.Ink)) Apply(); }
        public void OnPointerEnter(PointerEventData data){if(Surface!=null)Surface.OnPointerEnter(data);}
        public void OnPointerExit(PointerEventData data){if(Surface!=null)Surface.OnPointerExit(data);}
        public void OnSelect(BaseEventData data){if(Surface!=null)Surface.OnSelect(data);}
        public void OnDeselect(BaseEventData data){if(Surface!=null)Surface.OnDeselect(data);}
    }
    public sealed class HistoryMenuIcon:MaskableGraphic
    {
        protected override void OnPopulateMesh(VertexHelper mesh)
        {
            mesh.Clear();var rect=GetPixelAdjustedRect();var ink=StatisticsTheme.Focus;
            for(int bar=0;bar<3;bar++){
                float x=rect.xMin+rect.width*bar/3f,w=rect.width*.22f,h=rect.height*(.4f+.3f*bar);int start=mesh.currentVertCount;
                mesh.AddVert(new Vector2(x,rect.yMin),ink,Vector2.zero);mesh.AddVert(new Vector2(x,rect.yMin+h),ink,Vector2.zero);
                mesh.AddVert(new Vector2(x+w,rect.yMin+h),ink,Vector2.zero);mesh.AddVert(new Vector2(x+w,rect.yMin),ink,Vector2.zero);
                mesh.AddTriangle(start,start+1,start+2);mesh.AddTriangle(start,start+2,start+3);
            }
        }
    }
}
