using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using BrassPatch;
using BrassPatch.Tests.CityLocator;
using BoardGameRules.Entities.Board.BuildingSlots;
using BoardGameRules.Entities.Players;
using BrassApplication.UseCases.Common;
using DG.Tweening;
using Frameworks.Routing;
using Frameworks.Routing.Unity;
using Frameworks.View.Common;
using Frameworks.View.Game;
using Frameworks.View.Game.Replay;
using InterfaceAdapters.Common;
using InterfaceAdapters.Game.Board;
using InterfaceAdapters.Game.PlayersSummary;
using InterfaceAdapters.Game.IndustriesDetails;
using InterfaceAdapters.Game.GameProgressDetails;
using InterfaceAdapters.Game.GameLogView;
using InterfaceAdapters.Game.Replay;
using InterfaceAdapters.Game.EndOfEra;
using InterfaceAdapters.Game.EndOfGame;
using InterfaceAdapters.Routing;
using Newtonsoft.Json;
using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.SceneManagement;
using UnityEngine.UI;

// Loaded through the existing isolated test startup route; never starts storage.
public static class LocalAiUnityTests
{
    public static void Begin(string output)
    {
        var driver = new GameObject("Presentation lifecycle tests").AddComponent<PresentationDriver>();
        UnityEngine.Object.DontDestroyOnLoad(driver.gameObject); driver.Output = output;
    }
}
public sealed class PresentationDriver : MonoBehaviour
{
    public string Output;
    readonly List<object> cases = new List<object>();
    readonly List<string> errors = new List<string>();
    int checks, failures;
    static T Get<T>(object owner,string field) { return ReflectionAccess.Get<T>(owner,field); }
    static void Tick(float time) { UnityCityLocatorRegression.Tick(time); }
    void Check(bool passed,string name,object detail=null) { checks++; if(!passed)failures++; cases.Add(new {name,passed,detail}); }
    void Log(string message,string stack,LogType kind) { if(message.StartsWith("Error saving PlayerPrefs LastGoogleUpdate_I2Languages"))return; if(kind==LogType.Error || kind==LogType.Exception) errors.Add(message); }
    IEnumerator Start()
    {
        yield return SceneManager.LoadSceneAsync("GameScene"); yield return null;
        Application.logMessageReceived+=Log;
        Exception failure=null;
        var run=Run();
        while(true) { bool more=false; object current=null; try {more=run.MoveNext();if(more)current=run.Current;}catch(Exception e){failure=e;} if(failure!=null || !more)break;yield return current; }
        File.WriteAllText(Output,JsonConvert.SerializeObject(new {passed=failure==null && failures==0 && errors.Count==0,checks,failures,cases,nativeErrors=errors,error=failure==null?null:failure.ToString()},Formatting.Indented));
        Application.Quit(failure==null && failures==0 && errors.Count==0?0:1);
    }
    IEnumerator Run()
    {
        var prefabs=Resources.FindObjectsOfTypeAll<UnityViewFactory>().SelectMany(f=>f.ViewList.Views).OfType<BaseView>().ToArray();
        UnityCityLocatorRegression.Prefab=prefabs.OfType<BoardView>().First(v=>v.GetType()==typeof(BoardView));
        using(var f=new UnityCityLocatorRegression.Fixture(4))
        {
            foreach(var player in f.App.Game.PlayersTrack.Players)
            foreach(BuildingType type in Enum.GetValues(typeof(BuildingType)))
            {
            if(type==BuildingType.Undefined)continue;
            var slot=f.App.Game.Board.Regions.SelectMany(r=>r.BuildingSlots).First(s=>s.BuildingTypes.Contains(type));
            var industry=player.Mat.GetFirstBuildingOfType(type); slot.BuildedIndustry=industry;
            foreach(bool alreadyFlipped in new[]{false,true})
            {
                industry.IsFlipped=alreadyFlipped; f.Update();
                var animation=(Frameworks.View.Animation.DOTweenAnimationSequence)f.View.CreateIndustryFlippedAnimation(slot);
                animation.Play(); Tick(5f);
                var view=f.View.GetComponentsInChildren<BuildingSlotView>(true).First(v=>v.name==slot.UniqueId);
                var token=view.transform.Find("IndustryToken").GetComponent<Image>();
                Check(token.sprite!=null && token.sprite.name==player.Color.ToString()+type+"Flipped","flip has a valid owned sprite",new{color=player.Color.ToString(),type=type.ToString(),alreadyFlipped,sprite=token.sprite==null?null:token.sprite.name});
                animation=(Frameworks.View.Animation.DOTweenAnimationSequence)f.View.CreateIndustryFlippedAnimation(slot);
                animation.Play(); Tick(.6f); animation.Kill(); Tick(.1f);
                Check(token.sprite!=null && token.sprite.name==player.Color.ToString()+type+"Flipped" && token.transform.localScale==Vector3.one && token.transform.localRotation==Quaternion.identity,"interrupted flip restores sprite and transform");
                animation=(Frameworks.View.Animation.DOTweenAnimationSequence)f.View.CreateIndustryFlippedAnimation(slot);
                animation.Play(); Tick(.3f); industry.IsFlipped=false; f.Update(); Tick(4f);
                Check(token.sprite!=null && token.sprite.name==player.Color.ToString()+type,"slot rebind cancels obsolete flip");
            }
            slot.BuildedIndustry=null; f.Update();
            }
            DOTween.KillAll();
        }
        var root=new GameObject("Presentation canvas",typeof(RectTransform),typeof(Canvas),typeof(GraphicRaycaster),typeof(CanvasScaler));
        root.GetComponent<Canvas>().renderMode=RenderMode.ScreenSpaceOverlay;
        root.GetComponent<CanvasScaler>().uiScaleMode=CanvasScaler.ScaleMode.ScaleWithScreenSize;
        root.GetComponent<CanvasScaler>().referenceResolution=new Vector2(1920,1080);
        var factory=new Factory(root.transform,prefabs); var router=new Frameworks.Routing.Router(factory,new Input());
        factory.Router=router; router.UseCaseBuilder=new Builder(); Utils.Global.Instance.Router=router;
        var summary=(PlayersSummaryView)router.GetView<IPlayersSummaryView>(); summary.DestroyWhenClosed=false;
        router.ShowView<IPlayersSummaryView>(); Tick(1f); Canvas.ForceUpdateCanvases();
        yield return null;
        var rows=Get<Transform>(summary,"_playersContainer").Cast<Transform>().Select(t=>(RectTransform)t).ToArray();
        cases.Add(new{name="player row geometry",detail=rows.Select(r=>new{name=r.name,y=r.anchoredPosition.y,anchorMin=r.anchorMin.ToString(),anchorMax=r.anchorMax.ToString(),size=r.sizeDelta.ToString(),components=r.GetComponents<Component>().Select(c=>c.GetType().Name).ToArray()}).ToArray(),container=Get<Transform>(summary,"_playersContainer").GetComponents<Component>().Select(c=>c.GetType().Name).ToArray()});
        for(int i=0;i<rows.Length;i++) rows[i].GetComponent<PlayerView>().PlayerColor=(PlayerColor)i;
        // A layout pass (including a hidden/reopened HUD) may change Y after tween startup.
        var current=summary.CreateShowCurrentPlayerAnimation((PlayerColor)0); current.Play(); Tick(.2f);
        var intended=rows.Select((r,i)=>-100f-i*170f).ToArray();
        for(int i=0;i<rows.Length;i++) rows[i].anchoredPosition=new Vector2(rows[i].anchoredPosition.x,intended[i]);
        Tick(.1f);
        Check(rows.Select((r,i)=>Math.Abs(r.anchoredPosition.y-intended[i])<.01f).All(x=>x),"current-player animation preserves row layout",rows.Select(r=>r.anchoredPosition.y).ToArray());
        current.Kill(); DOTween.KillAll();
        for(int i=0;i<rows.Length;i++) rows[i].anchoredPosition=new Vector2(0, -100-i*170);
        current=summary.CreateShowCurrentPlayerAnimation((PlayerColor)0);current.Play();Tick(.15f);current.Kill();
        float stopped=rows[0].anchoredPosition.x;Tick(1f);
        Check(Math.Abs(rows[0].anchoredPosition.x-stopped)<.01f,"cancelling current-player animation cancels nested movement");
        DOTween.KillAll();
        foreach(string method in new[]{"CreateHidePlayerAnimation","CreateShowPlayerAnimation"})
        {
            var sequence=(Sequence)typeof(PlayersSummaryView).GetMethod(method,ReflectionAccess.Flags).Invoke(summary,new object[]{rows[1]});
            sequence.Play(); Tick(.03f); rows[1].anchoredPosition=new Vector2(rows[1].anchoredPosition.x,-345f); Tick(.03f);
            Check(Math.Abs(rows[1].anchoredPosition.y+345f)<.01f,method+" preserves row layout",rows[1].anchoredPosition.y);
            sequence.Kill();
        }
        var models=Enumerable.Range(0,4).Select(i=>new PlayerViewModel{Color=(PlayerColor)i,Name="Player "+i,IsCurrent=i==0}).ToList();
        summary.UpdatePlayersSummaryView(models.Take(2).ToList()); Tick(1f);
        summary.UpdatePlayersSummaryView(models); Tick(1f); Canvas.ForceUpdateCanvases(); yield return null;
        Check(rows.All(r=>r.gameObject.activeSelf),"HUD refresh restores all occupied seats");
        // Simulate a stale layout after a hidden/animated HUD refresh; recovery must be local to Y.
        for(int i=0;i<rows.Length;i++)rows[i].anchoredPosition=new Vector2(i==0?35:0,0);
        yield return null; yield return null;
        Check(rows.Select(r=>Math.Round(r.anchoredPosition.y,2)).Distinct().Count()==4,"HUD repairs overlapping rows",rows.Select(r=>r.anchoredPosition.ToString()).ToArray());
        foreach(var test in new[]{Panel<IIndustriesDetailsView>(router,root),Panel<IGameProgressDetailsView>(router,root),Panel<IGameLogView>(router,root)}) while(test.MoveNext())yield return test.Current;
        DOTween.KillAll(); UnityEngine.Object.DestroyImmediate(root);
    }
    IEnumerator Panel<T>(Frameworks.Routing.Router router,GameObject root) where T:class,IBaseView
    {
        if(router.IsViewVisible<IReplayView>()) { router.HideView<IReplayView>(); Tick(1f); }
        var panel=(BaseView)(object)router.GetView<T>(); panel.DestroyWhenClosed=false;
        router.ShowView<T>(); Tick(1f); Canvas.ForceUpdateCanvases();
        yield return null; Canvas.ForceUpdateCanvases();
        var buttons=panel.GetComponentsInChildren<Button>(true);
        var button=buttons.First(b=>b.name.IndexOf("Close",StringComparison.OrdinalIgnoreCase)>=0 || b.name.IndexOf("Back",StringComparison.OrdinalIgnoreCase)>=0 || Enumerable.Range(0,b.onClick.GetPersistentEventCount()).Any(i=>b.onClick.GetPersistentMethodName(i).IndexOf("Back",StringComparison.OrdinalIgnoreCase)>=0));
        if(panel is IndustriesDetailsView)panel.Controller=new IndustriesDetailsController(new BrassApplication.UseCases.Game.IndustriesDetails.IndustriesDetailsUseCase(new IndustriesDetailsPresenter(router,null),null));
        if(panel is GameProgressDetailsView)panel.Controller=new GameProgressDetailsController(new BrassApplication.UseCases.Game.GameProgressDetails.GameProgressDetailsUseCase(new GameProgressDetailsPresenter(router,null),null));
        if(panel is GameLogView)panel.Controller=new GameLogController(new LogInput(router));
        foreach(string stage in new[]{"thinking","replay","replay-moved-on-top","era-end","game-end"})
        {
            if(stage=="replay")router.ShowView<IReplayView>();
            if(stage=="replay-moved-on-top")router.MoveOnTop<IReplayView>();
            if(stage=="era-end")router.ShowView<IEndOfEraView>();
            if(stage=="game-end")router.ShowView<IEndOfGameView>();
            Tick(1f); router.ProcessInput();
            yield return null; Canvas.ForceUpdateCanvases();
            var es=EventSystem.current;
            if(es==null) es=new GameObject("Presentation event system",typeof(EventSystem)).GetComponent<EventSystem>();
            var data=new PointerEventData(es){position=RectTransformUtility.WorldToScreenPoint(null,button.transform.position)};
            var hits=new List<RaycastResult>(); es.RaycastAll(data,hits);
            var first=hits.FirstOrDefault();
            bool reachable=first.gameObject!=null && (first.gameObject==button.gameObject || first.gameObject.transform.IsChildOf(button.transform));
            Check(reachable,"information panel close hit remains reachable",new{panel=typeof(T).Name,stage,hit=first.gameObject==null?null:PathOf(first.gameObject.transform)});
            if(reachable)button.onClick.Invoke();else router.HideView<T>();
            Tick(1f);yield return null;
            Check(!router.IsViewVisible<T>(),"close button closes only the information panel",new{panel=typeof(T).Name,stage});
            router.ShowView<T>();Tick(1f);
        }
        if(panel is IndustriesDetailsView)
        {
            var replay=(ReplayView)router.GetView<IReplayView>();
            var animation=replay.CreateAnimation(new BoardGameRules.Entities.Actions.Events.IndustryDeveloped(PlayerColor.Blue,BuildingType.CoalMine));
            bool complete=false;animation.OnComplete(()=>complete=true);animation.Play();
            button.onClick.Invoke();Tick(1f);yield return null;
            Check(complete,"closing inspected development board cannot stall AI develop replay");
            router.ShowView<T>();Tick(1f);
        }
        router.ShowView<IQuestionPopupView>();Tick(1f);router.ProcessInput();
        Check(Get<List<IBaseView>>(router,"_viewStack")[0] is IQuestionPopupView,"confirmation modal retains priority");
        router.HideView<IQuestionPopupView>();Tick(1f);
        router.HideView<T>(); Tick(1f);
        if(router.IsViewVisible<IEndOfEraView>()){router.HideView<IEndOfEraView>();Tick(1f);}
        if(router.IsViewVisible<IEndOfGameView>()){router.HideView<IEndOfGameView>();Tick(1f);}
    }
    sealed class LogInput:BrassApplication.UseCases.Game.GameLog.IGameLogInput
    {
        readonly IRouter router;public LogInput(IRouter router){this.router=router;}
        public void CloseGameLog(){router.HideView<IGameLogView>();}
        public void UpdateGameLog(){}
        public void UpdateReplaySummary(IList<BoardGameRules.Entities.EventSourcing.Events.IEvent> events,Action dismiss){}
    }
    static string PathOf(Transform t) { return t.parent==null?t.name:PathOf(t.parent)+"/"+t.name; }
    sealed class Input:IInputSource { public bool GetKeyDown(KeyEnum k){return false;} public bool GetKey(KeyEnum k){return false;} public bool GetKeyUp(KeyEnum k){return false;} }
    sealed class Factory:IViewFactory
    {
        readonly Transform root; readonly BaseView[] prefabs; public IRouter Router;
        public Factory(Transform root,BaseView[] prefabs){this.root=root;this.prefabs=prefabs;}
        public T CreateView<T>() where T:class,IBaseView
        {var view=UnityEngine.Object.Instantiate(prefabs.First(v=>v is T),root);view.InputRegistration=Router;view.name=typeof(T).Name;view.DisableView();return view as T;}
        public void DestroyView(IBaseView v){UnityEngine.Object.Destroy(((BaseView)v).gameObject);}
    }
    sealed class Builder:IUseCaseBuilder
    {
        public void BuildUseCases(){} public void RestoreUseCases(){} public void RestoreUseCaseOfType(Type t){}
        public IController GetController(Type t){return null;} public T GetUseCase<T>() where T:IUseCase{throw new NotSupportedException();}
    }
}
