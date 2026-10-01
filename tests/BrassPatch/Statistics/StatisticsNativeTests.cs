using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using BoardGameRules.Entities.Actions;
using BoardGameRules.Entities.Board.BuildingSlots;
using BoardGameRules.Entities.Common;
using BoardGameRules.Entities.EventSourcing.Events;
using BoardGameRules.Entities.Game;
using BoardGameRules.Entities.Players;
using BrassApplication.AI;
using BrassApplication.AI.AiBehaviors;
using BrassApplication.AI.RuleBasedSystem;
using BrassApplication.AI.RuleBasedSystem.RuleSelectors;
using BrassApplication.Game;
using BrassApplication.GameHistory;
using BrassApplication.UseCases.Common;
using BrassApplication.UseCases.Game.FinishedGameReview;
using BrassApplication.UseCases.Game.Replay;
using BrassPatch;
using BrassPatch.Statistics;
using DG.Tweening;
using Frameworks.Persistence.GameFactory;
using Frameworks.Persistence.Repository;
using Frameworks.Persistence.RepositoryBackends.InMemory;
using Frameworks.Routing;
using Frameworks.Routing.Unity;
using Frameworks.View.Common;
using Frameworks.View.Game;
using InterfaceAdapters.Common;
using InterfaceAdapters.Game.EndOfGame;
using InterfaceAdapters.Game.FinishedGameReview;
using InterfaceAdapters.Routing;
using Newtonsoft.Json;
using TMPro;
using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.SceneManagement;
using UnityEngine.UI;

public static class LocalAiUnityTests
{
    public static void Begin(string output)
    {
        var driver=new GameObject("Statistics qualification").AddComponent<StatisticsDriver>();
        UnityEngine.Object.DontDestroyOnLoad(driver.gameObject);driver.Output=output;
    }
}
public sealed class StatisticsDriver : MonoBehaviour
{
    public string Output;
    readonly List<object> cases=new List<object>();
    readonly List<string> errors=new List<string>();
    int checks,failures,matches,pages;
    void Check(bool pass,string name,object detail=null){checks++;if(!pass)failures++;cases.Add(new{name,passed=pass,detail});}
    void Log(string text,string stack,LogType kind)
    {if(text.StartsWith("Error saving PlayerPrefs LastGoogleUpdate_I2Languages"))return;if(kind==LogType.Error||kind==LogType.Exception)errors.Add(text);}
    IEnumerator Start()
    {
        yield return SceneManager.LoadSceneAsync("GameScene");yield return null;Application.logMessageReceived+=Log;
        Exception failure=null;var run=Run();
        while(true) {
            bool more=false;object current=null;
            try{more=run.MoveNext();if(more)current=run.Current;}catch(Exception e){failure=e;}
            if(failure!=null||!more)break;yield return current;
        }
        File.WriteAllText(Output,JsonConvert.SerializeObject(new{passed=failure==null&&failures==0&&errors.Count==0,checks,failures,matches,pages,cases,nativeErrors=errors,error=failure==null?null:failure.ToString()},Formatting.Indented));
        Application.Quit(failure==null&&failures==0&&errors.Count==0?0:1);
    }
    sealed class Match
    {
        public BrassApplicationGame App;
        public BaseRepository Repository;
        public CompletedMatchStatistics Report;
        public IEvent[] Events;
    }
    Match Finish(int count,bool intro,int seed,bool passOnly=false)
    {
        var config=new GameStartConfiguration((NumberOfPlayers)count){Seed=seed,IsIntroductoryGame=intro};
        var metadata=new GameMetadata(config,GameType.Local);
        foreach(var start in config.Players) {
            var player=new PlayerMetadata{PlayerStartConfiguration=start,PlayerNetworkId="statistics-"+start.Color,NickName="Player "+start.Color,AiType=AIType.Hard};
            player.AIBehaviour=new HardAiBehavior(player.PlayerNetworkId);metadata.JoinGame(player);
        }
        var repository=new BaseRepository(new InMemoryGameStorage(),new InMemorySnapshotStorage(),new BrassBirminghamFactory());
        var app=repository.Create(metadata);app.Game.StartGame();int turns=0;
        while(!app.Game.GameProgressInformation.IsGameOver) {
            if(++turns>200)throw new Exception("Match did not finish");
            var behavior=app.Metadata.GetPlayerMetadataByColor(app.Game.GameProgressInformation.CurrentPlayer.Color).AIBehaviour;
            while(app.Game.TurnFlow.GetRemainingActionsCount()>0) {
                if(passOnly) {
                    var player=app.Game.GameProgressInformation.CurrentPlayer;app.Game.TurnFlow.StartPassAction().Pass(player.Hand[0],player);
                } else behavior.GetNextAIMove(app)();
                app.Game.TurnFlow.ConfirmCurrentAction();
            }
            app.Game.GameFlow.EndTurn();
            if(app.Game.GameFlow.IsLastTurnOver()) {
                if(app.Game.GameFlow.IsSellIndustryNecessary())
                    RuleSelectorFactory.CreatePrioritySelector(RuleSetFactory.CreateSellIndustryRuleSet()).Select(app).MakeAction(app,false)();
                app.Game.GameFlow.EndRound();
            }
        }
        var events=app.Game.GetUncommittedEvents().ToArray();
        string payload=JsonConvert.SerializeObject(events);int revision=app.Game.Revision;
        var bonus=DevelopMerchantBonus.Instance;
        var report=StatisticsCollector.Collect(app,new IEvent[0],events);matches++;
        Check(report.CompleteHistory,"complete match reconstruction",new{count,intro,seed,report.Availability,report.CollectionMilliseconds});
        Check(JsonConvert.SerializeObject(events)==payload&&app.Game.Revision==revision&&ReferenceEquals(bonus,DevelopMerchantBonus.Instance),"collection leaves live events, revision and bonus scope untouched");
        int actions=intro?(count==2?19:count==3?17:15):(count==2?39:count==3?35:31);
        foreach(var player in app.Game.PlayersTrack.Players) {
            var result=report.Players.First(p=>p.Color==player.Color);
            Check(result.FinalVP==player.VP&&result.FinalCash==player.Money&&result.FinalIncome==player.IncomeLevel,"final result reconciles",player.Color.ToString());
            Check(report.Actions.Count(a=>a.Player==player.Color)==actions,"confirmed action counts follow player-count/era rules",new{count,intro,color=player.Color.ToString(),actual=report.Actions.Count(a=>a.Player==player.Color),expected=actions});
            Check(result.OpeningCash+report.Entries.Where(e=>e.Player==player.Color&&e.Kind=="Balance").Sum(e=>e.Cash)==result.FinalCash,"cash reconciliation");
            Check(result.OpeningVP+report.Entries.Where(e=>e.Player==player.Color&&e.Kind=="Balance").Sum(e=>e.VP)==result.FinalVP,"VP reconciliation");
        }
        Check(report.Assets.Sum(a=>a.CanalVP+a.RailVP)==report.Entries.Where(e=>e.Kind=="Balance"&&e.Category=="Industries").Sum(e=>e.VP),"industry scoring reconciles");
        Check(report.Links.Sum(l=>l.VP)==report.Entries.Where(e=>e.Kind=="Balance"&&e.Category=="Links").Sum(e=>e.VP),"links reconcile including three-endpoint routes");
        foreach(var asset in report.Assets.Where(a=>a.Produced>0)) {
            int sold=report.Entries.Where(e=>e.Kind=="Market sale"&&e.AssetKey==asset.Key).Sum(e=>e.Quantity);
            Check(asset.Produced==asset.Supplied+sold+asset.Stock,"producer resource conservation",new{asset.Type,asset.Key,asset.Produced,asset.Supplied,sold,asset.Stock});
        }
        var filters=new StatisticsFilters();filters.Reset(report);
        for(int page=0;page<10;page++) {
            var sections=StatisticsPages.Build(page,report,filters);
            Check(sections.Count>0&&sections.All(s=>s.Rows.All(r=>r.Cells.Length==s.Columns.Length)),"page schema has complete usable ledgers",page);
        }
        var curve=StatisticsPages.Build(4,report,filters).First(s=>s.Chart!=null).Chart;
        Check(curve.Series.SelectMany(s=>s.Points).Max(p=>p.X)==report.Samples.Max(s=>s.Turn),"chart horizontal units are turns, not event revisions");
        var overlapping=StatisticsCollector.Collect(app,events.Take(events.Length/2),events.Skip(events.Length/3));
        Check(overlapping.CompleteHistory&&overlapping.Actions.Count==report.Actions.Count,"persisted/uncommitted overlap de-duplicates revisions");
        var gap=StatisticsCollector.Collect(app,events.Where(e=>e.Metadata.Revision!=10),new IEvent[0]);
        Check(!gap.CompleteHistory&&gap.Players.All(p=>p.FinalVP==report.Players.First(q=>q.Color==p.Color).FinalVP)&&gap.Actions.Count==0,"missing history retains final result without partial totals");
        var absent=StatisticsCollector.Collect(app,events.Skip(1),new IEvent[0]);
        Check(!absent.CompleteHistory&&StatisticsPages.Build(1,absent,filters).All(s=>s.Rows.Count==0),"missing setup is unavailable, never fabricated zero");
        var wrong=new BoardGameRules.Entities.Actions.Events.ActionPassed(null,PlayerColor.Red);wrong.Metadata.Revision=events[10].Metadata.Revision;
        var conflict=StatisticsCollector.Collect(app,events,new IEvent[]{wrong});
        Check(!conflict.CompleteHistory&&conflict.Availability.Contains("Conflicting events"),"conflicting overlapping revisions are rejected");
        if(passOnly) {
            Check(report.Players.All(p=>p.Rank==1)&&report.Actions.All(a=>a.Kind=="Pass"),"unresolved draw and Pass actions stay explicit");
            var table=StatisticsPages.Build(8,report,filters).First(s=>s.Title=="Conversion and efficiency");
            Check(table.Rows.All(r=>r.Cells[2]=="—"),"zero-VP denominators display unavailable ratio");
        }
        repository.Save(app);
        var history=new BrassGameHistory{Repository=repository};history.SetGame(app);
        Check(StatisticsCollector.Capture(history).CompleteHistory,"saved history plus current revision can be collected");
        return new Match{App=app,Repository=repository,Report=report,Events=events};
    }
    IEnumerator Run()
    {
        Match example=null;
        foreach(int count in new[]{2,3,4})foreach(bool intro in new[]{false,true}) {
            var match=Finish(count,intro,20260930+count+(intro?10:0));if(count==4&&!intro)example=match;yield return null;
        }
        foreach(int count in new[]{2,3,4}){Finish(count,false,20261010+count,true);yield return null;}
        if(example==null||!example.Report.CompleteHistory)throw new Exception("No complete report for native UI");
        File.WriteAllText(Path.Combine(Path.GetDirectoryName(Output),"statistics-example.json"),JsonConvert.SerializeObject(example.Report,Formatting.Indented));
        var viewFactory=Resources.FindObjectsOfTypeAll<UnityViewFactory>().First(f=>f.ViewList!=null&&f.ViewList.Views.OfType<Frameworks.View.Game.FinishedGameReview.FinishedGameReviewView>().Any());
        var root=new GameObject("Statistics test canvas",typeof(RectTransform),typeof(Canvas),typeof(GraphicRaycaster),typeof(CanvasScaler));
        root.GetComponent<Canvas>().renderMode=RenderMode.ScreenSpaceOverlay;
        root.GetComponent<CanvasScaler>().uiScaleMode=CanvasScaler.ScaleMode.ScaleWithScreenSize;root.GetComponent<CanvasScaler>().referenceResolution=new Vector2(1920,1080);
        var factory=root.AddComponent<UnityViewFactory>();factory.ParentTransform=root.transform;factory.ViewList=viewFactory.ViewList;
        var cameraNode=new GameObject("Statistics render camera",typeof(Camera));var camera=cameraNode.GetComponent<Camera>();
        camera.enabled=false;camera.orthographic=true;camera.clearFlags=CameraClearFlags.SolidColor;camera.backgroundColor=new Color(.065f,.06f,.05f);
        camera.transform.position=new Vector3(0,0,-100);camera.farClipPlane=1000;
        root.GetComponent<Canvas>().renderMode=RenderMode.ScreenSpaceCamera;root.GetComponent<Canvas>().worldCamera=camera;root.GetComponent<Canvas>().planeDistance=100;
        var input=new Input();var router=new Router(factory,input);var builder=new Builder();router.UseCaseBuilder=builder;Utils.Global.Instance.Router=router;
        var history=new BrassGameHistory{Repository=example.Repository};history.SetGame(example.App);
        var presenter=new Presenter(router);var owner=new FinishedGameReviewUseCase(presenter,history);builder.Owner=owner;
        owner.StartGameReview(example.App.Metadata.PlayerMetadata.First());
        Check(!root.GetComponentsInChildren<Button>(true).Any(b=>b.name=="Match statistics")&&!router.IsViewCached<StatisticsPanel>(),"entry remains hidden until final replay completes");
        owner.ReplayCompleted();
        Tick(1);yield return null;Canvas.ForceUpdateCanvases();
        var results=(EndOfGameView)router.GetView<IEndOfGameView>();
        var entry=results.transform.Find("SafeArea/Popup/Results statistics").GetComponent<Button>();
        Check(entry!=null,"statistics entry appears in original results");
        entry.onClick.Invoke();Tick(1);yield return null;
        var panel=router.GetView<StatisticsPanel>();Check(panel.IsOpen&&!router.IsViewVisible<IEndOfGameView>(),"statistics replaces the active result modal");
        var es=EventSystem.current;if(es==null)es=new GameObject("Statistics event system",typeof(EventSystem),typeof(StandaloneInputModule)).GetComponent<EventSystem>();
        foreach(var resolution in new[]{new Vector2Int(1280,720),new Vector2Int(1920,1080),new Vector2Int(2560,1440)}) {
            var target=new RenderTexture(resolution.x,resolution.y,24,RenderTextureFormat.ARGB32);target.Create();camera.targetTexture=target;
            Screen.SetResolution(resolution.x,resolution.y,false);yield return null;yield return null;yield return null;
            for(int page=0;page<10;page++) {
                panel.SelectPage(page);yield return null;yield return null;Canvas.ForceUpdateCanvases();
                Check(panel.PageButtons.Length==10&&panel.CurrentPage==page&&panel.BodyScroll.content.rect.height>=panel.BodyScroll.viewport.rect.height,"native page layout and scroll bounds",new{resolution=resolution.ToString(),page});
                Check(panel.GetComponentsInChildren<RectTransform>().Where(r=>r.GetComponent<Selectable>()!=null).All(r=>r.rect.width>0&&r.rect.height>0),"controls have positive geometry",new{resolution=resolution.ToString(),page});
                var close=panel.GetComponentsInChildren<Button>().First(b=>b.name=="Close statistics");
                var hits=new List<RaycastResult>();var data=new PointerEventData(es){position=RectTransformUtility.WorldToScreenPoint(camera,close.transform.TransformPoint(((RectTransform)close.transform).rect.center))};
                es.RaycastAll(data,hits);
                Check(hits.Count>0&&hits[0].gameObject.transform.IsChildOf(panel.transform),"statistics input shield wins raycast ordering");
                if(resolution.x==1920||page==0||page==4||page==8) {
                    Canvas.ForceUpdateCanvases();camera.Render();var previousTarget=RenderTexture.active;RenderTexture.active=target;
                    var image=new Texture2D(target.width,target.height,TextureFormat.RGB24,false);image.ReadPixels(new Rect(0,0,target.width,target.height),0,0);image.Apply();RenderTexture.active=previousTarget;
                    File.WriteAllBytes(Path.Combine(Path.GetDirectoryName(Output),"statistics-"+resolution.x+"-"+page+".png"),image.EncodeToPNG());Destroy(image);
                }
            }
            camera.targetTexture=null;target.Release();Destroy(target);
        }
        pages=10;
        panel.SelectPage(0);yield return null;yield return null;
        var wide=panel.GetComponentsInChildren<StatisticsTableScroll>().First();
        panel.BodyScroll.verticalNormalizedPosition=1;float sidewaysBefore=wide.horizontalNormalizedPosition;
        wide.OnScroll(new PointerEventData(es){scrollDelta=new Vector2(0,-3)});
        Check(panel.BodyScroll.verticalNormalizedPosition<.99f&&Math.Abs(wide.horizontalNormalizedPosition-sidewaysBefore)<.001f,"wheel over a wide table scrolls the page without horizontal drift");
        panel.BodyScroll.verticalNormalizedPosition=1;
        var start=RectTransformUtility.WorldToScreenPoint(camera,panel.BodyScroll.viewport.transform.position);
        var drag=new PointerEventData(es){position=start,pressPosition=start,button=PointerEventData.InputButton.Left};
        wide.OnInitializePotentialDrag(drag);wide.OnBeginDrag(drag);drag.position=start+new Vector2(0,100);drag.delta=new Vector2(0,100);wide.OnDrag(drag);wide.OnEndDrag(drag);
        Check(panel.BodyScroll.verticalNormalizedPosition<.99f,"vertical drag over a table scrolls the page");
        panel.ResetFilters();panel.Filters.Players.Remove(example.Report.Players[0].Color);panel.Filters.Era=1;panel.SelectPage(2);yield return null;
        Check(StatisticsPages.Build(2,example.Report,panel.Filters).SelectMany(s=>s.Rows).All(r=>!r.Player.HasValue||panel.Filters.Players.Contains(r.Player.Value)),"player subset excludes unwanted ledger rows");
        panel.Filters.Search="CoalMine";panel.SelectPage(2);yield return null;
        Check(StatisticsPages.Build(2,example.Report,panel.Filters).SelectMany(s=>s.Rows).All(r=>(String.Join(" ",r.Cells)+" "+r.Detail).IndexOf("CoalMine",StringComparison.OrdinalIgnoreCase)>=0),"ledger search includes details");
        panel.ResetFilters();panel.SelectPage(4);yield return null;yield return null;
        string secondName=panel.Report.Players[1].Name;
        ReflectionAccess.Set(panel.Report.Players[1],"Name",panel.Report.Players[0].Name);
        panel.SelectPage(4);yield return null;yield return null;
        panel.GetComponentsInChildren<Button>().First(b=>b.name.StartsWith("Legend ")).onClick.Invoke();yield return null;yield return null;
        Check(panel.GetComponentsInChildren<StatisticsChartGraphic>().First().Hidden.Count==1,"legend toggles use player identity even when names match");
        ReflectionAccess.Set(panel.Report.Players[1],"Name",secondName);
        panel.SelectPage(4);yield return null;yield return null;
        var row=panel.GetComponentsInChildren<Button>().First(b=>b.name.StartsWith("Row "));
        row.onClick.Invoke();yield return null;yield return null;
        Check(panel.GetComponentsInChildren<TextMeshProUGUI>().Any(t=>t.name.StartsWith("Expanded ")),"table row expands measured details");
        var sort=panel.GetComponentsInChildren<Button>().First(b=>b.name.StartsWith("Sort "));sort.onClick.Invoke();yield return null;
        Check(panel.GetComponentsInChildren<TextMeshProUGUI>().Any(t=>t.text.Contains(" ↑")),"column sorting is selectable");
        panel.BodyScroll.verticalNormalizedPosition=.37f;panel.SelectPage(9);yield return null;panel.SelectPage(4);yield return null;
        Check(Math.Abs(panel.BodyScroll.verticalNormalizedPosition-.37f)<.015f,"page scroll is retained");
        input.Held=KeyEnum.RightArrow;router.ProcessInput();Check(panel.IsOpen,"underlying board keys cannot escape modal");
        input.Down=KeyEnum.Return;router.ProcessInput();Check(panel.IsOpen,"return does not invoke old results review/exit");
        input.Down=KeyEnum.Escape;router.ProcessInput();Tick(1);yield return null;
        Check(!panel.IsOpen&&router.IsViewVisible<IEndOfGameView>(),"escape restores original results entry context");
        results=(EndOfGameView)router.GetView<IEndOfGameView>();results.transform.Find("SafeArea/Popup/Results statistics").GetComponent<Button>().onClick.Invoke();Tick(1);yield return null;
        panel.Close(false);Tick(1);yield return null;Check(!router.IsViewVisible<IEndOfGameView>(),"explicit board review returns to board");
        var review=(BaseView)router.GetView<IFinishedGameReviewView>();var boardButton=review.GetComponentsInChildren<Button>().First(b=>b.name=="Match statistics");
        for(int i=0;i<8;i++) {
            boardButton.onClick.Invoke();Tick(.1f);yield return null;panel.Close(false);Tick(.1f);yield return null;
        }
        Check(router.GetView<StatisticsPanel>()==panel&&root.GetComponentsInChildren<StatisticsPanel>(true).Length==1,"reopening reuses one panel and its state");
        Check(example.Repository.Load(example.App.Metadata.GameId)!=null,"panel closing never finishes review or deletes match");
        var staleClick=boardButton.onClick;owner.FinishGameReview();Tick(.1f);yield return null;
        Check(!router.IsViewCached<StatisticsPanel>()&&root.GetComponentsInChildren<StatisticsPanel>(true).Length==0,"review exit clears report UI and cached callbacks");
        staleClick.Invoke();Check(!router.IsViewCached<StatisticsPanel>(),"retired match callback cannot reopen a stale report");
        Destroy(root);Destroy(cameraNode);yield return null;
    }
    static void Tick(float seconds){for(float t=0;t<seconds;t+=.05f)DOTween.ManualUpdate(.05f,.05f);}
    sealed class Input:IInputSource {
        public KeyEnum Down,Held;public bool GetKeyDown(KeyEnum k){return k==Down;}public bool GetKey(KeyEnum k){return k==Held;}public bool GetKeyUp(KeyEnum k){return false;}
    }
    sealed class Builder:IUseCaseBuilder {
        public FinishedGameReviewUseCase Owner;
        public void BuildUseCases(){}public void RestoreUseCases(){}public void RestoreUseCaseOfType(Type t){}
        public IController GetController(Type t){if(typeof(IFinishedGameReviewView).IsAssignableFrom(t))return new FinishedGameReviewController(Owner);if(typeof(IEndOfGameView).IsAssignableFrom(t))return new EndOfGameController();return null;}
        public T GetUseCase<T>()where T:IUseCase{throw new NotSupportedException();}
    }
    sealed class Presenter:IFinishedGameReviewOutput {
        readonly Router _router;readonly FinishedGameReviewPresenter original;
        public Presenter(Router router){_router=router;original=new FinishedGameReviewPresenter(router);}
        public void ReplayUnseenActions(PlayerMetadata player,IReplayDelegate callback){}
        public void ShowSkipToMyNextTurnView(){}public void HidePlayerActionView(){}
        public void ShowButtonToLaunchGameSummaryTable(){original.ShowButtonToLaunchGameSummaryTable();}
        public void ShowGameSummaryTable(List<(PlayerColor,PlayerAvatar,string,int,int,int,int,int,int)> scores){original.ShowGameSummaryTable(scores);}
        public void ShowIntroductoryGameSummaryTable(List<(PlayerColor,PlayerAvatar,string,int,int,int,int,int,int,int,int)> scores){original.ShowIntroductoryGameSummaryTable(scores);}
    }
}
