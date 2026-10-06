using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using BoardGameRules.Entities.Common;
using BoardGameRules.Entities.Game;
using BoardGameRules.Entities.Players;
using BrassApplication.AI;
using BrassApplication.AI.AiBehaviors;
using BrassApplication.AI.RuleBasedSystem;
using BrassApplication.AI.RuleBasedSystem.RuleSelectors;
using BrassApplication.ApplicationSettings;
using BrassApplication.Game;
using BrassApplication.GameHistory;
using BrassApplication.UseCases.Common;
using BrassApplication.UseCases.Common.LoadScene;
using BrassApplication.UseCases.Game.Board;
using BrassPatch.History;
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
using InterfaceAdapters.Game.Board;
using InterfaceAdapters.Routing;
using InterfaceAdapters.Utils;
using Newtonsoft.Json;
using TMPro;
using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.SceneManagement;
using UnityEngine.UI;

public static class LocalAiUnityTests
{public static void Begin(string output){var driver=new GameObject("Statistics performance qualification").AddComponent<StatisticsPerformanceDriver>();UnityEngine.Object.DontDestroyOnLoad(driver.gameObject);driver.Output=output;}}
public sealed class StatisticsPerformanceDriver:MonoBehaviour
{
    public string Output;Camera probeCamera;
    readonly List<object> samples=new List<object>(),cases=new List<object>();readonly List<string> errors=new List<string>();int checks,failures;
    void Check(bool pass,string name,object detail=null){checks++;if(!pass)failures++;cases.Add(new{name,passed=pass,detail});}
    IEnumerator Start()
    {
        yield return SceneManager.LoadSceneAsync("GameScene");yield return null;
        Application.logMessageReceived+=(text,stack,type)=>{if((type==LogType.Error||type==LogType.Exception)&&!text.StartsWith("Error saving PlayerPrefs LastGoogleUpdate_I2Languages"))errors.Add(text);};
        Exception failure=null;var run=Run();while(true){bool more=false;object current=null;try{more=run.MoveNext();if(more)current=run.Current;}catch(Exception e){failure=e;}if(failure!=null||!more)break;yield return current;}
        File.WriteAllText(Output,JsonConvert.SerializeObject(new{passed=failure==null&&failures==0&&errors.Count==0,checks,failures,cases,nativeErrors=errors,error=failure==null?null:failure.ToString(),width=1920,height=1200,processor=SystemInfo.processorType,graphics=SystemInfo.graphicsDeviceName,definition="Matched completed four-player report, Large text, 1920x1200 native Unity board prefab. CPU frame intervals force Camera.Render at 1200p and include one frame yield, never PNG encoding. They are not displayed FPS or GPU-only timings. Batch-player fixture (board without player HUD), not a live-session FPS guarantee. History uses a codec-decoded clone and inactive board. Warm responses include UI build/layout and frame waits.",samples},Formatting.Indented));Application.Quit(failure==null&&failures==0&&errors.Count==0?0:1);
    }
    static double Percentile(IEnumerable<double> data,double p){var values=data.OrderBy(x=>x).ToArray();return values[(int)Math.Ceiling((values.Length-1)*p)];}
    IEnumerator Frames(StatisticsPanel panel,string mode,string phase,int count)
    {
        var values=new List<double>();var graph=panel.GetComponentsInChildren<StatisticsPlotGraphic>().FirstOrDefault();
        for(int i=0;i<count;i++) {
            var timer=System.Diagnostics.Stopwatch.StartNew();
            if(phase=="scroll")panel.BodyScroll.verticalNormalizedPosition=1f-(i%40)/39f;
            if(phase=="hover"&&graph!=null){var mark=graph.Plot.Marks.FirstOrDefault();if(mark!=null)panel.Tooltip(i%2==0?graph.Description(mark):null);}
            probeCamera.Render();yield return null;values.Add(timer.Elapsed.TotalMilliseconds);
        }
        samples.Add(new{mode,operation=phase,frames=count,medianMs=Percentile(values,.5),p95Ms=Percentile(values,.95),p99Ms=Percentile(values,.99)});
        panel.Tooltip(null);
    }
    IEnumerator Run()
    {
        QualitySettings.vSyncCount=0;Application.targetFrameRate=-1;Screen.SetResolution(1920,1200,false);
        var config=new GameStartConfiguration((NumberOfPlayers)4){Seed=20260934,IsIntroductoryGame=false};var metadata=new GameMetadata(config,GameType.Local);
        foreach(var start in config.Players){var p=new PlayerMetadata{PlayerStartConfiguration=start,PlayerNetworkId="statistics-"+start.Color,NickName="Player "+start.Color,AiType=AIType.Hard};p.AIBehaviour=new HardAiBehavior(p.PlayerNetworkId);metadata.JoinGame(p);}
        var repository=new BaseRepository(new InMemoryGameStorage(),new InMemorySnapshotStorage(),new BrassBirminghamFactory());var app=repository.Create(metadata);app.Game.StartGame();int turns=0;
        while(!app.Game.GameProgressInformation.IsGameOver){if(++turns>200)throw new Exception("Benchmark match did not finish");var ai=app.Metadata.GetPlayerMetadataByColor(app.Game.GameProgressInformation.CurrentPlayer.Color).AIBehaviour;while(app.Game.TurnFlow.GetRemainingActionsCount()>0){ai.GetNextAIMove(app)();app.Game.TurnFlow.ConfirmCurrentAction();}app.Game.GameFlow.EndTurn();if(app.Game.GameFlow.IsLastTurnOver()){if(app.Game.GameFlow.IsSellIndustryNecessary())RuleSelectorFactory.CreatePrioritySelector(RuleSetFactory.CreateSellIndustryRuleSet()).Select(app).MakeAction(app,false)();app.Game.GameFlow.EndRound();}}
        var report=StatisticsCollector.Collect(app,new BoardGameRules.Entities.EventSourcing.Events.IEvent[0],app.Game.GetUncommittedEvents().ToArray());var encoded=GameHistoryStatisticsCodec.Encode(report);Check(report.CompleteHistory,"E26-PERFORMANCE completed four-player report",report.Assets.Count);
        var template=Resources.FindObjectsOfTypeAll<UnityViewFactory>().First(f=>f.ViewList!=null&&f.ViewList.Views.OfType<BoardView>().Any());
        var root=new GameObject("Benchmark canvas",typeof(RectTransform),typeof(Canvas),typeof(CanvasScaler),typeof(GraphicRaycaster));var canvas=root.GetComponent<Canvas>();root.GetComponent<CanvasScaler>().uiScaleMode=CanvasScaler.ScaleMode.ScaleWithScreenSize;root.GetComponent<CanvasScaler>().referenceResolution=new Vector2(1920,1200);
        var cameraObject=new GameObject("Benchmark camera",typeof(Camera));var camera=cameraObject.GetComponent<Camera>();probeCamera=camera;camera.enabled=false;camera.orthographic=true;camera.clearFlags=CameraClearFlags.SolidColor;camera.backgroundColor=new Color(.07f,.07f,.06f);camera.transform.position=new Vector3(0,0,-100);camera.farClipPlane=1000;camera.targetTexture=new RenderTexture(1920,1200,24);camera.targetTexture.Create();canvas.renderMode=RenderMode.ScreenSpaceCamera;canvas.worldCamera=camera;canvas.planeDistance=100;
        var factory=root.AddComponent<UnityViewFactory>();factory.ParentTransform=root.transform;factory.ViewList=template.ViewList;var router=new Router(factory,new InputSource());Utils.Global.Instance.Router=router;var history=new BrassGameHistory{Repository=repository};history.SetGame(app);var builder=new Builder(router,history);router.UseCaseBuilder=builder;
        var board=router.GetView<IBoardView>() as BoardView;board.InputRegistration=router;board.UpdateBoardView(builder.Presenter.Model(app));board.CreateShowViewImmediatelyAnimation();yield return null;DOTween.CompleteAll();yield return null;
        if(EventSystem.current==null)new GameObject("Benchmark events",typeof(EventSystem),typeof(StandaloneInputModule));
        var font=Resources.FindObjectsOfTypeAll<TMP_FontAsset>().First(f=>f.name=="GARA SDF");var leaseType=typeof(StatisticsPanel).Assembly.GetType("BrassPatch.Statistics.StatisticsPresentationLease");
        foreach(string mode in leaseType==null?new[]{"baseline-live","history"}:new[]{"live-unsuspended-control","live-suspended","history"}) {
            var panelObject=new GameObject("Benchmark Statistics",typeof(RectTransform),typeof(Canvas),typeof(GraphicRaycaster),typeof(CanvasGroup));panelObject.transform.SetParent(root.transform,false);var rect=(RectTransform)panelObject.transform;rect.anchorMin=Vector2.zero;rect.anchorMax=Vector2.one;rect.offsetMin=rect.offsetMax=Vector2.zero;panelObject.GetComponent<Canvas>().overrideSorting=true;panelObject.GetComponent<Canvas>().sortingOrder=500;panelObject.GetComponent<CanvasGroup>().ignoreParentGroups=true;
            var panel=panelObject.AddComponent<StatisticsPanel>();panel.InputRegistration=router;var value=mode=="history"?GameHistoryStatisticsCodec.Decode(encoded):report;
            Check(GameHistoryStatisticsCodec.Encode(value).ToString()==encoded.ToString(),"E26-PERFORMANCE identical report across modes",mode);
            board.gameObject.SetActive(mode!="history");IDisposable lease=null;
            if(mode=="live-suspended")lease=(IDisposable)Activator.CreateInstance(leaseType,new object[]{root.transform,panel.transform});
            var first=System.Diagnostics.Stopwatch.StartNew();panel.Initialize(value,font,x=>{},mode=="history"?StatisticsOpeningContext.ArchivedHistory:StatisticsOpeningContext.LiveCompletedGame,null);panel.CreateShowViewAnimation();yield return null;yield return null;samples.Add(new{mode,operation="cold-open",milliseconds=first.Elapsed.TotalMilliseconds});
            panel.SelectPage(8);yield return null;yield return null;
            var stableGraph=panel.GetComponentsInChildren<StatisticsPlotGraphic>().First();
            foreach(string phase in new[]{"idle","scroll","hover"}){var run=Frames(panel,mode,phase,100);while(run.MoveNext())yield return run.Current;}
            Check(panel.GetComponentsInChildren<StatisticsPlotGraphic>().First()==stableGraph,"E26-PERFORMANCE idle, scroll and hover retain chart objects",mode);
            var responses=new List<double>();for(int i=0;i<12;i++){var timer=System.Diagnostics.Stopwatch.StartNew();panel.SelectPage(i%2==0?0:8);yield return null;yield return null;responses.Add(timer.Elapsed.TotalMilliseconds);}samples.Add(new{mode,operation="warm-page",medianMs=Percentile(responses,.5),p95Ms=Percentile(responses,.95),p99Ms=Percentile(responses,.99)});
            foreach(string operation in new[]{"warm-filter","warm-tab","text-size"}){
                var changeTab=typeof(StatisticsPanel).GetMethod("SetEfficiencyView");if(operation=="warm-tab"&&changeTab==null)continue;
                responses.Clear();for(int i=0;i<6;i++){var timer=System.Diagnostics.Stopwatch.StartNew();
                    if(operation=="warm-filter"){panel.Filters.Era=i%3;panel.SelectPage(8);}
                    else if(operation=="warm-tab")changeTab.Invoke(panel,new object[]{new[]{"Returns","Conversion","Transition & networks"}[i%3]});
                    else panel.GetComponentsInChildren<Button>().First(b=>b.name==(i%2==0?"Larger text":"Smaller text")).onClick.Invoke();
                    yield return null;yield return null;responses.Add(timer.Elapsed.TotalMilliseconds);
                }samples.Add(new{mode,operation,medianMs=Percentile(responses,.5),p95Ms=Percentile(responses,.95),p99Ms=Percentile(responses,.99)});
                panel.ResetFilters();panel.SelectPage(8);yield return null;yield return null;
            }
            Check(panel.IsOpen&&panel.GetComponentsInChildren<Button>().Any(b=>b.IsInteractable()),"E26-PERFORMANCE panel remains interactive",mode);
            panelObject.SetActive(false);if(lease!=null)lease.Dispose();Destroy(panelObject);yield return null;
        }
        Destroy(root);Destroy(camera.targetTexture);Destroy(cameraObject);yield return null;
    }
    sealed class InputSource:IInputSource{public bool GetKeyDown(KeyEnum k){return false;}public bool GetKey(KeyEnum k){return false;}public bool GetKeyUp(KeyEnum k){return false;}}
    sealed class Localization:ILocalization{public string GetTranslation(string key){return key.Split('/').Last();}}
    sealed class Settings:IGameSettings{public bool CameFromMenu(){return false;}public void SetMap(bool v){}public bool GetIsNightMap(){return false;}public void SetDefaultGameSettings(GameMetadata m,int n){}public int GetNumberOfPlayers(){return 4;}public bool IsSavedDefaultGameSettings(){return false;}public PlayerColor GetPlayerIndexSavedColor(int i){return (PlayerColor)i;}public PlayerAvatar GetPlayerIndexSavedAvatar(int i){return (PlayerAvatar)i;}public AIType GetPlayerIndexSavedAIType(int i){return AIType.Human;}}
    sealed class Presenter:BoardPresenter{public Presenter(IRouter r):base(r,new Localization()){}public BoardViewModel Model(BrassApplicationGame app){return PrepareBoardViewModel(app.Game.Board,app.Game.GameProgressInformation.CurrentGameEra,null);}}
    sealed class LoadPresenter:ILoadSceneOutput{public void DisplayTip(int i,bool l){}public void HideView(){}public bool GetWasLastTipNewGlobal(){return false;}public int GetLastTipIdGlobal(){return 1;}}
    sealed class Builder:IUseCaseBuilder{
        public readonly Presenter Presenter;readonly BoardUseCase board;readonly LoadSceneUseCase load=new LoadSceneUseCase(new LoadPresenter());
        public Builder(IRouter r,IGameHistory h){Presenter=new Presenter(r);board=new BoardUseCase(Presenter,h,new Settings());board.SetAllowBuidlingFromBoard(false);}
        public void BuildUseCases(){}public void RestoreUseCases(){}public void RestoreUseCaseOfType(Type t){}public IController GetController(Type t){return typeof(IBoardView).IsAssignableFrom(t)?new BoardController(board):null;}
        public T GetUseCase<T>()where T:IUseCase{if(typeof(T)==typeof(LoadSceneUseCase))return (T)(object)load;if(typeof(T)==typeof(BoardUseCase))return (T)(object)board;throw new NotSupportedException(typeof(T).Name);}
    }
}
