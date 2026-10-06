using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using BoardGameRules.Entities.Actions;
using BoardGameRules.Entities.Board.BuildingSlots;
using BoardGameRules.Entities.Board.Regions;
using BoardGameRules.Entities.Cards;
using BoardGameRules.Entities.Common;
using BoardGameRules.Entities.EventSourcing.Events;
using BoardGameRules.Entities.Game;
using BoardGameRules.Entities.GamePhases;
using BoardGameRules.Entities.Players;
using BrassApplication.AI;
using BrassApplication.ApplicationSettings;
using BrassApplication.Game;
using BrassApplication.GameHistory;
using BrassApplication.UseCases.Common;
using BrassApplication.UseCases.Common.LoadScene;
using BrassApplication.UseCases.Game.Board;
using BrassApplication.UseCases.Game.IndustriesSummary;
using BrassApplication.UseCases.Game.Market;
using BrassApplication.UseCases.Game.HandOfCards;
using BrassApplication.UseCases.Game.Replay;
using BrassPatch;
using BrassPatch.CardTracker;
using DG.Tweening;
using Frameworks.Persistence.GameFactory;
using Frameworks.Persistence.Repository;
using Frameworks.Persistence.RepositoryBackends.InMemory;
using Frameworks.Routing;
using Frameworks.Routing.Unity;
using Frameworks.Utils;
using Frameworks.View.Common;
using Frameworks.View.Game;
using InterfaceAdapters.Common;
using InterfaceAdapters.Game.Board;
using InterfaceAdapters.Game.IndustriesSummary;
using InterfaceAdapters.Game.Market;
using InterfaceAdapters.Game.HandOfCards;
using InterfaceAdapters.Game.PlayerActions;
using InterfaceAdapters.Game.PlayAiTurn;
using InterfaceAdapters.Routing;
using InterfaceAdapters.Utils;
using Newtonsoft.Json;
using TMPro;
using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.SceneManagement;
using UnityEngine.UI;

public static class LocalAiUnityTests
{
    public static void Begin(string output)
    { var driver=new GameObject("Card Tracker qualification").AddComponent<CardTrackerTestDriver>();UnityEngine.Object.DontDestroyOnLoad(driver.gameObject);driver.Output=output; }
}
public sealed class CardTrackerTestDriver : MonoBehaviour
{
    public string Output;
    int checks,failures;readonly List<object> cases=new List<object>();readonly List<string> errors=new List<string>();
    readonly Vector2Int[] resolutions={new Vector2Int(1920,1200),new Vector2Int(1920,1080)};
    void Check(bool pass,string name,object detail=null){checks++;if(!pass)failures++;cases.Add(new{name,passed=pass,detail});}
    void Log(string text,string stack,LogType kind){if(text.StartsWith("Error saving PlayerPrefs LastGoogleUpdate_I2Languages"))return;if(kind==LogType.Error||kind==LogType.Exception)errors.Add(text);}
    IEnumerator Start()
    {
        yield return SceneManager.LoadSceneAsync("GameScene");yield return null;Application.logMessageReceived+=Log;
        Exception failure=null;var run=Run();
        while(true){bool next=false;object current=null;try{next=run.MoveNext();if(next)current=run.Current;}catch(Exception e){failure=e;}if(failure!=null||!next)break;yield return current;}
        Time.timeScale=1;DOTween.timeScale=1;
        try{File.WriteAllText(Output,JsonConvert.SerializeObject(new{passed=failure==null&&failures==0&&errors.Count==0,checks,failures,cases,layouts=resolutions.Select(r=>new{width=r.x,height=r.y}).ToArray(),nativeErrors=errors,error=failure==null?null:failure.ToString()},Formatting.Indented));}
        catch(Exception reportError){failure=reportError;File.WriteAllText(Output,JsonConvert.SerializeObject(new{passed=false,checks,failures,error=reportError.ToString()},Formatting.Indented));}
        Application.Quit(failure==null&&failures==0&&errors.Count==0?0:1);
    }
    sealed class Match
    {
        public BrassApplicationGame App;public BaseRepository Repository;public BrassGameHistory History;
        public IEvent[] Events(){return Repository.Advanced.GameStorage.LoadEvents(App.Metadata.GameId,0,App.Game.Revision).Concat(App.Game.GetUncommittedEvents()).GroupBy(e=>e.Metadata.Revision).Select(g=>g.First()).OrderBy(e=>e.Metadata.Revision).ToArray();}
        public CardTrackerProjection Projection(){return CardTrackerProjection.Capture(History);}
    }
    Match NewMatch(int count,bool intro=false,int seed=20261001)
    {
        var config=new GameStartConfiguration((NumberOfPlayers)count){Seed=seed,IsIntroductoryGame=intro};
        var metadata=new GameMetadata(config,GameType.Local);
        foreach(var p in config.Players)metadata.JoinGame(new PlayerMetadata{PlayerStartConfiguration=p,PlayerNetworkId="tracker-"+p.Color,NickName="Player "+p.Color,AiType=AIType.Human});
        var repository=new BaseRepository(new InMemoryGameStorage(),new InMemorySnapshotStorage(),new BrassBirminghamFactory());
        var app=repository.Create(metadata);app.Game.StartGame();
        var history=new BrassGameHistory{Repository=repository};history.SetGame(app);
        return new Match{App=app,Repository=repository,History=history};
    }
    void EndTurn(Match match)
    {
        var game=match.App.Game;game.GameFlow.EndTurn();
        if(game.GameFlow.IsLastTurnOver())game.GameFlow.EndRound();
    }
    void Pass(Match match,BaseCard card=null)
    {
        var p=match.App.Game.GameProgressInformation.CurrentPlayer;
        match.App.Game.TurnFlow.StartPassAction().Pass(card??p.Hand[0],p);match.App.Game.TurnFlow.ConfirmCurrentAction();
    }
    void Scout(Match match)
    {
        var p=match.App.Game.GameProgressInformation.CurrentPlayer;var action=match.App.Game.TurnFlow.StartScoutAction();
        action.DiscardCard(p.Hand[0],p);action.Scout(p.Hand.Take(2).ToList(),p);match.App.Game.TurnFlow.ConfirmCurrentAction();
    }
    void ProjectionChecks()
    {
        foreach(bool intro in new[]{false,true}) {
            var match=NewMatch(4,intro,intro?20261101:20261001);var game=match.App.Game;
            var all=game.Board.CardDeck.Concat(game.PlayersTrack.Players.SelectMany(p=>p.Hand.Concat(p.DiscardPile))).ToArray();
            var initial=match.Projection();
            Check(initial.Complete&&initial.Played==0&&initial.Unseen==64,"initial public count includes hidden setup cards",intro);
            Check(all.Length==64&&all.Select(c=>c.UniqueId).Distinct().Count()==64&&game.Board.CardDeck.Count==28&&game.PlayersTrack.Players.All(p=>p.Hand.Count==8&&p.DiscardPile.Count==1),"actual four-player setup is 32 hands / 28 deck / 4 hidden",intro);
            foreach(var card in initial.Cards) {
                int total=all.Count(c=>card.City.HasValue?c is LocationCard&&((LocationCard)c).Region==card.City.Value:
                    c is IndustryCard&&(card.Key.EndsWith("CottonMill+Manufacturer")?((IndustryCard)c).BuildingTypes.Count==2:((IndustryCard)c).BuildingTypes.Count==1&&((IndustryCard)c).BuildingTypes[0].ToString()==card.Key.Substring(9)));
                Check(total==card.Total,"manifest matches actual physical card family",new{intro,card.Name,total});
            }
            var railStarts=new Dictionary<PlayerColor,int>();int turns=0;
            while(!game.GameProgressInformation.IsGameOver) {
                if(++turns>120)throw new Exception("Pass fixture did not end.");
                while(game.TurnFlow.GetRemainingActionsCount()>0)Pass(match);
                var projection=match.Projection();
                foreach(var p in game.PlayersTrack.Players)if(!railStarts.ContainsKey(p.Color))railStarts[p.Color]=0;
                var publicCards=game.PlayersTrack.Players.SelectMany(p=>p.DiscardPile.Skip(projection.Rail?railStarts[p.Color]:1)).ToArray();
                Check(projection.Complete&&projection.Played==publicCards.Length&&projection.Cards.All(c=>c.Played>=0&&c.Unseen>=0),"projection reconciles with independent public discard-pile oracle",new{intro,turns,projection.Rail});
                bool wasRail=projection.Rail;
                EndTurn(match);
                if(!wasRail&&game.GameProgressInformation.CurrentGameEra==GameEra.RailEra) {
                    foreach(var p in game.PlayersTrack.Players)railStarts[p.Color]=p.DiscardPile.Count;
                    var start=match.Projection();Check(start.Complete&&start.Rail&&start.Played==0&&start.Unseen==64,"Rail resets counts despite retained Canal discard aliases");
                }
            }
            var ended=match.Projection();Check(ended.Complete&&ended.Ended&&ended.Played==(intro?60:64),"ordinary era exhaustion reconciles",intro);
            var events=match.Events();var colors=game.PlayersTrack.Players.Select(p=>p.Color).ToArray();
            var duplicate=CardTrackerProjection.Build(events,events,game.Revision,colors);Check(duplicate.Complete&&duplicate.Played==ended.Played,"duplicate persisted/uncommitted delivery is counted once");
            var missing=CardTrackerProjection.Build(events.Skip(1),new IEvent[0],game.Revision,colors);Check(!missing.Complete,"missing setup produces unknown counts");
            var gap=CardTrackerProjection.Build(events.Where((e,i)=>i!=7),new IEvent[0],game.Revision,colors);Check(!gap.Complete,"history gap produces unknown counts");
            int before=events.First(e=>e is BoardGameRules.Entities.Actions.Events.CardDiscarded).Metadata.Revision-1;
            var prior=CardTrackerProjection.Build(events,new IEvent[0],before,colors);Check(prior.Complete&&prior.Played==0,"revision cap excludes future discards and supports rollback");
            match.Repository.Save(match.App);var loaded=match.Repository.Load(match.App.Metadata.GameId);
            match.History.SetGame(loaded);var resumed=match.Projection();
            Check(resumed.Complete&&resumed.Played==ended.Played&&resumed.Rail==ended.Rail,"persisted continue reconstructs same public counts");
        }
        var scout=NewMatch(4);
        for(int i=0;i<4;i++){Scout(scout);EndTurn(scout);}
        var scouted=scout.Projection();Check(scouted.Complete&&scouted.Played==12&&scouted.Wild.Values.All(w=>w[0]&&w[1]),"four Scouts account for twelve ordinary cards and eight public wild holders");
        var actor=scout.App.Game.GameProgressInformation.CurrentPlayer;int ordinary=scouted.Played;
        Pass(scout,actor.Hand.First(c=>c is WildLocationCard));var once=scout.Projection();
        Check(once.Complete&&!once.Wild[actor.Color][0]&&once.Wild[actor.Color][1]&&once.Played==ordinary,"wild Location spend updates only matching holder");
        Pass(scout,actor.Hand.First(c=>c is WildIndustryCard));var twice=scout.Projection();
        Check(twice.Complete&&!twice.Wild[actor.Color][0]&&!twice.Wild[actor.Color][1]&&twice.Played==ordinary,"wild Industry spend does not alter ordinary counts");
        var payload=JsonConvert.SerializeObject(scout.Events());var a=scout.Projection();
        var opponent=scout.App.Game.PlayersTrack.Players.First(p=>p.Color!=actor.Color);
        var hidden=opponent.DiscardPile[0];opponent.DiscardPile[0]=scout.App.Game.Board.CardDeck[0];scout.App.Game.Board.CardDeck[0]=hidden;
        opponent.Hand.Reverse();scout.App.Game.Board.CardDeck.Reverse();
        var b=scout.Projection();
        Check(a.Complete&&b.Complete&&a.Cards.Select(c=>c.Played).SequenceEqual(b.Cards.Select(c=>c.Played)),"hidden setup cards, opponent hands and draw order cannot influence public output");
        Check(JsonConvert.SerializeObject(scout.Events())==payload,"tracker capture does not mutate live events");
        Check(FourPlayerCardManifest.IndustryKey(BuildingType.CottonMill)==FourPlayerCardManifest.IndustryKey(BuildingType.Manufacturer)&&a.Cards.Sum(c=>c.Total)==64,"dual-purpose industry cards share one eight-card pool");
    }
    IEnumerator Run()
    {
        ProjectionChecks();yield return null;
        var templates=Resources.FindObjectsOfTypeAll<UnityViewFactory>().First(f=>f.ViewList!=null&&f.ViewList.Views.OfType<BoardView>().Any()).ViewList;
        var match=NewMatch(4);for(int i=0;i<4;i++){Scout(match);EndTurn(match);}
        var root=new GameObject("Tracker test canvas",typeof(RectTransform),typeof(Canvas),typeof(GraphicRaycaster),typeof(CanvasScaler));
        var canvas=root.GetComponent<Canvas>();var scaler=root.GetComponent<CanvasScaler>();scaler.uiScaleMode=CanvasScaler.ScaleMode.ScaleWithScreenSize;scaler.referenceResolution=new Vector2(1920,1080);
        var cameraNode=new GameObject("Tracker render camera",typeof(Camera));var camera=cameraNode.GetComponent<Camera>();camera.enabled=false;camera.orthographic=true;camera.clearFlags=CameraClearFlags.SolidColor;
        camera.backgroundColor=new Color(.09f,.085f,.075f);camera.transform.position=new Vector3(0,0,-100);camera.farClipPlane=1000;
        canvas.renderMode=RenderMode.ScreenSpaceCamera;canvas.worldCamera=camera;canvas.planeDistance=100;
        var factory=root.AddComponent<UnityViewFactory>();factory.ParentTransform=root.transform;factory.ViewList=templates;
        var input=new Input();var router=new Router(factory,input);var builder=new Builder(router,match.History);router.UseCaseBuilder=builder;Utils.Global.Instance.Router=router;
        var board=router.GetView<IBoardView>() as BoardView;board.InputRegistration=router;
        var market=router.GetView<IMarketView>() as MarketView;market.InputRegistration=router;
        var industries=router.GetView<IIndustriesSummaryView>() as IndustriesSummaryView;industries.InputRegistration=router;
        var hand=router.GetView<IHandOfCardsView>() as HandOfCardsView;hand.InputRegistration=router;
        var actions=router.GetView<IPlayerActionsView>() as BaseView;actions.InputRegistration=router;
        hand.UpdateHandCardsView(match.App.Game.GameProgressInformation.CurrentPlayer.Hand,false);
        board.UpdateBoardView(builder.BoardPresenter.Model(match.App));
        builder.Market.UpdateMarket();builder.Industries.UpdateIndustries();
        board.CreateShowViewImmediatelyAnimation();market.CreateShowViewImmediatelyAnimation();industries.CreateShowViewImmediatelyAnimation();hand.CreateShowViewImmediatelyAnimation();hand.ShowCards();
        actions.CreateShowViewImmediatelyAnimation();
        hand.EnableCardsLayoutGroup();
        yield return null;DOTween.CompleteAll();Canvas.ForceUpdateCanvases();
        LayoutRebuilder.ForceRebuildLayoutImmediate((RectTransform)hand.transform.Find("SafeArea/Popup/CardsContainer"));
        builder.Board.UpdateBoard();builder.Board.UpdateBoard();yield return null;
        Check(board.GetComponentsInChildren<CardTrackerSession>(true).Length==1,"woven BoardPresenter hook attaches exactly one tracker owner");
        var session=CardTrackerEntryPoints.Attach(router,match.History,board);
        Check(session!=null&&session.Eligible(),"tracker attaches to eligible human gameplay");
        var metadata=match.App.Metadata.GetPlayerMetadataByColor(match.App.Game.GameProgressInformation.CurrentPlayer.Color);
        metadata.AiType=AIType.Hard;session.Refresh();Check(!session.Toggle.interactable&&!session.Open(),"AI turn disables tracker and rejects stale/direct clicks");metadata.AiType=AIType.Human;
        ReflectionAccess.Set(builder.Replay,"_isReplayInProgress",true);Check(!session.Eligible(),"replay disables tracker");ReflectionAccess.Set(builder.Replay,"_isReplayInProgress",false);
        float animationValue=0;var scoring=DOTween.To(()=>animationValue,v=>animationValue=v,1,1).SetId("AnimationFactory");
        session.Refresh();Check(!session.Toggle.interactable&&!session.Open(),"turn/scoring animation disables tracker even for a human");scoring.Kill();
        var pending=match.App.Game.TurnFlow.StartPassAction();Check(!session.Eligible(),"in-progress human action disables tracker");match.History.CancelAction();match.App=match.History.CurrentGame;
        foreach(var resolution in resolutions) {
            var target=new RenderTexture(resolution.x,resolution.y,24,RenderTextureFormat.ARGB32);target.Create();camera.targetTexture=target;yield return null;Canvas.ForceUpdateCanvases();
            session.Overlay.Layout();Canvas.ForceUpdateCanvases();
            var fixedButton=ScreenBounds((RectTransform)session.Toggle.transform,camera);
            var iron=(RectTransform)market.transform.Find("SafeArea/Popup/Markets/IronMarket");
            var ironPosition=iron.anchoredPosition;
            metadata=match.App.Metadata.GetPlayerMetadataByColor(match.App.Game.GameProgressInformation.CurrentPlayer.Color);
            metadata.AiType=AIType.Hard;session.Refresh();
            var thinking=router.GetView<IPlayAiTurnView>() as PlayAiTurnView;thinking.InputRegistration=router;
            thinking.SetCurrentPlayerName("Worcester Iron-Cotton AI");thinking.CreateShowViewImmediatelyAnimation();
            var strategyStatus=BrassLegacyAI.AiTurnControls.Attach(router,"Worcester Iron-Cotton");
            yield return null;Canvas.ForceUpdateCanvases();
            var thinkingTexts=thinking.GetComponentsInChildren<TextMeshProUGUI>().Where(t=>!String.IsNullOrWhiteSpace(t.text)).ToArray();
            var thinkingBounds=thinkingTexts.Select(t=>ScreenBounds(t.rectTransform,camera)).ToArray();
            Check(thinkingTexts.Any(t=>t.name=="Strategy text")&&thinkingBounds.All(b=>!b.Overlaps(fixedButton)&&b.yMax+12<=fixedButton.yMin),"CT25-POSITION raised tracker clears native AI-thinking text",new{resolution.x,resolution.y,button=fixedButton.ToString(),labels=thinkingTexts.Select((t,i)=>new{t.name,t.text,bounds=thinkingBounds[i].ToString()}).ToArray()});
            Check(!session.Toggle.interactable&&!session.Open(),"CT25-INPUT tracker remains disabled throughout AI-thinking presentation");
            Capture(camera,target,"tracker-ai-thinking-"+resolution.x+"x"+resolution.y);
            var marketAnimation=iron.DOAnchorPos(ironPosition+new Vector2(180,180),.15f).SetId("AnimationFactory");
            yield return new WaitForSeconds(.08f);Canvas.ForceUpdateCanvases();
            Check(SameBounds(fixedButton,ScreenBounds((RectTransform)session.Toggle.transform,camera))&&!session.Toggle.interactable&&!session.Open(),"tracker stays fixed during another player's market animation",new{resolution.x,before=fixedButton.ToString(),after=ScreenBounds((RectTransform)session.Toggle.transform,camera).ToString(),session.Toggle.interactable,metadata.AiType});
            marketAnimation.Complete();iron.anchoredPosition=ironPosition;thinking.CreateHideViewAnimation();DOTween.CompleteAll();thinking.gameObject.SetActive(false);Destroy(strategyStatus.gameObject);metadata.AiType=AIType.Human;session.Refresh();
            yield return null;Canvas.ForceUpdateCanvases();Capture(camera,target,"tracker-button-placement-"+resolution.x+"x"+resolution.y);
            foreach(bool night in new[]{false,true}) {
                builder.Settings.Night=night;typeof(BoardView).GetMethod("SetMapDayNight",ReflectionAccess.Flags).Invoke(board,null);
                Time.timeScale=1.4f;DOTween.timeScale=.8f;var zoom=ReflectionAccess.Get<ZoomScrollRect>(board,"_zoomScrollRect");
                var oldPosition=zoom.content.anchoredPosition;var oldScale=zoom.content.localScale;bool oldEnabled=zoom.enabled;
                int revision=match.App.Game.Revision;var before=JsonConvert.SerializeObject(match.Events());
                Check(session.Open()&&Time.timeScale==0&&DOTween.timeScale==0,"opening freezes gameplay and all DOTween clocks",new{resolution.x,night});
                yield return null;yield return null;Canvas.ForceUpdateCanvases();
                Check(session.Overlay.CityBadges==20&&session.Overlay.IndustryBadges==6,"all twenty cities and six industry rows have counters");
                foreach(var control in session.GetComponentsInChildren<RectTransform>().Where(t=>t.name=="Tracker header"||t.name=="Wild cards strip"||t.name=="Card Tracker toggle"||t.name.StartsWith("Wild holder ")&&t.name!="Wild holder name"||t.name.StartsWith("City cards ")||t.name.StartsWith("Industry cards "))) {
                    var bounds=ScreenBounds(control,camera);
                    Check(bounds.xMin>=-1&&bounds.yMin>=-1&&bounds.xMax<=resolution.x+1&&bounds.yMax<=resolution.y+1,"tracker control stays inside display",new{control.name,resolution.x,bounds.xMin,bounds.xMax,bounds.yMin,bounds.yMax});
                }
                var headerBounds=ScreenBounds(session.GetComponentsInChildren<RectTransform>().Single(t=>t.name=="Tracker header"),camera);
                var wildBounds=ScreenBounds(session.GetComponentsInChildren<RectTransform>().Single(t=>t.name=="Wild cards strip"),camera);
                Check(headerBounds.height<=53&&headerBounds.width<=1261,"tracker toolbar uses compact physical dimensions",new{resolution.x,headerBounds.width,headerBounds.height});
                Check(headerBounds.height+wildBounds.height+6<=140,"default tracker grouping reduces the solid header footprint",new{resolution.x,height=headerBounds.height+wildBounds.height+6});
                Check(session.GetComponentsInChildren<RectTransform>().Where(t=>t.name.StartsWith("City cards ")||t.name.StartsWith("Industry cards ")).All(t=>!ScreenBounds(t,camera).Overlaps(headerBounds)&&!ScreenBounds(t,camera).Overlaps(wildBounds)),"all counters remain outside reserved header");
                var footer=ScreenBounds(session.GetComponentsInChildren<RectTransform>().Single(t=>t.name=="Tracker legend panel"),camera);
                Check(!footer.Overlaps(ScreenBounds((RectTransform)session.Toggle.transform,camera)),"footer legend and resume toggle do not overlap");
                var cardBounds=hand.GetComponentsInChildren<BaseCardView>().Select(c=>ScreenBounds((RectTransform)c.HolderObject,camera)).ToArray();
                var toggleBounds=ScreenBounds((RectTransform)session.Toggle.transform,camera);
                Check(cardBounds.Length>0&&cardBounds.All(b=>b.yMax+5<=footer.yMin&&!b.Overlaps(toggleBounds)),"tracker button and legend clear visible hand artwork",new{resolution.x,footerTop=footer.yMin,cards=cardBounds.Select(b=>new{b.xMin,b.xMax,b.yMin,b.yMax})});
                Check(Math.Abs(toggleBounds.xMax-(resolution.x-24))<1&&Math.Abs(toggleBounds.yMin-220)<1,"tracker occupies fixed bottom-right position above action band",resolution.x);
                var actionButtons=actions.GetComponentsInChildren<Button>().Where(b=>b.gameObject.activeInHierarchy).ToArray();
                Check(actionButtons.Length>0&&actionButtons.All(b=>!ScreenBounds((RectTransform)b.transform,camera).Overlaps(toggleBounds)&&ScreenBounds((RectTransform)b.transform,camera).yMax+8<=toggleBounds.yMin),"fixed tracker button clears the native action controls",new{resolution.x,buttons=actionButtons.Select(b=>new{b.name,bounds=ScreenBounds((RectTransform)b.transform,camera).ToString()}).ToArray()});
                var raisedCard=(RectTransform)hand.GetComponentsInChildren<BaseCardView>()[0].HolderObject;
                var cardPosition=raisedCard.anchoredPosition;var cardRotation=raisedCard.localRotation;
                raisedCard.anchoredPosition+=new Vector2(0,160);raisedCard.localRotation=Quaternion.Euler(0,0,14);
                session.Overlay.Layout();Canvas.ForceUpdateCanvases();
                Check(ScreenBounds(raisedCard,camera).yMax+5<=ScreenBounds(session.GetComponentsInChildren<RectTransform>().Single(t=>t.name=="Tracker legend"),camera).yMin&&SameBounds(toggleBounds,ScreenBounds((RectTransform)session.Toggle.transform,camera)),"raised and rotated hand moves only the legend, not tracker button",resolution.x);
                raisedCard.anchoredPosition=cardPosition;raisedCard.localRotation=cardRotation;session.Overlay.Layout();Canvas.ForceUpdateCanvases();
                var headerSurface=session.GetComponentsInChildren<BrassPatch.Statistics.StatisticsSurface>().Single(s=>s.name=="Tracker header");
                Check(headerSurface.canvasRenderer.GetAlpha()>.90f&&headerSurface.canvasRenderer.GetAlpha()<.98f,"tracker header is mildly translucent");
                Check(session.GetComponent<BrassPatch.Statistics.StatisticsTheme>()!=null&&board.GetComponent<BrassPatch.Statistics.StatisticsTheme>()==null,"CT25-THEME light theme is confined to tracker root");
                var headerMesh=GraphicMesh(headerSurface);
                Check(headerMesh.colors32.Length>0&&headerMesh.colors32[0].r>=245&&headerMesh.colors32[0].g>=240&&headerMesh.colors32[0].b>=220,"CT25-THEME header uses Game History ivory surface");Destroy(headerMesh);
                Check(session.GetComponentsInChildren<TextMeshProUGUI>().Where(t=>t.name=="Paused title"||t.name=="Public summary"||t.name=="Wild type name"||t.name=="Tracker legend").All(t=>t.color.r<.5f&&t.color.g<.5f&&t.color.b<.5f),"CT25-THEME dark text remains legible on light panels");
                Check(session.GetComponentsInChildren<CardTrackerGlyph>().Count(g=>g.Kind==4)==20,"CT25-CITY every city badge has a location-card symbol");
                var cityDots=session.GetComponentsInChildren<CardTrackerGlyph>().Where(g=>g.Kind==0&&g.name=="City card dots"&&g.Unseen>0).ToArray();
                Check(cityDots.Length>0&&cityDots.All(g=>{var mesh=GraphicMesh(g);bool red=mesh.colors32.Any(c=>c.r==195&&c.g==77&&c.b==72);Destroy(mesh);return red;}),"CT25-CITY unseen dots render red in actual geometry");
                var cityRects=session.GetComponentsInChildren<RectTransform>().Where(t=>t.name.StartsWith("City cards ")).Select(t=>new{Name=t.name,Bounds=ScreenBounds(t,camera)}).ToArray();
                var overlaps=cityRects.SelectMany((a,i)=>cityRects.Skip(i+1).Where(b=>a.Bounds.Overlaps(b.Bounds)).Select(b=>a.Name+" / "+b.Name)).ToArray();
                Check(overlaps.Length==0,"CT25-CITY city badge hit areas do not overlap",overlaps);
                Check(session.GetComponentsInChildren<TextMeshProUGUI>().Where(t=>t.name=="Paused title"||t.name=="Public summary"||t.name=="Wild holder name").All(t=>t.color.a==1),"tracker background transparency leaves text opaque");
                var industryContainer=industries.transform.Find("SafeArea/Popup/RightPanelBackground/IndustriesContainer");
                foreach(Transform row in industryContainer) {
                    var industry=row.GetComponent<IndustryView>();if(industry==null)continue;
                    var type=ReflectionAccess.Get<BuildingType>(industry,"BuildingType");
                    var badge=session.GetComponentsInChildren<Button>().Single(b=>b.name=="Industry cards "+type);
                    Check(ScreenBounds((RectTransform)badge.transform,camera).xMax<ScreenBounds((RectTransform)row,camera).xMin,"industry counts appear left of matching row",type);
                }
                Check(session.GetComponentsInChildren<CardTrackerGlyph>().Count(g=>g.Kind==1)==4&&session.GetComponentsInChildren<CardTrackerGlyph>().Count(g=>g.Kind==2)==4,"wild strip shows two distinct symbols per stable player");
                Check(session.GetComponentsInChildren<TextMeshProUGUI>().Any(t=>t.name=="Wild cards heading"&&t.text=="Wild cards")&&session.GetComponentsInChildren<TextMeshProUGUI>().Count(t=>t.name=="Wild type name"&&t.text=="Location")==4&&session.GetComponentsInChildren<TextMeshProUGUI>().Count(t=>t.name=="Wild type name"&&t.text=="Industry")==4,"wild cards are explicitly labeled with both card types");
                Check(session.GetComponentsInChildren<TextMeshProUGUI>().Where(t=>t.name=="Wild held count").All(t=>t.text=="1")&&session.GetComponentsInChildren<TextMeshProUGUI>().Any(t=>t.text=="1 held\n0 none"),"wild holder counts explain possession, rather than cards played");
                Check(session.GetComponentsInChildren<TextMeshProUGUI>().Where(t=>t.name=="Wild type name"||t.name=="Wild cards heading"||t.name=="Wild held count").All(t=>!t.isTextTruncated&&t.textInfo.characterInfo.Take(t.textInfo.characterCount).Any(c=>c.isVisible)),"full wild-card type labels fit the compact strip",resolution.x);
                Check(session.GetComponentsInChildren<TextMeshProUGUI>().Where(t=>!String.IsNullOrWhiteSpace(t.text)).All(t=>t.fontSize*canvas.scaleFactor>=17.5f),"tracker text stays readable at physical display size",resolution.x);
                var events=EventSystem.current;if(events==null)events=new GameObject("Tracker events",typeof(EventSystem)).GetComponent<EventSystem>();
                var hit=new PointerEventData(events){position=RectTransformUtility.WorldToScreenPoint(camera,board.transform.position)};var hits=new List<RaycastResult>();events.RaycastAll(hit,hits);
                Check(hits.Count>0&&hits[0].gameObject.transform.IsChildOf(session.transform),"paused input shield wins raycast ordering");
                Capture(camera,target,"tracker-"+resolution.x+"x"+resolution.y+"-"+(night?"night":"day"));
                if(resolution.x==1920&&!night) {
                    var glyph=session.GetComponentsInChildren<CardTrackerGlyph>().First(g=>g.Kind==0&&g.Total==3);int oldUnseen=glyph.Unseen;
                    var capsule=(RectTransform)glyph.transform.parent;var oldSize=capsule.sizeDelta;
                    capsule.sizeDelta=new Vector2(oldSize.x+20/canvas.scaleFactor,oldSize.y);
                    glyph.Unseen=0;glyph.SetVerticesDirty();yield return null;Capture(camera,target,"tracker-depleted-dots-"+resolution.x+"x"+resolution.y);glyph.Unseen=oldUnseen;capsule.sizeDelta=oldSize;glyph.SetVerticesDirty();
                }
                session.Overlay.ShowList();yield return null;
                Check(session.GetComponentsInChildren<TextMeshProUGUI>().Where(t=>t.name=="List heading"||t.name=="Card name"||t.name=="Played count").All(t=>t.color==BrassPatch.Statistics.StatisticsTheme.Ink),"CT25-THEME card list also uses the light theme");
                Check(session.Overlay.HasList&&session.GetComponentsInChildren<Button>().Count(t=>t.name.StartsWith("Tracked row "))==25,"scrolling card list includes all twenty-five physical categories");
                var scroll=session.GetComponentsInChildren<ScrollRect>().Single(s=>s.name=="All card categories");scroll.OnScroll(new PointerEventData(events){scrollDelta=new Vector2(0,-8)});
                Check(scroll.verticalNormalizedPosition<1,"card list scroll works while time is paused");
                if(resolution.x==1920&&!night)Capture(camera,target,"tracker-card-list-"+resolution.x+"x"+resolution.y);
                session.GetComponentsInChildren<Button>().First(t=>t.name=="Tracked row Birmingham").onClick.Invoke();yield return null;
                Check(session.Overlay.HasDetail&&session.GetComponentsInChildren<TextMeshProUGUI>().Any(t=>t.text.Contains("four hidden Canal")),"city detail explains public counting uncertainty");
                if(resolution.x==1920&&!night)Capture(camera,target,"tracker-card-detail-"+resolution.x+"x"+resolution.y);
                input.Down=KeyEnum.Escape;router.ProcessInput();Check(session.IsOpen&&!session.Overlay.HasDetail,"Escape first closes pinned details");
                router.ProcessInput();Check(session.IsOpen&&!session.Overlay.HasList,"Escape next closes card list");
                if(resolution.y==1200&&!night) {
                    Check(Math.Abs(session.Overlay.TextScale-1.15f)<.001f,"tracker defaults to the second text size");
                    var increase=session.GetComponentsInChildren<Button>().First(t=>t.name=="Tracker text size");increase.onClick.Invoke();yield return null;
                    Check(Math.Abs(session.Overlay.TextScale-1.3f)<.001f,"text selector reaches the third size from the default");
                    Check(session.GetComponentsInChildren<TextMeshProUGUI>().Where(t=>t.name=="Paused title"||t.name=="Public summary"||t.name=="Tracker legend").All(t=>!t.isTextTruncated),"large text preserves complete title, counts and legend");
                    Check(session.GetComponentsInChildren<TextMeshProUGUI>().Where(t=>t.name=="Wild type name"||t.name=="Wild cards heading"||t.name=="Wild held count").All(t=>!t.isTextTruncated&&t.textInfo.characterInfo.Take(t.textInfo.characterCount).Any(c=>c.isVisible)),"largest text retains complete wild-card labels");
                    var industryBadges=session.GetComponentsInChildren<RectTransform>().Where(t=>t.name.StartsWith("Industry cards ")).ToArray();
                    Check(industryBadges.All(a=>industryBadges.Where(b=>b!=a).All(b=>!ScreenBounds(a,camera).Overlaps(ScreenBounds(b,camera)))),"large industry counters do not overlap adjacent rows");
                    Capture(camera,target,"tracker-1920x1200-largest-text");
                    session.GetComponentsInChildren<Button>().First(t=>t.name=="Tracker text size").onClick.Invoke();yield return null;
                    Check(Math.Abs(session.Overlay.TextScale-1f)<.001f,"text selector still offers the first size");
                    session.GetComponentsInChildren<Button>().First(t=>t.name=="Tracker text size").onClick.Invoke();yield return null;
                }
                input.Down=KeyEnum.Return;router.ProcessInput();Check(session.IsOpen&&match.App.Game.Revision==revision,"Return cannot advance gameplay underneath tracker");
                session.Overlay.Pan(new Vector2(50,30));session.Overlay.Zoom(1.4f);yield return null;
                Check(zoom.content.anchoredPosition!=oldPosition||zoom.content.localScale!=oldScale,"paused map remains inspectable through owned pan and zoom");
                input.Down=KeyEnum.Escape;router.ProcessInput();input.Down=(KeyEnum)(-1);yield return null;
                Check(!session.IsOpen&&Math.Abs(Time.timeScale-1.4f)<.001&&Math.Abs(DOTween.timeScale-.8f)<.001,"close restores exact previous game and animation speeds");
                Check(zoom.enabled==oldEnabled&&zoom.content.anchoredPosition==oldPosition&&zoom.content.localScale==oldScale,"close restores original map pose and scroll state");
                Check(match.App.Game.Revision==revision&&JsonConvert.SerializeObject(match.Events())==before,"inspection leaves game and public history unchanged");
                Time.timeScale=DOTween.timeScale=1;
            }
            camera.targetTexture=null;target.Release();Destroy(target);
        }
        Check(session.Open(),"tracker can reopen");session.gameObject.SetActive(false);Check(Time.timeScale==1&&DOTween.timeScale==1&&!session.IsOpen,"disabled view restores pause state");session.gameObject.SetActive(true);
        foreach(int count in new[]{2,3}) {
            var smaller=NewMatch(count);Check(CardTrackerEntryPoints.Attach(router,smaller.History,board)==null,"two-/three-player games have no tracker",count);yield return null;
        }
        Check(root.GetComponentsInChildren<CardTrackerSession>(true).Length==0,"retired matches leave no overlay owner");
        var replacement=NewMatch(4);var replaced=CardTrackerEntryPoints.Attach(router,replacement.History,board);
        Check(replaced!=null&&replaced.Open(),"replacement four-player match can open its own tracker");
        Destroy(root);Destroy(cameraNode);yield return null;
        Check(Time.timeScale==1&&DOTween.timeScale==1,"destroying an open board releases the global pause");
    }
    static Mesh GraphicMesh(Graphic graphic)
    {
        var mesh=new Mesh();using(var vertices=new VertexHelper()) {
            graphic.GetType().GetMethod("OnPopulateMesh",ReflectionAccess.Flags,null,new[]{typeof(VertexHelper)},null).Invoke(graphic,new object[]{vertices});
            vertices.FillMesh(mesh);
        }return mesh;
    }
    static bool SameBounds(Rect a,Rect b){return Vector2.Distance(a.center,b.center)<.5f&&Vector2.Distance(a.size,b.size)<.5f;}
    Rect ScreenBounds(RectTransform rect,Camera camera)
    {
        var corners=new Vector3[4];rect.GetWorldCorners(corners);
        var points=corners.Select(c=>RectTransformUtility.WorldToScreenPoint(camera,c)).ToArray();
        return Rect.MinMaxRect(points.Min(p=>p.x),points.Min(p=>p.y),points.Max(p=>p.x),points.Max(p=>p.y));
    }
    void Capture(Camera camera,RenderTexture target,string name)
    {
        Canvas.ForceUpdateCanvases();camera.Render();var previous=RenderTexture.active;RenderTexture.active=target;
        var texture=new Texture2D(target.width,target.height,TextureFormat.RGB24,false);texture.ReadPixels(new Rect(0,0,target.width,target.height),0,0);texture.Apply();RenderTexture.active=previous;
        File.WriteAllBytes(Path.Combine(Path.GetDirectoryName(Output),name+".png"),texture.EncodeToPNG());Destroy(texture);
    }
    sealed class Input:IInputSource {public KeyEnum Down=(KeyEnum)(-1);public bool GetKeyDown(KeyEnum key){return key==Down;}public bool GetKey(KeyEnum key){return false;}public bool GetKeyUp(KeyEnum key){return false;}}
    sealed class Localization:ILocalization {public string GetTranslation(string text){return text.Split('/').Last();}}
    sealed class Settings:IGameSettings
    {
        public bool Night;public bool CameFromMenu(){return false;}public void SetMap(bool value){Night=value;}public bool GetIsNightMap(){return Night;}
        public void SetDefaultGameSettings(GameMetadata metadata,int count){}public int GetNumberOfPlayers(){return 4;}public bool IsSavedDefaultGameSettings(){return false;}
        public PlayerColor GetPlayerIndexSavedColor(int i){return (PlayerColor)i;}public PlayerAvatar GetPlayerIndexSavedAvatar(int i){return (PlayerAvatar)i;}public AIType GetPlayerIndexSavedAIType(int i){return AIType.Human;}
    }
    sealed class TestBoardPresenter:BoardPresenter
    {
        public TestBoardPresenter(IRouter router,ILocalization localization):base(router,localization){}
        public BoardViewModel Model(BrassApplicationGame app){return PrepareBoardViewModel(app.Game.Board,app.Game.GameProgressInformation.CurrentGameEra,null);}
    }
    sealed class LoadPresenter:ILoadSceneOutput
    {public void DisplayTip(int id,bool last){}public void HideView(){}public bool GetWasLastTipNewGlobal(){return false;}public int GetLastTipIdGlobal(){return 1;}}
    sealed class Builder:IUseCaseBuilder
    {
        readonly LoadSceneUseCase load=new LoadSceneUseCase(new LoadPresenter());
        public BoardUseCase Board;public MarketUseCase Market;public IndustriesSummaryUseCase Industries;public ReplayUseCase Replay;public Settings Settings=new Settings();public TestBoardPresenter BoardPresenter;
        public HandOfCardsUseCase Hand;
        public Builder(IRouter router,IGameHistory history){
            var localization=new Localization();BoardPresenter=new TestBoardPresenter(router,localization);Board=new BoardUseCase(BoardPresenter,history,Settings);Board.SetAllowBuidlingFromBoard(false);
            Market=new MarketUseCase(new MarketPresenter(router,localization),history);Industries=new IndustriesSummaryUseCase(new IndustriesSummaryPresenter(router,localization),history);Replay=new ReplayUseCase(null,history);
            Hand=new HandOfCardsUseCase(new HandOfCardsPresenter(router,localization),history);
        }
        public void BuildUseCases(){}public void RestoreUseCases(){}public void RestoreUseCaseOfType(Type type){}
        public IController GetController(Type type){
            if(typeof(IBoardView).IsAssignableFrom(type))return new BoardController(Board);
            if(typeof(IMarketView).IsAssignableFrom(type))return new MarketController(Market);
            if(typeof(IIndustriesSummaryView).IsAssignableFrom(type))return new IndustriesSummaryController(Industries);
            if(typeof(IHandOfCardsView).IsAssignableFrom(type))return new HandOfCardsController(Hand);
            return null;
        }
        public T GetUseCase<T>()where T:IUseCase{
            if(typeof(T)==typeof(BoardUseCase))return (T)(object)Board;if(typeof(T)==typeof(ReplayUseCase))return (T)(object)Replay;
            if(typeof(T)==typeof(LoadSceneUseCase))return (T)(object)load;
            if(typeof(T)==typeof(HandOfCardsUseCase))return (T)(object)Hand;
            throw new NotSupportedException(typeof(T).Name);
        }
    }
}
