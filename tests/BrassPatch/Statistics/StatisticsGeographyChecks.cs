using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using BoardGameRules.Entities.Players;
using BrassPatch.History;
using BrassPatch.Statistics;
using TMPro;
using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.UI;
public static class StatisticsGeographyChecks
{
    static T V<T>(params object[] pairs)where T:new(){var v=new T();for(int i=0;i<pairs.Length;i+=2)typeof(T).GetProperty((string)pairs[i]).SetValue(v,pairs[i+1],null);return v;}
    static void Add<T>(CompletedMatchStatistics r,string field,params T[] values){((List<T>)typeof(CompletedMatchStatistics).GetField(field,BindingFlags.Instance|BindingFlags.NonPublic).GetValue(r)).AddRange(values);}
    static StatisticsAsset Tile(string id,int built,int removed,int flip,PlayerColor owner,int level=2){return V<StatisticsAsset>("Key",id,"Slot",id=="replacement"||id=="old"?"shared":id,"IndustryId",id,"City","Birmingham","Type","CottonMill","Player",owner,"Era",built<100?1:2,"Revision",built,"RemovedRevision",removed,"FlipRevision",flip,"Level",level,"CanalVP",flip>0&&flip<100?5:0,"RailVP",flip>0&&flip<200&&removed==0?5:0);}
    static StatisticsLink Link(string key,int era,PlayerColor owner,params string[] cities){return V<StatisticsLink>("Key",key,"LinkId",key,"Era",era,"Player",owner,"Revision",era==1?40:140,"Cities",cities,"Scored",true);}
    public static void Data(Action<bool,string,object> check)
    {
        var r=V<CompletedMatchStatistics>("CompleteHistory",true);var colors=new[]{PlayerColor.Blue,PlayerColor.Red};
        foreach(var p in colors)Add(r,"players",V<StatisticsPlayer>("Color",p,"Name",p.ToString()));
        var assets=new[]{Tile("level-one",10,110,50,PlayerColor.Blue,1),Tile("carry",20,0,150,PlayerColor.Red),Tile("old",130,170,0,PlayerColor.Blue),Tile("replacement",170,0,180,PlayerColor.Red),Tile("liquidated",175,190,0,PlayerColor.Blue)};
        Add(r,"assets",assets);
        string[] names={"Birmingham","Walsall","Kidderminster","Worcester","SouthWest","Oxford"};
        foreach(var name in names)Add(r,"locations",V<StatisticsLocation>("City",name,"RegionColor","Violet","Farm",name=="SouthWest","Merchant",name=="Oxford"));
        foreach(int era in new[]{1,2})foreach(var p in colors){int rev=era==1?100:200;Add(r,"boundaries",V<StatisticsBoundary>("Name","Scoring","Era",era,"Revision",rev,"Player",p));foreach(string name in names.Where(n=>n!="Oxford")){var present=assets.Where(a=>a.City==name&&a.Revision<=rev&&(a.RemovedRevision==0||a.RemovedRevision>rev)).ToArray();Add(r,"cities",V<StatisticsCity>("Boundary","Scoring","Era",era,"City",name,"Player",p,"Slots",name=="Birmingham"?4:name=="SouthWest"?1:2,"Occupied",present.Length,"Owned",present.Count(a=>a.Player==p)));}}
        Add(r,"links",Link("canal",1,PlayerColor.Blue,"Birmingham","Walsall"),Link("rail",2,PlayerColor.Red,"Birmingham","Walsall"),Link("farm",2,PlayerColor.Blue,"Kidderminster","Worcester","SouthWest"));
        Add(r,"contributions",V<StatisticsLinkContribution>("LinkKey","canal","Era",1,"City","Birmingham","Owner",PlayerColor.Blue,"VP",7),V<StatisticsLinkContribution>("LinkKey","rail","Era",2,"City","Birmingham","Owner",PlayerColor.Red,"VP",3));
        var a1=StatisticsMapQueries.Get(r,1);var a2=StatisticsMapQueries.Get(r,2);var canal=a1.Cities.Single(c=>c.Location.City=="Birmingham");var rail=a2.Cities.Single(c=>c.Location.City=="Birmingham");
        check(a1.Available&&a2.Available&&canal.Available&&rail.Available,"GEO29-DATA coherent boundary snapshots",null);
        check(canal.Tiles.Any(a=>a.Level==1)&&rail.Tiles.All(a=>a.Level!=1),"GEO29-DATA level-I cleanup follows Canal scoring",null);
        check(canal.Tiles.Any(a=>a.Key=="carry")&&rail.Tiles.Any(a=>a.Key=="carry")&&!a1.Flipped(assets[1])&&a2.Flipped(assets[1]),"GEO29-DATA carryover and flip flags are historical",null);
        check(rail.Tiles.Select(a=>a.Key).OrderBy(x=>x).SequenceEqual(new[]{"carry","replacement"}),"GEO29-DATA overbuilt and liquidated industries excluded",null);
        var f=new StatisticsFilters();f.Reset(r);f.MapEra=1;f.Players.Clear();f.Players.Add(PlayerColor.Blue);
        check(canal.Slots==4&&canal.Occupied==2&&StatisticsMapQueries.Page(r,f).Map.Cities.Single(c=>c.Location.City=="Birmingham").Occupied==2,"GEO29-DATA selection never invents open slots or doubles capacity",null);
        check(canal.Links.Single().Player==PlayerColor.Blue&&rail.Links.Single().Player==PlayerColor.Red,"GEO29-DATA link ownership changes independently between eras",null);
        check(a2.Links.Count(l=>l.Key=="farm")==1&&a2.Cities.Count(c=>c.Links.Any(l=>l.Key=="farm"))==3&&a2.Links.All(l=>l.VP==0),"GEO29-DATA zero-VP branched link retained at all three endpoints",null);
        check(canal.Tiles.Any(a=>a.Player==PlayerColor.Red)&&canal.Links.All(l=>l.Player!=PlayerColor.Red),"GEO29-DATA industry ownership is not incident link ownership",null);
        f.Players.Add(PlayerColor.Red);check(StatisticsMapQueries.ContributionVP(r,a1,canal,f)==7&&StatisticsMapQueries.ContributionVP(r,a2,rail,f)==3,"GEO29-DATA selected era contribution does not accumulate previous points",null);
        check(a2.Cities.Single(c=>c.Location.City=="Walsall").Tiles.Length==0&&a2.Cities.Single(c=>c.Location.City=="Oxford").Location.Merchant,"GEO29-DATA empty cities and merchants are explicit",null);
        check(System.Object.ReferenceEquals(a2,StatisticsMapQueries.Get(r,2)),"GEO29-DATA snapshots cached per detached report",null);
        var intro=V<CompletedMatchStatistics>("CompleteHistory",true,"Introductory",true);check(!StatisticsMapQueries.Get(intro,2).Available,"GEO29-DATA introductory Rail is unavailable",null);
        var missing=V<CompletedMatchStatistics>("CompleteHistory",false,"Availability","Missing history");check(!StatisticsMapQueries.Get(missing,1).Available,"GEO29-DATA incomplete evidence not rendered as zero board",null);
        check(StatisticsBoardLayout.Points.Count==27&&StatisticsBoardLayout.Routes.Count(l=>l.Cities.Length==3)==2&&StatisticsBoardLayout.Routes.Any(l=>l.Era==2&&StatisticsBoardLayout.Identity(l.Cities)=="Birmingham|Redditch")&&!StatisticsBoardLayout.Routes.Any(l=>l.Era==1&&StatisticsBoardLayout.Identity(l.Cities)=="Birmingham|Redditch"),"GEO29-DATA verified 27 locations and era-specific route/farm topology",null);
        check(StatisticsBoardLayout.Points["Central"].x<StatisticsBoardLayout.Points["Cannock"].x&&StatisticsBoardLayout.Points["Gloucester"].x>StatisticsBoardLayout.Points["Worcester"].x,"GEO29-DATA corrected board-relative geography",null);
    }
    public static IEnumerator UI(StatisticsPanel panel,Camera camera,Action<bool,string,object> check,Action<RenderTexture,Camera,string> capture,string prefix)
    {
        Data(check);var rt=new RenderTexture(1920,1200,24);rt.Create();camera.targetTexture=rt;
        var decoded=GameHistoryStatisticsCodec.Decode(GameHistoryStatisticsCodec.Encode(panel.Report));
        foreach(int era in panel.Report.Introductory?new[]{1}:new[]{1,2}){
            var a=StatisticsMapQueries.Get(panel.Report,era);var b=StatisticsMapQueries.Get(decoded,era);
            check(a.Available&&b.Available&&a.Cities.Length==27&&a.Cities.All(c=>c.Available),"GEO29-DATA real recorded board reconciles",new{prefix,era,a.Problem,cities=a.Cities.Where(c=>!c.Available).Select(c=>new{c.Location.City,c.Problem})});
            check(a.Cities.Select(c=>c.Location.City+"/"+c.Occupied+"/"+String.Join(",",c.Tiles.Select(t=>t.Key+":"+a.Flipped(t)))).SequenceEqual(b.Cities.Select(c=>c.Location.City+"/"+c.Occupied+"/"+String.Join(",",c.Tiles.Select(t=>t.Key+":"+b.Flipped(t))))),"GEO29-DATA schema-1 History roundtrip gives identical snapshots",new{prefix,era});
        }
        foreach(int preset in new[]{1,2}){
            float desired=1+preset*.15f;while(Math.Abs(panel.TextScale-desired)>.01f){panel.GetComponentsInChildren<Button>().First(b=>b.name==(panel.TextScale>desired?"Smaller text":"Larger text")).onClick.Invoke();yield return null;yield return null;}
            Dictionary<string,Vector2> canalPositions=null;
            foreach(int era in new[]{1,2}){
                panel.ResetFilters();panel.SelectPage(3);panel.SetMapEra(era);yield return null;yield return null;Canvas.ForceUpdateCanvases();
                var map=panel.GetComponentsInChildren<StatisticsMapView>().Single();
                check(!map.Tag.gameObject.activeSelf&&panel.GetComponentsInChildren<TextMeshProUGUI>().All(t=>!t.name.StartsWith("City ")),"GEO29-UI initial map has no city tags or permanent labels",new{prefix,era,preset});
                var names=map.Snapshot.Cities.Select(c=>c.Location.City).ToArray();
                if(era==1)canalPositions=names.ToDictionary(n=>n,map.Point);else check(names.All(n=>(map.Point(n)-canalPositions[n]).magnitude<.1f),"GEO29-UI switching era never rearranges or rescales the board",new{prefix,preset});
                check(names.All(c=>map.Hit(map.Point(c))==c),"GEO29-UI every city has an unambiguous hit target",new{prefix,preset});
                var focus=map.GetComponentsInChildren<StatisticsMapFocus>().First(c=>c.City=="Birmingham");
                focus.OnSelect(new BaseEventData(EventSystem.current));
                check(map.Tag.City=="Birmingham","GEO29-INPUT keyboard focus exposes the same historical city tag",new{prefix,era,preset});
                focus.OnDeselect(new BaseEventData(EventSystem.current));
                var pointer=new PointerEventData(EventSystem.current){position=RectTransformUtility.WorldToScreenPoint(camera,map.rectTransform.TransformPoint(map.Point("Birmingham"))),pointerEnter=map.gameObject};
                pointer.pointerCurrentRaycast=new RaycastResult{module=panel.GetComponentInParent<GraphicRaycaster>(),gameObject=map.gameObject};
                map.OnPointerEnter(pointer);
                check(map.HoverCity=="Birmingham"&&!map.Tag.gameObject.activeSelf,"GEO29-INPUT pointer hit starts the hover delay without a stale tag",new{prefix,era,preset});
                map.OnPointerExit(pointer);
                check(map.HoverCity==null&&!map.Tag.gameObject.activeSelf,"GEO29-INPUT leaving the map clears its tag",prefix);
                capture(rt,camera,"geography-"+prefix+"-era"+era+"-text"+preset+"-overview");
                foreach(string city in new[]{"Birmingham","SouthWest","Oxford","Shrewsbury","Gloucester"}){
                    map.PreviewCity(city);Canvas.ForceUpdateCanvases();var dot=map.Point(city);var rect=map.Tag.Rect;
                    check(map.Tag.City==city&&rect.anchoredPosition.y<dot.y-map.NodeRadius&&rect.anchoredPosition.y-rect.rect.height>=-map.rectTransform.rect.height-1&&rect.anchoredPosition.x>=0&&rect.anchoredPosition.x+rect.rect.width<=map.rectTransform.rect.width,"GEO29-TAG tag is below dot and inside stage",new{prefix,city,era,preset,dot=dot.ToString(),position=rect.anchoredPosition.ToString(),size=rect.rect.ToString()});
                    var corners=new Vector3[4];rect.GetWorldCorners(corners);var viewport=panel.BodyScroll.viewport;
                    check(corners.All(c=>viewport.rect.Contains(viewport.InverseTransformPoint(c))),"GEO29-TAG complete tag stays in visible viewport",new{prefix,city,era,preset});
                    check(!map.Tag.GetComponent<CanvasGroup>().blocksRaycasts&&map.Tag.GetComponentsInChildren<Graphic>().All(g=>!g.raycastTarget),"GEO29-INPUT hover tag never intercepts input",city);
                    check(map.Tag.GetComponentsInChildren<TextMeshProUGUI>().All(t=>t.GetPreferredValues(t.text,t.rectTransform.rect.width,0).y<=t.rectTransform.rect.height+2),"GEO29-TAG tag text fits measured rows",new{prefix,city,preset});
                    if(city=="Birmingham"||era==2&&city=="Gloucester")capture(rt,camera,"geography-"+prefix+"-era"+era+"-text"+preset+"-"+city);
                }
                panel.BodyScroll.onValueChanged.Invoke(Vector2.zero);check(!map.Tag.gameObject.activeSelf,"GEO29-INPUT scrolling clears stale city tags",prefix);
                int objects=map.GetComponentsInChildren<Transform>(true).Length;var timer=System.Diagnostics.Stopwatch.StartNew();
                for(int i=0;i<100;i++){map.PreviewCity("Birmingham");map.PreviewCity("Oxford");}timer.Stop();
                check(objects==map.GetComponentsInChildren<Transform>(true).Length&&panel.GetComponentsInChildren<StatisticsMapView>().Single()==map,"GEO29-INPUT repeated inspection reuses one tag without rebuilding the page",new{prefix,era,preset,updates=200,milliseconds=timer.Elapsed.TotalMilliseconds});map.HideTag();
                panel.SetNetworkView("Network results");yield return null;yield return null;
                check(panel.GetComponentsInChildren<StatisticsMapView>().Length==0&&panel.GetComponentsInChildren<StatisticsPlotGraphic>().Any(g=>g.Plot.Id=="link-sources"),"GEO29-UI network results retain existing analytical charts",prefix);
            }
        }
        camera.targetTexture=null;rt.Release();UnityEngine.Object.Destroy(rt);panel.ResetFilters();panel.SelectPage(0);yield return null;yield return null;
    }
}

