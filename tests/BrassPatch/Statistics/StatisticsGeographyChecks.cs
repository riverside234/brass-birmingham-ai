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
        yield return null;yield return null;
        var edgeCheck=Antialiasing(panel,camera,rt,check,prefix);while(edgeCheck.MoveNext())yield return edgeCheck.Current;
        var sizing=Sizing(panel,camera,rt,check,capture,prefix);while(sizing.MoveNext())yield return sizing.Current;
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
                panel.BodyScroll.verticalNormalizedPosition=1;Canvas.ForceUpdateCanvases();
                float pixelScale=rt.width/((RectTransform)panel.transform).rect.width;
                Occupancy(map,pixelScale,check,prefix);
                check(map.Diagram.height*pixelScale>=(preset==1?490:450)&&map.Diagram.width/map.Diagram.height<1.85f,"GEO30-UI larger upright diagram at 1200p",new{prefix,era,preset,width=map.Diagram.width*pixelScale,height=map.Diagram.height*pixelScale});
                check(!map.Tag.gameObject.activeSelf&&panel.GetComponentsInChildren<TextMeshProUGUI>().All(t=>!t.name.StartsWith("City ")),"GEO29-UI initial map has no city tags or permanent labels",new{prefix,era,preset});
                var names=map.Snapshot.Cities.Select(c=>c.Location.City).ToArray();
                if(era==1)canalPositions=names.ToDictionary(n=>n,map.Point);else check(names.All(n=>(map.Point(n)-canalPositions[n]).magnitude<.1f),"GEO29-UI switching era never rearranges or rescales the board",new{prefix,preset});
                check(names.All(c=>map.Hit(map.Point(c))==c),"GEO29-UI every city has an unambiguous hit target",new{prefix,preset});
                check(names.SelectMany((n,i)=>names.Skip(i+1).Select(other=>(map.Point(n)-map.Point(other)).magnitude*pixelScale)).Min()>44,"GEO30-UI city hit areas have breathing room",new{prefix,era,preset});
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
                foreach(string city in names){
                    map.PreviewCity(city);Canvas.ForceUpdateCanvases();var dot=map.Point(city);var rect=map.Tag.Rect;
                    check(map.Tag.City==city&&rect.anchoredPosition.y<dot.y-map.Radius(city)&&rect.anchoredPosition.y-rect.rect.height>=-map.rectTransform.rect.height-1&&rect.anchoredPosition.x>=0&&rect.anchoredPosition.x+rect.rect.width<=map.rectTransform.rect.width,"GEO29-TAG tag is below dot and inside stage",new{prefix,city,era,preset,dot=dot.ToString(),position=rect.anchoredPosition.ToString(),size=rect.rect.ToString()});
                    var corners=new Vector3[4];rect.GetWorldCorners(corners);var viewport=panel.BodyScroll.viewport;
                    check(corners.All(c=>viewport.rect.Contains(viewport.InverseTransformPoint(c))),"GEO29-TAG complete tag stays in visible viewport",new{prefix,city,era,preset});
                    check(!map.Tag.GetComponent<CanvasGroup>().blocksRaycasts&&map.Tag.GetComponentsInChildren<Graphic>().All(g=>!g.raycastTarget),"GEO29-INPUT hover tag never intercepts input",city);
                    check(map.Tag.GetComponentsInChildren<TextMeshProUGUI>().All(t=>t.GetPreferredValues(t.text,t.rectTransform.rect.width,0).y<=t.rectTransform.rect.height+2),"GEO29-TAG tag text fits measured rows",new{prefix,city,preset});
                    var item=map.Snapshot.Cities.Single(c=>c.Location.City==city);var labels=map.Tag.GetComponentsInChildren<TextMeshProUGUI>();var glyphs=map.Tag.GetComponentsInChildren<StatisticsMapGlyph>();
                    check(item.Location.Merchant||glyphs.Count(g=>g.Kind=="Occupied")==item.Occupied&&glyphs.Count(g=>g.Kind=="Vacant")==item.Slots-item.Occupied,"GEO30-TAG slot symbols preserve actual occupancy",new{prefix,city,era,preset});
                    check(labels.Count(t=>t.name.StartsWith("Tile industry "))==item.Tiles.Length&&glyphs.Count(g=>g.name.StartsWith("Flip status ")&&g.Kind=="Check")==item.Tiles.Count(t=>t.FlipRevision>0&&t.FlipRevision<=map.Snapshot.Revision),"GEO30-TAG compact tiles retain each industry and its historical flip",new{prefix,city,era,preset});
                    check(rect.rect.height*pixelScale<190*panel.TextScale/1.15f&&labels.All(t=>t.name!="City era metric"&&t.name!="Flip key"),"GEO30-TAG occupancy tag omits repeated VP prose and flip legend",new{prefix,city,era,preset,height=rect.rect.height*pixelScale});
                    if(city=="Birmingham"||era==2&&(city=="Gloucester"||city=="Worcester"))capture(rt,camera,"geography-"+prefix+"-era"+era+"-text"+preset+"-"+city);
                }
                panel.BodyScroll.onValueChanged.Invoke(Vector2.zero);check(!map.Tag.gameObject.activeSelf,"GEO29-INPUT scrolling clears stale city tags",prefix);
                int objects=map.GetComponentsInChildren<Transform>(true).Length;var timer=System.Diagnostics.Stopwatch.StartNew();
                for(int i=0;i<100;i++){map.PreviewCity("Birmingham");map.PreviewCity("Oxford");}timer.Stop();
                check(objects==map.GetComponentsInChildren<Transform>(true).Length&&panel.GetComponentsInChildren<StatisticsMapView>().Single()==map,"GEO29-INPUT repeated inspection reuses one tag without rebuilding the page",new{prefix,era,preset,updates=200,milliseconds=timer.Elapsed.TotalMilliseconds});map.HideTag();
                foreach(string mode in new[]{"Industry VP","Link contributions"}){
                    panel.GetComponentsInChildren<Button>().First(b=>b.name=="Map measure").onClick.Invoke();yield return null;yield return null;Canvas.ForceUpdateCanvases();map=panel.GetComponentsInChildren<StatisticsMapView>().Single();map.PreviewCity("Birmingham");
                    var city=map.Snapshot.Cities.Single(c=>c.Location.City=="Birmingham");int vp=mode=="Industry VP"?StatisticsMapQueries.IndustryVP(map.Snapshot,city,panel.Filters):StatisticsMapQueries.ContributionVP(panel.Report,map.Snapshot,city,panel.Filters);
                    string value=(mode=="Industry VP"?"Industry VP: ":"Link VP: ")+vp;
                    check(panel.Filters.MapMode==mode&&map.Tag.GetComponentsInChildren<TextMeshProUGUI>().Any(t=>t.name=="City era metric"&&t.text.StartsWith(value)),"GEO30-DATA chosen measure retains labeled era VP",new{prefix,mode,era,preset});
                    var corners=new Vector3[4];map.Tag.Rect.GetWorldCorners(corners);check(corners.All(c=>panel.BodyScroll.viewport.rect.Contains(panel.BodyScroll.viewport.InverseTransformPoint(c))),"GEO30-TAG VP view still fits below its source dot",new{prefix,mode,era,preset});
                    if(era==2&&preset==1)capture(rt,camera,"geography-"+prefix+"-"+(mode=="Industry VP"?"industry-vp":"link-vp"));
                    Occupancy(map,pixelScale,check,prefix);
                    foreach(string name in names){map.PreviewCity(name);Canvas.ForceUpdateCanvases();map.Tag.Rect.GetWorldCorners(corners);check(corners.All(c=>panel.BodyScroll.viewport.rect.Contains(panel.BodyScroll.viewport.InverseTransformPoint(c))),"GEO31-LAYOUT all VP tags fit below larger markers",new{prefix,mode,era,preset,city=name});}
                    if(era==2&&preset==2){map.PreviewCity("Worcester");capture(rt,camera,"geography-"+prefix+"-largest-"+(mode=="Industry VP"?"industry-vp":"link-vp")+"-Worcester");}
                }
                map.HideTag();if(era==2&&preset==2){panel.GetComponentsInChildren<Button>().First(b=>b.name=="Map key").onClick.Invoke();Canvas.ForceUpdateCanvases();capture(rt,camera,"geography-"+prefix+"-key");panel.Tooltip(null);}
                panel.SetNetworkView("Network results");yield return null;yield return null;
                check(panel.GetComponentsInChildren<StatisticsMapView>().Length==0&&panel.GetComponentsInChildren<StatisticsPlotGraphic>().Any(g=>g.Plot.Id=="link-sources"),"GEO29-UI network results retain existing analytical charts",prefix);
            }
        }
        camera.targetTexture=null;rt.Release();UnityEngine.Object.Destroy(rt);panel.ResetFilters();panel.SelectPage(0);yield return null;yield return null;
    }
    static IEnumerator Sizing(StatisticsPanel panel,Camera camera,RenderTexture rt,Action<bool,string,object> check,Action<RenderTexture,Camera,string> capture,string prefix)
    {
        panel.ResetFilters();panel.SelectPage(3);panel.SetNetworkView("Geography");panel.SetMapEra(2);yield return null;yield return null;
        Vector2 size=Vector2.zero;Dictionary<string,Vector2> points=null;Dictionary<string,float> radii=null;float previousFont=0;
        // Exercise the real text-size buttons while the map is open, including
        // returning to the initial preset. Positions are measured in screen pixels.
        foreach(int preset in new[]{0,1,2,1}){
            float desired=1+preset*.15f;while(Math.Abs(panel.TextScale-desired)>.01f){panel.GetComponentsInChildren<Button>().First(b=>b.name==(panel.TextScale>desired?"Smaller text":"Larger text")).onClick.Invoke();yield return null;yield return null;}
            panel.BodyScroll.verticalNormalizedPosition=1;Canvas.ForceUpdateCanvases();var map=panel.GetComponentsInChildren<StatisticsMapView>().Single();float pixelScale=rt.width/((RectTransform)panel.transform).rect.width;
            var currentSize=map.Diagram.size*pixelScale;var currentPoints=map.Snapshot.Cities.ToDictionary(c=>c.Location.City,c=>RectTransformUtility.WorldToScreenPoint(camera,map.rectTransform.TransformPoint(map.Point(c.Location.City))));
            if(points==null){size=currentSize;points=currentPoints;radii=currentPoints.Keys.ToDictionary(n=>n,n=>map.Radius(n)*pixelScale);}
            check((currentSize-size).magnitude<.1f&&points.All(p=>(p.Value-currentPoints[p.Key]).magnitude<.5f),"GEO31-SIZE text buttons preserve map dimensions and screen positions",new{prefix,preset,width=currentSize.x,height=currentSize.y});
            check(radii.All(p=>Math.Abs(map.Radius(p.Key)*pixelScale-p.Value)<.1f),"GEO31-SIZE text buttons preserve occupancy marker sizes",new{prefix,preset});
            check(map.GetComponentsInChildren<StatisticsMapFocus>().All(f=>((RectTransform)f.transform).rect.width*pixelScale>=43.9f&&((RectTransform)f.transform).rect.width>=map.Radius(f.City)*2+11.9f/pixelScale)&&currentPoints.Keys.All(n=>map.Hit(map.Point(n)+Vector2.right*(map.Radius(n)+3/pixelScale))==n),"GEO31-SIZE pointer targets include each larger dot",new{prefix,preset});
            map.PreviewCity("Birmingham");var title=map.Tag.GetComponentsInChildren<TextMeshProUGUI>().Single(t=>t.name=="City tag title");
            check(preset!=2||title.fontSize>previousFont,"GEO31-SIZE text still enlarges inside a fixed diagram",new{prefix,preset,fontSize=title.fontSize*pixelScale});previousFont=title.fontSize;
            var names=currentPoints.Keys.ToArray();float spacing=names.SelectMany((n,i)=>names.Skip(i+1).Select(other=>(currentPoints[n]-currentPoints[other]).magnitude)).Min();
            check(spacing>=75,"GEO31-LAYOUT enlarged markers and hit targets remain separated",new{prefix,preset,minimumSpacing=spacing});
            float clearance=float.MaxValue;
            foreach(var route in StatisticsBoardLayout.Routes)foreach(var path in StatisticsBoardLayout.Paths(route.Cities))for(int i=1;i<path.Length;i++)foreach(var name in names.Where(n=>!route.Cities.Contains(n))){
                var a=Vector2.Scale(path[i-1],map.Diagram.size);var b=Vector2.Scale(path[i],map.Diagram.size);var q=Vector2.Scale(StatisticsBoardLayout.Points[name],map.Diagram.size);var d=b-a;float t=Mathf.Clamp01(Vector2.Dot(q-a,d)/d.sqrMagnitude);clearance=Math.Min(clearance,(q-(a+t*d)).magnitude*pixelScale);
            }
            check(clearance>=28,"GEO31-LAYOUT routes stay clear of unrelated city dots",new{prefix,preset,minimumClearance=clearance});
            map.HideTag();capture(rt,camera,"geography-"+prefix+"-fixed-text"+preset);yield return null;
        }
        var before=panel.GetComponentsInChildren<StatisticsMapView>().Single();var full=before.Snapshot.Cities.ToDictionary(c=>c.Location.City,c=>new{radius=before.Radius(c.Location.City),point=before.Point(c.Location.City)});
        foreach(var player in panel.Report.Players){panel.GetComponentsInChildren<Button>().Single(b=>b.name=="Filter "+player.Name).onClick.Invoke();yield return null;yield return null;}
        var filtered=panel.GetComponentsInChildren<StatisticsMapView>().Single();
        check(panel.Filters.Players.Count==0&&full.All(p=>Math.Abs(filtered.Radius(p.Key)-p.Value.radius)<.001f&&(filtered.Point(p.Key)-p.Value.point).magnitude<.01f),"GEO31-OCCUPANCY player filtering cannot shrink cities or hide physical occupancy",prefix);
        panel.ResetFilters();yield return null;yield return null;
    }
    static void Occupancy(StatisticsMapView map,float pixelScale,Action<bool,string,object> check,string prefix)
    {
        var cities=map.Snapshot.Cities.Where(c=>c.Available&&!c.Location.Merchant).ToArray();var groups=cities.GroupBy(c=>c.Occupied).OrderBy(g=>g.Key).ToArray();
        check(groups.All(g=>g.All(c=>Math.Abs(map.Radius(c.Location.City)-map.Radius(g.First().Location.City))<.001f))&&groups.Zip(groups.Skip(1),(a,b)=>map.Radius(a.First().Location.City)<map.Radius(b.First().Location.City)).All(v=>v),"GEO31-OCCUPANCY dot size follows total industries irrespective of flip state",new{prefix,era=map.Snapshot.Era,mode=map.Panel.Filters.MapMode,unflippedCities=cities.Count(c=>c.Tiles.Any(t=>!map.Snapshot.Flipped(t))),sizes=groups.Select(g=>new{occupied=g.Key,diameter=map.Radius(g.First().Location.City)*pixelScale*2})});
        check(cities.All(c=>map.Radius(c.Location.City)*pixelScale>=11.99f&&map.Radius(c.Location.City)*pixelScale<=20.01f)&&map.Snapshot.Cities.Where(c=>c.Location.Merchant).All(c=>Math.Abs(map.Radius(c.Location.City)*pixelScale-12)<.01f),"GEO31-OCCUPANCY visible size range and merchant symbols remain legible",prefix);
    }
    static IEnumerator Antialiasing(StatisticsPanel panel,Camera camera,RenderTexture rt,Action<bool,string,object> check,string prefix)
    {
        // Render the shipping mesh functions at 1x MSAA. Tests must not hide hard
        // edges by enabling a better render target than the real UI uses.
        var obj=new GameObject("Geography pixel coverage probe",typeof(RectTransform));var rect=(RectTransform)obj.transform;
        rect.SetParent(panel.transform,false);rect.anchorMin=Vector2.zero;rect.anchorMax=Vector2.one;rect.pivot=new Vector2(0,1);rect.offsetMin=rect.offsetMax=Vector2.zero;
        var probe=obj.AddComponent<GeographyCoverageProbe>();probe.Pixel=((RectTransform)panel.transform).rect.width/rt.width;probe.raycastTarget=false;probe.SetVerticesDirty();
        yield return null;Canvas.ForceUpdateCanvases();camera.Render();
        var previous=RenderTexture.active;RenderTexture.active=rt;var image=new Texture2D(rt.width,rt.height,TextureFormat.RGB24,false);image.ReadPixels(new Rect(0,0,rt.width,rt.height),0,0);image.Apply();RenderTexture.active=previous;
        var pixels=image.GetPixels32();
        for(int i=0;i<GeographyCoverageProbe.Angles.Length;i++){
            int cx=310+i%4*420,cy=1200-420-i/4*340;int blended=0,solid=0;double ink=0;
            for(int y=cy-145;y<=cy+145;y++)for(int x=cx-145;x<=cx+145;x++){
                byte value=pixels[y*rt.width+x].r;if(value>10&&value<245)blended++;if(value<10)solid++;ink+=1-value/255.0;
            }
            check(rt.antiAliasing==1&&blended>=160&&solid>=400&&ink>600&&ink<1050,"GEO30-AA smooth coverage and crisp width at 1x sampling",new{prefix,angle=GeographyCoverageProbe.Angles[i],blended,solid,ink});
        }
        // The production glyph uses the same physical fringe for both Canvas scales.
        int circleBlend=0,circleSolid=0;for(int y=180;y<220;y++)for(int x=180;x<220;x++){byte value=pixels[y*rt.width+x].r;if(value>10&&value<245)circleBlend++;if(value<10)circleSolid++;}
        check(circleBlend>=20&&circleSolid>=200,"GEO30-AA round markers preserve solid centres and soften boundaries",new{prefix,circleBlend,circleSolid});
        UnityEngine.Object.Destroy(image);UnityEngine.Object.Destroy(obj);yield return null;
    }
}

public sealed class GeographyCoverageProbe : MaskableGraphic
{
    public static readonly float[] Angles={0,7,23,45,68,83,90,137};public float Pixel=1;
    protected override void OnPopulateMesh(VertexHelper mesh)
    {
        mesh.Clear();var r=rectTransform.rect;foreach(var p in new[]{new Vector2(r.xMin,r.yMin),new Vector2(r.xMin,r.yMax),new Vector2(r.xMax,r.yMax),new Vector2(r.xMax,r.yMin)})mesh.AddVert(p,Color.white,Vector2.zero);mesh.AddTriangle(0,1,2);mesh.AddTriangle(0,2,3);
        for(int i=0;i<Angles.Length;i++){
            var center=new Vector2(310+i%4*420,-420-i/4*340)*Pixel;float angle=Angles[i]*Mathf.Deg2Rad;var half=new Vector2(Mathf.Cos(angle),Mathf.Sin(angle))*120*Pixel;
            StatisticsMapView.Line(mesh,center-half,center+half,Color.black,3.4f*Pixel,Pixel);
        }
        StatisticsMapView.Symbol(mesh,new Vector2(200,-1000)*Pixel,10*Pixel,Color.black,0,Pixel);
    }
}

