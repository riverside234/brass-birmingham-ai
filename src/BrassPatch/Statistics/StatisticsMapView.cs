using System;
using System.Collections.Generic;
using System.Linq;
using TMPro;
using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.UI;
namespace BrassPatch.Statistics
{
    public sealed class StatisticsMapView : MaskableGraphic,IPointerEnterHandler,IPointerExitHandler,IPointerClickHandler
    {
        public StatisticsPanel Panel {get;private set;}
        public StatisticsMapSnapshot Snapshot {get;private set;}
        public StatisticsCityTag Tag {get;private set;}
        public Rect Diagram {get;private set;}
        public string HoverCity {get;private set;}
        public float NodeRadius {get;private set;}
        float unit,scale,changedAt;bool entered,focused;Camera pointerCamera;Vector2 lastPointer;Vector3 lastPosition;
        readonly Dictionary<string,Vector2[][]> routes=new Dictionary<string,Vector2[][]>();
        readonly Dictionary<string,float> halo=new Dictionary<string,float>();
        public void Initialize(StatisticsPanel host,TMP_FontAsset font,StatisticsMapSnapshot data,float pixels,float text)
        {
            Panel=host;Snapshot=data;unit=pixels;scale=text;NodeRadius=10*unit;
            if(host.Filters.MapMode!="Occupancy"){
                var values=data.Cities.ToDictionary(c=>c.Location.City,c=>(float)(host.Filters.MapMode=="Industry VP"?StatisticsMapQueries.IndustryVP(data,c,host.Filters):StatisticsMapQueries.ContributionVP(host.Report,data,c,host.Filters)));
                float max=Math.Max(1,values.Values.DefaultIfEmpty(0).Max());foreach(var p in values)if(p.Value>0)halo[p.Key]=NodeRadius+8*unit*Mathf.Sqrt(p.Value/max);
            }
            var tag=StatisticsWidgets.Rect(transform,"City hover tag",Vector2.zero,new Vector2(Math.Min(470*scale,rectTransform.rect.width-24*unit),350*scale));
            Tag=tag.gameObject.AddComponent<StatisticsCityTag>();Tag.Initialize(host,font,data,scale,tag.sizeDelta.x);
            float height=Math.Max(160*unit,rectTransform.rect.height-Tag.MaximumHeight-48*unit);
            float width=Math.Min(rectTransform.rect.width-120*unit,height*2.7f);
            Diagram=new Rect((rectTransform.rect.width-width)/2,-height-20*unit,width,height);
            foreach(var route in StatisticsBoardLayout.Routes.Where(r=>r.Era==data.Era))routes[StatisticsBoardLayout.Identity(route.Cities)]=StatisticsBoardLayout.Paths(route.Cities).Select(path=>path.Select(Project).ToArray()).ToArray();
            foreach(var city in data.Cities.Where(c=>StatisticsBoardLayout.Points.ContainsKey(c.Location.City))){
                var p=Point(city.Location.City);var hit=StatisticsWidgets.Rect(transform,"Inspect city "+city.Location.City,p+new Vector2(-18,18)*unit,new Vector2(36,36)*unit);
                var image=hit.gameObject.AddComponent<Image>();image.color=Color.clear;image.raycastTarget=false;
                var b=hit.gameObject.AddComponent<Button>();b.targetGraphic=image;var colors=b.colors;colors.normalColor=colors.highlightedColor=colors.selectedColor=Color.clear;b.colors=colors;
                var focus=hit.gameObject.AddComponent<StatisticsMapFocus>();focus.Map=this;focus.City=city.Location.City;b.onClick.AddListener(()=>InspectRecords(city));
            }
            host.BodyScroll.onValueChanged.AddListener(Scrolled);lastPosition=transform.position;raycastTarget=true;SetVerticesDirty();
        }
        Vector2 Project(Vector2 p){return new Vector2(Diagram.x+p.x*Diagram.width,Diagram.y+p.y*Diagram.height);}
        public Vector2 Point(string city){return Project(StatisticsBoardLayout.Points[city]);}
        public string Hit(Vector2 p)
        {
            string result=null;float distance=18*unit;
            foreach(var city in Snapshot.Cities){if(!StatisticsBoardLayout.Points.ContainsKey(city.Location.City))continue;float d=(Point(city.Location.City)-p).magnitude;if(d<distance){result=city.Location.City;distance=d;}}
            return result;
        }
        public void FocusCity(string city){focused=true;SetHover(city,true);}
        public void ClearFocus(){focused=false;HideTag();}
        public void PreviewCity(string city){SetHover(city,true);} // Also used by deterministic native pointer/geometry checks.
        void SetHover(string city,bool immediate=false)
        {
            if(city!=HoverCity){HoverCity=city;changedAt=Time.unscaledTime;Tag.Hide();SetVerticesDirty();}
            if(city==null||Tag.City==city)return;
            if(immediate||Time.unscaledTime-changedAt>=.10f){var item=Snapshot.Cities.FirstOrDefault(c=>c.Location.City==city);if(item!=null)Tag.Show(item,Point(city),halo.ContainsKey(city)?halo[city]+3*unit:NodeRadius+3*unit,rectTransform.rect.width);}
        }
        public void HideTag(){HoverCity=null;if(Tag!=null)Tag.Hide();SetVerticesDirty();}
        void Scrolled(Vector2 value){HideTag();changedAt=Time.unscaledTime;lastPointer=new Vector2(float.NaN,float.NaN);}
        void Update()
        {
            if(Tag==null||Panel==null||!Panel.IsOpen)return;
            if(transform.position!=lastPosition){lastPosition=transform.position;Scrolled(Vector2.zero);}
            if(focused||!entered)return;
            Vector2 p;var mouse=(Vector2)Input.mousePosition;
            if(RectTransformUtility.ScreenPointToLocalPointInRectangle(rectTransform,mouse,pointerCamera,out p))SetHover(Hit(p));
            lastPointer=mouse;
        }
        public void OnPointerEnter(PointerEventData e){entered=true;focused=false;pointerCamera=e.enterEventCamera;Vector2 p;if(RectTransformUtility.ScreenPointToLocalPointInRectangle(rectTransform,e.position,pointerCamera,out p))SetHover(Hit(p));}
        public void OnPointerExit(PointerEventData e){entered=false;if(!focused)HideTag();}
        public void OnPointerClick(PointerEventData e){Vector2 p;if(!RectTransformUtility.ScreenPointToLocalPointInRectangle(rectTransform,e.position,e.pressEventCamera,out p))return;string city=Hit(p);if(city!=null)InspectRecords(Snapshot.Cities.First(c=>c.Location.City==city));}
        void InspectRecords(StatisticsMapCity city)
        {
            bool links=Panel.Filters.MapMode=="Link contributions";
            var target=new StatisticsDrill{Page=links?3:2,HasKeys=true,Era=links?Snapshot.Era:0};
            foreach(var key in links?city.Links.Where(l=>Panel.Filters.Players.Contains(l.Player)).Select(l=>l.Key):city.Tiles.Where(a=>Panel.Filters.Players.Contains(a.Player)).Select(a=>a.Key))target.Keys.Add(key);
            Panel.Drill(target);
        }
        protected override void OnDisable(){base.OnDisable();entered=focused=false;HideTag();}
        protected override void OnDestroy(){if(Panel!=null&&Panel.BodyScroll!=null)Panel.BodyScroll.onValueChanged.RemoveListener(Scrolled);base.OnDestroy();}
        protected override void OnPopulateMesh(VertexHelper mesh)
        {
            mesh.Clear();if(Snapshot==null)return;
            if(Panel.Filters.PrintedRoutes)foreach(var path in routes.Values.SelectMany(p=>p))Stroke(mesh,path,StatisticsTheme.Grid,1.1f*unit);
            foreach(var link in Snapshot.Links){
                Vector2[][] paths;if(!routes.TryGetValue(StatisticsBoardLayout.Identity(link.Cities),out paths))continue;
                bool region=Panel.Filters.Region==""||Snapshot.Cities.Any(c=>c.Location.RegionColor==Panel.Filters.Region&&link.Cities.Contains(c.Location.City));
                Color color=StatisticsTheme.Player(link.Player);if(!Panel.Filters.Players.Contains(link.Player)||!region)color=Color.Lerp(color,StatisticsTheme.Paper,.64f);
                bool hover=HoverCity!=null&&link.Cities.Contains(HoverCity);
                foreach(var path in paths){Stroke(mesh,path,StatisticsTheme.Paper,(hover?8:6)*unit);Stroke(mesh,path,color,(hover?4.8f:3.4f)*unit);}
                var main=paths[0];if(main.Length>1)StatisticsPlotGraphic.Symbol(mesh,(main[0]+main[1])*.5f,4.5f*unit,color,StatisticsPlotPalette.Shape(link.Player.ToString()));
            }
            foreach(var city in Snapshot.Cities){
                if(!StatisticsBoardLayout.Points.ContainsKey(city.Location.City))continue;
                var q=Point(city.Location.City);Color tint=StatisticsPlotPalette.Color(new StatisticsPlotSeries{Id="region/"+city.Location.RegionColor});int shape=city.Location.Merchant?3:city.Location.Farm?2:0;
                if(Panel.Filters.Region!=""&&Panel.Filters.Region!=city.Location.RegionColor)tint=Color.Lerp(tint,StatisticsTheme.Paper,.5f);
                if(halo.ContainsKey(city.Location.City))StatisticsPlotGraphic.Symbol(mesh,q,halo[city.Location.City],Color.Lerp(tint,StatisticsTheme.Paper,.78f),shape);
                if(HoverCity==city.Location.City){StatisticsPlotGraphic.Symbol(mesh,q,NodeRadius+5*unit,StatisticsTheme.Focus,shape);StatisticsPlotGraphic.Symbol(mesh,q,NodeRadius+3*unit,StatisticsTheme.Paper,shape);}
                StatisticsPlotGraphic.Symbol(mesh,q,NodeRadius,tint,shape);StatisticsPlotGraphic.Symbol(mesh,q,NodeRadius-3*unit,StatisticsTheme.Paper,shape);
                if(city.Tiles.Any(a=>Panel.Filters.Players.Contains(a.Player)))StatisticsPlotGraphic.Symbol(mesh,q,3.4f*unit,tint,shape);
            }
        }
        static void Stroke(VertexHelper mesh,Vector2[] path,Color tint,float width){for(int i=1;i<path.Length;i++)Line(mesh,path[i-1],path[i],tint,width);}
        public static void Line(VertexHelper mesh,Vector2 a,Vector2 b,Color tint,float width)
        {
            if((b-a).sqrMagnitude<.00001f)return;var normal=new Vector2(-(b-a).y,(b-a).x).normalized*width/2;int i=mesh.currentVertCount;foreach(var q in new[]{a-normal,a+normal,b+normal,b-normal})mesh.AddVert(q,tint,Vector2.zero);mesh.AddTriangle(i,i+1,i+2);mesh.AddTriangle(i,i+2,i+3);
        }
    }
    public sealed class StatisticsMapFocus : MonoBehaviour,ISelectHandler,IDeselectHandler
    {
        public StatisticsMapView Map;public string City;
        public void OnSelect(BaseEventData e){Map.FocusCity(City);}
        public void OnDeselect(BaseEventData e){Map.ClearFocus();}
    }
}
