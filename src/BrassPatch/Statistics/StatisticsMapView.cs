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
        readonly Dictionary<string,float> radii=new Dictionary<string,float>();
        public void Initialize(StatisticsPanel host,TMP_FontAsset font,StatisticsMapSnapshot data,float pixels,float text)
        {
            Panel=host;Snapshot=data;unit=pixels;scale=text;NodeRadius=12*unit;
            // Count physical industries at this era boundary, including unflipped
            // tiles and owners excluded by the emphasis filter. Added area per tile
            // is constant; the base ring keeps an empty location visible.
            foreach(var city in data.Cities)radii[city.Location.City]=unit*Mathf.Sqrt(144+64*(city.Available&&!city.Location.Merchant?Math.Max(0,city.Occupied):0));
            if(host.Filters.MapMode!="Occupancy"){
                var values=data.Cities.ToDictionary(c=>c.Location.City,c=>(float)(host.Filters.MapMode=="Industry VP"?StatisticsMapQueries.IndustryVP(data,c,host.Filters):StatisticsMapQueries.ContributionVP(host.Report,data,c,host.Filters)));
                float max=Math.Max(1,values.Values.DefaultIfEmpty(0).Max());foreach(var p in values)if(p.Value>0)halo[p.Key]=Radius(p.Key)+8*unit*Mathf.Sqrt(p.Value/max);
            }
            var tag=StatisticsWidgets.Rect(transform,"City hover tag",Vector2.zero,new Vector2(Math.Min(432*scale,rectTransform.rect.width-24*unit),190*scale));
            Tag=tag.gameObject.AddComponent<StatisticsCityTag>();Tag.Initialize(host,font,data,scale,tag.sizeDelta.x);
            // Diagram geometry uses display pixels, independently of text and tag
            // size. The low cities are spaced to leave room for their largest tags.
            float height=Math.Max(360*unit,rectTransform.rect.height-176*unit);
            float width=Math.Min(rectTransform.rect.width-90*unit,height*1.75f);
            height=Math.Min(height,width/1.75f);
            Diagram=new Rect((rectTransform.rect.width-width)/2,-height-24*unit,width,height);
            foreach(var route in StatisticsBoardLayout.Routes.Where(r=>r.Era==data.Era))routes[StatisticsBoardLayout.Identity(route.Cities)]=StatisticsBoardLayout.Paths(route.Cities).Select(path=>path.Select(Project).ToArray()).ToArray();
            foreach(var city in data.Cities.Where(c=>StatisticsBoardLayout.Points.ContainsKey(c.Location.City))){
                var p=Point(city.Location.City);float hitRadius=HitRadius(city.Location.City);var hit=StatisticsWidgets.Rect(transform,"Inspect city "+city.Location.City,p+new Vector2(-hitRadius,hitRadius),Vector2.one*hitRadius*2);
                var image=hit.gameObject.AddComponent<Image>();image.color=Color.clear;image.raycastTarget=false;
                var b=hit.gameObject.AddComponent<Button>();b.targetGraphic=image;var colors=b.colors;colors.normalColor=colors.highlightedColor=colors.selectedColor=Color.clear;b.colors=colors;
                var focus=hit.gameObject.AddComponent<StatisticsMapFocus>();focus.Map=this;focus.City=city.Location.City;b.onClick.AddListener(()=>InspectRecords(city));
            }
            host.BodyScroll.onValueChanged.AddListener(Scrolled);lastPosition=transform.position;raycastTarget=true;SetVerticesDirty();
        }
        Vector2 Project(Vector2 p){return new Vector2(Diagram.x+p.x*Diagram.width,Diagram.y+p.y*Diagram.height);}
        public Vector2 Point(string city){return Project(StatisticsBoardLayout.Points[city]);}
        public float Radius(string city){float value;return radii.TryGetValue(city,out value)?value:NodeRadius;}
        float HitRadius(string city){return Math.Max(22*unit,Radius(city)+6*unit);}
        public string Hit(Vector2 p)
        {
            string result=null;float distance=float.MaxValue;
            foreach(var city in Snapshot.Cities){if(!StatisticsBoardLayout.Points.ContainsKey(city.Location.City))continue;float d=(Point(city.Location.City)-p).magnitude;if(d<HitRadius(city.Location.City)&&d<distance){result=city.Location.City;distance=d;}}
            return result;
        }
        public void FocusCity(string city){focused=true;SetHover(city,true);}
        public void ClearFocus(){focused=false;HideTag();}
        public void PreviewCity(string city){SetHover(city,true);} // Also used by deterministic native pointer/geometry checks.
        void SetHover(string city,bool immediate=false)
        {
            if(city!=HoverCity){HoverCity=city;changedAt=Time.unscaledTime;Tag.Hide();SetVerticesDirty();}
            if(city==null||Tag.City==city)return;
            if(immediate||Time.unscaledTime-changedAt>=.10f){var item=Snapshot.Cities.FirstOrDefault(c=>c.Location.City==city);if(item!=null)Tag.Show(item,Point(city),halo.ContainsKey(city)?halo[city]+3*unit:Radius(city)+3*unit,rectTransform.rect.width);}
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
                var main=paths[0];if(main.Length>1)Symbol(mesh,(main[0]+main[1])*.5f,4.5f*unit,color,StatisticsPlotPalette.Shape(link.Player.ToString()),unit);
            }
            foreach(var city in Snapshot.Cities){
                if(!StatisticsBoardLayout.Points.ContainsKey(city.Location.City))continue;
                var q=Point(city.Location.City);float radius=Radius(city.Location.City);Color tint=StatisticsPlotPalette.Color(new StatisticsPlotSeries{Id="region/"+city.Location.RegionColor});int shape=city.Location.Merchant?3:city.Location.Farm?2:0;
                if(Panel.Filters.Region!=""&&Panel.Filters.Region!=city.Location.RegionColor)tint=Color.Lerp(tint,StatisticsTheme.Paper,.5f);
                if(halo.ContainsKey(city.Location.City))Symbol(mesh,q,halo[city.Location.City],Color.Lerp(tint,StatisticsTheme.Paper,.78f),shape,unit);
                if(HoverCity==city.Location.City){Symbol(mesh,q,radius+5*unit,StatisticsTheme.Focus,shape,unit);Symbol(mesh,q,radius+3*unit,StatisticsTheme.Paper,shape,unit);}
                Symbol(mesh,q,radius,tint,shape,unit);Symbol(mesh,q,radius-3.8f*unit,StatisticsTheme.Paper,shape,unit);
                if(city.Tiles.Any(a=>Panel.Filters.Players.Contains(a.Player)))Symbol(mesh,q,4.8f*unit,tint,shape,unit);
            }
        }
        void Stroke(VertexHelper mesh,Vector2[] path,Color tint,float width){Stroke(mesh,path,tint,width,unit);}
        // UI meshes do not receive reliable camera MSAA in every Canvas mode. A
        // narrow alpha fringe provides coverage at the actual display-pixel scale.
        // Shared cross-sections avoid overlapping fades at polyline bends.
        public static void Stroke(VertexHelper mesh,Vector2[] path,Color tint,float width,float pixel)
        {
            if(path==null||width<=0||tint.a<=0)return;
            var points=new List<Vector2>();foreach(var p in path)if(points.Count==0||(p-points[points.Count-1]).sqrMagnitude>.00001f)points.Add(p);
            if(points.Count<2)return;
            float fringe=Math.Max(.001f,pixel)*.9f,half=Math.Max(width/2,fringe/2),inner=Math.Max(.001f,half-fringe/2),outer=half+fringe/2;
            tint.a*=Math.Min(1,width/fringe);var clear=tint;clear.a=0;int first=mesh.currentVertCount;
            for(int n=0;n<points.Count;n++){
                var before=(points[n]-points[Math.Max(0,n-1)]).normalized;var after=(points[Math.Min(n+1,points.Count-1)]-points[n]).normalized;
                if(n==0)before=after;if(n==points.Count-1)after=before;
                var normal=new Vector2(-after.y,after.x);var miter=(normal+new Vector2(-before.y,before.x)).normalized;
                float dot=Vector2.Dot(miter,normal);if(dot<.25f)miter=normal;else miter/=dot;
                mesh.AddVert(points[n]-miter*outer,clear,Vector2.zero);mesh.AddVert(points[n]-miter*inner,tint,Vector2.zero);
                mesh.AddVert(points[n]+miter*inner,tint,Vector2.zero);mesh.AddVert(points[n]+miter*outer,clear,Vector2.zero);
                if(n>0)for(int band=0;band<3;band++){int a=first+(n-1)*4+band,b=a+4;mesh.AddTriangle(a,b,a+1);mesh.AddTriangle(a+1,b,b+1);}
            }
            Cap(mesh,points[0],(points[0]-points[1]).normalized,inner,outer,tint);
            Cap(mesh,points[points.Count-1],(points[points.Count-1]-points[points.Count-2]).normalized,inner,outer,tint);
        }
        static void Cap(VertexHelper mesh,Vector2 center,Vector2 direction,float inner,float outer,Color tint)
        {
            const int steps=12;var normal=new Vector2(-direction.y,direction.x);var clear=tint;clear.a=0;int first=mesh.currentVertCount;mesh.AddVert(center,tint,Vector2.zero);
            for(int n=0;n<=steps;n++){
                float angle=-Mathf.PI/2+Mathf.PI*n/steps;var ray=direction*Mathf.Cos(angle)+normal*Mathf.Sin(angle);
                mesh.AddVert(center+ray*inner,tint,Vector2.zero);mesh.AddVert(center+ray*outer,clear,Vector2.zero);
                if(n>0){int a=first+1+(n-1)*2,b=a+2;mesh.AddTriangle(first,a,b);mesh.AddTriangle(a,a+1,b);mesh.AddTriangle(b,a+1,b+1);}
            }
        }
        public static void Line(VertexHelper mesh,Vector2 a,Vector2 b,Color tint,float width,float pixel=1)
        {Stroke(mesh,new[]{a,b},tint,width,pixel);}
        public static void Symbol(VertexHelper mesh,Vector2 center,float size,Color tint,int shape,float pixel)
        {
            if(size<=0||tint.a<=0)return;
            if(shape==4||shape==5){Vector2 a=shape==4?new Vector2(size,0):new Vector2(size,size)*.7f,b=shape==4?new Vector2(0,size):new Vector2(size,-size)*.7f;Line(mesh,center-a,center+a,tint,size*.5f,pixel);Line(mesh,center-b,center+b,tint,size*.5f,pixel);return;}
            int count=shape==0?48:shape==2?3:shape==6?6:shape==7?5:4;float rotation=shape==1?Mathf.PI/4:0;
            float fringe=Math.Max(.001f,pixel)*.45f/Mathf.Cos(Mathf.PI/count),inner=Math.Max(0,size-fringe),outer=size+fringe;
            var clear=tint;clear.a=0;int first=mesh.currentVertCount;mesh.AddVert(center,tint,Vector2.zero);
            for(int n=0;n<=count;n++){
                float angle=rotation+Mathf.PI*2*n/count;var ray=new Vector2(Mathf.Sin(angle),Mathf.Cos(angle));
                mesh.AddVert(center+ray*inner,tint,Vector2.zero);mesh.AddVert(center+ray*outer,clear,Vector2.zero);
                if(n>0){int a=first+1+(n-1)*2,b=a+2;mesh.AddTriangle(first,a,b);mesh.AddTriangle(a,a+1,b);mesh.AddTriangle(b,a+1,b+1);}
            }
        }
    }
    public sealed class StatisticsMapFocus : MonoBehaviour,ISelectHandler,IDeselectHandler
    {
        public StatisticsMapView Map;public string City;
        public void OnSelect(BaseEventData e){Map.FocusCity(City);}
        public void OnDeselect(BaseEventData e){Map.ClearFocus();}
    }
}
