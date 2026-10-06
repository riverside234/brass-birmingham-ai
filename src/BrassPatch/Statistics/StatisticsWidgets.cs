using System;
using System.Collections.Generic;
using System.Linq;
using TMPro;
using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.UI;
namespace BrassPatch.Statistics
{
    internal static class StatisticsWidgets
    {
        public static readonly Color Ivory = new Color(250f/255,240f/255,218f/255,1);
        public static readonly Color Gold = new Color(226f/255,186f/255,105f/255,1);
        public static RectTransform Rect(Transform parent,string name,Vector2 position,Vector2 size)
        {
            var rect=new GameObject(name,typeof(RectTransform)).GetComponent<RectTransform>(); rect.SetParent(parent,false);
            rect.anchorMin=rect.anchorMax=new Vector2(0,1); rect.pivot=new Vector2(0,1); rect.anchoredPosition=position; rect.sizeDelta=size; return rect;
        }
        public static void Stretch(RectTransform rect,Vector2 min,Vector2 max)
        { rect.anchorMin=Vector2.zero; rect.anchorMax=Vector2.one; rect.offsetMin=min; rect.offsetMax=max; }
        public static TextMeshProUGUI Text(Transform parent,TMP_FontAsset font,string name,string value,Vector2 position,Vector2 size,float fontSize,bool wrap=false)
        {
            var rect=Rect(parent,name,position,size); var text=rect.gameObject.AddComponent<TextMeshProUGUI>();
            text.font=font; text.fontStyle=FontStyles.Normal; text.fontSize=fontSize; text.color=StatisticsTheme.Light(parent)?StatisticsTheme.Ink:Ivory; text.text=value;
            // Names and report values are literal text. A nickname containing TMP
            // tags must not change measured geometry or hide chart labels.
            if(parent.GetComponentInParent<StatisticsPanel>()!=null)text.richText=false;
            text.enableWordWrapping=wrap; text.overflowMode=TextOverflowModes.Ellipsis; text.alignment=TextAlignmentOptions.MidlineLeft; text.raycastTarget=false;
            rect.anchorMin=rect.anchorMax=new Vector2(0,1); rect.pivot=new Vector2(0,1); rect.sizeDelta=size; rect.anchoredPosition=position;
            return text;
        }
        public static Button Button(Transform parent,TMP_FontAsset font,string name,string value,Vector2 position,Vector2 size,float fontSize,Action click)
        {
            var rect=Rect(parent,name,position,size); var surface=rect.gameObject.AddComponent<StatisticsSurface>();
            var button=rect.gameObject.AddComponent<Button>(); button.targetGraphic=surface;
            var colors=button.colors; colors.normalColor=Color.white; colors.highlightedColor=new Color(1.3f,1.24f,1.1f,1); colors.selectedColor=colors.highlightedColor; colors.pressedColor=new Color(.8f,.8f,.8f,1); colors.fadeDuration=.08f; if(StatisticsTheme.Light(parent)){colors.highlightedColor=new Color(.94f,.92f,.88f,1);colors.selectedColor=new Color(.86f,.92f,.94f,1);colors.pressedColor=new Color(.85f,.81f,.73f,1);colors.disabledColor=new Color(.94f,.93f,.90f,1);colors.fadeDuration=0;} button.colors=colors;
            var label=Text(rect,font,"Label",value,Vector2.zero,size,fontSize); label.alignment=TextAlignmentOptions.Center;
            Stretch(label.rectTransform,new Vector2(8,2),new Vector2(-8,-2));
            if(click!=null)button.onClick.AddListener(()=>click()); return button;
        }
        public static void Selected(Button button,bool selected)
        { var surface=button.GetComponent<StatisticsSurface>(); surface.Selected=selected; surface.SetVerticesDirty(); }
        public static void Enabled(Button button,bool enabled)
        {
            if(button.interactable==enabled&&button.GetComponentInChildren<TextMeshProUGUI>().color.a==(enabled?1:.45f))return;
            button.interactable=enabled;var text=button.GetComponentInChildren<TextMeshProUGUI>();var tint=text.color;tint.a=enabled?1:.45f;text.color=tint;
        }
        public static TMP_InputField Search(Transform parent,TMP_FontAsset font,Vector2 position,Vector2 size,float fontSize,Action<string> changed)
        {
            var root=Rect(parent,"Find statistics",position,size); root.gameObject.AddComponent<StatisticsSurface>();
            var input=root.gameObject.AddComponent<TMP_InputField>();
            input.targetGraphic=root.GetComponent<StatisticsSurface>();
            var area=Rect(root,"Text area",Vector2.zero,size); Stretch(area,new Vector2(10,3),new Vector2(-10,-3)); area.gameObject.AddComponent<RectMask2D>();
            var label=Text(area,font,"Input","",Vector2.zero,size,fontSize); Stretch(label.rectTransform,Vector2.zero,Vector2.zero); label.raycastTarget=true;
            var placeholder=Text(area,font,"Placeholder","Find city, industry or action…",Vector2.zero,size,fontSize); Stretch(placeholder.rectTransform,Vector2.zero,Vector2.zero); placeholder.color=StatisticsTheme.Light(parent)?StatisticsTheme.Muted:new Color(.83f,.78f,.68f,1);
            input.textViewport=area; input.textComponent=label; input.placeholder=placeholder; input.lineType=TMP_InputField.LineType.SingleLine; input.characterLimit=100; input.onValueChanged.AddListener(text=>changed(text)); return input;
        }
        public static ScrollRect Scroll(Transform parent,string name,Vector2 position,Vector2 size,float unit,bool horizontal=false)
        {
            var root=Rect(parent,name,position,size); var scroll=horizontal?(ScrollRect)root.gameObject.AddComponent<StatisticsTableScroll>():root.gameObject.AddComponent<ScrollRect>();
            var viewport=Rect(root,"Viewport",Vector2.zero,size-new Vector2(horizontal?0:14*unit,horizontal?14*unit:0));
            viewport.gameObject.AddComponent<RectMask2D>(); var background=viewport.gameObject.AddComponent<Image>(); background.color=new Color(0,0,0,.015f);
            var content=Rect(viewport,"Content",Vector2.zero,viewport.sizeDelta); content.pivot=new Vector2(0,1);
            scroll.viewport=viewport; scroll.content=content; scroll.horizontal=horizontal; scroll.vertical=!horizontal; scroll.movementType=ScrollRect.MovementType.Clamped; scroll.scrollSensitivity=34*unit; scroll.inertia=false;
            var track=Rect(root,"Scrollbar",horizontal?new Vector2(0,-size.y+12*unit):new Vector2(size.x-12*unit,0),horizontal?new Vector2(size.x,12*unit):new Vector2(12*unit,size.y));
            track.gameObject.AddComponent<Image>().color=StatisticsTheme.Light(parent)?StatisticsTheme.Alternate:new Color(.16f,.14f,.11f,1);
            var sliding=Rect(track,"Sliding area",Vector2.zero,track.sizeDelta); Stretch(sliding,new Vector2(2,2),new Vector2(-2,-2));
            var handle=Rect(sliding,"Handle",Vector2.zero,sliding.sizeDelta); Stretch(handle,Vector2.zero,Vector2.zero); var graphic=handle.gameObject.AddComponent<Image>(); graphic.color=StatisticsTheme.Light(parent)?StatisticsTheme.Accent:Gold;
            var bar=track.gameObject.AddComponent<Scrollbar>(); bar.handleRect=handle; bar.targetGraphic=graphic; bar.direction=horizontal?Scrollbar.Direction.LeftToRight:Scrollbar.Direction.BottomToTop;
            if(horizontal)scroll.horizontalScrollbar=bar; else scroll.verticalScrollbar=bar; return scroll;
        }
        public static void Clear(Transform root)
        { foreach(Transform child in root.Cast<Transform>().ToArray()) { child.gameObject.SetActive(false); UnityEngine.Object.Destroy(child.gameObject); } }
        public static Color PlayerColor(BoardGameRules.Entities.Players.PlayerColor color)
        {
            switch(color.ToString()) {
                case "Blue":return new Color(.4f,.7f,.94f,1);
                case "Yellow":return new Color(.96f,.79f,.29f,1);
                case "Purple":case "Violet":return new Color(.76f,.57f,.9f,1);
                case "Red":return new Color(.92f,.43f,.36f,1);
                case "Green":return new Color(.43f,.78f,.56f,1);
                default:return Ivory;
            }
        }
    }
    public sealed class StatisticsTableScroll : ScrollRect
    {
        ScrollRect Page { get {var panel=GetComponentInParent<StatisticsPanel>();return panel==null?null:panel.BodyScroll;} }
        static bool Shift { get {return Input.GetKey(KeyCode.LeftShift)||Input.GetKey(KeyCode.RightShift);} }
        bool sideways;
        public override void OnScroll(PointerEventData data)
        {
            if(Shift||Mathf.Abs(data.scrollDelta.x)>Mathf.Abs(data.scrollDelta.y)||Page==null)base.OnScroll(data);
            else Page.OnScroll(data);
        }
        public override void OnInitializePotentialDrag(PointerEventData data)
        {base.OnInitializePotentialDrag(data);if(Page!=null)Page.OnInitializePotentialDrag(data);}
        public override void OnBeginDrag(PointerEventData data)
        {sideways=Shift;base.OnBeginDrag(data);if(Page!=null)Page.OnBeginDrag(data);}
        public override void OnDrag(PointerEventData data)
        {
            if(Mathf.Abs(data.delta.x)>Mathf.Abs(data.delta.y))sideways=true;
            if(sideways||Page==null)base.OnDrag(data);else Page.OnDrag(data);
        }
        public override void OnEndDrag(PointerEventData data)
        {base.OnEndDrag(data);if(Page!=null)Page.OnEndDrag(data);}
    }
    public sealed class StatisticsSurface : MaskableGraphic
    {
        public bool Selected;
        public bool Panel;
        public bool Row,Alternate;
        protected override void OnPopulateMesh(VertexHelper mesh)
        {
            mesh.Clear(); var rect=GetPixelAdjustedRect(); const int steps=6; int count=4*(steps+1);
            float radius=Mathf.Min(Panel?12:7,Mathf.Min(rect.width,rect.height)/2),border=Selected?2.2f:1;
            Color32 edge=Selected?StatisticsWidgets.Gold:Panel?new Color(.50f,.42f,.29f,1):new Color(.30f,.27f,.22f,1);
            Color32 fill=Selected?new Color(.31f,.25f,.16f,1):Row?(Alternate?new Color(.205f,.183f,.15f,1):new Color(.145f,.133f,.113f,1)):new Color(.105f,.095f,.08f,Panel ? .99f : 1);
            if(StatisticsTheme.Light(this)){edge=Selected?StatisticsTheme.Accent:StatisticsTheme.Border;fill=Selected?StatisticsTheme.Selected:Row?(Alternate?StatisticsTheme.Alternate:StatisticsTheme.Paper):Panel&&name=="Statistics window"?StatisticsTheme.Window:StatisticsTheme.Paper;}
            mesh.AddVert(rect.center,fill,Vector2.zero);
            for(int ring=0;ring<2;ring++)for(int corner=0;corner<4;corner++)for(int step=0;step<=steps;step++) {
                float angle=(corner*90+step*90f/steps)*Mathf.Deg2Rad;
                float cx=corner==0||corner==3?rect.xMax-radius:rect.xMin+radius,cy=corner<2?rect.yMax-radius:rect.yMin+radius;
                float rad=Mathf.Max(0,radius-(ring==0?0:border));
                mesh.AddVert(new Vector2(cx+Mathf.Cos(angle)*rad,cy+Mathf.Sin(angle)*rad),ring==0?edge:fill,Vector2.zero);
            }
            for(int i=0;i<count;i++) {
                int next=(i+1)%count;
                mesh.AddTriangle(0,1+count+i,1+count+next);
                mesh.AddTriangle(1+i,1+next,1+count+i); mesh.AddTriangle(1+next,1+count+next,1+count+i);
            }
        }
    }
    public sealed class StatisticsFocusReveal : MonoBehaviour,ISelectHandler
    {
        public StatisticsPanel Panel;
        public void OnSelect(BaseEventData data){if(Panel!=null)Panel.RevealFocus(GetComponent<Selectable>());}
    }
    public sealed class StatisticsTooltip : MonoBehaviour,IPointerEnterHandler,IPointerExitHandler,ISelectHandler,IDeselectHandler
    {
        public StatisticsPanel Panel;
        public string Description;
        public void OnPointerEnter(PointerEventData data){if(Panel!=null)Panel.Tooltip(Description);}
        public void OnPointerExit(PointerEventData data){if(Panel!=null)Panel.Tooltip(null);}
        public void OnSelect(BaseEventData data){if(Panel!=null)Panel.Tooltip(Description);}
        public void OnDeselect(BaseEventData data){if(Panel!=null)Panel.Tooltip(null);}
        void OnDisable(){if(Panel!=null)Panel.Tooltip(null);}
    }
    public sealed class StatisticsChartGraphic : MaskableGraphic,IPointerEnterHandler,IPointerExitHandler,IPointerClickHandler
    {
        public StatisticsChart Data;
        public StatisticsPanel Panel;
        public readonly HashSet<int> Hidden=new HashSet<int>();
        sealed class Hit { public Rect Rect; public StatisticsSeries Series; public StatisticsPoint Point; }
        readonly List<Hit> hits=new List<Hit>();
        bool hovering;
        Camera pointerCamera;
        protected override void OnPopulateMesh(VertexHelper mesh)
        {
            mesh.Clear(); hits.Clear(); if(Data==null)return;
            var bounds=GetPixelAdjustedRect(); var plot=new Rect(bounds.x+18,bounds.y+24,Mathf.Max(1,bounds.width-36),Mathf.Max(1,bounds.height-40));
            var points=Data.Series.SelectMany(s=>s.Points).ToArray(); if(points.Length==0)return;
            double minX=points.Min(p=>p.X),maxX=points.Max(p=>p.X),minY=Math.Min(0,points.Min(p=>p.Y)),maxY=Math.Max(1,points.Max(p=>p.Y));
            if(Data.Stacked) {
                maxY=Math.Max(1,Data.Series.SelectMany(s=>s.Points.GroupBy(p=>p.X).Select(g=>g.Where(p=>p.Y>0).Sum(p=>p.Y))).DefaultIfEmpty(1).Max());
                minY=Math.Min(0,Data.Series.SelectMany(s=>s.Points.GroupBy(p=>p.X).Select(g=>g.Where(p=>p.Y<0).Sum(p=>p.Y))).DefaultIfEmpty(0).Min());
            }
            if(maxX==minX)maxX=minX+1;if(maxY==minY)maxY=minY+1;
            Func<double,float> x=value=>plot.x+(float)((value-minX)/(maxX-minX))*plot.width;
            Func<double,float> y=value=>plot.y+(float)((value-minY)/(maxY-minY))*plot.height;
            for(int line=0;line<=4;line++)Segment(mesh,new Vector2(plot.x,plot.y+plot.height*line/4),new Vector2(plot.xMax,plot.y+plot.height*line/4),1,new Color(.42f,.36f,.26f,.42f));
            Segment(mesh,new Vector2(plot.x,y(0)),new Vector2(plot.xMax,y(0)),1.4f,StatisticsWidgets.Ivory);
            for(int s=0;s<Data.Series.Count;s++) {
                if(Hidden.Contains(s))continue; var series=Data.Series[s]; Color tint=StatisticsWidgets.PlayerColor(series.Player);
                if(Data.Lines) {
                    StatisticsPoint last=null;
                    foreach(var point in series.Points) {
                        var at=new Vector2(x(point.X),y(point.Y));
                        if(last!=null) { var prev=new Vector2(x(last.X),y(last.Y)); Segment(mesh,prev,new Vector2(at.x,prev.y),2.6f,tint);Segment(mesh,new Vector2(at.x,prev.y),at,2.6f,tint); }
                        Quad(mesh,new Rect(at.x-3.5f,at.y-3.5f,7,7),tint);
                        hits.Add(new Hit{Rect=new Rect(at.x-11,at.y-11,22,22),Series=series,Point=point}); last=point;
                    }
                } else if(Data.Stacked) {
                    var groups=series.Points.GroupBy(p=>p.X).ToArray();
                    for(int g=0;g<groups.Length;g++) {
                        float width=Mathf.Min(58,plot.width/(Math.Max(1,Data.Series.Count)*Math.Max(1,groups.Length)*2));
                        float cx=plot.x+plot.width*(s+.5f)/Data.Series.Count+(g-(groups.Length-1)*.5f)*(width+8);
                        double positive=0,negative=0;
                        foreach(var point in groups[g]) {
                            double start=point.Y>=0?positive:negative,end=start+point.Y;
                            var bar=new Rect(cx-width/2,Mathf.Min(y(start),y(end)),width,Mathf.Max(1,Mathf.Abs(y(end)-y(start))));
                            Color segment=CategoryColor(point.Category); Quad(mesh,bar,segment); hits.Add(new Hit{Rect=bar,Series=series,Point=point});
                            if(point.Y>=0)positive=end;else negative=end;
                        }
                    }
                } else {
                    float width=Mathf.Min(32,plot.width/(Math.Max(1,points.Length)*1.4f));
                    foreach(var point in series.Points) {
                        float cx=plot.x+plot.width*(float)((point.X+.5f)/Math.Max(1,points.Max(p=>p.X)+1))+(s-(Data.Series.Count-1)*.5f)*width;
                        var bar=new Rect(cx-width/2,Mathf.Min(y(0),y(point.Y)),width*.88f,Mathf.Max(1,Mathf.Abs(y(point.Y)-y(0))));Quad(mesh,bar,tint);hits.Add(new Hit{Rect=bar,Series=series,Point=point});
                    }
                }
            }
        }
        static Color CategoryColor(string category)
        {
            switch(category){case "Industries":return new Color(.49f,.72f,.55f,1);case "Links":return new Color(.39f,.65f,.86f,1);case "Merchant":return StatisticsWidgets.Gold;case "Penalty":return new Color(.86f,.39f,.35f,1);default:return new Color(.76f,.58f,.86f,1);}
        }
        static void Quad(VertexHelper mesh,Rect rect,Color tint)
        {
            int i=mesh.currentVertCount;mesh.AddVert(new Vector2(rect.xMin,rect.yMin),tint,Vector2.zero);mesh.AddVert(new Vector2(rect.xMin,rect.yMax),tint,Vector2.zero);mesh.AddVert(new Vector2(rect.xMax,rect.yMax),tint,Vector2.zero);mesh.AddVert(new Vector2(rect.xMax,rect.yMin),tint,Vector2.zero);mesh.AddTriangle(i,i+1,i+2);mesh.AddTriangle(i,i+2,i+3);
        }
        static void Segment(VertexHelper mesh,Vector2 a,Vector2 b,float thickness,Color tint)
        {
            if((b-a).sqrMagnitude<.001f)return;var normal=new Vector2(-(b-a).y,(b-a).x).normalized*thickness*.5f;int i=mesh.currentVertCount;
            foreach(var v in new[]{a-normal,a+normal,b+normal,b-normal})mesh.AddVert(v,tint,Vector2.zero);mesh.AddTriangle(i,i+1,i+2);mesh.AddTriangle(i,i+2,i+3);
        }
        Hit At(Vector2 point)
        { return hits.Where(h=>h.Rect.Contains(point)).OrderBy(h=>(h.Rect.center-point).sqrMagnitude).FirstOrDefault(); }
        void Update()
        {
            if(!hovering||Panel==null)return;Vector2 local;
            if(RectTransformUtility.ScreenPointToLocalPointInRectangle(rectTransform,Input.mousePosition,pointerCamera,out local)) {
                var hit=At(local);Panel.Tooltip(hit==null?Data.Unit+" · values available in the table":hit.Series.Label+"\n"+hit.Point.Label+"\n"+hit.Point.Y.ToString("0.##")+" "+Data.Unit+"\n"+hit.Point.Detail);
            }
        }
        public void OnPointerEnter(PointerEventData data){hovering=true;pointerCamera=data.enterEventCamera;}
        public void OnPointerExit(PointerEventData data){hovering=false;if(Panel!=null)Panel.Tooltip(null);}
        public void OnPointerClick(PointerEventData data)
        {
            Vector2 local;if(!RectTransformUtility.ScreenPointToLocalPointInRectangle(rectTransform,data.position,data.pressEventCamera,out local))return;var hit=At(local);
            if(hit!=null&&Panel!=null&&hit.Point.TargetPage>=0)Panel.Drill(hit.Point.TargetPage,hit.Series.Player,hit.Point.Search);
        }
        protected override void OnDisable(){base.OnDisable();hovering=false;if(Panel!=null)Panel.Tooltip(null);}
    }
}
