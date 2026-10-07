using System;
using System.Collections.Generic;
using System.Linq;
using TMPro;
using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.UI;
namespace BrassPatch.Statistics
{
    public static class StatisticsPlotPalette
    {
        public static Color Color(StatisticsPlotSeries s)
        {
            if(s.Player.HasValue)return StatisticsTheme.Player(s.Player.Value);
            string id=s.Id??"";if(id.Contains("/"))id=id.Substring(id.LastIndexOf('/')+1);
            switch(id) {
                case "base-cash": case "awarded": case "delay": case "Blue": case "Canal": case "Links": case "Network": case "Market": case "Location": case "L1": return StatisticsTheme.Hex(0x287A9C);
                case "resource-cash": case "Yellow": case "Rail": case "Merchant": case "Loan": case "Wild Location": case "L2":return StatisticsTheme.Hex(0x9D7024);
                case "Purple":case "Violet":case "Iron":case "Opponents":case "Industry":case "Develop":case "L3+":return StatisticsTheme.Hex(0x77539D);
                case "Red":case "Penalty":case "negative":case "Unsold goods":case "Removed unflipped":return StatisticsTheme.Hex(0xB34F45);
                case "positive":case "flipped":case "Own":case "Other flipped":case "Beer":case "Sell":return StatisticsTheme.Hex(0x367B65);
                case "Industries":return StatisticsTheme.Hex(0x396F4E);
                case "VP":return StatisticsTheme.Hex(0x287A9C);
                case "total":case "built":case "Build":case "Introductory":return StatisticsTheme.Hex(0x8A672B);
                case "Coal":return StatisticsTheme.Hex(0x505B62);
                case "Wild Industry":case "Scout":return StatisticsTheme.Hex(0xA44B78);
                case "seat1":return StatisticsTheme.Hex(0x287A9C);case "seat2":return StatisticsTheme.Hex(0x9D7024);case "seat3":return StatisticsTheme.Hex(0x77539D);case "seat4":return StatisticsTheme.Hex(0xB34F45);
                case "Cyan":case "Scored both eras":return StatisticsTheme.Hex(0x2E7C83);
                case "CottonMill":return StatisticsTheme.Hex(0xA86B30);case "Manufacturer":return StatisticsTheme.Hex(0x287A9C);case "Pottery":return StatisticsTheme.Hex(0xA44B78);case "Brewery":return StatisticsTheme.Hex(0x367B65);case "CoalMine":return StatisticsTheme.Hex(0x505B62);case "IronWorks":return StatisticsTheme.Hex(0x77539D);
                case "opportunity":return StatisticsTheme.Hex(0xCADBD9);case "unflipped":case "Unflipped supplier":case "Pass":case "Unclassified":return StatisticsTheme.Hex(0x657078);
                default:return StatisticsTheme.Hex(0x657078);
            }
        }
        public static int Shape(string id)
        {
            switch(id){case "Build":case "CottonMill":case "Blue":case "flipped":return 1;case "Rail":case "Network":case "CoalMine":case "Yellow":return 3;case "Develop":case "Pottery":case "Purple":case "Violet":case "farm":return 2;case "Loan":case "IronWorks":return 4;case "Scout":case "Manufacturer":return 5;case "Pass":return 6;case "Unclassified":return 7;case "merchant":return 3;default:return 0;}
        }
    }
    public sealed class StatisticsPlotGraphic : MaskableGraphic,IPointerEnterHandler,IPointerExitHandler,IPointerClickHandler
    {
        public StatisticsPlot Plot;
        public readonly HashSet<string> Hidden=new HashSet<string>();
        public StatisticsPanel Panel;
        public Rect Area;
        public double XMin,XMax,YMin,YMax;
        public StatisticsAxisLayout XAxis,YAxis;
        public float Unit=1,RowHeight;
        PointerEventData hover;Vector2 lastPointer;
        public bool Horizontal {get{return StatisticsPlotSemantics.Horizontal(Plot);}}
        public string Identity(StatisticsPlotMark m){return m.Series+"/"+m.Label+"/"+m.Row+"/"+m.Column+"/"+m.X+"/"+m.Y+"/"+(m.Target==null?"":String.Join(",",m.Target.Keys.OrderBy(x=>x)));}
        bool Visible(StatisticsPlotMark m){return !Hidden.Contains(m.Series)&&(!Horizontal||m.Row>=0)&&!m.Missing&&!Double.IsNaN(m.Y);}
        public Vector2 Point(StatisticsPlotMark m)
        {
            float y=Area.yMax-(m.Row+.5f)*RowHeight;
            if(Horizontal){double value=Plot.Kind==StatisticsPlotKind.Waterfall?m.X+m.Y:(m.Missing||Double.IsNaN(m.Y)?0:m.Y);
                if(Plot.Kind==StatisticsPlotKind.StackedBars){value=Plot.Marks.TakeWhile(x=>x!=m).Where(x=>x.Row==m.Row&&x.Facet==m.Facet&&Math.Sign(x.Y)==Math.Sign(m.Y)).Sum(x=>x.Y)+m.Y;double total=Plot.Marks.Where(x=>x.Row==m.Row&&x.Facet==m.Facet).Sum(x=>x.Y);if(Plot.Percent)value=total<=0?0:value*100/total;if(Plot.EraPairs)y+=(m.Facet==1?13:-13)*Unit;}
                if(Plot.Kind==StatisticsPlotKind.PairedBars||Plot.Kind==StatisticsPlotKind.PairedLollipop){int s=Plot.Series.FindIndex(x=>x.Id==m.Series);y+=((Plot.Series.Count-1)/2f-s)*Math.Min((Plot.Kind==StatisticsPlotKind.PairedLollipop?44:18)*Unit,RowHeight/(Plot.Series.Count+1));}
                if(Plot.Kind==StatisticsPlotKind.DelayDots)y+=((Plot.Marks.IndexOf(m)%5)-2)*3.5f*Unit;
                return new Vector2(X(value),y);
            }
            if(Plot.Kind==StatisticsPlotKind.Heatmap)return new Vector2(Area.xMin+(m.Column+.5f)*Area.width/Math.Max(1,Plot.Columns.Count),y);
            if(Plot.Kind==StatisticsPlotKind.MiniMap)return new Vector2(Area.xMin+(float)m.X*Area.width,Area.yMin+(float)m.Y*Area.height);
            if(Plot.Kind==StatisticsPlotKind.EventLanes||Plot.Kind==StatisticsPlotKind.Lifespans){if(Plot.Kind==StatisticsPlotKind.EventLanes)y+=((Plot.Marks.TakeWhile(x=>x!=m).Count(x=>x.Row==m.Row&&x.X==m.X)%2)*2-1)*9*Unit;return new Vector2(X(m.X),y);}
            var point=new Vector2(X(m.X),Y(m.Y));
            if(Plot.Kind==StatisticsPlotKind.Columns){int s=Plot.Series.FindIndex(x=>x.Id==m.Series);point.x+=(s-(Plot.Series.Count-1)/2f)*ColumnBand()/Math.Max(1,Plot.Series.Count);}
            if(Plot.Kind==StatisticsPlotKind.StackedColumns)point.y=Y(Plot.Marks.TakeWhile(x=>x!=m).Where(x=>x.Row==m.Row).Sum(x=>x.Y)+m.Y);
            if(Plot.Kind==StatisticsPlotKind.Scatter){int i=Plot.Marks.TakeWhile(x=>x!=m).Count(x=>x.X==m.X&&x.Y==m.Y);int n=Plot.Marks.Count(x=>x.X==m.X&&x.Y==m.Y);if(n>1){float angle=i*Mathf.PI*2/n;point+=new Vector2(Mathf.Cos(angle),Mathf.Sin(angle))*Math.Min(10,3+n)*Unit;}point.x=Mathf.Clamp(point.x,Area.xMin+8*Unit,Area.xMax-8*Unit);point.y=Mathf.Clamp(point.y,Area.yMin+8*Unit,Area.yMax-8*Unit);}
            return point;
        }
        float ColumnBand(){return Math.Min(74*Unit,Area.width/Math.Max(1,Plot.Columns.Count)*.72f);}
        public float X(double value){return Area.xMin+(float)((value-XMin)/Math.Max(1e-8,XMax-XMin))*Area.width;}
        public float Y(double value){return Area.yMin+(float)((value-YMin)/Math.Max(1e-8,YMax-YMin))*Area.height;}
        Color Tint(string series){return StatisticsPlotPalette.Color(Plot.Series.FirstOrDefault(s=>s.Id==series)??new StatisticsPlotSeries{Id=series});}
        public Color HeatColor(StatisticsPlotMark m)
        {
            if(m.Missing||Double.IsNaN(m.Y))return StatisticsTheme.Alternate;
            if(Plot.DiscreteHeat)return StatisticsPlotPalette.Color(new StatisticsPlotSeries{Id="seat"+(int)m.Y});
            double max=Plot.Marks.Where(x=>!Double.IsNaN(x.Y)).Select(x=>Math.Abs(x.Y)).DefaultIfEmpty(1).Max();
            return UnityEngine.Color.Lerp(StatisticsTheme.Alternate,m.Y<0?StatisticsTheme.Hex(0xB34F45):Tint(m.Series),(float)(max<=0?0:Math.Abs(m.Y)/max));
        }
        public Rect MarkBounds(StatisticsPlotMark m)
        {
            Vector2 q=Point(m);
            if(Plot.Kind==StatisticsPlotKind.Heatmap)return new Rect(q.x-Area.width/Math.Max(1,Plot.Columns.Count)/2+2*Unit,q.y-RowHeight/2+2*Unit,Area.width/Math.Max(1,Plot.Columns.Count)-4*Unit,RowHeight-4*Unit);
            if(Plot.Kind==StatisticsPlotKind.Bullet)return new Rect(X(0),q.y-14*Unit,Math.Max(Unit,X(Math.Max(m.End,m.Y))-X(0)),28*Unit);if(Plot.Kind==StatisticsPlotKind.StackedBars){double total=Plot.Marks.Where(x=>x.Row==m.Row&&x.Facet==m.Facet).Sum(x=>x.Y),size=Plot.Percent?(total==0?0:m.Y*100/total):m.Y;float start=q.x-(X(size)-X(0));float h=Plot.EraPairs?18*Unit:26*Unit;return new Rect(Math.Min(start,q.x),q.y-h/2,Math.Max(1*Unit,Math.Abs(q.x-start)),h);}
            if(Plot.Kind==StatisticsPlotKind.PairedBars||Plot.Kind==StatisticsPlotKind.Waterfall){float start=Plot.Kind==StatisticsPlotKind.Waterfall?X(m.X):X(0);float thick=Plot.Kind==StatisticsPlotKind.PairedBars?Math.Max(3*Unit,Math.Min(15*Unit,RowHeight/(Plot.Series.Count+1)-3*Unit)):26*Unit;return new Rect(Math.Min(start,q.x),q.y-thick/2,Math.Max(1*Unit,Math.Abs(q.x-start)),thick);}
            if(Plot.Kind==StatisticsPlotKind.Columns){float start=Y(0),width=ColumnBand()/Math.Max(1,Plot.Series.Count)*.84f;return new Rect(q.x-width/2,Math.Min(start,q.y),width,Math.Max(1*Unit,Math.Abs(q.y-start)));}
            if(Plot.Kind==StatisticsPlotKind.StackedColumns){float start=q.y-(Y(m.Y)-Y(0));return new Rect(q.x-ColumnBand()/2,Math.Min(start,q.y),ColumnBand(),Math.Max(1*Unit,Math.Abs(q.y-start)));}
            if(Plot.Kind==StatisticsPlotKind.Lifespans)return new Rect(Math.Min(q.x,X(m.End))-6*Unit,q.y-10*Unit,Math.Abs(X(m.End)-q.x)+12*Unit,20*Unit);
            if(Plot.Kind==StatisticsPlotKind.PairedLollipop||Plot.Kind==StatisticsPlotKind.Lollipop||Plot.Kind==StatisticsPlotKind.Dumbbell)return new Rect(q.x-12*Unit,q.y-12*Unit,24*Unit,24*Unit);
            float radius=Plot.Kind==StatisticsPlotKind.MiniMap?Math.Min(18,8+m.Row*2)*Unit+6*Unit:12*Unit;
            return new Rect(q.x-radius,q.y-radius,radius*2,radius*2);
        }
        protected override void OnPopulateMesh(VertexHelper mesh)
        {
            mesh.Clear();if(Plot==null)return;
            if(Horizontal||Plot.Kind==StatisticsPlotKind.EventLanes||Plot.Kind==StatisticsPlotKind.Lifespans)
                for(int row=0;row<Plot.Rows.Count;row++)if(row%2==0){var band=StatisticsTheme.Alternate;band.a=.46f;Quad(mesh,new Rect(Area.xMin,Area.yMax-(row+1)*RowHeight,Area.width,RowHeight),band);}
            if(Plot.Kind==StatisticsPlotKind.MiniMap&&Panel!=null){foreach(var m in Plot.Marks){var city=Panel.Report.Locations.FirstOrDefault(l=>l.City==m.Label);if(city==null)continue;foreach(var neighbor in city.Neighbors){var other=Plot.Marks.FirstOrDefault(x=>x.Label==neighbor);if(other!=null&&String.CompareOrdinal(m.Label,other.Label)<0)Line(mesh,Point(m),Point(other),StatisticsTheme.Grid,1.4f*Unit);}}}
            else if(Plot.Kind!=StatisticsPlotKind.Heatmap) {
                if(XAxis!=null)foreach(var tick in XAxis.Ticks)Line(mesh,new Vector2(X(tick.Value),Area.yMin),new Vector2(X(tick.Value),Area.yMax),Math.Abs(tick.Value)<1e-8?StatisticsTheme.Border:StatisticsTheme.Grid,(Math.Abs(tick.Value)<1e-8?1.8f:1)*Unit);
                if(YAxis!=null)foreach(var tick in YAxis.Ticks)Line(mesh,new Vector2(Area.xMin,Y(tick.Value)),new Vector2(Area.xMax,Y(tick.Value)),Math.Abs(tick.Value)<1e-8?StatisticsTheme.Border:StatisticsTheme.Grid,(Math.Abs(tick.Value)<1e-8?1.8f:1)*Unit);
            }
            if(Plot.Kind==StatisticsPlotKind.Bullet&&!Hidden.Contains("opportunity"))foreach(var m in Plot.Marks.Where(m=>!m.Missing&&m.Row>=0)){var q=Point(m);Quad(mesh,new Rect(X(0),q.y-14*Unit,Math.Max(Unit,X(m.End)-X(0)),28*Unit),Tint("opportunity"));}
            foreach(var m in Plot.Marks.Where(Visible)){
                var q=Point(m);Color tint=Tint(m.Series);int shape=StatisticsPlotPalette.Shape(m.Symbol??m.Series);
                if(Plot.Kind==StatisticsPlotKind.Heatmap)Quad(mesh,MarkBounds(m),HeatColor(m));
                else if(Plot.Kind==StatisticsPlotKind.StackedBars||Plot.Kind==StatisticsPlotKind.PairedBars||Plot.Kind==StatisticsPlotKind.Waterfall||Plot.Kind==StatisticsPlotKind.Columns||Plot.Kind==StatisticsPlotKind.StackedColumns){Quad(mesh,MarkBounds(m),tint);if(Plot.Kind==StatisticsPlotKind.Waterfall){var next=Plot.Marks.FirstOrDefault(x=>x.Row==m.Row+1);if(next!=null)Line(mesh,new Vector2(X(m.X+m.Y),q.y-13*Unit),new Vector2(X(m.X+m.Y),Point(next).y+13*Unit),StatisticsTheme.Border,1.5f*Unit);}}
                else if(Plot.Kind==StatisticsPlotKind.Step){var next=Plot.Marks.SkipWhile(x=>x!=m).Skip(1).FirstOrDefault(x=>x.Series==m.Series);if(next!=null){var to=Point(next);Line(mesh,q,new Vector2(to.x,q.y),tint,2.7f*Unit);Line(mesh,new Vector2(to.x,q.y),to,tint,2.7f*Unit);}Symbol(mesh,q,3.5f*Unit,tint,shape);}
                else if(Plot.Kind==StatisticsPlotKind.Dumbbell){var pair=Plot.Marks.FirstOrDefault(x=>x.Row==m.Row&&x!=m&&Visible(x));if(pair!=null)Line(mesh,q,Point(pair),StatisticsTheme.Border,2.5f*Unit);Symbol(mesh,q,7*Unit,tint,shape);}
                else if(Plot.Kind==StatisticsPlotKind.Bullet){Quad(mesh,new Rect(X(0),q.y-7*Unit,Math.Max(Unit,q.x-X(0)),14*Unit),tint);}else if(Plot.Kind==StatisticsPlotKind.Lollipop||Plot.Kind==StatisticsPlotKind.PairedLollipop){Line(mesh,new Vector2(X(0),q.y),q,tint,2.5f*Unit);Symbol(mesh,q,7*Unit,tint,shape);}
                else if(Plot.Kind==StatisticsPlotKind.Lifespans){Line(mesh,q,new Vector2(X(m.End),q.y),tint,4*Unit);Symbol(mesh,q,5*Unit,tint,1);Symbol(mesh,new Vector2(X(m.End),q.y),6*Unit,tint,m.Symbol=="removed"?2:m.Symbol=="open"?1:0);}
                else Symbol(mesh,q,Plot.Kind==StatisticsPlotKind.MiniMap?Math.Min(18,8+m.Row*2)*Unit:6.5f*Unit,tint,shape);
                string selected;if(Panel!=null&&Panel.SelectedMarks.TryGetValue(Plot.Id,out selected)&&selected==Identity(m)){var bounds=MarkBounds(m);bounds.xMin-=3*Unit;bounds.xMax+=3*Unit;bounds.yMin-=3*Unit;bounds.yMax+=3*Unit;Outline(mesh,bounds,StatisticsTheme.Focus,2.5f*Unit);}
            }
        }
        static void Quad(VertexHelper mesh,Rect rect,Color tint){int i=mesh.currentVertCount;foreach(var q in new[]{new Vector2(rect.xMin,rect.yMin),new Vector2(rect.xMin,rect.yMax),new Vector2(rect.xMax,rect.yMax),new Vector2(rect.xMax,rect.yMin)})mesh.AddVert(q,tint,Vector2.zero);mesh.AddTriangle(i,i+1,i+2);mesh.AddTriangle(i,i+2,i+3);}
        static void Line(VertexHelper mesh,Vector2 a,Vector2 b,Color tint,float width){if((b-a).sqrMagnitude<.00001f)return;var normal=new Vector2(-(b-a).y,(b-a).x).normalized*width/2;int i=mesh.currentVertCount;foreach(var q in new[]{a-normal,a+normal,b+normal,b-normal})mesh.AddVert(q,tint,Vector2.zero);mesh.AddTriangle(i,i+1,i+2);mesh.AddTriangle(i,i+2,i+3);}
        static void Outline(VertexHelper mesh,Rect r,Color tint,float width){var a=new Vector2(r.xMin,r.yMin);var b=new Vector2(r.xMin,r.yMax);var c=new Vector2(r.xMax,r.yMax);var d=new Vector2(r.xMax,r.yMin);Line(mesh,a,b,tint,width);Line(mesh,b,c,tint,width);Line(mesh,c,d,tint,width);Line(mesh,d,a,tint,width);}
        internal static void Symbol(VertexHelper mesh,Vector2 center,float size,Color tint,int shape)
        {
            if(shape==4||shape==5){Vector2 a=shape==4?new Vector2(size,0):new Vector2(size,size)*.7f,b=shape==4?new Vector2(0,size):new Vector2(size,-size)*.7f;Line(mesh,center-a,center+a,tint,size*.5f);Line(mesh,center-b,center+b,tint,size*.5f);return;}
            int count=shape==0?16:shape==2?3:shape==6?6:shape==7?5:4;float rotation=shape==1?Mathf.PI/4:0;int i=mesh.currentVertCount;mesh.AddVert(center,tint,Vector2.zero);for(int n=0;n<=count;n++){float angle=rotation+Mathf.PI*2*n/count;mesh.AddVert(center+new Vector2(Mathf.Sin(angle),Mathf.Cos(angle))*size,tint,Vector2.zero);if(n>0)mesh.AddTriangle(i,i+n,i+n+1);}
        }
        public IList<StatisticsPlotMark> Hits(Vector2 local)
        {
            var found=Plot.Marks.Where(m=>Visible(m)&&(!Horizontal||m.Row>=0)&&MarkBounds(m).Contains(local)).OrderBy(m=>(Point(m)-local).sqrMagnitude).ToList();
            if(Plot.Kind==StatisticsPlotKind.Scatter&&found.Count>0){var first=found[0];found=Plot.Marks.Where(m=>Visible(m)&&m.X==first.X&&m.Y==first.Y).ToList();}
            return found;
        }
        IList<StatisticsPlotMark> Hits(PointerEventData data){Vector2 local;return RectTransformUtility.ScreenPointToLocalPointInRectangle(rectTransform,data.position,data.pressEventCamera??data.enterEventCamera,out local)?Hits(local):new List<StatisticsPlotMark>();}
        public void OnPointerMove(PointerEventData data){var marks=Hits(data);if(Panel!=null)Panel.Tooltip(marks.Count==0?null:Description(marks[0])+(marks.Count>1?"\n"+marks.Count+" overlapping records; click to choose.":""));}
        public void OnPointerEnter(PointerEventData data){hover=data;lastPointer=data.position;OnPointerMove(data);}
        void Update(){if(hover==null)return;Vector2 pointer=Input.mousePosition;if(pointer==lastPointer)return;lastPointer=pointer;hover.position=pointer;OnPointerMove(hover);}
        public void OnPointerExit(PointerEventData data){hover=null;if(Panel!=null)Panel.Tooltip(null);}
        public void OnPointerClick(PointerEventData data){var marks=Hits(data);if(Panel==null||marks.Count==0)return;if(marks.Count>1)Panel.ChoosePlotMarks(this,marks);else Panel.SelectPlotMark(this,marks[0]);}
        public string Description(StatisticsPlotMark mark){return StatisticsPlotSemantics.Description(Plot,mark);}
    }
    public sealed class StatisticsLegendGlyph : MaskableGraphic
    {
        public string Identity;public int ShapeOverride=-1;
        protected override void OnPopulateMesh(VertexHelper mesh){mesh.Clear();var rect=GetPixelAdjustedRect();StatisticsPlotGraphic.Symbol(mesh,rect.center,Mathf.Min(rect.width,rect.height)*.42f,color,ShapeOverride<0?StatisticsPlotPalette.Shape(Identity??""):ShapeOverride);}
    }
    internal static class StatisticsPlotView
    {
        static TextMeshProUGUI Label(Transform parent,TMP_FontAsset font,string name,string value,Vector2 pos,Vector2 size,float pixels,bool wrap=false,TextAlignmentOptions align=TextAlignmentOptions.Center)
        {
            var label=StatisticsWidgets.Text(parent,font,name,value,pos,size,pixels,wrap);label.alignment=align;label.color=StatisticsTheme.Ink;label.overflowMode=TextOverflowModes.Overflow;return label;
        }
        public static float Create(Transform parent,TMP_FontAsset font,StatisticsPlot plot,float y,float width,float unit,float textScale,StatisticsPanel panel,HashSet<string> hidden,Action redraw)
        {
            float scale=unit*textScale;
            if(plot.Kind==StatisticsPlotKind.Efficiency){
                bool beside=width/unit>=1320&&textScale<=1.16f;
                float span=beside?(width-20*unit)/2:width,offset=0,used=0;
                var group=StatisticsWidgets.Rect(parent,"Efficiency comparison card",new Vector2(0,-y),new Vector2(width,1));
                foreach(string metric in new[]{"vp","cost"}){
                    var part=new StatisticsPlot{Id=plot.Id+"-"+metric,Title=metric=="vp"?"VP per confirmed action":"Cash spent per VP",Kind=StatisticsPlotKind.Lollipop,XLabel=metric=="vp"?"VP / confirmed action":"Cash spent per VP (£/VP)",XUnit=metric=="vp"?StatisticsAxisUnit.Ratio:StatisticsAxisUnit.Price,YLabel="Player",Definition=plot.Definition};part.Rows.AddRange(plot.Rows);part.Series.AddRange(plot.Series);part.Marks.AddRange(plot.Marks.Where(m=>m.Metric==metric));
                    var cell=StatisticsWidgets.Rect(group,"Efficiency panel "+metric,new Vector2(beside?offset:0,beside?0:-used),new Vector2(span,1));
                    Label(cell,font,"Efficiency unit "+metric,part.Title,Vector2.zero,new Vector2(span,34*scale),22*scale,false,TextAlignmentOptions.MidlineLeft);
                    float h=40*scale+Create(cell,font,part,40*scale,span,unit,textScale,panel,hidden,redraw);cell.sizeDelta=new Vector2(span,h);
                    if(beside){used=Math.Max(used,h);offset+=span+20*unit;}else used+=h;
                }
                group.sizeDelta=new Vector2(width,used);return used+12*unit;
            }
            bool horizontal=StatisticsPlotSemantics.Horizontal(plot),lanes=plot.Kind==StatisticsPlotKind.EventLanes||plot.Kind==StatisticsPlotKind.Lifespans;
            bool rows=horizontal||plot.Kind==StatisticsPlotKind.Heatmap||lanes;
            var measuring=Label(parent,font,"Plot label measurement","",Vector2.zero,Vector2.one,24*scale);measuring.gameObject.SetActive(false);
            Func<string,float> measure=value=>measuring.GetPreferredValues(value,0,0).x;
            float legendX=12*scale,legendY=10*scale,legendLine=0;var legendBoxes=new List<Rect>();measuring.fontSize=20*scale;measuring.enableWordWrapping=true;
            foreach(var s in plot.Series){
                string value="Off · "+s.Label;float w=Math.Min(Math.Min(width-24*scale,450*scale),Math.Max(92*scale,measure(value)+44*scale));
                float h=Math.Max(34*scale,measuring.GetPreferredValues(value,w-42*scale,0).y+12*scale);
                if(legendX+w>width-12*scale){legendX=12*scale;legendY+=legendLine+6*scale;legendLine=0;}
                legendBoxes.Add(new Rect(legendX,legendY,w,h));legendX+=w+8*scale;legendLine=Math.Max(legendLine,h);
            }
            float legendSpace=plot.Series.Count==0?12*scale:legendY+legendLine+16*scale;measuring.fontSize=24*scale;measuring.enableWordWrapping=false;
            float rowHeight=(plot.Kind==StatisticsPlotKind.Bullet?76:plot.Kind==StatisticsPlotKind.PairedLollipop?100:plot.Kind==StatisticsPlotKind.Heatmap?(plot.HeatTotals?56:76):plot.Kind==StatisticsPlotKind.PairedBars?Math.Max(62,plot.Series.Count*24):plot.Kind==StatisticsPlotKind.Dumbbell?78:56)*scale;
            float graphHeight=rows?Math.Max(132*scale,plot.Rows.Count*rowHeight):plot.Kind==StatisticsPlotKind.MiniMap?660*scale:330*scale;
            float labelWidth=rows?Math.Min(width*.30f,285*scale):90*scale;
            if(rows&&plot.Rows.Count>0){measuring.fontSize=21*scale;labelWidth=Math.Min(width*.36f,Math.Max(190*scale,plot.Rows.Max(s=>measure(s))+30*scale));measuring.fontSize=24*scale;}
            if(rows&&plot.Rows.Count>0){measuring.fontSize=21*scale;measuring.enableWordWrapping=true;rowHeight=Math.Max(rowHeight,plot.Rows.Max(s=>measuring.GetPreferredValues(s,labelWidth-24*scale,0).y)+12*scale);graphHeight=Math.Max(132*scale,plot.Rows.Count*rowHeight);measuring.enableWordWrapping=false;measuring.fontSize=24*scale;}
            float graphWidth=width;
            float heatColumn=100*scale;
            if(plot.Kind==StatisticsPlotKind.Heatmap){measuring.fontSize=21*scale;heatColumn=Math.Min(320*scale,Math.Max(100*scale,plot.Columns.Select(measure).DefaultIfEmpty(100*scale).Max()+20*scale));graphWidth=Math.Max(width,labelWidth+Math.Max(1,plot.Columns.Count)*heatColumn+100*scale);measuring.fontSize=24*scale;}
            if(plot.Kind==StatisticsPlotKind.Columns||plot.Kind==StatisticsPlotKind.StackedColumns)graphWidth=Math.Max(width,110*scale+plot.Columns.Count*(plot.Kind==StatisticsPlotKind.StackedColumns?172:70)*scale);
            if(plot.Kind==StatisticsPlotKind.MiniMap)graphWidth=Math.Max(width,1280*scale);
            float topSpace=plot.Kind==StatisticsPlotKind.Heatmap?100*scale:plot.Kind==StatisticsPlotKind.MiniMap?35*scale:horizontal||lanes?86*scale:38*scale;
            if(plot.Kind==StatisticsPlotKind.Heatmap){measuring.fontSize=21*scale;measuring.enableWordWrapping=true;float cellWidth=(graphWidth-labelWidth-95*scale)/Math.Max(1,plot.Columns.Count);topSpace=Math.Max(topSpace,plot.Columns.Select(s=>measuring.GetPreferredValues(s,cellWidth,0).y+44*scale).DefaultIfEmpty(topSpace).Max());measuring.enableWordWrapping=false;measuring.fontSize=24*scale;}
            float bottomSpace=plot.Kind==StatisticsPlotKind.StackedColumns?145*scale:plot.Kind==StatisticsPlotKind.Heatmap?95*scale:100*scale;
            float chartHeight=graphHeight+topSpace+bottomSpace;
            var root=StatisticsWidgets.Rect(parent,"Plot "+plot.Id,new Vector2(0,-y),new Vector2(width,legendSpace+chartHeight+14*unit));var backdrop=root.gameObject.AddComponent<StatisticsSurface>();backdrop.Panel=true;backdrop.raycastTarget=false;
            Transform container=StatisticsWidgets.Rect(root,"Plot frame",new Vector2(0,-legendSpace),new Vector2(width,chartHeight));
            if(graphWidth>width+1){var scroll=StatisticsWidgets.Scroll(container,"Plot horizontal scroll",Vector2.zero,new Vector2(width,chartHeight+14*unit),unit,true);scroll.content.sizeDelta=new Vector2(graphWidth,chartHeight);container=scroll.content;}
            var chartRect=StatisticsWidgets.Rect(container,"Plot marks",Vector2.zero,new Vector2(graphWidth,chartHeight));var graphic=chartRect.gameObject.AddComponent<StatisticsPlotGraphic>();graphic.Plot=plot;graphic.Panel=panel;graphic.Unit=scale;graphic.RowHeight=rowHeight;
            float rightGutter=plot.Kind==StatisticsPlotKind.Bullet?330*scale:plot.Id=="link-sources"||plot.Kind==StatisticsPlotKind.Dumbbell?235*scale:plot.Id=="conversion"?205*scale:plot.Kind==StatisticsPlotKind.Step||plot.Kind==StatisticsPlotKind.DelayDots?150*scale:plot.XUnit==StatisticsAxisUnit.Ratio||plot.XUnit==StatisticsAxisUnit.Price?195*scale:110*scale;
            graphic.Area=new Rect(labelWidth,-topSpace-graphHeight,graphWidth-labelWidth-rightGutter,graphHeight);
            if(plot.Kind==StatisticsPlotKind.Heatmap)graphic.Area=new Rect(labelWidth,-topSpace-graphHeight,graphWidth-labelWidth-95*scale,graphHeight);
            if(plot.Kind==StatisticsPlotKind.MiniMap)graphic.Area=new Rect(95*scale,-topSpace-graphHeight+55*scale,graphWidth-235*scale,graphHeight-120*scale);
            foreach(var s in plot.Series)if(hidden.Contains(plot.Id+"/"+s.Id))graphic.Hidden.Add(s.Id);
            var marks=plot.Marks.Where(m=>(!horizontal||m.Row>=0)&&!m.Missing&&!Double.IsNaN(m.Y)).ToArray();double minX=0,maxX=1,minY=0,maxY=1;
            if(horizontal){
                if(plot.Kind==StatisticsPlotKind.StackedBars){minX=Math.Min(0,marks.GroupBy(m=>new{m.Row,m.Facet}).Select(g=>g.Where(m=>m.Y<0).Sum(m=>m.Y)).DefaultIfEmpty().Min());maxX=plot.Percent?100:marks.GroupBy(m=>new{m.Row,m.Facet}).Select(g=>g.Where(m=>m.Y>0).Sum(m=>m.Y)).DefaultIfEmpty().Max();}
                else if(plot.Kind==StatisticsPlotKind.Waterfall){minX=marks.SelectMany(m=>new[]{m.X,m.X+m.Y}).DefaultIfEmpty().Min();maxX=marks.SelectMany(m=>new[]{m.X,m.X+m.Y}).DefaultIfEmpty().Max();}
                else{minX=marks.Select(m=>m.Y).DefaultIfEmpty().Min();maxX=marks.Select(m=>plot.Kind==StatisticsPlotKind.Bullet?Math.Max(m.Y,m.End):m.Y).DefaultIfEmpty().Max();}
            }else{minX=marks.Select(m=>m.X).DefaultIfEmpty().Min();maxX=marks.Select(m=>plot.Kind==StatisticsPlotKind.Lifespans?m.End:m.X).DefaultIfEmpty().Max();minY=marks.Select(m=>m.Y).DefaultIfEmpty().Min();maxY=plot.Kind==StatisticsPlotKind.StackedColumns?marks.GroupBy(m=>m.Row).Select(g=>g.Sum(m=>m.Y)).DefaultIfEmpty().Max():marks.Select(m=>m.Y).DefaultIfEmpty().Max();}
            if(plot.SharedMax.HasValue)maxX=plot.SharedMax.Value;
            bool category=plot.Kind==StatisticsPlotKind.Columns||plot.Kind==StatisticsPlotKind.StackedColumns;
            if(plot.Kind!=StatisticsPlotKind.Heatmap&&plot.Kind!=StatisticsPlotKind.MiniMap){
                if(horizontal||lanes){graphic.XAxis=StatisticsAxisLayout.Create(minX,maxX,plot.Percent?StatisticsAxisUnit.Percent:lanes?StatisticsAxisUnit.Turn:plot.XUnit,graphic.Area.width,measure);graphic.XMin=graphic.XAxis.Min;graphic.XMax=graphic.XAxis.Max;}
                else{graphic.YAxis=StatisticsAxisLayout.Create(minY,maxY,plot.YUnit,graphic.Area.height,s=>34*scale,true);float gutter=Math.Max(90*scale,graphic.YAxis.Ticks.Max(t=>measure(t.Label))+28*scale);graphic.Area=new Rect(gutter,graphic.Area.yMin,graphWidth-gutter-rightGutter,graphic.Area.height);graphic.YMin=graphic.YAxis.Min;graphic.YMax=graphic.YAxis.Max;
                    if(category){graphic.XMin=-.5;graphic.XMax=Math.Max(.5,plot.Columns.Count-.5);}
                    else {graphic.XAxis=StatisticsAxisLayout.Create(minX,maxX,plot.XUnit,graphic.Area.width,measure);graphic.XMin=graphic.XAxis.Min;graphic.XMax=graphic.XAxis.Max;}
                }
            }
            if(rows)for(int i=0;i<plot.Rows.Count;i++)Label(container,font,"Axis row "+i,plot.Rows[i],new Vector2(12*scale,graphic.Area.yMax-i*rowHeight),new Vector2(labelWidth-24*scale,rowHeight),21*scale,true,TextAlignmentOptions.MidlineRight);
            if(plot.Kind==StatisticsPlotKind.Heatmap){
                for(int c=0;c<plot.Columns.Count;c++)Label(container,font,"Axis column "+c,plot.Columns[c],new Vector2(graphic.Area.xMin+c*graphic.Area.width/plot.Columns.Count,-38*scale),new Vector2(graphic.Area.width/plot.Columns.Count,topSpace-42*scale),21*scale,true);
                foreach(var m in plot.Marks.Where(m=>!graphic.Hidden.Contains(m.Series))){var q=graphic.Point(m);var text=Label(container,font,"Heat value",m.Missing||Double.IsNaN(m.Y)?"—":m.Y.ToString("0.##"),new Vector2(q.x-38*scale,q.y+18*scale),new Vector2(76*scale,36*scale),22*scale);text.color=StatisticsTheme.ReadableOn(graphic.HeatColor(m));if(!plot.HeatTotals&&m.OtherLabel!=null){text.rectTransform.anchoredPosition+=new Vector2(0,12*scale);var sample=Label(container,font,"Heat sample",m.OtherLabel,new Vector2(q.x-45*scale,q.y-4*scale),new Vector2(90*scale,28*scale),18*scale);sample.color=text.color;}}
                if(!plot.DiscreteHeat&&plot.HeatTotals){foreach(int row in Enumerable.Range(0,plot.Rows.Count))Label(container,font,"Heat row total",plot.Marks.Where(m=>m.Row==row&&!m.Missing).Sum(m=>m.Y).ToString("0"),new Vector2(graphic.Area.xMax+8*scale,graphic.Area.yMax-row*rowHeight),new Vector2(76*scale,rowHeight),21*scale);foreach(int col in Enumerable.Range(0,plot.Columns.Count))Label(container,font,"Heat column total",plot.Marks.Where(m=>m.Column==col&&!m.Missing).Sum(m=>m.Y).ToString("0"),new Vector2(graphic.Area.xMin+col*graphic.Area.width/Math.Max(1,plot.Columns.Count),graphic.Area.yMin-8*scale),new Vector2(graphic.Area.width/Math.Max(1,plot.Columns.Count),34*scale),21*scale);Label(container,font,"Heat totals key","Totals",new Vector2(graphic.Area.xMax,graphic.Area.yMin-8*scale),new Vector2(85*scale,34*scale),19*scale);}
                Label(container,font,"Heat row axis",plot.YLabel,new Vector2(12*scale,-2*scale),new Vector2(labelWidth-24*scale,32*scale),20*scale,false,TextAlignmentOptions.MidlineRight);
                Label(container,font,"Heat column axis",plot.XLabel,new Vector2(graphic.Area.xMin,graphic.Area.yMin-52*scale),new Vector2(graphic.Area.width,32*scale),21*scale);
                HeatKey(container,font,graphic,scale,panel.Report.Players.Count,width);
                if(!plot.DiscreteHeat&&plot.HeatTotals)Label(container,font,"Heat row totals heading","Total",new Vector2(graphic.Area.xMax+8*scale,-38*scale),new Vector2(76*scale,topSpace-42*scale),21*scale);
            }
            else {
                if(graphic.XAxis!=null)foreach(var tick in graphic.XAxis.Ticks) {
                    float w=Math.Max(44*scale,measure(tick.Label)+8*scale),x=graphic.X(tick.Value);Label(container,font,"Value tick X "+tick.Value,tick.Label,new Vector2(x-w/2,graphic.Area.yMin-10*scale),new Vector2(w,34*scale),24*scale);
                    if(horizontal||lanes)Label(container,font,"Value tick top "+tick.Value,tick.Label,new Vector2(x-w/2,graphic.Area.yMax+42*scale),new Vector2(w,34*scale),24*scale);
                    var line=StatisticsWidgets.Rect(container,"X tick mark",new Vector2(x,graphic.Area.yMin),new Vector2(1.5f*scale,7*scale));line.gameObject.AddComponent<Image>().color=StatisticsTheme.Border;
                }
                if(graphic.YAxis!=null)foreach(var tick in graphic.YAxis.Ticks)Label(container,font,"Value tick Y "+tick.Value,tick.Label,new Vector2(8*scale,graphic.Y(tick.Value)+18*scale),new Vector2(graphic.Area.xMin-20*scale,36*scale),24*scale,false,TextAlignmentOptions.MidlineRight);
                if(category){int every=Math.Max(1,(int)Math.Ceiling(plot.Columns.Count/(graphic.Area.width/(plot.Kind==StatisticsPlotKind.StackedColumns?150:66)/scale)));for(int c=0;c<plot.Columns.Count;c+=every){float band=graphic.Area.width/plot.Columns.Count;Label(container,font,"X category "+c,plot.Columns[c],new Vector2(graphic.X(c)-band/2,graphic.Area.yMin-12*scale),new Vector2(band,plot.Kind==StatisticsPlotKind.StackedColumns?94*scale:32*scale),21*scale,plot.Kind==StatisticsPlotKind.StackedColumns);}}
                Label(container,font,"X axis",plot.XLabel,new Vector2(graphic.Area.xMin,graphic.Area.yMin-(category&&plot.Kind==StatisticsPlotKind.StackedColumns?106:55)*scale),new Vector2(graphic.Area.width,32*scale),21*scale);
                if(horizontal||lanes)Label(container,font,"Top axis unit",plot.XLabel,new Vector2(graphic.Area.xMin,-2*scale),new Vector2(graphic.Area.width,32*scale),20*scale,false,TextAlignmentOptions.MidlineLeft);
                if(!horizontal&&!lanes)Label(container,font,"Y axis",plot.YLabel,new Vector2(graphic.Area.xMin,-2*scale),new Vector2(graphic.Area.width,32*scale),21*scale);
                DirectLabels(container,font,graphic,scale);
                if((plot.Kind==StatisticsPlotKind.Step||lanes)&&panel!=null){var entry=panel.Report.Boundaries.FirstOrDefault(b=>b.Name=="Rail entry");if(entry!=null&&entry.Turn>=graphic.XMin&&entry.Turn<=graphic.XMax){var line=StatisticsWidgets.Rect(container,"Rail era boundary",new Vector2(graphic.X(entry.Turn),graphic.Area.yMax),new Vector2(2*scale,graphic.Area.height));line.gameObject.AddComponent<Image>().color=StatisticsTheme.Accent;line.GetComponent<Image>().raycastTarget=false;Label(container,font,"Era Canal band","Canal",new Vector2(graphic.Area.xMin,graphic.Area.yMin-78*scale),new Vector2(Math.Max(60*scale,graphic.X(entry.Turn)-graphic.Area.xMin),24*scale),19*scale);Label(container,font,"Era Rail band","Rail",new Vector2(graphic.X(entry.Turn),graphic.Area.yMin-78*scale),new Vector2(Math.Max(60*scale,graphic.Area.xMax-graphic.X(entry.Turn)),24*scale),19*scale);}}
            }
            // Transparent keyboard proxies never intercept mouse raycasts. Actual
            // pointer hits use the whole rendered shape in StatisticsPlotGraphic.
            var keyboard=plot.Marks.Where(m=>(!horizontal||m.Row>=0)&&!graphic.Hidden.Contains(m.Series)).ToArray();if(keyboard.Length>80)keyboard=keyboard.Where((m,i)=>i==0||i==keyboard.Length-1||i%Math.Max(1,keyboard.Length/60)==0).ToArray();
            foreach(var m in keyboard){var q=graphic.Point(m);var hit=StatisticsWidgets.Rect(chartRect,"Select plot value",q+new Vector2(-12,12)*scale,new Vector2(24,24)*scale);var image=hit.gameObject.AddComponent<Image>();image.color=new Color(0,0,0,0);image.raycastTarget=false;var button=hit.gameObject.AddComponent<Button>();button.targetGraphic=image;var colors=button.colors;colors.highlightedColor=new Color(.16f,.43f,.51f,.35f);colors.selectedColor=colors.highlightedColor;colors.normalColor=new Color(0,0,0,0);colors.fadeDuration=0;button.colors=colors;var tooltip=hit.gameObject.AddComponent<StatisticsTooltip>();tooltip.Panel=panel;tooltip.Description=graphic.Description(m);button.onClick.AddListener(()=>panel.SelectPlotMark(graphic,m));}
            int at=0;foreach(var series in plot.Series){
                var s=series;var box=legendBoxes[at++];bool shown=!graphic.Hidden.Contains(s.Id);
                var button=StatisticsWidgets.Button(root,font,"Legend "+plot.Id+"/"+s.Id,(shown?"On · ":"Off · ")+s.Label,new Vector2(box.x,-box.y),new Vector2(box.width,box.height),20*scale,()=>{string id=plot.Id+"/"+s.Id;if(!hidden.Remove(id))hidden.Add(id);redraw();});StatisticsWidgets.Selected(button,shown);
                var text=button.GetComponentInChildren<TextMeshProUGUI>();text.color=StatisticsTheme.Ink;text.rectTransform.offsetMin=new Vector2(30*scale,2*scale);text.enableWordWrapping=true;text.overflowMode=TextOverflowModes.Overflow;
                var glyph=StatisticsWidgets.Rect(button.transform,"Legend symbol",new Vector2(7,-8)*scale,new Vector2(18,18)*scale).gameObject.AddComponent<StatisticsLegendGlyph>();glyph.Identity=s.Id;glyph.color=StatisticsPlotPalette.Color(s);glyph.raycastTarget=false;
                var tip=button.gameObject.AddComponent<StatisticsTooltip>();tip.Panel=panel;tip.Description=s.Label+" · "+s.Role+" encoding\nToggle visibility; numerical scales and percentage denominators retain all series.";
            }
            UnityEngine.Object.Destroy(measuring.gameObject);return root.sizeDelta.y+12*unit;
        }
        static void HeatKey(Transform parent,TMP_FontAsset font,StatisticsPlotGraphic graphic,float scale,int players,float visibleWidth)
        {
            var plot=graphic.Plot;float left=graphic.Area.xMin,available=Math.Min(graphic.Area.width,visibleWidth-left-28*scale);
            if(plot.DiscreteHeat){
                for(int i=0;i<players;i++){
                    float x=left+i*available/players;var glyph=StatisticsWidgets.Rect(parent,"Position key "+(i+1),new Vector2(x,-8*scale),new Vector2(18,18)*scale).gameObject.AddComponent<StatisticsLegendGlyph>();glyph.Identity="seat"+(i+1);glyph.color=StatisticsPlotPalette.Color(new StatisticsPlotSeries{Id=glyph.Identity});glyph.raycastTarget=false;
                    Label(parent,font,"Position label "+(i+1),new[]{"1st · first","2nd","3rd","4th"}[i],new Vector2(x+25*scale,-2*scale),new Vector2(available/players-30*scale,32*scale),20*scale,false,TextAlignmentOptions.MidlineLeft);
                }
                return;
            }
            double max=plot.Marks.Where(m=>!m.Missing&&!Double.IsNaN(m.Y)).Select(m=>Math.Abs(m.Y)).DefaultIfEmpty().Max();
            float barWidth=Math.Min(210*scale,available*.28f),start=left+34*scale;
            Label(parent,font,"Heat scale zero","0",new Vector2(left,-2*scale),new Vector2(26*scale,32*scale),20*scale);
            var tint=StatisticsPlotPalette.Color(plot.Series.FirstOrDefault()??new StatisticsPlotSeries());
            for(int i=0;i<24;i++){var strip=StatisticsWidgets.Rect(parent,"Heat scale band",new Vector2(start+i*barWidth/24,-9*scale),new Vector2(barWidth/24+1,14*scale));var fill=strip.gameObject.AddComponent<Image>();fill.color=Color.Lerp(StatisticsTheme.Alternate,tint,i/23f);fill.raycastTarget=false;}
            Label(parent,font,"Heat scale",max.ToString("0.##")+" "+(plot.HeatUnit??(plot.Id=="industry-vp"?"VP":"units"))+" · shared scale",new Vector2(start+barWidth+12*scale,-2*scale),new Vector2(Math.Max(1,available-barWidth-46*scale),32*scale),20*scale,false,TextAlignmentOptions.MidlineLeft);
        }
        static void DirectLabels(Transform parent,TMP_FontAsset font,StatisticsPlotGraphic graphic,float scale)
        {
            var plot=graphic.Plot;var marks=plot.Marks.Where(m=>!graphic.Hidden.Contains(m.Series)&&m.Row>=0).ToArray();
            if(plot.Kind==StatisticsPlotKind.StackedBars)foreach(var m in marks.Where(m=>!m.Missing&&m.Y>0)){
                var bounds=graphic.MarkBounds(m);double total=plot.Marks.Where(x=>x.Row==m.Row&&x.Facet==m.Facet).Sum(x=>x.Y);
                string value=plot.Percent?(total>0?(m.Y*100/total).ToString("0.#")+"%":""):m.Y.ToString("0.##");
                var label=Label(parent,font,"Segment value",value,new Vector2(bounds.xMin+4*scale,bounds.center.y+14*scale),new Vector2(Math.Max(1,bounds.width-8*scale),28*scale),18*scale);
                label.color=StatisticsTheme.ReadableOn(StatisticsPlotPalette.Color(plot.Series.First(s=>s.Id==m.Series)));
                if(label.GetPreferredValues(value,0,0).x>bounds.width-12*scale)label.gameObject.SetActive(false);
            }
            if(plot.Kind==StatisticsPlotKind.StackedBars){foreach(var row in plot.Marks.GroupBy(m=>new{m.Row,m.Facet}).Where(g=>g.Key.Row>=0)){
                double total=row.Sum(m=>m.Y);float x=graphic.X(plot.Percent&&total>0?100:row.Where(m=>m.Y>0).Sum(m=>m.Y));
                float cy=graphic.Area.yMax-(row.Key.Row+.5f)*graphic.RowHeight+(plot.EraPairs?(row.Key.Facet==1?13:-13)*scale:0);
                string value=plot.Percent?(total>0?total.ToString("0")+" total":"—"):StatisticsAxisLayout.Format(total,plot.XUnit);if(plot.Id=="link-sources")value+= " VP\n"+row.First().OtherLabel;if(plot.EraPairs)value=(row.Key.Facet==1?"C ":"R ")+value;
                Label(parent,font,"Stack total",value,new Vector2(x+8*scale,cy+16*scale),new Vector2((plot.Id=="link-sources"?225:95)*scale,(plot.Id=="link-sources"?52:32)*scale),(plot.Id=="link-sources"?19:20)*scale,plot.Id=="link-sources",TextAlignmentOptions.MidlineLeft);
            }}
            if(plot.Kind==StatisticsPlotKind.StackedColumns){foreach(var row in plot.Marks.GroupBy(m=>m.Row)){double total=row.Sum(m=>m.Y);Label(parent,font,"Development total",total.ToString("0"),new Vector2(graphic.X(row.Key)-30*scale,graphic.Y(total)+34*scale),new Vector2(60*scale,32*scale),21*scale);}}
            if(plot.Kind==StatisticsPlotKind.Bullet||plot.Kind==StatisticsPlotKind.PairedLollipop||plot.Kind==StatisticsPlotKind.PairedBars||plot.Kind==StatisticsPlotKind.Waterfall||plot.Kind==StatisticsPlotKind.Lollipop||plot.Kind==StatisticsPlotKind.Dumbbell)foreach(var m in marks){
                var q=graphic.Point(m);string value=m.Missing?m.ValueLabel??"—":m.ValueLabel??StatisticsAxisLayout.Format(m.Y,plot.XUnit,1,true);
                if(plot.Id=="conversion"&&m.Series=="flipped"){double built=plot.Marks.Where(x=>x.Row==m.Row&&x.Series=="built").Sum(x=>x.Y);value+=" · "+StatisticsFormat.Percent(m.Y,built);}
                if(plot.Kind==StatisticsPlotKind.Waterfall&&m.Y>0&&m.Series!="total")value="+"+value;
                bool market=plot.Id.StartsWith("market-");if(market&&m.OtherLabel!=null)value+="\n"+m.OtherLabel;
                if(plot.Kind==StatisticsPlotKind.Bullet)value+="\n"+m.OtherLabel;float w=plot.Kind==StatisticsPlotKind.Bullet?305*scale:plot.Id=="conversion"?185*scale:market?190*scale:plot.Kind==StatisticsPlotKind.Dumbbell?80*scale:plot.XUnit==StatisticsAxisUnit.Price||plot.XUnit==StatisticsAxisUnit.Ratio?165*scale:95*scale;
                float labelY=q.y+(plot.Kind==StatisticsPlotKind.Bullet?30:16)*scale;if(plot.Kind==StatisticsPlotKind.Dumbbell)labelY+=plot.Series.IndexOf(plot.Series.First(s=>s.Id==m.Series))==0?18*scale:-18*scale;
                var label=Label(parent,font,"Direct value",value,new Vector2((plot.Kind==StatisticsPlotKind.Bullet?graphic.Area.xMax+12*scale:Math.Min(q.x+10*scale,graphic.Area.xMax+10*scale)),labelY),new Vector2(w,(plot.Kind==StatisticsPlotKind.Bullet?60:market?50:32)*scale),(market?19:20)*scale,market||plot.Kind==StatisticsPlotKind.Bullet,TextAlignmentOptions.MidlineLeft);
                if(plot.Kind==StatisticsPlotKind.Waterfall&&m.Y<0){var background=StatisticsWidgets.Rect(parent,"Signed value surface",label.rectTransform.anchoredPosition-new Vector2(3*scale,0),new Vector2(label.GetPreferredValues(value,0,0).x+10*scale,32*scale));var surface=background.gameObject.AddComponent<StatisticsSurface>();surface.Panel=true;surface.raycastTarget=false;background.SetSiblingIndex(label.transform.GetSiblingIndex());}
            }
            if(plot.Kind==StatisticsPlotKind.Dumbbell)foreach(var row in marks.GroupBy(m=>m.Row).Where(g=>g.Count()==2)){double difference=row.Last().Y-row.First().Y;Label(parent,font,"Era difference","Δ "+(difference>0?"+":"")+StatisticsAxisLayout.Format(difference,StatisticsAxisUnit.Count)+" VP",new Vector2(graphic.Area.xMax+95*scale,graphic.Area.yMax-(row.Key+.5f)*graphic.RowHeight+16*scale),new Vector2(125*scale,32*scale),20*scale,false,TextAlignmentOptions.MidlineLeft);}
            if(plot.Kind==StatisticsPlotKind.Step){
                var placed=new List<Rect>();foreach(var series in marks.GroupBy(m=>m.Series)){
                    var last=series.Last();var q=graphic.Point(last);float labelY=q.y+(plot.Kind==StatisticsPlotKind.Bullet?30:16)*scale;Rect chosen=new Rect();
                    foreach(float offset in new[]{0f,36,-36,72,-72,108,-108,144,-144,180,-180}){
                        float candidateY=Mathf.Clamp(labelY+offset*scale,graphic.Area.yMin+34*scale,graphic.Area.yMax+16*scale);var candidate=new Rect(graphic.Area.xMax+8*scale,candidateY-32*scale,132*scale,32*scale);
                        if(!placed.Any(p=>p.Overlaps(candidate))){chosen=candidate;break;}
                    }
                    placed.Add(chosen);var label=Label(parent,font,"Curve endpoint",StatisticsAxisLayout.Format(last.Y,plot.YUnit),new Vector2(chosen.x,chosen.yMax),chosen.size,20*scale,false,TextAlignmentOptions.MidlineLeft);label.color=StatisticsPlotPalette.Color(plot.Series.Single(s=>s.Id==last.Series));
                    Vector2 end=new Vector2(chosen.x-3*scale,chosen.center.y);var delta=end-q;var leader=StatisticsWidgets.Rect(parent,"Curve endpoint leader",q,new Vector2(delta.magnitude,1.5f*scale));leader.pivot=new Vector2(0,.5f);leader.localEulerAngles=new Vector3(0,0,Mathf.Atan2(delta.y,delta.x)*Mathf.Rad2Deg);var image=leader.gameObject.AddComponent<Image>();image.color=label.color;image.raycastTarget=false;
                }
            }
            if(plot.Kind==StatisticsPlotKind.DelayDots)foreach(var row in plot.Marks.GroupBy(m=>m.Row)){
                var ordered=row.Select(m=>m.Y).OrderBy(x=>x).ToArray();if(ordered.Length==0)continue;double median=(ordered[(ordered.Length-1)/2]+ordered[ordered.Length/2])/2;
                Label(parent,font,"Delay sample",ordered.Length+" tiles · median "+median.ToString("0.#"),new Vector2(graphic.Area.xMax+8*scale,graphic.Area.yMax-(row.Key+.5f)*graphic.RowHeight+22*scale),new Vector2(140*scale,50*scale),19*scale,true,TextAlignmentOptions.MidlineLeft);
                var tick=StatisticsWidgets.Rect(parent,"Delay median",new Vector2(graphic.X(median),graphic.Area.yMax-(row.Key+.5f)*graphic.RowHeight+14*scale),new Vector2(2*scale,28*scale));var image=tick.gameObject.AddComponent<Image>();image.color=StatisticsTheme.Ink;image.raycastTarget=false;
            }
        }
    }
}
