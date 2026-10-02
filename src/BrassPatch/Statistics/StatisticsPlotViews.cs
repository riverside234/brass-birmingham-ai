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
            if(s.Player.HasValue)return StatisticsWidgets.PlayerColor(s.Player.Value);
            if(s.Id.StartsWith("region/")){switch(s.Id.Substring(7)){case "Violet":return new Color(.76f,.57f,.9f);case "Red":return new Color(.92f,.43f,.36f);case "Yellow":return new Color(.96f,.79f,.29f);case "Blue":return new Color(.4f,.7f,.94f);case "Cyan":return new Color(.37f,.82f,.85f);default:return new Color(.70f,.70f,.70f);}}
            switch(s.Id){
                case "positive":case "flipped":case "Own":case "Other flipped":return new Color(.36f,.79f,.67f);
                case "negative":case "Penalty":case "Unsold goods":case "Removed unflipped":return new Color(.94f,.48f,.43f);
                case "total":case "built":return StatisticsWidgets.Gold;
                case "Canal":case "Coal":case "Market":return new Color(.43f,.69f,.88f);
                case "Rail":case "Beer":case "Merchant":return new Color(.95f,.72f,.36f);
                case "Iron":case "Opponents":return new Color(.76f,.61f,.90f);
                case "unflipped":case "Unflipped supplier":return new Color(.68f,.70f,.72f);
                case "Scored both eras":return new Color(.45f,.78f,.90f);
            }
            if(s.Id.EndsWith("/Penalty"))return new Color(.94f,.48f,.43f);
            uint hash=2166136261;foreach(char c in s.Id)hash=(hash^c)*16777619;
            Color[] palette={new Color(.43f,.72f,.87f),new Color(.92f,.68f,.37f),new Color(.64f,.79f,.52f),new Color(.77f,.62f,.88f),new Color(.91f,.55f,.64f),new Color(.56f,.81f,.76f),new Color(.83f,.76f,.56f),new Color(.65f,.70f,.84f)};
            return palette[hash%palette.Length];
        }
        public static int Shape(string id){uint hash=0;foreach(char c in id)hash=hash*31+c;return (int)(hash%4);}
    }
    public sealed class StatisticsPlotGraphic : MaskableGraphic, IPointerEnterHandler, IPointerExitHandler, IPointerClickHandler
    {
        public StatisticsPlot Plot;
        public readonly HashSet<string> Hidden=new HashSet<string>();
        public StatisticsPanel Panel;
        public Rect Area;
        public double XMin,XMax,YMin,YMax;
        public float Unit=1,RowHeight;
        PointerEventData hover;
        Vector2 lastPointer;
        public bool Horizontal {get{return Plot.Kind==StatisticsPlotKind.StackedBars||Plot.Kind==StatisticsPlotKind.PairedBars||Plot.Kind==StatisticsPlotKind.Lollipop||Plot.Kind==StatisticsPlotKind.Dumbbell||Plot.Kind==StatisticsPlotKind.Waterfall||Plot.Kind==StatisticsPlotKind.DelayDots;}}
        public Vector2 Point(StatisticsPlotMark m)
        {
            float y=Area.yMax-(m.Row+.5f)*RowHeight;
            if(Horizontal){double value=Plot.Kind==StatisticsPlotKind.Waterfall?m.X+m.Y:m.Y;
                if(Plot.Kind==StatisticsPlotKind.StackedBars){var preceding=Plot.Marks.TakeWhile(x=>x!=m).Where(x=>x.Row==m.Row&&Math.Sign(x.Y)==Math.Sign(m.Y));value=preceding.Sum(x=>x.Y)+m.Y;
                    if(Plot.Percent){double total=Plot.Marks.Where(x=>x.Row==m.Row).Sum(x=>x.Y);value=total<=0?0:value*100/total;}}
                if(Plot.Kind==StatisticsPlotKind.PairedBars){int s=Plot.Series.FindIndex(x=>x.Id==m.Series);y+=((Plot.Series.Count-1)/2f-s)*Math.Min(12*Unit,RowHeight/(Plot.Series.Count+1));}
                if(Plot.Kind==StatisticsPlotKind.DelayDots)y+=((Plot.Marks.IndexOf(m)%5)-2)*3.5f*Unit;
                return new Vector2(X(value),y);
            }
            if(Plot.Kind==StatisticsPlotKind.Heatmap)return new Vector2(Area.xMin+(m.Column+.5f)*Area.width/Math.Max(1,Plot.Columns.Count),y);
            if(Plot.Kind==StatisticsPlotKind.MiniMap)return new Vector2(Area.xMin+(float)m.X*Area.width,Area.yMin+(float)m.Y*Area.height);
            if(Plot.Kind==StatisticsPlotKind.EventLanes||Plot.Kind==StatisticsPlotKind.Lifespans){
                if(Plot.Kind==StatisticsPlotKind.EventLanes)y+=((Plot.Marks.TakeWhile(x=>x!=m).Count(x=>x.Row==m.Row&&x.X==m.X)%2)*2-1)*7*Unit;
                return new Vector2(X(m.X),y);
            }
            var point=new Vector2(X(m.X),Y(m.Y));
            if(Plot.Kind==StatisticsPlotKind.Columns){int s=Plot.Series.FindIndex(x=>x.Id==m.Series);point.x+=((s-(Plot.Series.Count-1)/2f))*Math.Min(9*Unit,Area.width/Math.Max(1,Plot.Columns.Count)/(Plot.Series.Count+1));}
            if(Plot.Kind==StatisticsPlotKind.Scatter){int overlap=Plot.Marks.TakeWhile(x=>x!=m).Count(x=>x.X==m.X&&x.Y==m.Y);point+=new Vector2(overlap%3,overlap/3)*3*Unit;}
            return point;
        }
        public float X(double value){return Area.xMin+(float)((value-XMin)/Math.Max(1e-8,XMax-XMin))*Area.width;}
        public float Y(double value){return Area.yMin+(float)((value-YMin)/Math.Max(1e-8,YMax-YMin))*Area.height;}
        Color Tint(string series){return StatisticsPlotPalette.Color(Plot.Series.FirstOrDefault(s=>s.Id==series)??new StatisticsPlotSeries{Id=series});}
        protected override void OnPopulateMesh(VertexHelper vh)
        {
            vh.Clear();if(Plot==null)return;
            if(Plot.Kind==StatisticsPlotKind.MiniMap){foreach(var m in Plot.Marks){var location=Panel.Report.Locations.FirstOrDefault(l=>l.City==m.Label);if(location==null)continue;foreach(var neighbor in location.Neighbors){var other=Plot.Marks.FirstOrDefault(x=>x.Label==neighbor);if(other!=null&&String.CompareOrdinal(m.Label,other.Label)<0)Line(vh,Point(m),Point(other),new Color(.57f,.53f,.43f,.30f),1.3f*Unit);}}}
            else if(Plot.Kind!=StatisticsPlotKind.Heatmap){
                if(Horizontal||Plot.Kind==StatisticsPlotKind.EventLanes||Plot.Kind==StatisticsPlotKind.Lifespans){for(int i=0;i<=4;i++)Line(vh,new Vector2(Area.xMin+Area.width*i/4,Area.yMin),new Vector2(Area.xMin+Area.width*i/4,Area.yMax),new Color(.72f,.68f,.56f,.16f),Unit);}
                else {for(int i=0;i<=4;i++)Line(vh,new Vector2(Area.xMin,Area.yMin+Area.height*i/4),new Vector2(Area.xMax,Area.yMin+Area.height*i/4),new Color(.72f,.68f,.56f,.16f),Unit);}
                if(Horizontal)Line(vh,new Vector2(X(0),Area.yMin),new Vector2(X(0),Area.yMax),new Color(.8f,.77f,.68f,.7f),1.6f*Unit);
                else if(Plot.Kind!=StatisticsPlotKind.Lifespans&&Plot.Kind!=StatisticsPlotKind.EventLanes)Line(vh,new Vector2(Area.xMin,Y(0)),new Vector2(Area.xMax,Y(0)),new Color(.8f,.77f,.68f,.7f),1.6f*Unit);
            }
            foreach(var m in Plot.Marks.Where(m=>!Hidden.Contains(m.Series)&&(!Horizontal||m.Row>=0))){Vector2 q=Point(m);Color tint=Tint(m.Series);int shape=StatisticsPlotPalette.Shape(m.Series);
                if(Plot.Kind==StatisticsPlotKind.Heatmap){float max=(float)Plot.Marks.Select(x=>x.Y).DefaultIfEmpty(1).Max();float intensity=max<=0?0:(float)m.Y/max;var fill=UnityEngine.Color.Lerp(new Color(.15f,.16f,.17f),tint,.12f+.78f*intensity);Quad(vh,new Rect(q.x-Area.width/Math.Max(1,Plot.Columns.Count)/2+2*Unit,q.y-RowHeight/2+2*Unit,Area.width/Math.Max(1,Plot.Columns.Count)-4*Unit,RowHeight-4*Unit),fill);}
                else if(Plot.Kind==StatisticsPlotKind.StackedBars){double total=Plot.Marks.Where(x=>x.Row==m.Row).Sum(x=>x.Y);double size=Plot.Percent?(total==0?0:m.Y*100/total):m.Y;float start=q.x-(X(size)-X(0));Quad(vh,new Rect(Math.Min(start,q.x),q.y-12*Unit,Math.Abs(q.x-start),24*Unit),tint);}
                else if(Plot.Kind==StatisticsPlotKind.PairedBars||Plot.Kind==StatisticsPlotKind.Waterfall){float start=Plot.Kind==StatisticsPlotKind.Waterfall?X(m.X):X(0);float thickness=Plot.Kind==StatisticsPlotKind.PairedBars?Math.Min(10*Unit,RowHeight/(Plot.Series.Count+1)-2*Unit):22*Unit;Quad(vh,new Rect(Math.Min(start,q.x),q.y-thickness/2,Math.Abs(q.x-start),Math.Max(2*Unit,thickness)),tint);}
                else if(Plot.Kind==StatisticsPlotKind.Columns){float start=Y(0);Quad(vh,new Rect(q.x-3.5f*Unit,Math.Min(start,q.y),7*Unit,Math.Abs(q.y-start)),tint);}
                else if(Plot.Kind==StatisticsPlotKind.Step){var next=Plot.Marks.SkipWhile(x=>x!=m).Skip(1).FirstOrDefault(x=>x.Series==m.Series);if(next!=null){var to=Point(next);Line(vh,q,new Vector2(to.x,q.y),tint,2.6f*Unit);Line(vh,new Vector2(to.x,q.y),to,tint,2.6f*Unit);}}
                else if(Plot.Kind==StatisticsPlotKind.Dumbbell){var pair=Plot.Marks.FirstOrDefault(x=>x.Row==m.Row&&x!=m&&!Hidden.Contains(x.Series));if(pair!=null)Line(vh,q,Point(pair),new Color(.8f,.77f,.68f,.5f),2*Unit);Symbol(vh,q,7*Unit,tint,shape);}
                else if(Plot.Kind==StatisticsPlotKind.Lollipop){Line(vh,new Vector2(X(0),q.y),q,tint,2.5f*Unit);Symbol(vh,q,7*Unit,tint,shape);}
                else if(Plot.Kind==StatisticsPlotKind.Lifespans){Line(vh,q,new Vector2(X(m.End),q.y),tint,4*Unit);Symbol(vh,q,5*Unit,tint,1);Symbol(vh,new Vector2(X(m.End),q.y),6*Unit,tint,m.Series=="flipped"?0:2);}
                else if(Plot.Kind==StatisticsPlotKind.MiniMap){Symbol(vh,q,Math.Min(18,8+m.Row*2)*Unit,tint,m.Series=="merchant"?3:m.Series=="farm"?2:shape);}
                else Symbol(vh,q,5.5f*Unit,tint,shape);
            }
        }
        static void Quad(VertexHelper vh,Rect rect,Color color){int i=vh.currentVertCount;vh.AddVert(new Vector3(rect.xMin,rect.yMin),color,Vector2.zero);vh.AddVert(new Vector3(rect.xMin,rect.yMax),color,Vector2.zero);vh.AddVert(new Vector3(rect.xMax,rect.yMax),color,Vector2.zero);vh.AddVert(new Vector3(rect.xMax,rect.yMin),color,Vector2.zero);vh.AddTriangle(i,i+1,i+2);vh.AddTriangle(i,i+2,i+3);}
        static void Line(VertexHelper vh,Vector2 a,Vector2 b,Color color,float width){var normal=new Vector2(-(b-a).y,(b-a).x).normalized*width/2;int i=vh.currentVertCount;foreach(var p in new[]{a-normal,a+normal,b+normal,b-normal})vh.AddVert(p,color,Vector2.zero);vh.AddTriangle(i,i+1,i+2);vh.AddTriangle(i,i+2,i+3);}
        internal static void Symbol(VertexHelper vh,Vector2 center,float size,Color color,int shape){int count=shape==0?12:shape==2?3:4;float rotation=shape==1?Mathf.PI/4:0;int i=vh.currentVertCount;vh.AddVert(center,color,Vector2.zero);for(int n=0;n<=count;n++){float a=rotation+Mathf.PI*2*n/count;vh.AddVert(center+new Vector2(Mathf.Sin(a),Mathf.Cos(a))*size,color,Vector2.zero);if(n>0)vh.AddTriangle(i,i+n,i+n+1);}}
        StatisticsPlotMark Hit(PointerEventData data){Vector2 local;if(!RectTransformUtility.ScreenPointToLocalPointInRectangle(rectTransform,data.position,data.pressEventCamera??data.enterEventCamera,out local))return null;return Plot.Marks.Where(m=>!Hidden.Contains(m.Series)&&(!Horizontal||m.Row>=0)).OrderBy(m=>(Point(m)-local).sqrMagnitude).FirstOrDefault(m=>(Point(m)-local).sqrMagnitude<Math.Pow((Plot.Kind==StatisticsPlotKind.Heatmap?Math.Min(RowHeight/2,Area.width/Math.Max(1,Plot.Columns.Count)/2):18*Unit),2));}
        public void OnPointerMove(PointerEventData data){var mark=Hit(data);Panel.Tooltip(mark==null?null:Description(mark));}
        public void OnPointerEnter(PointerEventData data){hover=data;lastPointer=data.position;OnPointerMove(data);}
        void Update(){if(hover==null)return;Vector2 pointer=Input.mousePosition;if(pointer==lastPointer)return;lastPointer=pointer;hover.position=pointer;OnPointerMove(hover);}
        public void OnPointerExit(PointerEventData data){hover=null;Panel.Tooltip(null);}
        public void OnPointerClick(PointerEventData data){var mark=Hit(data);if(mark!=null&&mark.Target!=null)Panel.Drill(mark.Target);}
        public string Description(StatisticsPlotMark mark){var series=Plot.Series.FirstOrDefault(s=>s.Id==mark.Series);return mark.Label+" · "+(series==null?mark.Series:series.Label)+"\n"+mark.Y.ToString("0.##")+" · "+(Horizontal?Plot.XLabel:Plot.YLabel)+(Plot.Kind==StatisticsPlotKind.Scatter?" · cost £"+mark.X.ToString("0.##"):"")+"\n"+mark.Detail+"\nOpen the exact-data ledger for all values and supporting records.";}
    }
    public sealed class StatisticsLegendGlyph : MaskableGraphic
    {
        public string Identity;
        public int ShapeOverride=-1;
        protected override void OnPopulateMesh(VertexHelper mesh){mesh.Clear();var rect=GetPixelAdjustedRect();StatisticsPlotGraphic.Symbol(mesh,rect.center,Mathf.Min(rect.width,rect.height)*.42f,color,ShapeOverride<0?StatisticsPlotPalette.Shape(Identity??""):ShapeOverride);}
    }
    internal static class StatisticsPlotView
    {
        public static float Create(Transform parent,TMP_FontAsset font,StatisticsPlot plot,float y,float width,float unit,float textScale,StatisticsPanel panel,HashSet<string> hidden,Action redraw)
        {
            float scale=unit*textScale;
            bool horizontal=plot.Kind==StatisticsPlotKind.StackedBars||plot.Kind==StatisticsPlotKind.PairedBars||plot.Kind==StatisticsPlotKind.Lollipop||plot.Kind==StatisticsPlotKind.Dumbbell||plot.Kind==StatisticsPlotKind.Waterfall||plot.Kind==StatisticsPlotKind.DelayDots;
            bool rows=horizontal||plot.Kind==StatisticsPlotKind.Heatmap||plot.Kind==StatisticsPlotKind.EventLanes||plot.Kind==StatisticsPlotKind.Lifespans;
            float rowHeight=plot.Kind==StatisticsPlotKind.Heatmap?50*scale:Math.Max(48,plot.Kind==StatisticsPlotKind.PairedBars?plot.Series.Count*14:48)*scale;
            float graphHeight=rows?Math.Max(140*scale,plot.Rows.Count*rowHeight):plot.Kind==StatisticsPlotKind.MiniMap?640*scale:300*scale;
            float labelWidth=rows?Math.Min(width*.36f,250*scale):64*scale;
            float graphWidth=width;
            if(plot.Kind==StatisticsPlotKind.Heatmap)graphWidth=Math.Max(width,labelWidth+Math.Max(1,plot.Columns.Count)*76*scale+20*scale);
            if(plot.Kind==StatisticsPlotKind.MiniMap)graphWidth=Math.Max(width,1440*scale);
            float legendY=10*scale;
            float legendRows=Math.Max(1,(float)Math.Ceiling(plot.Series.Count/Math.Max(1,Math.Floor((width-24*scale)/(232*scale)))));
            float legendSpace=legendRows*44*scale+16*scale;
            float axisSpace=(plot.Kind==StatisticsPlotKind.Heatmap?170:100)*scale;
            float total=legendSpace+graphHeight+axisSpace;
            var root=StatisticsWidgets.Rect(parent,"Plot "+plot.Id,new Vector2(0,-y),new Vector2(width,total));
            var backdrop=root.gameObject.AddComponent<StatisticsSurface>();backdrop.Panel=true;backdrop.raycastTarget=false;
            Transform container=StatisticsWidgets.Rect(root,"Plot frame",new Vector2(0,-legendSpace),new Vector2(width,graphHeight+axisSpace-10*scale));
            if(graphWidth>width+1){var scroll=StatisticsWidgets.Scroll(container,"Plot horizontal scroll",Vector2.zero,new Vector2(width,graphHeight+axisSpace-10*scale),unit,true);scroll.content.sizeDelta=new Vector2(graphWidth,graphHeight+axisSpace-10*scale-14*unit);container=scroll.content;}
            var chartRect=StatisticsWidgets.Rect(container,"Plot marks",Vector2.zero,new Vector2(graphWidth,graphHeight+axisSpace-10*scale));
            var graphic=chartRect.gameObject.AddComponent<StatisticsPlotGraphic>();graphic.Plot=plot;graphic.Panel=panel;graphic.Unit=scale;graphic.RowHeight=rowHeight;
            graphic.Area=new Rect(labelWidth,-graphHeight-26*scale,graphWidth-labelWidth-28*scale,graphHeight);
            if(plot.Kind==StatisticsPlotKind.Heatmap)graphic.Area=new Rect(labelWidth,-graphHeight-90*scale,graphWidth-labelWidth-28*scale,graphHeight);
            if(plot.Kind==StatisticsPlotKind.MiniMap)graphic.Area=new Rect(45*scale,-graphHeight-20*scale,graphWidth-140*scale,graphHeight-65*scale);
            foreach(var s in plot.Series)if(hidden.Contains(plot.Id+"/"+s.Id))graphic.Hidden.Add(s.Id);
            var marks=plot.Marks.Where(m=>!horizontal||m.Row>=0).ToList();double xMin=0,xMax=1,yMin=0,yMax=1;
            if(horizontal){
                if(plot.Kind==StatisticsPlotKind.StackedBars){xMin=Math.Min(0,marks.GroupBy(m=>m.Row).Select(g=>g.Where(m=>m.Y<0).Sum(m=>m.Y)).DefaultIfEmpty().Min());xMax=plot.Percent?100:Math.Max(1,marks.GroupBy(m=>m.Row).Select(g=>g.Where(m=>m.Y>0).Sum(m=>m.Y)).DefaultIfEmpty().Max());}
                else if(plot.Kind==StatisticsPlotKind.Waterfall){xMin=Math.Min(0,marks.SelectMany(m=>new[]{m.X,m.X+m.Y}).DefaultIfEmpty().Min());xMax=Math.Max(1,marks.SelectMany(m=>new[]{m.X,m.X+m.Y}).DefaultIfEmpty().Max());}
                else{xMin=Math.Min(0,marks.Select(m=>m.Y).DefaultIfEmpty().Min());xMax=Math.Max(1,marks.Select(m=>m.Y).DefaultIfEmpty().Max());}
            }else{xMin=Math.Min(0,marks.Select(m=>m.X).DefaultIfEmpty().Min());xMax=Math.Max(1,marks.Select(m=>plot.Kind==StatisticsPlotKind.Lifespans?m.End:m.X).DefaultIfEmpty().Max());yMin=Math.Min(0,marks.Select(m=>m.Y).DefaultIfEmpty().Min());yMax=Math.Max(1,marks.Select(m=>m.Y).DefaultIfEmpty().Max());}
            graphic.XMin=xMin;graphic.XMax=xMax;graphic.YMin=yMin;graphic.YMax=yMax;
            Action<string,string,float,float,float,float,bool> label=(name,value,x,ly,w,h,wrap)=>{var t=StatisticsWidgets.Text(container,font,name,value,new Vector2(x,ly),new Vector2(w,h),19*scale,wrap);t.alignment=TextAlignmentOptions.Center;t.overflowMode=TextOverflowModes.Ellipsis;};
            if(rows)for(int i=0;i<plot.Rows.Count;i++){var t=StatisticsWidgets.Text(container,font,"Axis row "+i,plot.Rows[i],new Vector2(12*scale,graphic.Area.yMax-(i+.5f)*rowHeight+rowHeight/2),new Vector2(labelWidth-20*scale,rowHeight),19*scale,true);t.alignment=TextAlignmentOptions.MidlineRight;}
            if(plot.Kind==StatisticsPlotKind.Heatmap){
                for(int c=0;c<plot.Columns.Count;c++)label("Axis column "+c,plot.Columns[c],graphic.Area.xMin+c*graphic.Area.width/plot.Columns.Count,-32*scale,graphic.Area.width/plot.Columns.Count,48*scale,true);
                foreach(var m in marks.Where(m=>!graphic.Hidden.Contains(m.Series))){var point=graphic.Point(m);label("Heat value",m.Y.ToString("0.##"),point.x-35*scale,point.y+16*scale,70*scale,32*scale,false);}
                label("Heat scale","Shared intensity: 0 → "+marks.Select(m=>m.Y).DefaultIfEmpty().Max().ToString("0.##"),labelWidth,-2*scale,graphWidth-labelWidth,22*scale,false);
            }else if(plot.Kind==StatisticsPlotKind.MiniMap){foreach(var m in marks.Where(m=>!graphic.Hidden.Contains(m.Series))){var point=graphic.Point(m);label("City "+m.Label,CityLabel(m.Label)+"\n"+m.Row+" "+(plot.YLabel=="Owned tiles"?"tiles":"VP"),point.x-90*scale,point.y-14*scale,180*scale,52*scale,true);}}
            else{
                for(int tick=0;tick<=4;tick++){
                    double value=horizontal||plot.Kind==StatisticsPlotKind.Lifespans||plot.Kind==StatisticsPlotKind.EventLanes?xMin+(xMax-xMin)*tick/4:yMin+(yMax-yMin)*tick/4;
                    if(horizontal||plot.Kind==StatisticsPlotKind.Lifespans||plot.Kind==StatisticsPlotKind.EventLanes)label("Value tick "+tick,value.ToString("0.#"),graphic.Area.xMin+graphic.Area.width*tick/4-36*scale,-graphHeight-29*scale,72*scale,27*scale,false);
                    else label("Value tick "+tick,value.ToString("0.#"),0,graphic.Area.yMin+graphic.Area.height*tick/4+14*scale,labelWidth-8*scale,28*scale,false);
                }
                if(horizontal)for(int tick=0;tick<=4;tick++)label("Top value tick "+tick,(xMin+(xMax-xMin)*tick/4).ToString("0.#"),graphic.Area.xMin+graphic.Area.width*tick/4-36*scale,-2*scale,72*scale,22*scale,false);
                if(!horizontal&&plot.Kind!=StatisticsPlotKind.Lifespans&&plot.Kind!=StatisticsPlotKind.EventLanes){
                    if(plot.Columns.Count>0){for(int tick=0;tick<plot.Columns.Count;tick++)label("X category "+tick,plot.Columns[tick],graphic.X(tick)-28*scale,-graphHeight-29*scale,56*scale,27*scale,false);}
                    else for(int tick=0;tick<=4;tick++)label("X tick "+tick,(xMin+(xMax-xMin)*tick/4).ToString("0.#"),graphic.Area.xMin+graphic.Area.width*tick/4-36*scale,-graphHeight-29*scale,72*scale,27*scale,false);
                }
                label("X axis",plot.XLabel,graphic.Area.xMin,-graphHeight-60*scale,graphic.Area.width,30*scale,false);
                if(!horizontal&&!rows)label("Y axis",plot.YLabel,labelWidth,-2*scale,graphic.Area.width,22*scale,false);
            }
            // Keyboard proxies use exact values, never rounded coordinates. Every record is also in the paged data table.
            var keyboard=marks.Where(m=>!graphic.Hidden.Contains(m.Series)).ToArray();
            if(keyboard.Length>60)keyboard=keyboard.Where((m,i)=>i==0||i==keyboard.Length-1||i%Math.Max(1,keyboard.Length/40)==0).ToArray();
            foreach(var m in keyboard){var point=graphic.Point(m);var hit=StatisticsWidgets.Rect(chartRect,"Select plot value",point+new Vector2(-12,12)*scale,new Vector2(24,24)*scale);var image=hit.gameObject.AddComponent<Image>();image.color=new Color(1,1,1,0);var b=hit.gameObject.AddComponent<Button>();b.targetGraphic=image;
                var colors=b.colors;colors.highlightedColor=new Color(1,1,1,.25f);colors.selectedColor=colors.highlightedColor;colors.normalColor=new Color(1,1,1,0);b.colors=colors;
                var tip=hit.gameObject.AddComponent<StatisticsTooltip>();tip.Panel=panel;tip.Description=graphic.Description(m);
                b.onClick.AddListener(()=>{if(m.Target!=null)panel.Drill(m.Target);else panel.Tooltip(graphic.Description(m));});
            }
            float legendX=12*scale,legendTop=legendY;foreach(var s in plot.Series){float w=Math.Min(226*scale,width-24*scale);if(legendX+w>width-12*scale){legendX=12*scale;legendTop+=44*scale;}
                var b=StatisticsWidgets.Button(root,font,"Legend "+plot.Id+"/"+s.Id,s.Label,new Vector2(legendX,-legendTop),new Vector2(w,38*scale),18*scale,()=>{string id=plot.Id+"/"+s.Id;if(!hidden.Remove(id))hidden.Add(id);redraw();});
                var legendLabel=b.GetComponentInChildren<TextMeshProUGUI>();legendLabel.color=StatisticsPlotPalette.Color(s);legendLabel.rectTransform.offsetMin=new Vector2(32*scale,2*scale);legendLabel.enableWordWrapping=true;StatisticsWidgets.Selected(b,!graphic.Hidden.Contains(s.Id));
                var symbol=StatisticsWidgets.Rect(b.transform,"Legend symbol",new Vector2(8,-10)*scale,new Vector2(18,18)*scale).gameObject.AddComponent<StatisticsLegendGlyph>();symbol.Identity=s.Id;symbol.color=StatisticsPlotPalette.Color(s);symbol.raycastTarget=false;
                if(plot.Kind==StatisticsPlotKind.MiniMap&&s.Id=="merchant")symbol.ShapeOverride=3;
                if(plot.Kind==StatisticsPlotKind.MiniMap&&s.Id=="farm")symbol.ShapeOverride=2;
                var tip=b.gameObject.AddComponent<StatisticsTooltip>();tip.Panel=panel;tip.Description=s.Label+" · "+s.Role+" color\nToggle this series. Numerical scale stays fixed; exact-data tables retain all values.";legendX+=w+6*scale;
            }
            root.sizeDelta=new Vector2(width,total);return total+12*unit;
        }
        static string CityLabel(string value){return value=="SouthWest"?"South-west farm":value=="Central"?"Central farm":value.Replace("OnTrent"," on Trent");}
    }
}
