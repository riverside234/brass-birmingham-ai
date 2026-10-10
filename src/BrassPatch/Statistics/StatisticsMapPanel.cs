using System;
using System.Collections.Generic;
using System.Linq;
using TMPro;
using UnityEngine;
namespace BrassPatch.Statistics
{
    public sealed partial class StatisticsPanel
    {
        readonly Dictionary<string,float> networkPositions=new Dictionary<string,float>();
        public bool MapActive {get{return CurrentPage==3&&Filters.NetworkView=="Geography";}}
        public void SetNetworkView(string view)
        {
            if(view!="Network results")view="Geography";
            if(BodyScroll!=null)networkPositions[Filters.NetworkView]=BodyScroll.verticalNormalizedPosition;
            Filters.NetworkView=view;positions[3]=networkPositions.ContainsKey(view)?networkPositions[view]:1;
            CloseCategories();Tooltip(null);renderPending=true;
        }
        public void SetMapEra(int era){if(era!=1&&era!=2||era==2&&Report.Introductory)return;Filters.MapEra=era;QueueRender(false);}
        float NetworkTabs(float y)
        {
            float x=0;foreach(var value in new[]{"Geography","Network results"}){string tab=value;
                var b=StatisticsWidgets.Button(BodyScroll.content,font,"Network view "+tab,tab,new Vector2(x,-y),Size(210*TextScale,42*TextScale),F(20),()=>SetNetworkView(tab));StatisticsWidgets.Selected(b,Filters.NetworkView==tab);x+=220*unit*TextScale;
            }
            if(MapActive){float s=unit*TextScale/1.15f;
                StatisticsWidgets.Button(BodyScroll.content,font,"Map measure","View: "+Filters.MapMode,new Vector2(x+12*unit,-y),new Vector2(282*s,42*s),21*s,()=>{string[] values={"Occupancy","Industry VP","Link contributions"};Filters.MapMode=values[(Array.IndexOf(values,Filters.MapMode)+1)%values.Length];QueueRender(true);});
                StatisticsWidgets.Button(BodyScroll.content,font,"Printed routes","Printed routes: "+(Filters.PrintedRoutes?"On":"Off"),new Vector2(x+306*s,-y),new Vector2(205*s,42*s),21*s,()=>{Filters.PrintedRoutes=!Filters.PrintedRoutes;QueueRender(true);});
                var key=StatisticsWidgets.Button(BodyScroll.content,font,"Map key","Map key",new Vector2(x+523*s,-y),new Vector2(112*s,42*s),21*s,()=>Tooltip(MapKey));Tip(key,MapKey);
            }return 54*unit*TextScale;
        }
        const string MapKey="ERA-END BOARD\nBefore link removal and Canal level-I cleanup. Hover a location; click for exact records.\n\nMAP\nDot rim = board region. Diamond = merchant; triangle = farm. Lines and owner shapes match the player key. Pale routes are printed routes.\n\nCITY TAG\nOpen slot rings = vacancies; solid dots = occupied slots. Vacancies include all players, not build permission. Tile marker = owner; number = industry level. A check means flipped; a dash means unflipped. Cotton / Goods / Pottery / Beer / Coal / Iron identify the six industries.\n\nLINKS & VP\nLinks counts each owner's directly touching links, not all reachable players. Choose a VP view for that era's selected-player value and magnitude halos. Link VP may reward several incident links; it is separate from industry VP.\n\nPlayer and region choices emphasize the full board. Other detail filters are suspended; search affects only the ledger.";
        float MapContent(StatisticsPageView view,float start)
        {
            float y=start,x=0,s=unit*TextScale/1.15f;
            var snapshot=view.Map;
            if(snapshot==null||!snapshot.Available)return y-start+Paragraph(BodyScroll.content,"Map unavailable",snapshot==null?"Historical board unavailable.":snapshot.Problem,y,22);
            x=0;foreach(var p in Report.Players){
                var sample=StatisticsWidgets.Rect(BodyScroll.content,"Player route key "+p.Color,new Vector2(x,-y-12*s),new Vector2(32*s,5*s));var line=sample.gameObject.AddComponent<UnityEngine.UI.Image>();line.color=StatisticsTheme.Player(p.Color);line.raycastTarget=false;
                var glyph=StatisticsWidgets.Rect(BodyScroll.content,"Route owner symbol",new Vector2(x+9*s,-y-6*s),new Vector2(16*s,16*s)).gameObject.AddComponent<StatisticsMapGlyph>();glyph.Pixel=unit;glyph.Kind="Owner";glyph.Owner=p.Color;glyph.color=line.color;glyph.raycastTarget=false;
                var label=StatisticsWidgets.Text(BodyScroll.content,font,"Route owner "+p.Color,StatisticsMetrics.PlayerName(Report,p),new Vector2(x+40*s,-y),new Vector2((bodyWidth/Report.Players.Count)-50*s,32*s),21*s);label.color=line.color;x+=bodyWidth/Report.Players.Count;
            }y+=39*s;
            float height=Math.Max(490*unit,BodyScroll.viewport.rect.height-y-10*unit);
            var stage=StatisticsWidgets.Rect(BodyScroll.content,"Geography map stage",new Vector2(0,-y),new Vector2(bodyWidth,height));
            stage.gameObject.AddComponent<StatisticsSurface>().Panel=true;
            var graphic=StatisticsWidgets.Rect(stage,"Geography map",Vector2.zero,stage.sizeDelta);
            var map=graphic.gameObject.AddComponent<StatisticsMapView>();map.Initialize(this,font,snapshot,unit,s);y+=height+12*unit;
            if(Filters.Typed)y+=Paragraph(BodyScroll.content,"Map detail scope","Industry, action and evidence filters are suspended here. Region and player choices emphasize the complete historical board.",y,18);
            return y-start;
        }
    }
}
