using System;
using System.Linq;
using BoardGameRules.Entities.Players;
using TMPro;
using UnityEngine;
using UnityEngine.UI;
namespace BrassPatch.Statistics
{
    public sealed class StatisticsCityTag : MonoBehaviour
    {
        TextMeshProUGUI title,slots,empty,metric,footer,none;
        readonly RectTransform[] chips=new RectTransform[4];
        readonly TextMeshProUGUI[] industries=new TextMeshProUGUI[4],connectionCounts=new TextMeshProUGUI[4];
        readonly StatisticsMapGlyph[] owners=new StatisticsMapGlyph[4],symbols=new StatisticsMapGlyph[4],status=new StatisticsMapGlyph[4],vacancies=new StatisticsMapGlyph[4],connectionSymbols=new StatisticsMapGlyph[4];
        RectTransform rect;float scale,width;StatisticsPanel panel;StatisticsMapSnapshot snapshot;
        public float MaximumHeight {get;private set;}
        public string City {get;private set;}
        public RectTransform Rect {get{return rect;}}
        public void Initialize(StatisticsPanel host,TMP_FontAsset font,StatisticsMapSnapshot data,float s,float w)
        {
            panel=host;snapshot=data;scale=s;width=w;rect=(RectTransform)transform;
            // A two-column tile grid reserves the same maximum in both eras and all views.
            MaximumHeight=188*s;
            var surface=gameObject.AddComponent<StatisticsSurface>();surface.Panel=true;surface.Row=true;surface.Alternate=true;surface.raycastTarget=false;
            var input=gameObject.AddComponent<CanvasGroup>();input.blocksRaycasts=false;input.interactable=false;
            title=Text(transform,font,"City tag title",14,5,w/s-137,36,23);title.fontStyle=FontStyles.Bold;
            slots=Text(transform,font,"City slot vacancies",w/s-117,5,103,23,18);slots.alignment=TextAlignmentOptions.MidlineRight;
            for(int i=0;i<4;i++){vacancies[i]=Glyph(transform,"Slot occupancy "+i,w/s-84+i*18,30,10);}
            float cell=(w/s-40)/2;
            for(int i=0;i<4;i++){
                chips[i]=StatisticsWidgets.Rect(transform,"Industry tile "+i,new Vector2(14+(i%2)*(cell+12),-46-(i/2)*38)*s,new Vector2(cell,34)*s);
                var chip=chips[i].gameObject.AddComponent<StatisticsSurface>();chip.Row=true;chip.raycastTarget=false;
                owners[i]=Glyph(chips[i],"Tile owner "+i,7,11,13);owners[i].Kind="Owner";
                symbols[i]=Glyph(chips[i],"Industry icon "+i,27,5,24);
                industries[i]=Text(chips[i],font,"Tile industry "+i,59,1,cell-84,32,20);
                status[i]=Glyph(chips[i],"Flip status "+i,cell-23,9,17);
            }
            empty=Text(transform,font,"Empty city",14,44,w/s-28,32,20);empty.color=StatisticsTheme.Muted;
            metric=Text(transform,font,"City era metric",14,0,w/s-28,25,20);metric.color=StatisticsTheme.Accent;
            footer=Text(transform,font,"City connections",14,0,53,26,19);footer.text="Links";footer.color=StatisticsTheme.Muted;
            none=Text(transform,font,"No owned links",76,0,100,26,20);none.text="—";none.color=StatisticsTheme.Muted;
            for(int i=0;i<4;i++){
                connectionSymbols[i]=Glyph(transform,"Connection owner "+i,78+i*79,0,15);connectionSymbols[i].Kind="Owner";
                connectionCounts[i]=Text(transform,font,"Connection count "+i,100+i*79,0,40,26,20);
            }
            gameObject.SetActive(false);
        }
        TextMeshProUGUI Text(Transform parent,TMP_FontAsset font,string name,float x,float y,float w,float h,float size)
        {var t=StatisticsWidgets.Text(parent,font,name,"",new Vector2(x,-y)*scale,new Vector2(w,h)*scale,size*scale,true);t.overflowMode=TextOverflowModes.Overflow;return t;}
        StatisticsMapGlyph Glyph(Transform parent,string name,float x,float y,float size)
        {var g=StatisticsWidgets.Rect(parent,name,new Vector2(x,-y)*scale,new Vector2(size,size)*scale).gameObject.AddComponent<StatisticsMapGlyph>();g.Pixel=scale*1.15f/panel.TextScale;g.raycastTarget=false;return g;}
        static string Industry(string type){switch(type){case "CottonMill":return "Cotton";case "Manufacturer":return "Goods";case "IronWorks":return "Iron";case "CoalMine":return "Coal";case "Brewery":return "Beer";default:return type;}}
        Color OwnerColor(PlayerColor owner){var tint=StatisticsTheme.Player(owner);return panel.Filters.Players.Contains(owner)?tint:Color.Lerp(tint,StatisticsTheme.Paper,.46f);}
        public void Show(StatisticsMapCity city,Vector2 dot,float dotRadius,float stageWidth)
        {
            City=city.Location.City;title.text=StatisticsPlotPages.CityName(City);
            bool valid=city.Available;
            slots.text=city.Location.Merchant?"Merchant":!valid?"Unavailable":(city.Slots-city.Occupied)+" open";
            for(int i=0;i<4;i++){
                bool present=valid&&!city.Location.Merchant&&i<city.Slots;vacancies[i].gameObject.SetActive(present);
                if(present){vacancies[i].Kind=i<city.Occupied?"Occupied":"Vacant";vacancies[i].color=StatisticsTheme.Muted;vacancies[i].SetVerticesDirty();}
                present=valid&&i<city.Tiles.Length;chips[i].gameObject.SetActive(present);if(!present)continue;
                var a=city.Tiles[i];var tint=OwnerColor(a.Player);
                owners[i].Owner=a.Player;owners[i].color=tint;owners[i].SetVerticesDirty();
                industries[i].text=Industry(a.Type)+" "+a.Level;
                symbols[i].Kind=a.Type;symbols[i].color=tint;symbols[i].SetVerticesDirty();
                status[i].Kind=snapshot.Flipped(a)?"Check":"Open";status[i].color=snapshot.Flipped(a)?StatisticsTheme.Focus:StatisticsTheme.Muted;status[i].SetVerticesDirty();
            }
            empty.gameObject.SetActive(!valid||!city.Location.Merchant&&city.Tiles.Length==0);empty.text=!valid?city.Problem:"No industries";
            float emptyHeight=Math.Max(32*scale,empty.GetPreferredValues(empty.text,empty.rectTransform.rect.width,0).y+4*scale);
            empty.rectTransform.sizeDelta=new Vector2(empty.rectTransform.sizeDelta.x,emptyHeight);
            float bodyEnd=city.Location.Merchant?44:!valid||city.Tiles.Length==0?44+emptyHeight/scale:46+((city.Tiles.Length+1)/2)*38-4;
            bool score=panel.Filters.MapMode!="Occupancy";
            metric.gameObject.SetActive(score);
            if(score){
                bool industry=panel.Filters.MapMode=="Industry VP";
                int value=industry?StatisticsMapQueries.IndustryVP(snapshot,city,panel.Filters):StatisticsMapQueries.ContributionVP(panel.Report,snapshot,city,panel.Filters);
                int bonus=industry&&snapshot.Era==1?city.Tiles.Where(a=>panel.Filters.Players.Contains(a.Player)).Sum(a=>a.IntroVP):0;
                metric.text=(industry?"Industry VP: ":"Link VP: ")+(!valid&&industry?"—":value.ToString())+(bonus>0?"   + "+bonus+" bonus":"");
                metric.rectTransform.anchoredPosition=new Vector2(14,-bodyEnd-4)*scale;bodyEnd+=26;
            }
            float footerY=bodyEnd+8;
            footer.rectTransform.anchoredPosition=new Vector2(14,-footerY)*scale;
            none.rectTransform.anchoredPosition=new Vector2(76,-footerY)*scale;
            var linksByOwner=city.Links.GroupBy(l=>l.Player).OrderBy(g=>(int)g.Key).ToArray();none.gameObject.SetActive(linksByOwner.Length==0);
            for(int i=0;i<4;i++){
                bool present=i<linksByOwner.Length;connectionSymbols[i].gameObject.SetActive(present);connectionCounts[i].gameObject.SetActive(present);if(!present)continue;
                var owner=linksByOwner[i].Key;var tint=OwnerColor(owner);connectionSymbols[i].Owner=owner;connectionSymbols[i].color=tint;connectionSymbols[i].SetVerticesDirty();
                connectionCounts[i].text=linksByOwner[i].Select(l=>l.Key).Distinct().Count().ToString();connectionCounts[i].color=tint;
                connectionSymbols[i].rectTransform.anchoredPosition=new Vector2(78+i*79,-footerY-5)*scale;
                connectionCounts[i].rectTransform.anchoredPosition=new Vector2(100+i*79,-footerY)*scale;
            }
            rect.sizeDelta=new Vector2(width,(footerY+34)*scale);rect.anchoredPosition=new Vector2(Mathf.Clamp(dot.x-width/2,8*scale,Math.Max(8*scale,stageWidth-width-8*scale)),dot.y-dotRadius-12*scale);
            gameObject.SetActive(true);rect.SetAsLastSibling();
            foreach(var text in GetComponentsInChildren<TextMeshProUGUI>())text.ForceMeshUpdate();
            foreach(var graphic in GetComponentsInChildren<Graphic>())graphic.raycastTarget=false;
        }
        public void Hide(){City=null;gameObject.SetActive(false);}
    }
    public sealed class StatisticsMapGlyph : MaskableGraphic
    {
        public string Kind="Owner";public PlayerColor Owner;public float Pixel=1;
        protected override void OnPopulateMesh(VertexHelper mesh)
        {
            mesh.Clear();var r=rectTransform.rect;float s=Math.Min(r.width,r.height);var c=r.center;
            Action<float,float,float,float> line=(x,y,u,v)=>StatisticsMapView.Line(mesh,c+new Vector2(x,y)*s,c+new Vector2(u,v)*s,color,s*.085f,Pixel);
            if(Kind=="Occupied"||Kind=="Vacant"){StatisticsMapView.Symbol(mesh,c,s*.45f,color,0,Pixel);if(Kind=="Vacant")StatisticsMapView.Symbol(mesh,c,s*.29f,StatisticsTheme.Paper,0,Pixel);return;}
            if(Kind=="Owner"){StatisticsMapView.Symbol(mesh,c,s*.4f,color,StatisticsPlotPalette.Shape(Owner.ToString()),Pixel);return;}
            if(Kind=="Check"){line(-.35f,0,-.1f,-.26f);line(-.1f,-.26f,.38f,.3f);return;}
            if(Kind=="Open"){line(-.28f,0,.28f,0);return;}
            if(Kind=="Brewery"){line(-.27f,-.35f,-.37f,0);line(-.37f,0,-.27f,.35f);line(-.27f,.35f,.27f,.35f);line(.27f,.35f,.37f,0);line(.37f,0,.27f,-.35f);line(.27f,-.35f,-.27f,-.35f);line(-.3f,.16f,.3f,.16f);line(-.3f,-.16f,.3f,-.16f);return;}
            if(Kind=="CoalMine"||Kind=="IronWorks"){StatisticsMapView.Symbol(mesh,c,s*.36f,color,Kind=="CoalMine"?6:1,Pixel);return;}
            if(Kind=="Pottery"){line(-.12f,.4f,.12f,.4f);line(-.1f,.3f,-.3f,-.15f);line(.1f,.3f,.3f,-.15f);line(-.3f,-.15f,-.2f,-.35f);line(.3f,-.15f,.2f,-.35f);line(-.2f,-.35f,.2f,-.35f);return;}
            line(-.4f,-.3f,.4f,-.3f);line(-.4f,-.3f,-.4f,.1f);line(-.4f,.1f,-.12f,.3f);line(-.12f,.3f,-.12f,.1f);line(-.12f,.1f,.13f,.3f);line(.13f,.3f,.13f,.1f);line(.13f,.1f,.4f,.1f);line(.4f,.1f,.4f,-.3f);
            if(Kind=="CottonMill")line(.25f,.1f,.25f,.44f);else line(-.25f,-.1f,.23f,-.1f);
        }
    }
}
