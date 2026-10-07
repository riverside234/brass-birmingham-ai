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
        TextMeshProUGUI title,slots,empty,summary,footer,legend,unflippedLabel;
        TextMeshProUGUI[] owners,industries;
        StatisticsMapGlyph[] symbols,status;
        StatisticsMapGlyph[] connectionSymbols;
        TextMeshProUGUI[] connectionLabels;
        StatisticsMapGlyph flippedKey,openKey;
        RectTransform rect;float scale,width;int capacity;StatisticsPanel panel;StatisticsMapSnapshot snapshot;
        public float MaximumHeight {get;private set;}
        public string City {get;private set;}
        public RectTransform Rect {get{return rect;}}
        public void Initialize(StatisticsPanel host,TMP_FontAsset font,StatisticsMapSnapshot data,float s,float w)
        {
            // The board's maximum is four industry slots. Reserve the same card
            // height in BOTH eras, even when Canal has fewer occupied tiles.
            panel=host;snapshot=data;scale=s;width=w;rect=(RectTransform)transform;capacity=4;
            MaximumHeight=(160+capacity*32+(host.Report.Introductory?20:0))*s;
            var surface=gameObject.AddComponent<StatisticsSurface>();surface.Panel=true;surface.raycastTarget=false;
            var input=gameObject.AddComponent<CanvasGroup>();input.blocksRaycasts=false;input.interactable=false;
            title=Text(font,"City tag title",18,8,w/s-150,48,24);title.fontStyle=FontStyles.Bold;
            slots=Text(font,"City slot vacancies",w/s-137,10,122,44,20);slots.alignment=TextAlignmentOptions.MidlineRight;
            owners=new TextMeshProUGUI[capacity];industries=new TextMeshProUGUI[capacity];symbols=new StatisticsMapGlyph[capacity];status=new StatisticsMapGlyph[capacity];
            for(int i=0;i<capacity;i++){
                float y=54+i*32;
                owners[i]=Text(font,"Tile owner "+i,16,y,87,30,19);
                symbols[i]=Glyph("Industry icon "+i,104,y+4,22);
                industries[i]=Text(font,"Tile industry "+i,136,y,w/s-194,30,20);
                status[i]=Glyph("Flip status "+i,w/s-45,y+5,21);
            }
            empty=Text(font,"Empty city",18,54,w/s-36,34,20);
            summary=Text(font,"City era statistics",18,0,w/s-36,host.Report.Introductory?45:26,19);
            footer=Text(font,"City connections",18,0,101,26,18);footer.text="Connections";
            connectionSymbols=new StatisticsMapGlyph[4];connectionLabels=new TextMeshProUGUI[4];
            for(int i=0;i<4;i++){connectionSymbols[i]=Glyph("Connection owner "+i,119+i*82,0,10);connectionLabels[i]=Text(font,"Connection label "+i,132+i*82,0,69,26,17);}
            flippedKey=Glyph("Flipped legend",18,0,16);flippedKey.Kind="Check";flippedKey.color=StatisticsTheme.Focus;
            openKey=Glyph("Unflipped legend",122,0,16);openKey.Kind="Open";openKey.color=StatisticsTheme.Muted;
            legend=Text(font,"Flip key",40,0,80,25,17);legend.text="Flipped";legend.color=StatisticsTheme.Muted;
            unflippedLabel=Text(font,"Unflipped key",146,0,w/s-164,25,17);unflippedLabel.text="Unflipped at this era end";unflippedLabel.color=StatisticsTheme.Muted;
            gameObject.SetActive(false);
        }
        TextMeshProUGUI Text(TMP_FontAsset font,string name,float x,float y,float w,float h,float size)
        {var t=StatisticsWidgets.Text(transform,font,name,"",new Vector2(x,-y)*scale,new Vector2(w,h)*scale,size*scale,true);t.overflowMode=TextOverflowModes.Overflow;return t;}
        StatisticsMapGlyph Glyph(string name,float x,float y,float size)
        {var g=StatisticsWidgets.Rect(transform,name,new Vector2(x,-y)*scale,new Vector2(size,size)*scale).gameObject.AddComponent<StatisticsMapGlyph>();g.raycastTarget=false;return g;}
        static string Industry(string type){switch(type){case "CottonMill":return "Cotton mill";case "Manufacturer":return "Manufactory";case "IronWorks":return "Iron works";case "CoalMine":return "Coal mine";default:return type;}}
        public void Show(StatisticsMapCity city,Vector2 dot,float dotRadius,float stageWidth)
        {
            City=city.Location.City;title.text=StatisticsPlotPages.CityName(City);
            slots.text=city.Location.Merchant?"Merchant":!city.Available?"Unavailable":city.Slots==city.Occupied?"Full":(city.Slots-city.Occupied)+" of "+city.Slots+" open";
            bool valid=city.Available;
            for(int i=0;i<capacity;i++){
                bool present=valid&&i<city.Tiles.Length;owners[i].gameObject.SetActive(present);industries[i].gameObject.SetActive(present);symbols[i].gameObject.SetActive(present);status[i].gameObject.SetActive(present);if(!present)continue;
                var a=city.Tiles[i];var color=StatisticsTheme.Player(a.Player);if(!panel.Filters.Players.Contains(a.Player))color=Color.Lerp(color,StatisticsTheme.Paper,.46f);
                owners[i].text="● "+a.Player;owners[i].color=color;
                industries[i].text=Industry(a.Type)+" Lv."+a.Level;symbols[i].Kind=a.Type;symbols[i].color=StatisticsTheme.Ink;symbols[i].SetVerticesDirty();
                status[i].Kind=snapshot.Flipped(a)?"Check":"Open";status[i].color=snapshot.Flipped(a)?StatisticsTheme.Focus:StatisticsTheme.Muted;status[i].SetVerticesDirty();
            }
            empty.gameObject.SetActive(!valid||city.Tiles.Length==0);empty.text=!valid?city.Problem:city.Location.Merchant?"Merchant location · no industry slots":"No industries at this era end";
            float rowEnd=(54+Math.Max(1,valid?city.Tiles.Length:1)*32)*scale;
            int vp=StatisticsMapQueries.IndustryVP(snapshot,city,panel.Filters),links=StatisticsMapQueries.ContributionVP(panel.Report,snapshot,city,panel.Filters);
            int bonus=snapshot.Era==1?city.Tiles.Where(a=>panel.Filters.Players.Contains(a.Player)).Sum(a=>a.IntroVP):0;
            summary.text="Selected: "+(valid?vp.ToString():"—")+" industry VP · "+links+" link VP"+(bonus>0?"\n+ "+bonus+" introductory bonus VP":"");
            summary.rectTransform.anchoredPosition=new Vector2(18*scale,-rowEnd-3*scale);
            float footerY=rowEnd+(panel.Report.Introductory?51:31)*scale;
            footer.rectTransform.anchoredPosition=new Vector2(18*scale,-footerY);
            var linksByOwner=city.Links.GroupBy(l=>l.Player).OrderBy(g=>(int)g.Key).ToArray();
            for(int i=0;i<4;i++){bool show=i<linksByOwner.Length;connectionSymbols[i].gameObject.SetActive(show);connectionLabels[i].gameObject.SetActive(show||i==0&&linksByOwner.Length==0);if(show){var owner=linksByOwner[i].Key;var tint=StatisticsTheme.Player(owner);if(!panel.Filters.Players.Contains(owner))tint=Color.Lerp(tint,StatisticsTheme.Paper,.46f);connectionSymbols[i].Kind="Owner";connectionSymbols[i].Owner=owner;connectionSymbols[i].color=tint;connectionSymbols[i].SetVerticesDirty();connectionLabels[i].color=tint;connectionLabels[i].text=owner+" "+linksByOwner[i].Count();}else if(i==0)connectionLabels[i].text="None";connectionSymbols[i].rectTransform.anchoredPosition=new Vector2((119+i*82)*scale,-footerY-8*scale);connectionLabels[i].rectTransform.anchoredPosition=new Vector2((132+i*82)*scale,-footerY);}
            float keyY=footerY+31*scale;
            legend.rectTransform.anchoredPosition=new Vector2(40*scale,-keyY);
            unflippedLabel.rectTransform.anchoredPosition=new Vector2(146*scale,-keyY);
            flippedKey.rectTransform.anchoredPosition=new Vector2(18*scale,-keyY-4*scale);openKey.rectTransform.anchoredPosition=new Vector2(122*scale,-keyY-4*scale);
            float height=keyY+34*scale;
            rect.sizeDelta=new Vector2(width,height);rect.anchoredPosition=new Vector2(Mathf.Clamp(dot.x-width/2,8*scale,Math.Max(8*scale,stageWidth-width-8*scale)),dot.y-dotRadius-12*scale);
            gameObject.SetActive(true);rect.SetAsLastSibling();
            // TMP can create fallback submeshes for player bullets on activation.
            foreach(var text in GetComponentsInChildren<TextMeshProUGUI>())text.ForceMeshUpdate();
            foreach(var graphic in GetComponentsInChildren<Graphic>())graphic.raycastTarget=false;
        }
        public void Hide(){City=null;gameObject.SetActive(false);}
    }
    public sealed class StatisticsMapGlyph : MaskableGraphic
    {
        public string Kind="Owner";public PlayerColor Owner;
        protected override void OnPopulateMesh(VertexHelper mesh)
        {
            mesh.Clear();var r=rectTransform.rect;float s=Math.Min(r.width,r.height);var c=r.center;
            Action<float,float,float,float> line=(x,y,u,v)=>StatisticsMapView.Line(mesh,c+new Vector2(x,y)*s,c+new Vector2(u,v)*s,color,s*.085f);
            if(Kind=="Owner"){StatisticsPlotGraphic.Symbol(mesh,c,s*.4f,color,StatisticsPlotPalette.Shape(Owner.ToString()));return;}
            if(Kind=="Check"){line(-.35f,0,-.1f,-.26f);line(-.1f,-.26f,.38f,.3f);return;}
            if(Kind=="Open"){line(-.28f,0,.28f,0);return;}
            if(Kind=="Brewery"){line(-.27f,-.35f,-.37f,0);line(-.37f,0,-.27f,.35f);line(-.27f,.35f,.27f,.35f);line(.27f,.35f,.37f,0);line(.37f,0,.27f,-.35f);line(.27f,-.35f,-.27f,-.35f);line(-.3f,.16f,.3f,.16f);line(-.3f,-.16f,.3f,-.16f);return;}
            if(Kind=="CoalMine"||Kind=="IronWorks"){StatisticsPlotGraphic.Symbol(mesh,c,s*.36f,color,Kind=="CoalMine"?6:1);return;}
            if(Kind=="Pottery"){line(-.12f,.4f,.12f,.4f);line(-.1f,.3f,-.3f,-.15f);line(.1f,.3f,.3f,-.15f);line(-.3f,-.15f,-.2f,-.35f);line(.3f,-.15f,.2f,-.35f);line(-.2f,-.35f,.2f,-.35f);return;}
            line(-.4f,-.3f,.4f,-.3f);line(-.4f,-.3f,-.4f,.1f);line(-.4f,.1f,-.12f,.3f);line(-.12f,.3f,-.12f,.1f);line(-.12f,.1f,.13f,.3f);line(.13f,.3f,.13f,.1f);line(.13f,.1f,.4f,.1f);line(.4f,.1f,.4f,-.3f);
            if(Kind=="CottonMill")line(.25f,.1f,.25f,.44f);else line(-.25f,-.1f,.23f,-.1f);
        }
    }
}
