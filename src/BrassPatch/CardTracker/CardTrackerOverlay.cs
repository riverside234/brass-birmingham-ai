using System;
using System.Collections.Generic;
using System.Linq;
using BoardGameRules.Entities.Board.BuildingSlots;
using Frameworks.Utils;
using Frameworks.View.Common;
using Frameworks.View.Game;
using InterfaceAdapters.Game.IndustriesSummary;
using InterfaceAdapters.Game.Market;
using InterfaceAdapters.Game.HandOfCards;
using TMPro;
using UnityEngine;
using UnityEngine.UI;
using BrassPatch.Statistics;
using BrassPatch.UI;

namespace BrassPatch.CardTracker
{
    public sealed class CardTrackerOverlay
    {
        sealed class Anchor { public RectTransform Badge, Target; public CardTrackerGlyph Leader; public bool City; }
        readonly CardTrackerSession session;
        readonly RectTransform root;
        readonly TMP_FontAsset font;
        readonly List<Anchor> anchors = new List<Anchor>();
        RectTransform layer, header, wildHeader, detail, list, legend;
        ZoomScrollRect map;
        Vector3 oldMapScale;
        Vector2 oldMapPosition;
        bool oldMapEnabled, pinned, listing;
        float unit=1,textScale=1.15f,inspectionMinScale;
        CardTrackerProjection snapshot;
        TextMeshProUGUI toggleLabel;
        public Button Toggle { get; private set; }
        public bool HasDetail { get { return detail != null; } }
        public bool HasList { get { return list != null; } }
        public int CityBadges { get { return anchors.Count(a=>a.City); } }
        public int IndustryBadges { get { return anchors.Count(a=>!a.City); } }
        public float TextScale { get { return textScale; } }
        Camera Camera { get { return root.GetComponent<Canvas>().rootCanvas.worldCamera; } }
        public CardTrackerOverlay(CardTrackerSession owner,RectTransform parent)
        {
            session=owner;root=parent;
            // Opt in only on the tracker root; the board and other game panels
            // retain their own appearance and input behavior.
            root.gameObject.AddComponent<StatisticsTheme>();
            root.gameObject.AddComponent<LightUiTheme>();
            var text=owner.Board.GetComponentsInChildren<TextMeshProUGUI>(true).FirstOrDefault(t=>t.font!=null);
            font=text==null?TMP_Settings.defaultFontAsset:text.font;
            UpdateUnit();
            Toggle=StatisticsWidgets.Button(root,font,"Card Tracker toggle","Card Tracker",Vector2.zero,new Vector2(200,50)*unit,20*unit*textScale,()=>{
                if(session.IsOpen)session.Close();else session.Open();
            });
            Translucent(Toggle.gameObject,.96f);
            toggleLabel=Toggle.GetComponentInChildren<TextMeshProUGUI>();
            Hover(Toggle.gameObject,"Four-player public card counts.\nAvailable on your turn, between actions, after animations finish.\nOpening pauses the game. Closing resumes it.");
            Layout();
        }
        void UpdateUnit()
        {
            var canvas=root.GetComponent<Canvas>().rootCanvas;
            float pixels=canvas.worldCamera!=null?canvas.worldCamera.pixelWidth:Screen.width;
            unit=root.rect.width/Mathf.Max(1,pixels);if(unit<=0)unit=1;
            root.GetComponent<LightUiTheme>().PixelUnit=unit;
        }
        RectTransform Rect(Transform parent,string name,Vector2 size)
        {
            var rect=StatisticsWidgets.Rect(parent,name,Vector2.zero,size*unit);
            rect.anchorMin=rect.anchorMax=new Vector2(.5f,.5f);rect.pivot=Vector2.zero;return rect;
        }
        TextMeshProUGUI Text(Transform parent,string name,string value,Vector2 pos,Vector2 size,float pixels=20,bool wrap=false)
        {
            var label=StatisticsWidgets.Text(parent,font,name,value,pos*unit,size*unit,pixels*unit*textScale,wrap);
            label.richText=false;return label;
        }
        Button Button(Transform parent,string name,string value,Vector2 pos,Vector2 size,Action click,float pixels=20)
        {
            var button=StatisticsWidgets.Button(parent,font,name,value,pos*unit,size*unit,pixels*unit*textScale,click);
            Translucent(button.gameObject,.95f);return button;
        }
        static void Translucent(GameObject node,float opacity)
        {
            var surface=node.GetComponent<StatisticsSurface>();if(surface==null)return;
            surface.canvasRenderer.SetAlpha(opacity);
            var button=node.GetComponent<Button>();if(button==null)return;
            var colors=button.colors;var tint=colors.normalColor;tint.a=opacity;colors.normalColor=tint;
            tint=colors.highlightedColor;tint.a=Mathf.Min(1,opacity+.04f);colors.highlightedColor=tint;
            tint=colors.selectedColor;tint.a=Mathf.Min(1,opacity+.04f);colors.selectedColor=tint;
            tint=colors.pressedColor;tint.a=opacity;colors.pressedColor=tint;
            tint=colors.disabledColor;tint.a=opacity;colors.disabledColor=tint;
            button.colors=colors;
        }
        void Hover(GameObject node,string value)
        { var hover=node.AddComponent<CardTrackerHover>();hover.Owner=this;hover.Description=value; }
        static void Destroy(RectTransform item)
        { if(item!=null){item.gameObject.SetActive(false);UnityEngine.Object.Destroy(item.gameObject);} }
        public void RefreshToggle(bool enabled,bool open)
        {
            if(Toggle==null||toggleLabel==null)return;
            StatisticsWidgets.Enabled(Toggle,enabled);StatisticsWidgets.Selected(Toggle,open);
            toggleLabel.color=enabled?StatisticsTheme.Ink:StatisticsTheme.Muted;
            toggleLabel.text=open?"Resume game":"Card Tracker";
        }
        public void Show(CardTrackerProjection state)
        {
            snapshot=state;
            map=ReflectionAccess.Get<ZoomScrollRect>(session.Board,"_zoomScrollRect");
            oldMapScale=map.content.localScale;oldMapPosition=map.content.anchoredPosition;oldMapEnabled=map.enabled;
            map.enabled=false;
            // Inspection starts at the whole board. The exact prior camera pose
            // and ScrollRect state are restored before animations resume.
            map.content.localScale=new Vector3(map.MinContentScaleFactor,map.MinContentScaleFactor,1);
            map.content.anchoredPosition=Vector2.zero;
            FitBoard();Build();
        }
        void FitBoard()
        {
            var targets=session.Board.GetComponentsInChildren<RegionView>().Select(r=>r.transform.Find("NameImage") as RectTransform)
                .Concat(session.Board.GetComponentsInChildren<BuildingSlotView>().Select(s=>s.transform as RectTransform)).Where(t=>t!=null).ToArray();
            inspectionMinScale=map.MinContentScaleFactor;if(targets.Length==0)return;
            var bounds=Bounds(targets[0]);foreach(var target in targets.Skip(1)){var b=Bounds(target);bounds=UnityEngine.Rect.MinMaxRect(Mathf.Min(bounds.xMin,b.xMin),Mathf.Min(bounds.yMin,b.yMin),Mathf.Max(bounds.xMax,b.xMax),Mathf.Max(bounds.yMax,b.yMax));}
            var area=UnityEngine.Rect.MinMaxRect(root.rect.xMin+175*unit,FooterBottom()+76*unit,root.rect.xMax-295*unit,root.rect.yMax-(80+3*Mathf.Ceil(21*textScale+3))*unit);
            float factor=Mathf.Min(1,area.width/(bounds.width+60*unit),area.height/(bounds.height+60*unit));
            inspectionMinScale*=factor;map.content.localScale=new Vector3(inspectionMinScale,inspectionMinScale,1);
            var fitted=Bounds(targets[0]);foreach(var target in targets.Skip(1)){var b=Bounds(target);fitted=UnityEngine.Rect.MinMaxRect(Mathf.Min(fitted.xMin,b.xMin),Mathf.Min(fitted.yMin,b.yMin),Mathf.Max(fitted.xMax,b.xMax),Mathf.Max(fitted.yMax,b.yMax));}
            map.content.anchoredPosition+=(Vector2)map.content.parent.InverseTransformVector(root.TransformVector(area.center-fitted.center));
        }
        void Build()
        {
            Destroy(layer);anchors.Clear();detail=null;list=null;pinned=false;
            UpdateUnit();
            layer=Rect(root,"Paused card inspection",Vector2.zero);layer.pivot=new Vector2(.5f,.5f);StatisticsWidgets.Stretch(layer,Vector2.zero,Vector2.zero);
            var shield=layer.gameObject.AddComponent<Image>();shield.color=new Color(0,0,0,.035f);shield.raycastTarget=true;
            var input=layer.gameObject.AddComponent<CardTrackerMapInput>();input.Owner=this;
            float headerWidth=Mathf.Min(1260,root.rect.width/unit-310);
            header=Rect(layer,"Tracker header",new Vector2(headerWidth,48));
            header.anchoredPosition=new Vector2(root.rect.xMin+12*unit,root.rect.yMax-60*unit);
            header.gameObject.AddComponent<StatisticsSurface>().Panel=true;
            Translucent(header.gameObject,.94f);
            Text(header,"Paused title","Card tracker",new Vector2(14,-6),new Vector2(212,36),22).fontStyle=FontStyles.Bold;
            Text(header,"Paused state","Paused",new Vector2(224,-6),new Vector2(78,36),18).color=StatisticsTheme.Accent;
            string summary=snapshot.Complete?(snapshot.Rail?"Rail":"Canal")+" · "+snapshot.Played+" played · "+snapshot.Unseen+" unseen":"Counts unknown";
            Text(header,"Public summary",summary,new Vector2(308,-6),new Vector2(headerWidth-668,36),18).color=StatisticsTheme.Muted;
            Button(header,"Tracker card list","Card list",new Vector2(headerWidth-348,-6),new Vector2(108,36),()=>ShowList(),18);
            Button(header,"Tracker text size","Text "+(textScale<1.1f?"1":textScale<1.25f?"2":"3")+"/3",new Vector2(headerWidth-232,-6),new Vector2(108,36),()=>{textScale=textScale<1.1f?1.15f:textScale<1.25f?1.3f:1;Build();Layout();},18);
            Button(header,"Tracker resume","Resume",new Vector2(headerWidth-116,-6),new Vector2(104,36),()=>session.Close(),18);
            float wildRow=Mathf.Ceil(20*textScale+3),wildHeight=3*wildRow+2;
            wildHeader=StatisticsWidgets.Rect(header,"Wild cards strip",new Vector2(0,-54)*unit,new Vector2(headerWidth,wildHeight)*unit);
            var wildKey=StatisticsWidgets.Rect(wildHeader,"Wild cards key",Vector2.zero,new Vector2(142,wildHeight)*unit);
            wildKey.gameObject.AddComponent<StatisticsSurface>();Translucent(wildKey.gameObject,.94f);
            Text(wildKey,"Wild cards heading","Wild cards",new Vector2(8,-2),new Vector2(128,wildRow),18).color=StatisticsTheme.Accent;
            Text(wildKey,"Wild cards status key","1 held\n0 none",new Vector2(8,-wildRow-2),new Vector2(128,2*wildRow),17.5f,true);
            var players=session.History.CurrentGame.Metadata.PlayerMetadata.OrderBy(p=>(int)p.PlayerStartConfiguration.Color).ToArray();
            float available=headerWidth-150,chip=available/4;
            for(int i=0;i<players.Length;i++) {
                var p=players[i];var color=p.PlayerStartConfiguration.Color;float x=150+i*chip;
                var node=StatisticsWidgets.Rect(wildHeader,"Wild holder "+color,new Vector2(x,0)*unit,new Vector2(chip-8,wildHeight)*unit);
                node.gameObject.AddComponent<StatisticsSurface>();
                Translucent(node.gameObject,.92f);
                Text(node,"Wild holder name",p.NickName,new Vector2(8,0),new Vector2(chip-24,wildRow),18).color=StatisticsTheme.Player(color);
                for(int type=0;type<2;type++) {
                    bool[] held;bool known=snapshot.Complete&&snapshot.Wild.TryGetValue(color,out held);
                    bool acquired=known&&snapshot.Wild[color][type];
                    string message=p.NickName+" · Wild "+(type==0?"Location":"Industry")+"\n"+(!known?"Unknown: history incomplete.":acquired?"Held: acquired through Scout.":"Not currently held.")+
                        "\nPin = Wild Location. Factory = Wild Industry.";
                    var button=Button(node,"Wild "+color+" "+type,"",new Vector2(4,-wildRow*(type+1)),new Vector2(chip-16,wildRow),()=>ShowDetail(message));
                    StatisticsWidgets.Selected(button,acquired);
                    Text(button.transform,"Wild type name",type==0?"Location":"Industry",new Vector2(32,0),new Vector2(chip-82,wildRow),18);
                    var graphic=StatisticsWidgets.Rect(button.transform,"Wild symbol",new Vector2(8,-(wildRow-18)/2)*unit,new Vector2(18,18)*unit).gameObject.AddComponent<CardTrackerGlyph>();
                    graphic.Kind=type+1;graphic.Held=acquired;graphic.Known=known;graphic.raycastTarget=false;Hover(button.gameObject,message);
                    var count=Text(button.transform,"Wild held count",!known?"?":acquired?"1":"0",new Vector2(chip-45,0),new Vector2(25,wildRow),19);
                    count.color=acquired?StatisticsTheme.Accent:StatisticsTheme.Muted;count.fontStyle=FontStyles.Bold;
                }
            }
            foreach(var card in snapshot.Cards.Where(c=>c.City.HasValue)) {
                var region=session.Board.GetRegionViewForGivenRegionName(card.City.Value) as RegionView;
                if(region==null)continue;
                var target=region.transform.Find("NameImage") as RectTransform;
                float width=34+card.Total*20+(snapshot.Complete&&card.Unseen==0?20:0);
                var badge=Button(layer,"City cards "+card.Name,"",Vector2.zero,new Vector2(width,34),()=>ShowDetail(snapshot.Description(card)));
                Translucent(badge.gameObject,.96f);
                var cardIcon=StatisticsWidgets.Rect(badge.transform,"City card symbol",new Vector2(8,-8)*unit,new Vector2(16,18)*unit).gameObject.AddComponent<CardTrackerGlyph>();
                cardIcon.Kind=4;cardIcon.raycastTarget=false;
                if(snapshot.Complete) {
                    var graphic=StatisticsWidgets.Rect(badge.transform,"City card dots",Vector2.zero,new Vector2(width-34,28)*unit).gameObject.AddComponent<CardTrackerGlyph>();
                    StatisticsWidgets.Stretch((RectTransform)graphic.transform,new Vector2(29,3)*unit,new Vector2(-5,-3)*unit);
                    graphic.Total=card.Total;graphic.Unseen=card.Unseen;graphic.raycastTarget=false;
                }else Text(badge.transform,"Unknown city count","?",new Vector2(30,0),new Vector2(width-34,34),20);
                var leader=Rect(layer,"City leader "+card.Name,Vector2.zero).gameObject.AddComponent<CardTrackerGlyph>();
                ((RectTransform)leader.transform).pivot=new Vector2(.5f,.5f);
                StatisticsWidgets.Stretch((RectTransform)leader.transform,Vector2.zero,Vector2.zero);leader.Kind=3;leader.raycastTarget=false;leader.transform.SetAsFirstSibling();
                Hover(badge.gameObject,snapshot.Description(card));anchors.Add(new Anchor{Badge=(RectTransform)badge.transform,Target=target,City=true,Leader=leader});
            }
            if(session.Router.IsViewCached<IIndustriesSummaryView>()) {
                var sidebar=session.Router.GetView<IIndustriesSummaryView>() as IndustriesSummaryView;
                var container=sidebar==null?null:sidebar.transform.Find("SafeArea/Popup/RightPanelBackground/IndustriesContainer");
                if(container!=null)foreach(Transform row in container) {
                    var industry=row.GetComponent<IndustryView>();if(industry==null)continue;
                    var type=ReflectionAccess.Get<BuildingType>(industry,"BuildingType");
                    var card=snapshot.Find(FourPlayerCardManifest.IndustryKey(type));
                    bool shared=type==BuildingType.CottonMill||type==BuildingType.Manufacturer;
                    string label=snapshot.Complete?card.Unseen+" unseen"+(shared?" =":"")+"\n"+card.Played+" played":"? unseen\n? played";
                    var badge=Button(layer,"Industry cards "+type,label,Vector2.zero,new Vector2(124*textScale,Mathf.Min(64,58*textScale)),()=>ShowDetail(snapshot.Description(card)),20);
                    badge.GetComponentInChildren<TextMeshProUGUI>().enableWordWrapping=true;
                    Hover(badge.gameObject,snapshot.Description(card));
                    anchors.Add(new Anchor{Badge=(RectTransform)badge.transform,Target=row as RectTransform});
                }
            }
            string legend=snapshot.Rail?"Rail: unseen cards are in hands or the draw deck.":"Canal includes 4 hidden setup discards.";
            this.legend=Rect(layer,"Tracker legend panel",new Vector2(Math.Max(460,root.rect.width/unit-410),60));
            this.legend.gameObject.AddComponent<StatisticsSurface>();Translucent(this.legend.gameObject,.92f);
            var key=StatisticsWidgets.Rect(this.legend,"Count key",new Vector2(12,-18)*unit,new Vector2(40,22)*unit).gameObject.AddComponent<CardTrackerGlyph>();
            key.Total=2;key.Unseen=1;key.raycastTarget=false;
            var legendText=Text(this.legend,"Tracker legend","Red: unseen · crossed gray: played.  "+legend,Vector2.zero,Vector2.zero,18,true);
            StatisticsWidgets.Stretch(legendText.rectTransform,new Vector2(66,6)*unit,new Vector2(-14,-6)*unit);
            Toggle.transform.SetAsLastSibling();header.SetAsLastSibling();
            if(listing)ShowList();
            Layout();Canvas.ForceUpdateCanvases();
            foreach(var label in layer.GetComponentsInChildren<TextMeshProUGUI>())label.ForceMeshUpdate();
        }
        Rect Bounds(RectTransform target)
        {
            var corners=new Vector3[4];target.GetWorldCorners(corners);
            var points=corners.Select(c=>(Vector2)root.InverseTransformPoint(c)).ToArray();
            return UnityEngine.Rect.MinMaxRect(points.Min(p=>p.x),points.Min(p=>p.y),points.Max(p=>p.x),points.Max(p=>p.y));
        }
        IEnumerable<RectTransform> VisibleHandCards()
        {
            if(!session.Router.IsViewCached<IHandOfCardsView>())return Enumerable.Empty<RectTransform>();
            var hand=session.Router.GetView<IHandOfCardsView>() as HandOfCardsView;
            return hand==null||!hand.gameObject.activeInHierarchy||!hand.IsViewVisible()?Enumerable.Empty<RectTransform>():
                hand.GetComponentsInChildren<BaseCardView>().Select(c=>(c.HolderObject as RectTransform)??c.transform as RectTransform).Where(t=>t!=null&&t.gameObject.activeInHierarchy);
        }
        float FooterBottom()
        {
            float bottom=root.rect.yMin+60*unit;
            foreach(var card in VisibleHandCards()) {
                var b=Bounds(card);
                if(b.xMax>root.rect.xMin+130*unit)bottom=Mathf.Max(bottom,b.yMax+20*unit);
            }
            return bottom;
        }
        public void Layout()
        {
            if(Toggle==null)return;UpdateUnit();
            // A screen corner is stable throughout hand, market and map animations.
            // Reserve the lower action band; never follow animated artwork bounds.
            var toggle=(RectTransform)Toggle.transform;
            toggle.anchorMin=toggle.anchorMax=new Vector2(1,0);toggle.pivot=new Vector2(1,0);
            toggle.sizeDelta=new Vector2(200,50)*unit;toggle.anchoredPosition=new Vector2(-24,220)*unit;
            toggleLabel.fontSize=20*unit*textScale;
            if(layer==null)return;
            if(legend!=null) {
                legend.anchorMin=legend.anchorMax=new Vector2(.5f,.5f);legend.pivot=Vector2.zero;
                legend.anchoredPosition=new Vector2(root.rect.xMin+160*unit,FooterBottom());
                legend.sizeDelta=new Vector2(root.rect.width-408*unit,64*unit);
            }
            var occupied=new List<Rect>{Bounds(header),Bounds(wildHeader),Bounds((RectTransform)Toggle.transform),Bounds(legend)};
            occupied.AddRange(VisibleHandCards().Select(Bounds));
            var names=anchors.Where(a=>a.City&&a.Target!=null).Select(a=>Bounds(a.Target))
                .Concat(session.Board.GetComponentsInChildren<BuildingSlotView>().Select(s=>Bounds((RectTransform)s.transform))).ToArray();
            foreach(var anchor in anchors.Where(a=>!a.City)) {
                if(anchor.Target==null)continue;var b=Bounds(anchor.Target);var rect=anchor.Badge;
                rect.anchorMin=rect.anchorMax=new Vector2(.5f,.5f);rect.pivot=Vector2.zero;
                rect.anchoredPosition=new Vector2(b.xMin-rect.rect.width-8*unit,b.center.y-rect.rect.height/2);
                occupied.Add(new Rect(rect.anchoredPosition,rect.rect.size));
            }
            var viewport=map==null?root.rect:Bounds(map.viewport);
            foreach(var anchor in anchors.Where(a=>a.City)) {
                var b=Bounds(anchor.Target);var size=anchor.Badge.rect.size;
                bool visible=viewport.Overlaps(b);anchor.Badge.gameObject.SetActive(visible);anchor.Leader.gameObject.SetActive(visible);
                if(!visible)continue;
                var candidates=new List<Vector2>{new Vector2(b.center.x-size.x/2,b.yMin-size.y-4*unit),new Vector2(b.xMax+6*unit,b.center.y-size.y/2),
                    new Vector2(b.xMin-size.x-6*unit,b.center.y-size.y/2),new Vector2(b.center.x-size.x/2,b.yMax+4*unit),
                    new Vector2(b.center.x-size.x/2,b.yMin-size.y-24*unit),new Vector2(b.xMin-size.x-26*unit,b.center.y-size.y/2),
                    new Vector2(b.xMax+26*unit,b.center.y-size.y/2)};
                // Dense central cities need more than the four immediate sides.
                // Keep the nearest clear badge and connect it to its city by a leader.
                for(int ring=1;ring<=5;ring++) {
                    float gap=(6+ring*18)*unit;
                    for(int y=-1;y<=1;y++)for(int x=-1;x<=1;x++) {
                        if(x==0&&y==0)continue;
                        candidates.Add(b.center-size/2+new Vector2(x*(b.width/2+size.x/2+gap),y*(b.height/2+size.y/2+gap)));
                    }
                }
                var point=candidates.Last();
                foreach(var candidate in candidates) {
                    var possible=new Rect(candidate,size);
                    if(possible.xMin<viewport.xMin||possible.xMax>viewport.xMax||possible.yMin<viewport.yMin||possible.yMax>viewport.yMax)continue;
                    var clearance=new Rect(possible.x-2*unit,possible.y-2*unit,possible.width+4*unit,possible.height+4*unit);
                    if(occupied.Any(o=>o.Overlaps(clearance))||names.Any(n=>n.Overlaps(clearance)))continue;
                    point=candidate;break;
                }
                anchor.Badge.anchorMin=anchor.Badge.anchorMax=new Vector2(.5f,.5f);anchor.Badge.pivot=Vector2.zero;
                anchor.Badge.anchoredPosition=point;occupied.Add(new Rect(point,size));
                anchor.Leader.From=b.center;anchor.Leader.To=point+size/2;anchor.Leader.SetVerticesDirty();
            }
        }
        public void Pan(Vector2 screenDelta)
        { if(map!=null&&session.IsOpen){map.content.anchoredPosition+=screenDelta*unit;Layout();} }
        public void Zoom(float multiplier)
        {
            if(map==null||!session.IsOpen)return;
            float old=map.content.localScale.x,value=Mathf.Clamp(old*multiplier,inspectionMinScale,map.MaxContentScaleFactor);
            map.content.localScale=new Vector3(value,value,1);map.content.anchoredPosition*=value/old;Layout();
        }
        public void Tooltip(string value)
        { if(!session.IsOpen||pinned||listing)return;if(value==null){Destroy(detail);detail=null;}else MakeDetail(value,false); }
        public void ShowDetail(string value) { MakeDetail(value,true); }
        void MakeDetail(string value,bool pin)
        {
            if(layer==null)return;Destroy(detail);pinned=pin;
            float width=Mathf.Min(540,root.rect.width/unit-32);
            var measured=Text(layer,"Detail measurement",value,Vector2.zero,new Vector2(width-32,1),19,true);
            float height=Mathf.Max(110,measured.GetPreferredValues(value,(width-32)*unit,0).y/unit+54);Destroy(measured.rectTransform);
            detail=Rect(layer,"Tracker details",new Vector2(width,height));
            detail.anchoredPosition=new Vector2(root.rect.xMin+16*unit,FooterBottom()+70*unit);
            var surface=detail.gameObject.AddComponent<StatisticsSurface>();surface.Panel=true;surface.raycastTarget=pin;
            Translucent(detail.gameObject,.97f);
            Text(detail,"Card definition",value,new Vector2(16,-12),new Vector2(width-32,height-42),19,true);
            if(pin)Button(detail,"Close tracker detail","Close",new Vector2(width-92,-height+35),new Vector2(78,28),()=>DismissDetail(),18);
            detail.SetAsLastSibling();Toggle.transform.SetAsLastSibling();
        }
        public void ShowList()
        {
            if(layer==null)return;Destroy(list);Destroy(detail);detail=null;pinned=false;listing=true;
            float width=Mathf.Min(830,root.rect.width/unit-60),height=Mathf.Min(760,root.rect.height/unit-174);
            list=Rect(layer,"Tracker card list",new Vector2(width,height));list.pivot=new Vector2(.5f,.5f);list.anchoredPosition=new Vector2(0,-18*unit);
            list.gameObject.AddComponent<StatisticsSurface>().Panel=true;
            Translucent(list.gameObject,.97f);
            Text(list,"List heading","Public cards · "+(snapshot.Rail?"Rail":"Canal")+" era",new Vector2(18,-9),new Vector2(width-160,34),22).fontStyle=FontStyles.Bold;
            Button(list,"Close card list","Close",new Vector2(width-105,-8),new Vector2(85,32),()=>{Destroy(list);list=null;listing=false;},18);
            float rowWidth=width-49;
            Text(list,"List card label","Card",new Vector2(26,-47),new Vector2(rowWidth*.52f,35),19);
            Text(list,"List played label","Played",new Vector2(16+rowWidth*.55f,-47),new Vector2(80,35),19);
            Text(list,"List unseen label","Unseen",new Vector2(16+rowWidth*.73f,-47),new Vector2(80,35),19);
            Text(list,"List total label","Total",new Vector2(16+rowWidth*.9f,-47),new Vector2(70,35),19);
            var scroll=StatisticsWidgets.Scroll(list,"All card categories",new Vector2(16,-87)*unit,new Vector2(width-32,height-104)*unit,unit);
            float y=0,rowHeight=47*textScale;
            foreach(var card in snapshot.Cards) {
                var captured=card;string played=snapshot.Complete?card.Played.ToString():"?",unseen=snapshot.Complete?card.Unseen.ToString():"?";
                var row=Button(scroll.content,"Tracked row "+card.Name,"",new Vector2(0,-y),new Vector2(width-49,rowHeight-4),()=>ShowDetail(snapshot.Description(captured)),19);
                var rowSurface=row.GetComponent<StatisticsSurface>();rowSurface.Row=true;rowSurface.Alternate=((int)Math.Round(y/rowHeight))%2==1;
                Text(row.transform,"Card name",card.Name,new Vector2(10,0),new Vector2((width-49)*.52f,rowHeight-4),19);
                Text(row.transform,"Played count",played,new Vector2((width-49)*.55f,0),new Vector2(60,rowHeight-4),19);
                Text(row.transform,"Unseen count",unseen,new Vector2((width-49)*.73f,0),new Vector2(60,rowHeight-4),19).color=CardTrackerGlyph.Red;
                Text(row.transform,"Total count",card.Total.ToString(),new Vector2((width-49)*.9f,0),new Vector2(50,rowHeight-4),19);y+=rowHeight;
            }
            scroll.content.sizeDelta=new Vector2(scroll.content.sizeDelta.x,y);list.SetAsLastSibling();Toggle.transform.SetAsLastSibling();
        }
        public bool DismissDetail()
        {
            if(detail!=null&&pinned){Destroy(detail);detail=null;pinned=false;return true;}
            if(list!=null){Destroy(list);list=null;listing=false;return true;}
            return false;
        }
        public void Hide()
        {
            Destroy(layer);layer=null;header=null;wildHeader=null;detail=null;list=null;legend=null;pinned=false;listing=false;anchors.Clear();
            if(map!=null){if(map.content!=null){map.content.localScale=oldMapScale;map.content.anchoredPosition=oldMapPosition;}map.enabled=oldMapEnabled;map=null;}
            snapshot=null;
        }
    }
}
