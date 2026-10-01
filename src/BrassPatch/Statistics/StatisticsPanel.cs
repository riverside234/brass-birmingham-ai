using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using BoardGameRules.Entities.Players;
using DG.Tweening;
using Frameworks.View.Common;
using InterfaceAdapters.Common;
using InterfaceAdapters.Routing;
using TMPro;
using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.UI;

namespace BrassPatch.Statistics
{
    // Owns only newly created controls. No game object, tile or player portrait is
    // moved into this panel, and the report contains no live game references.
    public sealed class StatisticsPanel : BaseView
    {
        sealed class Table {
            public StatisticsSection Section;
            public ScrollRect Scroll;
            public float Y,Height,Width,Column;
        }
        sealed class NullController : IController {
            public void OnViewCreated() {} public void OnViewDestroyed() {}
        }
        public CompletedMatchStatistics Report { get; private set; }
        public StatisticsFilters Filters { get; private set; } = new StatisticsFilters();
        public int CurrentPage { get; private set; }
        public ScrollRect BodyScroll { get; private set; }
        public Button[] PageButtons { get; private set; }
        public bool IsOpen { get { return _isViewVisible && gameObject.activeInHierarchy; } }
        public override IController Controller { get; set; } = new NullController();
        public Action<bool?> OnClose;
        TMP_FontAsset font;
        RectTransform window,sticky,tip,dropdown;
        TextMeshProUGUI tipText,pageLabel;
        TMP_InputField search;
        Button closeButton;
        GameObject opener;
        readonly float[] positions=Enumerable.Repeat(1f,10).ToArray();
        readonly Dictionary<string,int> limits=new Dictionary<string,int>(),sortColumns=new Dictionary<string,int>();
        readonly HashSet<string> descending=new HashSet<string>(),expanded=new HashSet<string>(),hiddenSeries=new HashSet<string>();
        readonly List<Table> tables=new List<Table>();
        readonly List<Button> playerButtons=new List<Button>(),eraButtons=new List<Button>();
        float unit,width,bodyWidth;
        bool renderPending,closing;
        int screenWidth,screenHeight;
        Table pinned;

        protected override void Awake() { DestroyWhenClosed=false; DisableMenusUnderneath=false; }
        protected override void Start() { if(InputRegistration!=null)InputRegistration.RegisterKey(KeyEnum.Escape,this); }
        public void Initialize(CompletedMatchStatistics report,TMP_FontAsset typeface,Action<bool?> close)
        {
            Report=report; font=typeface; OnClose=close; Filters.Reset(report); BuildLayout();
        }
        public void SetOpener(GameObject value) { opener=value; }
        public override IAnimation CreateShowViewAnimation()
        {
            closing=false; _isViewVisible=true; EnableView(); MoveToTop(); Render();
            if(EventSystem.current!=null)EventSystem.current.SetSelectedGameObject(closeButton.gameObject);
            return new Frameworks.View.Animation.DOTweenAnimation(DOTween.Sequence().AppendInterval(.001f));
        }
        public override IAnimation CreateShowViewImmediatelyAnimation() { return CreateShowViewAnimation(); }
        public override IAnimation CreateHideViewAnimation()
        {
            _isViewVisible=false; Tooltip(null); gameObject.SetActive(false);
            return new Frameworks.View.Animation.DOTweenAnimation(DOTween.Sequence().AppendInterval(.001f));
        }
        public override void SetInteractable(bool value) { GetComponent<CanvasGroup>().interactable=value; }
        void OnDestroy() { if(InputRegistration!=null)InputRegistration.UnregisterAllKeys(this); }
        public void Close(bool? results=null)
        {
            if(closing||!IsOpen)return; closing=true; positions[CurrentPage]=BodyScroll.verticalNormalizedPosition;
            Tooltip(null); if(EventSystem.current!=null)EventSystem.current.SetSelectedGameObject(null);
            if(OnClose!=null)OnClose(results); else { _isViewVisible=false; gameObject.SetActive(false); }
            if(results!=true&&opener!=null&&opener.activeInHierarchy&&EventSystem.current!=null)EventSystem.current.SetSelectedGameObject(opener);
        }
        public void SelectPage(int page)
        {
            if(page<0||page>=10)return; if(BodyScroll!=null)positions[CurrentPage]=BodyScroll.verticalNormalizedPosition;
            CurrentPage=page; Tooltip(null); renderPending=true;
        }
        public void Drill(int page,PlayerColor player,string query)
        {
            Filters.Players.Clear();Filters.Players.Add(player);Filters.Search=query??"";
            if(search!=null)search.SetTextWithoutNotify(Filters.Search);
            positions[page]=1; SelectPage(page);
        }
        public void ResetFilters()
        { Filters.Reset(Report);if(search!=null)search.SetTextWithoutNotify("");QueueRender(false); }
        void QueueRender(bool retain)
        { if(retain&&BodyScroll!=null)positions[CurrentPage]=BodyScroll.verticalNormalizedPosition;else positions[CurrentPage]=1;renderPending=true; }
        void LateUpdate()
        {
            if(!IsOpen)return;
            if(screenWidth!=Screen.width||screenHeight!=Screen.height) {positions[CurrentPage]=BodyScroll.verticalNormalizedPosition;BuildLayout();renderPending=true;}
            if(renderPending) {renderPending=false; Render();}
            UpdateSticky();
        }
        void BuildLayout()
        {
            screenWidth=Screen.width;screenHeight=Screen.height; StatisticsWidgets.Clear(transform);
            Canvas.ForceUpdateCanvases();var root=(RectTransform)transform;unit=root.rect.width/Math.Max(1,Screen.width);
            if(unit<=0)unit=1; var safe=Screen.safeArea;
            width=Mathf.Min(safe.width*.94f,1680)*unit;float height=Mathf.Min(safe.height*.94f,960)*unit;
            var backdrop=StatisticsWidgets.Rect(transform,"Statistics input shield",Vector2.zero,root.rect.size);
            StatisticsWidgets.Stretch(backdrop,Vector2.zero,Vector2.zero);backdrop.gameObject.AddComponent<Image>().color=new Color(0,0,0,.74f);
            window=StatisticsWidgets.Rect(transform,"Statistics window",Vector2.zero,new Vector2(width,height));
            window.anchorMin=window.anchorMax=window.pivot=new Vector2(.5f,.5f);
            window.anchoredPosition=new Vector2(safe.center.x-Screen.width/2f,safe.center.y-Screen.height/2f)*unit;
            window.gameObject.AddComponent<StatisticsSurface>().Panel=true;
            StatisticsWidgets.Text(window,font,"Title","MATCH STATISTICS",V(22,12),Size(480,34),25*unit);
            StatisticsWidgets.Text(window,font,"Context",(Report.Introductory?"Introductory":"Full game")+" · "+Report.Players.Count+" players · "+(Report.CompleteHistory?"complete history":"partial history"),V(22,45),new Vector2(width-100*unit,26*unit),17*unit);
            closeButton=StatisticsWidgets.Button(window,font,"Close statistics","×",new Vector2(width-66*unit,-12*unit),Size(44,42),26*unit,()=>Close());
            float x=22*unit;playerButtons.Clear();
            foreach(var p in Report.Players) {
                var player=p;float w=Mathf.Min(148,(width/unit-390)/Report.Players.Count)*unit;
                var b=StatisticsWidgets.Button(window,font,"Filter "+p.Name,p.Name,new Vector2(x,-78*unit),new Vector2(w,34*unit),17*unit,()=>{
                    if(!Filters.Players.Remove(player.Color))Filters.Players.Add(player.Color);QueueRender(false);
                });
                var label=b.GetComponentInChildren<TextMeshProUGUI>();label.color=StatisticsWidgets.PlayerColor(p.Color);
                Tip(b,p.Name+"\n"+p.Opponent+"\nToggle this player in all pages.");playerButtons.Add(b);x+=w+8*unit;
            }
            eraButtons.Clear();
            for(int i=0;i<(Report.Introductory?2:3);i++) {
                int era=i;string name=new[]{"Whole game","Canal","Rail"}[i];float w=(i==0?108:68)*unit;
                var b=StatisticsWidgets.Button(window,font,"Era "+name,name,new Vector2(x,-78*unit),new Vector2(w,34*unit),16*unit,()=>{Filters.Era=era;QueueRender(false);});
                eraButtons.Add(b);x+=w+6*unit;
            }
            StatisticsWidgets.Button(window,font,"Reset filters","Reset",new Vector2(width-106*unit,-78*unit),Size(84,34),17*unit,ResetFilters);
            search=StatisticsWidgets.Search(window,font,V(22,121),new Vector2(Mathf.Min(510*unit,width-330*unit),32*unit),17*unit,value=>{Filters.Search=value;QueueRender(false);});
            search.SetTextWithoutNotify(Filters.Search);
            StatisticsWidgets.Button(window,font,"Find category","Find category...",new Vector2(search.GetComponent<RectTransform>().sizeDelta.x+32*unit,-121*unit),Size(164,32),16*unit,ShowCategories);
            float nav=(Screen.width<1500?176:216)*unit,bodyY=168*unit,bodyHeight=height-bodyY-64*unit;
            var navigation=StatisticsWidgets.Scroll(window,"Page navigation",new Vector2(16*unit,-bodyY),new Vector2(nav,bodyHeight),unit);
            navigation.content.sizeDelta=new Vector2(nav-14*unit,10*53*unit);PageButtons=new Button[10];
            for(int i=0;i<10;i++) {
                int page=i; PageButtons[i]=StatisticsWidgets.Button(navigation.content,font,"Page "+i,StatisticsPages.Names[i],new Vector2(0,-i*53*unit),new Vector2(nav-19*unit,47*unit),17*unit,()=>SelectPage(page));
                PageButtons[i].GetComponentInChildren<TextMeshProUGUI>().enableWordWrapping=true;
            }
            BodyScroll=StatisticsWidgets.Scroll(window,"Statistics page",new Vector2(nav+32*unit,-bodyY),new Vector2(width-nav-48*unit,bodyHeight),unit);
            bodyWidth=BodyScroll.viewport.sizeDelta.x-8*unit;
            sticky=StatisticsWidgets.Rect(BodyScroll.viewport,"Pinned column headings",Vector2.zero,new Vector2(bodyWidth,37*unit));sticky.gameObject.AddComponent<RectMask2D>();sticky.gameObject.SetActive(false);
            pinned=null;
            StatisticsWidgets.Button(window,font,"Return to results","Return to results",new Vector2(22*unit,-height+48*unit),Size(188,36),17*unit,()=>Close(true));
            StatisticsWidgets.Button(window,font,"Return to board","Close · review board",new Vector2(width-236*unit,-height+48*unit),Size(214,36),17*unit,()=>Close(false));
            tip=StatisticsWidgets.Rect(window,"Exact values",new Vector2(width-454*unit,-height+210*unit),Size(424,144));
            tip.gameObject.AddComponent<StatisticsSurface>().Panel=true;
            tipText=StatisticsWidgets.Text(tip,font,"Definition","",V(14,8),Size(398,128),16*unit,true);tip.gameObject.SetActive(false);
        }
        Vector2 V(float x,float y) { return new Vector2(x,-y)*unit; }
        Vector2 Size(float x,float y) { return new Vector2(x,y)*unit; }
        void Tip(Component target,string description) { var t=target.gameObject.AddComponent<StatisticsTooltip>();t.Panel=this;t.Description=description; }
        public void Tooltip(string value)
        {
            if(tip==null)return;tip.gameObject.SetActive(!String.IsNullOrEmpty(value));if(value!=null) {tipText.text=value;tip.SetAsLastSibling();}
        }
        void ShowCategories()
        {
            if(dropdown!=null){Destroy(dropdown.gameObject);dropdown=null;return;}
            IEnumerable<string> values=CurrentPage==2?StatisticsPages.IndustryTypes:CurrentPage==5?new[]{"Coal","Iron","Beer"}:
                CurrentPage==6?StatisticsPages.ActionTypes:CurrentPage==3?Report.Assets.Select(a=>a.City).Distinct().OrderBy(c=>c).ToArray():
                new[]{"Build","Network","Develop","Sell","Loan","Scout","Merchant","Penalty"};
            var options=new[]{"All ledger rows"}.Concat(values).ToArray();
            dropdown=StatisticsWidgets.Rect(window,"Category choices",V(22,158),Size(226,Mathf.Min(11,options.Length)*34));
            dropdown.gameObject.AddComponent<StatisticsSurface>().Panel=true;
            var scroll=StatisticsWidgets.Scroll(dropdown,"Categories",Vector2.zero,dropdown.sizeDelta,unit);scroll.content.sizeDelta=Size(210,options.Length*34);
            for(int i=0;i<options.Length;i++) {
                string value=options[i];StatisticsWidgets.Button(scroll.content,font,"Find "+value,value,V(0,i*34),Size(209,32),16*unit,()=>{
                    search.text=value=="All ledger rows"?"":value;Destroy(dropdown.gameObject);dropdown=null;
                });
            }
            dropdown.SetAsLastSibling();
        }
        void Render()
        {
            if(Report==null||BodyScroll==null)return;
            StatisticsWidgets.Clear(BodyScroll.content);tables.Clear();pinned=null;sticky.gameObject.SetActive(false); Tooltip(null);
            for(int i=0;i<10;i++)StatisticsWidgets.Selected(PageButtons[i],CurrentPage==i);
            for(int i=0;i<playerButtons.Count;i++)StatisticsWidgets.Selected(playerButtons[i],Filters.Players.Contains(Report.Players[i].Color));
            for(int i=0;i<eraButtons.Count;i++)StatisticsWidgets.Selected(eraButtons[i],Filters.Era==i);
            float y=0;
            pageLabel=StatisticsWidgets.Text(BodyScroll.content,font,"Page title",StatisticsPages.Names[CurrentPage],Vector2.zero,new Vector2(bodyWidth,34*unit),25*unit);y+=43*unit;
            string availability=Report.Availability;
            if(Report.Warnings.Count>0)availability+=" · "+String.Join(" · ",Report.Warnings);
            if(!String.IsNullOrWhiteSpace(Filters.Search))availability+="\nSearch filters ledger rows; charts retain the player and era selection.";
            y+=Paragraph(BodyScroll.content,"Data coverage",availability,y,16);
            if(CurrentPage==8) {
                var payments=StatisticsWidgets.Button(BodyScroll.content,font,"Cost denominator",Filters.AllCashPayments?"Cost: all cash payments (incl. debt)":"Cost: action spending",new Vector2(0,-y),Size(350,34),16*unit,()=>{Filters.AllCashPayments=!Filters.AllCashPayments;QueueRender(true);});
                y+=44*unit;
            }
            foreach(var section in StatisticsPages.Build(CurrentPage,Report,Filters)) {
                StatisticsWidgets.Text(BodyScroll.content,font,section.Title,section.Title,new Vector2(0,-y),new Vector2(bodyWidth,33*unit),21*unit).color=StatisticsWidgets.Gold;y+=36*unit;
                y+=Paragraph(BodyScroll.content,"Definition",section.Definition,y,16);
                if(section.Chart!=null)y+=Chart(section.Chart,y);
                if(section.Columns.Length>0) {
                    if(section.Rows.Count==0)y+=Paragraph(BodyScroll.content,"No rows","No records match these filters.",y,17);
                    else y+=TableView(section,y);
                }
                y+=24*unit;
            }
            BodyScroll.content.sizeDelta=new Vector2(bodyWidth,Mathf.Max(y,BodyScroll.viewport.sizeDelta.y));
            Canvas.ForceUpdateCanvases();BodyScroll.verticalNormalizedPosition=positions[CurrentPage];
            WireNavigation();
        }
        float Paragraph(Transform parent,string name,string value,float y,float size)
        {
            if(String.IsNullOrEmpty(value))return 0;
            var text=StatisticsWidgets.Text(parent,font,name,value,new Vector2(0,-y),new Vector2(bodyWidth,40*unit),size*unit,true);
            float height=Mathf.Max(size*1.4f*unit,text.GetPreferredValues(value,bodyWidth,0).y+6*unit);text.rectTransform.sizeDelta=new Vector2(bodyWidth,height);return height+8*unit;
        }
        string Key(StatisticsSection section) { return CurrentPage+":"+section.Title; }
        float TableView(StatisticsSection section,float y)
        {
            string key=Key(section);int column=sortColumns.ContainsKey(key)?sortColumns[key]:-1;
            var rows=section.Rows.ToList();
            if(column>=0)rows.Sort((a,b)=>Compare(a.Cells[column],b.Cells[column])*(descending.Contains(key)?-1:1));
            int limit=limits.ContainsKey(key)?limits[key]:60;int visible=Math.Min(limit,rows.Count);
            float col=Mathf.Max(110*unit,(bodyWidth-42*unit)/section.Columns.Length),w=col*section.Columns.Length+42*unit,h=38*unit;
            var heights=new float[visible];
            for(int i=0;i<visible;i++) {
                float extra=0;if(expanded.Contains(key+":"+rows[i].Key)) {
                    var t=pageLabel;extra=Mathf.Min(350*unit,t.GetPreferredValues(rows[i].Detail,w-32*unit,0).y*17/25+25*unit);
                }
                heights[i]=39*unit+extra;h+=heights[i];
            }
            if(visible<rows.Count)h+=40*unit;h+=14*unit;
            var scroll=StatisticsWidgets.Scroll(BodyScroll.content,section.Title+" table",new Vector2(0,-y),new Vector2(bodyWidth,h),unit,true);
            scroll.content.sizeDelta=new Vector2(w,h-14*unit);scroll.viewport.gameObject.GetComponent<Image>().color=new Color(.13f,.12f,.1f,.75f);
            var table=new Table{Section=section,Scroll=scroll,Y=y,Height=h,Width=w,Column=col};tables.Add(table);
            DrawHeaders(scroll.content,table);
            float rowY=38*unit;
            for(int i=0;i<visible;i++) {
                var row=rows[i];string id=key+":"+row.Key;
                var b=StatisticsWidgets.Button(scroll.content,font,"Row "+row.Key,"",new Vector2(0,-rowY),new Vector2(w,37*unit),17*unit,()=>{
                    if(!expanded.Remove(id))expanded.Add(id);QueueRender(true);
                });
                b.GetComponent<StatisticsSurface>().Selected=expanded.Contains(id);Tip(b,row.Detail);
                if(row.Player.HasValue) {
                    var stripe=StatisticsWidgets.Rect(b.transform,"Player marker",Size(3,-5),Size(3,27));stripe.gameObject.AddComponent<Image>().color=StatisticsWidgets.PlayerColor(row.Player.Value);
                }
                for(int c=0;c<row.Cells.Length;c++)
                    StatisticsWidgets.Text(b.transform,font,"Cell "+c,row.Cells[c],new Vector2(11*unit+c*col,0),new Vector2(col-14*unit,37*unit),17*unit);
                if(row.TargetPage>=0)StatisticsWidgets.Button(b.transform,font,"Open ledger","›",new Vector2(w-36*unit,0),Size(33,37),23*unit,()=>Drill(row.TargetPage,row.Player??Filters.Players.FirstOrDefault(),row.TargetSearch));
                if(expanded.Contains(id)) {
                    var detail=StatisticsWidgets.Text(scroll.content,font,"Expanded "+row.Key,row.Detail,new Vector2(14*unit,-rowY-39*unit),new Vector2(w-28*unit,heights[i]-40*unit),17*unit,true);
                    detail.overflowMode=TextOverflowModes.Truncate;
                }
                rowY+=heights[i];
            }
            if(visible<rows.Count)StatisticsWidgets.Button(scroll.content,font,"More "+section.Title,"Show next 60 · "+visible+" of "+rows.Count,new Vector2(4*unit,-rowY),Size(290,36),17*unit,()=>{limits[key]=limit+60;QueueRender(true);});
            return h+8*unit;
        }
        static int Compare(string a,string b)
        {
            double x,y;Func<string,string> clean=s=>s.Replace("£","").Replace("%","").Replace(",","").Replace("−","-");
            if(Double.TryParse(clean(a),NumberStyles.Float,CultureInfo.InvariantCulture,out x)&&Double.TryParse(clean(b),NumberStyles.Float,CultureInfo.InvariantCulture,out y))return x.CompareTo(y);
            return StringComparer.OrdinalIgnoreCase.Compare(a,b);
        }
        void DrawHeaders(Transform parent,Table table)
        {
            string key=Key(table.Section);
            for(int c=0;c<table.Section.Columns.Length;c++) {
                int index=c;string title=table.Section.Columns[c]+(sortColumns.ContainsKey(key)&&sortColumns[key]==c?(descending.Contains(key)?" ↓":" ↑"):"");
                var b=StatisticsWidgets.Button(parent,font,"Sort "+table.Section.Columns[c],title,new Vector2(c*table.Column,0),new Vector2(table.Column,37*unit),16*unit,()=>{
                    if(sortColumns.ContainsKey(key)&&sortColumns[key]==index) {if(!descending.Remove(key))descending.Add(key);}
                    else {sortColumns[key]=index;descending.Remove(key);}QueueRender(true);
                });b.GetComponent<StatisticsSurface>().Selected=true;
                Tip(b,"Sort by "+table.Section.Columns[c]+". Click again to reverse.");
            }
        }
        void UpdateSticky()
        {
            if(BodyScroll==null)return;float top=BodyScroll.content.anchoredPosition.y;
            var table=tables.FirstOrDefault(t=>top>t.Y+37*unit&&top<t.Y+t.Height-40*unit);
            if(table!=pinned) {
                StatisticsWidgets.Clear(sticky);pinned=table;
                if(table!=null) {
                    var header=StatisticsWidgets.Rect(sticky,"Headings",Vector2.zero,new Vector2(table.Width,37*unit));DrawHeaders(header,table);
                }
                sticky.gameObject.SetActive(table!=null);
            }
            if(table!=null&&sticky.childCount>0)((RectTransform)sticky.GetChild(sticky.childCount-1)).anchoredPosition=new Vector2(table.Scroll.content.anchoredPosition.x,0);
        }
        float Chart(StatisticsChart data,float y)
        {
            var label=StatisticsWidgets.Text(BodyScroll.content,font,data.Title,data.Title,new Vector2(0,-y),new Vector2(bodyWidth,29*unit),18*unit);y+=31*unit;
            float start=y,chartHeight=224*unit;var r=StatisticsWidgets.Rect(BodyScroll.content,data.Title+" plot",new Vector2(58*unit,-y),new Vector2(bodyWidth-66*unit,chartHeight));
            var chart=r.gameObject.AddComponent<StatisticsChartGraphic>();chart.Data=data;chart.Panel=this;
            var points=data.Series.SelectMany(s=>s.Points).ToArray();
            double max=points.Length==0?0:points.Max(p=>p.Y),min=points.Length==0?0:Math.Min(0,points.Min(p=>p.Y));
            if(data.Stacked) {
                max=data.Series.SelectMany(s=>s.Points.GroupBy(p=>p.X).Select(g=>g.Where(p=>p.Y>0).Sum(p=>p.Y))).DefaultIfEmpty(0).Max();
                min=data.Series.SelectMany(s=>s.Points.GroupBy(p=>p.X).Select(g=>g.Where(p=>p.Y<0).Sum(p=>p.Y))).DefaultIfEmpty(0).Min();
            }
            StatisticsWidgets.Text(BodyScroll.content,font,"Maximum",max.ToString("0.#"),new Vector2(0,-y),Size(54,28),15*unit);
            StatisticsWidgets.Text(BodyScroll.content,font,"Minimum",min.ToString("0.#"),new Vector2(0,-y-chartHeight+30*unit),Size(54,28),15*unit);
            if(data.Lines&&points.Length>0) {
                double left=points.Min(p=>p.X),right=points.Max(p=>p.X);
                for(int i=0;i<5;i++)StatisticsWidgets.Text(BodyScroll.content,font,"Turn tick "+i,(left+(right-left)*i/4).ToString("0"),new Vector2(75*unit+(bodyWidth-134*unit)*i/4,-y-chartHeight+24*unit),Size(50,24),14*unit);
            }
            y+=chartHeight+5*unit;
            string axes=data.Lines?"Turn number · "+data.Unit:data.Stacked?"Player bars · "+(Filters.Era==0&&!Report.Introductory?"Canal on left / Rail on right":Filters.Era==2?"Rail":"Canal")+" · "+data.Unit:"Actions in order: "+String.Join(", ",StatisticsPages.ActionTypes);
            y+=Paragraph(BodyScroll.content,"Chart axes",axes,y,15);
            float x=0;
            for(int i=0;i<data.Series.Count;i++) {
                string id=CurrentPage+":"+data.Title+":"+data.Series[i].Player;
                if(hiddenSeries.Contains(id))chart.Hidden.Add(i);
                float w=Mathf.Min(230*unit,bodyWidth/Math.Max(1,data.Series.Count));
                var legend=StatisticsWidgets.Button(BodyScroll.content,font,"Legend "+data.Series[i].Label,data.Series[i].Label,new Vector2(x,-y),new Vector2(w-6*unit,32*unit),16*unit,()=>{
                    if(!hiddenSeries.Remove(id))hiddenSeries.Add(id);QueueRender(true);
                });StatisticsWidgets.Selected(legend,!hiddenSeries.Contains(id));legend.GetComponentInChildren<TextMeshProUGUI>().color=StatisticsWidgets.PlayerColor(data.Series[i].Player);x+=w;
            }
            y+=41*unit;
            if(data.Stacked)y+=Paragraph(BodyScroll.content,"Score legend","Green: industries · Blue: links · Gold: merchant · Red: penalties · Purple: introductory additions",y,15);
            return y-start+31*unit;
        }
        void WireNavigation()
        {
            var controls=window.GetComponentsInChildren<Selectable>().Where(s=>s.IsActive()&&s.IsInteractable()).ToArray();
            for(int i=0;i<controls.Length;i++) {
                var n=controls[i].navigation;n.mode=Navigation.Mode.Explicit;n.selectOnUp=controls[(i-1+controls.Length)%controls.Length];n.selectOnDown=controls[(i+1)%controls.Length];
                n.selectOnLeft=n.selectOnUp;n.selectOnRight=n.selectOnDown;controls[i].navigation=n;
            }
        }
        public override void HandleKeyDown(KeyEnum key)
        {
            if(key==KeyEnum.Escape){Close();return;}
            var selected=EventSystem.current==null?null:EventSystem.current.currentSelectedGameObject;
            if(key==KeyEnum.Tab) {
                var controls=window.GetComponentsInChildren<Selectable>().Where(s=>s.IsActive()&&s.IsInteractable()).ToArray();
                if(controls.Length==0)return;int i=Array.FindIndex(controls,c=>c.gameObject==selected);
                int direction=Input.GetKey(KeyCode.LeftShift)||Input.GetKey(KeyCode.RightShift)?-1:1;
                EventSystem.current.SetSelectedGameObject(controls[(i+direction+controls.Length)%controls.Length].gameObject);return;
            }
            if(selected!=null&&selected.GetComponent<TMP_InputField>()!=null)return;
            if(key==KeyEnum.Home)BodyScroll.verticalNormalizedPosition=1;
            else if(key==KeyEnum.End)BodyScroll.verticalNormalizedPosition=0;
            else if(key==KeyEnum.PageUp||key==KeyEnum.PageDown)Scroll(key==KeyEnum.PageUp?1:-1,BodyScroll.viewport.rect.height*.85f);
        }
        public override void HandleKey(KeyEnum key)
        {
            var selected=EventSystem.current==null?null:EventSystem.current.currentSelectedGameObject;
            if(selected!=null&&selected.GetComponent<TMP_InputField>()!=null)return;
            if(key==KeyEnum.UpArrow||key==KeyEnum.DownArrow)Scroll(key==KeyEnum.UpArrow?1:-1,360*unit*Time.unscaledDeltaTime);
        }
        void Scroll(int direction,float distance)
        { float range=BodyScroll.content.rect.height-BodyScroll.viewport.rect.height;if(range>0)BodyScroll.verticalNormalizedPosition=Mathf.Clamp01(BodyScroll.verticalNormalizedPosition+direction*distance/range); }
    }
}
