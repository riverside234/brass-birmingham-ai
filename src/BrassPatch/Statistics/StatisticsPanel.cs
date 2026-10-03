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
    public sealed partial class StatisticsPanel : BaseView
    {
        sealed class Table {
            public StatisticsSection Section;
            public string Key;
            public ScrollRect Scroll;
            public float Y,Height,Width,Column,Header;
        }
        sealed class NullController : IController {
            public void OnViewCreated() {} public void OnViewDestroyed() {}
        }
        public CompletedMatchStatistics Report { get; private set; }
        public StatisticsFilters Filters { get; private set; } = new StatisticsFilters();
        public int CurrentPage { get; private set; } = 0;
        public ScrollRect BodyScroll { get; private set; }
        public Button[] PageButtons { get; private set; }
        public float TextScale { get; private set; } = 1.15f;
        public bool IsOpen { get { return _isViewVisible && gameObject.activeInHierarchy; } }
        public override IController Controller { get; set; } = new NullController();
        public Action<bool?> OnClose;
        TMP_FontAsset font;
        RectTransform window,sticky,tip,dropdown;
        TextMeshProUGUI tipText,pageLabel,filterStatus,measurement,sizeLabel;
        TMP_InputField search;
        Button closeButton,smallerText,largerText,clearSearch,collapseRows,topButton;
        GameObject opener;
        readonly float[] positions=Enumerable.Repeat(1f,StatisticsPages.Names.Length).ToArray();
        readonly Dictionary<string,int> limits=new Dictionary<string,int>(),sortColumns=new Dictionary<string,int>();
        readonly Dictionary<string,float> horizontalPositions=new Dictionary<string,float>();
        readonly HashSet<string> descending=new HashSet<string>(),expanded=new HashSet<string>(),hiddenSeries=new HashSet<string>();
        readonly List<Table> tables=new List<Table>();
        readonly List<Button> playerButtons=new List<Button>(),eraButtons=new List<Button>();
        float unit,width,bodyWidth,bodyY;
        bool renderPending,closing,revealPage=true;
        int screenWidth,screenHeight;
        Table pinned;
        protected override void Awake() { DestroyWhenClosed=false; DisableMenusUnderneath=false; }
        protected override void Start() { if(InputRegistration!=null)InputRegistration.RegisterKey(KeyEnum.Escape,this); }
        public void Initialize(CompletedMatchStatistics report,TMP_FontAsset typeface,Action<bool?> close)
        {
            if(GetComponent<StatisticsTheme>()==null)gameObject.AddComponent<StatisticsTheme>();
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
            page=StatisticsPages.NormalizePage(page); if(BodyScroll!=null)positions[CurrentPage]=BodyScroll.verticalNormalizedPosition;
            CurrentPage=page; CloseCategories(); Tooltip(null); revealPage=true;renderPending=true;
        }
        public void Drill(int page,PlayerColor player,string query)
        {
            page=StatisticsPages.NormalizePage(page);
            Filters.Players.Clear();Filters.Players.Add(player);Filters.Search=query??"";
            if(search!=null)search.SetTextWithoutNotify(Filters.Search);
            positions[page]=1; SelectPage(page);
        }
        public void ResetFilters()
        { Filters.Reset(Report);hiddenSeries.Clear();if(search!=null)search.SetTextWithoutNotify("");QueueRender(false); }
        void QueueRender(bool retain)
        { CloseCategories();if(retain&&BodyScroll!=null)positions[CurrentPage]=BodyScroll.verticalNormalizedPosition;else positions[CurrentPage]=1;renderPending=true; }
        void ChangeTextSize(int direction)
        {
            float next=Mathf.Clamp(TextScale+direction*.15f,1f,1.3f);
            if(Mathf.Abs(next-TextScale)<.01f)return;
            positions[CurrentPage]=BodyScroll.verticalNormalizedPosition;RememberTables();
            TextScale=next;BuildLayout();renderPending=true;
        }
        void RememberTables()
        { foreach(var table in tables)if(table.Scroll!=null)horizontalPositions[table.Key]=table.Scroll.horizontalNormalizedPosition; }
        void LateUpdate()
        {
            if(!IsOpen)return;
            var pixels=PixelBounds();
            float currentUnit=((RectTransform)transform).rect.width/Math.Max(1,pixels.width);
            if(screenWidth!=(int)pixels.width||screenHeight!=(int)pixels.height||Mathf.Abs(unit-currentUnit)>.001f) {
                positions[CurrentPage]=BodyScroll.verticalNormalizedPosition;BuildLayout();renderPending=true;
            }
            if(renderPending) {renderPending=false; Render();}
            UpdateSticky();
            if(topButton!=null)StatisticsWidgets.Enabled(topButton,BodyScroll.verticalNormalizedPosition<.995f);
        }
        Rect PixelBounds()
        {
            var canvas=GetComponentInParent<Canvas>().rootCanvas;
            // Camera render targets also use physical pixels. This keeps native
            // preview captures faithful when batch mode ignores SetResolution.
            return canvas.renderMode==RenderMode.ScreenSpaceCamera&&canvas.worldCamera!=null&&canvas.worldCamera.targetTexture!=null
                ?canvas.worldCamera.pixelRect:new Rect(0,0,Screen.width,Screen.height);
        }
        void BuildLayout()
        {
            RememberTables();CloseCategories();tables.Clear();
            var pixels=PixelBounds();screenWidth=(int)pixels.width;screenHeight=(int)pixels.height; StatisticsWidgets.Clear(transform);
            Canvas.ForceUpdateCanvases();var root=(RectTransform)transform;unit=root.rect.width/Math.Max(1,pixels.width);
            if(unit<=0)unit=1; var safe=screenWidth==Screen.width&&screenHeight==Screen.height?Screen.safeArea:pixels;
            width=Mathf.Min(safe.width*.96f,1920)*unit;float height=Mathf.Min(safe.height*.96f,1200)*unit;
            var backdrop=StatisticsWidgets.Rect(transform,"Statistics input shield",Vector2.zero,root.rect.size);
            StatisticsWidgets.Stretch(backdrop,Vector2.zero,Vector2.zero);backdrop.gameObject.AddComponent<Image>().color=new Color(0,0,0,.50f);
            window=StatisticsWidgets.Rect(transform,"Statistics window",Vector2.zero,new Vector2(width,height));
            window.anchorMin=window.anchorMax=window.pivot=new Vector2(.5f,.5f);
            window.anchoredPosition=new Vector2(safe.center.x-pixels.center.x,safe.center.y-pixels.center.y)*unit;
            window.gameObject.AddComponent<StatisticsSurface>().Panel=true;
            StatisticsWidgets.Text(window,font,"Title","MATCH STATISTICS",V(22,10),new Vector2(width-360*unit,48*unit),F(30)).fontStyle=FontStyles.Bold;
            StatisticsWidgets.Text(window,font,"Context",(Report.Introductory?"Introductory":"Full game")+" · "+Report.Players.Count+" players · "+(Report.CompleteHistory?"complete history":"partial history"),V(22,48),new Vector2(width-100*unit,30*unit),F(19));
            smallerText=StatisticsWidgets.Button(window,font,"Smaller text","A−",new Vector2(width-298*unit,-16*unit),Size(50,42),23*unit,()=>ChangeTextSize(-1));
            sizeLabel=StatisticsWidgets.Text(window,font,"Text size",new[]{"Large","Larger","Largest"}[Mathf.RoundToInt((TextScale-1)/.15f)],new Vector2(width-244*unit,-16*unit),Size(100,42),20*unit);
            sizeLabel.alignment=TextAlignmentOptions.Center;
            largerText=StatisticsWidgets.Button(window,font,"Larger text","A+",new Vector2(width-140*unit,-16*unit),Size(50,42),23*unit,()=>ChangeTextSize(1));
            StatisticsWidgets.Enabled(smallerText,TextScale>1.01f);StatisticsWidgets.Enabled(largerText,TextScale<1.29f);
            Tip(smallerText,"Reduce text size; the selected page and filters are retained.");
            Tip(largerText,"Increase text size. Tables and navigation adapt; page position and filters are retained.");
            closeButton=StatisticsWidgets.Button(window,font,"Close statistics","×",new Vector2(width-70*unit,-16*unit),Size(48,42),28*unit,()=>Close());
            float x=22*unit;playerButtons.Clear();
            foreach(var p in Report.Players) {
                var player=p;float w=Mathf.Min(192,(width/unit-330-130*TextScale-66)/Report.Players.Count-8)*unit;
                var b=StatisticsWidgets.Button(window,font,"Filter "+p.Name,p.Name,new Vector2(x,-72*unit),new Vector2(w,44*unit),F(21),()=>{
                    if(!Filters.Players.Remove(player.Color))Filters.Players.Add(player.Color);QueueRender(false);
                });
                var label=b.GetComponentInChildren<TextMeshProUGUI>();label.color=StatisticsTheme.Player(p.Color);
                Tip(b,p.Name+"\n"+p.Opponent+"\nToggle this player in all pages.");playerButtons.Add(b);x+=w+8*unit;
            }
            eraButtons.Clear();
            for(int i=0;i<(Report.Introductory?2:3);i++) {
                int era=i;string name=new[]{"Whole game","Canal","Rail"}[i];float w=new[]{142,90,80}[i]*unit;
                var b=StatisticsWidgets.Button(window,font,"Era "+name,name,new Vector2(x,-72*unit),new Vector2(w,44*unit),F(19),()=>{Filters.Era=era;QueueRender(false);});
                eraButtons.Add(b);x+=w+6*unit;
            }
            var reset=StatisticsWidgets.Button(window,font,"Reset filters","Reset filters",new Vector2(width-22*unit-130*unit*TextScale,-72*unit),new Vector2(130*unit*TextScale,44*unit),F(19),ResetFilters);
            Tip(reset,"Show all players, the whole game and every chart series; clear the search.");
            float searchWidth=Mathf.Min(570*unit,width-44*unit-470*unit*TextScale);
            search=StatisticsWidgets.Search(window,font,V(22,118),new Vector2(searchWidth,44*unit),F(21),value=>{Filters.Search=value;QueueRender(false);});
            search.textViewport.offsetMax=new Vector2(-50*unit,-3*unit);
            search.SetTextWithoutNotify(Filters.Search);
            ((TextMeshProUGUI)search.placeholder).text="Search ledger: city, industry or action…";
            clearSearch=StatisticsWidgets.Button(search.transform,font,"Clear search","×",new Vector2(searchWidth-43*unit,-4*unit),Size(38,36),25*unit,()=>{search.text="";});
            Tip(clearSearch,"Clear the ledger search.");
            float toolsX=searchWidth+32*unit;
            StatisticsWidgets.Button(window,font,"Find category","Filter detail…",new Vector2(toolsX,-118*unit),new Vector2(200*unit*TextScale,44*unit),F(19),ShowCategories);
            toolsX+=210*unit*TextScale;
            collapseRows=StatisticsWidgets.Button(window,font,"Collapse rows","Collapse rows",new Vector2(toolsX,-118*unit),new Vector2(160*unit*TextScale,44*unit),F(19),()=>{expanded.Clear();QueueRender(true);});
            Tip(collapseRows,"Collapse all expanded row details. Filters, sorting and scroll position are retained.");
            toolsX+=170*unit*TextScale;
            topButton=StatisticsWidgets.Button(window,font,"Back to top","↑ Top",new Vector2(toolsX,-118*unit),new Vector2(80*unit*TextScale,44*unit),F(19),()=>{BodyScroll.verticalNormalizedPosition=1;positions[CurrentPage]=1;});
            filterStatus=StatisticsWidgets.Text(window,font,"Filter summary","",new Vector2(Math.Max(toolsX+100*unit*TextScale,width*.70f),-118*unit),new Vector2(width-Math.Max(toolsX+100*unit*TextScale,width*.70f)-22*unit,44*unit),F(18));
            filterStatus.color=StatisticsTheme.Muted;
            float nav=258*unit;bodyY=174*unit;float bodyHeight=height-bodyY-66*unit;
            var navigation=StatisticsWidgets.Scroll(window,"Page navigation",new Vector2(16*unit,-bodyY),new Vector2(nav,bodyHeight),unit);
            float pageHeight=56*unit*TextScale;
            navigation.content.sizeDelta=new Vector2(nav-14*unit,StatisticsPages.Names.Length*pageHeight);PageButtons=new Button[StatisticsPages.Names.Length];
            for(int i=0;i<StatisticsPages.NavigationOrder.Length;i++) {
                int page=StatisticsPages.NavigationOrder[i]; PageButtons[page]=StatisticsWidgets.Button(navigation.content,font,"Page "+page,(page==3?"Networks &\nGeography":page==8?"Efficiency &\nTransition":StatisticsPages.Names[page]),new Vector2(0,-i*pageHeight),new Vector2(nav-19*unit,pageHeight-6*unit),F(21),()=>SelectPage(page));
                PageButtons[page].GetComponentInChildren<TextMeshProUGUI>().enableWordWrapping=true;PageButtons[page].GetComponentInChildren<TextMeshProUGUI>().overflowMode=TextOverflowModes.Overflow;
            }
            BodyScroll=StatisticsWidgets.Scroll(window,"Statistics page",new Vector2(nav+32*unit,-bodyY),new Vector2(width-nav-48*unit,bodyHeight),unit);
            bodyWidth=BodyScroll.viewport.sizeDelta.x-8*unit;
            sticky=StatisticsWidgets.Rect(BodyScroll.viewport,"Pinned column headings",Vector2.zero,new Vector2(bodyWidth,54*unit));sticky.gameObject.AddComponent<RectMask2D>();sticky.gameObject.SetActive(false);
            pinned=null;
            StatisticsWidgets.Button(window,font,"Return to results","Return to results",new Vector2(22*unit,-height+54*unit),new Vector2(222*unit*TextScale,44*unit),F(21),()=>Close(true));
            StatisticsWidgets.Button(window,font,"Return to board","Close · review board",new Vector2(width-22*unit-258*unit*TextScale,-height+54*unit),new Vector2(258*unit*TextScale,44*unit),F(21),()=>Close(false));
            var hint=StatisticsWidgets.Text(window,font,"Scroll hint","Wheel: scroll · Shift: sideways",new Vector2(238*unit*TextScale,-height+54*unit),new Vector2(width-518*unit*TextScale,44*unit),F(18));
            hint.alignment=TextAlignmentOptions.Center;hint.color=filterStatus.color;
            tip=StatisticsWidgets.Rect(window,"Exact values",Vector2.zero,Size(520,100));
            var tipSurface=tip.gameObject.AddComponent<StatisticsSurface>();tipSurface.Panel=true;tipSurface.raycastTarget=false;
            tipText=StatisticsWidgets.Text(tip,font,"Definition","",V(16,12),Size(488,76),F(19),true);tipText.alignment=TextAlignmentOptions.TopLeft;tip.gameObject.SetActive(false);
            measurement=StatisticsWidgets.Text(window,font,"Text measurement","",Vector2.zero,Size(1,1),F(21),true);measurement.fontStyle=FontStyles.Bold;measurement.gameObject.SetActive(false);
        }
        float F(float size) { return size*unit*TextScale; }
        float Measure(string value,float lineWidth,float size)
        { measurement.fontSize=F(size);return measurement.GetPreferredValues(value,lineWidth,0).y; }
        Vector2 V(float x,float y) { return new Vector2(x,-y)*unit; }
        Vector2 Size(float x,float y) { return new Vector2(x,y)*unit; }
        void Tip(Component target,string description) { var t=target.gameObject.AddComponent<StatisticsTooltip>();t.Panel=this;t.Description=description; }
        public void Tooltip(string value)
        {
            if(tip==null)return;tip.gameObject.SetActive(!String.IsNullOrEmpty(value));if(!String.IsNullOrEmpty(value)) {
                float w=Mathf.Min(540*unit*TextScale,width-44*unit),h=Mathf.Min(window.rect.height-130*unit,tipText.GetPreferredValues(value,w-32*unit,0).y+28*unit);
                tip.sizeDelta=new Vector2(w,h);tip.anchoredPosition=new Vector2(width-w-22*unit,-window.rect.height+h+66*unit);
                tipText.rectTransform.sizeDelta=new Vector2(w-32*unit,h-24*unit);tipText.text=value;tip.SetAsLastSibling();
            }
        }
        void CloseCategories()
        { if(dropdown==null)return;dropdown.gameObject.SetActive(false);Destroy(dropdown.gameObject);dropdown=null; }
        void ShowCategories()
        {
            if(dropdown!=null){CloseCategories();return;}
            bool industry=CurrentPage==1||CurrentPage==2||CurrentPage==8||CurrentPage==5&&Filters.ResourceView=="Merchants";
            IEnumerable<string> values=industry?StatisticsPages.IndustryTypes:CurrentPage==5?new[]{"Coal","Iron","Beer"}:
                CurrentPage==3?Report.Locations.Select(l=>l.RegionColor).Distinct().OrderBy(c=>c).ToArray():StatisticsPages.ActionTypes;
            var options=new[]{"Clear detail filters"}.Concat(values).ToArray();
            float w=320*unit*TextScale,rowHeight=46*unit*TextScale;
            dropdown=StatisticsWidgets.Rect(window,"Category choices",new Vector2(search.GetComponent<RectTransform>().sizeDelta.x+32*unit,-198*unit),new Vector2(w,Mathf.Min(options.Length*rowHeight,window.rect.height-270*unit)));
            dropdown.gameObject.AddComponent<StatisticsSurface>().Panel=true;
            var scroll=StatisticsWidgets.Scroll(dropdown,"Categories",Vector2.zero,dropdown.sizeDelta,unit);scroll.content.sizeDelta=new Vector2(w-14*unit,options.Length*rowHeight);
            for(int i=0;i<options.Length;i++) {
                string value=options[i];StatisticsWidgets.Button(scroll.content,font,"Find "+value,value,new Vector2(0,-i*rowHeight),new Vector2(w-18*unit,rowHeight-4*unit),F(21),()=>{
                    Filters.Evidence.Clear();Filters.EvidenceRestricted=false;Filters.Industry=Filters.Region=Filters.Action="";
                    if(value!="Clear detail filters") {if(industry)Filters.Industry=value;else if(CurrentPage==3)Filters.Region=value;else if(CurrentPage==5)Filters.Resource=value;else Filters.Action=value;}
                    QueueRender(false);CloseCategories();
                });
            }
            dropdown.SetAsLastSibling();
            WireNavigation();
        }
        void Render()
        {
            if(Report==null||BodyScroll==null)return;
            RememberTables();StatisticsWidgets.Clear(BodyScroll.content);tables.Clear();pinned=null;sticky.gameObject.SetActive(false); Tooltip(null);
            for(int i=0;i<StatisticsPages.Names.Length;i++) {
                StatisticsWidgets.Selected(PageButtons[i],CurrentPage==i);
                PageButtons[i].GetComponentInChildren<TextMeshProUGUI>().fontStyle=CurrentPage==i?FontStyles.Bold:FontStyles.Normal;
            }
            for(int i=0;i<playerButtons.Count;i++)StatisticsWidgets.Selected(playerButtons[i],Filters.Players.Contains(Report.Players[i].Color));
            for(int i=0;i<eraButtons.Count;i++)StatisticsWidgets.Selected(eraButtons[i],Filters.Era==i);
            var analytic=View();
            var sections=analytic.Ledgers.ToList();
            // Older historical ledgers remain available when no typed evidence scope is active.
            if(!Filters.Typed&&CurrentPage!=5)sections.AddRange(StatisticsPages.Build(CurrentPage,Report,Filters).Where(s=>s.Columns.Length>0));
            if(!String.IsNullOrWhiteSpace(Filters.Search))foreach(var s in sections)s.Rows.RemoveAll(r=>!(String.Join(" ",r.Cells)+" "+r.Detail).ToLowerInvariant().Contains(Filters.Search.Trim().ToLowerInvariant()));
            StatisticsWidgets.Enabled(clearSearch,!String.IsNullOrEmpty(Filters.Search));StatisticsWidgets.Enabled(collapseRows,expanded.Count>0);
            filterStatus.text=(Filters.Players.Count==0?"No players selected · select a player above":Filters.Players.Count+"/"+Report.Players.Count+" players")+" · "+new[]{"Whole game","Canal era","Rail era"}[Filters.Era]+" · "+sections.Sum(s=>s.Rows.Count)+" rows"+(String.IsNullOrWhiteSpace(Filters.Search)?"":" · Search filters tables");
            float y=0;
            float titleHeight=Mathf.Max(44*unit,Measure(StatisticsPages.Names[CurrentPage],bodyWidth,30)+8*unit);
            pageLabel=StatisticsWidgets.Text(BodyScroll.content,font,"Page title",StatisticsPages.Names[CurrentPage],Vector2.zero,new Vector2(bodyWidth,titleHeight),F(30),true);
            pageLabel.fontStyle=FontStyles.Bold;y+=titleHeight+10*unit;
            string availability=Report.CompleteHistory?"":Report.Availability;
            if(Report.Warnings.Count>0)availability+=(availability==""?"History note: ":" · ")+String.Join(" · ",Report.Warnings);
            if(!String.IsNullOrWhiteSpace(Filters.Search))availability+="\nSearch filters ledger rows; charts retain the player and era selection.";
            y+=Paragraph(BodyScroll.content,"Data coverage",availability,y,19);
            y+=PlotContent(analytic,y);
            foreach(var section in sections) {
                string ledgerKey=Key(section);
                var disclosure=StatisticsWidgets.Button(BodyScroll.content,font,"Data "+section.Title,(openLedgers.Contains(ledgerKey)?"− Hide data · ":"+ Show data · ")+section.Title+" · "+section.Rows.Count+" records",new Vector2(0,-y),new Vector2(bodyWidth,56*unit*TextScale),F(21),()=>{if(!openLedgers.Remove(ledgerKey))openLedgers.Add(ledgerKey);QueueRender(true);});
                disclosure.GetComponentInChildren<TextMeshProUGUI>().enableWordWrapping=true;y+=64*unit*TextScale;
                if(!openLedgers.Contains(ledgerKey))continue;
                y+=Paragraph(BodyScroll.content,"Definition",section.Definition,y,19);
                if(section.Columns.Length>0) {
                    if(section.Rows.Count==0)y+=Paragraph(BodyScroll.content,"No rows","No records match these filters. Select a player, clear the search or reset the filters.",y,21);
                    else y+=TableView(section,y);
                }
                y+=30*unit;
            }
            BodyScroll.content.sizeDelta=new Vector2(bodyWidth,Mathf.Max(y,BodyScroll.viewport.sizeDelta.y));
            Canvas.ForceUpdateCanvases();BodyScroll.verticalNormalizedPosition=positions[CurrentPage];
            if(evidenceSection!=null){var target=tables.FirstOrDefault(t=>t.Section.Title==evidenceSection);if(target!=null){float range=BodyScroll.content.rect.height-BodyScroll.viewport.rect.height;BodyScroll.verticalNormalizedPosition=range<=0?1:Mathf.Clamp01(1-target.Y/range);positions[CurrentPage]=BodyScroll.verticalNormalizedPosition;}evidenceSection=null;}
            WireNavigation();
            if(revealPage){RevealFocus(PageButtons[CurrentPage]);revealPage=false;}
        }
        float Paragraph(Transform parent,string name,string value,float y,float size)
        {
            if(String.IsNullOrEmpty(value))return 0;
            var text=StatisticsWidgets.Text(parent,font,name,value,new Vector2(0,-y),new Vector2(bodyWidth,40*unit),F(size),true);
            text.alignment=TextAlignmentOptions.TopLeft;
            float height=Mathf.Max(F(size)*1.4f,text.GetPreferredValues(value,bodyWidth,0).y+8*unit);text.rectTransform.sizeDelta=new Vector2(bodyWidth,height);return height+12*unit;
        }
        string Key(StatisticsSection section) { return CurrentPage+":"+section.Title; }
        float TableView(StatisticsSection section,float y)
        {
            string key=Key(section);int column=sortColumns.ContainsKey(key)?sortColumns[key]:-1;
            var rows=section.Rows.ToList();
            if(column>=0)rows.Sort((a,b)=>Compare(a.Cells[column],b.Cells[column])*(descending.Contains(key)?-1:1));
            int limit=limits.ContainsKey(key)?limits[key]:60;int visible=Math.Min(limit,rows.Count);
            float col=Mathf.Max(170*unit*TextScale,(bodyWidth-44*unit)/section.Columns.Length),w=col*section.Columns.Length+44*unit;
            float header=Mathf.Max(54*unit*TextScale,section.Columns.Max(c=>Measure(c+" ↓",col-20*unit,19))+12*unit);
            float rowHeight=50*unit*TextScale,detailWidth=Mathf.Min(w-32*unit,bodyWidth-32*unit),h=header;
            var heights=new float[visible];
            for(int i=0;i<visible;i++) {
                float extra=0;if(expanded.Contains(key+":"+rows[i].Key)) {
                    extra=Measure(rows[i].Detail,detailWidth,21)+24*unit;
                }
                heights[i]=rowHeight+extra;h+=heights[i];
            }
            if(visible<rows.Count)h+=52*unit*TextScale;h+=18*unit;
            var scroll=StatisticsWidgets.Scroll(BodyScroll.content,section.Title+" table",new Vector2(0,-y),new Vector2(bodyWidth,h),unit,true);
            scroll.content.sizeDelta=new Vector2(w,h-14*unit);scroll.viewport.gameObject.GetComponent<Image>().color=StatisticsTheme.Paper;
            if(horizontalPositions.ContainsKey(key))scroll.horizontalNormalizedPosition=horizontalPositions[key];
            var table=new Table{Section=section,Key=key,Scroll=scroll,Y=y,Height=h,Width=w,Column=col,Header=header};tables.Add(table);
            DrawHeaders(scroll.content,table);
            float rowY=header;
            for(int i=0;i<visible;i++) {
                var row=rows[i];string id=key+":"+row.Key;
                var b=StatisticsWidgets.Button(scroll.content,font,"Row "+row.Key,"",new Vector2(0,-rowY),new Vector2(w,rowHeight-2*unit),F(21),()=>{
                    if(!expanded.Remove(id))expanded.Add(id);QueueRender(true);
                });
                var surface=b.GetComponent<StatisticsSurface>();surface.Selected=expanded.Contains(id);surface.Row=true;surface.Alternate=i%2!=0;
                Tip(b,"Click to "+(expanded.Contains(id)?"collapse":"expand")+" details.\n"+row.Detail);
                if(row.Player.HasValue) {
                    var stripe=StatisticsWidgets.Rect(b.transform,"Player marker",Size(4,-6),new Vector2(4*unit,rowHeight-14*unit));stripe.gameObject.AddComponent<Image>().color=StatisticsTheme.Player(row.Player.Value);
                }
                for(int c=0;c<row.Cells.Length;c++)
                    StatisticsWidgets.Text(b.transform,font,"Cell "+c,row.Cells[c],new Vector2(14*unit+c*col,0),new Vector2(col-22*unit,rowHeight-2*unit),F(21));
                if(row.TargetPage>=0||row.Drill!=null) {
                    var open=StatisticsWidgets.Button(b.transform,font,"Open ledger","›",new Vector2(w-42*unit,0),new Vector2(40*unit,rowHeight-2*unit),F(28),()=>{if(row.Drill!=null)Drill(row.Drill);else Drill(row.TargetPage,row.Player??Filters.Players.FirstOrDefault(),row.TargetSearch);});
                    Tip(open,"Open the related ledger for this player.");
                }
                if(expanded.Contains(id)) {
                    var detail=StatisticsWidgets.Text(scroll.content,font,"Expanded "+row.Key,row.Detail,new Vector2(16*unit,-rowY-rowHeight-8*unit),new Vector2(detailWidth,heights[i]-rowHeight-16*unit),F(21),true);
                    detail.alignment=TextAlignmentOptions.TopLeft;
                }
                rowY+=heights[i];
            }
            if(visible<rows.Count)StatisticsWidgets.Button(scroll.content,font,"More "+section.Title,"Show next 60 · "+visible+" of "+rows.Count,new Vector2(4*unit,-rowY),new Vector2(Mathf.Min(bodyWidth-8*unit,420*unit*TextScale),46*unit*TextScale),F(21),()=>{limits[key]=limit+60;QueueRender(true);});
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
                var b=StatisticsWidgets.Button(parent,font,"Sort "+table.Section.Columns[c],title,new Vector2(c*table.Column,0),new Vector2(table.Column,table.Header-2*unit),F(19),()=>{
                    if(sortColumns.ContainsKey(key)&&sortColumns[key]==index) {if(!descending.Remove(key))descending.Add(key);}
                    else {sortColumns[key]=index;descending.Remove(key);}QueueRender(true);
                });
                StatisticsWidgets.Selected(b,sortColumns.ContainsKey(key)&&sortColumns[key]==c);
                var label=b.GetComponentInChildren<TextMeshProUGUI>();label.enableWordWrapping=true;label.fontStyle=FontStyles.Bold;
                Tip(b,"Sort by "+table.Section.Columns[c]+". Click again to reverse.");
            }
        }
        void UpdateSticky()
        {
            if(BodyScroll==null)return;float top=BodyScroll.content.anchoredPosition.y;
            var table=tables.FirstOrDefault(t=>top>t.Y+1*unit&&top<t.Y+t.Height-t.Header);
            if(table!=pinned) {
                StatisticsWidgets.Clear(sticky);pinned=table;
                if(table!=null) {
                    sticky.sizeDelta=new Vector2(bodyWidth,table.Header);
                    var header=StatisticsWidgets.Rect(sticky,"Headings",Vector2.zero,new Vector2(table.Width,table.Header));DrawHeaders(header,table);
                }
                sticky.gameObject.SetActive(table!=null);
            }
            if(table!=null&&sticky.childCount>0)((RectTransform)sticky.GetChild(sticky.childCount-1)).anchoredPosition=new Vector2(table.Scroll.content.anchoredPosition.x,0);
        }
        float Chart(StatisticsChart data,float y)
        {
            float start=y,titleHeight=Mathf.Max(36*unit,Measure(data.Title,bodyWidth,23)+8*unit);
            StatisticsWidgets.Text(BodyScroll.content,font,data.Title,data.Title,new Vector2(0,-y),new Vector2(bodyWidth,titleHeight),F(23),true);y+=titleHeight+6*unit;
            float chartHeight=240*unit*TextScale;var r=StatisticsWidgets.Rect(BodyScroll.content,data.Title+" plot",new Vector2(72*unit*TextScale,-y),new Vector2(bodyWidth-80*unit*TextScale,chartHeight));
            var chart=r.gameObject.AddComponent<StatisticsChartGraphic>();chart.Data=data;chart.Panel=this;
            var points=data.Series.SelectMany(s=>s.Points).ToArray();
            double max=points.Length==0?0:points.Max(p=>p.Y),min=points.Length==0?0:Math.Min(0,points.Min(p=>p.Y));
            if(data.Stacked) {
                max=data.Series.SelectMany(s=>s.Points.GroupBy(p=>p.X).Select(g=>g.Where(p=>p.Y>0).Sum(p=>p.Y))).DefaultIfEmpty(0).Max();
                min=data.Series.SelectMany(s=>s.Points.GroupBy(p=>p.X).Select(g=>g.Where(p=>p.Y<0).Sum(p=>p.Y))).DefaultIfEmpty(0).Min();
            }
            StatisticsWidgets.Text(BodyScroll.content,font,"Maximum",max.ToString("0.#"),new Vector2(0,-y),new Vector2(70*unit*TextScale,32*unit*TextScale),F(19));
            StatisticsWidgets.Text(BodyScroll.content,font,"Minimum",min.ToString("0.#"),new Vector2(0,-y-chartHeight+36*unit*TextScale),new Vector2(70*unit*TextScale,32*unit*TextScale),F(19));
            if(data.Lines&&points.Length>0) {
                double left=points.Min(p=>p.X),right=points.Max(p=>p.X);
                for(int i=0;i<5;i++)StatisticsWidgets.Text(BodyScroll.content,font,"Turn tick "+i,(left+(right-left)*i/4).ToString("0"),new Vector2(88*unit*TextScale+(bodyWidth-170*unit*TextScale)*i/4,-y-chartHeight+32*unit),new Vector2(65*unit*TextScale,32*unit),F(18));
            }
            y+=chartHeight+5*unit;
            string axes=data.Lines?"Turn number · "+data.Unit:data.Stacked?"Player bars · "+(Filters.Era==0&&!Report.Introductory?"Canal on left / Rail on right":Filters.Era==2?"Rail":"Canal")+" · "+data.Unit:"Actions in order: "+String.Join(", ",StatisticsPages.ActionTypes);
            y+=Paragraph(BodyScroll.content,"Chart axes",axes,y,19);
            float x=0,legendHeight=44*unit*TextScale;
            for(int i=0;i<data.Series.Count;i++) {
                string id=CurrentPage+":"+data.Title+":"+data.Series[i].Player;
                if(hiddenSeries.Contains(id))chart.Hidden.Add(i);
                float w=Mathf.Min(270*unit*TextScale,bodyWidth/Math.Max(1,data.Series.Count));
                var legend=StatisticsWidgets.Button(BodyScroll.content,font,"Legend "+data.Series[i].Label,data.Series[i].Label,new Vector2(x,-y),new Vector2(w-6*unit,legendHeight),F(21),()=>{
                    if(!hiddenSeries.Remove(id))hiddenSeries.Add(id);QueueRender(true);
                });StatisticsWidgets.Selected(legend,!hiddenSeries.Contains(id));
                legend.GetComponentInChildren<TextMeshProUGUI>().color=hiddenSeries.Contains(id)?new Color(.65f,.63f,.59f,1):StatisticsWidgets.PlayerColor(data.Series[i].Player);
                Tip(legend,data.Series[i].Label+"\nClick to "+(hiddenSeries.Contains(id)?"show":"hide")+" this player's series. Reset filters restores all series.");x+=w;
            }
            y+=legendHeight+12*unit;
            if(data.Stacked)y+=Paragraph(BodyScroll.content,"Score legend","Green: industries · Blue: links · Gold: merchant · Red: penalties · Purple: introductory additions",y,19);
            return y-start;
        }
        void WireNavigation()
        {
            var controls=window.GetComponentsInChildren<Selectable>().Where(s=>s.IsActive()&&s.IsInteractable()).ToArray();
            for(int i=0;i<controls.Length;i++) {
                var n=controls[i].navigation;n.mode=Navigation.Mode.Explicit;n.selectOnUp=controls[(i-1+controls.Length)%controls.Length];n.selectOnDown=controls[(i+1)%controls.Length];
                n.selectOnLeft=n.selectOnUp;n.selectOnRight=n.selectOnDown;controls[i].navigation=n;
                if(controls[i].GetComponent<StatisticsFocusReveal>()==null) {
                    var reveal=controls[i].gameObject.AddComponent<StatisticsFocusReveal>();reveal.Panel=this;
                }
            }
        }
        public void RevealFocus(Selectable control)
        {
            if(control==null)return;
            var scroll=control.GetComponentsInParent<ScrollRect>().FirstOrDefault(s=>s.vertical&&control.transform.IsChildOf(s.content));
            if(scroll==null)return;
            Canvas.ForceUpdateCanvases();var corners=new Vector3[4];((RectTransform)control.transform).GetWorldCorners(corners);
            float top=scroll.viewport.InverseTransformPoint(corners[1]).y,bottom=scroll.viewport.InverseTransformPoint(corners[0]).y;
            float shift=top>scroll.viewport.rect.yMax?scroll.viewport.rect.yMax-top:bottom<scroll.viewport.rect.yMin?scroll.viewport.rect.yMin-bottom:0;
            var at=scroll.content.anchoredPosition;at.y=Mathf.Clamp(at.y+shift,0,Mathf.Max(0,scroll.content.rect.height-scroll.viewport.rect.height));scroll.content.anchoredPosition=at;
        }
        public override void HandleKeyDown(KeyEnum key)
        {
            if(key==KeyEnum.Escape){if(dropdown!=null){CloseCategories();WireNavigation();}else Close();return;}
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
