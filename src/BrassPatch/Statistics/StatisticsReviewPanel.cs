using System;
using System.Collections.Generic;
using System.Linq;
using BoardGameRules.Entities.Players;
using TMPro;
using UnityEngine;
namespace BrassPatch.Statistics
{
    public sealed partial class StatisticsPanel
    {
        readonly HashSet<string> openLedgers=new HashSet<string>();
        readonly Dictionary<string,StatisticsPageView> views=new Dictionary<string,StatisticsPageView>();
        // Cache analytical values only. UI objects and ledger-search results are never cached.
        StatisticsPageView View()
        {
            string key=CurrentPage+"/"+Filters.Era+"/"+String.Join(",",Filters.Players.OrderBy(p=>(int)p))+"/"+Filters.Industry+"/"+Filters.Region+"/"+Filters.Action+"/"+Filters.Resource+"/"+Filters.Outgoing+"/"+Filters.ShowAll+"/"+Filters.Boundary+"/"+Filters.BoundaryEra+"/"+Filters.EvidenceRestricted+"/"+Filters.MapMode+"/"+Filters.Focus+"/"+Filters.Reference+"/"+Filters.AllCashPayments+"/"+Filters.ResourceView+"/"+Filters.EfficiencyView+"/"+Filters.DevelopmentSource+"/"+Filters.ByPlayer+"/"+Filters.ActionShare+"/"+Filters.CardShare+"/"+Filters.LinkShare+"/"+String.Join(",",Filters.Evidence.OrderBy(x=>x));
            StatisticsPageView view;if(!views.TryGetValue(key,out view)){if(views.Count>64)views.Clear();view=StatisticsPlotPages.Build(CurrentPage,Report,Filters);views[key]=view;}
            // Clone row lists because text search must not mutate a cached view.
            var copy=new StatisticsPageView{Context=view.Context};copy.Headlines.AddRange(view.Headlines);copy.Plots.AddRange(view.Plots);
            foreach(var s in view.Ledgers){var section=new StatisticsSection(s.Title,s.Definition,s.Columns){Id=s.Id};section.Rows.AddRange(s.Rows);copy.Ledgers.Add(section);}return copy;
        }
        public void Drill(StatisticsDrill target)
        {
            if(target==null)return;target.Page=StatisticsPages.NormalizePage(target.Page);Filters.Industry=target.Industry;Filters.Region=target.Region;Filters.Action=target.Action;Filters.Evidence.Clear();foreach(var key in target.Keys)Filters.Evidence.Add(key);
            Filters.EvidenceRestricted=target.HasKeys||target.Keys.Count>0;
            if(target.Era.HasValue)Filters.Era=target.Era.Value;
            if(target.Resource!="")Filters.Resource=target.Resource;
            if(target.ResourceView!=null)Filters.ResourceView=target.ResourceView;if(target.EfficiencyView!=null)Filters.EfficiencyView=target.EfficiencyView;
            if(target.AllCashPayments.HasValue)Filters.AllCashPayments=target.AllCashPayments.Value;
            if(target.Outgoing.HasValue)Filters.Outgoing=target.Outgoing.Value;
            if(target.Player.HasValue){Filters.Players.Clear();Filters.Players.Add(target.Player.Value);}
            Filters.Search="";if(search!=null)search.SetTextWithoutNotify("");positions[target.Page]=1;evidenceSection="Supporting records by identity";SelectPage(target.Page);
            // Evidence is visible immediately, without requiring another disclosure click.
            var view=StatisticsPlotPages.Build(target.Page,Report,Filters);foreach(var s in view.Ledgers)openLedgers.Add(target.Page+":"+s.Identity);
        }
        float PlotContent(StatisticsPageView view,float start)
        {
            float y=start;
            if(CurrentPage==1||CurrentPage==4){
                var focus=StatisticsMetrics.Focus(Report,Filters);var reference=StatisticsMetrics.Reference(Report,Filters);
                float w=Math.Min(bodyWidth/2-6*unit,380*unit*TextScale);
                var b=StatisticsWidgets.Button(BodyScroll.content,font,"Player","Player: "+focus.Name,new Vector2(0,-y),new Vector2(w,48*unit*TextScale),F(21),()=>PlayerChoice(false));
                Tip(b,"Choose the selected player for the whole-match score-gap or cash reconciliation. Detail-player toggles remain independent.");
                if(CurrentPage!=4){b=StatisticsWidgets.Button(BodyScroll.content,font,"Reference player","Compare with: "+(reference==null?"—":reference.Name),new Vector2(w+12*unit,-y),new Vector2(w,48*unit*TextScale),F(21),()=>PlayerChoice(true));Tip(b,"Default: highest-ranked other player (winner for a non-winner; runner-up for a winner). VP ties stay tied in the score-gap plot.");}
                y+=58*unit*TextScale;
            }
            if(CurrentPage==5){
                float tabsX=0;foreach(string localView in new[]{"Supply","Market","Merchants"}){string selected=localView;var tab=StatisticsWidgets.Button(BodyScroll.content,font,"Resource view "+selected,selected,new Vector2(tabsX,-y),Size(180*TextScale,44*TextScale),F(21),()=>SetResourceView(selected));StatisticsWidgets.Selected(tab,Filters.ResourceView==selected);tabsX+=190*unit*TextScale;}y+=56*unit*TextScale;
            }
            if(CurrentPage==5&&Filters.ResourceView=="Supply"){
                float x=0;foreach(string resource in new[]{"Coal","Iron","Beer"}){string name=resource;var b=StatisticsWidgets.Button(BodyScroll.content,font,"Resource "+name,name,new Vector2(x,-y),Size(100*TextScale,44*TextScale),F(21),()=>{Filters.Resource=name;Filters.Evidence.Clear();Filters.EvidenceRestricted=false;QueueRender(false);});StatisticsWidgets.Selected(b,Filters.Resource==name);x+=110*unit*TextScale;}
                StatisticsWidgets.Button(BodyScroll.content,font,"Flow direction",Filters.Outgoing?"Outgoing supply":"Incoming demand",new Vector2(x,-y),Size(200*TextScale,44*TextScale),F(21),()=>{Filters.Outgoing=!Filters.Outgoing;Filters.Evidence.Clear();Filters.EvidenceRestricted=false;QueueRender(false);});y+=56*unit*TextScale;
            }
            if(CurrentPage==3||CurrentPage==9){
                StatisticsWidgets.Button(BodyScroll.content,font,"Show all plot records",Filters.ShowAll?(CurrentPage==9?"Show first 8 intervals":"Group top 8 + Other"):"Show all plot records",new Vector2(0,-y),new Vector2(Math.Min(bodyWidth,300*unit*TextScale),46*unit*TextScale),F(21),()=>{Filters.ShowAll=!Filters.ShowAll;QueueRender(true);});
                if(CurrentPage==3){var b=StatisticsWidgets.Button(BodyScroll.content,font,"Map boundary","Boundary: "+Filters.Boundary+(Filters.BoundaryEra==0?"":Filters.BoundaryEra==1?" · Canal":" · Rail"),new Vector2(310*unit*TextScale,-y),new Vector2(Math.Max(160*unit,bodyWidth-310*unit*TextScale),46*unit*TextScale),F(21),BoundaryChoice);Tip(b,"The schematic uses historical occupancy at this boundary, not final board state for every era.");}y+=58*unit*TextScale;
                if(CurrentPage==3){StatisticsWidgets.Button(BodyScroll.content,font,"Map measure","Map: "+Filters.MapMode,new Vector2(0,-y),new Vector2(Math.Min(bodyWidth,320*unit*TextScale),46*unit*TextScale),F(21),()=>{string[] modes={"Occupancy","Industry VP","Link contributions"};Filters.MapMode=modes[(Array.IndexOf(modes,Filters.MapMode)+1)%modes.Length];QueueRender(true);});y+=58*unit*TextScale;}
            }
            y+=PlotControls(y);
            if(CurrentPage==0)y+=ResultStrip(y);
            if(CurrentPage!=0)y+=Paragraph(BodyScroll.content,"Plot context",view.Context,y,19);
            // Wrapping metric strip avoids tiny cards at 720p and the largest text setting.
            if(CurrentPage!=0)y+=Paragraph(BodyScroll.content,"Headline values",String.Join("   ·   ",view.Headlines),y,24);
            foreach(var plot in view.Plots){
                var heading=StatisticsWidgets.Text(BodyScroll.content,font,"Plot title",plot.Title,new Vector2(0,-y),new Vector2(bodyWidth,40*unit),F(24),true);heading.color=StatisticsTheme.Accent;heading.fontStyle=FontStyles.Bold;
                float h=Math.Max(40*unit,heading.GetPreferredValues(plot.Title,bodyWidth,0).y+8*unit);heading.rectTransform.sizeDelta=new Vector2(bodyWidth,h);y+=h+6*unit;
                var help=StatisticsWidgets.Button(BodyScroll.content,font,"About "+plot.Id,"About this chart",new Vector2(0,-y),new Vector2(200*unit*TextScale,36*unit*TextScale),F(18),()=>Tooltip(plot.Definition));Tip(help,plot.Definition);y+=44*unit*TextScale;
                if(plot.Marks.Count==0)y+=Paragraph(BodyScroll.content,"Empty plot","No recorded values match this scope. Reset filters to restore the full view.",y,21);
                else y+=StatisticsPlotView.Create(BodyScroll.content,font,plot,y,bodyWidth,unit,TextScale,this,hiddenSeries,()=>QueueRender(true));
                var exact=StatisticsWidgets.Button(BodyScroll.content,font,"Exact data "+plot.Id,"Exact data · "+plot.Marks.Count+" marks",new Vector2(0,-y),new Vector2(280*unit*TextScale,40*unit*TextScale),F(19),()=>OpenChartData("plot/"+plot.Id));y+=48*unit*TextScale;
                if(plot.Context.Count>0)y+=Paragraph(BodyScroll.content,"Plot coverage",String.Join("\n",plot.Context),y,19);
                y+=20*unit;
            }
            return y-start;
        }
        void PlayerChoice(bool reference)
        {
            var focus=StatisticsMetrics.Focus(Report,Filters);
            var players=Report.Players.Where(p=>!reference||p.Color!=focus.Color).ToArray();
            Choices(reference?"Compare with":"Player",players.Select(p=>p.Name+" · "+p.Color).ToArray(),index=>{if(reference)Filters.Reference=players[index].Color;else{Filters.Focus=players[index].Color;if(Filters.Reference==Filters.Focus)Filters.Reference=null;}QueueRender(false);});
        }
        void BoundaryChoice()
        {
            string[] names=Report.Introductory?new[]{"Setup","Canal scoring","Final"}:new[]{"Setup","Canal scoring","After cleanup","Rail entry","Rail scoring","Final"};
            Choices("Historical boundary",names,index=>{var name=names[index];Filters.Boundary=name.EndsWith("scoring")?"Scoring":name;Filters.BoundaryEra=name=="Canal scoring"?1:name=="Rail scoring"?2:0;QueueRender(false);});
        }
        void Choices(string title,string[] values,Action<int> select)
        {
            CloseCategories();float w=Math.Min(480*unit*TextScale,width-44*unit),rh=52*unit*TextScale;
            dropdown=StatisticsWidgets.Rect(window,title,V(24,240),new Vector2(w,Math.Min(values.Length*rh,window.rect.height-310*unit)));dropdown.gameObject.AddComponent<StatisticsSurface>().Panel=true;
            var scroll=StatisticsWidgets.Scroll(dropdown,title+" choices",Vector2.zero,dropdown.sizeDelta,unit);scroll.content.sizeDelta=new Vector2(w-14*unit,values.Length*rh);
            for(int i=0;i<values.Length;i++){int n=i;StatisticsWidgets.Button(scroll.content,font,title+" "+n,values[i],new Vector2(0,-i*rh),new Vector2(w-18*unit,rh-4*unit),F(21),()=>{select(n);CloseCategories();});}dropdown.SetAsLastSibling();WireNavigation();
        }
    }
}
