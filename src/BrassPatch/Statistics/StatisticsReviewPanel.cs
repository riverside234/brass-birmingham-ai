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
        bool moreObservations;
        // Cache analytical values only. UI objects and ledger-search results are never cached.
        StatisticsPageView View()
        {
            string key=CurrentPage+"/"+Filters.Era+"/"+String.Join(",",Filters.Players.OrderBy(p=>(int)p))+"/"+Filters.Industry+"/"+Filters.Region+"/"+Filters.Action+"/"+Filters.Resource+"/"+Filters.Outgoing+"/"+Filters.ShowAll+"/"+Filters.Boundary+"/"+Filters.BoundaryEra+"/"+Filters.EvidenceRestricted+"/"+Filters.MapMode+"/"+Filters.Focus+"/"+Filters.Reference+"/"+Filters.AllCashPayments+"/"+String.Join(",",Filters.Evidence.OrderBy(x=>x));
            StatisticsPageView view;if(!views.TryGetValue(key,out view)){if(views.Count>64)views.Clear();view=StatisticsPlotPages.Build(CurrentPage,Report,Filters);views[key]=view;}
            // Clone row lists because text search must not mutate a cached view.
            var copy=new StatisticsPageView{Context=view.Context};copy.Headlines.AddRange(view.Headlines);copy.Plots.AddRange(view.Plots);
            foreach(var s in view.Ledgers){var section=new StatisticsSection(s.Title,s.Definition,s.Columns);section.Rows.AddRange(s.Rows);copy.Ledgers.Add(section);}return copy;
        }
        public void Drill(StatisticsDrill target)
        {
            if(target==null)return;Filters.Industry=target.Industry;Filters.Region=target.Region;Filters.Action=target.Action;Filters.Evidence.Clear();foreach(var key in target.Keys)Filters.Evidence.Add(key);
            Filters.EvidenceRestricted=target.HasKeys||target.Keys.Count>0;
            if(target.Era.HasValue)Filters.Era=target.Era.Value;
            if(target.Resource!="")Filters.Resource=target.Resource;
            if(target.Outgoing.HasValue)Filters.Outgoing=target.Outgoing.Value;
            if(target.Player.HasValue){Filters.Players.Clear();Filters.Players.Add(target.Player.Value);}
            Filters.Search="";if(search!=null)search.SetTextWithoutNotify("");positions[target.Page]=1;SelectPage(target.Page);
            // Evidence is visible immediately, without requiring another disclosure click.
            var view=StatisticsPlotPages.Build(target.Page,Report,Filters);foreach(var s in view.Ledgers)openLedgers.Add(target.Page+":"+s.Title);
        }
        float ReviewContent(StatisticsPageView view,float start)
        {
            float y=start;
            if(CurrentPage==10||CurrentPage==1||CurrentPage==4){
                var focus=StatisticsMetrics.Focus(Report,Filters);var reference=StatisticsMetrics.Reference(Report,Filters);
                float w=Math.Min(bodyWidth/2-6*unit,380*unit*TextScale);
                var b=StatisticsWidgets.Button(BodyScroll.content,font,"Review player","Review: "+focus.Name,new Vector2(0,-y),new Vector2(w,48*unit*TextScale),F(21),()=>PlayerChoice(false));
                Tip(b,"Choose the player whose whole-match review and cash waterfall are shown. Detail-player toggles remain independent.");
                if(CurrentPage!=4){b=StatisticsWidgets.Button(BodyScroll.content,font,"Reference player","Compare: "+(reference==null?"—":reference.Name),new Vector2(w+12*unit,-y),new Vector2(w,48*unit*TextScale),F(21),()=>PlayerChoice(true));Tip(b,"Default: highest-ranked other player (winner for a non-winner; runner-up for a winner). VP ties stay tied in the score-gap plot.");}
                y+=58*unit*TextScale;
            }
            if(CurrentPage==5){
                float x=0;foreach(string resource in new[]{"Coal","Iron","Beer"}){string name=resource;var b=StatisticsWidgets.Button(BodyScroll.content,font,"Resource "+name,name,new Vector2(x,-y),Size(100*TextScale,44*TextScale),F(21),()=>{Filters.Resource=name;Filters.Evidence.Clear();Filters.EvidenceRestricted=false;QueueRender(false);});StatisticsWidgets.Selected(b,Filters.Resource==name);x+=110*unit*TextScale;}
                StatisticsWidgets.Button(BodyScroll.content,font,"Flow direction",Filters.Outgoing?"Outgoing supply":"Incoming demand",new Vector2(x,-y),Size(200*TextScale,44*TextScale),F(21),()=>{Filters.Outgoing=!Filters.Outgoing;Filters.Evidence.Clear();Filters.EvidenceRestricted=false;QueueRender(false);});y+=56*unit*TextScale;
            }
            if(CurrentPage==3||CurrentPage==9){
                StatisticsWidgets.Button(BodyScroll.content,font,"Show all plot records",Filters.ShowAll?"Group top 8 + Other":"Show all plot records",new Vector2(0,-y),new Vector2(Math.Min(bodyWidth,300*unit*TextScale),46*unit*TextScale),F(21),()=>{Filters.ShowAll=!Filters.ShowAll;QueueRender(true);});
                if(CurrentPage==3){var b=StatisticsWidgets.Button(BodyScroll.content,font,"Map boundary","Boundary: "+Filters.Boundary+(Filters.BoundaryEra==0?"":Filters.BoundaryEra==1?" · Canal":" · Rail"),new Vector2(310*unit*TextScale,-y),new Vector2(Math.Max(160*unit,bodyWidth-310*unit*TextScale),46*unit*TextScale),F(21),BoundaryChoice);Tip(b,"The schematic uses historical occupancy at this boundary, not final board state for every era.");}y+=58*unit*TextScale;
                if(CurrentPage==3){StatisticsWidgets.Button(BodyScroll.content,font,"Map measure","Map: "+Filters.MapMode,new Vector2(0,-y),new Vector2(Math.Min(bodyWidth,320*unit*TextScale),46*unit*TextScale),F(21),()=>{string[] modes={"Occupancy","Industry VP","Link contributions"};Filters.MapMode=modes[(Array.IndexOf(modes,Filters.MapMode)+1)%modes.Length];QueueRender(true);});y+=58*unit*TextScale;}
            }
            y+=Paragraph(BodyScroll.content,"Plot context",view.Context,y,21);
            // Wrapping metric strip avoids tiny cards at 720p and the largest text setting.
            y+=Paragraph(BodyScroll.content,"Headline values",String.Join("   ·   ",view.Headlines),y,24);
            foreach(var plot in view.Plots){
                var heading=StatisticsWidgets.Text(BodyScroll.content,font,"Plot title",plot.Title,new Vector2(0,-y),new Vector2(bodyWidth,40*unit),F(24),true);heading.color=StatisticsWidgets.Gold;heading.fontStyle=FontStyles.Bold;
                float h=Math.Max(40*unit,heading.GetPreferredValues(plot.Title,bodyWidth,0).y+8*unit);heading.rectTransform.sizeDelta=new Vector2(bodyWidth,h);y+=h+6*unit;
                y+=Paragraph(BodyScroll.content,"Plot definition",plot.Definition,y,21);
                if(plot.Marks.Count==0)y+=Paragraph(BodyScroll.content,"Empty plot","No recorded values match this scope. Reset filters to restore the full view.",y,21);
                else y+=StatisticsPlotView.Create(BodyScroll.content,font,plot,y,bodyWidth,unit,TextScale,this,hiddenSeries,()=>QueueRender(true));
                if(plot.Context.Count>0)y+=Paragraph(BodyScroll.content,"Plot coverage",String.Join("\n",plot.Context),y,19);
                y+=20*unit;
            }
            if(CurrentPage==10)y+=Observations(y);
            return y-start;
        }
        float Observations(float start)
        {
            float y=start;var focus=StatisticsMetrics.Focus(Report,Filters);var observations=StatisticsInsights.Build(Report,focus.Color);
            if(!Report.CompleteHistory)return Paragraph(BodyScroll.content,"Advice unavailable","Tactical observations are unavailable because the recorded history is incomplete.",y,21);
            var strengths=observations.Where(o=>o.Strength).Take(3).ToArray();var concerns=observations.Where(o=>!o.Strength).OrderBy(o=>o.Title.StartsWith("Debt")?0:o.Title.StartsWith("Unsold")?1:2).Take(3).ToArray();
            foreach(var group in new[]{strengths,concerns}){
                y+=Paragraph(BodyScroll.content,"Observation heading",group==strengths?"WHAT WORKED":"REVIEW NEXT TIME",y,24);
                if(group.Length==0)y+=Paragraph(BodyScroll.content,"No supported observations","No supported "+(group==strengths?"strength pattern":"concern")+" was found by these rules. Explore the plots for further context.",y,21);
                foreach(var o in group)y+=Observation(o,y);
            }
            y+=Paragraph(BodyScroll.content,"Experiment heading","NEXT-GAME EXPERIMENTS",y,24);
            int i=1;foreach(var o in concerns.Take(3))y+=Paragraph(BodyScroll.content,"Experiment",(i++)+". "+o.Experiment,y,21);
            if(concerns.Length==0&&strengths.Length>0)y+=Paragraph(BodyScroll.content,"Experiment",strengths[0].Experiment,y,21);
            var more=observations.Except(strengths).Except(concerns).ToArray();if(more.Length>0){StatisticsWidgets.Button(BodyScroll.content,font,"More observations",(moreObservations?"Hide":"More")+" observations · "+more.Length,new Vector2(0,-y),new Vector2(Math.Min(bodyWidth,350*unit*TextScale),48*unit*TextScale),F(21),()=>{moreObservations=!moreObservations;QueueRender(true);});y+=60*unit*TextScale;if(moreObservations)foreach(var o in more)y+=Observation(o,y);}
            y+=Paragraph(BodyScroll.content,"Review limitations","These are local, deterministic observations, not an optimal replay or strategy grade. Loans, shared supplies and zero-VP links can be useful. The report cannot establish opponent intent or a better legal move.",y,19);return y-start;
        }
        float Observation(StatisticsObservation o,float start)
        {
            float y=start;y+=Paragraph(BodyScroll.content,"Observation title",o.Title,y,24);
            y+=Paragraph(BodyScroll.content,"Observed","Observed: "+o.Fact,y,21);y+=Paragraph(BodyScroll.content,"Interpretation","Interpretation: "+o.Interpretation,y,21);
            var b=StatisticsWidgets.Button(BodyScroll.content,font,"Evidence "+o.Title,"Show evidence",new Vector2(0,-y),new Vector2(Math.Min(bodyWidth,220*unit*TextScale),44*unit*TextScale),F(21),()=>Drill(o.Evidence));Tip(b,"Supporting record IDs. Review facts and their evidence cover the whole match.");return y-start+60*unit*TextScale;
        }
        void PlayerChoice(bool reference)
        {
            var focus=StatisticsMetrics.Focus(Report,Filters);
            var players=Report.Players.Where(p=>!reference||p.Color!=focus.Color).ToArray();
            Choices(reference?"Compare with":"Review player",players.Select(p=>p.Name+" · "+p.Color).ToArray(),index=>{if(reference)Filters.Reference=players[index].Color;else{Filters.Focus=players[index].Color;if(Filters.Reference==Filters.Focus)Filters.Reference=null;}QueueRender(false);});
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
