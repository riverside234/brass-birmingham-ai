using System;
using System.Collections.Generic;
using System.Linq;
using TMPro;
using UnityEngine;
namespace BrassPatch.Statistics
{
    public sealed partial class StatisticsPanel
    {
        string evidenceSection;
        void OpenChartData(string title){openLedgers.Add(CurrentPage+":"+title);evidenceSection=title;QueueRender(true);}
        readonly Dictionary<string,float> resourcePositions=new Dictionary<string,float>();
        public readonly Dictionary<string,string> SelectedMarks=new Dictionary<string,string>();
        void SetResourceView(string view)
        {
            resourcePositions[Filters.ResourceView]=BodyScroll.verticalNormalizedPosition;
            Filters.ResourceView=view;Filters.Evidence.Clear();Filters.EvidenceRestricted=false;
            Filters.Action=Filters.Industry=Filters.Region="";Filters.Outgoing=false;
            positions[5]=resourcePositions.ContainsKey(view)?resourcePositions[view]:1;
            renderPending=true;CloseCategories();Tooltip(null);
        }
        float PlotControls(float y)
        {
            float start=y;
            if(CurrentPage==8) {
                var b=StatisticsWidgets.Button(BodyScroll.content,font,"Cost denominator",Filters.AllCashPayments?"Cost: all cash payments (incl. debt)":"Cost: action spending",new Vector2(0,-y),new Vector2(Math.Min(bodyWidth,480*unit*TextScale),46*unit*TextScale),F(21),()=>{Filters.AllCashPayments=!Filters.AllCashPayments;QueueRender(true);});
                Tip(b,"Both efficiency panels and the ratio ledger use player/era totals. Market costs are already included in action spending.");y+=56*unit*TextScale;
            }
            if(CurrentPage==2||CurrentPage==6) {
                StatisticsWidgets.Button(BodyScroll.content,font,"Group new plot",Filters.ByPlayer?"By player":"Selected players total",new Vector2(0,-y),Size(260*TextScale,44*TextScale),F(21),()=>{Filters.ByPlayer=!Filters.ByPlayer;QueueRender(true);});
                if(CurrentPage==2)StatisticsWidgets.Button(BodyScroll.content,font,"Development source","Source: "+Filters.DevelopmentSource,new Vector2(274*unit*TextScale,-y),Size(280*TextScale,44*TextScale),F(21),()=>Choices("Development source",new[]{"All","Develop action","Merchant"},i=>{Filters.DevelopmentSource=new[]{"All","Develop action","Merchant"}[i];QueueRender(true);}));
                else {StatisticsWidgets.Button(BodyScroll.content,font,"Card display",Filters.CardShare?"Cards: Share":"Cards: Counts",new Vector2(274*unit*TextScale,-y),Size(240*TextScale,44*TextScale),F(21),()=>{Filters.CardShare=!Filters.CardShare;QueueRender(true);});StatisticsWidgets.Button(BodyScroll.content,font,"Action display",Filters.ActionShare?"Actions: Share":"Actions: Counts",new Vector2(528*unit*TextScale,-y),Size(260*TextScale,44*TextScale),F(21),()=>{Filters.ActionShare=!Filters.ActionShare;QueueRender(true);});}
                y+=56*unit*TextScale;
            }
            if(CurrentPage==3){StatisticsWidgets.Button(BodyScroll.content,font,"Link display",Filters.LinkShare?"Link sources: Share":"Link sources: VP",new Vector2(335*unit*TextScale,-y+58*unit*TextScale),Size(280*TextScale,46*TextScale),F(21),()=>{Filters.LinkShare=!Filters.LinkShare;QueueRender(true);});}
            if(CurrentPage==5&&Filters.ResourceView=="Merchants") {
                StatisticsWidgets.Button(BodyScroll.content,font,"Merchant industry","Industry: "+(Filters.Industry==""?"All goods":StatisticsMetrics.IndustryName(Filters.Industry)),new Vector2(0,-y),Size(400*TextScale,44*TextScale),F(21),()=>Choices("Goods industry",new[]{"All goods","Cotton mills","Manufactories","Potteries"},i=>{Filters.Industry=new[]{"","CottonMill","Manufacturer","Pottery"}[i];Filters.Evidence.Clear();Filters.EvidenceRestricted=false;QueueRender(false);}));y+=56*unit*TextScale;
            }
            return y-start;
        }
        float ResultStrip(float y)
        {
            float start=y,rowHeight=76*unit*TextScale,gap=12*unit,cardWidth=(bodyWidth-gap)/2;
            var heading=StatisticsWidgets.Text(BodyScroll.content,font,"Final ranking heading","FINAL RESULT · whole match",new Vector2(0,-y),new Vector2(bodyWidth,34*unit*TextScale),F(22));heading.fontStyle=FontStyles.Bold;heading.color=StatisticsTheme.Accent;y+=40*unit*TextScale;
            int index=0;foreach(var p in Report.Players.OrderBy(p=>p.Rank).ThenBy(p=>(int)p.Color)) {
                float x=(index%2)*(cardWidth+gap),cy=y+(index/2)*(rowHeight+6*unit);
                var rect=StatisticsWidgets.Rect(BodyScroll.content,"Final result "+p.Color,new Vector2(x,-cy),new Vector2(cardWidth,rowHeight));var surface=rect.gameObject.AddComponent<StatisticsSurface>();surface.Row=true;surface.Alternate=p.Rank%2==0;surface.raycastTarget=false;
                string score=p.FinalVP+" VP";measurement.fontSize=F(21);float scoreWidth=Math.Max(100*unit*TextScale,measurement.GetPreferredValues(score,0,0).x+20*unit);
                var name=StatisticsWidgets.Text(rect,font,"Rank and player",p.Rank+" · "+StatisticsMetrics.PlayerName(Report,p),new Vector2(14*unit,-3*unit),new Vector2(cardWidth-42*unit-scoreWidth,34*unit*TextScale),F(21));name.color=StatisticsTheme.Player(p.Color);name.fontStyle=FontStyles.Bold;surface.raycastTarget=true;Tip(rect,StatisticsMetrics.PlayerName(Report,p));
                var value=StatisticsWidgets.Text(rect,font,"Final VP",score,new Vector2(cardWidth-scoreWidth-14*unit,-3*unit),new Vector2(scoreWidth,34*unit*TextScale),F(21));value.color=StatisticsTheme.Player(p.Color);value.fontStyle=FontStyles.Bold;value.alignment=TextAlignmentOptions.MidlineRight;value.overflowMode=TextOverflowModes.Overflow;
                string eras=Report.CompleteHistory?"C "+Report.Entries.Where(e=>e.Kind=="Balance"&&e.Player==p.Color&&e.Era==1).Sum(e=>e.VP)+(Report.Introductory?"":" / R "+Report.Entries.Where(e=>e.Kind=="Balance"&&e.Player==p.Color&&e.Era==2).Sum(e=>e.VP)):"Era details unavailable";
                StatisticsWidgets.Text(rect,font,"Final totals",eras+"    ·    Income "+p.FinalIncome+"    ·    "+StatisticsFormat.Money(p.FinalCash)+" cash",new Vector2(14*unit,-36*unit*TextScale),new Vector2(cardWidth-28*unit,34*unit*TextScale),F(21));index++;
            }
            y+=(float)Math.Ceiling(index/2.0)*(rowHeight+6*unit);return y-start+12*unit;
        }
        public void SelectPlotMark(StatisticsPlotGraphic graphic,StatisticsPlotMark mark)
        {
            SelectedMarks[graphic.Plot.Id]=graphic.Identity(mark);graphic.SetVerticesDirty();Tooltip(graphic.Description(mark));
            if(mark.Target!=null)Drill(mark.Target);
        }
        public void ChoosePlotMarks(StatisticsPlotGraphic graphic,IList<StatisticsPlotMark> marks)
        {
            Choices("Overlapping records",marks.Select(m=>m.Label+" · "+StatisticsPlotSemantics.Value(graphic.Plot,m)).ToArray(),i=>SelectPlotMark(graphic,marks[i]));
        }
    }
}
