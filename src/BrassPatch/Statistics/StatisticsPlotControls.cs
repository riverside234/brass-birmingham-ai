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
        readonly Dictionary<string,float> efficiencyPositions=new Dictionary<string,float>();
        public void SetEfficiencyView(string view)
        {
            if(view!="Returns"&&view!="Conversion"&&view!="Transition & networks")view="Returns";
            if(BodyScroll!=null)efficiencyPositions[Filters.EfficiencyView]=BodyScroll.verticalNormalizedPosition;
            Filters.EfficiencyView=view;
            positions[8]=efficiencyPositions.ContainsKey(view)?efficiencyPositions[view]:1;
            CloseCategories();Tooltip(null);renderPending=true;
        }
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
                float x=0;foreach(string view in new[]{"Returns","Conversion","Transition & networks"}) {
                    string selected=view;float w=(selected=="Transition & networks"?300:190)*unit*TextScale;
                    var tab=StatisticsWidgets.Button(BodyScroll.content,font,"Efficiency view "+selected,selected,new Vector2(x,-y),new Vector2(w,44*unit*TextScale),F(21),()=>SetEfficiencyView(selected));StatisticsWidgets.Selected(tab,Filters.EfficiencyView==selected);x+=w+10*unit;
                }
                y+=56*unit*TextScale;
                if(Filters.EfficiencyView!="Conversion") {
                    var group=StatisticsWidgets.Button(BodyScroll.content,font,"Group efficiency",Filters.ByPlayer?"Industries: by player":"Industries: players combined",new Vector2(0,-y),Size(350*TextScale,44*TextScale),F(21),()=>{Filters.ByPlayer=!Filters.ByPlayer;QueueRender(true);});
                    Tip(group,"Construction return and final unflipped investment only. Player panels share one numeric scale.");y+=54*unit*TextScale;
                }
            }
            if(CurrentPage==8&&Filters.EfficiencyView=="Returns") {
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
            int index=0;foreach(var p in Report.Players.OrderBy(p=>p.Rank>0?p.Rank:Int32.MaxValue).ThenBy(p=>(int)p.Color)) {
                float x=(index%2)*(cardWidth+gap),cy=y+(index/2)*(rowHeight+6*unit);
                var rect=StatisticsWidgets.Rect(BodyScroll.content,"Final result "+p.Color,new Vector2(x,-cy),new Vector2(cardWidth,rowHeight));var surface=rect.gameObject.AddComponent<StatisticsSurface>();surface.Row=true;surface.Alternate=(index/2)%2==1;surface.raycastTarget=false;
                string score=p.FinalVP+" VP";measurement.fontSize=F(21);float scoreWidth=Math.Max(100*unit*TextScale,measurement.GetPreferredValues(score,0,0).x+20*unit);
                float nameLeft=14*unit;
                if(p.Rank==1||p.Rank==2){var medal=StatisticsWidgets.Rect(rect,"Rank medal "+p.Rank,new Vector2(12*unit,-3*unit),new Vector2(38*unit,38*unit));var award=medal.gameObject.AddComponent<StatisticsMedal>();award.Rank=p.Rank;award.raycastTarget=false;var digit=StatisticsWidgets.Text(medal,font,"Medal rank",p.Rank.ToString(),new Vector2(4*unit,-1*unit),new Vector2(30*unit,28*unit),20*unit);digit.alignment=TextAlignmentOptions.Center;digit.fontStyle=FontStyles.Bold;digit.color=StatisticsTheme.Hex(0x39434C);nameLeft=58*unit;}
                var name=StatisticsWidgets.Text(rect,font,"Rank and player",(p.Rank>0?p.Rank.ToString():"Unranked")+" · "+StatisticsMetrics.PlayerName(Report,p),new Vector2(nameLeft,-3*unit),new Vector2(cardWidth-nameLeft-28*unit-scoreWidth,34*unit*TextScale),F(21));name.color=StatisticsTheme.Player(p.Color);name.fontStyle=FontStyles.Bold;surface.raycastTarget=true;Tip(rect,StatisticsMetrics.PlayerName(Report,p)+" · "+score+(p.Rank==1?" · Gold · first place":p.Rank==2?" · Silver · second place":p.Rank<=0?" · Rank unavailable":" · Rank "+p.Rank));
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
