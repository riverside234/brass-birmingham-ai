using System;
using System.Collections.Generic;
using System.Linq;
using BoardGameRules.Entities.Players;
namespace BrassPatch.Statistics
{
    public static partial class StatisticsPlotPages
    {
        static StatisticsPlot Plot(StatisticsPageView v,string id,string title,StatisticsPlotKind kind,string definition,string x="",string y="")
        {var p=new StatisticsPlot{Id=id,Title=title,Kind=kind,Definition=definition,XLabel=x,YLabel=y};if(kind==StatisticsPlotKind.Step){p.XUnit=StatisticsAxisUnit.Turn;p.YUnit=id=="income-curve"?StatisticsAxisUnit.Income:StatisticsAxisUnit.Cash;}if(kind==StatisticsPlotKind.Scatter)p.XUnit=StatisticsAxisUnit.Cash;if(id=="cash-flow")p.XUnit=StatisticsAxisUnit.Cash;if(id=="spending")p.YUnit=StatisticsAxisUnit.Cash;v.Plots.Add(p);return p;}
        static void Series(StatisticsPlot p,string id,string label,StatisticsColorRole role=StatisticsColorRole.Category,PlayerColor? player=null)
        {if(p.Series.All(s=>s.Id!=id))p.Series.Add(new StatisticsPlotSeries{Id=id,Label=label,Role=role,Player=player});}
        static StatisticsDrill Target(int page,PlayerColor? player=null,IEnumerable<string> keys=null,string industry="",string action="",string region="")
        {var t=new StatisticsDrill{Page=page,Player=player,Industry=industry,Action=action,Region=region,HasKeys=keys!=null};if(keys!=null)foreach(var key in keys)t.Keys.Add(key);return t;}
        static void Mark(StatisticsPlot p,string series,string label,double x,double y,int row=0,int column=0,string detail="",StatisticsDrill target=null,double end=0)
        {p.Marks.Add(new StatisticsPlotMark{Series=series,Label=label,X=x,Y=y,Row=row,Column=column,Detail=detail,Target=target,End=end});}
        public static StatisticsPageView Build(int page,CompletedMatchStatistics r,StatisticsFilters f)
        {
            page=StatisticsPages.NormalizePage(page);
            if(page==3&&f.NetworkView=="Geography")return StatisticsMapQueries.Page(r,f);
            var v=new StatisticsPageView();
            v.Context="Select a mark for exact values and evidence. Player / era filters affect plots; text search affects ledgers.";
            if(f.Typed)v.Context+="\nEvidence scope: "+String.Join(" · ",new[]{f.Industry,f.Region,f.Action,f.EvidenceRestricted||f.Evidence.Count>0?f.Evidence.Count+" record IDs":""}.Where(s=>s!=""))+". Reset filters to restore the complete view.";
            if(!r.CompleteHistory){v.Context=r.Availability+" Missing history is unavailable, not zero. Only the final result can be shown.";return v;}
            var assets=StatisticsMetrics.Assets(r,f).ToArray();var actions=StatisticsMetrics.Actions(r,f).ToArray();
            v.Headlines.Add(assets.Length+" tiles built");v.Headlines.Add(assets.Count(a=>a.Flipped&&(f.Era==0||f.Era==a.FlipEra))+" cohort tiles flipped");
            v.Headlines.Add(actions.Length+" actions");v.Headlines.Add(StatisticsFormat.Money(actions.Sum(a=>a.Spending))+" action spending");
            switch(page){
                case 0:Composition(v,r,f);EraComparison(v,r,f);break;
                case 1:IndustryHeat(v,r,f);Gap(v,r,StatisticsMetrics.Focus(r,f),StatisticsMetrics.Reference(r,f));break;
                case 2:Conversion(v,r,f,assets);FlipDelay(v,r,f,assets);Development(v,r,f);break;
                case 3:Routes(v,r,f);LinkSources(v,r,f);break;
                case 4:Curve(v,r,f,false);Curve(v,r,f,true);CashFlow(v,r,f);break;
                case 5:if(f.ResourceView=="Market")MarketProfile(v,r,f);else if(f.ResourceView=="Merchants")MerchantDestinations(v,r,f);else{Flows(v,r,f);Sources(v,r,f);}break;
                case 6:ActionShare(v,r,f,actions);SaleHistogram(v,r,f,actions);CardPurpose(v,r,f);break;
                case 7:Seats(v,r,f);Spending(v,r,f);PositionShare(v,r,f);break;
                case 8:EfficiencyTabs(v,r,f,assets);break;
                case 9:Events(v,r,f,actions);Lifespans(v,r,f,assets);break;
            }
            Headlines(v,page,r,f,assets,actions);
            foreach(var plot in v.Plots){
                var data=new StatisticsSection(plot.Title+" · exact data",plot.Definition,"Label","Series","Value","Other value / axis","Evidence"){Id="plot/"+plot.Id};
                foreach(var m in plot.Marks){var series=plot.Series.FirstOrDefault(s=>s.Id==m.Series);data.Rows.Add(new StatisticsRow("plot-"+plot.Id+"-"+data.Rows.Count,StatisticsPlotSemantics.Description(plot,m),m.Label,series==null?m.Series:series.Label,StatisticsPlotSemantics.Value(plot,m),StatisticsPlotSemantics.Other(plot,m),m.Detail){Player=m.Target==null?null:m.Target.Player,Drill=m.Target});}
                v.Ledgers.Add(data);
            }
            if(page!=8&&(page!=5||f.ResourceView=="Supply"))v.Ledgers.AddRange(StatisticsMetrics.AdditionalLedgers(r,f,page));
            if(page!=8&&(page!=5||f.ResourceView=="Supply"))v.Ledgers.AddRange(StatisticsDetailMetrics.Build(r,f,page));
            if(f.EvidenceRestricted||f.Evidence.Count>0)v.Ledgers.Insert(0,StatisticsMetrics.SupportingRecords(r,f));
            return v;
        }
        static void Headlines(StatisticsPageView v,int page,CompletedMatchStatistics r,StatisticsFilters f,StatisticsAsset[] assets,StatisticsAction[] actions)
        {
            if(page==3)return;
            if(page==2){var flipped=assets.Where(a=>a.Flipped&&(f.Era==0||f.Era==a.FlipEra)).ToArray();v.Headlines.Clear();v.Headlines.Add(assets.Length+" tiles in build cohort");v.Headlines.Add(flipped.Length+" flipped in scope");v.Headlines.Add(StatisticsFormat.Percent(flipped.Length,assets.Length)+" conversion");double median=StatisticsDetailMetrics.Median(flipped.Select(a=>a.FlipOwnTurn-a.OwnTurn));v.Headlines.Add((Double.IsNaN(median)?"—":median.ToString("0.#"))+" median own turns to flip");}
            if(page==4){var entries=StatisticsMetrics.Entries(r,f).ToArray();v.Headlines.Clear();v.Headlines.Add(actions.Count(a=>a.Kind=="Loan")+" loan actions");v.Headlines.Add(StatisticsFormat.Money(actions.Sum(a=>a.Borrowed))+" borrowed");v.Headlines.Add(StatisticsFormat.Money(entries.Where(e=>e.Kind=="Balance"&&e.Category=="Income payment").Sum(e=>Math.Max(0,e.Cash)))+" actual positive income paid");v.Headlines.Add(StatisticsFormat.Money(actions.Sum(a=>a.Spending))+" action spending incl. markets");}
            if(page==5&&f.ResourceView=="Supply"){var flow=StatisticsMetrics.Flow(r,f).ToArray();v.Headlines.Clear();v.Headlines.Add(flow.Sum(e=>e.Quantity)+" "+f.Resource.ToLowerInvariant()+" units · "+(f.Outgoing?"outgoing to actions":"incoming"));
                if(f.Outgoing){v.Headlines.Add(flow.Where(e=>e.Supplier==e.Player).Sum(e=>e.Quantity)+" units used by owner");v.Headlines.Add(flow.Where(e=>e.Supplier!=e.Player).Sum(e=>e.Quantity)+" units used by opponents");v.Headlines.Add(flow.Where(e=>e.Quantity>0).Select(e=>e.Player).Distinct().Count()+" consumers in scope");}
                else{v.Headlines.Add(flow.Where(e=>e.Supplier==e.Player).Sum(e=>e.Quantity)+" own-source units");v.Headlines.Add(flow.Where(e=>e.Supplier.HasValue&&e.Supplier!=e.Player).Sum(e=>e.Quantity)+" opponent-source units");v.Headlines.Add(flow.Where(e=>e.Source=="Market").Sum(e=>e.Quantity)+" market / "+flow.Where(e=>e.Source=="Merchant").Sum(e=>e.Quantity)+" merchant units");}}
            if(page==6){var sales=actions.Where(a=>a.Kind=="Sell").ToArray();v.Headlines.Clear();v.Headlines.Add(actions.Length+" actions (incl. Pass)");v.Headlines.Add(sales.Length+" Sell actions");v.Headlines.Add(StatisticsFormat.Ratio(sales.Sum(a=>a.Goods),sales.Length)+" goods tiles / Sell");v.Headlines.Add(sales.Select(a=>a.Goods).DefaultIfEmpty().Max()+" largest sale batch");}
            if(page==7){var rounds=StatisticsChartQueries.Rounds(r,f).ToArray();int doubles=0;for(int i=1;i<r.Rounds.Count;i++){var a=r.Rounds[i-1];var b=r.Rounds[i];if(a.Era==b.Era&&rounds.Contains(b)&&a.Order.Count>0&&b.Order.Count>0&&a.Order.Last()==b.Order.First()&&f.Players.Contains(b.Order.First()))doubles++;}v.Headlines.Clear();v.Headlines.Add(rounds.Length+" rounds");v.Headlines.Add(doubles+" within-era back-to-back turns");v.Headlines.Add(StatisticsFormat.Money(rounds.Sum(a=>f.Players.Sum(p=>a.Spent(p))))+" round spending");}
            if(page==8){var totals=StatisticsChartQueries.Efficiency(r,f).ToArray();int vp=totals.Sum(t=>t.VP),count=totals.Sum(t=>t.Actions),spend=totals.Sum(t=>t.Spending);v.Headlines.Clear();v.Headlines.Add(StatisticsFormat.Ratio(vp,count)+" VP / confirmed action · player/era totals");v.Headlines.Add((vp>0?"£"+StatisticsFormat.Ratio(spend,vp):"—")+" "+(f.AllCashPayments?"all payments":"action spending")+" / VP");v.Headlines.Add(StatisticsChartQueries.Carryover(r,f).Count()+" tiles carried into Rail");v.Headlines.Add(assets.Count(a=>a.Remaining&&!a.Flipped&&a.Sellable)+" final unsold goods in build cohort");}
        }
        static IEnumerable<StatisticsPlayer> Players(CompletedMatchStatistics r,StatisticsFilters f){return r.Players.Where(p=>f.Players.Contains(p.Color));}
        static void Composition(StatisticsPageView v,CompletedMatchStatistics r,StatisticsFilters f)
        {
            var plot=Plot(v,"composition","Where the VP came from",StatisticsPlotKind.StackedBars,"Actual awarded VP, grouped by scoring category and era. Penalties extend left of zero. Final result is independent of filters.","Awarded VP","Player");
            foreach(var category in new[]{"Industries","Links","Merchant","Penalty","Introductory"}.Where(c=>c=="Industries"||c=="Links"||Players(r,f).Any(p=>StatisticsMetrics.Score(r,p.Color,c,f.Era)!=0)))Series(plot,category,category);
            plot.EraPairs=!r.Introductory&&f.Era==0;
            foreach(var p in Players(r,f)){int row=plot.Rows.Count;plot.Rows.Add(StatisticsMetrics.PlayerName(r,p));foreach(int era in r.Introductory?new[]{1}:new[]{1,2})if(f.Era==0||f.Era==era)foreach(var s in plot.Series){int value=StatisticsMetrics.Score(r,p.Color,s.Id,era);Mark(plot,s.Id,p.Name,0,value,row,detail:value+" VP · "+s.Label+" · "+(era==1?"Canal":"Rail"),target:Target(1,p.Color));plot.Marks.Last().Facet=era;}}
            plot.Context.Add(plot.EraPairs?"Upper bar: Canal · lower bar: Rail. Category colors are shared between eras.":f.Era==2?"Rail era awards":"Canal era awards");
        }
        static void EraComparison(StatisticsPageView v,CompletedMatchStatistics r,StatisticsFilters f)
        {
            if(r.Introductory)return;var plot=Plot(v,"era-comparison","Canal → Rail scoring",StatisticsPlotKind.Dumbbell,"Both endpoints are net era VP. Their connecting line shows change, without implying an income trajectory.","Net VP","Player");
            Series(plot,"Canal","Canal era",StatisticsColorRole.Era);Series(plot,"Rail","Rail era",StatisticsColorRole.Era);
            foreach(var p in Players(r,f)){int row=plot.Rows.Count;plot.Rows.Add(StatisticsMetrics.PlayerName(r,p));foreach(int era in new[]{1,2})if(f.Era==0||f.Era==era){int value=r.Entries.Where(e=>e.Player==p.Color&&e.Kind=="Balance"&&e.Era==era).Sum(e=>e.VP);Mark(plot,era==1?"Canal":"Rail",p.Name,0,value,row,target:Target(1,p.Color));}}
        }
        static void Gap(StatisticsPageView v,CompletedMatchStatistics r,StatisticsPlayer focus,StatisticsPlayer reference)
        {
            if(reference==null)return;var plot=Plot(v,"gap","Score gap: "+focus.Name+" vs "+reference.Name,StatisticsPlotKind.Waterfall,"Whole match: category differences sum exactly to final VP difference. Positive contributions favor the selected player. VP ties remain zero even when tie-breaks determine rank.","VP difference","Scoring category");plot.WholeMatch=true;
            Series(plot,"positive",focus.Name+" − "+reference.Name+" · positive");Series(plot,"negative",focus.Name+" − "+reference.Name+" · negative");Series(plot,"total","Final gap");
            int opening=focus.OpeningVP-reference.OpeningVP;double running=opening;
            if(opening!=0){plot.Rows.Add("Opening VP");Mark(plot,opening<0?"negative":"positive","Opening VP",0,opening,0);}
            foreach(int era in r.Introductory?new[]{1}:new[]{1,2})foreach(string category in new[]{"Industries","Links","Merchant","Penalty","Introductory"}){
                int value=StatisticsMetrics.Score(r,focus.Color,category,era)-StatisticsMetrics.Score(r,reference.Color,category,era);if(value==0)continue;
                string label=(era==1?"Canal":"Rail")+" "+category;int row=plot.Rows.Count;plot.Rows.Add(label);Mark(plot,value<0?"negative":"positive",label,running,value,row,detail:value+" VP difference · "+focus.Name+" minus "+reference.Name,target:Target(1,focus.Color));running+=value;
            }
            plot.Rows.Add("Final gap");Mark(plot,"total","Final gap",0,focus.FinalVP-reference.FinalVP,plot.Rows.Count-1,detail:"Final VP "+focus.FinalVP+" − "+reference.FinalVP+" = "+(focus.FinalVP-reference.FinalVP));
        }
        static void IndustryHeat(StatisticsPageView v,CompletedMatchStatistics r,StatisticsFilters f)
        {
            var plot=Plot(v,"industry-vp","Industry VP by player",StatisticsPlotKind.Heatmap,"Actual industry points at the selected scoring boundary, including Canal carryover scored in Rail. Heat intensity uses one shared scale.","Player","Industry");Series(plot,"VP","Industry VP");
            plot.Columns.AddRange(Players(r,f).Select(p=>StatisticsMetrics.PlayerName(r,p)));
            foreach(var type in StatisticsPages.IndustryTypes.Where(t=>f.Industry==""||f.Industry==t)){int row=plot.Rows.Count;plot.Rows.Add(StatisticsMetrics.IndustryName(type));int col=0;foreach(var p in Players(r,f)){var assets=r.Assets.Where(a=>a.Player==p.Color&&a.Type==type&&(f.Region==""||a.RegionColor==f.Region)&&(!f.EvidenceRestricted&&f.Evidence.Count==0||f.Evidence.Contains(a.Key))).ToArray();Mark(plot,"VP",StatisticsMetrics.IndustryName(type)+" · "+p.Name,0,assets.Sum(a=>StatisticsMetrics.AssetVP(a,f.Era)),row,col++,target:Target(2,p.Color,assets.Select(a=>a.Key),type));}}
        }
        static void Conversion(StatisticsPageView v,CompletedMatchStatistics r,StatisticsFilters f,StatisticsAsset[] assets)
        {
            var plot=Plot(v,"conversion","Build → flip conversion",StatisticsPlotKind.PairedBars,"Selected players' total build cohort by industry. A flipped tile counts only when its flip occurs by the selected era; whole match includes later flips. Counts are not strategy grades.","Tiles","Industry");Series(plot,"built","Built");Series(plot,"flipped","Flipped");
            foreach(var type in StatisticsPages.IndustryTypes.Where(t=>f.Industry==""||f.Industry==t)){int row=plot.Rows.Count;plot.Rows.Add(StatisticsMetrics.IndustryName(type));var group=assets.Where(a=>a.Type==type).ToArray();Mark(plot,"built",StatisticsMetrics.IndustryName(type),0,group.Length,row,target:Target(2,keys:group.Select(a=>a.Key),industry:type));var flipped=group.Where(a=>a.Flipped&&(f.Era==0||f.Era==a.FlipEra)).ToArray();Mark(plot,"flipped",StatisticsMetrics.IndustryName(type),0,flipped.Length,row,detail:StatisticsFormat.Percent(flipped.Length,group.Length)+" of built cohort",target:Target(2,keys:group.Select(a=>a.Key),industry:type));}
        }
        static void FlipDelay(StatisticsPageView v,CompletedMatchStatistics r,StatisticsFilters f,StatisticsAsset[] assets)
        {
            var plot=Plot(v,"flip-delay","How long industries waited to flip",StatisticsPlotKind.DelayDots,"One dot per flipped tile, grouped by industry. Delay uses the owner's turn count; exact data also shows global-turn delay. Unflipped tiles are excluded from the distribution.","Own turns to flip","Industry");
            foreach(var player in Players(r,f))Series(plot,player.Color.ToString(),StatisticsMetrics.PlayerName(r,player),StatisticsColorRole.Player,player.Color);
            plot.Rows.AddRange(StatisticsPages.IndustryTypes.Where(t=>f.Industry==""||f.Industry==t).Select(StatisticsMetrics.IndustryName));
            foreach(var a in assets.Where(a=>a.Flipped&&(f.Era==0||f.Era==a.FlipEra)))Mark(plot,a.Player.ToString(),a.City+" · L"+a.Level,0,a.FlipOwnTurn-a.OwnTurn,plot.Rows.IndexOf(StatisticsMetrics.IndustryName(a.Type)),detail:(a.FlipTurn-a.Turn)+" global turns · "+(a.FlipTrigger??"Goods sale"),target:Target(2,a.Player,new[]{a.Key}));
        }
        static void Routes(StatisticsPageView v,CompletedMatchStatistics r,StatisticsFilters f)
        {
            var links=StatisticsChartQueries.Links(r,f).OrderByDescending(l=>l.VP).ThenBy(l=>l.Key).ToArray();
            var plot=Plot(v,"routes","Route scoring",StatisticsPlotKind.Lollipop,"Routes sorted by actual VP. Top 8 plus Other preserves the total; Show all exposes every route. A zero-VP link can still provide access.","Link VP","Route");
            foreach(var p in Players(r,f))Series(plot,p.Color.ToString(),StatisticsMetrics.PlayerName(r,p),StatisticsColorRole.Player,p.Color);
            int visible=f.ShowAll?links.Length:Math.Min(8,links.Length);
            foreach(var l in links.Take(visible)){int row=plot.Rows.Count;plot.Rows.Add(l.Route);Mark(plot,l.Player.ToString(),l.Route,0,l.VP,row,detail:r.Name(l.Player)+" · own "+l.OwnVP+", opponents "+l.OthersVP+", merchant "+l.MerchantVP+" VP\n"+l.Contributions,target:Target(3,l.Player,new[]{l.Key}));}
            if(links.Length>visible){Series(plot,"other","Other routes (aggregate)");Mark(plot,"other","Other · "+(links.Length-visible)+" routes",0,links.Skip(visible).Sum(l=>l.VP),-1,detail:"Exact sum of remaining routes; shown separately from individual-route scale",target:Target(3,keys:links.Skip(visible).Select(l=>l.Key)));plot.Context.Add("Other "+(links.Length-visible)+" routes contributed "+links.Skip(visible).Sum(l=>l.VP)+" VP in total. The route axis compares individual routes; this aggregate is in exact data.");}
            v.Headlines.Clear();v.Headlines.Add(links.Length+" links");v.Headlines.Add(links.Sum(l=>l.VP)+" link VP");v.Headlines.Add(StatisticsFormat.Ratio(links.Sum(l=>l.VP),links.Length)+" VP / link");var checks=r.Connections.Where(c=>f.Includes(c.Player,c.Era)).ToArray();v.Headlines.Add(checks.Count(c=>c.OpponentRequired)+" / "+checks.Length+" endpoint checks needed opponents' links");
        }
        static void Curve(StatisticsPageView v,CompletedMatchStatistics r,StatisticsFilters f,bool income)
        {
            var plot=Plot(v,income?"income-curve":"cash-curve",income?"Income level through the match":"Cash through the match",StatisticsPlotKind.Step,"Step changes from recorded balance events. X is global turn count; multiple changes within a turn share its x value. Hover or select a mark for the event.","Global turn",income?"Income level":"Cash (£)");
            foreach(var p in Players(r,f)){Series(plot,p.Color.ToString(),StatisticsMetrics.PlayerName(r,p),StatisticsColorRole.Player,p.Color);foreach(var s in r.Samples.Where(s=>s.Player==p.Color&&(f.Era==0||f.Era==s.Era)))Mark(plot,p.Color.ToString(),s.Position,s.Turn,income?s.Income:s.Cash,detail:s.Annotation+" · revision "+s.Revision,target:Target(4,p.Color));}
        }
        static void CashFlow(StatisticsPageView v,CompletedMatchStatistics r,StatisticsFilters f)
        {
            var focus=StatisticsMetrics.Focus(r,f);var entries=r.Entries.Where(e=>e.Player==focus.Color&&e.Kind=="Balance").ToArray();
            var plot=Plot(v,"cash-flow","Cash reconciliation · "+focus.Name,StatisticsPlotKind.Waterfall,"Whole match: opening cash plus every signed cash category equals final cash. Market resource payments are already inside action spending and are not added twice.","Cash (£)","Cash category");plot.WholeMatch=true;
            Series(plot,"positive","Cash in");Series(plot,"negative","Cash out");Series(plot,"total","Closing cash");double running=focus.OpeningCash;
            plot.Rows.Add("Opening cash");Mark(plot,"positive","Opening cash",0,running,0);
            foreach(var g in entries.GroupBy(e=>e.Category).Where(g=>g.Sum(e=>e.Cash)!=0)){int row=plot.Rows.Count;plot.Rows.Add(g.Key);int value=g.Sum(e=>e.Cash);Mark(plot,value<0?"negative":"positive",g.Key,running,value,row,detail:g.Count()+" balance events",target:Target(4,focus.Color));running+=value;}
            plot.Rows.Add("Final cash");Mark(plot,"total","Final cash",0,focus.FinalCash,plot.Rows.Count-1);
        }
        static void Flows(StatisticsPageView v,CompletedMatchStatistics r,StatisticsFilters f)
        {
            var flow=StatisticsMetrics.Flow(r,f).ToArray();var plot=Plot(v,"resource-flow",f.Resource+" · supplier → consumer",StatisticsPlotKind.Heatmap,(f.Outgoing?"Outgoing: selected suppliers; all consumers remain context.":"Incoming: selected consumers; all suppliers remain context.")+" Each unit is counted once. Industry/region filters refer to source tiles; action filters refer to the consuming action. Merchant and market supplies have no player owner.","Consumer","Supplier");Series(plot,f.Resource,f.Resource+" units",StatisticsColorRole.Resource);
            plot.Rows.AddRange((f.Outgoing?Players(r,f):r.Players).Select(p=>StatisticsMetrics.PlayerName(r,p)));if(!f.Outgoing){plot.Rows.Add("Merchant");plot.Rows.Add("Market");}
            var consumers=(f.Outgoing?r.Players:Players(r,f)).ToArray();plot.Columns.AddRange(consumers.Select(p=>StatisticsMetrics.PlayerName(r,p)));
            for(int row=0;row<plot.Rows.Count;row++)for(int col=0;col<consumers.Length;col++){
                var suppliers=(f.Outgoing?Players(r,f):r.Players).ToArray();var supplier=row<suppliers.Length?suppliers[row]:null;
                var records=flow.Where(e=>e.Player==consumers[col].Color&&(supplier!=null?e.Supplier==supplier.Color:e.Source==plot.Rows[row])).ToArray();
                var target=Target(5,f.Outgoing&&supplier!=null?supplier.Color:consumers[col].Color,records.Select(e=>"entry-"+e.Revision+"-"+e.Player));target.Resource=f.Resource;target.Outgoing=f.Outgoing;
                Mark(plot,f.Resource,plot.Rows[row]+" → "+StatisticsMetrics.PlayerName(r,consumers[col]),0,records.Sum(e=>e.Quantity),row,col,detail:records.Length+" recorded supply events",target:target);
            }
        }
        static void Sources(StatisticsPageView v,CompletedMatchStatistics r,StatisticsFilters f)
        {
            var flow=StatisticsMetrics.Flow(r,f).ToArray();var categories=f.Outgoing?new[]{"Own","Opponents"}:new[]{"Own","Opponents","Merchant","Market"};
            var plot=Plot(v,"sources",f.Resource+(f.Outgoing?" · supplied to actions":" supply mix"),StatisticsPlotKind.StackedBars,f.Outgoing?"Outgoing units supplied by selected players, split by own use and opponents' use. Includes all counterpart consumers and uses the same typed scope as the matrix. Market sales are separate transactions.":"Incoming units used by selected players, grouped by ownership/source. Uses the same typed scope and exact records as the matrix.","Units",f.Outgoing?"Supplier":"Consumer");
            foreach(var key in categories)Series(plot,key,f.Outgoing?(key=="Own"?"Own use":"Opponents' use"):key);
            foreach(var p in Players(r,f)){int row=plot.Rows.Count;plot.Rows.Add(StatisticsMetrics.PlayerName(r,p));foreach(var key in categories){
                var records=flow.Where(e=>f.Outgoing?e.Supplier==p.Color&&(key=="Own"?e.Player==p.Color:e.Player!=p.Color):e.Player==p.Color&&(key=="Own"?e.Supplier==p.Color:key=="Opponents"?e.Supplier.HasValue&&e.Supplier!=p.Color:e.Source==key)).ToArray();var target=Target(5,p.Color,records.Select(e=>"entry-"+e.Revision+"-"+e.Player));target.Resource=f.Resource;target.Outgoing=f.Outgoing;Mark(plot,key,StatisticsMetrics.PlayerName(r,p),0,records.Sum(e=>e.Quantity),row,target:target);
            }}
        }
        static void ActionShare(StatisticsPageView v,CompletedMatchStatistics r,StatisticsFilters f,StatisticsAction[] actions)
        {
            var plot=Plot(v,"actions","Action mix",StatisticsPlotKind.StackedBars,"Counts and shares of recorded actions per player. Includes Pass and any explicitly unclassified event. Sale VP is not credited again to the action that sold it.",f.ActionShare?"Actions (%)":"Confirmed actions","Player");plot.Percent=f.ActionShare;
            foreach(var kind in StatisticsPages.ActionTypes.Where(k=>actions.Any(a=>a.Kind==k)))Series(plot,kind,kind);
            foreach(var p in Players(r,f)){int row=plot.Rows.Count;plot.Rows.Add(StatisticsMetrics.PlayerName(r,p));var own=actions.Where(a=>a.Player==p.Color).ToArray();foreach(var s in plot.Series){var records=own.Where(a=>a.Kind==s.Id).ToArray();Mark(plot,s.Id,p.Name,0,records.Length,row,detail:StatisticsFormat.Percent(records.Length,own.Length)+" · "+records.Length+" / "+own.Length+" actions",target:Target(6,p.Color,records.Select(a=>a.Key),action:s.Id));}}
        }
        static void SaleHistogram(StatisticsPageView v,CompletedMatchStatistics r,StatisticsFilters f,StatisticsAction[] actions)
        {
            var plot=Plot(v,"sale-batches","Goods sold per Sell action",StatisticsPlotKind.PairedBars,"Action batching: 3+ groups all sales of three or more tiles; exact values remain in the action ledger. No legal alternative batch is inferred.","Sell actions","Tiles per sale");plot.Rows.AddRange(new[]{"1 tile","2 tiles","3+ tiles"});
            foreach(var p in Players(r,f)){Series(plot,p.Color.ToString(),StatisticsMetrics.PlayerName(r,p),StatisticsColorRole.Player,p.Color);for(int i=0;i<3;i++){var records=actions.Where(a=>a.Player==p.Color&&a.Kind=="Sell"&&(i==2?a.Goods>=3:a.Goods==i+1)).ToArray();Mark(plot,p.Color.ToString(),plot.Rows[i],0,records.Length,i,target:Target(6,p.Color,records.Select(a=>a.Key),action:"Sell"));}}
        }
        static void Seats(StatisticsPageView v,CompletedMatchStatistics r,StatisticsFilters f)
        {
            var plot=Plot(v,"seats","Turn-order positions by round",StatisticsPlotKind.Heatmap,"Each cell is the player's actual position in the round. Position 1 acts first. Round order is determined by the preceding round's spending, with ties resolved by existing order.","Round","Player");Series(plot,"seat","Turn position (1 = first)");plot.DiscreteHeat=true;
            var rounds=StatisticsChartQueries.Rounds(r,f).ToArray();plot.Columns.AddRange(rounds.Select(t=>(t.Era==1?"C":"R")+t.Round));
            foreach(var p in Players(r,f)){int row=plot.Rows.Count;plot.Rows.Add(StatisticsMetrics.PlayerName(r,p));for(int col=0;col<rounds.Length;col++)Mark(plot,"seat",p.Name+" · "+plot.Columns[col],0,rounds[col].Order.Contains(p.Color)?rounds[col].Order.IndexOf(p.Color)+1:Double.NaN,row,col,target:Target(7,p.Color,new[]{StatisticsChartQueries.RoundKey(rounds[col])}));}
        }
        static void Spending(StatisticsPageView v,CompletedMatchStatistics r,StatisticsFilters f)
        {
            var plot=Plot(v,"spending","Spending tempo",StatisticsPlotKind.Columns,"Recorded action spending per round. Market costs are already included. A spike changes order in the following round; variation is descriptive, not a strategy rating.","Round","Spending (£)");
            var rounds=StatisticsChartQueries.Rounds(r,f).ToArray();plot.Columns.AddRange(rounds.Select(t=>(t.Era==1?"C":"R")+t.Round));
            foreach(var p in Players(r,f)){Series(plot,p.Color.ToString(),StatisticsMetrics.PlayerName(r,p),StatisticsColorRole.Player,p.Color);for(int i=0;i<rounds.Length;i++)Mark(plot,p.Color.ToString(),p.Name+" · "+plot.Columns[i],i,rounds[i].Spent(p.Color),target:Target(7,p.Color,new[]{StatisticsChartQueries.RoundKey(rounds[i])}));}
        }
        static void Efficiency(StatisticsPageView v,CompletedMatchStatistics r,StatisticsFilters f,StatisticsAsset[] assets)
        {
            var plot=Plot(v,"tile-efficiency","Construction cost vs realized industry VP",StatisticsPlotKind.Scatter,"Each tile: base construction + bought resource cost versus actual industry VP. Income and network value are excluded, so suppliers can be useful at low VP. Exact overlap is offset visually, not numerically.","Construction + market resources (£)","Actual industry VP");
            foreach(var p in Players(r,f))Series(plot,p.Color.ToString(),StatisticsMetrics.PlayerName(r,p),StatisticsColorRole.Player,p.Color);
            foreach(var a in assets){Mark(plot,a.Player.ToString(),StatisticsMetrics.IndustryName(a.Type)+" L"+a.Level+" · "+a.City,a.BaseCost+a.MarketCost,StatisticsMetrics.AssetVP(a,0),detail:TileFacts(a),target:EfficiencyTarget("Conversion",a.Player,new[]{a.Key}));plot.Marks.Last().Symbol=a.Type;}
            plot.Context.Add(Cohort(f));
            plot.Context.Add("Player color; industry symbol: cotton square, manufactory cross, pottery triangle, brewery circle, coal diamond, iron plus. Exact data preserves overlapping tiles.");
        }
        static void Outcomes(StatisticsPageView v,CompletedMatchStatistics r,StatisticsFilters f,StatisticsAsset[] assets)
        {
            var plot=Plot(v,"outcomes","Tile outcomes",StatisticsPlotKind.StackedBars,"Mutually exclusive final outcomes for the selected build cohort. An unflipped supplier is distinct from unsold goods; surviving Canal tiles retain their actual later points.","Tiles","Industry");
            var states=new[]{"Scored both eras","Other flipped","Unsold goods","Unflipped supplier","Removed unflipped"};foreach(var s in states)Series(plot,s,s);
            Func<StatisticsAsset,string> state=a=>a.Carried&&a.CanalVP>0&&a.RailVP>0?states[0]:a.Flipped?states[1]:!a.Remaining?states[4]:a.Sellable?states[2]:states[3];
            foreach(var type in StatisticsPages.IndustryTypes.Where(t=>f.Industry==""||f.Industry==t)){int row=plot.Rows.Count;plot.Rows.Add(StatisticsMetrics.IndustryName(type));foreach(var s in states){var records=assets.Where(a=>a.Type==type&&state(a)==s).ToArray();Mark(plot,s,StatisticsMetrics.IndustryName(type),0,records.Length,row,target:Target(8,keys:records.Select(a=>a.Key),industry:type));}}
        }
        static void Events(StatisticsPageView v,CompletedMatchStatistics r,StatisticsFilters f,StatisticsAction[] actions)
        {
            var plot=Plot(v,"events","Actions across the match",StatisticsPlotKind.EventLanes,"One symbol per action on the global-turn axis. Two actions in the same turn use adjacent marks. Color identifies the action; the lane identifies the player.","Global turn","Player");
            foreach(var kind in StatisticsPages.ActionTypes.Where(k=>actions.Any(a=>a.Kind==k)))Series(plot,kind,kind);plot.Rows.AddRange(Players(r,f).Select(p=>StatisticsMetrics.PlayerName(r,p)));
            foreach(var a in actions)Mark(plot,a.Kind,r.Name(a.Player)+" · "+a.Kind,a.Turn,a.OwnTurn,Array.FindIndex(Players(r,f).ToArray(),p=>p.Color==a.Player),detail:a.Position+" · "+a.Details,target:Target(6,a.Player,new[]{a.Key},action:a.Kind));
        }
        static void Lifespans(StatisticsPageView v,CompletedMatchStatistics r,StatisticsFilters f,StatisticsAsset[] assets)
        {
            var plot=Plot(v,"lifespans","Industry lifespans",StatisticsPlotKind.Lifespans,"Build-to-flip intervals by industry. Open-ended lines indicate unflipped tiles, ending at removal or match end. Intervals are grouped by industry; Show all exposes the remaining records. No averaged Other interval is drawn.","Global turn","Tile");Series(plot,"flipped","Built → flipped");Series(plot,"unflipped","Unflipped interval");
            var sorted=assets.OrderBy(a=>Array.IndexOf(StatisticsPages.IndustryTypes,a.Type)).ThenBy(a=>a.Turn).ThenBy(a=>a.Key).ToArray();var selected=f.ShowAll?sorted:sorted.Take(8).ToArray();int end=r.Actions.Select(a=>a.Turn).DefaultIfEmpty().Max();
            foreach(var a in selected){int row=plot.Rows.Count;plot.Rows.Add(StatisticsMetrics.IndustryName(a.Type)+" L"+a.Level+" · "+a.City);var last=r.Entries.LastOrDefault(e=>e.Revision<=a.RemovedRevision&&a.RemovedRevision>0);Mark(plot,a.Flipped?"flipped":"unflipped",plot.Rows[row],a.Turn,a.Flipped?a.FlipTurn-a.Turn:Math.Max(0,(last==null?end:last.Turn)-a.Turn),row,detail:r.Name(a.Player)+" · "+(a.FlipTrigger??a.Removal??"Still on board"),target:Target(2,a.Player,new[]{a.Key}),end:a.Flipped?a.FlipTurn:last==null?end:last.Turn);plot.Marks.Last().Symbol=a.Flipped?"flipped":a.RemovedRevision>0?"removed":"open";plot.Marks.Last().OtherLabel=a.Flipped?"Endpoint: flipped":a.RemovedRevision>0?"Endpoint: removed · "+a.Removal:"Endpoint: still open at match end";}
            plot.Context.Add("Endpoint symbols: circle = flipped; triangle = removed unflipped; square = still open at match end.");
            if(selected.Length<sorted.Length)plot.Context.Add("Other "+(sorted.Length-selected.Length)+" tiles: "+sorted.Skip(selected.Length).Count(a=>a.Flipped)+" flipped, "+sorted.Skip(selected.Length).Count(a=>!a.Flipped)+" unflipped. Show all or open the industry ledger for every record.");
        }
    }
}
