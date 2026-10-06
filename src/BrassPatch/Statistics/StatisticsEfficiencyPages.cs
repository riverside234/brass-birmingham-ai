using System;
using System.Linq;
using BoardGameRules.Entities.Players;
namespace BrassPatch.Statistics
{
    public static partial class StatisticsPlotPages
    {
        static string Cohort(StatisticsFilters f)
        { return (f.Era==0?"Built in either era":f.Era==1?"Built in Canal":"Built in Rail")+" · lifetime outcomes through game end. Action filters do not allocate tile returns."; }
        static StatisticsDrill EfficiencyTarget(string view,PlayerColor? player,System.Collections.Generic.IEnumerable<string> keys,int? era=null)
        {var t=Target(8,player,keys);t.EfficiencyView=view;t.Era=era;return t;}
        static string TileFacts(StatisticsAsset a)
        {return a.City+" · "+StatisticsMetrics.IndustryName(a.Type)+" L"+a.Level+" · built "+a.Position+" / own turn "+a.OwnTurn+"; "+(a.Flipped?"flipped "+(a.FlipEra==1?"Canal":"Rail")+" / own turn "+a.FlipOwnTurn+" / global turn "+a.FlipTurn:"unflipped")+"; Canal "+a.CanalVP+" / Rail "+a.RailVP+" / intro "+a.IntroVP+" VP; base "+StatisticsFormat.Money(a.BaseCost)+" + market "+StatisticsFormat.Money(a.MarketCost);}
        static void EfficiencyTabs(StatisticsPageView v,CompletedMatchStatistics r,StatisticsFilters f,StatisticsAsset[] assets)
        {
            v.Context="Player and era filters are retained between views. Each chart states whether era means scoring era or build cohort. Text search affects exact-data tables.";
            if(f.Typed)v.Context+=" Detail scope: "+String.Join(" · ",new[]{f.Industry,f.Region,f.Action,f.EvidenceRestricted?f.Evidence.Count+" evidence IDs":""}.Where(x=>x!=""))+".";
            if(f.EfficiencyView=="Conversion") {
                Efficiency(v,r,f,assets);Outcomes(v,r,f,assets);CompletionHeat(v,r,f,assets);
            } else if(f.EfficiencyView=="Transition & networks") {
                UnflippedInvestment(v,r,f,assets);CarryoverScore(v,r,f);LinkReturn(v,r,f);
                var cleanup=new StatisticsSection("Level-1 Canal cleanup","Canal links score before cleanup. Removed icons do not subtract points already earned.","Player","Tile","Flipped","Canal VP","Stock removed");
                foreach(var a in assets.Where(a=>a.Removal=="Canal cleanup"))cleanup.Rows.Add(new StatisticsRow(a.Key,TileFacts(a),r.Name(a.Player),a.City+" · "+StatisticsMetrics.IndustryName(a.Type),a.Flipped?"Yes":"No",a.CanalVP.ToString(),a.Stock.ToString()){Player=a.Player,Drill=EfficiencyTarget(f.EfficiencyView,a.Player,new[]{a.Key})});
                v.Ledgers.Add(cleanup);
            } else {PlayerEfficiency(v,r,f);Realization(v,r,f);ConstructionReturn(v,r,f,assets);}
        }
        static void Realization(StatisticsPageView v,CompletedMatchStatistics r,StatisticsFilters f)
        {
            var p=Plot(v,"industry-realization","Industry VP · awarded / recorded opportunity",StatisticsPlotKind.Bullet,"Filled bars show ordinary industry VP awarded; pale tracks show printed VP recorded at actual scoring snapshots. Earlier-built carried tiles can score again. This is not the optimal achievable score and excludes tiles removed before scoring.","Industry VP","Player");
            Series(p,"awarded","Ordinary industry VP");Series(p,"opportunity","Recorded printed opportunity");
            p.Context.Add("Scoring era · player totals. Industry, region, action and evidence filters do not narrow these totals. Introductory bonus is separate.");
            foreach(var t in StatisticsChartQueries.Efficiency(r,f)) {
                int row=p.Rows.Count;p.Rows.Add(StatisticsMetrics.PlayerName(r,t.Player));
                var tiles=r.Assets.Where(a=>a.Player==t.Player.Color&&((f.Era!=2&&(a.CanalPrinted!=0||a.CanalVP!=0||a.IntroVP!=0))||(f.Era!=1&&(a.RailPrinted!=0||a.RailVP!=0)))).ToArray();
                var keys=tiles.Select(a=>a.Key).Concat(r.Entries.Where(e=>e.Player==t.Player.Color&&e.Kind=="Balance"&&e.VP!=0&&(f.Era==0||f.Era==e.Era)).Select(StatisticsChartQueries.EntryKey));
                // Evidence lookup is identity-based, so Rail scoring can include a Canal-built tile.
                Mark(p,"awarded",StatisticsMetrics.PlayerName(r,t.Player),0,t.IndustryVP,row,detail:"Ordinary industry VP "+t.IndustryVP+"; printed opportunity "+t.Printed+"; separate introductory bonus +"+t.IntroVP+" VP.",target:EfficiencyTarget("Returns",t.Player.Color,keys),end:t.Printed);
                p.Marks.Last().ValueLabel=t.IndustryVP+" / "+t.Printed+" VP · "+StatisticsFormat.Percent(t.IndustryVP,t.Printed);
                p.Marks.Last().OtherLabel="Intro bonus +"+t.IntroVP+" VP";
            }
        }
        static void ConstructionReturn(StatisticsPageView v,CompletedMatchStatistics r,StatisticsFilters f,StatisticsAsset[] assets)
        {
            var facets=f.ByPlayer?Players(r,f).Select(p=>(PlayerColor?)p.Color).ToArray():new PlayerColor?[]{null};
            double max=0;
            foreach(var owner in facets)foreach(var type in StatisticsPages.IndustryTypes){var a=assets.Where(x=>(!owner.HasValue||x.Player==owner.Value)&&x.Type==type).ToArray();int cost=a.Sum(x=>x.BaseCost+x.MarketCost);if(cost>0)max=Math.Max(max,(double)a.Sum(x=>StatisticsMetrics.AssetVP(x,0))/cost);}
            foreach(var owner in facets) {
                var p=Plot(v,"construction-return"+(owner.HasValue?"-"+owner:""),"Construction return by industry"+(owner.HasValue?" · "+StatisticsMetrics.PlayerName(r,r.Players.Single(x=>x.Color==owner.Value)):" · selected players"),StatisticsPlotKind.Lollipop,"Lifetime awarded industry VP divided by total base plus purchased-resource construction cash for the same tiles. Ratio of sums; includes zero-VP tiles and introductory bonuses. Carried construction is counted once. Excludes income, network utility and resource-sharing benefits.","Industry VP / £ construction","Industry");p.XUnit=StatisticsAxisUnit.Ratio;p.SharedMax=max;p.Context.Add(Cohort(f));
                foreach(var type in StatisticsPages.IndustryTypes) {
                    var a=assets.Where(x=>(!owner.HasValue||x.Player==owner.Value)&&x.Type==type).ToArray();int cost=a.Sum(x=>x.BaseCost+x.MarketCost),vp=a.Sum(x=>StatisticsMetrics.AssetVP(x,0));int row=p.Rows.Count;p.Rows.Add(StatisticsMetrics.IndustryName(type));Series(p,type,StatisticsMetrics.IndustryName(type),StatisticsColorRole.Industry);
                    Mark(p,type,p.Rows[row],0,cost>0?(double)vp/cost:Double.NaN,row,detail:a.Length+" tiles · "+vp+" VP (intro +"+a.Sum(x=>x.IntroVP)+") · base "+StatisticsFormat.Money(a.Sum(x=>x.BaseCost))+" + market "+StatisticsFormat.Money(a.Sum(x=>x.MarketCost)),target:EfficiencyTarget("Returns",owner,a.Select(x=>x.Key)));
                    p.Marks.Last().Missing=cost<=0;p.Marks.Last().OtherLabel=vp+" VP / "+StatisticsFormat.Money(cost)+" · "+a.Length+" tiles";
                }
            }
        }
        static void CompletionHeat(StatisticsPageView v,CompletedMatchStatistics r,StatisticsFilters f,StatisticsAsset[] assets)
        {
            var p=Plot(v,"completion-speed","Completion speed · median own turns",StatisticsPlotKind.Heatmap,"Median own-turn interval from construction to flipping. Later-era flips remain included. Same-turn flips are zero; unfinished tiles are excluded from the median. Darker means longer, without a tactical grade.","Industry · n = flipped tiles","Player");p.HeatTotals=false;p.HeatUnit="own turns";Series(p,"delay","Median own turns · darker = longer");p.Columns.AddRange(StatisticsPages.IndustryTypes.Select(StatisticsMetrics.IndustryName));p.Context.Add(Cohort(f));
            var ledger=new StatisticsSection("Completion intervals by tile",Cohort(f),"Player","Tile","Build era / own turn","Flip era / own turn","Own turns","Global turns","Outcome");
            foreach(var player in Players(r,f)) {
                int row=p.Rows.Count;p.Rows.Add(StatisticsMetrics.PlayerName(r,player));
                for(int col=0;col<StatisticsPages.IndustryTypes.Length;col++) {
                    var a=assets.Where(x=>x.Player==player.Color&&x.Type==StatisticsPages.IndustryTypes[col]).ToArray();var flipped=a.Where(x=>x.Flipped).ToArray();double median=StatisticsDetailMetrics.Median(flipped.Select(x=>x.FlipOwnTurn-x.OwnTurn));
                    Mark(p,"delay",p.Rows[row]+" · "+p.Columns[col],0,median,row,col,detail:flipped.Length+" flipped; "+a.Count(x=>!x.Flipped&&x.Remaining)+" remaining unflipped; "+a.Count(x=>!x.Flipped&&!x.Remaining)+" removed unflipped.\n"+String.Join("\n",a.Select(TileFacts)),target:EfficiencyTarget("Conversion",player.Color,a.Select(x=>x.Key)));
                    p.Marks.Last().Missing=flipped.Length==0;p.Marks.Last().OtherLabel="n = "+flipped.Length;p.Marks.Last().ValueLabel=Double.IsNaN(median)?"— · no observed flips":median.ToString("0.#")+" own turns";
                }
            }
            foreach(var a in assets)ledger.Rows.Add(new StatisticsRow(a.Key,TileFacts(a),r.Name(a.Player),a.City+" · "+StatisticsMetrics.IndustryName(a.Type),(a.Era==1?"Canal":"Rail")+" / "+a.OwnTurn,a.Flipped?(a.FlipEra==1?"Canal":"Rail")+" / "+a.FlipOwnTurn:"—",a.Flipped?(a.FlipOwnTurn-a.OwnTurn).ToString():"—",a.Flipped?(a.FlipTurn-a.Turn).ToString():"—",a.Remaining?(a.Flipped?"Flipped":"Unflipped"):a.Removal){Player=a.Player,Drill=EfficiencyTarget("Conversion",a.Player,new[]{a.Key})});v.Ledgers.Add(ledger);
        }
        static void UnflippedInvestment(StatisticsPageView v,CompletedMatchStatistics r,StatisticsFilters f,StatisticsAsset[] assets)
        {
            var all=assets.Where(a=>a.Remaining&&!a.Flipped).ToArray();var facets=f.ByPlayer?Players(r,f).Select(p=>(PlayerColor?)p.Color).ToArray():new PlayerColor?[]{null};
            double max=facets.SelectMany(owner=>StatisticsPages.IndustryTypes.Select(type=>all.Where(a=>(!owner.HasValue||a.Player==owner.Value)&&a.Type==type).Sum(a=>a.BaseCost+a.MarketCost))).DefaultIfEmpty().Max();
            foreach(var owner in facets) {
                var p=Plot(v,"unflipped-investment"+(owner.HasValue?"-"+owner:""),"Final unflipped investment"+(owner.HasValue?" · "+StatisticsMetrics.PlayerName(r,r.Players.Single(x=>x.Color==owner.Value)):" · selected players"),StatisticsPlotKind.StackedBars,"Base construction cash plus bought resources in tiles still on the final board, unflipped. This is investment, not proven waste. Receipts are shown separately; free resource units have no invented cash price.","Construction cash (£)","Industry");p.XUnit=StatisticsAxisUnit.Cash;p.SharedMax=max;Series(p,"base-cash","Base construction cash");Series(p,"resource-cash","Bought resources");p.Context.Add(Cohort(f));
                foreach(var type in StatisticsPages.IndustryTypes) {
                    var a=all.Where(x=>(!owner.HasValue||x.Player==owner.Value)&&x.Type==type).ToArray();int row=p.Rows.Count;p.Rows.Add(StatisticsMetrics.IndustryName(type)+(new[]{"CottonMill","Manufacturer","Pottery"}.Contains(type)?" · goods":" · supplier"));
                    foreach(string series in new[]{"base-cash","resource-cash"})Mark(p,series,p.Rows[row],0,a.Sum(x=>series=="base-cash"?x.BaseCost:x.MarketCost),row,detail:a.Length+" tiles · printed "+a.Sum(x=>x.PrintedVP)+" VP · supplied "+a.Sum(x=>x.Supplied)+" units · remaining stock "+a.Sum(x=>x.Stock)+" · market receipts "+StatisticsFormat.Money(a.Sum(x=>x.MarketReceipts)),target:EfficiencyTarget("Transition & networks",owner,a.Select(x=>x.Key)));
                }
            }
        }
        static void CarryoverScore(StatisticsPageView v,CompletedMatchStatistics r,StatisticsFilters f)
        {
            var p=Plot(v,"carryover-scoring","Canal carryover · industry scoring",StatisticsPlotKind.StackedColumns,"Actual tiles surviving into Rail, including zero-VP tiles. Era selects the scoring segment of this fixed Rail-entry cohort. Industry and region filters narrow the cohort. Introductory games have no Rail carryover.","Player · Rail-entry cohort","Industry VP");
            if(r.Introductory){p.Context.Add("Unavailable in introductory games: there is no Rail-entry cohort.");return;}
            var all=StatisticsChartQueries.Carryover(r,f).ToArray();foreach(int era in new[]{1,2})if(f.Era==0||f.Era==era)Series(p,era==1?"Canal":"Rail",era==1?"Canal VP":"Rail VP",StatisticsColorRole.Era);
            foreach(var player in Players(r,f)){int row=p.Rows.Count;p.Rows.Add(player.Name);p.Columns.Add(StatisticsMetrics.PlayerName(r,player));var a=all.Where(x=>x.Player==player.Color).ToArray();foreach(var s in p.Series)Mark(p,s.Id,StatisticsMetrics.PlayerName(r,player),row,a.Sum(x=>s.Id=="Canal"?x.CanalVP:x.RailVP),row,detail:a.Length+" carried tiles; "+a.Count(x=>x.CanalVP+x.RailVP==0)+" with no ordinary VP.\n"+String.Join("\n",a.Select(TileFacts)),target:EfficiencyTarget("Transition & networks",player.Color,a.Select(x=>x.Key)));}
            p.Context.Add("Scoring-era segments · fixed Rail-entry cohort. Use the Industry filter to inspect its composition. Canal links scored before level-1 cleanup.");
        }
        static void LinkReturn(StatisticsPageView v,CompletedMatchStatistics r,StatisticsFilters f)
        {
            var p=Plot(v,"link-return","Network return · VP per scored link",StatisticsPlotKind.PairedLollipop,"Awarded link VP divided by distinct scored links, including zero-VP links. Two links built by one Network action count as two links. Region selects links with a matching contribution endpoint; a link is counted once. Industry and action filters cannot allocate these ratios.","VP / scored link","Player");p.XUnit=StatisticsAxisUnit.Ratio;
            var all=StatisticsChartQueries.Links(r,f).Where(l=>l.Scored).ToArray();foreach(int era in r.Introductory?new[]{1}:new[]{1,2})if(f.Era==0||f.Era==era)Series(p,era==1?"Canal":"Rail",era==1?"Canal · circle":"Rail · diamond",StatisticsColorRole.Era);
            foreach(var player in Players(r,f)){int row=p.Rows.Count;p.Rows.Add(StatisticsMetrics.PlayerName(r,player));foreach(var s in p.Series){int era=s.Id=="Canal"?1:2;var links=all.Where(l=>l.Player==player.Color&&l.Era==era).GroupBy(l=>l.Key).Select(g=>g.First()).ToArray();Mark(p,s.Id,StatisticsMetrics.PlayerName(r,player),0,links.Length>0?(double)links.Sum(l=>l.VP)/links.Length:Double.NaN,row,detail:links.Sum(l=>l.VP)+" VP / "+links.Length+" scored links; "+links.Count(l=>l.VP==0)+" zero-VP links.\n"+String.Join("\n",links.Select(l=>l.Route+" · "+l.VP+" VP")),target:EfficiencyTarget("Transition & networks",player.Color,links.Select(l=>l.Key),era));p.Marks.Last().Missing=links.Length==0;p.Marks.Last().OtherLabel=links.Sum(l=>l.VP)+" VP / "+links.Length+" links";}}
            p.Context.Add("Scoring era · distinct scored links. Empty eras show —; zero-VP links remain in the denominator.");
        }
    }
}
