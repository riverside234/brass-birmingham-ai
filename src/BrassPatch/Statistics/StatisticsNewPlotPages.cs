using System;
using System.Collections.Generic;
using System.Linq;
namespace BrassPatch.Statistics
{
    public static partial class StatisticsPlotPages
    {
        static void Development(StatisticsPageView v,CompletedMatchStatistics r,StatisticsFilters f)
        {
            var records=StatisticsChartQueries.Development(r,f).ToArray();
            var plot=Plot(v,"development","Tiles removed through development",StatisticsPlotKind.StackedColumns,"Counts tiles, not tile levels or Develop actions. Level 3+ groups the original levels; exact data retains each level. Source: "+f.DevelopmentSource+". Merchant developments remain part of Sell.","Industry","Tiles removed");
            foreach(string level in new[]{"L1","L2","L3+"})Series(plot,level,level);
            foreach(var player in f.ByPlayer?Players(r,f).Cast<StatisticsPlayer>():new StatisticsPlayer[]{null})foreach(string type in StatisticsPages.IndustryTypes.Where(t=>f.Industry==""||t==f.Industry)) {
                int row=plot.Rows.Count;plot.Rows.Add((player==null?"":StatisticsMetrics.PlayerName(r,player)+" · ")+StatisticsMetrics.IndustryName(type));plot.Columns.Add(plot.Rows[row]);
                foreach(var series in plot.Series){var group=records.Where(e=>e.Resource==type&&(player==null||e.Player==player.Color)&&(series.Id=="L3+"?e.Quantity>=3:e.Quantity==(series.Id=="L1"?1:2))).ToArray();
                    var drill=Target(2,player==null?null:(BoardGameRules.Entities.Players.PlayerColor?)player.Color,group.Select(StatisticsChartQueries.EntryKey),industry:type);
                    Mark(plot,series.Id,plot.Rows[row],row,group.Length,row,column:row,detail:String.Join("\n",group.Select(e=>r.Name(e.Player)+" · original level "+e.Quantity+" · "+e.Position+" · "+e.Detail)),target:drill);
                }
            }
            plot.Context.Add(records.Length+" tiles removed · "+records.Where(e=>r.Actions.Any(a=>a.Key==e.ActionKey&&a.Kind=="Develop")).Select(e=>e.ActionKey).Distinct().Count()+" Develop actions represented");
        }
        static void LinkSources(StatisticsPageView v,CompletedMatchStatistics r,StatisticsFilters f)
        {
            var plot=Plot(v,"link-sources","Who supplied the link points",StatisticsPlotKind.StackedBars,"Scored links only, including zero-VP links in the average. Own industry, other players' industry and merchant endpoint VP sum to the actual link score. Point attribution does not establish route dependency.",f.LinkShare?"Link VP (%)":"Link VP","Player / era");plot.Percent=f.LinkShare;
            foreach(string source in new[]{"Own","Opponents","Merchant"})Series(plot,source,source=="Own"?"Own industries":source=="Opponents"?"Others' industries":"Merchants");
            foreach(var player in Players(r,f))foreach(int era in r.Introductory?new[]{1}:new[]{1,2})if(f.Era==0||era==f.Era) {
                var links=StatisticsChartQueries.Links(r,f).Where(l=>l.Scored&&l.Player==player.Color&&l.Era==era).ToArray();int row=plot.Rows.Count;plot.Rows.Add(StatisticsMetrics.PlayerName(r,player)+" · "+(era==1?"Canal":"Rail"));
                foreach(var s in plot.Series){Mark(plot,s.Id,plot.Rows[row],0,links.Sum(l=>s.Id=="Own"?l.OwnVP:s.Id=="Opponents"?l.OthersVP:l.MerchantVP),row,detail:links.Length+" scored links · "+StatisticsFormat.Ratio(links.Sum(l=>l.VP),links.Length)+" VP/link (zero-VP links included)",target:Target(3,player.Color,links.Select(l=>l.Key)));plot.Marks.Last().OtherLabel=links.Length+" links · "+StatisticsFormat.Ratio(links.Sum(l=>l.VP),links.Length)+" VP/link";}
            }
        }
        static void MarketProfile(StatisticsPageView v,CompletedMatchStatistics r,StatisticsFilters f)
        {
            var prices=new Dictionary<string,StatisticsEntry[]>();foreach(string resource in new[]{"Coal","Iron"})prices[resource]=StatisticsChartQueries.Purchases(r,f,resource).ToArray();
            double max=prices.Values.SelectMany(es=>es.GroupBy(e=>e.Player).Select(g=>(double)g.Sum(e=>e.TotalPrice)/Math.Max(1,g.Sum(e=>e.Quantity)))).DefaultIfEmpty(0).Max();
            foreach(string resource in new[]{"Coal","Iron"}) {
                var plot=Plot(v,"market-"+resource.ToLowerInvariant(),resource+" · weighted purchase price",StatisticsPlotKind.Lollipop,"Recorded total paid ÷ total units bought. Receipts and market quotes are separate. Purchases are already included once in action spending.","Price per unit (£)","Player");plot.XUnit=StatisticsAxisUnit.Price;plot.SharedMax=max;
                foreach(var p in Players(r,f)){
                    Series(plot,p.Color.ToString(),StatisticsMetrics.PlayerName(r,p),StatisticsColorRole.Player,p.Color);int row=plot.Rows.Count;plot.Rows.Add(StatisticsMetrics.PlayerName(r,p));var entries=prices[resource].Where(e=>e.Player==p.Color).ToArray();int units=entries.Sum(e=>e.Quantity),paid=entries.Sum(e=>e.TotalPrice);
                    var target=Target(5,p.Color,entries.Select(StatisticsChartQueries.EntryKey));target.ResourceView="Market";target.Resource=resource;target.Outgoing=false;
                    Mark(plot,p.Color.ToString(),p.Name,0,units>0?(double)paid/units:0,row,detail:units+" units bought · "+StatisticsFormat.Money(paid)+" total paid",target:target);
                    var m=plot.Marks.Last();m.Missing=units==0;m.ValueLabel=units==0?"No purchases":StatisticsAxisLayout.Format((double)paid/units,StatisticsAxisUnit.Price,1,true)+" / unit";m.OtherLabel=units+" units · "+StatisticsFormat.Money(paid)+" paid";
                }
            }
            var transactions=new StatisticsSection("Market transactions","Purchases and sale receipts are separate. Transaction quantities weight the average price; quotes are observations, not the paid price.","Player","Resource","Direction","Units","Total paid / received","Price / unit","Quote before → after");
            foreach(string resource in new[]{"Coal","Iron"})foreach(var e in prices[resource].Concat(StatisticsMetrics.Entries(r,f).Where(e=>e.Kind=="Market sale"&&e.Resource==resource)))transactions.Rows.Add(new StatisticsRow(StatisticsChartQueries.EntryKey(e),e.Position+" · "+e.Detail,r.Name(e.Player),resource,e.Kind,e.Quantity.ToString(),StatisticsFormat.Money(e.TotalPrice),StatisticsChartQueries.Number(e.Quantity>0?(double)e.TotalPrice/e.Quantity:Double.NaN),e.BeforeQuote+" → "+e.AfterQuote){Player=e.Player});
            v.Ledgers.Add(transactions);v.Headlines.Clear();v.Headlines.Add("Player / era / action scope · purchase prices; receipts separate");
        }
        static void CardPurpose(StatisticsPageView v,CompletedMatchStatistics r,StatisticsFilters f)
        {
            var records=StatisticsMetrics.Entries(r,f).Where(e=>e.Kind=="Card").ToArray();
            var plot=Plot(v,"card-purpose","Cards used · purpose and type",StatisticsPlotKind.StackedBars,"Recorded card payments and Scout extra trades. Draws, acquisitions, setup discards and shuffles are excluded. Counts and share use the same played-card denominator.",f.CardShare?"Cards (%)":"Cards","Purpose");plot.Percent=f.CardShare;
            foreach(string type in new[]{"Location","Industry","Wild Location","Wild Industry"})Series(plot,type,type);
            foreach(var player in f.ByPlayer?Players(r,f).Cast<StatisticsPlayer>():new StatisticsPlayer[]{null})foreach(string category in new[]{"Build","Non-Build payment","Scout extra trade"}) {
                int row=plot.Rows.Count;string label=category=="Build"?"Build authorization":category;plot.Rows.Add((player==null?"":StatisticsMetrics.PlayerName(r,player)+" · ")+label);
                foreach(var series in plot.Series){var entries=records.Where(e=>(player==null||e.Player==player.Color)&&(category=="Scout extra trade"?e.Category=="Scout trade":e.Category==category)&&e.Label==series.Id).ToArray();Mark(plot,series.Id,plot.Rows[row],0,entries.Length,row,detail:entries.Length+" played cards · "+category,target:Target(6,player==null?null:(BoardGameRules.Entities.Players.PlayerColor?)player.Color,entries.Select(StatisticsChartQueries.EntryKey)));}
            }
        }
        static void MerchantDestinations(StatisticsPageView v,CompletedMatchStatistics r,StatisticsFilters f)
        {
            var records=StatisticsMetrics.Entries(r,f).Where(e=>e.Kind=="Goods sale").ToArray();
            var plot=Plot(v,"merchant-sales","Goods tiles sold by destination",StatisticsPlotKind.PairedBars,"Tiles sold, not Sell actions or beer uses. Recorded and uniquely derived destinations remain distinguishable; unattributed destinations are never guessed.","Goods tiles sold","Merchant destination");
            var destinations=r.Locations.Where(l=>l.Merchant).Select(l=>l.City).Concat(records.Select(e=>String.IsNullOrEmpty(e.Destination)?"Unattributed":e.Destination)).Concat(new[]{"Unattributed"}).Distinct().OrderBy(s=>s=="Unattributed"?"zzzz":s).ToArray();
            plot.Rows.AddRange(destinations.Select(CityName));foreach(var p in Players(r,f)){
                Series(plot,p.Color.ToString(),StatisticsMetrics.PlayerName(r,p),StatisticsColorRole.Player,p.Color);
                for(int row=0;row<destinations.Length;row++){string city=destinations[row];var entries=records.Where(e=>e.Player==p.Color&&(String.IsNullOrEmpty(e.Destination)?"Unattributed":e.Destination)==city).ToArray();var target=Target(5,p.Color,entries.Select(StatisticsChartQueries.EntryKey),industry:f.Industry);target.ResourceView="Merchants";
                    Mark(plot,p.Color.ToString(),CityName(city),0,entries.Sum(e=>e.Quantity),row,detail:entries.Count(e=>city!="Unattributed"&&!e.DestinationDerived)+" recorded · "+entries.Count(e=>e.DestinationDerived)+" uniquely derived · "+entries.Count(e=>city=="Unattributed")+" unattributed",target:target);
                }
            }
            plot.Context.Add(records.Length+" sold tiles: "+records.Count(e=>e.Destination!="Unattributed"&&!String.IsNullOrEmpty(e.Destination)&&!e.DestinationDerived)+" recorded, "+records.Count(e=>e.DestinationDerived)+" uniquely derived, "+records.Count(e=>e.Destination=="Unattributed"||String.IsNullOrEmpty(e.Destination))+" unattributed.");
            v.Headlines.Clear();v.Headlines.Add(records.Sum(e=>e.Quantity)+" sold goods tiles");
            var bonus=new StatisticsSection("Merchant bonuses · separate units","Player/era bonus ledger; industry and goods-sale evidence filters apply to the destination chart only. Activations and actual balance changes are separate records, not summed duplicates. Different units are not combined into a score.","Player","When","Recorded bonus","Cash Δ","Marker spaces Δ","Income levels Δ","VP Δ");
            foreach(var e in r.Entries.Where(e=>f.Includes(e.Player,e.Era)&&(e.Kind=="Balance"&&e.Category=="Merchant"||e.Kind=="Merchant bonus")))bonus.Rows.Add(new StatisticsRow(StatisticsChartQueries.EntryKey(e),e.Detail,r.Name(e.Player),e.Position,e.Kind+" · "+e.Label,StatisticsFormat.Money(e.Cash),e.Marker.ToString(),e.Income.ToString(),e.VP.ToString()){Player=e.Player});v.Ledgers.Add(bonus);
        }
        static void PositionShare(StatisticsPageView v,CompletedMatchStatistics r,StatisticsFilters f)
        {
            var plot=Plot(v,"position-share","Share of recorded turn positions",StatisticsPlotKind.StackedBars,"Each player's denominator includes only rounds in which their position is known. Missing observations are unavailable; back-to-back turns are a separate measure.","Known rounds (%)","Player");plot.Percent=true;
            for(int i=1;i<=r.Players.Count;i++)Series(plot,"seat"+i,i==1?"1st":i==2?"2nd":i==3?"3rd":"4th");
            var rounds=StatisticsChartQueries.Rounds(r,f).ToArray();foreach(var p in Players(r,f)){
                int row=plot.Rows.Count;plot.Rows.Add(StatisticsMetrics.PlayerName(r,p));var known=rounds.Where(t=>t.Order.Contains(p.Color)).ToArray();
                foreach(var series in plot.Series){int seat=Int32.Parse(series.Id.Substring(4));var group=known.Where(t=>t.Order.IndexOf(p.Color)+1==seat).ToArray();var target=Target(7,p.Color,group.Select(StatisticsChartQueries.RoundKey));
                    Mark(plot,series.Id,p.Name,0,group.Length,row,detail:group.Length+" / "+known.Length+" known rounds · "+(rounds.Length-known.Length)+" unavailable",target:target);plot.Marks.Last().Missing=known.Length==0;
                }
            }
            var evidence=new StatisticsSection("Round position evidence","The preceding round's spending determines this order. No preceding spending is assigned across the era maintenance boundary.","Player","Era / round","Position","Previous spending","Back-to-back");
            foreach(var p in Players(r,f))foreach(var round in rounds.Where(t=>StatisticsChartQueries.Evidence(f,StatisticsChartQueries.RoundKey(t)))){
                int index=r.Rounds.IndexOf(round);var previous=index>0?r.Rounds[index-1]:null;bool consecutive=previous!=null&&previous.Era==round.Era&&previous.Order.Count>0&&round.Order.Count>0&&previous.Order.Last()==p.Color&&round.Order.First()==p.Color;
                evidence.Rows.Add(new StatisticsRow(StatisticsChartQueries.RoundKey(round)+"-"+p.Color,"Recorded order: "+String.Join(" → ",round.Order.Select(r.Name)),r.Name(p.Color),(round.Era==1?"Canal ":"Rail ")+round.Round,round.Order.Contains(p.Color)?(round.Order.IndexOf(p.Color)+1).ToString():"Unavailable",previous!=null&&previous.Era==round.Era?StatisticsFormat.Money(previous.Spent(p.Color)):"—",consecutive?"Yes":"No"){Player=p.Color});
            }v.Ledgers.Add(evidence);
        }
        static void PlayerEfficiency(StatisticsPageView v,CompletedMatchStatistics r,StatisticsFilters f)
        {
            var plot=Plot(v,"player-efficiency","Economics and VP efficiency",StatisticsPlotKind.Efficiency,"Player/era totals. Detail filters below do not narrow either denominator. Confirmed actions include Pass. Spending basis: "+(f.AllCashPayments?"all negative cash payments, including debt":"action spending, including market resources once")+". Whole game uses authoritative final VP; era uses actual net awards.","VP per confirmed action","Cash spent per VP (£/VP)");
            foreach(var t in StatisticsChartQueries.Efficiency(r,f)){
                Series(plot,t.Player.Color.ToString(),t.Player.Name,StatisticsColorRole.Player,t.Player.Color);int row=plot.Rows.Count;plot.Rows.Add(StatisticsMetrics.PlayerName(r,t.Player));
                foreach(bool cost in new[]{false,true}){double value=cost?t.CostPerVP:t.VPPerAction;Mark(plot,t.Player.Color.ToString(),t.Player.Name,0,Double.IsNaN(value)?0:value,row,detail:t.VP+" VP ÷ "+t.Actions+" confirmed actions; "+StatisticsFormat.Money(t.Spending)+" "+(f.AllCashPayments?"all cash payments":"action spending")+" · "+(f.Era==0?"Whole game":f.Era==1?"Canal":"Rail"),target:cost?t.CostEvidence:t.ActionEvidence);var m=plot.Marks.Last();m.Metric=cost?"cost":"vp";m.Missing=Double.IsNaN(value);m.ValueLabel=m.Missing?"—":(cost?"£":"")+StatisticsChartQueries.Number(value)+(cost?" / VP":" VP / action");m.OtherLabel=t.VP+" VP · "+t.Actions+" actions · "+StatisticsFormat.Money(t.Spending)+" spent";}
            }
            plot.Context.Add((f.Era==0?"Whole-game":f.Era==1?"Canal":"Rail")+" totals for selected players · "+(f.AllCashPayments?"all cash payments":"action spending")+". Negative era VP is valid for VP/action; £/VP is unavailable for nonpositive VP.");
            v.Ledgers.Add(StatisticsChartQueries.EfficiencyLedger(r,f));
        }
        public static string CityName(string value){return value=="SouthWest"?"South-west farm":value=="Central"?"Central farm":value=="StokeOnTrent"?"Stoke-on-Trent":value=="BurtonOnTrent"?"Burton upon Trent":value=="Coalbrookdale"?"Coalbrookdale":value=="Unattributed"?value:System.Text.RegularExpressions.Regex.Replace(value??"", "([a-z])([A-Z])", "$1 $2");}
    }
}
