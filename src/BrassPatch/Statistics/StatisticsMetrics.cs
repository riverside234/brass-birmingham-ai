using System;
using System.Collections.Generic;
using System.Linq;
using BoardGameRules.Entities.Players;
namespace BrassPatch.Statistics
{
    public sealed class StatisticsDrill
    {
        public int Page;
        public int? Era;
        public PlayerColor? Player;
        public string Industry="", Region="", Action="", Resource="";
        public bool? Outgoing;
        public string ResourceView;
        public bool? AllCashPayments;
        public bool HasKeys;
        public readonly HashSet<string> Keys=new HashSet<string>();
    }
    public sealed class StatisticsLoanRecovery
    {
        public StatisticsAction Loan;
        public int Baseline, After;
        public int? RecoveredOwnTurn;
        public int? RecoveredRevision,ActionsToRecover;
        public int ActualIncomePaid;
    }
    // Pure queries over a detached report. No live game, inferred paths or hidden hands.
    public static class StatisticsMetrics
    {
        public static string IndustryName(string name)
        { switch(name){case "CottonMill":return "Cotton mills";case "Manufacturer":return "Manufactories";case "CoalMine":return "Coal mines";case "IronWorks":return "Ironworks";default:return name=="Pottery"?"Potteries":name=="Brewery"?"Breweries":name;} }
        public static IEnumerable<StatisticsAsset> Assets(CompletedMatchStatistics r,StatisticsFilters f)
        {return r.Assets.Where(f.Asset);}
        public static IEnumerable<StatisticsAction> Actions(CompletedMatchStatistics r,StatisticsFilters f)
        {return r.Actions.Where(f.Frame);}
        public static IEnumerable<StatisticsEntry> Entries(CompletedMatchStatistics r,StatisticsFilters f)
        {
            var keys=new HashSet<string>(r.Assets.Where(a=>f.Players.Contains(a.Player)&&(f.Industry==""||f.Industry==a.Type)&&(f.Region==""||f.Region==a.RegionColor)).Select(a=>a.Key));
            var actions=new HashSet<string>(Actions(r,f).Select(a=>a.Key));
            return r.Entries.Where(e=>f.Includes(e.Player,e.Era)&&(f.Industry==""&&f.Region==""||e.AssetKey!=null&&keys.Contains(e.AssetKey))
                &&(f.Action==""||actions.Contains(e.ActionKey??""))&&(!f.EvidenceRestricted&&f.Evidence.Count==0||f.Evidence.Contains(e.AssetKey??"")||f.Evidence.Contains(e.ActionKey??"")||f.Evidence.Contains("entry-"+e.Revision+"-"+e.Player)));
        }
        public static int AssetVP(StatisticsAsset a,int era=0)
        {return (era==2?0:a.CanalVP)+(era==1?0:a.RailVP)+(era==2?0:a.IntroVP);}
        public static int Score(CompletedMatchStatistics r,PlayerColor p,string category,int era=0)
        {return r.Entries.Where(e=>e.Player==p&&e.Kind=="Balance"&&e.Category==category&&(era==0||e.Era==era)).Sum(e=>e.VP);}
        public static StatisticsPlayer Focus(CompletedMatchStatistics r,StatisticsFilters f)
        {return r.Players.FirstOrDefault(p=>f.Focus==p.Color)??r.Players.FirstOrDefault(p=>(p.Opponent??"").StartsWith("Human",StringComparison.OrdinalIgnoreCase))??r.Players.First();}
        public static StatisticsPlayer Reference(CompletedMatchStatistics r,StatisticsFilters f)
        {
            var focus=Focus(r,f);var explicitPlayer=r.Players.FirstOrDefault(p=>p.Color==f.Reference&&p.Color!=focus.Color);
            return explicitPlayer??r.Players.Where(p=>p.Color!=focus.Color).OrderBy(p=>p.Rank).ThenByDescending(p=>p.FinalVP).ThenBy(p=>(int)p.Color).FirstOrDefault();
        }
        public static List<StatisticsLoanRecovery> Loans(CompletedMatchStatistics r,PlayerColor player)
        {
            return r.Actions.Where(a=>a.Player==player&&a.Kind=="Loan").Select(a=>{
                var before=r.Samples.LastOrDefault(s=>s.Player==player&&s.Revision<a.Revision);
                var after=r.Samples.LastOrDefault(s=>s.Player==player&&s.Revision<=a.EndRevision);
                int baseline=before==null?0:before.Income;
                var recovered=r.Samples.FirstOrDefault(s=>s.Player==player&&s.Revision>a.EndRevision&&s.Income>=baseline);
                return new StatisticsLoanRecovery{Loan=a,Baseline=baseline,After=after==null?baseline:after.Income,RecoveredOwnTurn=recovered==null?(int?)null:recovered.OwnTurn,RecoveredRevision=recovered==null?(int?)null:recovered.Revision,ActionsToRecover=recovered==null?(int?)null:r.Actions.Count(x=>x.Player==player&&x.Revision>a.EndRevision&&x.Revision<=recovered.Revision),ActualIncomePaid=r.Entries.Where(e=>e.Player==player&&e.Kind=="Balance"&&e.Category=="Income payment"&&e.Revision>a.EndRevision&&(recovered==null||e.Revision<=recovered.Revision)).Sum(e=>Math.Max(0,e.Cash))};
            }).ToList();
        }
        public static int DevelopmentBuilds(CompletedMatchStatistics r,StatisticsEntry e,int era=0)
        {var boundary=r.Boundaries.LastOrDefault(b=>b.Player==e.Player&&(era==0?b.Name=="Final":b.Name=="Scoring"&&b.Era==era));return r.Assets.Count(a=>a.Player==e.Player&&a.Type==e.Resource&&a.Revision>e.Revision&&a.Level>e.Quantity&&(boundary==null||a.Revision<=boundary.Revision));}
        public static IEnumerable<StatisticsEntry> Flow(CompletedMatchStatistics r,StatisticsFilters f)
        {
            // Source industry/region is independent of consumer ownership and of the
            // era in which the source was built. Carried Canal suppliers can serve Rail.
            var sources=new HashSet<string>(r.Assets.Where(a=>(f.Industry==""||a.Type==f.Industry)&&(f.Region==""||a.RegionColor==f.Region)).Select(a=>a.Key));
            var actions=new HashSet<string>(r.Actions.Where(a=>f.Action==""||a.Kind==f.Action).Select(a=>a.Key));
            return r.Entries.Where(e=>e.Kind=="Resource use"&&e.Resource==f.Resource&&(f.Era==0||e.Era==f.Era)&&
                (f.Industry==""&&f.Region==""||sources.Contains(e.AssetKey??""))&&(f.Action==""||actions.Contains(e.ActionKey??""))&&
                (!f.EvidenceRestricted&&f.Evidence.Count==0||f.Evidence.Contains(e.AssetKey??"")||f.Evidence.Contains(e.ActionKey??"")||f.Evidence.Contains("entry-"+e.Revision+"-"+e.Player))&&
                (f.Outgoing?e.Supplier.HasValue&&f.Players.Contains(e.Supplier.Value):f.Players.Contains(e.Player)));
        }
        public static StatisticsBoundary Boundary(CompletedMatchStatistics r,PlayerColor p,string name,int era=0)
        {return r.Boundaries.LastOrDefault(b=>b.Player==p&&b.Name==name&&(era==0||b.Era==era));}
        public static bool At(StatisticsAsset a,StatisticsBoundary b)
        {return b!=null&&a.Revision<=b.Revision&&(a.RemovedRevision==0||a.RemovedRevision>b.Revision);}
        public static string PlayerName(CompletedMatchStatistics r,StatisticsPlayer p){return r.Players.Count(q=>q.Name==p.Name)>1?p.Name+" ["+p.Color+"]":p.Name;}
        public static List<StatisticsSection> AdditionalLedgers(CompletedMatchStatistics r,StatisticsFilters f,int page)
        {
            var output=new List<StatisticsSection>();if(!r.CompleteHistory)return output;
            Action<StatisticsSection,StatisticsPosition,string,string[]> row=(s,p,detail,cells)=>s.Rows.Add(new StatisticsRow("entry-"+p.Revision+"-"+p.Player,detail,cells){Player=p.Player});
            if(page==2||page==8||page==9){
                var flips=new StatisticsSection("Flip timing and trigger","Own turns and global turns are separate units. Opponent use can flip a supplier between its turns.","Player","Industry / city","Own-turn delay","Global-turn delay","Trigger","Actual VP");
                foreach(var a in Assets(r,f))flips.Rows.Add(new StatisticsRow(a.Key,a.Position+" · built revision "+a.Revision+"; flip revision "+a.FlipRevision,new[]{r.Name(a.Player),IndustryName(a.Type)+" · "+a.City,a.Flipped?(a.FlipOwnTurn-a.OwnTurn).ToString():"Unflipped",a.Flipped?(a.FlipTurn-a.Turn).ToString():"—",a.FlipTrigger??(a.Sellable?"Goods sale":"Unrecorded"),AssetVP(a,f.Era).ToString()}){Player=a.Player});
                output.Add(flips);
                var develop=new StatisticsSection("Development follow-through","Later higher-level builds are observed follow-through, not proof development was necessary. Next mat tile uses the actual era restriction.","Player","Developed tile","Next level","Rail allowed","Later higher builds");
                foreach(var e in StatisticsChartQueries.Development(r,f))row(develop,e,e.Detail+" · tile "+e.TileId,new[]{r.Name(e.Player),IndustryName(e.Resource)+" L"+e.Quantity,e.NextLevel==0?"Exhausted":"L"+e.NextLevel,e.NextLevel==0?"—":e.NextRailBlocked?"Blocked (Canal only)":"Yes",DevelopmentBuilds(r,e,f.Era).ToString()});
                output.Add(develop);
            }
            if(page==4||page==8){
                var loans=new StatisticsSection("Income recovery after loans","Era filters loan acquisition; recovery is observed through match end, including a later era. Recovery means the income level first reaches its value immediately before this loan. Multiple episodes may overlap; this is not cash payback. Intervening income is actual cash, not loan profit.","Player","Loan turn / amount","Pre → post level","Recovery own turns / actions","Intervening income cash");
                foreach(var p in r.Players.Where(p=>f.Players.Contains(p.Color)))foreach(var l in Loans(r,p.Color).Where(l=>f.Era==0||l.Loan.Era==f.Era))row(loans,l.Loan,l.Loan.Position,new[]{p.Name,l.Loan.OwnTurn+" / "+StatisticsFormat.Money(l.Loan.Borrowed),l.Baseline+" → "+l.After,l.RecoveredOwnTurn.HasValue?(l.RecoveredOwnTurn-l.Loan.OwnTurn)+" / "+l.ActionsToRecover:"Not recovered",StatisticsFormat.Money(l.ActualIncomePaid)});
                output.Add(loans);
                var rail=new StatisticsSection("Rail entry and opening actions","Snapshot after Canal cleanup, before the first Rail turn. A Canal-only tile can block Rail construction even when its level is above I.","Player","Cash / income","Carried tiles","Rail-blocked industries","First two Rail actions");
                if(!r.Introductory)foreach(var p in r.Players.Where(p=>f.Players.Contains(p.Color))){
                    var b=Boundary(r,p.Color,"Rail entry");var mat=r.MatTiles.Where(m=>m.Player==p.Color&&m.Boundary=="Rail entry");
                    rail.Rows.Add(new StatisticsRow("rail-"+p.Color,"Carried suppliers: "+String.Join(", ",r.Assets.Where(a=>a.Player==p.Color&&a.Carried&&a.Produced>0).Select(a=>IndustryName(a.Type)+" "+a.City)),p.Name,b==null?"Unavailable":StatisticsFormat.Money(b.Cash)+" / "+b.Income,r.Assets.Count(a=>a.Player==p.Color&&a.Carried).ToString(),String.Join(", ",mat.Where(m=>m.RailBlocked).Select(m=>IndustryName(m.Type)+" L"+m.Level)),String.Join(" → ",r.Actions.Where(a=>a.Player==p.Color&&a.Era==2).Take(2).Select(a=>a.Kind))){Player=p.Color});
                }output.Add(rail);
                var income=new StatisticsSection("Actual income payments","Cash paid at Income events; negative levels can require payments. Gross positive income is distinct from the final income marker.","Player","Paydays","Income cash received","Negative-income cash paid","Debt settlement cash","Debt VP penalty");
                foreach(var p in r.Players.Where(p=>f.Players.Contains(p.Color))){var entries=Entries(r,f).Where(e=>e.Player==p.Color&&e.Kind=="Balance").ToArray();var paid=entries.Where(e=>e.Category=="Income payment").ToArray();income.Rows.Add(new StatisticsRow("paid-"+p.Color,"Includes zero-payment paydays. Debt cash is recorded spending outside an action; liquidation receipts appear separately.",p.Name,Entries(r,f).Count(e=>e.Player==p.Color&&e.Kind=="Payday").ToString(),StatisticsFormat.Money(paid.Where(e=>e.Cash>0).Sum(e=>e.Cash)),StatisticsFormat.Money(-paid.Where(e=>e.Cash<0).Sum(e=>e.Cash)),StatisticsFormat.Money(-entries.Where(e=>e.Category=="Debt settlement").Sum(e=>Math.Min(0,e.Cash))),(-entries.Where(e=>e.Category=="Penalty").Sum(e=>e.VP)).ToString()){Player=p.Color});}output.Add(income);
            }
            if(page==5){
                var flow=new StatisticsSection("Resource supply records",(f.Outgoing?"Selected suppliers → all recorded consumers.":"All recorded suppliers → selected consumers.")+" Same units and typed scope as both resource plots. Industry/region refers to the source tile; Action refers to the consuming action.","Supplier","Consumer","Resource / purpose","Units","Source tile / city","When");
                foreach(var e in Flow(r,f)){
                    var source=r.Assets.FirstOrDefault(a=>a.Key==e.AssetKey);var owner=f.Outgoing?e.Supplier:(PlayerColor?)e.Player;
                    var drill=new StatisticsDrill{Page=5,Player=owner,Resource=f.Resource,Outgoing=f.Outgoing,HasKeys=true};drill.Keys.Add("entry-"+e.Revision+"-"+e.Player);
                    flow.Rows.Add(new StatisticsRow(StatisticsChartQueries.RecordRowKey(e),e.Detail,e.Supplier.HasValue?r.Name(e.Supplier.Value):e.Source,r.Name(e.Player),e.Resource+" / "+e.Category,e.Quantity.ToString(),source==null?e.Source:IndustryName(source.Type)+" · "+source.City,e.Position){Player=owner,Drill=drill});
                }output.Add(flow);
                var prices=new StatisticsSection("Recorded market transaction prices",(f.Outgoing?"Outgoing mode: selected suppliers' market sales only. Switch to incoming for purchase records. ":"")+"Unit price = recorded total transaction price / quantity. Market sale receipts are not negative spending and are separate from units supplied to actions.","Player","Resource","Direction","Units","Total price","Price / unit","Quote before → after");
                foreach(var e in Entries(r,f).Where(e=>(e.Kind=="Market sale"||!f.Outgoing&&e.Kind=="Market purchase")&&e.Resource==f.Resource))row(prices,e,e.Position,new[]{r.Name(e.Player),e.Resource,e.Kind,e.Quantity.ToString(),StatisticsFormat.Money(e.TotalPrice),StatisticsFormat.Ratio(e.TotalPrice,e.Quantity),e.BeforeQuote+" → "+e.AfterQuote});output.Add(prices);
            }
            if(page==6){
                var wild=new StatisticsSection("Wild card timing","Holding time counts global turns. A wildcard opens options; its use alone does not establish a tactical turnaround.","Player","Card","Acquired","Used","Action / timing");
                foreach(var e in Entries(r,f).Where(e=>e.Kind=="Wildcard acquired"||e.Kind=="Card"&&e.Label.StartsWith("Wild")))row(wild,e,e.Detail,new[]{r.Name(e.Player),e.Kind=="Card"?e.Label:e.Category,e.Kind=="Wildcard acquired"?e.OwnTurn.ToString():"—",e.Kind=="Card"?e.OwnTurn.ToString():"—",e.Label+" · "+e.Detail});output.Add(wild);
            }
            return output;
        }
        public static StatisticsSection SupportingRecords(CompletedMatchStatistics r,StatisticsFilters f)
        {
            var s=new StatisticsSection("Supporting records by identity","Exact evidence IDs and related recorded operations. Actor and supplier can differ; final tile state is labeled separately from historical event timing.","Record ID","Actor / owner","When","Operation / tile","Values","Context");
            foreach(var a in r.Assets.Where(a=>f.Evidence.Contains(a.Key)))s.Rows.Add(new StatisticsRow(a.Key,a.Position+" · flip revision "+a.FlipRevision+"; removal revision "+a.RemovedRevision,a.Key,r.Name(a.Player),a.Position,IndustryName(a.Type)+" L"+a.Level+" · "+a.City,"Cost "+StatisticsFormat.Money(a.BaseCost+a.MarketCost)+" · actual VP "+AssetVP(a,f.Era),"Final: "+(a.Remaining?(a.Flipped?"flipped":"unflipped"):a.Removal)){Player=a.Player});
            foreach(var l in r.Links.Where(l=>f.Evidence.Contains(l.Key)))s.Rows.Add(new StatisticsRow(l.Key,l.Contributions,l.Key,r.Name(l.Player),l.Position,l.Route,l.VP+" VP",l.OwnVP+" own / "+l.OthersVP+" opponents / "+l.MerchantVP+" merchant VP"){Player=l.Player});
            foreach(var round in r.Rounds.Where(t=>f.Evidence.Contains(StatisticsChartQueries.RoundKey(t)))){
                int index=r.Rounds.IndexOf(round);var previous=index>0?r.Rounds[index-1]:null;
                s.Rows.Add(new StatisticsRow(StatisticsChartQueries.RoundKey(round),"Preceding spending: "+(previous==null||previous.Era!=round.Era?"Era entry; no preceding round":String.Join(" · ",previous.Order.Select(p=>r.Name(p)+" "+StatisticsFormat.Money(previous.Spent(p))))),StatisticsChartQueries.RoundKey(round),"Recorded round",(round.Era==1?"Canal":"Rail")+" · round "+round.Round,"Turn order",String.Join(" → ",round.Order.Select(r.Name)),"Current-round spending: "+String.Join(" · ",round.Order.Select(p=>r.Name(p)+" "+StatisticsFormat.Money(round.Spent(p))))));
            }
            var entries=r.Entries.Where(e=>f.Evidence.Contains("entry-"+e.Revision+"-"+e.Player)||f.Evidence.Contains(e.AssetKey??"")||f.Evidence.Contains(e.ActionKey??"")).ToArray();
            var actions=new HashSet<string>(entries.Where(e=>e.ActionKey!=null).Select(e=>e.ActionKey));
            foreach(var a in r.Actions.Where(a=>f.Evidence.Contains(a.Key)||actions.Contains(a.Key)))s.Rows.Add(new StatisticsRow(a.Key,a.Details,a.Key,r.Name(a.Player),a.Position,a.Kind,StatisticsFormat.Money(a.Spending)+" spent · "+a.Goods+" goods · "+a.Links+" links",a.Cards){Player=a.Player});
            foreach(var e in entries)s.Rows.Add(new StatisticsRow(StatisticsChartQueries.RecordRowKey(e),e.Detail,"entry-"+e.Revision+"-"+e.Player,r.Name(e.Player)+(e.Supplier.HasValue?" ← "+r.Name(e.Supplier.Value):""),e.Position,e.Kind+" · "+e.Label,e.Quantity+" units · cash Δ "+StatisticsFormat.Money(e.Cash)+" · VP Δ "+e.VP,(e.Source??"")+" → "+(e.Destination??"")+" · "+e.Detail){Player=e.Player});
            return s;
        }
    }
}
