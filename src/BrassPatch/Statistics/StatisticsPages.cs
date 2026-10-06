using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using BoardGameRules.Entities.Players;
namespace BrassPatch.Statistics
{
    public sealed class StatisticsFilters
    {
        public readonly HashSet<PlayerColor> Players = new HashSet<PlayerColor>();
        public int Era;
        public string Search = "";
        public bool AllCashPayments;
        public string ResourceView="Supply",DevelopmentSource="All",EfficiencyView="Returns";
        public bool ByPlayer,CardShare,LinkShare;
        public bool ActionShare=true;
        public string Industry = "", Region = "", Action = "", Resource = "Beer";
        public string Boundary = "Final";
        public string MapMode = "Occupancy";
        public int BoundaryEra;
        public bool Outgoing, ShowAll;
        public bool EvidenceRestricted;
        public PlayerColor? Focus, Reference;
        public readonly HashSet<string> Evidence = new HashSet<string>();
        public bool Asset(StatisticsAsset a) { return Includes(a.Player,a.Era) && (Industry==""||Industry==a.Type) && (Region==""||Region==a.RegionColor) && (!EvidenceRestricted&&Evidence.Count==0||Evidence.Contains(a.Key)); }
        public bool Frame(StatisticsAction a) { return Includes(a.Player,a.Era) && (Action==""||Action==a.Kind) && (!EvidenceRestricted&&Evidence.Count==0||Evidence.Contains(a.Key)); }
        public bool Typed { get {return Industry!=""||Region!=""||Action!=""||EvidenceRestricted||Evidence.Count>0;} }
        public bool Includes(PlayerColor player, int era) { return Players.Contains(player) && (Era == 0 || Era == era); }
        public void Reset(CompletedMatchStatistics report) { Players.Clear(); foreach (var p in report.Players) Players.Add(p.Color); Era = 0; Search = ""; AllCashPayments = false; Industry=Region=Action="";Evidence.Clear();EvidenceRestricted=false;ShowAll=Outgoing=false;Boundary="Final";BoundaryEra=0;MapMode="Occupancy";Resource="Beer";ResourceView="Supply";EfficiencyView="Returns";DevelopmentSource="All";ByPlayer=CardShare=LinkShare=false;ActionShare=true; }
    }
    public sealed class StatisticsRow
    {
        public string Key, Detail;
        public string[] Cells;
        public PlayerColor? Player;
        public int TargetPage = -1;
        public string TargetSearch;
        public StatisticsDrill Drill;
        public StatisticsRow(string key, string detail, params string[] cells) { Key = key; Detail = detail; Cells = cells; }
    }
    public sealed class StatisticsSection
    {
        public string Title, Definition;
        public string Id; public string Identity { get { return Id??Title; } }
        public string[] Columns;
        public readonly List<StatisticsRow> Rows = new List<StatisticsRow>();
        public StatisticsChart Chart;
        public StatisticsSection(string title, string definition, params string[] columns) { Title = title; Definition = definition; Columns = columns; }
    }
    public sealed class StatisticsChart
    {
        public string Title, Unit;
        public bool Lines, Stacked;
        public readonly List<StatisticsSeries> Series = new List<StatisticsSeries>();
    }
    public sealed class StatisticsSeries
    {
        public string Label;
        public PlayerColor Player;
        public readonly List<StatisticsPoint> Points = new List<StatisticsPoint>();
    }
    public sealed class StatisticsPoint
    {
        public double X, Y;
        public string Label, Detail, Search;
        public int TargetPage = -1;
        public string Category;
    }
    public static class StatisticsPages
    {
        public static readonly string[] Names = { "Overview", "Scoring", "Industries", "Networks & Geography", "Economy", "Resources & Trade", "Actions & Cards", "Turn Order & Tempo", "Efficiency & Transition", "Timeline" };
        public static readonly int[] NavigationOrder = {0,1,2,3,4,5,6,7,8,9};
        public static int NormalizePage(int page) {return page>=0&&page<Names.Length?page:0;}
        public static readonly string[] IndustryTypes = { "CottonMill", "Manufacturer", "Pottery", "Brewery", "CoalMine", "IronWorks" };
        public static readonly string[] ActionTypes = { "Build", "Network", "Develop", "Sell", "Loan", "Scout", "Pass", "Unclassified" };
        public static List<StatisticsSection> Build(int page, CompletedMatchStatistics report, StatisticsFilters filters)
        {
            var sections = new List<StatisticsSection>();
            if (page == 0) Overview(report, filters, sections);
            if (!report.CompleteHistory) {
                sections.Add(new StatisticsSection("Detailed statistics unavailable", report.Availability + "\nMissing history is not zero. The final result above remains authoritative."));
                return sections;
            }
            switch (page) {
                case 1: Scoring(report, filters, sections); break;
                case 2: Industries(report, filters, sections); break;
                case 3: Networks(report, filters, sections); break;
                case 4: Economy(report, filters, sections); break;
                case 5: Resources(report, filters, sections); break;
                case 6: Actions(report, filters, sections); break;
                case 7: Tempo(report, filters, sections); break;
                case 8: Efficiency(report, filters, sections); break;
                case 9: Timeline(report, filters, sections); break;
            }
            if (!String.IsNullOrWhiteSpace(filters.Search))
                foreach (var section in sections) section.Rows.RemoveAll(row => !(String.Join(" ", row.Cells) + " " + row.Detail).ToLowerInvariant().Contains(filters.Search.Trim().ToLowerInvariant()));
            return sections;
        }
        static IEnumerable<StatisticsPlayer> Players(CompletedMatchStatistics r, StatisticsFilters f) { return r.Players.Where(p => f.Players.Contains(p.Color)); }
        static IEnumerable<StatisticsEntry> Entries(CompletedMatchStatistics r, StatisticsFilters f) { return r.Entries.Where(e => f.Includes(e.Player, e.Era)); }
        static IEnumerable<StatisticsAction> Frames(CompletedMatchStatistics r, StatisticsFilters f) { return r.Actions.Where(a => f.Includes(a.Player, a.Era)); }
        static string N(int value) { return value.ToString(CultureInfo.InvariantCulture); }
        static string F(double value) { return value.ToString("F2", CultureInfo.InvariantCulture); }
        static string M(int value) { return StatisticsFormat.Money(value); }
        static StatisticsSection Section(List<StatisticsSection> output, string title, string definition, params string[] columns)
        { var section = new StatisticsSection(title, definition, columns); output.Add(section); return section; }
        static StatisticsRow Row(StatisticsSection section, string key, PlayerColor? player, string detail, params string[] values)
        { var row = new StatisticsRow(key, detail, values) { Player = player }; section.Rows.Add(row); return row; }
        static int Score(CompletedMatchStatistics r, PlayerColor p, int era, string category)
        { return r.Entries.Where(e => e.Player == p && e.Era == era && e.Kind == "Balance" && e.Category == category).Sum(e => e.VP); }
        static string AssetDetail(StatisticsAsset a)
        {
            return a.Position + "\n" + a.City + " · " + a.Type + " level " + a.Level + "\nBuilt revision " + a.Revision +
                "\n" + (a.Flipped ? "Flipped after " + (a.FlipTurn - a.Turn) + " turns · revision " + a.FlipRevision : "Remained unflipped") +
                "\nIncome marker gain: " + a.IncomeSpaces + " spaces\nConstruction " + M(a.BaseCost) + "; bought resources " + M(a.MarketCost) +
                "\nResources used: coal " + a.CoalUsed + ", iron " + a.IronUsed +
                "\nProduced " + a.Produced + "; supplied to actions " + a.Supplied + "; market receipts " + M(a.MarketReceipts) +
                "\n" + (a.Remaining ? "Still on the board" : a.Removal + " · revision " + a.RemovedRevision) + "; stock " + a.Stock +
                "\nActual industry VP: Canal " + a.CanalVP + ", Rail " + a.RailVP + ", introductory addition " + a.IntroVP;
        }
        static void Overview(CompletedMatchStatistics r, StatisticsFilters f, List<StatisticsSection> output)
        {
            bool draw = r.Players.Count(p => p.Rank == 1) > 1;
            var result = Section(output, "Final result" + (draw ? " · Draw" : ""), "Final ranking is independent of era filters. Ties compare income level, then cash. Era totals already include merchant points and penalties.", "Rank", "Player", "Final VP", "Canal", "Rail", "Merchant VP", "VP lost", "Final cash", "Final income");
            foreach (var p in Players(r, f).OrderBy(p => p.Rank)) {
                int canal = r.Entries.Where(e => e.Kind == "Balance" && e.Player == p.Color && e.Era == 1).Sum(e => e.VP);
                int rail = r.Entries.Where(e => e.Kind == "Balance" && e.Player == p.Color && e.Era == 2).Sum(e => e.VP);
                var row = Row(result, "result-" + p.Color, p.Color, p.Opponent + " · " + p.Avatar + "\nTie-break values: income " + p.FinalIncome + ", cash " + M(p.FinalCash),
                    N(p.Rank), p.Name, N(p.FinalVP), r.CompleteHistory ? N(canal) : "—", r.CompleteHistory && !r.Introductory ? N(rail) : "—",
                    r.CompleteHistory ? N(Score(r,p.Color,1,"Merchant")+Score(r,p.Color,2,"Merchant")) : "—",
                    r.CompleteHistory ? N(-r.Entries.Where(e=>e.Player==p.Color&&e.Kind=="Balance"&&e.Category=="Penalty").Sum(e=>e.VP)) : "—", M(p.FinalCash), N(p.FinalIncome));
                row.TargetPage = 1;
            }
            if (!r.CompleteHistory) return;
            AddScoreChart(r, f, output);
            var activity = Section(output, "Activity", "Distinct built/flipped tiles and placed links. Click a player row for their actions.", "Player", "Built", "Flipped", "Links", "Loans", "Goods sold", "Actions");
            foreach (var p in Players(r,f)) {
                var assets = r.Assets.Where(a=>f.Includes(a.Player,a.Era)&&a.Player==p.Color).ToArray();
                int flipped=r.Assets.Count(a=>a.Player==p.Color&&a.Flipped&&(f.Era==0||a.FlipEra==f.Era));
                var actions = Frames(r,f).Where(a=>a.Player==p.Color).ToArray();
                Row(activity,"activity-"+p.Color,p.Color,"Builds and flips are counted in the era in which they occurred. Completed actions include Pass.",p.Name,N(assets.Length),N(flipped),
                    N(r.Links.Count(l=>l.Player==p.Color&&f.Includes(l.Player,l.Era))),N(actions.Count(a=>a.Kind=="Loan")),N(actions.Sum(a=>a.Goods)),N(actions.Length)).TargetPage=6;
            }
            if(r.Warnings.Count>0) Section(output,"History notes",String.Join("\n",r.Warnings));
        }
        static void AddScoreChart(CompletedMatchStatistics r, StatisticsFilters f, List<StatisticsSection> output)
        {
            var section=Section(output,"Score composition","Awarded VP by era/category. Penalties extend below zero; flipping alone is not a VP award.");
            var chart=section.Chart=new StatisticsChart{Title="Awarded VP",Unit="VP",Stacked=true};
            foreach(var p in Players(r,f)) {
                var series=new StatisticsSeries{Label=p.Name,Player=p.Color};
                foreach(int era in r.Introductory?new[]{1}:new[]{1,2}) {
                    if(f.Era!=0&&f.Era!=era)continue;
                    foreach(string category in new[]{"Industries","Links","Merchant","Penalty","Introductory"}) {
                        int value=Score(r,p.Color,era,category);
                        if(value==0)continue;
                        series.Points.Add(new StatisticsPoint{X=era,Y=value,Label=(era==1?"Canal":"Rail")+" · "+category,Category=category,Detail=p.Name+" · "+value+" VP",TargetPage=1});
                    }
                }
                chart.Series.Add(series);
            }
        }
        static void Scoring(CompletedMatchStatistics r, StatisticsFilters f, List<StatisticsSection> output)
        {
            AddScoreChart(r,f,output);
            var scores=Section(output,"Era / category ledger","Each category sums actual balance changes once. Game-ended summaries are validation, not extra awards.","Player","Era","Industry","Links","Merchant","Penalty","Introductory","Net VP");
            foreach(var p in Players(r,f)) foreach(int era in r.Introductory?new[]{1}:new[]{1,2}) {
                if(f.Era!=0&&f.Era!=era)continue;
                Row(scores,"score-"+p.Color+"-"+era,p.Color,"Final VP reconciles to opening VP plus net awarded VP.",p.Name,era==1?"Canal":"Rail",
                    N(Score(r,p.Color,era,"Industries")),N(Score(r,p.Color,era,"Links")),N(Score(r,p.Color,era,"Merchant")),N(Score(r,p.Color,era,"Penalty")),N(Score(r,p.Color,era,"Introductory")),
                    N(r.Entries.Where(e=>e.Player==p.Color&&e.Era==era&&e.Kind=="Balance").Sum(e=>e.VP)));
            }
            var assets=Section(output,"Industry scoring ledger","One asset can earn points in both eras. A removed tile retains previously awarded points.","Player","City / industry","Level","Canal VP","Rail VP","Intro VP");
            foreach(var a in r.Assets.Where(a=>f.Players.Contains(a.Player)&&(f.Era==0||f.Era==1&&a.CanalPrinted>0||f.Era==2&&a.RailPrinted>0)))
                Row(assets,a.Key,a.Player,AssetDetail(a),r.Name(a.Player),a.City+" · "+a.Type,N(a.Level),N(a.CanalVP),N(a.RailVP),N(a.IntroVP));
            LinkLedger(r,f,output);
            var changes=Section(output,"Bonus / penalty awards","Recorded VP deltas; merchant cash/income/development do not become invented VP.","Player","When","Category","VP");
            foreach(var e in Entries(r,f).Where(e=>e.Kind=="Balance"&&(e.Category=="Merchant"||e.Category=="Penalty"||e.Category=="Introductory")))
                Row(changes,"award-"+e.Revision+"-"+e.Player,e.Player,e.Detail,r.Name(e.Player),e.Position,e.Label,N(e.VP));
        }
        static void Industries(CompletedMatchStatistics r, StatisticsFilters f, List<StatisticsSection> output)
        {
            var summary=Section(output,"Industry conversion","Build cohorts use the selected era. Flip counts require flipping within that era; whole game includes later flips.","Player","Industry","Built","Flipped","Flip rate","Canal VP","Rail VP","Developed L1–2");
            foreach(var p in Players(r,f)) foreach(string type in IndustryTypes) {
                var all=r.Assets.Where(a=>a.Player==p.Color&&a.Type==type).ToArray();
                var built=all.Where(a=>f.Era==0||a.Era==f.Era).ToArray();
                int flips=built.Count(a=>a.Flipped&&(f.Era==0||a.FlipEra==f.Era));
                int low=Entries(r,f).Count(e=>e.Player==p.Color&&e.Kind=="Development"&&e.Resource==type&&e.Quantity<=2);
                Row(summary,"type-"+p.Color+type,p.Color,"Scored VP is filtered by scoring era, independently of build cohort.",p.Name,type,N(built.Length),N(flips),StatisticsFormat.Percent(flips,built.Length),N(f.Era==2?0:all.Sum(a=>a.CanalVP)),N(f.Era==1?0:all.Sum(a=>a.RailVP)),N(low));
            }
            var ledger=Section(output,"Industry activity ledger","Expand for resource stock, spending, income, carryover and removal reason.","Player","City","Industry","Level","Built","Flip lag","State");
            foreach(var a in r.Assets.Where(a=>f.Players.Contains(a.Player)&&(f.Era==0||a.Era==f.Era||a.Carried&&f.Era==2)))
                Row(ledger,a.Key,a.Player,AssetDetail(a),r.Name(a.Player),a.City,a.Type,N(a.Level),a.Position,a.Flipped?N(a.FlipTurn-a.Turn)+" turns":"—",a.Remaining?(a.Flipped?"Flipped":"Unflipped"):a.Removal);
            var development=Section(output,"Development ledger","Merchant development is part of a sale, with no additional Develop action.","Player","When","Source","Tile removed");
            foreach(var e in Entries(r,f).Where(e=>e.Kind=="Development"))Row(development,"dev-"+e.Revision,e.Player,e.Detail,r.Name(e.Player),e.Position,e.Category,e.Label);
        }
        static void LinkLedger(CompletedMatchStatistics r, StatisticsFilters f, List<StatisticsSection> output)
        {
            var ledger=Section(output,"Link scoring ledger","Scores use the board at that era's scoring boundary, including every connected region. Expand a row for endpoint contributions.","Player","Era","Route","VP","Own tiles","Other tiles","Merchants");
            foreach(var l in r.Links.Where(l=>f.Includes(l.Player,l.Era)))
                Row(ledger,l.Key,l.Player,l.Position+"\n"+l.Contributions,r.Name(l.Player),l.Era==1?"Canal":"Rail",l.Route,l.Scored?N(l.VP):"—",N(l.OwnVP),N(l.OthersVP),N(l.MerchantVP));
        }
        static void Networks(CompletedMatchStatistics r, StatisticsFilters f, List<StatisticsSection> output)
        {
            var summary=Section(output,"Network outcomes","A double rail is one Network action and two links. Zero-point links remain in averages.","Player","Era","Network actions","Links","Double rails","Link VP","VP / link");
            foreach(var p in Players(r,f))foreach(int era in r.Introductory?new[]{1}:new[]{1,2}) {
                if(f.Era!=0&&f.Era!=era)continue;
                var frames=r.Actions.Where(a=>a.Player==p.Color&&a.Era==era&&a.Kind=="Network").ToArray();
                var links=r.Links.Where(l=>l.Player==p.Color&&l.Era==era).ToArray();
                Row(summary,"network-"+p.Color+era,p.Color,"Only actual link VP; no hypothetical route value.",p.Name,era==1?"Canal":"Rail",N(frames.Length),N(links.Length),N(frames.Count(a=>a.Links==2)),N(links.Sum(l=>l.VP)),StatisticsFormat.Ratio(links.Sum(l=>l.VP),links.Length));
            }
            LinkLedger(r,f,output);
            var cities=Section(output,"City occupancy snapshots","Density = owned / all slots. Ownership share = owned / occupied slots. These are board snapshots, not a historical build count.","Player","Boundary","City","Owned","Slots","Occupied","Density","Share");
            foreach(var c in r.Cities.Where(c=>f.Includes(c.Player,c.Era)))
                Row(cities,"city-"+c.Player+c.Era+c.Boundary+c.City,c.Player,"Snapshot: "+(c.Era==1?"Canal":"Rail")+" · "+c.Boundary,r.Name(c.Player),(c.Era==1?"Canal":"Rail")+" · "+c.Boundary,c.City,N(c.Owned),N(c.Slots),N(c.Occupied),StatisticsFormat.Percent(c.Owned,c.Slots),StatisticsFormat.Percent(c.Owned,c.Occupied));
            var reliance=Section(output,"Recorded connection reliance","Opponent links were required for these recorded endpoints. Another supplier/destination might have been possible. Iron and own-beer use are excluded.","Player","Purpose","From","To","Opponent links required");
            foreach(var c in r.Connections.Where(c=>f.Includes(c.Player,c.Era)))
                Row(reliance,"connection-"+c.Revision+"-"+c.Purpose,c.Player,c.Position,r.Name(c.Player),c.Purpose,c.From,c.To,c.OpponentRequired?"Yes":"No");
            var competition=Section(output,"Slot competition","Measured capacity changes and actual overbuilds, without assuming an opponent's intention.","Player","When","City","Industry","Event");
            foreach(var a in r.Assets.Where(a=>f.Includes(a.Player,a.Era)&&(a.LastSlot||a.OpponentOverbuild)))
                Row(competition,a.Key+"-competition",a.Player,AssetDetail(a),r.Name(a.Player),a.Position,a.City,a.Type,a.OpponentOverbuild?"Opponent overbuild":"Last vacant matching slot");
        }
        static void Economy(CompletedMatchStatistics r, StatisticsFilters f, List<StatisticsSection> output)
        {
            LineChart(r,f,output,"Cash through the match","£",s=>s.Cash);
            LineChart(r,f,output,"Income levels through the match","Income level",s=>s.Income);
            var totals=Section(output,"Cash / loan accounting","Opening cash + actual receipts − payments reconciles to final cash. Market spending is already included in action spending.","Player","Borrowed","Loans","Loan levels lost","Action spend","Market cost","Income paid","Income debt","Market receipts");
            foreach(var p in Players(r,f)) {
                var frames=Frames(r,f).Where(a=>a.Player==p.Color).ToArray(); var entries=Entries(r,f).Where(e=>e.Player==p.Color&&e.Kind=="Balance").ToArray();
                Row(totals,"economy-"+p.Color,p.Color,"Final cash "+M(p.FinalCash)+"; opening cash "+M(p.OpeningCash)+". Cash movements below are counted once.",p.Name,M(frames.Sum(a=>a.Borrowed)),N(frames.Count(a=>a.Kind=="Loan")),N(frames.Sum(a=>a.IncomeDrop)),
                    M(frames.Sum(a=>a.Spending)),M(frames.Sum(a=>a.MarketCost)),M(entries.Where(e=>e.Category=="Income payment"&&e.Cash>0).Sum(e=>e.Cash)),M(-entries.Where(e=>e.Category=="Income payment"&&e.Cash<0).Sum(e=>e.Cash)),
                    M(entries.Where(e=>e.Category=="Market receipt").Sum(e=>e.Cash)));
            }
            var ledger=Section(output,"Cash and income movements","Only actual balance changes. No income payment is added after the final Rail round.","Player","When","Movement","Cash Δ","Income Δ","Marker spaces Δ");
            foreach(var e in Entries(r,f).Where(e=>e.Kind=="Balance"&&(e.Cash!=0||e.Income!=0||e.Marker!=0)))
                Row(ledger,"cash-"+e.Revision+"-"+e.Player,e.Player,e.Detail,r.Name(e.Player),e.Position,e.Label,M(e.Cash),N(e.Income),N(e.Marker));
        }
        static void LineChart(CompletedMatchStatistics r,StatisticsFilters f,List<StatisticsSection> output,string title,string unit,Func<StatisticsSample,int> value)
        {
            var section=Section(output,title,"Actual state at recorded operations. Hover/focus for the value; click to inspect its timeline. Turn 0 is setup.", "Player","When","Value","Event");
            var chart=section.Chart=new StatisticsChart{Title=title,Unit=unit,Lines=true};
            foreach(var p in Players(r,f)) {
                var series=new StatisticsSeries{Player=p.Color,Label=p.Name};
                foreach(var sample in r.Samples.Where(s=>s.Player==p.Color&&(f.Era==0||s.Era==f.Era))) {
                    series.Points.Add(new StatisticsPoint{X=sample.Turn,Y=value(sample),Label=sample.Position,Detail=p.Name+" · "+sample.Annotation+" · "+value(sample)+" "+unit,TargetPage=9,Search=sample.Annotation});
                    Row(section,"sample-"+sample.Revision+"-"+sample.Player,p.Color,sample.Annotation,p.Name,sample.Position,N(value(sample)),sample.Annotation);
                }
                chart.Series.Add(series);
            }
        }
        static void Resources(CompletedMatchStatistics r,StatisticsFilters f,List<StatisticsSection> output)
        {
            var summary=Section(output,"Resource sources and uses","Market buys are quantities/real payments; industry resources are free transfers. Merchant beer is distinct from brewery beer used in a sale.","Consumer","Resource","Own industry","Opponents","Market bought","Merchant","Sell use","Network use");
            foreach(var p in Players(r,f))foreach(string resource in new[]{"Coal","Iron","Beer"}) {
                var uses=Entries(r,f).Where(e=>e.Player==p.Color&&e.Kind=="Resource use"&&e.Resource==resource).ToArray();
                Row(summary,"resource-"+p.Color+resource,p.Color,"Supplier ownership is captured before depletion/removal.",p.Name,resource,
                    N(uses.Where(e=>e.Supplier==p.Color).Sum(e=>e.Quantity)),N(uses.Where(e=>e.Supplier.HasValue&&e.Supplier!=p.Color).Sum(e=>e.Quantity)),
                    N(Entries(r,f).Where(e=>e.Player==p.Color&&e.Kind=="Market purchase"&&e.Resource==resource).Sum(e=>e.Quantity)),
                    N(uses.Where(e=>e.Source=="Merchant").Sum(e=>e.Quantity)),N(uses.Where(e=>e.Category=="Sell").Sum(e=>e.Quantity)),N(uses.Where(e=>e.Category=="Network").Sum(e=>e.Quantity)));
            }
            var transfers=Section(output,"Supplier → consumer matrix","Quantity supplied to the recorded actor. Counts do not imply payments or transfer the producer's points.","Resource","Supplier","Consumer","Quantity");
            foreach(var group in Entries(r,f).Where(e=>e.Kind=="Resource use"&&e.Supplier.HasValue).GroupBy(e=>new{e.Resource,Owner=e.Supplier.Value,e.Player}))
                Row(transfers,"transfer-"+group.Key,group.Key.Player,String.Join("\n",group.Select(e=>e.Position+" · "+e.Label+" × "+e.Quantity)),group.Key.Resource,r.Name(group.Key.Owner),r.Name(group.Key.Player),N(group.Sum(e=>e.Quantity)));
            var beer=Entries(r,f).Where(e=>e.Kind=="Resource use"&&e.Resource=="Beer"&&e.Supplier.HasValue&&e.Supplier.Value!=e.Player).GroupBy(e=>e.Supplier.Value).OrderByDescending(g=>g.Sum(e=>e.Quantity)).ToArray();
            if(beer.Length>0)Section(output,"Most beer supplied to opponents",String.Join(" / ",beer.Where(g=>g.Sum(e=>e.Quantity)==beer[0].Sum(e=>e.Quantity)).Select(g=>r.Name(g.Key)))+" · "+beer[0].Sum(e=>e.Quantity)+" barrels. Defined by quantity, with ties.");
            var market=Section(output,"Market transactions and quote impact","Next-unit quotes before/after a transaction describe its immediate price effect. They do not prove a downstream cost to opponents.","Player","When","Resource","Operation","Quantity","Next quote before","After");
            foreach(var e in Entries(r,f).Where(e=>e.Kind=="Market purchase"||e.Kind=="Market sale"))
                Row(market,"market-"+e.Revision,e.Player,e.Detail,r.Name(e.Player),e.Position,e.Resource,e.Kind,N(e.Quantity),M(e.BeforeQuote),M(e.AfterQuote));
            var goods=Section(output,"Goods sales","Merchant beer records a destination; uniquely eligible locations are marked derived. Other sales remain unattributed.","Player","When","Industry / city","Merchant","Attribution");
            foreach(var e in Entries(r,f).Where(e=>e.Kind=="Goods sale"))
                Row(goods,"sale-"+e.Revision,e.Player,e.Detail,r.Name(e.Player),e.Position,e.Label,e.Destination,e.Destination=="Unattributed"?"Not recorded":e.DestinationDerived?"Derived":"Recorded beer");
            var merchants=Section(output,"Merchant benefits","Cash, income spaces/levels, VP and development have separate units. No Free Network merchant bonus exists.","Player","Location / bonus","Uses","Cash","Marker spaces","Income levels","VP");
            foreach(var group in Entries(r,f).Where(e=>e.Kind=="Merchant bonus").GroupBy(e=>new{e.Player,e.Destination,e.Category}))
                Row(merchants,"bonus-"+group.Key,group.Key.Player,"Development details appear under Industries; this does not count another action.",r.Name(group.Key.Player),group.Key.Destination+" · "+group.Key.Category,N(group.Count()),M(group.Sum(e=>e.Cash)),N(group.Sum(e=>e.Marker)),N(group.Sum(e=>e.Income)),N(group.Sum(e=>e.VP)));
            var usage=Section(output,"Resource-use ledger","Expand to see source ownership, purpose and destination.","Consumer","When","Resource","Supplier","Purpose","Quantity");
            foreach(var e in Entries(r,f).Where(e=>e.Kind=="Resource use"))
                Row(usage,"use-"+e.Revision,e.Player,e.Detail+"\n"+e.Label+"\nDestination: "+e.Destination,r.Name(e.Player),e.Position,e.Resource,e.Supplier.HasValue?r.Name(e.Supplier.Value):e.Source,e.Category,N(e.Quantity));
        }
        static void Actions(CompletedMatchStatistics r,StatisticsFilters f,List<StatisticsSection> output)
        {
            var summary=Section(output,"Completed actions","Six core actions plus Pass. Percentages include Pass. Scout's extra card trades are not extra actions.","Player","Action","Count","Share","Developed / action","Goods / action","Links / action");
            var chart=summary.Chart=new StatisticsChart{Title="Action distribution",Unit="Actions"};
            foreach(var p in Players(r,f)) {
                var frames=Frames(r,f).Where(a=>a.Player==p.Color).ToArray(); var series=new StatisticsSeries{Player=p.Color,Label=p.Name};
                for(int i=0;i<ActionTypes.Length;i++) {
                    string type=ActionTypes[i]; var category=frames.Where(a=>a.Kind==type).ToArray(); if(type=="Unclassified"&&category.Length==0)continue;
                    Row(summary,"actions-"+p.Color+type,p.Color,String.Join("\n",category.Select(a=>a.Position)),p.Name,type,N(category.Length),StatisticsFormat.Percent(category.Length,frames.Length),
                        StatisticsFormat.Ratio(category.Sum(a=>a.Developed),category.Length),StatisticsFormat.Ratio(category.Sum(a=>a.Goods),category.Length),StatisticsFormat.Ratio(category.Sum(a=>a.Links),category.Length)).TargetPage=9;
                    series.Points.Add(new StatisticsPoint{X=i,Y=category.Length,Label=type,Detail=p.Name+" · "+category.Length+" actions",TargetPage=9,Search=type});
                }
                chart.Series.Add(series);
            }
            var cards=Section(output,"Card purpose distribution","Build authorization, non-Build action payment and Scout's extra trades are counted separately. Setup discard and reshuffles are excluded.","Player","Purpose","Card type","Count");
            foreach(var group in Entries(r,f).Where(e=>e.Kind=="Card").GroupBy(e=>new{e.Player,e.Category,e.Label}))
                Row(cards,"cards-"+group.Key,group.Key.Player,"Card purpose is derived from its confirmed action.",r.Name(group.Key.Player),group.Key.Category,group.Key.Label,N(group.Count()));
            var wild=Section(output,"Wildcard conversion","Acquisition episodes are counted separately even when the same card returns to the pool. Uses show actual flexibility, not an asserted turnaround.","Player","Acquired","Build uses","Other payments","Unspent episodes");
            foreach(var p in Players(r,f)) {
                int acquired=Entries(r,f).Count(e=>e.Player==p.Color&&e.Kind=="Wildcard acquired");
                var used=Entries(r,f).Where(e=>e.Player==p.Color&&e.Kind=="Card"&&e.Label.StartsWith("Wild")).ToArray();
                Row(wild,"wild-"+p.Color,p.Color,"Era-filtered counts show acquisitions and uses in that era. Unspent episodes are shown for the whole match.",p.Name,N(acquired),N(used.Count(e=>e.Category=="Build")),N(used.Count(e=>e.Category!="Build")),f.Era==0?N(acquired-used.Length):"—");
            }
            var ledger=Section(output,"Wildcard-use ledger","A feasibility note refers to this same placement and the remaining ordinary cards in hand.","Player","When","Card","Purpose");
            foreach(var e in Entries(r,f).Where(e=>e.Kind=="Card"&&e.Label.StartsWith("Wild")))
                Row(ledger,"wild-use-"+e.Revision,e.Player,e.Detail,r.Name(e.Player),e.Position,e.Label,e.Category);
        }
        static void Tempo(CompletedMatchStatistics r,StatisticsFilters f,List<StatisticsSection> output)
        {
            var seats=Section(output,"Round positions","Round percentages use known turns; nonexistent seats are omitted.","Player","Position","Rounds","Share");
            var rounds=StatisticsChartQueries.Rounds(r,f).ToArray();
            foreach(var p in Players(r,f))for(int seat=0;seat<r.Players.Count;seat++) {
                int count=rounds.Count(x=>x.Order.Count>seat&&x.Order[seat]==p.Color);
                Row(seats,"seat-"+p.Color+seat,p.Color,"Recorded round order, unaffected by temporary debt-choice ownership.",p.Name,N(seat+1),N(count),StatisticsFormat.Percent(count,rounds.Count(x=>x.Order.Contains(p.Color))));
            }
            var spend=Section(output,"Spending tempo","Action spending only. Volatility is population standard deviation; relative volatility = deviation / mean. No fixed strategy grade.","Player","Mean","Median","Min","Max","Std. deviation","Relative","Zero rounds");
            foreach(var p in Players(r,f)) {
                var values=rounds.Select(x=>(double)x.Spent(p.Color)).OrderBy(x=>x).ToArray(); if(values.Length==0)continue;
                double mean=values.Average(), median=(values[(values.Length-1)/2]+values[values.Length/2])/2, deviation=Math.Sqrt(values.Average(x=>(x-mean)*(x-mean)));
                Row(spend,"tempo-"+p.Color,p.Color,"Spend ranges and volatility are descriptive, without arbitrary steady/volatile thresholds.",p.Name,F(mean),F(median),F(values.First()),F(values.Last()),F(deviation),StatisticsFormat.Ratio(deviation,mean),N(values.Count(x=>x==0)));
            }
            var order=Section(output,"Round order and spending","Round N spending determines round N+1 order. Equal-spend ties retain order.","Era / round","Turn order","Spending");
            foreach(var x in rounds)Row(order,"round-"+x.Era+x.Round,null,"Round order is recorded at round start.",(x.Era==1?"Canal":"Rail")+" "+x.Round,
                String.Join(" → ",x.Order.Select(r.Name)),String.Join(" · ",x.Order.Where(f.Players.Contains).Select(p=>r.Name(p)+" "+M(x.Spent(p)))));
            var frequency=Section(output,"Back-to-back frequency","Opportunities count adjacent rounds within the selected era. Canal-to-Rail boundaries are reported separately because maintenance intervenes.","Player","Within-era doubles","Opportunities","Frequency","Era-transition doubles");
            foreach(var p in Players(r,f)) {
                int opportunities=0,doubles=0,transition=0;
                for(int i=1;i<r.Rounds.Count;i++) {
                    var a=r.Rounds[i-1];var b=r.Rounds[i];
                    if(!rounds.Contains(b)||!a.Order.Contains(p.Color)||!b.Order.Contains(p.Color))continue;
                    bool consecutive=a.Order.Count>0&&b.Order.Count>0&&a.Order.Last()==p.Color&&b.Order.First()==p.Color;
                    if(a.Era==b.Era){opportunities++;if(consecutive)doubles++;}else if(consecutive)transition++;
                }
                Row(frequency,"double-rate-"+p.Color,p.Color,"Observed turns, with no inferred tactical intent.",p.Name,N(doubles),N(opportunities),StatisticsFormat.Percent(doubles,opportunities),N(transition));
            }
            var back=Section(output,"Back-to-back turn sequences","Last in one round, first in the next. Era transitions are labeled separately; no intentional combo is assumed.","Player","Boundary","Actions in previous turn","Next turn","Eligible boundary");
            for(int i=1;i<r.Rounds.Count;i++) {
                var previous=r.Rounds[i-1]; var next=r.Rounds[i];
                if(previous.Order.Count==0||next.Order.Count==0||previous.Order.Last()!=next.Order.First())continue;
                var p=next.Order.First(); if(!f.Players.Contains(p)||!rounds.Contains(next))continue;
                var left=r.Actions.Where(a=>a.Player==p&&a.Era==previous.Era&&a.Round==previous.Round).ToArray();
                var right=r.Actions.Where(a=>a.Player==p&&a.Era==next.Era&&a.Round==next.Round).ToArray();
                string label=previous.Era==next.Era?"Round "+previous.Round+" → "+next.Round:"Canal → Rail transition";
                Row(back,"double-"+i,p,String.Join("\n",left.Concat(right).Select(a=>a.Position+"\n"+a.Details)),r.Name(p),label,String.Join(" → ",left.Select(a=>a.Kind)),String.Join(" → ",right.Select(a=>a.Kind)),previous.Era==next.Era?"Within era":"Maintenance intervenes");
            }
        }
        static void Efficiency(CompletedMatchStatistics r,StatisticsFilters f,List<StatisticsSection> output)
        {
            output.Add(StatisticsChartQueries.EfficiencyLedger(r,f));
            var cleanup=Section(output,"Level-1 Canal cleanup","Link points score before cleanup. Removed link icons do not subtract previously earned Canal VP.","Player","City / industry","Flipped","Canal VP earned","Stock removed","Link icons removed");
            foreach(var a in r.Assets.Where(a=>f.Players.Contains(a.Player)&&a.Removal=="Canal cleanup"&&(f.Era==0||f.Era==1)))
                Row(cleanup,a.Key+"-cleanup",a.Player,AssetDetail(a),r.Name(a.Player),a.City+" · "+a.Type,a.Flipped?"Yes":"No",N(a.CanalVP),N(a.Stock),N(a.Flipped?a.LinkIcons:0));
            var carry=Section(output,"Canal carryover conversion","Actual tiles surviving Canal maintenance and their later outcome.","Player","City / industry","Level","Rail VP","Outcome");
            foreach(var a in r.Assets.Where(a=>f.Players.Contains(a.Player)&&a.Carried))
                Row(carry,a.Key+"-carry",a.Player,AssetDetail(a),r.Name(a.Player),a.City+" · "+a.Type,N(a.Level),N(a.RailVP),a.Remaining?(a.Flipped?"Flipped":"Unflipped"):a.Removal);
            var remaining=Section(output,"Investment remaining unflipped","These costs are investment, not proven waste. A partly depleted supplier may have produced useful resources or market cash.","Player","City / industry","Printed VP","Base cash","Bought resources","Free/input units","Supplied","Market receipts");
            foreach(var a in r.Assets.Where(a=>f.Players.Contains(a.Player)&&a.Remaining&&!a.Flipped))
                Row(remaining,a.Key+"-unfinished",a.Player,AssetDetail(a),r.Name(a.Player),a.City+" · "+a.Type+" "+a.Level,N(a.PrintedVP),M(a.BaseCost),M(a.MarketCost),"Coal "+a.CoalUsed+" / Iron "+a.IronUsed,N(a.Supplied),M(a.MarketReceipts));
        }
        static void Timeline(CompletedMatchStatistics r,StatisticsFilters f,List<StatisticsSection> output)
        {
            LineChart(r,f,output,"Awarded VP through the match","VP",s=>s.VP);
            var actions=Section(output,"Action timeline","Expand a confirmed action for its actual ordered operations and resource suppliers. No live-board rewind occurs.","Player","When","Action","Spending","Loans","Goods","Links");
            foreach(var a in Frames(r,f))Row(actions,a.Key,a.Player,a.Details+"\nCards: "+a.Cards+"\nRevisions "+a.Revision+"–"+a.EndRevision,r.Name(a.Player),a.Position,a.Kind,M(a.Spending),M(a.Borrowed),N(a.Goods),N(a.Links));
            var phases=Section(output,"Scoring / settlement and state changes","Phase rows are distinct from completed actions. Revision identifies the exact recorded operation.","Player","Revision","When","Event","Cash Δ","Income Δ","VP Δ");
            foreach(var e in Entries(r,f).Where(e=>e.Kind=="Balance"))
                Row(phases,"event-"+e.Revision+"-"+e.Player,e.Player,e.Detail,r.Name(e.Player),N(e.Revision),e.Position,e.Label,M(e.Cash),N(e.Income),N(e.VP));
        }
    }
}
