using System;
using System.Collections.Generic;
using System.Linq;
using BoardGameRules.Entities.Players;
namespace BrassPatch.Statistics
{
    public sealed class StatisticsEfficiencyTotal
    {
        public StatisticsPlayer Player;
        public int VP,Actions,Spending,IndustryVP,Printed;
        public double VPPerAction { get {return Actions>0?(double)VP/Actions:Double.NaN;} }
        public double CostPerVP { get {return VP>0?(double)Spending/VP:Double.NaN;} }
        public StatisticsDrill ActionEvidence,CostEvidence;
    }
    public static class StatisticsChartQueries
    {
        public static string EntryKey(StatisticsEntry e) {return "entry-"+e.Revision+"-"+e.Player;}
        public static string RecordRowKey(StatisticsEntry e) {return e.Kind=="Wildcard acquired"&&e.AssetKey!=null?e.AssetKey:EntryKey(e)+"/"+e.Kind+"/"+e.Label+"/"+e.RecordOrdinal;}
        public static string RoundKey(StatisticsRound round) {return "round-"+round.Era+"-"+round.Round;}
        public static IEnumerable<StatisticsRound> Rounds(CompletedMatchStatistics r,StatisticsFilters f) {return r.Rounds.Where(t=>(f.Era==0||f.Era==t.Era)&&Evidence(f,RoundKey(t)));}
        public static bool Evidence(StatisticsFilters f,string key) {return !f.EvidenceRestricted&&f.Evidence.Count==0||f.Evidence.Contains(key??"");}
        public static IEnumerable<StatisticsLink> Links(CompletedMatchStatistics r,StatisticsFilters f)
        {
            return r.Links.Where(l=>f.Includes(l.Player,l.Era)&&Evidence(f,l.Key)&&(f.Region==""||r.Contributions.Any(c=>c.LinkKey==l.Key&&c.RegionColor==f.Region)));
        }
        public static IEnumerable<StatisticsEfficiencyTotal> Efficiency(CompletedMatchStatistics r,StatisticsFilters f)
        {
            // Detail filters cannot allocate a player's total score to a subset of actions.
            foreach(var p in r.Players.Where(p=>f.Players.Contains(p.Color))) {
                var actions=r.Actions.Where(a=>a.Player==p.Color&&(f.Era==0||a.Era==f.Era)).ToArray();
                var balances=r.Entries.Where(e=>e.Player==p.Color&&(f.Era==0||e.Era==f.Era)&&e.Kind=="Balance").ToArray();
                var t=new StatisticsEfficiencyTotal{Player=p,VP=f.Era==0?p.FinalVP:balances.Sum(e=>e.VP),Actions=actions.Length,Spending=f.AllCashPayments?-balances.Where(e=>e.Cash<0).Sum(e=>e.Cash):actions.Sum(a=>a.Spending)};
                t.IndustryVP=r.Assets.Where(a=>a.Player==p.Color).Sum(a=>StatisticsMetrics.AssetVP(a,f.Era));
                t.Printed=r.Assets.Where(a=>a.Player==p.Color).Sum(a=>(f.Era==2?0:a.CanalPrinted)+(f.Era==1?0:a.RailPrinted));
                t.ActionEvidence=new StatisticsDrill{Page=8,Player=p.Color,Era=f.Era,HasKeys=true,AllCashPayments=f.AllCashPayments};
                t.CostEvidence=new StatisticsDrill{Page=8,Player=p.Color,Era=f.Era,HasKeys=true,AllCashPayments=f.AllCashPayments};
                foreach(var a in actions){t.ActionEvidence.Keys.Add(a.Key);t.CostEvidence.Keys.Add(a.Key);}
                foreach(var e in balances.Where(e=>e.VP!=0||f.AllCashPayments&&e.Cash<0)){t.ActionEvidence.Keys.Add(EntryKey(e));t.CostEvidence.Keys.Add(EntryKey(e));}
                yield return t;
            }
        }
        public static StatisticsSection EfficiencyLedger(CompletedMatchStatistics r,StatisticsFilters f)
        {
            var s=new StatisticsSection("Conversion and efficiency","Player/era totals; industry, action, region and evidence filters apply to the tile plots, not these ratios. Market costs are included once. Printed opportunity is nominal, not an optimal score.","Player","VP / action","£ spent / VP","Industry VP","Printed opportunity","Realization");
            foreach(var t in Efficiency(r,f))s.Rows.Add(new StatisticsRow("ratio-"+t.Player.Color,"VP "+t.VP+"; confirmed actions "+t.Actions+"; "+(f.AllCashPayments?"all negative cash payments ":"action spending ")+StatisticsFormat.Money(t.Spending)+". Printed opportunity sums per-era snapshots, including carryover.",StatisticsMetrics.PlayerName(r,t.Player),Number(t.VPPerAction),Number(t.CostPerVP),t.IndustryVP.ToString(),t.Printed.ToString(),StatisticsFormat.Percent(t.IndustryVP,t.Printed)){Player=t.Player.Color,Drill=t.CostEvidence});
            return s;
        }
        public static string Number(double value){return Double.IsNaN(value)?"—":value.ToString("0.00",System.Globalization.CultureInfo.InvariantCulture);}
        public static IEnumerable<StatisticsAsset> Carryover(CompletedMatchStatistics r,StatisticsFilters f)
        {return r.Assets.Where(a=>a.Carried&&f.Players.Contains(a.Player)&&(f.Industry==""||f.Industry==a.Type)&&(f.Region==""||f.Region==a.RegionColor)&&Evidence(f,a.Key));}
        public static IEnumerable<StatisticsEntry> Purchases(CompletedMatchStatistics r,StatisticsFilters f,string resource)
        {
            var deliveries=StatisticsMetrics.Flow(r,ForResource(f,resource)).Where(e=>e.Source=="Market").ToArray();
            return r.Entries.Where(e=>e.Kind=="Market purchase"&&e.Resource==resource&&f.Includes(e.Player,e.Era)&&
                (f.Action==""||r.Actions.Any(a=>a.Key==e.ActionKey&&a.Kind==f.Action))&&
                (f.Industry==""&&f.Region=="")&&
                (Evidence(f,EntryKey(e))||f.Evidence.Contains(e.ActionKey??"")||deliveries.Any(d=>d.Player==e.Player&&d.ActionKey!=null&&d.ActionKey==e.ActionKey)));
        }
        public static StatisticsFilters ForResource(StatisticsFilters f,string resource)
        {
            var result=new StatisticsFilters{Era=f.Era,Industry=f.Industry,Region=f.Region,Action=f.Action,Resource=resource,Outgoing=f.Outgoing,EvidenceRestricted=f.EvidenceRestricted};
            foreach(var p in f.Players)result.Players.Add(p);foreach(var key in f.Evidence)result.Evidence.Add(key);return result;
        }
        public static IEnumerable<StatisticsEntry> Development(CompletedMatchStatistics r,StatisticsFilters f)
        {
            return r.Entries.Where(e=>e.Kind=="Development"&&f.Includes(e.Player,e.Era)&&(f.Industry==""||e.Resource==f.Industry)&&(f.Action==""||r.Actions.Any(a=>a.Key==e.ActionKey&&a.Kind==f.Action))&&
                (f.DevelopmentSource=="All"||f.DevelopmentSource=="Merchant"&&e.Category=="Merchant"||f.DevelopmentSource=="Develop action"&&e.Category=="Develop action")&&
                (Evidence(f,EntryKey(e))||f.Evidence.Contains(e.ActionKey??"")));
        }
        public static StatisticsEntry WildUse(CompletedMatchStatistics r,StatisticsEntry episode)
        {return r.Entries.FirstOrDefault(e=>e.Kind=="Card"&&e.Player==episode.Player&&e.AssetKey==episode.AssetKey&&episode.AssetKey!=null);}
    }
}
