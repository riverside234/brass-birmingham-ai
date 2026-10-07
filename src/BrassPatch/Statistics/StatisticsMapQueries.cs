using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using BoardGameRules.Entities.Players;
namespace BrassPatch.Statistics
{
    // Derived presentation values; deliberately outside the schema-1 persistence contract.
    public sealed class StatisticsMapCity
    {
        public StatisticsLocation Location;
        public StatisticsAsset[] Tiles;
        public StatisticsLink[] Links;
        public int Slots, Occupied;
        public string Problem;
        public bool Available { get { return String.IsNullOrEmpty(Problem); } }
    }
    public sealed class StatisticsMapSnapshot
    {
        public int Era,Revision;
        public string Problem;
        public StatisticsMapCity[] Cities=new StatisticsMapCity[0];
        public StatisticsLink[] Links=new StatisticsLink[0];
        public bool Available { get { return String.IsNullOrEmpty(Problem); } }
        public bool Flipped(StatisticsAsset a) { return a.FlipRevision>0&&a.FlipRevision<=Revision; }
    }
    public static class StatisticsMapQueries
    {
        sealed class Pair { public readonly Dictionary<int,StatisticsMapSnapshot> Values=new Dictionary<int,StatisticsMapSnapshot>(); }
        static readonly ConditionalWeakTable<CompletedMatchStatistics,Pair> cache=new ConditionalWeakTable<CompletedMatchStatistics,Pair>();
        public static StatisticsMapSnapshot Get(CompletedMatchStatistics report,int era)
        {
            var pair=cache.GetOrCreateValue(report);StatisticsMapSnapshot result;
            if(!pair.Values.TryGetValue(era,out result)){result=Build(report,era);pair.Values[era]=result;}return result;
        }
        static StatisticsMapSnapshot Build(CompletedMatchStatistics r,int era)
        {
            var s=new StatisticsMapSnapshot{Era=era};
            if(!r.CompleteHistory){s.Problem=r.Availability??"Historical board unavailable.";return s;}
            if(era!=1&&era!=2||r.Introductory&&era==2){s.Problem="No Rail era in this match.";return s;}
            var boundaries=r.Boundaries.Where(b=>b.Name=="Scoring"&&b.Era==era).ToArray();
            if(boundaries.Length!=r.Players.Count||boundaries.Select(b=>b.Player).Distinct().Count()!=r.Players.Count||boundaries.Select(b=>b.Revision).Distinct().Count()!=1){s.Problem="The era scoring snapshot is unavailable or inconsistent.";return s;}
            s.Revision=boundaries[0].Revision;
            var tiles=r.Assets.Where(a=>a.Revision<=s.Revision&&(a.RemovedRevision==0||a.RemovedRevision>s.Revision)).ToArray();
            s.Links=r.Links.Where(l=>l.Era==era&&l.Revision<=s.Revision).ToArray();
            var names=new HashSet<string>(r.Locations.Select(l=>l.City));
            if(s.Links.Any(l=>l.Cities==null||l.Cities.Length<2||l.Cities.Length>3||l.Cities.Distinct().Count()!=l.Cities.Length||l.Cities.Any(c=>!names.Contains(c)))){s.Problem="Link endpoints are missing or inconsistent for this era.";return s;}
            s.Cities=r.Locations.Select(location=>{
                var city=new StatisticsMapCity{Location=location,Tiles=tiles.Where(a=>a.City==location.City).OrderBy(a=>a.Slot,StringComparer.Ordinal).ToArray(),Links=s.Links.Where(l=>l.Cities.Contains(location.City)).ToArray()};
                if(location.Merchant)return city;
                var capacities=r.Cities.Where(c=>c.City==location.City&&c.Era==era&&c.Boundary=="Scoring").ToArray();
                if(capacities.Length!=r.Players.Count||capacities.Select(c=>c.Player).Distinct().Count()!=r.Players.Count||capacities.Select(c=>c.Slots).Distinct().Count()!=1||capacities.Select(c=>c.Occupied).Distinct().Count()!=1){city.Problem="Slot snapshot unavailable.";return city;}
                city.Slots=capacities[0].Slots;city.Occupied=capacities[0].Occupied;
                if(city.Slots<0||city.Slots>4||city.Occupied<0||city.Occupied>city.Slots||city.Occupied!=city.Tiles.Length||city.Tiles.Any(a=>String.IsNullOrEmpty(a.Slot))||city.Tiles.Select(a=>a.Slot).Distinct().Count()!=city.Tiles.Length||capacities.Any(c=>c.Owned!=city.Tiles.Count(a=>a.Player==c.Player)))city.Problem="Industry occupancy does not reconcile with this snapshot.";
                return city;
            }).ToArray();return s;
        }
        public static int IndustryVP(StatisticsMapSnapshot s,StatisticsMapCity c,StatisticsFilters f)
        {return c.Tiles.Where(a=>f.Players.Contains(a.Player)).Sum(a=>s.Era==1?a.CanalVP:a.RailVP);}
        public static int ContributionVP(CompletedMatchStatistics r,StatisticsMapSnapshot s,StatisticsMapCity c,StatisticsFilters f)
        {var keys=new HashSet<string>(c.Links.Where(l=>f.Players.Contains(l.Player)).Select(l=>l.Key));return r.Contributions.Where(v=>v.Era==s.Era&&v.City==c.Location.City&&keys.Contains(v.LinkKey)).Sum(v=>v.VP);}
        public static StatisticsPageView Page(CompletedMatchStatistics r,StatisticsFilters f)
        {
            var s=Get(r,f.MapEra);var v=new StatisticsPageView{Map=s,Context="Board at era scoring, before links are removed. Open slots count every player; player buttons emphasize ownership."};
            var rows=new StatisticsSection("Era-end city snapshot","All occupants and direct link owners are retained. VP columns use selected players only. Detail/action/evidence filters are suspended on this board view.","City","Slots open","Industries at scoring","Incident link owners","Selected industry VP","Selected link contribution VP"){Id="geography-snapshot"};
            foreach(var c in s.Cities){
                string tiles=String.Join("; ",c.Tiles.Select(a=>r.Name(a.Player)+" · "+StatisticsMetrics.IndustryName(a.Type)+" Lv."+a.Level+" · "+(s.Flipped(a)?"flipped":"unflipped")));
                string owners=String.Join("; ",c.Links.GroupBy(l=>l.Player).Select(g=>r.Name(g.Key)+" · "+g.Count()));
                rows.Rows.Add(new StatisticsRow("city-"+s.Era+"-"+c.Location.City,c.Problem??"Physical vacancies, not a build permission.",StatisticsPlotPages.CityName(c.Location.City),c.Location.Merchant?"Not applicable":!c.Available?"Unavailable":(c.Slots-c.Occupied)+" / "+c.Slots,tiles,owners,c.Available?IndustryVP(s,c,f).ToString():"—",ContributionVP(r,s,c,f).ToString()));
            }
            v.Ledgers.Add(rows);
            var unmapped=s.Cities.Where(c=>!StatisticsBoardLayout.Points.ContainsKey(c.Location.City)).ToArray();
            if(unmapped.Length>0){var missing=new StatisticsSection("Unmapped records","These locations have no verified diagram position.","Location");foreach(var c in unmapped)missing.Rows.Add(new StatisticsRow(c.Location.City,"",c.Location.City));v.Ledgers.Add(missing);}
            return v;
        }
    }
}
