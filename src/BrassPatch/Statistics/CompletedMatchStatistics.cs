using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Globalization;
using BoardGameRules.Entities.Players;
namespace BrassPatch.Statistics
{
    // Value-only report. No domain objects, Unity transforms, RNG or repository survive collection.
    public sealed class CompletedMatchStatistics
    {
        public string GameId { get; internal set; }
        public int FinalRevision { get; internal set; }
        public bool Introductory { get; internal set; }
        public bool CompleteHistory { get; internal set; }
        public string Availability { get; internal set; }
        public double CollectionMilliseconds { get; internal set; }
        internal readonly List<StatisticsPlayer> players = new List<StatisticsPlayer>();
        internal readonly List<StatisticsAsset> assets = new List<StatisticsAsset>();
        internal readonly List<StatisticsLink> links = new List<StatisticsLink>();
        internal readonly List<StatisticsAction> actions = new List<StatisticsAction>();
        internal readonly List<StatisticsEntry> entries = new List<StatisticsEntry>();
        internal readonly List<StatisticsSample> samples = new List<StatisticsSample>();
        internal readonly List<StatisticsRound> rounds = new List<StatisticsRound>();
        internal readonly List<StatisticsCity> cities = new List<StatisticsCity>();
        internal readonly List<StatisticsConnection> connections = new List<StatisticsConnection>();
        internal readonly List<StatisticsBoundary> boundaries = new List<StatisticsBoundary>();
        internal readonly List<StatisticsMatTile> matTiles = new List<StatisticsMatTile>();
        internal readonly List<StatisticsLinkContribution> contributions = new List<StatisticsLinkContribution>();
        internal readonly List<StatisticsLocation> locations = new List<StatisticsLocation>();
        internal readonly List<StatisticsStock> stocks = new List<StatisticsStock>();
        internal readonly List<string> warnings = new List<string>();
        public ReadOnlyCollection<StatisticsPlayer> Players { get { return players.AsReadOnly(); } }
        public ReadOnlyCollection<StatisticsAsset> Assets { get { return assets.AsReadOnly(); } }
        public ReadOnlyCollection<StatisticsLink> Links { get { return links.AsReadOnly(); } }
        public ReadOnlyCollection<StatisticsAction> Actions { get { return actions.AsReadOnly(); } }
        public ReadOnlyCollection<StatisticsEntry> Entries { get { return entries.AsReadOnly(); } }
        public ReadOnlyCollection<StatisticsSample> Samples { get { return samples.AsReadOnly(); } }
        public ReadOnlyCollection<StatisticsRound> Rounds { get { return rounds.AsReadOnly(); } }
        public ReadOnlyCollection<StatisticsCity> Cities { get { return cities.AsReadOnly(); } }
        public ReadOnlyCollection<StatisticsConnection> Connections { get { return connections.AsReadOnly(); } }
        public ReadOnlyCollection<StatisticsBoundary> Boundaries { get { return boundaries.AsReadOnly(); } }
        public ReadOnlyCollection<StatisticsMatTile> MatTiles { get { return matTiles.AsReadOnly(); } }
        public ReadOnlyCollection<StatisticsLinkContribution> Contributions { get { return contributions.AsReadOnly(); } }
        public ReadOnlyCollection<StatisticsLocation> Locations { get { return locations.AsReadOnly(); } }
        public ReadOnlyCollection<StatisticsStock> Stocks { get { return stocks.AsReadOnly(); } }
        public ReadOnlyCollection<string> Warnings { get { return warnings.AsReadOnly(); } }
        public string Name(PlayerColor color) { var player = players.Find(p => p.Color == color); return player == null ? color.ToString() : player.Name; }
    }
    public sealed class StatisticsPlayer
    {
        public PlayerColor Color { get; internal set; }
        public string Name { get; internal set; }
        public string Avatar { get; internal set; }
        public string Opponent { get; internal set; }
        public int FinalVP { get; internal set; }
        public int FinalCash { get; internal set; }
        public int FinalIncome { get; internal set; }
        public int Rank { get; internal set; }
        public int OpeningCash { get; internal set; }
        public int OpeningVP { get; internal set; }
    }
    public abstract class StatisticsPosition
    {
        public int Revision { get; internal set; }
        public int Era { get; internal set; }
        public int Round { get; internal set; }
        public int Turn { get; internal set; }
        public int OwnTurn { get; internal set; }
        public PlayerColor Player { get; internal set; }
        public string Position { get { return (Era == 2 ? "Rail" : "Canal") + " · round " + Round + " · turn " + Turn; } }
    }
    public sealed class StatisticsAsset : StatisticsPosition
    {
        public string Key { get; internal set; }
        public string IndustryId { get; internal set; }
        public string Slot { get; internal set; }
        public string City { get; internal set; }
        public string Type { get; internal set; }
        public int Level { get; internal set; }
        public int PrintedVP { get; internal set; }
        public int LinkIcons { get; internal set; }
        public int BaseCost { get; internal set; }
        public int MarketCost { get; internal set; }
        public int CoalUsed { get; internal set; }
        public int IronUsed { get; internal set; }
        public int Produced { get; internal set; }
        public int Stock { get; internal set; }
        public int MarketReceipts { get; internal set; }
        public int Supplied { get; internal set; }
        public int FlipTurn { get; internal set; }
        public int FlipEra { get; internal set; }
        public int FlipRevision { get; internal set; }
        public int FlipOwnTurn { get; internal set; }
        public string FlipTrigger { get; internal set; }
        public PlayerColor? FlipActor { get; internal set; }
        public bool Sellable { get; internal set; }
        public string RegionColor { get; internal set; }
        public int IncomeSpaces { get; internal set; }
        public int CanalVP { get; internal set; }
        public int RailVP { get; internal set; }
        public int IntroVP { get; internal set; }
        public int CanalPrinted { get; internal set; }
        public int RailPrinted { get; internal set; }
        public int RemovedRevision { get; internal set; }
        public int RemovedEra { get; internal set; }
        public string Removal { get; internal set; }
        public bool Carried { get; internal set; }
        public bool LastSlot { get; internal set; }
        public bool OpponentOverbuild { get; internal set; }
        public bool Flipped { get { return FlipRevision > 0; } }
        public bool Remaining { get { return RemovedRevision == 0; } }
    }
    public sealed class StatisticsLink : StatisticsPosition
    {
        public string Key { get; internal set; }
        public string LinkId { get; internal set; }
        public string Route { get; internal set; }
        public int VP { get; internal set; }
        public int OwnVP { get; internal set; }
        public int OthersVP { get; internal set; }
        public int MerchantVP { get; internal set; }
        public bool Scored { get; internal set; }
        public string Contributions { get; internal set; }
        public string[] Cities { get; internal set; }
    }
    public sealed class StatisticsAction : StatisticsPosition
    {
        public string Key { get; internal set; }
        public string Kind { get; internal set; }
        public int EndRevision { get; internal set; }
        public int Spending { get; internal set; }
        public int MarketCost { get; internal set; }
        public int Borrowed { get; internal set; }
        public int IncomeDrop { get; internal set; }
        public int Goods { get; internal set; }
        public int Developed { get; internal set; }
        public int Links { get; internal set; }
        internal readonly List<string> notes = new List<string>();
        internal readonly List<string> cards = new List<string>();
        public string Details { get { return String.Join("\n", notes); } }
        public string Cards { get { return String.Join(", ", cards); } }
    }
    public sealed class StatisticsEntry : StatisticsPosition
    {
        public int RecordOrdinal { get; internal set; }
        public string ActionKey { get; internal set; }
        public string Kind { get; internal set; }
        public string Category { get; internal set; }
        public string Label { get; internal set; }
        public string Detail { get; internal set; }
        public string AssetKey { get; internal set; }
        public int Cash { get; internal set; }
        public int Income { get; internal set; }
        public int Marker { get; internal set; }
        public int VP { get; internal set; }
        public int Quantity { get; internal set; }
        public string Resource { get; internal set; }
        public string Source { get; internal set; }
        public PlayerColor? Supplier { get; internal set; }
        public string Destination { get; internal set; }
        public bool DestinationDerived { get; internal set; }
        public int BeforeQuote { get; internal set; }
        public int AfterQuote { get; internal set; }
        public int TotalPrice { get; internal set; }
        public string TileId { get; internal set; }
        public int NextLevel { get; internal set; }
        public bool NextRailBlocked { get; internal set; }
    }
    public sealed class StatisticsSample : StatisticsPosition
    {
        public int Cash { get; internal set; }
        public int Income { get; internal set; }
        public int VP { get; internal set; }
        public string Annotation { get; internal set; }
    }
    public sealed class StatisticsRound
    {
        public int Era { get; internal set; }
        public int Round { get; internal set; }
        internal readonly List<PlayerColor> order = new List<PlayerColor>();
        internal readonly Dictionary<PlayerColor, int> spending = new Dictionary<PlayerColor, int>();
        public ReadOnlyCollection<PlayerColor> Order { get { return order.AsReadOnly(); } }
        public int Spent(PlayerColor color) { int amount; return spending.TryGetValue(color, out amount) ? amount : 0; }
    }
    public sealed class StatisticsCity
    {
        public string Boundary { get; internal set; }
        public int Era { get; internal set; }
        public string City { get; internal set; }
        public int Slots { get; internal set; }
        public int Occupied { get; internal set; }
        public PlayerColor Player { get; internal set; }
        public int Owned { get; internal set; }
    }
    public sealed class StatisticsConnection : StatisticsPosition
    {
        public string ActionKey { get; internal set; }
        public string Purpose { get; internal set; }
        public string From { get; internal set; }
        public string To { get; internal set; }
        public bool OpponentRequired { get; internal set; }
    }
    public sealed class StatisticsBoundary : StatisticsPosition
    {
        public string Name { get; internal set; }
        public int Cash { get; internal set; }
        public int Income { get; internal set; }
    }
    public sealed class StatisticsMatTile : StatisticsPosition
    {
        public string Boundary {get;internal set;}
        public string Type {get;internal set;}
        public string TileId {get;internal set;}
        public int Level {get;internal set;}
        public bool RailBlocked {get;internal set;}
    }
    public sealed class StatisticsStock : StatisticsPosition
    {
        public string Boundary {get;internal set;}
        public string AssetKey {get;internal set;}
        public int Stock {get;internal set;}
    }
    public sealed class StatisticsLinkContribution
    {
        public string LinkKey {get;internal set;}
        public string City {get;internal set;}
        public string RegionColor {get;internal set;}
        public int Era {get;internal set;}
        public int VP {get;internal set;}
        public PlayerColor Owner {get;internal set;}
        public PlayerColor? Supplier {get;internal set;}
    }
    public sealed class StatisticsLocation
    {
        public string City {get;internal set;}
        public string RegionColor {get;internal set;}
        public bool Merchant {get;internal set;}
        public bool Farm {get;internal set;}
        string[] neighbors=new string[0];
        public string[] Neighbors {get{return (string[])neighbors.Clone();}internal set{neighbors=value==null?new string[0]:(string[])value.Clone();}}
    }
    public static class StatisticsFormat
    {
        public static string Ratio(double numerator, double denominator, int decimals = 2)
        { return denominator <= 0 ? "—" : (numerator / denominator).ToString("F" + decimals, CultureInfo.InvariantCulture); }
        public static string Percent(double numerator, double denominator)
        { return denominator <= 0 ? "—" : (100 * numerator / denominator).ToString("F1", CultureInfo.InvariantCulture) + "%"; }
        public static string Money(int value) { return "£" + value.ToString(CultureInfo.InvariantCulture); }
    }
}
