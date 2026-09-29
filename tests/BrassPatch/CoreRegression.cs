using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Runtime.Serialization;
using System.Text;
using BoardGameRules.Entities.Actions.Events;
using BoardGameRules.Entities.Board.Events;
using BoardGameRules.Entities.Board.BuildingSlots;
using BoardGameRules.Entities.Board.Merchants;
using BoardGameRules.Entities.Board.Regions;
using BoardGameRules.Entities.Cards;
using BoardGameRules.Entities.Common;
using BoardGameRules.Entities.EventSourcing.Events;
using BoardGameRules.Entities.Game;
using BoardGameRules.Entities.GamePhases;
using BoardGameRules.Entities.GamePhases.Events;
using BoardGameRules.Entities.Industries;
using BoardGameRules.Entities.NumericalResources.Events;
using BoardGameRules.Entities.Players;
using BrassApplication.Game;
using BrassApplication.GameHistory;
using BrassPatch;
using Frameworks.View.Game.Replay;
using InterfaceAdapters.Common;
using InterfaceAdapters.Game.PlayersSummary;
using InterfaceAdapters.Routing;
using Utils;

public static class CoreRegression
{
    static int checks;
    static void Check(bool condition, string name) { checks++; if (!condition) throw new Exception("Core: " + name); }
    static BoardGameRulesGame New(int seed, bool legacy = false)
    {
        var cfg = new GameStartConfiguration(NumberOfPlayers.Four) { Seed = seed };
        var game = new BoardGameRulesGame(cfg);
        if (legacy) { game.ApplyEvent(new GameStarted(cfg)); game.GameFlow.StartGame(); }
        else game.StartGame();
        return game;
    }
    static IEnumerable<BaseIndustry> Tiles(BoardGameRulesGame game)
    {
        return game.PlayersTrack.Players.SelectMany(p => Enum.GetValues(typeof(BuildingType)).Cast<BuildingType>()
            .Where(t => t != BuildingType.Undefined).SelectMany(t => p.Mat.GetIndustryList(t)));
    }
    static List<string> Ids(BoardGameRulesGame game)
    {
        return Tiles(game).Select(t => t.UniqueId).Concat(game.Board.Regions.SelectMany(r => r.BuildingSlots).Select(s => s.UniqueId))
            .Concat(game.Board.Regions.OfType<BorderRegion>().SelectMany(r => r.MerchantSlots).Select(s => s.UniqueId))
            .Concat(game.Board.CardDeck.Concat(game.Board.WildIndustryCardDeck).Concat(game.Board.WildLocationCardDeck)
                .Concat(game.PlayersTrack.Players.SelectMany(p => p.Hand.Concat(p.DiscardPile))).Select(c => c.UniqueId)).ToList();
    }
    static string Cards(BoardGameRulesGame game)
    {
        return String.Join("|", game.Board.CardDeck.Concat(game.PlayersTrack.Players.SelectMany(p => p.Hand.Concat(p.DiscardPile))).Select(c => c.ToString()));
    }
    static byte[] Serialize<T>(T value)
    {
        var stream = new MemoryStream();
        new Frameworks.Persistence.Serialization.JsonSerializer().Serialize(stream, value);
        return stream.ToArray();
    }
    static T Deserialize<T>(byte[] bytes)
    {
        return new Frameworks.Persistence.Serialization.JsonSerializer().Deserialize<T>(new MemoryStream(bytes));
    }
    static BoardGameRulesGame Reconstruct(byte[] bytes)
    {
        var events = Deserialize<List<IEvent>>(bytes);
        var copy = new BoardGameRulesGame(((GameStarted)events[0]).GameStartConfiguration);
        foreach (var e in events) copy.ApplyEvent(e);
        return copy;
    }
    static void Identities()
    {
        foreach (var seed in Enumerable.Range(0, 128).Concat(new[] { 445060, 677697, 754778, 984130 }))
        {
            var game = New(seed); var legacy = New(seed, true);
            var ids = Ids(game);
            Check(ids.Count == ids.Distinct().Count(), "unique setup IDs " + seed);
            Check(IdentityFixes.Version(game.GameStartConfiguration) == 1, "new rules version");
            Check(IdentityFixes.Version(legacy.GameStartConfiguration) == 0, "legacy rules version");
            Check(game.Randomer.RandomerInitData.FastForwardCount == legacy.Randomer.RandomerInitData.FastForwardCount, "unchanged PRNG advancement");
            Check(Cards(game) == Cards(legacy), "unchanged seeded deck/hands");
            Check(game.Board.Regions.OfType<BorderRegion>().SelectMany(r => r.MerchantSlots).Select(m => m.MerchantTile == null ? "null" : String.Join(",", m.MerchantTile.BuildingTypes)).SequenceEqual(
                legacy.Board.Regions.OfType<BorderRegion>().SelectMany(r => r.MerchantSlots).Select(m => m.MerchantTile == null ? "null" : String.Join(",", m.MerchantTile.BuildingTypes))), "unchanged merchant arrangement");
            var copy = Reconstruct(Serialize(game.GetUncommittedEvents().ToList()));
            Check(Ids(game).SequenceEqual(Ids(copy)), "JSON event reconstruction preserves v1 IDs");
            var legacyJson = Encoding.UTF8.GetString(Serialize(legacy.GetUncommittedEvents().ToList()));
            legacyJson = System.Text.RegularExpressions.Regex.Replace(legacyJson, @"\s*""BrassRulesVersion"":\s*0,?", "");
            // The field is usually followed by another member; handle a trailing comma defensively.
            legacyJson = System.Text.RegularExpressions.Regex.Replace(legacyJson, @",\s*}", "}");
            var oldCopy = Reconstruct(Encoding.UTF8.GetBytes(legacyJson));
            Check(IdentityFixes.Version(oldCopy.GameStartConfiguration) == 0 && Ids(legacy).SequenceEqual(Ids(oldCopy)), "missing-version legacy JSON preserves original IDs");
        }
        var collision = New(984130, true);
        var red = collision.PlayersTrack.GetPlayerByColor(PlayerColor.Red);
        var brewery = (Brewery)red.Mat.GetIndustryList(BuildingType.Brewery).Single(t => t.UniqueId == "industry944625454");
        var cotton = red.Mat.GetIndustryList(BuildingType.CottonMill).Single(t => t.UniqueId == brewery.UniqueId);
        red.Mat.RemoveGivenBuildingFromMat(brewery); red.Mat.RemoveGivenBuildingFromMat(cotton);
        var cottonSlot = collision.Board.GetRegionByName(RegionName.Birmingham).BuildingSlots.First(s => s.CanBuildIndustryType(BuildingType.CottonMill));
        var brewSlot = collision.Board.GetRegionByName(RegionName.Central).BuildingSlots.Single();
        cottonSlot.BuildIndustry(cotton); brewSlot.BuildIndustry(brewery);
        collision.GameProgressInformation.CurrentGameEra = GameEra.RailEra; brewery.Build();
        var consumer = collision.TurnFlow.StartSellAction().GeneralAction.ConsumingBeer;
        consumer.SupplyBeer(brewery, 2, 2, red, new List<Region> { collision.Board.GetRegionByName(RegionName.Central) });
        Check(brewery.IsFlipped && !cotton.IsFlipped && brewery.AvailableBeer == 0, "legacy collision flip uses recorded region");
    }
    static void Oxford()
    {
        foreach (int marker in new[] { 0, 10, 97, 98, 99 })
        {
            var game = New(12); var player = game.GameProgressInformation.CurrentPlayer;
            game.ApplyEvent(new IncomeMarkerPositionChanged(player.Color, marker - player.IncomeMarkerPosition, "fixture"));
            game.ApplyEvent(new MerchantBonusReceived(BonusType.Income, player.Color, RegionName.Oxford));
            Check(player.IncomeMarkerPosition == Math.Min(99, marker + 2), "Oxford intermediate state is capped");
            var copy = Reconstruct(Serialize(game.GetUncommittedEvents().ToList()));
            Check(copy.PlayersTrack.GetPlayerByColor(player.Color).IncomeMarkerPosition == player.IncomeMarkerPosition, "Oxford cap survives JSON event replay");
            Check(PresentationFixes.CappedIncome(marker, 2) == player.IncomeMarkerPosition, "Oxford displayed target matches state");
        }
        Check(PresentationFixes.CappedIncome(98, 9) == 99, "industry income animation cap");
    }
    static void Presentation(bool unity)
    {
        var game = New(14);
        var metadata = new GameMetadata(game.GameStartConfiguration, GameType.Local);
        var app = new BrassApplicationGame(game, metadata);
        var history = new BrassGameHistory(); history.SetGame(app); Global.Instance.BrassGameHistory = history;
        var summary = new Summary(); var router = new Router(summary); Global.Instance.Router = router;
        var view = (ReplayView)FormatterServices.GetUninitializedObject(typeof(ReplayView));
        ReflectionAccess.Set(view, "_router", router);
        var method = typeof(ReplayView).GetMethod("CreateAnimation", ReflectionAccess.Flags, null, new[] { typeof(MerchantBonusReceived) }, null);
        var player = game.GameProgressInformation.CurrentPlayer;
        foreach (int vp in new[] { 0, 1, 20, 101 })
        foreach (var region in new[] { RegionName.Shrewsbury, RegionName.Nottingham })
        {
            player.VP = vp; method.Invoke(view, new object[] { new MerchantBonusReceived(BonusType.VP, player.Color, region) });
            Check(summary.Vp == (vp + (region == RegionName.Shrewsbury ? 4 : 3)).ToString(), "B11 actual replay method target");
        }
        if (unity)
        {
            foreach (int marker in new[] { 97, 98, 99 })
            {
                player.IncomeMarkerPosition = marker;
                var animation = (IAnimation)method.Invoke(view, new object[] { new MerchantBonusReceived(BonusType.Income, player.Color, RegionName.Oxford) });
                Check(summary.Marker == Math.Min(99, marker + 2) + "/99", "actual Unity replay Oxford label");
                animation.Kill();
            }
            // A historical source may be absent or ambiguous; the summary animation must still finish.
            var incomeMethod = typeof(ReplayView).GetMethod("CreateAnimation", ReflectionAccess.Flags, null, new[] { typeof(IncomeMarkerPositionChanged) }, null);
            var missingAnimation = (IAnimation)incomeMethod.Invoke(view, new object[] { new IncomeMarkerPositionChanged(player.Color, 2, "missing-industry") });
            Check(summary.Marker == "99/99", "missing income source does not stall replay");
            missingAnimation.Kill();
            var collision = New(984130, true);
            var red = collision.PlayersTrack.GetPlayerByColor(PlayerColor.Red);
            var brewery = red.Mat.GetIndustryList(BuildingType.Brewery).Single(t => t.UniqueId == "industry944625454");
            var cotton = red.Mat.GetIndustryList(BuildingType.CottonMill).Single(t => t.UniqueId == brewery.UniqueId);
            red.Mat.RemoveGivenBuildingFromMat(brewery); red.Mat.RemoveGivenBuildingFromMat(cotton);
            collision.Board.GetRegionByName(RegionName.Birmingham).BuildingSlots.First(s => s.CanBuildIndustryType(BuildingType.CottonMill)).BuildIndustry(cotton);
            collision.Board.GetRegionByName(RegionName.Central).BuildingSlots.Single().BuildIndustry(brewery);
            history.SetGame(new BrassApplicationGame(collision, new GameMetadata(collision.GameStartConfiguration, GameType.Local)));
            red.IncomeMarkerPosition = 98;
            var ambiguousAnimation = (IAnimation)incomeMethod.Invoke(view, new object[] { new IncomeMarkerPositionChanged(red.Color, 2, brewery.UniqueId) });
            Check(summary.Marker == "99/99", "ambiguous legacy income source does not stall replay");
            ambiguousAnimation.Kill();
        }
    }
    static void BreweryLiquidationJson()
    {
        var game = New(32703); var brewery = (Brewery)game.PlayersTrack.Players[0].Mat.GetFirstBuildingOfType(BuildingType.Brewery);
        brewery.AvailableBeer = 1;
        var ended = new BoardGameRules.Entities.SellIndustry.Events.IndustrySellingEnded(
            new Dictionary<PlayerColor, List<BaseIndustry>> { [brewery.Color] = new List<BaseIndustry> { brewery } },
            new Dictionary<PlayerColor, int> { [brewery.Color] = 0 });
        byte[] bytes = Serialize(new List<IEvent> { ended });
        Check(!Encoding.UTF8.GetString(bytes).Contains("_numberOfBeersFunc"), "brewery liquidation does not serialize runtime callback");
        var copy = (BoardGameRules.Entities.SellIndustry.Events.IndustrySellingEnded)Deserialize<List<IEvent>>(bytes)[0];
        var tile = (Brewery)copy.SoldInLastRound[brewery.Color][0];
        Check(tile.UniqueId == brewery.UniqueId && tile.AvailableBeer == 1 && tile.MoneyCost == brewery.MoneyCost, "brewery liquidation summary roundtrip");
        var json = Newtonsoft.Json.Linq.JObject.Parse(Encoding.UTF8.GetString(bytes).TrimStart('\uFEFF'));
        var oldTile = json["$values"][0]["SoldInLastRound"][brewery.Color.ToString()]["$values"][0];
        oldTile["_numberOfBeersFunc"] = new Newtonsoft.Json.Linq.JObject {
            ["$type"] = typeof(Func<int>).AssemblyQualifiedName,
            ["Method"] = new Newtonsoft.Json.Linq.JObject { ["$type"] = "System.Reflection.RuntimeMethodInfo, mscorlib" } };
        copy = (BoardGameRules.Entities.SellIndustry.Events.IndustrySellingEnded)Deserialize<List<IEvent>>(Encoding.UTF8.GetBytes(json.ToString()))[0];
        Check(copy.SoldInLastRound[brewery.Color][0].UniqueId == brewery.UniqueId, "legacy unserializable brewery callback ignored on read");
        int expected = brewery._numberOfBeersFunc(); brewery.Build();
        Check(brewery.AvailableBeer == expected, "live setup brewery retains its dynamic production callback");
    }
    public static int Main(string[] args)
    {
        try { Identities(); Oxford(); BreweryLiquidationJson(); Presentation(args.Contains("unity")); Console.WriteLine("Core passed: " + checks + " assertions; actual JSON roundtrips and replay-method calls."); return 0; }
        catch (Exception e) { Console.Error.WriteLine(e); return 1; }
    }
    sealed class Router : IRouter
    {
        readonly Summary summary; public Router(Summary value) { summary = value; }
        public IUseCaseBuilder UseCaseBuilder { get; set; }
        public T GetView<T>() where T : class, IBaseView { return summary as T; }
        public IAnimation ShowView<T>() where T : class, IBaseView { return null; }
        public IAnimation HideView<T>() where T : class, IBaseView { return null; }
        public void MoveOnTop<T>() where T : class, IBaseView { }
        public bool IsViewVisible<T>() where T : class, IBaseView { return false; }
        public bool IsViewCached<T>() where T : class, IBaseView { return false; }
        public void ProcessInput() { }
        public void RegisterKey(KeyEnum k, IBaseView v) { }
        public void UnregisterKey(KeyEnum k, IBaseView v) { }
        public void UnregisterAllKeys(IBaseView v) { }
    }
    sealed class Summary : IPlayersSummaryView
    {
        public string Vp, Marker;
        public IInputRegistration InputRegistration { get; set; }
        public bool DisableMenusUnderneath { get; set; }
        public bool DestroyWhenClosed { get; set; }
        public IController Controller { get; set; }
        public void UpdatePlayersSummaryView(List<PlayerViewModel> rows) { }
        public IAnimation CreateChangeVPAnimation(string value, PlayerColor p) { Vp = value; return null; }
        public IAnimation CreateChangeIncomeAnimation(string level, string marker, PlayerColor p) { Marker = marker; return null; }
        public IAnimation CreateShowCurrentPlayerAnimation(PlayerColor p) { return null; }
        public IAnimation CreateShuffleNewPlayersOrderAnimation(RoundStarted e) { return null; }
        public IAnimation CreateEndOfRoundAnimation(RoundEnded e) { return null; }
        public IAnimation CreateChangeMoneyAnimation(string a, PlayerColor p) { return null; }
        public IAnimation CreateChangeMoneySpentAnimation(string a, PlayerColor p, string s) { return null; }
        public IAnimation CreateVictoryPointsForMoneyAnimation(int a, PlayerColor p) { return null; }
        public IAnimation CreateVictoryPointsForIncomeAnimation(int a, PlayerColor p) { return null; }
        public void HandleKeyDown(KeyEnum k) { } public void HandleKey(KeyEnum k) { } public void HandleKeyUp(KeyEnum k) { }
        public void EnableView() { } public void DisableView() { } public void SetInteractable(bool v) { }
        public bool IsViewVisible() { return false; } public void MoveToTop() { } public void MoveToBottom() { }
        public IAnimation CreateShowViewAnimation() { return null; } public IAnimation CreateHideViewAnimation() { return null; } public IAnimation CreateShowViewImmediatelyAnimation() { return null; }
    }
}
