using System;
using System.Collections.Generic;
using System.Linq;
using BoardGameRules.Entities.Actions;
using BoardGameRules.Entities.Board.BuildingSlots;
using BoardGameRules.Entities.Board.Merchants;
using BoardGameRules.Entities.Board.Regions;
using BoardGameRules.Entities.Cards;
using BoardGameRules.Entities.Common;
using BoardGameRules.Entities.GamePhases;
using BoardGameRules.Entities.Industries;
using BoardGameRules.Utils.Logging;
using BrassApplication.AI.AIHelpers;
using BrassApplication.Game;
using BrassApplication.GameHistory;
using BrassApplication.UseCases.Game.Actions;
using BrassApplication.UseCases.Game.Actions.BaseAction;
using BrassApplication.UseCases.Game.Actions.SellAction;
using BrassApplication.UseCases.Game.Actions.NetworkAction;
using BrassApplication.UseCases.Game.Actions.CoalDelivery;
using BrassPatch;

public static class ResourcesRegression
{
    static int checks;
    static void Check(bool condition, string label) { checks++; if (!condition) throw new Exception(label); }
    static void Rejected(Action command, string label)
    {
        bool rejected = false; try { command(); } catch (InvalidOperationException) { rejected = true; }
        Check(rejected, label);
    }
    static BrassApplicationGame NewGame(int count) { return ExtendedNetworkMatrix.NewGame(count, 20260927); }
    static BaseIndustry Put(BrassApplicationGame app, RegionName region, BuildingType type, int level, bool opponent = false)
    { return ExtendedNetworkMatrix.Put(app, region, type, level, opponent); }
    static LocationCard Card(BrassApplicationGame app, RegionName name)
    {
        var p = app.Game.GameProgressInformation.CurrentPlayer;
        var region = app.Game.Board.GetRegionByName(name);
        var card = new LocationCard("test-location", NumberOfPlayers.Two, name, region.Color);
        p.Hand.Clear(); p.Hand.Add(card); return card;
    }
    static BorderRegion PrepareMerchants(BrassApplicationGame app, RegionName region, BuildingType type, int count)
    {
        foreach (var r in app.Game.Board.Regions.OfType<BorderRegion>()) foreach (var m in r.MerchantSlots)
            m.SetMerchant(new MerchantTile(NumberOfPlayers.Two, new List<BuildingType>()));
        var target = (BorderRegion)app.Game.Board.GetRegionByName(region);
        for (int i = 0; i < count; i++) target.MerchantSlots[i].SetMerchant(new MerchantTile(NumberOfPlayers.Two, new List<BuildingType> { type }));
        return target;
    }
    public static object Run()
    {
        checks = 0;
        foreach (int count in new[] {2,3,4})
        {
            BuildCoal(count); Overbuild(count); RailCoal(count); SaleMerchant(count); NoTargetBonus(count);
        }
        return new { assertions = checks, findings = new[] { "B02", "B03", "B04", "B08", "B10" }, failures = new string[0] };
    }
    static void BuildCoal(int count)
    {
        var app = NewGame(count); var g = app.Game; var p = g.GameProgressInformation.CurrentPlayer; var b = g.Board;
        g.GameProgressInformation.CurrentGameEra = GameEra.RailEra; p.Money = 22;
        var target = b.GetRegionByName(RegionName.Coventry);
        p.Mat.DevelopIndustry(BuildingType.Pottery); p.Mat.DevelopIndustry(BuildingType.Pottery);
        var pottery = p.Mat.GetFirstBuildingOfType(BuildingType.Pottery); var card = Card(app, RegionName.Coventry);
        var near = (CoalMine)Put(app, RegionName.Coventry, BuildingType.CoalMine, 2, true);
        var far = (CoalMine)Put(app, RegionName.Nuneaton, BuildingType.CoalMine, 2, true);
        near.AvailableCoal = far.AvailableCoal = 1;
        b.GetLinkById("RailCoventryNuneaton").SetOwner(p);
        var slot = target.BuildingSlots.First(s => s.BuildingTypes.Contains(BuildingType.Pottery)); var action = g.TurnFlow.BuildAction;
        Check(BuildAction.CanUseBuildActionForGivenSlot(slot, pottery, p, b, g.GameProgressInformation), "B02 legal two-mine build");
        Check(action.GetCantUseBuildActionReasonForGivenSlot(slot, pottery, p, b, g.GameProgressInformation).Count == 0, "B02 reasons agree");
        action.DiscardCard(card, p); action.Build(slot, BuildingType.Pottery);
        Rejected(() => action.SupplyCoal(far, 1), "B02 farther mine remains illegal until depletion");
        action.SupplyCoal(near, 1); action.SupplyCoal(far, 1);
        Check(p.Money == 0 && action.NeededCoal == 0 && near.IsFlipped && far.IsFlipped, "B02 execution settles exactly");
        var disconnected = NewGame(count); var gd = disconnected.Game; var pd = gd.GameProgressInformation.CurrentPlayer;
        gd.GameProgressInformation.CurrentGameEra = GameEra.RailEra;
        Put(disconnected, RegionName.Nuneaton, BuildingType.CoalMine, 2, true);
        Check(ResourcesFixes.CostOfCoalForGivenIndustry(gd.Board.GetRegionByName(RegionName.Coventry), gd.Board, 2) == -1, "B02 disconnected coal cannot satisfy estimate");
        gd.Board.GetLinkById("RailBirminghamOxford").SetOwner(pd);
        Check(ResourcesFixes.CostOfCoalForGivenIndustry(gd.Board.GetRegionByName(RegionName.Birmingham), gd.Board, 2) == gd.Board.CoalMarket.CoalPrice(2), "B02 market shortfall price");
    }
    static void Overbuild(int count)
    {
        var app = NewGame(count); var g = app.Game; var p = g.GameProgressInformation.CurrentPlayer; var b = g.Board;
        g.GameProgressInformation.CurrentGameEra = GameEra.RailEra; p.Money = 100;
        var r = b.GetRegionByName(RegionName.Cannock); p.Mat.DevelopIndustry(BuildingType.CoalMine);
        var old = p.Mat.GetFirstBuildingOfType(BuildingType.CoalMine); p.Mat.RemoveGivenBuildingFromMat(old); r.BuildingSlots[0].BuildIndustry(old); old.Build();
        p.Mat.DevelopIndustry(BuildingType.CoalMine); var next = p.Mat.GetFirstBuildingOfType(BuildingType.CoalMine);
        var card = Card(app, RegionName.Cannock); var slot = r.BuildingSlots[0]; var action = g.TurnFlow.BuildAction;
        Check(BuildAction.CanUseBuildActionForGivenSlot(slot, next, p, b, g.GameProgressInformation), "B08 dual-space overbuild legal");
        Check(!action.GetCantUseBuildActionReasonForGivenSlot(slot, next, p, b, g.GameProgressInformation).Any(), "B08 reasons agree");
        action.DiscardCard(card, p); action.Build(slot, BuildingType.CoalMine);
        Check(ReferenceEquals(slot.BuildedIndustry, next) && b.RemovedIndustries.Contains(old), "B08 original tile removed");
        var fresh = NewGame(count); var region = fresh.Game.Board.GetRegionByName(RegionName.Cannock);
        Check(!BuildAction.IsGivenSlotTheBestOneInGivenRegionForGivenTypeStatic(region.BuildingSlots[0], BuildingType.CoalMine, fresh.Game.Board), "B08 vacant dual space still loses to vacant single");
        Check(!BuildAction.IsSingleSlotOrDoubleSlotButLastPossibleStatic(region.BuildingSlots[0], BuildingType.CoalMine, region), "B08 duplicate predicate agrees");
    }
    static void RailCoal(int count)
    {
        var app = NewGame(count); var g = app.Game; var p = g.GameProgressInformation.CurrentPlayer; var b = g.Board;
        g.GameProgressInformation.CurrentGameEra = GameEra.RailEra; p.Money = 30;
        var near = (CoalMine)Put(app, RegionName.Dudley, BuildingType.CoalMine, 2);
        var far = (CoalMine)Put(app, RegionName.Redditch, BuildingType.CoalMine, 2);
        b.GetLinkById("RailBirminghamRedditch").SetOwner(p);
        var rail = b.GetLinkById("RailBirminghamDudley"); var action = g.TurnFlow.StartNetworkAction();
        action.DiscardCard(p.Hand[0], p); action.PlaceLink(rail);
        var mines = ResourcesFixes.ClosestCoalMines(b, new[] {rail.Region1, rail.Region2, rail.Region3});
        Check(mines.Count == 1 && ReferenceEquals(mines[0], near), "B03 compare rail endpoints globally");
        int before = far.AvailableCoal;
        Rejected(() => action.SupplyCoal(far, 1), "B03 domain rejects farther source");
        Rejected(() => action.SupplyCoal(b.CoalMarket, 1), "B03 market prohibited before board exhaustion");
        Check(before == far.AvailableCoal && action.NeededCoal == 1, "B03 rejected source changes nothing");
        action.SupplyCoal(near, 1); Check(action.NeededCoal == 0, "B03 closest source completes rail");
        var ai = NewGame(count); var ga = ai.Game; var pa = ga.GameProgressInformation.CurrentPlayer;
        ga.GameProgressInformation.CurrentGameEra = GameEra.RailEra; pa.Money = 30; pa.LinkTileCount = 1;
        var aiNear = (CoalMine)Put(ai, RegionName.Dudley, BuildingType.CoalMine, 2);
        var aiFar = (CoalMine)Put(ai, RegionName.Redditch, BuildingType.CoalMine, 2);
        ga.Board.GetLinkById("RailBirminghamRedditch").SetOwner(pa); int nearBefore = aiNear.AvailableCoal, farBefore = aiFar.AvailableCoal;
        BoardHelper.Init(ai, BrassApplication.AI.AIType.Hard);
        NetworkActionAIHelper.PlaceLink(ga.TurnFlow.StartNetworkAction(),pa.Hand[0],ga.Board.GetLinkById("RailBirminghamDudley"),ga.Board,pa,false);
        Check(aiNear.AvailableCoal == nearBefore - 1 && aiFar.AvailableCoal == farBefore, "B03 original AI entry consumes nearest mine");
        var ui = NewGame(count); var gu = ui.Game; var pu = gu.GameProgressInformation.CurrentPlayer;
        gu.GameProgressInformation.CurrentGameEra = GameEra.RailEra; pu.Money = 30;
        var uiNear = (CoalMine)Put(ui,RegionName.Dudley,BuildingType.CoalMine,2); Put(ui,RegionName.Redditch,BuildingType.CoalMine,2);
        gu.Board.GetLinkById("RailBirminghamRedditch").SetOwner(pu);
        var history = new BrassGameHistory(); history.SetGame(ui); var output = new CoalOutput(); var useCase = new NetworkActionUseCase(output,history);
        useCase.StartNetworkAction();useCase.SelectCard(pu.Hand[0]);output.Pump();useCase.LinkSelected(gu.Board.GetLinkById("RailBirminghamDudley"));output.Pump();
        Check(output.Sources.Count == 1 && ReferenceEquals(output.Sources[0],uiNear), "B03 human picker offers global nearest only");
    }
    static void SaleMerchant(int count)
    {
        var app = NewGame(count); var g = app.Game; var p = g.GameProgressInformation.CurrentPlayer; var b = g.Board;
        var tile = Put(app, RegionName.Birmingham, BuildingType.Manufacturer, 5);
        var oxford = PrepareMerchants(app, RegionName.Oxford, BuildingType.Manufacturer, 2); b.GetLinkById("CanalBirminghamOxford").SetOwner(p);
        var action = g.TurnFlow.StartSellAction(); var slot = b.GetBuildingSlotContainingGivenIndustry(tile);
        Check(!action.GetValidBuildingSlotsForSellingIndustries(b, p).Contains(slot), "B04 two merchants cannot fund one industry");
        Check(action.GetSellActionNotPossibleReasons(b, g.GameProgressInformation).Contains(CantUseSellActionReason.NO_BEER), "B04 menu agrees");
        var candidates = new List<BuildingSlot> { slot };
        Check(SellActionAIHelper.SlotToSell(action, p.Hand[0], b, p, candidates) == null, "B04 stale AI candidate revalidated");
        var brewery = (Brewery)Put(app, RegionName.Stafford, BuildingType.Brewery, 2); brewery.AvailableBeer = 1;
        Check(action.GetValidBuildingSlotsForSellingIndustries(b, p).Contains(slot), "B04 own disconnected brewery plus merchant legal");
        action.DiscardCard(p.Hand[0], p); action.StartSellProcedure(tile); action.SupplyBeer(oxford.MerchantSlots[0], 1);
        Rejected(() => action.SupplyBeer(oxford.MerchantSlots[1], 1), "B04 engine locks the merchant for this industry");
        Check(ReferenceEquals(SellActionAIHelper.FindBeerSource(action,b,p,slot), brewery), "B04 AI selects brewery after merchant");
        action.SupplyBeer(brewery,1); action.SellIndustry();
        Check(tile.IsFlipped && oxford.MerchantSlots[1].HasBeer && action.NeededBeer == 0, "B04 preserves other merchant barrel");
        // Next industry in the same sell action can choose another merchant.
        var second = Put(app, RegionName.Birmingham, BuildingType.Manufacturer, 2);
        action.StartSellProcedure(second); action.SupplyBeer(oxford.MerchantSlots[1],1); action.SellIndustry();
        Check(second.IsFlipped && !oxford.MerchantSlots[1].HasBeer, "B04 context resets between industries");
        var ai = NewGame(count); var ga = ai.Game; var pa = ga.GameProgressInformation.CurrentPlayer;
        var aiIndustry = Put(ai,RegionName.Birmingham,BuildingType.Manufacturer,5); var merchants = PrepareMerchants(ai,RegionName.Oxford,BuildingType.Manufacturer,2);
        ga.Board.GetLinkById("CanalBirminghamOxford").SetOwner(pa); var aiBeer=(Brewery)Put(ai,RegionName.Stafford,BuildingType.Brewery,2);aiBeer.AvailableBeer=1;
        SellActionAIHelper.SellIndustries(ga.TurnFlow.StartSellAction(),pa.Hand[0],ga.Board,pa,ga.DevelopMerchantBonus,LogFactory.BuildLogger(typeof(SellAction)));
        Check(aiIndustry.IsFlipped && merchants.MerchantSlots.Count(m=>m.HasBeer)==1 && aiBeer.AvailableBeer==0,"B04 full AI sale uses one merchant plus brewery");
    }
    static void NoTargetBonus(int count)
    {
        foreach(int mode in new[]{0,1,2})
        {
            var app=NewGame(count);var g=app.Game;var p=g.GameProgressInformation.CurrentPlayer;var b=g.Board;
            var tile=Put(app,RegionName.Worcester,BuildingType.CottonMill,1);var gloucester=PrepareMerchants(app,RegionName.Gloucester,BuildingType.CottonMill,1);
            b.GetLinkById("CanalWorcesterGloucester").SetOwner(p);
            foreach(BuildingType type in Enum.GetValues(typeof(BuildingType)))if(type!=BuildingType.Undefined)
            {
                if(mode==1&&type==BuildingType.Pottery || mode==2&&type==BuildingType.CoalMine)
                {
                    // Pottery I cannot develop; retaining it must still count as no target.
                    var first=p.Mat.GetFirstBuildingOfType(type);p.Mat.GetIndustryList(type).Clear();p.Mat.GetIndustryList(type).Add(first);
                }
                else p.Mat.GetIndustryList(type).Clear();
            }
            var history=new BrassGameHistory();history.SetGame(app);var output=new BonusOutput(history);var useCase=new SellActionUseCase(output,history);
            useCase.StartSellAction();useCase.SelectCard(p.Hand[0]);output.Pump();useCase.SelectIndustrySlot(b.GetBuildingSlotContainingGivenIndustry(tile));output.Pump();
            if(output.Sources.Count>0){useCase.SelectBeerSource(gloucester.MerchantSlots[0],1);output.Pump();}
            Check(tile.IsFlipped&&!g.DevelopMerchantBonus.MerchantDevelopmentNeeded&&g.TurnFlow.CurrentAction==null,"B10 human bonus completes sale");
            Check(output.ModalCount==(mode==2?1:0), "B10 modal is offered exactly when a legal target exists");
        }
    }
    public class CoalOutput : ExtendedNetworkMatrix.NetworkOutput, INetworkActionOutput
    {
        public List<CoalMine> Sources = new List<CoalMine>();
        public new void StartCoalDeliveryUseCase(int count,IEnumerable<CoalMine> sources,ICoalDeliveryResultDelegate callback) { Sources=sources.ToList(); }
    }
    public class BonusOutput : ExtendedNetworkMatrix.SellOutput, ISellActionOutput
    {
        public int ModalCount;
        public BonusOutput(BrassGameHistory history) : base(history) { }
        public new void ShowMerchantDevelopBonus(BrassApplicationGame game,ISelectIndustryOfTypeDelegate callback)
        {
            ModalCount++;
            var types=Enum.GetValues(typeof(BuildingType)).Cast<BuildingType>().Where(t=>t!=BuildingType.Undefined && game.Game.GameProgressInformation.CurrentPlayer.Mat.CanIndustryBeDeveloped(t)).ToList();
            if(types.Count==0)throw new Exception("B10 unusable development modal");
            callback.SelectIndustryOfType(types[0]);
        }
    }
}
