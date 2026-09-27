using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using System.Threading.Tasks;
using BoardGameRules.Entities.Actions;
using BoardGameRules.Entities.Board;
using BoardGameRules.Entities.Board.BuildingSlots;
using BoardGameRules.Entities.Board.Links;
using BoardGameRules.Entities.Board.Merchants;
using BoardGameRules.Entities.Board.Regions;
using BoardGameRules.Entities.Common;
using BoardGameRules.Entities.EventSourcing.Events;
using BoardGameRules.Entities.Game;
using BoardGameRules.Entities.GamePhases;
using BoardGameRules.Entities.Industries;
using BoardGameRules.Entities.Players;
using BrassApplication.Game;
using BrassApplication.GameHistory;
using BrassApplication.UseCases.Game.Actions;
using BrassApplication.UseCases.Game.Actions.BaseAction;
using BrassApplication.UseCases.Game.Actions.BeerDelivery;
using BrassApplication.UseCases.Game.Actions.SellAction;
using BrassApplication.UseCases.Game.Actions.NetworkAction;
using BrassApplication.UseCases.Game.Actions.CoalDelivery;
using BrassApplication.UseCases.Game.Replay;
using BoardGameRules.Entities.Actions.Events;
using BoardGameRules.Entities.GamePhases.Events;
using BoardGameRules.Entities.NumericalResources.Events;
using Frameworks.Persistence.Repository;
using Frameworks.Persistence.RepositoryBackends.InMemory;
using Frameworks.Persistence.GameFactory;

public static class ExtendedNetworkMatrix
{
    static List<object> failures = new List<object>();
    static int assertions;
    static void Check(bool ok, string label, object details) { assertions++; if (!ok) failures.Add(new { label, details }); }
    public static BrassApplicationGame NewGame(int players, int seed)
    {
        var cfg = new GameStartConfiguration((NumberOfPlayers)players); cfg.Seed=seed;
        var g = new BoardGameRulesGame(cfg); var md = new GameMetadata(cfg, GameType.Local); md.GameId="matrix";
        foreach(var pc in cfg.Players) md.JoinGame(new PlayerMetadata { PlayerNetworkId="probe-"+pc.Color, NickName="probe", PlayerStartConfiguration=pc });
        g.StartGame(); return new BrassApplicationGame(g,md);
    }
    public static BaseIndustry Put(BrassApplicationGame app, RegionName rn, BuildingType bt, int level, bool opponent)
    {
        var g=app.Game; var p=opponent ? g.PlayersTrack.Players.First(x=>x!=g.GameProgressInformation.CurrentPlayer) : g.GameProgressInformation.CurrentPlayer;
        var ind=p.Mat.GetIndustryList(bt).First(x=>x.Level==level); p.Mat.RemoveGivenBuildingFromMat(ind);
        g.Board.GetRegionByName(rn).BuildingSlots.First(x=>x.BuildedIndustry==null && x.BuildingTypes.Contains(bt)).BuildIndustry(ind); ind.Build(); return ind;
    }
    static IEnumerable<Link> Links(Board b) { return b.Regions.SelectMany(x=>x.Links).Distinct(); }
    static IEnumerable<Region> Ends(Link l) { return new []{l.Region1,l.Region2,l.Region3}.Where(x=>x!=null); }
    // Independent graph traversal; deliberately does not call the game's route APIs.
    static HashSet<Region> Reach(IEnumerable<Region> start, Link proposed)
    {
        var reached=new HashSet<Region>(start); var todo=new Queue<Region>(reached);
        while(todo.Count>0) foreach(var l in todo.Dequeue().Links.Where(x=>x.HasOwner||x==proposed)) foreach(var n in Ends(l)) if(reached.Add(n)) todo.Enqueue(n);
        return reached;
    }
    static int MarketPrice(int available) { return available<=0?8:(14-available)/2+1; }
    static bool ReferenceEligible(BrassApplicationGame app, Link link, bool second)
    {
        var g=app.Game; var p=g.GameProgressInformation.CurrentPlayer; bool rail=g.GameProgressInformation.CurrentGameEra==GameEra.RailEra;
        if(p.LinkTileCount<=0||link.HasOwner|| (rail ? !(link is Rail):!(link is Canal)))return false;
        var own = g.Board.Regions.Where(r=>r.BuildingSlots.Any(s=>s.BuildedIndustry!=null&&s.BuildedIndustry.Color==p.Color)||r.Links.Any(l=>l.Owner==p)).ToList();
        if(own.Count>0&&!Ends(link).Any(own.Contains))return false;
        if(!rail)return p.Money>=3;
        var connected=Reach(Ends(link),link);
        bool hasCoal=connected.SelectMany(r=>r.BuildingSlots).Any(s=>s.BuildedIndustry is CoalMine && ((CoalMine)s.BuildedIndustry).AvailableCoal>0);
        if(!hasCoal&&!connected.Any(r=>r is BorderRegion))return false;
        int cost=(second?10:5)+(hasCoal?0:MarketPrice(g.Board.CoalMarket.AvailableCoal));
        if(p.Money<cost)return false;
        if(second&&!g.Board.Regions.Any(r=>r.BuildingSlots.Any(s=>s.BuildedIndustry is Brewery && ((Brewery)s.BuildedIndustry).AvailableBeer>0 && (s.BuildedIndustry.Color==p.Color||connected.Contains(r)))))return false;
        return true;
    }
    static void SetMarket(Board b, int amount) { b.CoalMarket.Buy(100); b.CoalMarket.Sell(amount); }
    static void SetOwned(BrassApplicationGame app,string id,bool opponent) { var g=app.Game;g.Board.GetLinkById(id).SetOwner(opponent?g.PlayersTrack.Players.First(x=>x!=g.GameProgressInformation.CurrentPlayer):g.GameProgressInformation.CurrentPlayer); }
    public static object Run()
    {
        assertions=0; failures.Clear(); int queries=0,placements=0,beerCases=0;
        var rng=new Random(708012);
        // Money boundary, all player counts, both eras, empty/full markets,
        // link inventory, ownership, disconnected industries and farm branches.
        for(int caseNo=0;caseNo<600;caseNo++)
        {
            int players=2+caseNo%3; var app=NewGame(players,123+caseNo);var g=app.Game;var p=g.GameProgressInformation.CurrentPlayer;
            bool rail=caseNo%4!=0;g.GameProgressInformation.CurrentGameEra=rail?GameEra.RailEra:GameEra.CanalEra;
            p.Money=new[]{0,2,3,4,5,6,7,10,12,13,15,16,23,30}[rng.Next(14)];p.LinkTileCount=caseNo%5==0?0:(caseNo%5==1?1:14);
            int market=new[]{0,1,2,3,12,13,14}[rng.Next(7)];SetMarket(g.Board,market);
            string era=rail?"Rail":"Canal";
            int topology=caseNo%8;
            if(topology>=1)SetOwned(app,era+"BirminghamOxford",topology==1);
            if(topology>=2)SetOwned(app,era+"BirminghamWalsall",false);
            if(topology>=3)Put(app,RegionName.Stafford,BuildingType.Brewery,2,false);
            if(topology>=4)SetOwned(app,era+"WorcesterSouthWest",false);
            if(topology>=5)Put(app,RegionName.Redditch,BuildingType.CoalMine,2,true);
            if(topology>=6)Put(app,RegionName.Worcester,BuildingType.CottonMill,2,false);
            if(topology>=7){SetOwned(app,era+"RedditchOxford",true);Put(app,RegionName.Central,BuildingType.Brewery,2,true);}
            var expected=Links(g.Board).Where(l=>ReferenceEligible(app,l,false)).Select(l=>l.LinkId).OrderBy(x=>x).ToArray();
            var actual=g.TurnFlow.NetworkAction.GetValidConnectionSlots(g.Board,g.GameProgressInformation).Select(l=>l.LinkId).OrderBy(x=>x).ToArray();queries++;
            Check(expected.SequenceEqual(actual),"first-link candidate set",new{caseNo,players,rail,money=p.Money,market,topology,expected,actual});
        }
        // Every legal ordered rail pair on a resource-free board with own,
        // disconnected beer. Each placement consumes actual market coal.
        foreach(int market in new[]{0,1,2,13,14})
        {
            var template=NewGame(2,900);template.Game.GameProgressInformation.CurrentGameEra=GameEra.RailEra;
            var firstIds=Links(template.Game.Board).Where(l=>l is Rail&&Ends(l).Any(r=>r is BorderRegion)).Select(l=>l.LinkId).ToArray();
            foreach(var firstId in firstIds)foreach(int availableTiles in new[]{1,2})
            {
                var app=NewGame(2,900);var g=app.Game;var p=g.GameProgressInformation.CurrentPlayer;g.GameProgressInformation.CurrentGameEra=GameEra.RailEra;p.Money=100;p.LinkTileCount=availableTiles;
                // Place first while player has no network; brewery is added
                // afterwards solely to isolate second-rail beer availability.
                SetMarket(g.Board,market);var action=g.TurnFlow.StartNetworkAction();action.DiscardCard(p.Hand[0],p);
                var first=g.Board.GetLinkById(firstId);int expectedFirst=5+MarketPrice(market);action.PlaceLink(first);action.SupplyCoal(g.Board.CoalMarket,1);placements++;
                Check(p.Money==100-expectedFirst && action.NeededCoal==0 && first.Owner==p && p.LinkTileCount==availableTiles-1,"first rail payment",new{market,firstId,availableTiles,money=p.Money,expectedFirst});
                Put(app,RegionName.Stafford,BuildingType.Brewery,2,false);
                var expected=Links(g.Board).Where(l=>ReferenceEligible(app,l,true)).Select(l=>l.LinkId).OrderBy(x=>x).ToArray();
                var actual=action.GetValidConnectionSlots(g.Board,g.GameProgressInformation).Select(l=>l.LinkId).OrderBy(x=>x).ToArray();queries++;
                Check(expected.SequenceEqual(actual),"second-link candidate set",new{market,firstId,availableTiles,expected,actual});
                if(actual.Length>0)
                {
                    var second=g.Board.GetLinkById(actual[0]);int before=p.Money, expectedSecond=10+MarketPrice(g.Board.CoalMarket.AvailableCoal);int beforeBeer=g.Board.GetUnflippedBreweries().Sum(b=>b.AvailableBeer);
                    action.PlaceLink(second);action.SupplyCoal(g.Board.CoalMarket,1);action.SupplyBeer(g.Board.GetUnflippedBreweries().First(),1);placements++;
                    Check(p.Money==before-expectedSecond && action.NeededCoal==0 && action.NeededBeer==0 && p.LinkTileCount==0 && g.Board.GetUnflippedBreweries().Sum(b=>b.AvailableBeer)==beforeBeer-1,"double rail final payment",new{market,firstId,second=second.LinkId,money=p.Money,expectedSecond});
                }
            }
        }
        // All brewery positions, ownership alternatives and sold-industry
        // positions on partially connected graphs, checked against own BFS.
        foreach(bool opponent in new[]{false,true})foreach(var breweryRegion in new[]{RegionName.Stafford,RegionName.Coalbrookdale,RegionName.Central,RegionName.SouthWest,RegionName.Derby,RegionName.BurtonOnTrent})
        {
            var app=NewGame(3,77);var g=app.Game;var p=g.GameProgressInformation.CurrentPlayer;g.GameProgressInformation.CurrentGameEra=GameEra.RailEra;
            foreach(string id in new[]{"RailBirminghamOxford","RailBirminghamWorcester","RailWorcesterSouthWest","RailKidderminsterCoalbrookdale","RailBirminghamDudley"})SetOwned(app,id,true);
            var brewery=(Brewery)Put(app,breweryRegion,BuildingType.Brewery,2,opponent);
            foreach(var target in g.Board.Regions)
            {
                bool expected=!opponent||Reach(new[]{target},null).Contains(g.Board.GetRegionContainingGivenIndustry(brewery));
                bool actual=g.TurnFlow.GeneralAction.ConsumingBeer.CanBeerBeSuppliedToSlotFromSource(brewery,target,p);beerCases++;
                Check(expected==actual,"beer reachability",new{opponent,breweryRegion,target=target.Name,expected,actual});
            }
        }
        var ui=RunSellFlowMatrix();var transactionCases=RunNetworkTransactions();var multiSaleCases=RunMultiSaleMatrix();var sellTransactionCases=RunSellTransactions();
        return new{assertions,queryFixtures=queries,executedRailPlacements=placements,beerReachabilityCases=beerCases,sellFlowCases=ui,transactionCases,multiSaleCases,sellTransactionCases,failures=failures.ToArray()};
    }
    static int RunSellFlowMatrix()
    {
        int cases=0;
        foreach(int players in new[]{2,3,4})foreach(bool rail in new[]{false,true})foreach(int required in new[]{0,1,2})foreach(bool merchant in new[]{false,true})
        {
            if(required==0&&merchant)continue;
            var app=NewGame(players,878);var g=app.Game;var p=g.GameProgressInformation.CurrentPlayer;g.GameProgressInformation.CurrentGameEra=rail?GameEra.RailEra:GameEra.CanalEra;
            var i=Put(app,RegionName.Birmingham,BuildingType.Manufacturer,required==0?3:required==1?2:5,false);
            var reg=(BorderRegion)g.Board.GetRegionByName(RegionName.Oxford);foreach(var br in g.Board.Regions.OfType<BorderRegion>())foreach(var s in br.MerchantSlots)s.SetMerchant(new MerchantTile(NumberOfPlayers.Two,new List<BuildingType>()));
            reg.MerchantSlots[0].SetMerchant(new MerchantTile(NumberOfPlayers.Two,new List<BuildingType>{BuildingType.Manufacturer})); if(!merchant)reg.MerchantSlots[0].ConsumeBeer(1);
            SetOwned(app,(rail?"Rail":"Canal")+"BirminghamOxford",false);
            int breweryBeer=required-(merchant?1:0);if(breweryBeer>0){var b=(Brewery)Put(app,RegionName.Stafford,BuildingType.Brewery,2,false);b.AvailableBeer=breweryBeer;}
            var history=new BrassGameHistory();history.SetGame(app);var output=new SellOutput(history);var uc=new SellActionUseCase(output,history);output.UseCase=uc;
            int hand=p.Hand.Count,actions=g.TurnFlow.GetRemainingActionsCount(),income=p.IncomeMarkerPosition;
            try
            {
                uc.StartSellAction();uc.SelectCard(p.Hand[0]);output.Pump();uc.SelectIndustrySlot(g.Board.GetBuildingSlotContainingGivenIndustry(i));output.Pump();
                int safety=0;while(g.TurnFlow.CurrentAction!=null&&safety++<10)
                {
                    if(output.Sources.Count>0){var src=output.Sources.FirstOrDefault(x=>x is MerchantSlot)??output.Sources[0];output.Sources.Clear();uc.SelectBeerSource(src,1);output.Pump();}
                    else if(output.Choice){output.Choice=false;uc.EndSellAction();output.Pump();}
                    else break;
                }
                Check(i.IsFlipped && p.Hand.Count==hand-1 && g.TurnFlow.GetRemainingActionsCount()==actions-1 && g.TurnFlow.CurrentAction==null,"human sell flow completion",new{players,rail,required,merchant,flipped=i.IsFlipped,currentAction=g.TurnFlow.CurrentAction!=null,pending=output.Queue.Count,sourceCount=output.Sources.Count});
                Check(reg.MerchantSlots[0].AvailableBeer==(merchant&&required>0?0:merchant?1:0),"merchant consumption",new{players,rail,required,merchant});
            }
            catch(Exception e){Check(false,"human sell exception",new{players,rail,required,merchant,error=e.ToString()});}
            cases++;
        }
        return cases;
    }
    static string Fingerprint(BrassApplicationGame app)
    {
        var g=app.Game;return String.Join("|",new[]{g.GameProgressInformation.CurrentGameEra.ToString(),g.GameProgressInformation.CurrentRoundNumber.ToString(),g.TurnFlow.GetRemainingActionsCount().ToString(),g.Board.CoalMarket.AvailableCoal.ToString(),g.Board.IronMarket.AvailableIron.ToString()}
        .Concat(g.PlayersTrack.Players.Select(p=>p.Color+":"+p.Money+":"+p.LinkTileCount+":"+p.IncomeMarkerPosition+":"+String.Join(",",p.Hand.Select(c=>c.ToString()))+":"+p.DiscardPile.Count+":"+String.Join(",",Enum.GetValues(typeof(BuildingType)).Cast<BuildingType>().Where(t=>t!=BuildingType.Undefined).Select(t=>t+"="+String.Join(";",p.Mat.GetIndustryList(t).Select(i=>i.UniqueId))))))
        .Concat(Links(g.Board).OrderBy(l=>l.LinkId).Select(l=>l.LinkId+":"+(l.Owner==null?"-":l.Owner.Color.ToString())))
        .Concat(g.Board.Regions.SelectMany(r=>r.BuildingSlots).Select(s=>s.UniqueId+":"+(s.BuildedIndustry==null?"-":s.BuildedIndustry.UniqueId+":"+s.BuildedIndustry.IsFlipped+":"+s.BuildedIndustry.GetResource()))));
    }
    static void PutViaEvents(BrassApplicationGame app,RegionName rn,BuildingType bt,int level)
    {
        var g=app.Game;var p=g.GameProgressInformation.CurrentPlayer;
        while(p.Mat.GetFirstBuildingOfType(bt).Level<level)g.ApplyEvent(new IndustryDeveloped(p.Color,bt));
        var slot=g.Board.GetRegionByName(rn).BuildingSlots.First(s=>s.BuildedIndustry==null&&s.BuildingTypes.Contains(bt));g.ApplyEvent(new IndustryBuilt(p.Color,bt,slot.UniqueId));
    }
    static int RunNetworkTransactions()
    {
        int cases=0;
        for(int players=2;players<=4;players++)for(int stage=0;stage<=6;stage++)
        {
            var app=NewGame(players,442);var g=app.Game;var p=g.GameProgressInformation.CurrentPlayer;
            // Event-based fixture enables original repository/history rebuild.
            g.ApplyEvent(new RailEraStarted());g.ApplyEvent(new MoneyLoaned(p.Color,100));g.ApplyEvent(new LinkPlaced(p.Color,"RailBirminghamOxford"));
            PutViaEvents(app,RegionName.Stafford,BuildingType.Brewery,2);PutViaEvents(app,RegionName.Derby,BuildingType.Brewery,2);
            var repo=new BaseRepository(new InMemoryGameStorage(),new InMemorySnapshotStorage(),new BrassBirminghamFactory());repo.Save(app);
            var history=new BrassGameHistory();history.Repository=repo;history.SetGame(app);var output=new NetworkOutput();var uc=new NetworkActionUseCase(output,history);
            string baseline=Fingerprint(app);int baselineMoney=p.Money;
            try
            {
                uc.StartNetworkAction();uc.SelectCard(p.Hand[0]);output.Pump();
                if(stage>=1){uc.LinkSelected(g.Board.GetLinkById("RailBirminghamDudley"));output.Pump();}
                if(stage>=2){uc.SelectCoalSource(g.Board.CoalMarket,1);output.Pump();}
                if(stage>=3){uc.SelectSecondLink();}
                if(stage>=4){uc.LinkSelected(g.Board.GetLinkById("RailBirminghamWalsall"));output.Pump();}
                if(stage>=5){uc.SelectCoalSource(g.Board.CoalMarket,1);output.Pump();}
                if(stage>=6){uc.SelectBeerSource(g.Board.GetUnflippedBreweries().First(),1);output.Pump();history.UndoAction();}
                else uc.CancelAction();
                Check(Fingerprint(history.CurrentGame)==baseline,"network cancel/undo restores state",new{players,stage,equal=Fingerprint(history.CurrentGame)==baseline});
                var restored=history.CurrentGame.Game;Check(restored.TurnFlow.CurrentAction==null && restored.TurnFlow.NetworkAction.NeededCoal==0 && restored.TurnFlow.NetworkAction.NeededBeer==0,"network reset completion",new{players,stage});
                // A new action after cancellation must not retain counters.
                uc.StartNetworkAction();uc.SelectCard(restored.GameProgressInformation.CurrentPlayer.Hand[0]);output.Pump();uc.LinkSelected(restored.Board.GetLinkById("RailBirminghamDudley"));output.Pump();uc.SelectCoalSource(restored.Board.CoalMarket,1);output.Pump();uc.EndNetworkAction();output.Pump();
                Check(restored.GameProgressInformation.CurrentPlayer.Money==baselineMoney-6 && restored.TurnFlow.NetworkAction.NeededBeer==0,"new one-rail action after cancel",new{players,stage,money=restored.GameProgressInformation.CurrentPlayer.Money,baselineMoney});
            }
            catch(Exception e){Check(false,"network transaction exception",new{players,stage,error=e.ToString()});}
            cases++;
        }
        return cases;
    }
    static int RunMultiSaleMatrix()
    {
        int cases=0;
        foreach(int players in new[]{2,3,4})foreach(bool rail in new[]{false,true})foreach(bool stopEarly in new[]{false,true})
        {
            var app=NewGame(players,981);var g=app.Game;var p=g.GameProgressInformation.CurrentPlayer;g.GameProgressInformation.CurrentGameEra=rail?GameEra.RailEra:GameEra.CanalEra;
            var a=Put(app,RegionName.Birmingham,BuildingType.Manufacturer,3,false);var b=Put(app,RegionName.Birmingham,BuildingType.Manufacturer,2,false);var c=Put(app,RegionName.Birmingham,BuildingType.Manufacturer,5,false);
            foreach(var br in g.Board.Regions.OfType<BorderRegion>())foreach(var s in br.MerchantSlots)s.SetMerchant(new MerchantTile(NumberOfPlayers.Two,new List<BuildingType>()));
            var ox=(BorderRegion)g.Board.GetRegionByName(RegionName.Oxford);ox.MerchantSlots[0].SetMerchant(new MerchantTile(NumberOfPlayers.Two,new List<BuildingType>{BuildingType.Manufacturer}));ox.MerchantSlots[1].SetMerchant(new MerchantTile(NumberOfPlayers.Two,new List<BuildingType>{BuildingType.CottonMill,BuildingType.Manufacturer,BuildingType.Pottery}));
            var brew=(Brewery)Put(app,RegionName.Stafford,BuildingType.Brewery,2,false);brew.AvailableBeer=1;SetOwned(app,(rail?"Rail":"Canal")+"BirminghamOxford",false);
            var h=new BrassGameHistory();h.SetGame(app);var o=new SellOutput(h);var uc=new SellActionUseCase(o,h);int hand=p.Hand.Count,actions=g.TurnFlow.GetRemainingActionsCount(),income=p.IncomeMarkerPosition;
            try
            {
                uc.StartSellAction();uc.SelectCard(p.Hand[0]);o.Pump();uc.SelectIndustrySlot(g.Board.GetBuildingSlotContainingGivenIndustry(a));o.Pump();
                if(stopEarly){uc.EndSellAction();o.Pump();}
                else
                {
                    uc.SelectNextIndustryToSell();uc.SelectIndustrySlot(g.Board.GetBuildingSlotContainingGivenIndustry(b));o.Pump();uc.SelectBeerSource(ox.MerchantSlots[0],1);o.Pump();
                    uc.SelectNextIndustryToSell();uc.SelectIndustrySlot(g.Board.GetBuildingSlotContainingGivenIndustry(c));o.Pump();uc.SelectBeerSource(ox.MerchantSlots[1],1);o.Pump();
                }
                Check(a.IsFlipped && b.IsFlipped==!stopEarly && c.IsFlipped==!stopEarly && p.Hand.Count==hand-1 && g.TurnFlow.GetRemainingActionsCount()==actions-1 && g.TurnFlow.CurrentAction==null,"multi-sale one card one action",new{players,rail,stopEarly,a=a.IsFlipped,b=b.IsFlipped,c=c.IsFlipped});
                Check(stopEarly?brew.AvailableBeer==1:brew.AvailableBeer==0&&brew.IsFlipped,"multi-sale brewery depletion",new{players,rail,stopEarly});
            }
            catch(Exception e){Check(false,"multisale exception",new{players,rail,stopEarly,error=e.ToString()});}
            cases++;
        }
        return cases;
    }
    static int RunSellTransactions()
    {
        int cases=0;
        foreach(int players in new[]{2,3,4})for(int stage=0;stage<=5;stage++)
        {
            var app=NewGame(players,878);var g=app.Game;var p=g.GameProgressInformation.CurrentPlayer;g.ApplyEvent(new RailEraStarted());
            PutViaEvents(app,RegionName.Birmingham,BuildingType.Manufacturer,2);PutViaEvents(app,RegionName.Birmingham,BuildingType.Manufacturer,3);PutViaEvents(app,RegionName.Birmingham,BuildingType.Manufacturer,5);
            PutViaEvents(app,RegionName.Stafford,BuildingType.Brewery,2);PutViaEvents(app,RegionName.Derby,BuildingType.Brewery,2);
            var birmingham=g.Board.GetRegionByName(RegionName.Birmingham);var target=g.Board.Regions.OfType<BorderRegion>().First(r=>r.MerchantSlots.Any(s=>s.MerchantTile!=null&&s.MerchantTile.BuildingTypes.Contains(BuildingType.Manufacturer)));
            var path=g.Board.GetPossibleConnections(birmingham,typeof(Rail))[g.Board.Regions.IndexOf(target)];var previous=birmingham;
            foreach(var next in path){var link=previous.Links.First(l=>l is Rail&&Ends(l).Contains(next));g.ApplyEvent(new LinkPlaced(p.Color,link.LinkId));previous=next;}
            var repo=new BaseRepository(new InMemoryGameStorage(),new InMemorySnapshotStorage(),new BrassBirminghamFactory());repo.Save(app);var h=new BrassGameHistory();h.Repository=repo;h.SetGame(app);var o=new SellOutput(h);var uc=new SellActionUseCase(o,h);
            string baseline=Fingerprint(app);var slots=birmingham.BuildingSlots.Where(s=>s.BuildedIndustry!=null).ToList();
            try
            {
                uc.StartSellAction();uc.SelectCard(p.Hand[0]);o.Pump();
                if(stage>=1){uc.SelectIndustrySlot(slots.First(s=>s.BuildedIndustry.Level==5));o.Pump();}
                if(stage>=2){uc.SelectBeerSource(g.Board.GetUnflippedBreweries().First(),1);o.Pump();}
                if(stage>=3){uc.SelectBeerSource(g.Board.GetUnflippedBreweries().First(),1);o.Pump();}
                if(stage>=4){uc.SelectNextIndustryToSell();uc.SelectIndustrySlot(slots.First(s=>s.BuildedIndustry.Level==3));o.Pump();}
                if(stage>=5){uc.SelectNextIndustryToSell();uc.SelectIndustrySlot(slots.First(s=>s.BuildedIndustry.Level==2));o.Pump();if(g.TurnFlow.CurrentAction!=null){uc.SelectBeerSource(g.Board.GetUnflippedBreweries().First(),1);o.Pump();}h.UndoAction();}
                else uc.CancelAction();
                Check(Fingerprint(h.CurrentGame)==baseline,"sell cancel/undo restores all state",new{players,stage,equal=Fingerprint(h.CurrentGame)==baseline});
                Check(h.CurrentGame.Game.TurnFlow.CurrentAction==null,"sell cancel leaves no action",new{players,stage});
            }
            catch(Exception e){Check(false,"sell transaction exception",new{players,stage,error=e.ToString()});}
            cases++;
        }
        return cases;
    }
    public class NetworkOutput:INetworkActionOutput
    {
        public Queue<IReplayDelegate> Queue=new Queue<IReplayDelegate>();
        public void Pump(){int guard=0;while(Queue.Count>0&&guard++<25)Queue.Dequeue().ReplayCompleted();if(guard>=25)throw new Exception("Replay loop");}
        public void PresentEvents(ReadOnlyCollection<IEvent> e,PlayerMetadata pm,IReplayDelegate d){Queue.Enqueue(d);}
        public void HighlightValidConnections(BrassApplicationGame g,List<Link> l,ILinkSelectedDelegate d){}public void ShowAnotherTrackSelectionPanel(List<CantUseLinkActionReason> r){}
        public void HideNumberOfTracksToBuildSelectionPanel(){}public void ShowCardsToDiscard(ISelectCardDelegate d){}public void HideCards(){}public void HideCoalDelivery(){}public void HideBeerDelivery(){}
        public void ShowActionInProgressView(IActionInProgressCancelDelegate d,int a,int b){}public void ShowPlayerInfoToSelectCardToDiscard(){}public void ShowPlayerInfoToSelectLinks(){}public void ShowPlayerInfoToSelectIfAnotherTrackShouldBePlaced(){}public void AllowRegionInteractions(bool b){}
        public void StartBeerDeliveryUseCase(int n,IEnumerable<Brewery> s,IBeerDeliveryResultDelegate d){}public void StartCoalDeliveryUseCase(int n,IEnumerable<CoalMine> s,ICoalDeliveryResultDelegate d){}
        public void ReplayEvents(ReadOnlyCollection<IEvent> e){}public void SetFastForwardButtonActive(bool b){}
        public void HideActionInProgressView(){}public void ShowPlayerActionsButtons(){}public void UpdateGameStatus(PlayerMetadata m){}
        public void RestoreCardsView(){}public void ShowEndTurnButton(){}public void HideReplayView(){}public Task WaitForTweensEnd(){return Task.FromResult(0);}public void HideIronDelivery(){}
    }
    public class SellOutput : ISellActionOutput
    {
        public Queue<IReplayDelegate> Queue=new Queue<IReplayDelegate>();public List<IBeerSource> Sources=new List<IBeerSource>();public bool Choice;public SellActionUseCase UseCase;
        BrassGameHistory history; BeerDeliveryUseCase beer;
        public SellOutput(BrassGameHistory h){history=h;beer=new BeerDeliveryUseCase(new BeerOutput(this),h);}
        public void Pump(){int guard=0;while(Queue.Count>0&&guard++<25)Queue.Dequeue().ReplayCompleted();if(guard>=25)throw new Exception("Replay loop");}
        public void PresentEvents(ReadOnlyCollection<IEvent> e,PlayerMetadata pm,IReplayDelegate d){Queue.Enqueue(d);}
        public void StartBeerDeliveryUseCase(int n,IEnumerable<IBeerSource> src,IBeerDeliveryResultDelegate d){Sources=src.ToList();beer.StartBeerDelivery(n,Sources,d);if(Sources.Count==1)Sources.Clear();}
        public void ShowMerchantDevelopBonus(BrassApplicationGame g,ISelectIndustryOfTypeDelegate d){throw new Exception("Unexpected develop bonus in Oxford fixture");}
        public void ShowNextSellChoicePanel(List<BuildingSlot> l){Choice=true;}
        public void HideNextSellChoicePanel(){Choice=false;}
        public void ShowPlayerInfoToSelectCardToDiscard(IActionInProgressCancelDelegate d,int a,int b){}
        public void ShowCardsToDiscard(ISelectCardDelegate d){} public void HideCards(){}
        public void HighlightValidBuildingSlots(Board b,List<BuildingSlot> l,ISelectIndustryDelegate d){}
        public void ShowPlayerInfoToSelectIndustry(IActionInProgressCancelDelegate d){}
        public void HideIndustriesDetailsView(){}public void SetFastForwardButtonActive(bool b){}
        public void HideActionInProgressView(){}public void ShowPlayerActionsButtons(){}public void UpdateGameStatus(PlayerMetadata m){}
        public void RestoreCardsView(){}public void ShowEndTurnButton(){}public void HideReplayView(){}public Task WaitForTweensEnd(){return Task.FromResult(0);}public void HideIronDelivery(){}
        class BeerOutput:IBeerDeliveryOutput
        {
            SellOutput owner;public BeerOutput(SellOutput o){owner=o;}
            public void HighlightBreweries(Board b,List<BuildingSlot> s,ISelectIndustryDelegate d){}
            public void HighlightMerchantSlots(Board b,IEnumerable<MerchantSlot> s,ISelectMerchantSlotDelegate d){}
            public void ShowPlayerInfoToSelectBrewery(){}public void ReplayEvents(ReadOnlyCollection<IEvent> e){}
        }
    }
}
