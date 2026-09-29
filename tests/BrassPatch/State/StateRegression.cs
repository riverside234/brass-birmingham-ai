using System;
using System.Collections.Generic;
using System.Linq;
using BoardGameRules.Entities.Actions;
using BoardGameRules.Entities.Actions.Events;
using BoardGameRules.Entities.Board;
using BoardGameRules.Entities.Board.BuildingSlots;
using BoardGameRules.Entities.Board.Events;
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
using BrassApplication.AI;
using BrassApplication.AI.AiBehaviors;
using BrassApplication.AI.RuleBasedSystem;
using BrassApplication.AI.RuleBasedSystem.RuleSelectors;
using BrassApplication.Game;
using BrassApplication.GameHistory;
using BrassApplication.UseCases.Game.FinishedGameReview;
using BrassApplication.UseCases.Game.Replay;
using Frameworks.Persistence.GameFactory;
using Frameworks.Persistence.Repository;
using Frameworks.Persistence.RepositoryBackends.InMemory;
using BrassApplication.UseCases.Menu.MainMenu;
using InterfaceAdapters.Menu.MainMenu;
using InterfaceAdapters.Common;
using InterfaceAdapters.Routing;

// Compile against the staging Assembly-CSharp.dll. All assertions enter production APIs.
// In-memory fixtures do not read or write user saves, invoke Unity, or transmit analytics.
public static class StateRegression
{
    private static int assertions;
    private static int replayCutoffs;
    private static int fullGames;
    private static void Check(bool condition, string description)
    {
        assertions++;
        if (!condition) throw new Exception("State regression failed: " + description);
    }

    private sealed class Case
    {
        public BaseRepository Repository;
        public BrassApplicationGame App;
    }

    private static Case NewCase(int players = 2, bool intro = false, int seed = 20260927)
    {
        var config = new GameStartConfiguration((NumberOfPlayers)players) { Seed = seed, IsIntroductoryGame = intro };
        var metadata = new GameMetadata(config, GameType.Local);
        foreach (var start in config.Players)
        {
            var player = new PlayerMetadata { PlayerStartConfiguration = start, PlayerNetworkId = "fixture-" + start.Color,
                NickName = "Fixture " + start.Color, AiType = AIType.Hard };
            player.AIBehaviour = AiBehaviorFactory.CreateAiBehavior(player);
            metadata.JoinGame(player);
        }
        var repository = new BaseRepository(new InMemoryGameStorage(), new InMemorySnapshotStorage(), new BrassBirminghamFactory());
        var app = repository.Create(metadata);
        app.Game.StartGame();
        repository.Save(app);
        return new Case { Repository = repository, App = app };
    }

    private static void Pass(BrassApplicationGame app)
    {
        while (app.Game.TurnFlow.GetRemainingActionsCount() > 0)
        {
            var player = app.Game.GameProgressInformation.CurrentPlayer;
            var action = app.Game.TurnFlow.StartPassAction();
            action.Pass(player.Hand[0], player);
            app.Game.TurnFlow.ConfirmCurrentAction();
        }
    }

    private static void CompleteTurn(BrassApplicationGame app)
    {
        app.Game.GameFlow.EndTurn();
        if (app.Game.GameFlow.IsLastTurnOver())
        {
            if (app.Game.GameFlow.IsSellIndustryNecessary())
            {
                var selector = RuleSelectorFactory.CreatePrioritySelector(RuleSetFactory.CreateSellIndustryRuleSet());
                selector.Select(app).MakeAction(app, false)();
            }
            app.Game.GameFlow.EndRound();
        }
    }

    private static void AiTurn(BrassApplicationGame app)
    {
        var behavior = app.Metadata.GetPlayerMetadataByColor(app.Game.GameProgressInformation.CurrentPlayer.Color).AIBehaviour;
        while (app.Game.TurnFlow.GetRemainingActionsCount() > 0)
        {
            behavior.GetNextAIMove(app)();
            app.Game.TurnFlow.ConfirmCurrentAction();
        }
    }

    private static void Loans()
    {
        for (int income = -10; income <= -6; income++)
        {
            var game = NewCase().App.Game;
            var player = game.GameProgressInformation.CurrentPlayer;
            player.IncomeMarkerPosition = income + 10;
            int revision = game.Revision, money = player.Money, cards = player.Hand.Count;
            var reasons = game.TurnFlow.LoanAction.GetLoanActionNotPossibleReasons(game.GameProgressInformation);
            Check((reasons.Count == 0) == (income >= -7), "B07 menu boundary " + income);
            bool rejected = false;
            try { game.TurnFlow.LoanAction.Loan(player.Color); }
            catch (InvalidOperationException) { rejected = true; }
            Check(rejected == (income < -7), "B07 execution boundary " + income);
            if (rejected)
            {
                Check(game.Revision == revision && player.Money == money && player.Hand.Count == cards && player.IncomeLevel == income,
                    "B07 rejected loan has no side effects");
            }
            else Check(player.Money == money + 30 && player.IncomeLevel == income - 3, "B07 loan result");
        }
    }

    private static void FinalRound()
    {
        foreach (int count in new[] { 2, 3, 4 })
        foreach (GameEra era in new[] { GameEra.CanalEra, GameEra.RailEra })
        foreach (bool last in new[] { false, true })
        foreach (int income in new[] { -1, 2 })
        {
            var game = NewCase(count).App.Game;
            var progress = game.GameProgressInformation;
            progress.CurrentGameEra = era;
            progress.CurrentRoundNumber = game.GameFlow.GetNumberOfRounds((NumberOfPlayers)count) - (last ? 0 : 1);
            progress.CurrentTurnNumber = count;
            var player = progress.CurrentPlayer;
            player.Money = 0;
            player.IncomeMarkerPosition = income < 0 ? income + 10 : 15; // marker 15 is income +2
            player.VP = 10;
            while (game.TurnFlow.GetRemainingActionsCount() > 0) game.TurnFlow.DecreaseRemainingActionsCount();
            int beforeEvents = game.GetUncommittedEvents().Count;
            game.GameFlow.EndTurn();
            bool skipped = era == GameEra.RailEra && last;
            Check(game.GameFlow.IsSellIndustryNecessary() == (income < 0 && !skipped), "B01 debt gate");
            game.GameFlow.EndRound();
            var events = game.GetUncommittedEvents().Skip(beforeEvents).ToList();
            Check(events.Count(e => e is RoundEnded) == 1, "B01 round completes once");
            Check(events.Any(e => e is IncomeMoneyPaid) == !skipped, "B01 income event boundary");
            if (skipped)
            {
                Check(player.VP == 10 && player.Money == 0, "B01 final Rail preserves cash and VP");
                Check(!events.Any(e => e.GetType().Name == "IndustrySellingStarted"), "B01 no debt event");
            }
        }
        // Scorable industry must survive final-round insolvency.
        var withAsset = NewCase().App.Game;
        var p = withAsset.GameProgressInformation.CurrentPlayer;
        var tile = p.Mat.GetIndustryList(BuildingType.CottonMill).First(i => i.Level == 2);
        p.Mat.RemoveGivenBuildingFromMat(tile);
        var slot = withAsset.Board.GetRegionByName(RegionName.Birmingham).BuildingSlots.First(s => s.CanBuildIndustryType(BuildingType.CottonMill));
        slot.BuildIndustry(tile); tile.IsFlipped = true;
        p.Money = 0; p.IncomeMarkerPosition = 9; p.VP = 10;
        withAsset.GameProgressInformation.CurrentGameEra = GameEra.RailEra;
        withAsset.GameProgressInformation.CurrentRoundNumber = 10;
        withAsset.GameProgressInformation.CurrentTurnNumber = 2;
        while (withAsset.TurnFlow.GetRemainingActionsCount() > 0) withAsset.TurnFlow.DecreaseRemainingActionsCount();
        withAsset.GameFlow.EndTurn(); withAsset.GameFlow.EndRound();
        Check(ReferenceEquals(slot.BuildedIndustry, tile) && p.VP == 10 + tile.VPForBuilding, "B01 final Rail retains scorable asset");
    }

    private static void Finish(BoardGameRulesGame game)
    {
        var zero = game.PlayersTrack.Players.ToDictionary(p => p.Color, p => 0);
        game.ApplyEvent(new GameEnded(zero, zero, zero));
    }

    private static void Rankings()
    {
        foreach (int count in new[] { 2, 3, 4 })
        {
            foreach (int winner in Enumerable.Range(0, count))
            {
                var game = NewCase(count).App.Game;
                foreach (var player in game.PlayersTrack.Players)
                { player.VP = 100; player.IncomeMarkerPosition = 12; player.Money = 0; }
                var expected = game.PlayersTrack.Players[winner];
                expected.IncomeMarkerPosition = 11; expected.Money = 10;
                Finish(game);
                Check(game.GameProgressInformation.EndGameRanking[0] == expected.Color, "B05 cash beats income band position for every seat");
                Check(game.GameProgressInformation.EndGamePoints[expected.Color].Item5 == 0, "B05 winner rank");
            }
            var tied = NewCase(count);
            foreach (var player in tied.App.Game.PlayersTrack.Players)
            { player.VP = 100; player.IncomeMarkerPosition = 11 + ((int)player.Color % 2); player.Money = 10; }
            Finish(tied.App.Game);
            Check(tied.App.Game.GameProgressInformation.EndGamePoints.Values.All(row => row.Item5 == 0), "B06 shared ranks for all tied winners");
            var output = new SummaryOutput();
            var history = new BrassGameHistory { Repository = tied.Repository }; history.SetGame(tied.App);
            new FinishedGameReviewUseCase(output, history).ShowGameSummaryTable();
            Check(output.Names.All(n => n.Contains("joint winner")), "B06 each tied winner explicitly labelled");
            Check(output.Places.All(p => p == 0), "B06 presenter receives shared ranks");
        }
        var lower = NewCase(4).App.Game;
        foreach (var p in lower.PlayersTrack.Players) { p.VP = 90; p.IncomeMarkerPosition = 10; p.Money = 0; }
        lower.PlayersTrack.Players[0].VP = 100;
        lower.PlayersTrack.Players[3].VP = 80;
        Finish(lower);
        Check(lower.PlayersTrack.Players.Select(p => lower.GameProgressInformation.EndGamePoints[p.Color].Item5).SequenceEqual(new[] { 0, 1, 1, 3 }), "B06 competition ranks with lower tie");
        var bigCash = NewCase().App.Game;
        bigCash.PlayersTrack.Players[0].VP = 1; bigCash.PlayersTrack.Players[0].Money = 0;
        bigCash.PlayersTrack.Players[1].VP = 0; bigCash.PlayersTrack.Players[1].Money = int.MaxValue;
        Finish(bigCash);
        Check(bigCash.GameProgressInformation.EndGameRanking[0] == bigCash.PlayersTrack.Players[0].Color, "B05 lexicographic order has no cash weight overflow");
    }

    private static void BonusOwnership()
    {
        var first = NewCase().App.Game;
        var second = NewCase(seed: 31).App.Game;
        first.ApplyEvent(new MerchantBonusReceived(BonusType.Develop, first.GameProgressInformation.CurrentPlayer.Color, RegionName.Gloucester));
        Check(first.DevelopMerchantBonus.MerchantDevelopmentNeeded && !second.DevelopMerchantBonus.MerchantDevelopmentNeeded, "B12 receipt belongs to originating game");
        while (first.TurnFlow.GetRemainingActionsCount() > 0) first.TurnFlow.DecreaseRemainingActionsCount();
        int revision = first.Revision, count = first.GetUncommittedEvents().Count;
        bool rejected = false;
        try { first.GameFlow.EndTurn(); } catch (InvalidOperationException) { rejected = true; }
        Check(rejected && first.Revision == revision && first.GetUncommittedEvents().Count == count, "B12 pending bonus rejection preserves revision");
        first.DevelopMerchantBonus.FinishDevelop();
        Check(!first.DevelopMerchantBonus.MerchantDevelopmentNeeded && !second.DevelopMerchantBonus.MerchantDevelopmentNeeded, "B12 finish belongs to originating game");
        second.ApplyEvent(new MerchantBonusReceived(BonusType.Develop, second.GameProgressInformation.CurrentPlayer.Color, RegionName.Gloucester));
        first.GameFlow.EndTurn();
        Check(second.DevelopMerchantBonus.MerchantDevelopmentNeeded, "B12 ending other game's turn preserves pending choice");
    }

    private static void ReplayCutoffs()
    {
        Case found = null;
        int startRevision = 0;
        for (int seed = 20260927; seed < 20260947 && found == null; seed++)
        {
            var candidate = NewCase(seed: seed);
            for (int turn = 0; turn < 15 && found == null; turn++)
            {
                startRevision = candidate.App.Game.Revision;
                AiTurn(candidate.App);
                bool bonus = candidate.App.Game.GetUncommittedEvents().OfType<MerchantBonusReceived>().Any(e => e.BonusType == BonusType.Develop);
                CompleteTurn(candidate.App); candidate.Repository.Save(candidate.App);
                if (bonus) found = candidate;
            }
        }
        Check(found != null, "B12 locate naturally played Gloucester trace");
        var events = found.Repository.Advanced.GameStorage.LoadEvents(found.App.Metadata.GameId, startRevision, int.MaxValue);
        for (int cutoff = 0; cutoff <= events.Count; cutoff++)
        {
            var latest = found.Repository.Load(found.App.Metadata.GameId);
            var history = new BrassGameHistory { Repository = found.Repository }; history.SetGame(latest);
            var watcher = latest.Metadata.GetPlayerMetadataByColor(latest.Game.GameProgressInformation.CurrentPlayer.Color);
            watcher.LastSeenEventRevision = startRevision;
            var output = new ReplayOutput();
            var replay = new ReplayUseCase(output, history);
            replay.PresentEvents(events, watcher, null);
            for (int step = 0; step < cutoff && replay.GetIsReplayInProgress(); step++) output.Callback.EventReplayCompleted(output.LastEvent);
            int guard = 0;
            while (replay.GetIsReplayInProgress()) { replay.SkipReplay(); if (++guard > 20) throw new Exception("Replay skip guard"); }
            Check(ReferenceEquals(history.CurrentGame, latest), "B12 skip restores latest object");
            Check(!latest.Game.DevelopMerchantBonus.MerchantDevelopmentNeeded, "B12 cutoff leaves latest bonus complete");
            Pass(latest); CompleteTurn(latest);
            replayCutoffs++;
        }
    }

    private static void IntroReload()
    {
        foreach (int players in new[] { 2, 3, 4 })
        {
            var test = NewCase(players, true);
            int turns = 0;
            while (!test.App.Game.GameProgressInformation.IsGameOver)
            {
                Pass(test.App); CompleteTurn(test.App);
                if (++turns > 100) throw new Exception("Intro game guard");
            }
            test.Repository.Save(test.App); fullGames++;
            for (int load = 0; load < 3; load++)
            {
                var restored = test.Repository.Load(test.App.Metadata.GameId);
                var progress = restored.Game.GameProgressInformation;
                Check(progress.IntroductoryPoints != null && progress.IntroductoryPoints.Count == players, "B13 reconstructed component dictionary");
                foreach (var player in restored.Game.PlayersTrack.Players)
                {
                    Check(progress.IntroductoryPoints[player.Color].Equals(test.App.Game.GameProgressInformation.IntroductoryPoints[player.Color]), "B13 identical component breakdown");
                    Check(player.VP == test.App.Game.PlayersTrack.GetPlayerByColor(player.Color).VP, "B13 reload doesn't duplicate VP");
                    Check(progress.EndGamePoints[player.Color].Equals(test.App.Game.GameProgressInformation.EndGamePoints[player.Color]), "B06 persisted shared ranks");
                }
                var history = new BrassGameHistory { Repository = test.Repository }; history.SetGame(restored);
                var output = new SummaryOutput();
                new FinishedGameReviewUseCase(output, history).ShowGameSummaryTable();
                Check(output.Names.Count == players && output.Intro, "B13 production results table succeeds after reload");
            }
        }
        var componentsCase = NewCase();
        var game = componentsCase.App.Game; var p = game.GameProgressInformation.CurrentPlayer;
        game.ApplyEvent(new IntroductoryGameAdditionalMoneyVictoryPointsChanged(p.Color, 15));
        game.ApplyEvent(new IntroductoryGameAdditionalIncomeVictoryPointsChanged(p.Color, -7));
        game.ApplyEvent(new IntroductoryGameAdditionalIndustryVictoryPointsChanged(p.Color, 22));
        componentsCase.Repository.Save(componentsCase.App);
        var row = componentsCase.Repository.Load(componentsCase.App.Metadata.GameId).Game.GameProgressInformation.IntroductoryPoints[p.Color];
        Check(row.Item1 == 15 && row.Item2 == -7 && row.Item3 == 22, "B13 recorded values preserve negative income and industry bonus");
    }

    private static void EraReload()
    {
        var test = NewCase();
        int turns = 0;
        while (test.App.Game.GameProgressInformation.CurrentGameEra == GameEra.CanalEra)
        {
            AiTurn(test.App); CompleteTurn(test.App);
            if (++turns > 35) throw new Exception("Canal game guard");
        }
        test.Repository.Save(test.App);
        var restored = test.Repository.Load(test.App.Metadata.GameId);
        foreach (var p in restored.Game.PlayersTrack.Players)
            Check(p.VPFromBonusesInCurrentEra == 0, "B16 Canal merchant counter reset on reload");
        foreach (var app in new[] { test.App, restored })
        {
            turns = 0;
            while (!app.Game.GameProgressInformation.IsGameOver)
            { Pass(app); CompleteTurn(app); if (++turns > 30) throw new Exception("Rail game guard"); }
            fullGames++;
        }
        foreach (var p in restored.Game.PlayersTrack.Players)
        {
            var original = test.App.Game.PlayersTrack.GetPlayerByColor(p.Color);
            Check(p.VP == original.VP, "B16 total score identical after reload");
            Check(restored.Game.GameProgressInformation.EndGamePoints[p.Color].Equals(test.App.Game.GameProgressInformation.EndGamePoints[p.Color]), "B16 all final result components identical");
        }
    }

    private static void ExhaustedManufacturer()
    {
        foreach (int players in new[] { 3, 4 })
        foreach (GameEra era in new[] { GameEra.CanalEra, GameEra.RailEra })
        foreach (int round in new[] { 1, 2 })
        foreach (AIType difficulty in new[] { AIType.Easy, AIType.Medium, AIType.Hard })
        {
            var test = NewCase(players);
            var game = test.App.Game; var player = game.GameProgressInformation.CurrentPlayer;
            game.GameProgressInformation.CurrentGameEra = era; game.GameProgressInformation.CurrentRoundNumber = round;
            game.TurnFlow.StartTurn(player, round, era);
            player.Money = 0; player.IncomeMarkerPosition = 1;
            while (player.Mat.GetFirstBuildingOfType(BuildingType.Manufacturer) != null) player.Mat.DevelopIndustry(BuildingType.Manufacturer);
            var factory = new CardsFactory(game.Randomer);
            player.Hand[0] = factory.GetLocationAndIndustryCards((NumberOfPlayers)players).OfType<IndustryCard>().First(c => c.BuildingTypes.Contains(BuildingType.Manufacturer));
            var opponent = game.PlayersTrack.Players.First(p => p.Color != player.Color);
            var iron = new IronWorks("state-regression-iron", opponent.Color, 2, 5, 1, 3, true, false, false, 7, 1, 0, 1);
            game.Board.GetRegionByName(RegionName.Birmingham).BuildingSlots[2].BuildIndustry(iron);
            var metadata = test.App.Metadata.GetPlayerMetadataByColor(player.Color); metadata.AiType = difficulty;
            var behavior = AiBehaviorFactory.CreateAiBehavior(metadata);
            int before = game.TurnFlow.GetRemainingActionsCount();
            behavior.GetNextAIMove(test.App)(); game.TurnFlow.ConfirmCurrentAction();
            Check(game.TurnFlow.GetRemainingActionsCount() == before - 1 && player.Money >= 0, "B14 public AI decision and confirmation " + difficulty + " " + era + " " + round);
        }
    }

    private static void Ambiguous(Action action, string description)
    {
        bool rejected = false;
        try { action(); }
        catch (InvalidOperationException exception) { rejected = exception.Message.Contains("Ambiguous legacy"); }
        Check(rejected, description);
    }

    private static void LegacyIdentitySafety()
    {
        foreach (BuildingType kind in new[] { BuildingType.CoalMine, BuildingType.IronWorks, BuildingType.Brewery })
        {
            var game = NewCase().App.Game;
            var owner = game.GameProgressInformation.CurrentPlayer;
            var slots = game.Board.Regions.Where(r => r.BuildingSlots.Any(s => s.CanBuildIndustryType(kind)))
                .Take(2).Select(r => r.BuildingSlots.First(s => s.CanBuildIndustryType(kind))).ToArray();
            var tiles = new List<BaseIndustry>();
            foreach (var slot in slots)
            {
                BaseIndustry tile;
                if (kind == BuildingType.CoalMine) tile = new CoalMine("legacy-collision", owner.Color, 2, 2, 1, 7, true, false, false, 7, 0, 0, 3);
                else if (kind == BuildingType.IronWorks) tile = new IronWorks("legacy-collision", owner.Color, 2, 5, 1, 3, true, false, false, 7, 1, 0, 3);
                else tile = new Brewery("legacy-collision", owner.Color, 2, 5, 2, 5, true, false, false, 7, 0, 1, () => 2);
                slot.BuildIndustry(tile); tile.Build(); tiles.Add(tile);
            }
            int revision = game.Revision, events = game.GetUncommittedEvents().Count;
            var stock = tiles.Select(t => t.GetResource()).ToArray();
            if (kind == BuildingType.CoalMine)
            {
                Ambiguous(() => game.Board.GetCoalSourceById("legacy-collision"), "B15 legacy coal lookup rejects duplicate");
                Ambiguous(() => game.ApplyEvent(new CoalSupplied("legacy-collision", 1)), "B15 legacy coal event rejected before revision");
            }
            else if (kind == BuildingType.IronWorks)
            {
                Ambiguous(() => game.Board.GetIronSourceById("legacy-collision"), "B15 legacy iron lookup rejects duplicate");
                Ambiguous(() => game.ApplyEvent(new IronSupplied("legacy-collision", 1)), "B15 legacy iron event rejected before revision");
            }
            else
            {
                Ambiguous(() => game.Board.GetBeerSourceById("legacy-collision"), "B15 legacy beer lookup rejects duplicate");
                Ambiguous(() => game.ApplyEvent(new BeerSupplied("legacy-collision", 1)), "B15 legacy beer event rejected before revision");
            }
            Check(game.Revision == revision && game.GetUncommittedEvents().Count == events && stock.SequenceEqual(tiles.Select(t => t.GetResource())), "B15 rejected legacy event has no mutation");
            Ambiguous(() => game.Board.GetIndustryByUniqueId("legacy-collision"), "B15 generic board industry lookup refuses guessing");
            game.ApplyEvent(new IndustryFlipped("legacy-collision", game.Board.GetRegionContainingGivenSlot(slots[1]).Name));
            Check(!tiles[0].IsFlipped && tiles[1].IsFlipped, "B15 region-specific legacy flip remains safe");
        }
        var mat = NewCase().App.Game;
        var player = mat.GameProgressInformation.CurrentPlayer;
        var duplicated = player.Mat.GetFirstBuildingOfType(BuildingType.IronWorks);
        player.Mat.GetIndustryList(BuildingType.IronWorks).Add(new IronWorks(duplicated.UniqueId, player.Color, 4, 9, 1, 1, true, false, false, 12, 1, 0, 6));
        Ambiguous(() => mat.PlayersTrack.GetIndustryByUniqueId(duplicated.UniqueId), "B15 mat lookup catches duplicates within one stack");
    }

    private static void OfflineEntry()
    {
        var router = new NoticeRouter();
        var presenter = new MainMenuPresenter(router, null);
        // Null credentials/backend deliberately fail if the old authentication flow is touched.
        var menu = new MainMenuUseCase(presenter, null, null);
        menu.ShowOnlineGamesView();
        Check(router.Shown == 1 && router.Popup.Text.Contains("Online play is disabled"), "Offline guard shows existing information popup");
        Check(router.Popup.Button == "Close" && router.Popup.Confirm != null, "Offline guard has a dismissible Close action");
        router.Popup.Confirm();
        Check(router.Hidden == 1, "Offline notice closes without authentication/backend calls");
    }

    public static int Main(string[] args)
    {
        try
        {
            Loans(); FinalRound(); Rankings(); BonusOwnership(); ReplayCutoffs(); IntroReload(); EraReload(); ExhaustedManufacturer(); LegacyIdentitySafety(); OfflineEntry();
            Console.WriteLine("{\"suite\":\"state\",\"assertions\":" + assertions + ",\"replayCutoffs\":" + replayCutoffs + ",\"completedGames\":" + fullGames + ",\"failures\":0}");
            return 0;
        }
        catch (Exception exception) { Console.Error.WriteLine(exception); return 1; }
    }

    private sealed class ReplayOutput : IReplayOutput
    {
        public IEvent LastEvent;
        public IEventReplayDelegate Callback;
        public void UpdateCurrentPlayer(string text) { }
        public void UpdateCurrentPlayerAction(IEvent e) { }
        public void UpdateCurrentPlayerActionText(string text) { }
        public void UpdateGamePresentation(PlayerMetadata player) { }
        public void ReplayEvent(IEvent e, IEventReplayDelegate callback) { LastEvent = e; Callback = callback; }
        public void CancelCurrentReplay() { }
        public void ShowReplaySummary(IList<IEvent> events, Action done) { if (done != null) done(); }
        public void ShowIndustrySellingSummary(IEvent e, Action done) { if (done != null) done(); }
        public void UpdateCurrentPlayerActionsView() { }
        public void FinishReplay() { }
    }

    private sealed class SummaryOutput : IFinishedGameReviewOutput
    {
        public bool Intro;
        public List<string> Names = new List<string>();
        public List<int> Places = new List<int>();
        public void ReplayUnseenActions(PlayerMetadata player, IReplayDelegate callback) { }
        public void ShowSkipToMyNextTurnView() { }
        public void HidePlayerActionView() { }
        public void ShowButtonToLaunchGameSummaryTable() { }
        public void ShowGameSummaryTable(List<(PlayerColor, PlayerAvatar, string, int, int, int, int, int, int)> rows)
        { Names = rows.Select(r => r.Item3).ToList(); Places = rows.Select(r => r.Item9).ToList(); }
        public void ShowIntroductoryGameSummaryTable(List<(PlayerColor, PlayerAvatar, string, int, int, int, int, int, int, int, int)> rows)
        { Intro = true; Names = rows.Select(r => r.Item3).ToList(); Places = rows.Select(r => r.Item8).ToList(); }
    }

    private sealed class NoticeRouter : IRouter
    {
        public int Shown, Hidden;
        public readonly NoticePopup Popup = new NoticePopup();
        public IUseCaseBuilder UseCaseBuilder { get; set; }
        public IAnimation ShowView<T>() where T : class, IBaseView { Check(typeof(T) == typeof(IInfoPopupView), "Offline guard never opens login or connection"); Shown++; return null; }
        public IAnimation HideView<T>() where T : class, IBaseView { Check(typeof(T) == typeof(IInfoPopupView), "Offline guard closes its own popup"); Hidden++; return null; }
        public T GetView<T>() where T : class, IBaseView { return (T)(object)Popup; }
        public void MoveOnTop<T>() where T : class, IBaseView { }
        public bool IsViewVisible<T>() where T : class, IBaseView { return false; }
        public bool IsViewCached<T>() where T : class, IBaseView { return false; }
        public void ProcessInput() { }
        public void RegisterKey(KeyEnum key, IBaseView view) { }
        public void UnregisterKey(KeyEnum key, IBaseView view) { }
        public void UnregisterAllKeys(IBaseView view) { }
    }

    private sealed class NoticePopup : IInfoPopupView
    {
        public string Text, Button;
        public Action Confirm;
        public IInputRegistration InputRegistration { get; set; }
        public bool DisableMenusUnderneath { get; set; }
        public bool DestroyWhenClosed { get; set; }
        public IController Controller { get; set; }
        public void Setup(string text, string button, Action confirm) { Text = text; Button = button; Confirm = confirm; }
        public void HandleKeyDown(KeyEnum key) { }
        public void HandleKey(KeyEnum key) { }
        public void HandleKeyUp(KeyEnum key) { }
        public void EnableView() { }
        public void DisableView() { }
        public IAnimation CreateShowViewAnimation() { return null; }
        public IAnimation CreateHideViewAnimation() { return null; }
        public IAnimation CreateShowViewImmediatelyAnimation() { return null; }
        public void SetInteractable(bool interactable) { }
        public bool IsViewVisible() { return true; }
        public void MoveToTop() { }
        public void MoveToBottom() { }
    }
}
