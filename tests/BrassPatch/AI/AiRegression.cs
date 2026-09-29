using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using BoardGameRules.Entities.Actions;
using BoardGameRules.Entities.Board.BuildingSlots;
using BoardGameRules.Entities.Board.Regions;
using BoardGameRules.Entities.Cards;
using BoardGameRules.Entities.Common;
using BoardGameRules.Entities.Game;
using BoardGameRules.Entities.GamePhases;
using BoardGameRules.Entities.Industries;
using BoardGameRules.Entities.Players;
using BrassApplication.AI;
using BrassApplication.AI.AIHelpers;
using BrassApplication.AI.AiBehaviors;
using BrassApplication.Game;

// All fixtures use the staged production engine. No user storage or Unity APIs
// are needed, so Run() also executes in the explicit Unity Mono self-test.
public static class AiRegression
{
    private static int assertions;
    private static int decisionCases;
    private static int legalHistories;
    private static int openingControls;

    private static void Check(bool condition, string label)
    {
        assertions++;
        if (!condition) throw new Exception("AI regression failed: " + label);
    }

    private static BrassApplicationGame NewGame(int players, int seed = 20260928)
    {
        var config = new GameStartConfiguration((NumberOfPlayers)players) { Seed = seed };
        var game = new BoardGameRulesGame(config);
        var metadata = new GameMetadata(config, GameType.Local);
        foreach (var start in config.Players)
            metadata.JoinGame(new PlayerMetadata { PlayerStartConfiguration = start,
                PlayerNetworkId = "ai-regression-" + start.Color, NickName = "AI regression", AiType = AIType.Hard });
        game.StartGame();
        return new BrassApplicationGame(game, metadata);
    }

    private static object BreweryRule()
    {
        var type = typeof(HardAiBehavior).Assembly.GetType("BrassApplication.AI.RuleBasedSystem.Rules.BuildBreweryForCanalEra", true);
        return Activator.CreateInstance(type, true);
    }

    private static bool Applicable(object rule, BrassApplicationGame app, bool original)
    {
        var name = original ? "BrassPatch_Original_IsApplicable" : "IsApplicable";
        var method = rule.GetType().GetMethod(name, BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic);
        Check(method != null, "woven brewery rule contains " + name);
        try { return (bool)method.Invoke(rule, new object[] { app }); }
        catch (TargetInvocationException error) { throw error.InnerException ?? error; }
    }

    // Synthetic boundary fixtures deliberately isolate rule applicability, rather
    // than claiming that every round/stock combination is a played history.
    private static void BreweryBoundaries()
    {
        foreach (int players in new[] { 2, 3, 4 })
        foreach (GameEra era in new[] { GameEra.CanalEra, GameEra.RailEra })
        foreach (int round in new[] { 1, 2, 3, 7 })
        foreach (int nextLevel in new[] { 0, 1, 2, 4 })
        foreach (int ownedState in new[] { 0, 1, 2 }) // absent, unflipped, flipped
        {
            var app = NewGame(players); var game = app.Game; var p = game.GameProgressInformation.CurrentPlayer;
            game.GameProgressInformation.CurrentGameEra = era;
            game.GameProgressInformation.CurrentRoundNumber = round;
            var stack = p.Mat.GetIndustryList(BuildingType.Brewery);
            var boardBrewery = stack[1];
            var next = nextLevel == 0 ? null : stack.First(x => x.Level == nextLevel);
            stack.Clear(); if (next != null) stack.Add(next);
            if (ownedState != 0)
            {
                game.Board.GetRegionByName(RegionName.Stafford).BuildingSlots.First(s => s.BuildingTypes.Contains(BuildingType.Brewery)).BuildIndustry(boardBrewery);
                boardBrewery.Build(); boardBrewery.IsFlipped = ownedState == 2;
                ((Brewery)boardBrewery).AvailableBeer = ownedState == 2 ? 0 : 1;
            }
            p.Money = 100;
            p.Hand.Clear();
            p.Hand.Add(new IndustryCard("brewery-opening-card", NumberOfPlayers.Two, new List<BuildingType> { BuildingType.Brewery }));
            BoardHelper.Init(app, AIType.Hard);
            var rule = BreweryRule(); int revision = game.Revision;
            bool actual = Applicable(rule, app, false);
            bool expected = era == GameEra.CanalEra && round <= 2 && nextLevel == 1 && ownedState != 1 &&
                BoardHelper.PossibleBuildingSlots[BuildingType.Brewery].Count > 0;
            string label = players + "/" + era + "/r" + round + "/next" + nextLevel + "/owned" + ownedState;
            Check(actual == expected, "brewery applicability " + label);
            Check(game.Revision == revision && p.Hand.Count == 1 && p.Money == 100,
                "applicability is read-only " + label);
            if (next != null)
            {
                Check(actual == Applicable(rule, app, true), "preserve original non-null rule behavior " + label);
                if (actual) openingControls++;
            }
        }
        Check(openingControls > 0, "matrix actually exercises applicable original opening rules");
    }

    private static BaseAiBehavior Behavior(AIType difficulty)
    {
        if (difficulty == AIType.Easy) return new EasyAiBehavior("regression");
        if (difficulty == AIType.Medium) return new NormalAiBehavior("regression");
        return new HardAiBehavior("regression");
    }

    private static void ExecuteDecision(BrassApplicationGame app, AIType difficulty, string label)
    {
        var game = app.Game; var p = game.GameProgressInformation.CurrentPlayer;
        int hand = p.Hand.Count, remaining = game.TurnFlow.GetRemainingActionsCount();
        var move = Behavior(difficulty).GetNextAIMove(app);
        Check(move != null, "public AI returns a move " + label);
        move();
        var action = game.TurnFlow.CurrentAction;
        Check(action != null, "public AI starts an action " + label);
        var build = action as BuildAction;
        var develop = action as DevelopAction;
        var network = action as NetworkAction;
        var sell = action as SellAction;
        Check((build == null || build.NeededCoal + build.NeededIron == 0) &&
            (develop == null || develop.NeededIron == 0) &&
            (network == null || network.NeededCoal + network.NeededBeer == 0) &&
            (sell == null || sell.NeededBeer == 0), "AI action pays its resources " + label);
        game.TurnFlow.ConfirmCurrentAction();
        Check(p.Hand.Count == hand - 1, "AI consumes one net card " + label);
        Check(game.TurnFlow.GetRemainingActionsCount() == remaining - 1,
            "AI consumes one action " + label);
        Check(game.TurnFlow.CurrentAction == null && p.Money >= 0 && !game.DevelopMerchantBonus.MerchantDevelopmentNeeded,
            "AI completes its action " + label);
        decisionCases++;
    }

    private static void PublicDecisionBoundaries()
    {
        foreach (int players in new[] { 2, 3, 4 })
        foreach (AIType difficulty in new[] { AIType.Easy, AIType.Medium, AIType.Hard })
        foreach (GameEra era in new[] { GameEra.CanalEra, GameEra.RailEra })
        foreach (int round in new[] { 1, 2, 3, 7 })
        foreach (bool empty in new[] { false, true })
        foreach (bool ownedUnflipped in new[] { false, true })
        {
            var app = NewGame(players); var game = app.Game; var p = game.GameProgressInformation.CurrentPlayer;
            var stack = p.Mat.GetIndustryList(BuildingType.Brewery);
            var next = stack[0]; var boardBrewery = stack[1];
            stack.Clear(); if (!empty) stack.Add(next);
            if (ownedUnflipped)
            {
                game.Board.GetRegionByName(RegionName.Stafford).BuildingSlots.First(s => s.BuildingTypes.Contains(BuildingType.Brewery)).BuildIndustry(boardBrewery);
                boardBrewery.Build();
            }
            game.GameProgressInformation.CurrentGameEra = era;
            game.GameProgressInformation.CurrentRoundNumber = round;
            game.TurnFlow.StartTurn(p, round, era);
            p.Money = 0; p.IncomeMarkerPosition = 1; // no loan; no affordable build/develop/network
            ExecuteDecision(app, difficulty, players + "/" + difficulty + "/" + era + "/r" + round + "/empty" + empty + "/owned" + ownedUnflipped);
        }
    }

    private static void LegalHistory(int players)
    {
        var app = NewGame(players); var game = app.Game;
        var target = game.GameProgressInformation.CurrentPlayer.Color;
        int step = 0, allActions = 0;
        while (game.GameProgressInformation.CurrentRoundNumber < 7 || game.GameProgressInformation.CurrentPlayer.Color != target)
        {
            Check(game.GameProgressInformation.CurrentRoundNumber <= 7, "legal trace reaches target in round seven");
            var p = game.GameProgressInformation.CurrentPlayer;
            while (game.TurnFlow.GetRemainingActionsCount() > 0)
            {
                int hand = p.Hand.Count, remaining = game.TurnFlow.GetRemainingActionsCount();
                if (p.Color == target && step < 3)
                {
                    var loan = game.TurnFlow.StartLoanAction();
                    Check(loan.GetLoanActionNotPossibleReasons(game.GameProgressInformation).Count == 0, "trace loan legal");
                    loan.DiscardCard(p.Hand[0], p); loan.Loan(p.Color); step++;
                }
                else if (p.Color == target && step < 11)
                {
                    var type = step < 7 ? BuildingType.Brewery : BuildingType.CoalMine;
                    int quantity = step == 6 || step == 10 ? 1 : 2;
                    var develop = game.TurnFlow.StartDevelopAction();
                    Check(develop.GetDevelopActionNotPossibleReasons(game.GameProgressInformation).Count == 0,
                        "trace development available");
                    Check(p.Money >= game.Board.IronMarket.IronPrice(quantity) && !game.Board.GetUnflippedIronWorks().Any(),
                        "trace market iron legal and affordable");
                    develop.DiscardCard(p.Hand[0], p);
                    for (int n = 0; n < quantity; n++)
                    {
                        Check(p.Mat.CanIndustryBeDeveloped(type), "trace next tile may be developed");
                        develop.Develop(type);
                    }
                    develop.SupplyIron(game.Board.IronMarket, quantity);
                    Check(develop.NeededIron == 0, "trace development pays all iron"); step++;
                }
                else game.TurnFlow.StartPassAction().Pass(p.Hand[0], p);
                game.TurnFlow.ConfirmCurrentAction(); allActions++;
                Check(p.Hand.Count == hand - 1 && p.Money >= 0, "trace card and money accounting");
                Check(game.TurnFlow.GetRemainingActionsCount() == remaining - 1, "trace exact action decrement");
            }
            game.GameFlow.EndTurn();
            if (game.GameFlow.IsLastTurnOver()) game.GameFlow.EndRound();
        }
        var current = game.GameProgressInformation.CurrentPlayer;
        Check(step == 11 && allActions == 13 * players - 2, "trace action count " + players);
        Check(current.Money == 0 && current.IncomeLevel == -9 && current.VP == 0 &&
            current.Mat.GetIndustryList(BuildingType.Brewery).Count == 0 &&
            current.Mat.GetIndustryList(BuildingType.CoalMine).Count == 0 &&
            !game.Board.GetIndustriesOnBoardOfGivenPlayer(target).Any(), "legal original crash boundary " + players);
        Check(game.GameProgressInformation.CurrentGameEra == GameEra.CanalEra && !game.GameFlow.IsSellIndustryNecessary(),
            "trace debt settled and still Canal " + players);

        var config = game.GameStartConfiguration;
        var replay = new BoardGameRulesGame(config);
        foreach (var item in game.GetUncommittedEvents()) replay.ApplyEvent(item);
        var rp = replay.GameProgressInformation.CurrentPlayer;
        Check(replay.Revision == game.Revision && rp.Color == target && rp.Money == current.Money &&
            rp.IncomeLevel == current.IncomeLevel && rp.Hand.Select(c => c.UniqueId).SequenceEqual(current.Hand.Select(c => c.UniqueId)) &&
            rp.Mat.GetIndustryList(BuildingType.Brewery).Count == 0 && rp.Mat.GetIndustryList(BuildingType.CoalMine).Count == 0 &&
            replay.GameProgressInformation.CurrentRoundNumber == 7, "event replay preserves full trigger " + players);
        ExecuteDecision(app, AIType.Hard, "legal history " + players);
        ExecuteDecision(new BrassApplicationGame(replay, new GameMetadata(config, GameType.Local)), AIType.Hard,
            "replayed legal history " + players);
        legalHistories++;
    }

    private static void ManufacturerRegression()
    {
        // Preserve B14: its distinct exhausted-manufacturer guard still runs at
        // the public decision boundary before any R03 opening-brewery rule.
        foreach (int players in new[] { 3, 4 })
        {
            var app = NewGame(players); var game = app.Game; var p = game.GameProgressInformation.CurrentPlayer;
            game.GameProgressInformation.CurrentGameEra = GameEra.RailEra;
            game.GameProgressInformation.CurrentRoundNumber = 1;
            game.TurnFlow.StartTurn(p, 1, GameEra.RailEra);
            p.Money = 0; p.IncomeMarkerPosition = 1;
            p.Mat.GetIndustryList(BuildingType.Manufacturer).Clear();
            p.Hand[0] = new IndustryCard("manufacturer-boundary", NumberOfPlayers.Three,
                new List<BuildingType> { BuildingType.CottonMill, BuildingType.Manufacturer });
            var opponent = game.PlayersTrack.Players.First(x => x.Color != p.Color);
            var iron = (IronWorks)opponent.Mat.GetIndustryList(BuildingType.IronWorks).First(x => x.Level == 2);
            opponent.Mat.RemoveGivenBuildingFromMat(iron);
            game.Board.GetRegionByName(RegionName.Birmingham).BuildingSlots.First(s => s.BuildingTypes.Contains(BuildingType.IronWorks)).BuildIndustry(iron);
            iron.Build(); iron.AvailableIron = 1;
            ExecuteDecision(app, AIType.Hard, "B14 exhausted manufacturers " + players);
        }
    }

    public static int Run()
    {
        assertions = decisionCases = legalHistories = openingControls = 0;
        BreweryBoundaries();
        PublicDecisionBoundaries();
        foreach (int players in new[] { 2, 3, 4 }) LegalHistory(players);
        ManufacturerRegression();
        return assertions;
    }

    public static int Main()
    {
        try
        {
            int total = Run();
            Console.WriteLine("AI regressions passed: " + total + " assertions; " + decisionCases +
                " executed public AI decisions; " + legalHistories + " legal histories plus event replay; " +
                openingControls + " applicable original opening controls.");
            return 0;
        }
        catch (Exception error) { Console.Error.WriteLine(error); return 1; }
    }
}
