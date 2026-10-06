using System;
using System.Linq;
using BoardGameRules.Entities.Common;
using BoardGameRules.Entities.EventSourcing.Events;
using BoardGameRules.Entities.Game;
using BrassApplication.AI;
using BrassApplication.AI.AiBehaviors;
using BrassApplication.AI.RuleBasedSystem;
using BrassApplication.AI.RuleBasedSystem.RuleSelectors;
using BrassApplication.Game;
using BrassPatch.Statistics;
using Frameworks.Persistence.GameFactory;
using Frameworks.Persistence.Repository;
using Frameworks.Persistence.RepositoryBackends.InMemory;

public sealed class HistoryFixture
{
    public BrassApplicationGame App;
    public BaseRepository Repository;
    public CompletedMatchStatistics Report;
    public IEvent[] Events;
    public static HistoryFixture Start(int count, bool intro, int seed,GameType type=GameType.Local)
    {
        var config = new GameStartConfiguration((NumberOfPlayers)count) { Seed = seed, IsIntroductoryGame = intro };
        var metadata = new GameMetadata(config, type);
        foreach (var start in config.Players) {
            var player = new PlayerMetadata { PlayerStartConfiguration = start, PlayerNetworkId = "history-" + start.Color,
                NickName = "Player " + start.Color, AiType = AIType.Hard };
            player.AIBehaviour = new HardAiBehavior(player.PlayerNetworkId); metadata.JoinGame(player);
        }
        var repository = new BaseRepository(new InMemoryGameStorage(), new InMemorySnapshotStorage(), new BrassBirminghamFactory());
        var app = repository.Create(metadata); app.Game.StartGame();
        return new HistoryFixture{App=app,Repository=repository};
    }
    public static HistoryFixture Finish(int count, bool intro, int seed, bool passOnly = false,GameType type=GameType.Local)
    {
        var fixture=Start(count,intro,seed,type);var app=fixture.App;int turns=0;
        while (!app.Game.GameProgressInformation.IsGameOver) {
            if (++turns > 200) throw new Exception("History fixture did not finish.");
            var behavior = app.Metadata.GetPlayerMetadataByColor(app.Game.GameProgressInformation.CurrentPlayer.Color).AIBehaviour;
            while (app.Game.TurnFlow.GetRemainingActionsCount() > 0) {
                if (passOnly) {
                    var player = app.Game.GameProgressInformation.CurrentPlayer;
                    if (count == 4 && turns == 1) { var scout = app.Game.TurnFlow.StartScoutAction(); scout.DiscardCard(player.Hand[0], player); scout.Scout(player.Hand.Take(2).ToList(), player); }
                    else app.Game.TurnFlow.StartPassAction().Pass(player.Hand.FirstOrDefault(c => c.GetType().Name == "WildLocationCard") ?? player.Hand[0], player);
                } else behavior.GetNextAIMove(app)();
                app.Game.TurnFlow.ConfirmCurrentAction();
            }
            app.Game.GameFlow.EndTurn();
            if (app.Game.GameFlow.IsLastTurnOver()) {
                if (app.Game.GameFlow.IsSellIndustryNecessary()) RuleSelectorFactory.CreatePrioritySelector(RuleSetFactory.CreateSellIndustryRuleSet()).Select(app).MakeAction(app, false)();
                app.Game.GameFlow.EndRound();
            }
        }
        var events = app.Game.GetUncommittedEvents().ToArray();
        fixture.Events=events;fixture.Report=StatisticsCollector.Collect(app,new IEvent[0],events);return fixture;
    }
}
