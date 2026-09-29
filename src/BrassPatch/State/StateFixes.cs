using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Runtime.ExceptionServices;
using BoardGameRules.Entities.Actions;
using BoardGameRules.Entities.Board.BuildingSlots;
using BoardGameRules.Entities.Game;
using BoardGameRules.Entities.GamePhases;
using BoardGameRules.Entities.GamePhases.Events;
using BoardGameRules.Entities.NumericalResources;
using BoardGameRules.Entities.NumericalResources.Events;
using BoardGameRules.Entities.Players;
using BrassApplication.Game;
using BrassApplication.GameHistory;
using BrassApplication.UseCases.Game.FinishedGameReview;

namespace BrassPatch
{
    /// <summary>Game-owned state and event-reconstructed result repairs. No global current-game state.</summary>
    public static class StateFixes
    {
        private const BindingFlags Flags = BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.DeclaredOnly;

        private static FieldInfo FindField(Type type, string name)
        {
            for (var current = type; current != null; current = current.BaseType)
            {
                var field = current.GetField(name, Flags);
                if (field != null) return field;
            }
            return null;
        }

        private static object Read(object owner, string name)
        {
            var field = FindField(owner.GetType(), name);
            if (field == null) throw new MissingFieldException(owner.GetType().FullName, name);
            return field.GetValue(owner);
        }

        private static void Write(object owner, string name, object value)
        {
            var field = FindField(owner.GetType(), name);
            if (field == null) throw new MissingFieldException(owner.GetType().FullName, name);
            field.SetValue(owner, value);
        }

        private static object Original(object owner, string name, params object[] args)
        {
            var parameters = args.Select(a => a.GetType()).ToArray();
            MethodInfo method = null;
            for (var current = owner.GetType(); current != null && method == null; current = current.BaseType)
                method = current.GetMethod("BrassPatch_Original_" + name, Flags, null, parameters, null);
            if (method == null) throw new MissingMethodException(owner.GetType().FullName, name);
            try { return method.Invoke(owner, args); }
            catch (TargetInvocationException exception)
            {
                ExceptionDispatchInfo.Capture(exception.InnerException ?? exception).Throw();
                throw;
            }
        }

        /// <summary>Resolve a domain object's aggregate, or an application use case's own history.</summary>
        public static BoardGameRulesGame GameForOwner(object owner)
        {
            if (owner == null) throw new ArgumentNullException("owner");
            var game = owner as BoardGameRulesGame;
            if (game != null) return game;
            var app = owner as BrassApplicationGame;
            if (app != null) return app.Game;
            var rootField = FindField(owner.GetType(), "_aggregateRoot");
            if (rootField != null)
            {
                game = rootField.GetValue(owner) as BoardGameRulesGame;
                if (game != null) return game;
            }
            var historyField = FindField(owner.GetType(), "_gameHistory");
            if (historyField != null) return ((IGameHistory)historyField.GetValue(owner)).CurrentGame.Game;
            // Board owns a PlayersTrack; that track owns precisely the same aggregate.
            var playersField = FindField(owner.GetType(), "_playersTrack");
            if (playersField != null) return GameForOwner(playersField.GetValue(owner));
            throw new InvalidOperationException("Cannot resolve the owning game of " + owner.GetType().FullName);
        }

        public static DevelopMerchantBonus BonusForOwner(object owner)
        {
            return GameForOwner(owner).DevelopMerchantBonus;
        }

        public static bool ShouldSettleIncomeThisRound(GameFlow flow)
        {
            var game = GameForOwner(flow);
            var progress = game.GameProgressInformation;
            return progress.CurrentGameEra != GameEra.RailEra ||
                progress.CurrentRoundNumber != flow.GetNumberOfRounds(game.GameStartConfiguration.NumberOfPlayers);
        }

        public static void BeginEndRound(GameFlow flow)
        {
            if (ShouldSettleIncomeThisRound(flow))
            {
                Original(flow, "BeginEndRound");
                return;
            }
            Write(flow, "_shouldEndRoundBeCalled", true);
            Write(flow, "_isSellIndustryNecessary", false);
        }

        public static void EndTurn(GameFlow flow)
        {
            // Validate before AggregateRoot.ApplyEvent advances revision and invokes TurnFlow.
            if (BonusForOwner(flow).MerchantDevelopmentNeeded)
                throw new InvalidOperationException("You can't end turn before finishing develop merchant bonus");
            Original(flow, "EndTurn");
        }

        public static List<CantUseLoanActionReason> LoanReasons(LoanAction action, GameProgressInformation progress)
        {
            var result = new List<CantUseLoanActionReason>();
            if (progress.CurrentPlayer.IncomeLevel - LoanAction.LevelsToTakeWithLoan < -10)
                result.Add(CantUseLoanActionReason.INCOME_TOO_LOW);
            return result;
        }

        public static bool ManufacturerRuleApplicable(object rule, BrassApplicationGame app)
        {
            if (app.Game.GameProgressInformation.CurrentPlayer.Mat.GetFirstBuildingOfType(BuildingType.Manufacturer) == null)
                return false;
            return (bool)Original(rule, "IsApplicable", app);
        }

        public static void ApplyGameEnded(EndOfGame end, GameEnded ended)
        {
            var game = GameForOwner(end);
            var progress = game.GameProgressInformation;
            var ordered = game.PlayersTrack.Players.OrderByDescending(p => p.VP)
                .ThenByDescending(p => p.IncomeLevel).ThenByDescending(p => p.Money).ToList();
            var points = new Dictionary<PlayerColor, ValueTuple<int, int, int, int, int>>();
            int rank = 0;
            for (int index = 0; index < ordered.Count; index++)
            {
                var player = ordered[index];
                if (index > 0 && !SameResult(player, ordered[index - 1])) rank = index;
                points[player.Color] = ValueTuple.Create(ended.LastEraLinksVPs[player.Color],
                    ended.LastEraIndustriesVPs[player.Color], ended.LastEraBonusesVPs[player.Color], player.VP, rank);
            }
            progress.IsGameOver = true;
            progress.EndGameRanking = ordered.Select(p => p.Color).ToList();
            progress.EndGamePoints = points;
        }

        private static bool SameResult(Player left, Player right)
        {
            return left.VP == right.VP && left.IncomeLevel == right.IncomeLevel && left.Money == right.Money;
        }

        private static void RecordIntroComponent(VictoryPointsTrack track, PlayerColor color, int amount, int component)
        {
            var progress = GameForOwner(track).GameProgressInformation;
            if (progress.IntroductoryPoints == null)
                progress.IntroductoryPoints = new Dictionary<PlayerColor, ValueTuple<int, int, int>>();
            ValueTuple<int, int, int> row;
            progress.IntroductoryPoints.TryGetValue(color, out row);
            if (component == 0) row.Item1 = amount;
            if (component == 1) row.Item2 = amount;
            if (component == 2) row.Item3 = amount;
            progress.IntroductoryPoints[color] = row;
        }

        public static void IntroMoney(VictoryPointsTrack track, IntroductoryGameAdditionalMoneyVictoryPointsChanged change)
        {
            var player = GameForOwner(track).PlayersTrack.GetPlayerByColor(change.Player);
            player.VP = Math.Max(0, player.VP + change.Amount);
            RecordIntroComponent(track, change.Player, change.Amount, 0);
        }

        public static void IntroIncome(VictoryPointsTrack track, IntroductoryGameAdditionalIncomeVictoryPointsChanged change)
        {
            GameForOwner(track).PlayersTrack.GetPlayerByColor(change.Player).VP += change.Amount;
            RecordIntroComponent(track, change.Player, change.Amount, 1);
        }

        public static void IntroIndustry(VictoryPointsTrack track, IntroductoryGameAdditionalIndustryVictoryPointsChanged change)
        {
            GameForOwner(track).PlayersTrack.GetPlayerByColor(change.Player).VP += change.Amount;
            RecordIntroComponent(track, change.Player, change.Amount, 2);
        }

        private static void RestoreEraSummary(BaseEra era, Dictionary<PlayerColor, int> links,
            Dictionary<PlayerColor, int> industries, Dictionary<PlayerColor, int> bonuses)
        {
            // Copy event payloads: later command execution must not mutate recorded history.
            Write(era, "EndOfEraLinksVPs", new Dictionary<PlayerColor, int>(links));
            Write(era, "EndOfEraIndustriesVPs", new Dictionary<PlayerColor, int>(industries));
            Write(era, "EndOfEraBonusesVPs", new Dictionary<PlayerColor, int>(bonuses));
            foreach (var player in GameForOwner(era).PlayersTrack.Players) player.VPFromBonusesInCurrentEra = 0;
        }

        public static void CanalEnded(CanalEra era, CanalEraEnded ended)
        {
            RestoreEraSummary(era, ended.EndOfEraLinksVPs, ended.EndOfEraIndustriesVPs, ended.CurrentEraBonusesVPs);
            Original(era, "ApplyEvent", ended);
        }

        public static void RailEnded(RailEra era, RailEraEnded ended)
        {
            RestoreEraSummary(era, ended.EndOfEraLinksVPs, ended.EndOfEraIndustriesVPs, ended.CurrentEraBonusesVPs);
        }

        public static void ShowSummary(FinishedGameReviewUseCase review)
        {
            var history = (IGameHistory)Read(review, "_gameHistory");
            var presenter = (IFinishedGameReviewOutput)Read(review, "_presenter");
            var app = history.CurrentGame;
            var game = app.Game;
            var points = game.GameProgressInformation.EndGamePoints;
            int winnerCount = points.Values.Count(row => row.Item5 == 0);
            Func<Player, string> name = player => app.Metadata.GetPlayerMetadataByColor(player.Color).NickName +
                (winnerCount > 1 && points[player.Color].Item5 == 0 ? " (joint winner)" : "");
            if (app.Metadata.GameStartConfiguration.IsIntroductoryGame)
            {
                var components = game.GameProgressInformation.IntroductoryPoints;
                var rows = new List<(PlayerColor, PlayerAvatar, string, int, int, int, int, int, int, int, int)>();
                foreach (var player in game.PlayersTrack.Players)
                {
                    ValueTuple<int, int, int> intro = default(ValueTuple<int, int, int>);
                    bool available = components != null && components.TryGetValue(player.Color, out intro);
                    var score = points[player.Color];
                    rows.Add((player.Color, app.Metadata.GetPlayerMetadataByColor(player.Color).PlayerAvatar,
                        name(player) + (available ? "" : " (breakdown unavailable)"), score.Item1, score.Item2,
                        score.Item3, player.VP, score.Item5, intro.Item1, intro.Item2, intro.Item3));
                }
                presenter.ShowIntroductoryGameSummaryTable(rows);
                return;
            }
            var normal = new List<(PlayerColor, PlayerAvatar, string, int, int, int, int, int, int)>();
            foreach (var player in game.PlayersTrack.Players)
            {
                var score = points[player.Color];
                normal.Add((player.Color, app.Metadata.GetPlayerMetadataByColor(player.Color).PlayerAvatar, name(player),
                    player.VP - score.Item1 - score.Item2 - score.Item3, score.Item1, score.Item2, score.Item3, player.VP, score.Item5));
            }
            presenter.ShowGameSummaryTable(normal);
        }

        public static int CompareIntroResultRows(
            (PlayerColor, PlayerAvatar, string, int, int, int, int, int, int, int, int) left,
            (PlayerColor, PlayerAvatar, string, int, int, int, int, int, int, int, int) right)
        {
            return left.Item8.CompareTo(right.Item8);
        }

        public static void AnalyticsGameOver(object analytics, GameMetadata metadata)
        {
            // The old event has only one winner field. Do not report an arbitrary sole winner for draws.
            var points = metadata.GameProgressInformation.EndGamePoints;
            if (points != null && points.Values.Count(row => row.Item5 == 0) > 1) return;
            Original(analytics, "GameOver", metadata);
        }
    }
}
