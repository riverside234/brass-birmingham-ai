using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using BoardGameRules.Entities.EventSourcing.Events;
using BoardGameRules.Entities.GamePhases.Events;
using BrassApplication.Game;
using BrassApplication.GameHistory;
using BrassPatch.Persistence;
using Newtonsoft.Json;

namespace BrassPatch.Statistics
{
    public static class StatisticsCollector
    {
        public static CompletedMatchStatistics Capture(IGameHistory history)
        {
            if (history == null || history.CurrentGame == null) throw new ArgumentNullException("history");
            var app = history.CurrentGame;
            try {
                return Collect(app, history.Repository.Advanced.GameStorage.LoadEvents(app.Metadata.GameId, 0, app.Game.Revision), app.Game.GetUncommittedEvents());
            } catch (Exception error) { return Summary(app, "Detailed history unavailable: " + error.Message); }
        }
        public static CompletedMatchStatistics Summary(BrassApplicationGame app, string reason)
        {
            var report = new CompletedMatchStatistics { GameId = app.Metadata.GameId, FinalRevision = app.Game.Revision,
                Introductory = app.Game.GameStartConfiguration.IsIntroductoryGame, Availability = reason };
            var extension = SaveExtensions.Get(app.Metadata);
            foreach (var player in app.Game.PlayersTrack.Players) {
                var metadata = app.Metadata.PlayerMetadata.FirstOrDefault(p => p.PlayerStartConfiguration.Color == player.Color);
                int rank = app.Game.GameProgressInformation.EndGamePoints != null && app.Game.GameProgressInformation.EndGamePoints.ContainsKey(player.Color)
                    ? app.Game.GameProgressInformation.EndGamePoints[player.Color].Item5 + 1 : 0;
                string opponent = metadata == null ? "Player" : metadata.AiType.ToString();
                if (extension != null && metadata != null && metadata.AiType.ToString() != "Human") {
                    try {
                    var assignments = extension["assignments"] as Newtonsoft.Json.Linq.JArray;
                    var assignment = assignments == null ? null : assignments.FirstOrDefault(a => (int?)a["color"] == (int)player.Color);
                    if (assignment != null) {
                        string strategy=(string)assignment["strategy"] ?? "custom";
                        var execution=assignment["execution"];
                        opponent="AI · starting style: "+strategy;
                        if(execution!=null)opponent+=" · "+(string)execution["difficulty"]+" ("+(int?)execution["search_ms"]/1000+" s)";
                    }
                    } catch(Exception) { opponent="AI · configuration unavailable"; }
                }
                report.players.Add(new StatisticsPlayer { Color = player.Color, Name = metadata == null ? player.Color.ToString() : metadata.NickName,
                    Avatar = metadata == null ? "" : metadata.PlayerAvatar.ToString(), Opponent = opponent,
                    FinalVP = player.VP, FinalCash = player.Money, FinalIncome = player.IncomeLevel, Rank = rank });
            }
            return report;
        }
        public static CompletedMatchStatistics Collect(BrassApplicationGame app, IEnumerable<IEvent> persisted, IEnumerable<IEvent> uncommitted)
        {
            var clock = Stopwatch.StartNew();
            var summary = Summary(app, "Final result recorded; collecting history.");
            if (!app.Game.GameProgressInformation.IsGameOver) { summary.Availability = "Statistics become available when this match is complete."; return summary; }
            try {
                // ApplyEvent mutates the supplied event. Detach through the game's own
                // serializer before replay, including nested configuration/card objects.
                var combined = new SortedDictionary<int, IEvent>();
                foreach (var item in persisted.Concat(uncommitted)) {
                    int revision = item.Metadata.Revision;
                    if (revision > app.Game.Revision) continue;
                    IEvent previous;
                    if (combined.TryGetValue(revision, out previous)) {
                        if (!ReferenceEquals(previous, item) && JsonConvert.SerializeObject(previous) != JsonConvert.SerializeObject(item))
                            throw new InvalidDataException("Conflicting events at revision " + revision + ".");
                    } else combined.Add(revision, item);
                }
                var retained = combined.Values.ToList();
                int finalIndex = retained.FindLastIndex(e => e is GameEnded);
                if (finalIndex < 0) throw new InvalidDataException("The retained history has no completed-game event.");
                retained = retained.Take(finalIndex + 1).ToList();
                if (!(retained.FirstOrDefault() is GameStarted)) throw new InvalidDataException("The initial setup record is missing.");
                for (int i = 0; i < retained.Count; i++)
                    if (retained[i].Metadata.Revision != i + 1) throw new InvalidDataException("History is incomplete at revision " + (i + 1) + ".");
                var serializer = new Frameworks.Persistence.Serialization.JsonSerializer();
                List<IEvent> events;
                byte[] detached;
                using (var stream = new MemoryStream()) {
                    serializer.Serialize(stream, retained); detached = stream.ToArray();
                }
                using (var stream = new MemoryStream(detached)) events = serializer.Deserialize<List<IEvent>>(stream);
                var oldBonus = BoardGameRules.Entities.Actions.DevelopMerchantBonus.Instance;
                try { new StatisticsProjection(summary, events).Run(); }
                finally { BoardGameRules.Entities.Actions.DevelopMerchantBonus.Instance = oldBonus; }
                summary.CompleteHistory = true;
                summary.Availability = "Recorded results · details derived from the retained event history";
            } catch (Exception error) {
                // A partly applied projection cannot masquerade as a complete result.
                summary = Summary(app, "Detailed history unavailable: " + error.Message);
            }
            summary.CollectionMilliseconds = clock.Elapsed.TotalMilliseconds;
            return summary;
        }
    }
}
