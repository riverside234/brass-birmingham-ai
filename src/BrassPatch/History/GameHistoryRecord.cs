using System;
using System.Collections.Generic;
using BoardGameRules.Entities.Players;
using BrassPatch.Statistics;

namespace BrassPatch.History
{
    public sealed class GameHistoryRecord
    {
        public string Key { get; internal set; }
        public long Sequence { get; internal set; }
        public DateTime RecordedUtc { get; internal set; }
        public string Producer { get; internal set; }
        public CompletedMatchStatistics Report { get; internal set; }
    }
    public sealed class GameHistorySummary
    {
        public string Key { get; internal set; }
        public long Sequence { get; internal set; }
        public DateTime RecordedUtc { get; internal set; }
        public bool Introductory { get; internal set; }
        public bool CompleteHistory { get; internal set; }
        public string Unavailable { get; internal set; }
        public IReadOnlyList<StatisticsPlayer> Players { get; internal set; }
    }
    public sealed class GameHistoryList
    {
        public IReadOnlyList<GameHistorySummary> Games { get; internal set; }
        public string Error { get; internal set; }
        public string Diagnostic { get; internal set; }
    }
    public enum HistoryAdmission { Added, Updated, Unchanged, AlreadyRetired }
}
