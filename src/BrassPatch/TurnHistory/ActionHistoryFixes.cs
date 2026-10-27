using System;
using BoardGameRules.Entities.Actions.Events;
using BoardGameRules.Entities.GamePhases.Events;
using BrassApplication.GameHistory;

namespace BrassPatch.TurnHistory
{
    public static class ActionHistoryFixes
    {
        // A checkpoint can precede several AI turns. Uncommitted events are a
        // persistence buffer, not the boundary of the human's current action.
        public static int BeginningOfAction(BrassGameHistory history)
        {
            var game = history.CurrentGame.Game;
            var events = game.GetUncommittedEvents();
            int index = events.Count - 1;
            // Undo a completed action; Cancel instead keeps the preceding one.
            if (index >= 0 && events[index] is ActionCountDecreased) index--;
            for (; index >= 0; index--)
                if (events[index] is ActionCountDecreased || events[index] is TurnStarted)
                    return events[index].Metadata.Revision;
            return game.Revision - events.Count;
        }

        public static bool CanCancel(BrassGameHistory history)
        {
            if (history.CurrentGame == null || ReflectionAccess.Get<bool>(history, "_movedBackInHistory"))
                return false;
            var game = history.CurrentGame.Game;
            var events = game.GetUncommittedEvents();
            if (events.Count == 0 || events[events.Count - 1] is ActionCountDecreased)
                return false;
            // Opening and closing a city emits no events. Do not reconstruct
            // the game (or touch earlier AI turns) merely to dismiss that UI.
            return BeginningOfAction(history) < game.Revision;
        }
    }
}
