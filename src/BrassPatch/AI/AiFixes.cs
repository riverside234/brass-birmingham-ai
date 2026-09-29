using System;
using System.Reflection;
using System.Runtime.ExceptionServices;
using BoardGameRules.Entities.Board.BuildingSlots;
using BoardGameRules.Entities.GamePhases;
using BrassApplication.Game;

namespace BrassPatch
{
    /// <summary>Boundary guards for the shipped AI; preserves its opening brewery strategy.</summary>
    public static class AiFixes
    {
        public static bool BreweryRuleApplicable(object rule, BrassApplicationGame app)
        {
            var progress = app.Game.GameProgressInformation;
            // This heuristic only applies to the first two Canal rounds. Check the
            // window before looking at tiles that may have been exhausted later.
            if (progress.CurrentGameEra != GameEra.CanalEra || progress.CurrentRoundNumber > 2)
                return false;
            if (progress.CurrentPlayer.Mat.GetFirstBuildingOfType(BuildingType.Brewery) == null)
                return false;

            // Keep the original owned-brewery, tile-era and valid-slot decisions.
            var original = rule.GetType().GetMethod("BrassPatch_Original_IsApplicable",
                BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic,
                null, new[] { typeof(BrassApplicationGame) }, null);
            if (original == null) throw new MissingMethodException(rule.GetType().FullName, "BrassPatch_Original_IsApplicable");
            try { return (bool)original.Invoke(rule, new object[] { app }); }
            catch (TargetInvocationException error)
            {
                ExceptionDispatchInfo.Capture(error.InnerException ?? error).Throw();
                throw;
            }
        }
    }
}
