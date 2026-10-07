using System;

namespace BrassLegacyAI
{
    // Game execution only. No profile/save edits and no change to lab search.
    public static class TurnFinishingBudget
    {
        public static double MarginMilliseconds(double remainingTurnMilliseconds)
        {
            if(double.IsNaN(remainingTurnMilliseconds) || double.IsInfinity(remainingTurnMilliseconds) || remainingTurnMilliseconds<0)
                throw new ArgumentOutOfRangeException(nameof(remainingTurnMilliseconds));
            // Native laptop measurements: worst baseline retirement/publication
            // after 2/5/12 s was 164/187/228 ms. Bounded headroom, interpolated
            // for existing 10 s saves and reduced-clock replanning requests.
            if(remainingTurnMilliseconds<=2000)return remainingTurnMilliseconds*.125;
            if(remainingTurnMilliseconds<=5000)return 250+(remainingTurnMilliseconds-2000)/60;
            return Math.Min(400,300+(remainingTurnMilliseconds-5000)/70);
        }
        public static bool MayPublish(double elapsed,double deadline,bool hasCompletedPlan)
        {
            // A missed deadline cannot replace the already available legal
            // fallback. If mandatory incumbent construction itself runs late,
            // retain its first complete result rather than invent a partial move.
            return !hasCompletedPlan || elapsed<deadline;
        }
    }
    public sealed class TurnSearchTiming
    {
        public double budget_ms,margin_ms,search_cutoff_ms,elapsed_ms,finalization_ms,final_publish_ms,publish_copy_ms;
        public int workers,completed,interrupted,nodes,failures,late_publications_ignored;
        public bool cancelled,deadline_overrun;
    }
}
