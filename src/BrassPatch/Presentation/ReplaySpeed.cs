using Frameworks.View.Animation;
using Frameworks.View.Game.Replay;
using InterfaceAdapters.Game.Replay;

namespace BrassPatch.Presentation
{
    public static class ReplaySpeed
    {
        public static void BeforePlay(DOTweenAnimationSequence animation)
        {
            // Sell's ReplayCompleted can start the next replay and then disable
            // fast-forward in its selection-state epilogue. Play is dispatched
            // afterward, so restore the replay input when playback actually begins.
            // View transitions and selection highlights have no replay ID.
            if (animation.Sequence == null || (animation.Sequence.stringId != "AnimationFactory"
                && !Equals(animation.Sequence.id, "AnimationFactory"))) return;
            var router = Utils.Global.Instance.Router;
            if (router == null || !router.IsViewVisible<IReplayView>()) return;
            var replay = router.GetView<IReplayView>() as ReplayView;
            if (replay != null && replay.isActiveAndEnabled) replay.SetFastForwardActive(true);
        }
    }
}
