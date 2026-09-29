using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using BoardGameRules.Entities.Board.BuildingSlots;
using BoardGameRules.Entities.Industries;
using BoardGameRules.Entities.Actions.Events;
using BoardGameRules.Entities.Players;
using DG.Tweening;
using Frameworks.Routing;
using Frameworks.Utils;
using Frameworks.View.Animation;
using Frameworks.View.Common;
using Frameworks.View.Game;
using Frameworks.View.Game.Replay;
using InterfaceAdapters.Common;
using InterfaceAdapters.Game.Board;
using InterfaceAdapters.Game.IndustriesDetails;
using InterfaceAdapters.Game.PlayersSummary;
using InterfaceAdapters.Routing;
using UnityEngine;
using UnityEngine.UI;

namespace BrassPatch.Presentation
{
    public static class PresentationLifecycle
    {
        public static IAnimation Flip(BoardView board, BuildingSlot slot)
        {
            // Replay can already be at the post-event state. The stock callback appends
            // "Flipped" to a mutable image name, sometimes requesting FlippedFlipped.
            var industry = (BaseIndustry)typeof(object).GetMethod("MemberwiseClone", ReflectionAccess.Flags).Invoke(slot.BuildedIndustry, null);
            industry.IsFlipped = false;
            var snapshot = new BuildingSlot(slot.UniqueId, industry.BuildingType) { BuildedIndustry = industry };
            var view = board.GetComponentsInChildren<BuildingSlotView>(true).First(v => v.name == slot.UniqueId);
            var owner = view.GetComponent<SlotFlipLifetime>() ?? view.gameObject.AddComponent<SlotFlipLifetime>();
            owner.Cancel();
            var animation = (DOTweenAnimationSequence)ReflectionAccess.Original(board, "CreateIndustryFlippedAnimation", new[] { typeof(BuildingSlot) }, snapshot);
            var token = view.transform.Find("IndustryToken").GetComponent<Image>();
            var sprite = UnityEngine.Resources.Load<Sprite>("Board/Industries/" + snapshot.GetCurrentImageName() + "Flipped");
            var previous = animation.Sequence.onKill;
            owner.Animation = animation.Sequence;
            animation.Sequence.OnKill(() => {
                if (owner != null) owner.Animation = null;
                if (token == null) return;
                if (previous != null) previous();
                if (sprite != null) token.sprite = sprite;
                token.transform.localRotation = Quaternion.identity;
                token.transform.localScale = Vector3.one;
            });
            return animation;
        }

        public static void BeforeSlotUpdate(BuildingSlotView view, BuildingSlotViewModel model, bool highlightUpdate)
        {
            // Finish the old presentation before binding a newer event/owner to this slot.
            if (!highlightUpdate) { var owner = view.GetComponent<SlotFlipLifetime>(); if (owner != null) owner.Cancel(); }
        }

        public static void UpdatePlayers(PlayersSummaryView view, List<PlayerViewModel> models)
        {
            var layout = view.GetComponent<PlayerRowsLayout>() ?? view.gameObject.AddComponent<PlayerRowsLayout>();
            layout.Initialize(view);
            layout.CancelHighlight();
            for (int i = 0; i < layout.Rows.childCount; i++) {
                var row = layout.Rows.GetChild(i); row.gameObject.SetActive(i < models.Count);
                if (i < models.Count) row.GetComponent<PlayerView>().PlayerColor = models[i].Color;
            }
            ReflectionAccess.Original(view, "UpdatePlayersSummaryView", new[] { typeof(List<PlayerViewModel>) }, models);
            layout.RebuildVertical();
        }

        public static IAnimation CurrentPlayer(PlayersSummaryView view, PlayerColor color)
        {
            var layout = view.GetComponent<PlayerRowsLayout>() ?? view.gameObject.AddComponent<PlayerRowsLayout>();
            layout.Initialize(view); layout.CancelHighlight();
            var sequence = DOTween.Sequence();
            sequence.AppendInterval(.1f);
            sequence.AppendCallback(() => { if (view != null) view.GetComponent<AudioSource>().PlayOneShot(view.PlayerChangedClip); });
            for (int i = 0; i < layout.Rows.childCount; i++)
            {
                var row = (RectTransform)layout.Rows.GetChild(i);
                sequence.Insert(.1f, row.DOAnchorPosX(row.GetComponent<PlayerView>().PlayerColor == color ? 35f : 0f, UIProperties.AnimationDuration));
            }
            // The stock animation starts a detached tween inside a callback. Nesting it
            // makes replay cancellation and a fresh HUD update cancel the whole movement.
            layout.Highlight = sequence;
            sequence.OnKill(() => { if (layout != null && layout.Highlight == sequence) layout.Highlight = null; });
            return new DOTweenAnimationSequence(sequence);
        }

        public static void BeforeRound(PlayersSummaryView view, BoardGameRules.Entities.GamePhases.Events.RoundEnded change)
        {
            var layout = view.GetComponent<PlayerRowsLayout>();
            if (layout != null) { layout.CancelHighlight(); layout.RebuildVertical(); }
        }

        sealed class AutomaticPanel { }
        static readonly ConditionalWeakTable<BaseView, AutomaticPanel> AutomaticPanels = new ConditionalWeakTable<BaseView, AutomaticPanel>();
        static int automaticIndustryDepth;
        public static void AfterPush(Router router, IBaseView view)
        {
            var panel = view as IndustriesDetailsView;
            if (panel != null) {
                AutomaticPanels.Remove(panel);
                if (automaticIndustryDepth > 0) AutomaticPanels.Add(panel, new AutomaticPanel());
            }
            KeepInformationAccessible(router);
        }
        static bool Information(IBaseView view)
        {
            var baseView = view as BaseView;
            if (baseView == null || !baseView.IsViewVisible()) return false;
            if (baseView is GameProgressDetailsView || baseView is GameLogView) return true;
            var industries = baseView as IndustriesDetailsView;
            AutomaticPanel automatic;
            return industries != null && !AutomaticPanels.TryGetValue(industries, out automatic)
                && ReflectionAccess.Get<GameObject>(industries, "_backButton").activeSelf
                && !ReflectionAccess.Get<bool>(industries, "bMerchantBonus");
        }
        public static void KeepInformationAccessible(Router router)
        {
            var stack = ReflectionAccess.Get<List<IBaseView>>(router, "_viewStack");
            if (stack.Count == 0) return;
            // Only presentation overlays yield to an already-open information panel.
            // Save, confirmation and action-selection dialogs retain their modal order.
            var top = stack[0];
            if (!(top is ReplayView) && !(top is EndOfEraView) && !(top is EndOfGameView)) return;
            foreach (var panel in stack.Where(Information).Reverse().ToArray())
            {
                ((BaseView)panel).transform.SetAsLastSibling();
                stack.Remove(panel); stack.Insert(0, panel);
            }
        }
        public static IAnimation Develop(ReplayView replay, IndustryDeveloped change)
        {
            var router = ReflectionAccess.Get<IRouter>(replay, "_router");
            if (router.IsViewVisible<IIndustriesDetailsView>() && Information(router.GetView<IIndustriesDetailsView>()))
            {
                // This is the user's inspection board. Replay must not animate its
                // tokens or retain references to it after the user closes the panel.
                return new DOTweenAnimationSequence(DOTween.Sequence().AppendInterval(.3f));
            }
            automaticIndustryDepth++;
            try { return (IAnimation)ReflectionAccess.Original(replay, "CreateAnimation", new[] { typeof(IndustryDeveloped) }, change); }
            finally { automaticIndustryDepth--; }
        }
    }

    public sealed class SlotFlipLifetime : MonoBehaviour
    {
        public Sequence Animation;
        public void Cancel() { if (Animation != null && Animation.IsActive()) Animation.Kill(); Animation = null; }
        void OnDestroy() { Cancel(); }
    }
    public sealed class PlayerRowsLayout : MonoBehaviour
    {
        public Transform Rows;
        public Sequence Highlight;
        public void Initialize(PlayersSummaryView view) { Rows = ReflectionAccess.Get<Transform>(view, "_playersContainer"); }
        public void CancelHighlight() { if (Highlight != null && Highlight.IsActive()) Highlight.Kill(); Highlight = null; }
        public void RebuildVertical()
        {
            if (Rows == null) return;
            var group = Rows.GetComponent<VerticalLayoutGroup>();
            if (group == null) return;
            // Preserve horizontal current-player/round animation; only the layout's
            // vertical axis owns seat spacing. Recalculate even after a hidden refresh.
            group.CalculateLayoutInputHorizontal(); group.CalculateLayoutInputVertical(); group.SetLayoutVertical();
        }
        void LateUpdate()
        {
            if (Rows == null) return;
            RectTransform previous = null;
            foreach (Transform child in Rows)
            {
                var row = child as RectTransform;
                if (row == null || !row.gameObject.activeSelf) continue;
                if (previous != null && previous.anchoredPosition.y - row.anchoredPosition.y < (previous.rect.height + row.rect.height) * .49f)
                { RebuildVertical(); return; }
                previous = row;
            }
        }
        void OnDisable() { CancelHighlight(); }
        void OnDestroy() { CancelHighlight(); }
    }
}
