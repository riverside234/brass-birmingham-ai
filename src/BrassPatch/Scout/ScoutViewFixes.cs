using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using DG.Tweening;
using Frameworks.Utils;
using Frameworks.View.Game;
using BrassApplication.UseCases.Game.Actions.ScoutAction;
using BrassApplication.UseCases.Game.Actions;
using BrassApplication.Game;
using InterfaceAdapters.Game.Actions.ScoutAction;
using InterfaceAdapters.Routing;
using UnityEngine;
using UnityEngine.UI;

namespace BrassPatch.Scout
{
    public sealed class ScoutViewLifetime : MonoBehaviour
    {
        internal ScoutActionView Owner;
        internal ScoutActionUseCase UseCase;
        internal object Operation;
        private void OnDestroy()
        {
            ScoutViewFixes.StopAnimations(Owner, true);
            ScoutFixes.InvalidateOperation(UseCase, Operation);
        }
    }

    public static class ScoutViewFixes
    {
        private sealed class CardPosition
        {
            public Transform Card;
            public Transform Parent;
            public int Index;
            public Vector3 Position;
            public Vector3 Scale;
            public Vector3 HolderPosition;
            public bool Ignore;
            public bool Interactable;
            public bool Raycasts;
        }
        private sealed class ViewState
        {
            public readonly List<Sequence> Tweens = new List<Sequence>();
            public readonly List<CardPosition> Hand = new List<CardPosition>();
            public readonly List<CardPosition> Selected = new List<CardPosition>();
            public readonly List<CardPosition> Wild = new List<CardPosition>();
            public Transform Parent;
            public Vector3 Slot0Scale;
            public Vector3 Slot1Scale;
            public bool SlotSnapshot;
            public bool Stopped;
            public bool Completed;
        }
        private static readonly ConditionalWeakTable<ScoutActionView, ViewState> States = new ConditionalWeakTable<ScoutActionView, ViewState>();
        private static Transform Field(ScoutActionView view, string name) { return ReflectionAccess.Get<Transform>(view, name); }
        private static CardPosition Position(Transform card)
        {
            BaseCardView view = card.GetComponent<BaseCardView>(); CanvasGroup canvas = card.GetComponent<CanvasGroup>();
            return new CardPosition { Card = card, Parent = card.parent, Index = card.GetSiblingIndex(), Position = card.localPosition, Scale = card.localScale,
                HolderPosition = view != null && view.HolderObject != null ? view.HolderObject.localPosition : Vector3.zero,
                Ignore = view != null && view.ignoreMouseOver, Interactable = canvas == null || canvas.interactable, Raycasts = canvas == null || canvas.blocksRaycasts };
        }
        private static ViewState State(ScoutActionView view)
        {
            ViewState state = States.GetOrCreateValue(view);
            if (!state.SlotSnapshot)
            {
                state.Slot0Scale = Field(view, "_slot0").localScale;
                state.Slot1Scale = Field(view, "_slot1").localScale; state.SlotSnapshot = true;
                state.Wild.Add(Position(Field(view, "_wildIndustryCard")));
                state.Wild.Add(Position(Field(view, "_wildLocationCard")));
            }
            if (view.GetComponent<ScoutViewLifetime>() == null)
            {
                var lifetime = view.gameObject.AddComponent<ScoutViewLifetime>(); lifetime.Owner = view;
                Bind(view, lifetime);
            }
            return state;
        }
        private static void Bind(ScoutActionView view, ScoutViewLifetime lifetime)
        {
            var controller = view.Controller as ScoutActionController;
            lifetime.UseCase = controller == null ? null : ReflectionAccess.Get<object>(controller, "_useCase") as ScoutActionUseCase;
            lifetime.Operation = ScoutFixes.CaptureOperation(lifetime.UseCase);
        }
        public static void PrepareForShow(ScoutActionView view)
        {
            // Router can reuse a cached view while its previous hide animation is pending.
            // Reset before Show (not its delayed completion), then bind the new attempt.
            ViewState old;
            if (States.TryGetValue(view, out old))
            {
                StopAnimations(view, true);
                if (old.SlotSnapshot)
                { Field(view, "_slot0").localScale = old.Slot0Scale; Field(view, "_slot1").localScale = old.Slot1Scale; }
                States.Remove(view);
            }
            State(view);
            Bind(view, view.GetComponent<ScoutViewLifetime>());
        }
        public static void ShowExchange(ScoutActionPresenter presenter, BrassApplicationGame game, ISelectCardDelegate selector)
        {
            IRouter router = ReflectionAccess.Get<IRouter>(presenter, "_router");
            ScoutActionView view = router.GetView<IScoutActionView>() as ScoutActionView;
            if (view != null) PrepareForShow(view);
            ReflectionAccess.Original(presenter, "ShowExchangeForWildCardsView", new[] { typeof(BrassApplicationGame), typeof(ISelectCardDelegate) }, game, selector);
        }
        private static void Restore(CardPosition position)
        {
            if (position.Card == null || position.Parent == null) return;
            position.Card.SetParent(position.Parent, false); position.Card.SetSiblingIndex(position.Index);
            position.Card.localPosition = position.Position; position.Card.localScale = position.Scale;
            BaseCardView card = position.Card.GetComponent<BaseCardView>();
            if (card != null)
            {
                card.ignoreMouseOver = position.Ignore;
                if (card.HolderObject != null) card.HolderObject.localPosition = position.HolderPosition;
            }
            CanvasGroup canvas = position.Card.GetComponent<CanvasGroup>();
            if (canvas != null) { canvas.interactable = position.Interactable; canvas.blocksRaycasts = position.Raycasts; }
        }

        public static void StopAnimations(ScoutActionView view, bool restoreCards)
        {
            if (System.Object.ReferenceEquals(view, null)) return;
            ViewState state; if (!States.TryGetValue(view, out state) || state.Stopped) return;
            state.Stopped = true;
            foreach (Sequence tween in state.Tweens.ToArray()) if (tween != null && tween.IsActive()) tween.Kill(false);
            state.Tweens.Clear();
            if (restoreCards && !state.Completed)
            {
                // Restore the original whole-hand ordering: the second selection's current
                // sibling index has already shifted after the first card left the hand.
                foreach (CardPosition item in state.Hand.OrderBy(x => x.Index)) Restore(item);
                foreach (CardPosition item in state.Wild) Restore(item);
            }
            if (state.Parent != null)
            {
                HorizontalLayoutGroup layout = state.Parent.GetComponent<HorizontalLayoutGroup>();
                if (layout != null) layout.enabled = true;
            }
        }

        public static void HideExchange(ScoutActionPresenter presenter)
        {
            IRouter router = ReflectionAccess.Get<IRouter>(presenter, "_router");
            if (router.IsViewCached<IScoutActionView>())
            {
                ScoutActionView view = router.GetView<IScoutActionView>() as ScoutActionView;
                if (view != null) StopAnimations(view, true);
            }
            if (router.IsViewVisible<IScoutActionView>())
                ReflectionAccess.Original(presenter, "HideExchangeForWildCardsView", Type.EmptyTypes);
        }

        public static void ShowViewCompleted(ScoutActionView view)
        {
            ViewState state = State(view); if (state.Stopped) return;
            Transform slot0 = Field(view, "_slot0"), slot1 = Field(view, "_slot1");
            Transform industrySlot = Field(view, "_wildIndustrySlot"), locationSlot = Field(view, "_wildLocationSlot");
            Transform industry = Field(view, "_wildIndustryCard"), location = Field(view, "_wildLocationCard");
            slot0.gameObject.SetActive(true); slot1.gameObject.SetActive(true); industrySlot.gameObject.SetActive(true); locationSlot.gameObject.SetActive(true);
            Sequence tween = DOTween.Sequence(); state.Tweens.Add(tween); tween.OnKill(delegate { state.Tweens.Remove(tween); });
            tween.Append(industry.DOMove(industrySlot.position, 5f * UIProperties.FrameDuration)).SetEase(Ease.OutQuart);
            tween.Insert(2f * UIProperties.FrameDuration, location.DOMove(locationSlot.position, 5f * UIProperties.FrameDuration)).SetEase(Ease.OutQuart);
            tween.AppendInterval(1.5f); tween.Append(slot0.DOScale(0.35f * Vector3.one, 4f * UIProperties.FrameDuration));
            tween.Append(slot0.DOScale(0.3f * Vector3.one, 3f * UIProperties.FrameDuration));
        }

        public static void AnimateCardToSlot(ScoutActionView view, int index, Tuple<object, int> info, Action callback)
        {
            ViewState state = State(view);
            if (state.Stopped || state.Completed || index != state.Selected.Count || index < 0 || index > 1 || info == null) return;
            GameObject go = info.Item1 as GameObject;
            if (go == null || state.Selected.Any(x => x.Card == go.transform)) return;
            BaseCardView card = go.GetComponent<BaseCardView>();
            if (card == null || card.HolderObject == null || go.transform.parent == null) return;
            Transform slot0 = Field(view, "_slot0"), slot1 = Field(view, "_slot1");
            Transform industry = Field(view, "_wildIndustryCard"), location = Field(view, "_wildLocationCard");
            if (state.Parent == null)
            {
                state.Parent = go.transform.parent;
                foreach (Transform child in state.Parent) state.Hand.Add(Position(child));
            }
            CardPosition selected = state.Hand.FirstOrDefault(x => x.Card == go.transform);
            if (selected == null) return;
            state.Selected.Add(selected);
            // Reserve immediately, before the first DOTween update or another click.
            card.ignoreMouseOver = true;
            CanvasGroup group = go.GetComponent<CanvasGroup>(); if (group != null) { group.interactable = false; group.blocksRaycasts = false; }
            HorizontalLayoutGroup layout = state.Parent.GetComponent<HorizontalLayoutGroup>(); if (layout != null) layout.enabled = false;
            Transform slot = index == 0 ? slot0 : slot1;
            Sequence sequence = DOTween.Sequence(); state.Tweens.Add(sequence); sequence.OnKill(delegate { state.Tweens.Remove(sequence); });
            sequence.AppendCallback(delegate { go.transform.SetParent(slot); });
            sequence.Append(go.transform.DOLocalMove(Vector3.zero, 11f * UIProperties.FrameDuration)).Join(card.HolderObject.DOLocalMove(Vector3.zero, 11f * UIProperties.FrameDuration));
            sequence.AppendInterval(7f * UIProperties.FrameDuration); sequence.Append(go.transform.DOScale(Vector3.one, 7f * UIProperties.FrameDuration));
            if (index == 0)
            {
                sequence.Append(industry.DOBlendableMoveBy(new Vector3(-30f, 30f, 0f), 13f * UIProperties.FrameDuration));
                sequence.AppendCallback(delegate { slot1.GetChild(0).gameObject.SetActive(false); slot1.GetChild(1).gameObject.SetActive(true); });
                sequence.Append(slot1.DOScale(0.35f * Vector3.one, 4f * UIProperties.FrameDuration));
                sequence.Append(slot1.DOScale(0.3f * Vector3.one, 3f * UIProperties.FrameDuration));
            }
            else
            {
                BaseCardView industryCard = industry.GetComponent<BaseCardView>(), locationCard = location.GetComponent<BaseCardView>();
                industryCard.bonusY = 140f; locationCard.bonusY = 140f;
                sequence.AppendCallback(delegate {
                    industry.SetParent(state.Parent); location.SetParent(state.Parent);
                    // Two wilds replace the two selected positions in the original hand order.
                    int first = state.Selected[0].Index, second = state.Selected[1].Index;
                    industry.SetSiblingIndex(first - (second < first ? 1 : 0));
                    location.SetSiblingIndex(second);
                });
                sequence.Append(location.DOBlendableMoveBy(new Vector3(-30f, 30f, 0f), 13f * UIProperties.FrameDuration));
                sequence.Append(industry.DOBlendableMoveBy(new Vector3(0f, 50f, 0f), 6f * UIProperties.FrameDuration)).Join(location.DOBlendableMoveBy(new Vector3(0f, 50f, 0f), 6f * UIProperties.FrameDuration));
                Vector3 firstPosition = state.Parent.TransformPoint(state.Selected[0].Position), secondPosition = state.Parent.TransformPoint(state.Selected[1].Position);
                sequence.Append(industry.DOMove(firstPosition - new Vector3(0f, 20f, 0f), 7f * UIProperties.FrameDuration)).Join(location.DOMove(secondPosition - new Vector3(0f, 20f, 0f), 7f * UIProperties.FrameDuration))
                    .Join(industry.DOScale(0.4f * Vector3.one, 6f * UIProperties.FrameDuration)).Join(location.DOScale(0.4f * Vector3.one, 6f * UIProperties.FrameDuration))
                    .Join(industryCard.HolderObject.GetComponent<RectTransform>().DOAnchorPosY(industryCard.bonusY + 130f, 6f * UIProperties.FrameDuration))
                    .Join(locationCard.HolderObject.GetComponent<RectTransform>().DOAnchorPosY(locationCard.bonusY + 130f, 6f * UIProperties.FrameDuration))
                    .Join(slot0.DOScale(Vector3.zero, 6f * UIProperties.FrameDuration)).Join(slot1.DOScale(Vector3.zero, 6f * UIProperties.FrameDuration));
                sequence.Append(industry.DOMove(firstPosition, 6f * UIProperties.FrameDuration)).Join(location.DOMove(secondPosition, 6f * UIProperties.FrameDuration));
                sequence.AppendCallback(delegate { if (layout != null) layout.enabled = true; });
                sequence.AppendInterval(1f);
            }
            sequence.AppendCallback(delegate {
                if (state.Stopped || view == null) return;
                if (index == 1) state.Completed = true;
                if (callback != null) callback();
            });
            sequence.Play();
        }
    }
}
