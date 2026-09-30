using System;
using System.Runtime.CompilerServices;
using BoardGameRules.Entities.Actions.Events;
using BoardGameRules.Entities.EventSourcing.Events;
using BoardGameRules.Entities.Players;
using BrassApplication.UseCases.Game.Replay;
using DG.Tweening;
using Frameworks.Utils;
using Frameworks.View.Animation;
using Frameworks.View.Game.Replay;
using InterfaceAdapters.Common;
using InterfaceAdapters.Common.LoadScene;
using InterfaceAdapters.Game.Replay;
using InterfaceAdapters.Routing;
using InterfaceAdapters.Utils.MainThreadDispatcher;
using UnityEngine;
using UnityEngine.UI;

namespace BrassPatch.Presentation
{
    public static class LinkReplay
    {
        internal sealed class ReplayOwner { public Ticket Current; }
        static readonly ConditionalWeakTable<ReplayPresenter, ReplayOwner> Owners = new ConditionalWeakTable<ReplayPresenter, ReplayOwner>();

        public static void ReplayEvent(ReplayPresenter presenter, IEvent change, IEventReplayDelegate callback)
        {
            Cancel(presenter);
            if (!(change is LinkPlaced)) {
                ReflectionAccess.Original(presenter, "ReplayEvent", new[] { typeof(IEvent), typeof(IEventReplayDelegate) }, change, callback);
                return;
            }
            var owner = Owners.GetOrCreateValue(presenter);
            var ticket = new Ticket(owner, ReflectionAccess.Get<IRouter>(presenter, "_router"), change, callback);
            owner.Current = ticket;
            // No detached Task.Run: a cancelled queued start must never play an old event.
            ReflectionAccess.Get<IMainThreadDispatcher>(presenter, "_mainThreadDispatcher").InvokeAsync(ticket.Start);
        }

        public static void Cancel(ReplayPresenter presenter)
        {
            ReplayOwner owner;
            if (Owners.TryGetValue(presenter, out owner) && owner.Current != null) owner.Current.Cancel();
        }

        internal sealed class Ticket
        {
            readonly ReplayOwner owner;
            readonly IRouter router;
            readonly IEvent change;
            readonly IEventReplayDelegate callback;
            ReplayView view;
            Sequence sequence;
            bool closed, started;
            float elapsed, allowance;
            internal Ticket(ReplayOwner owner, IRouter router, IEvent change, IEventReplayDelegate callback)
            { this.owner = owner; this.router = router; this.change = change; this.callback = callback; }
            bool Current { get { return !closed && ReferenceEquals(owner.Current, this); } }
            bool Valid()
            {
                var replay = callback as ReplayUseCase;
                return Current && view != null && view.isActiveAndEnabled && view.IsViewVisible()
                    && ReferenceEquals(Utils.Global.Instance.Router, router)
                    && !router.IsViewVisible<ILoadSceneView>()
                    && (replay == null || replay.GetIsReplayInProgress());
            }
            internal void Start()
            {
                if (!Current) return;
                view = router.GetView<IReplayView>() as ReplayView;
                if (!Valid()) { Cancel(); return; }
                var guard = view.GetComponent<LinkReplayGuard>() ?? view.gameObject.AddComponent<LinkReplayGuard>();
                guard.Current = this;
                try { sequence = ((DOTweenAnimationSequence)view.CreateAnimation(change)).Sequence; }
                catch (Exception failure) {
                    // Link state has already been applied. A missing presentation
                    // object must not prevent resource prompts/action controls.
                    Debug.LogWarning("Brass patch: link animation unavailable: " + failure.GetType().Name);
                    sequence = DOTween.Sequence().AppendInterval(.01f).SetId("AnimationFactory");
                }
                allowance = Math.Max(3f, sequence.Duration(false) + 2f);
                sequence.OnComplete(Complete);
                started = true;
                sequence.Play();
            }
            internal void Tick(float delta)
            {
                if (!started || !Current) return;
                if (!Valid()) { Cancel(); return; }
                // A killed tween never calls OnComplete. A paused/stuck active tween
                // can also lose its callback; only this short, non-interactive event
                // gets a bounded presentation allowance. Explicit replay pause and
                // Unity time pause do not consume that allowance.
                if (!sequence.IsActive() || sequence.IsComplete()) { Recover(); return; }
                if (sequence.timeScale > 0f) elapsed += delta * sequence.timeScale;
                if (elapsed >= allowance) Recover();
            }
            void Close()
            {
                closed = true;
                if (ReferenceEquals(owner.Current, this)) owner.Current = null;
            }
            internal void Cancel()
            {
                Close();
                if (sequence != null && sequence.IsActive()) sequence.Kill();
            }
            void Complete()
            {
                if (!Valid()) { Cancel(); return; }
                Close();
                callback.EventReplayCompleted(change);
            }
            void Recover()
            {
                if (!Valid()) { Cancel(); return; }
                Close();
                if (sequence.IsActive()) sequence.Kill();
                Debug.LogWarning("Brass patch: recovered interrupted link replay " + ((LinkPlaced)change).LinkId);
                // Advance the existing replay queue. Resource selection and action
                // confirmation still run through the normal network use case.
                callback.EventReplayCompleted(change);
            }
        }

        public static IAnimation PlaceLink(BoardView board, string id, PlayerColor color)
        {
            var links = ReflectionAccess.Get<Transform>(board, "_linksContainer");
            bool canal = id.StartsWith("Canal", StringComparison.Ordinal);
            var link = links.Find(id.Substring(canal ? 5 : 4));
            var sequence = DOTween.Sequence();
            // A temporarily unavailable visual target must not strand the action.
            if (link == null) { sequence.AppendInterval(.01f); return new DOTweenAnimationSequence(sequence); }
            foreach (Transform item in links) {
                var old = item.GetComponent<LinkVisualLifetime>(); if (old != null) old.CancelPulse();
                var outlineNode = item.Find("LinkToken/Outline"); if (outlineNode != null) outlineNode.gameObject.SetActive(false);
            }
            var lifetime = link.GetComponent<LinkVisualLifetime>() ?? link.gameObject.AddComponent<LinkVisualLifetime>();
            lifetime.CancelScore(); lifetime.CancelPlacement();
            var linkView = link.GetComponent<LinkView>();
            var token = ReflectionAccess.Get<Image>(linkView, "_tokenImage");
            var outlineObject = ReflectionAccess.Get<GameObject>(linkView, "_outlineObject");
            var outline = outlineObject == null ? null : outlineObject.GetComponent<Image>();
            if (token == null || outline == null) { sequence.AppendInterval(.01f); return new DOTweenAnimationSequence(sequence); }
            lifetime.Placement = sequence;
            var sprite = UnityEngine.Resources.Load<Sprite>("Board/Links/Token" + (canal ? "CanalEra" : "RailEra") + color);
            Action restore = () => {
                if (token != null) { token.transform.localScale = Vector3.one; if (sprite != null) token.sprite = sprite; token.color = Color.white; }
                if (outline != null) { outline.gameObject.SetActive(false); outline.transform.localScale = Vector3.one; outline.color = Color.white; }
                if (lifetime != null && lifetime.Placement == sequence) lifetime.Placement = null;
            };
            var scroll = (Sequence)typeof(BoardView).GetMethod("ScrollBoardToGivenLocation", ReflectionAccess.Flags,
                null, new[] { typeof(RectTransform) }, null).Invoke(board, new object[] { (RectTransform)link });
            sequence.Append(scroll);
            sequence.AppendCallback(() => {
                if (link == null || token == null || outline == null) return;
                if (linkView != null && linkView.GetComponent<AudioSource>() != null && linkView.PlaceLinkClip != null) linkView.PlayPlaceLinkClip();
                token.gameObject.SetActive(true); if (sprite != null) token.sprite = sprite; token.color = Color.white;
                token.transform.localScale = Vector3.one * .7f;
                outline.transform.position = token.transform.position;
                outline.transform.localScale = Vector3.one; outline.color = Color.white; outline.gameObject.SetActive(true);
            });
            // All effects belong to the replay sequence, so speed and cancellation
            // apply to the token, outline and camera together.
            var effects = DOTween.Sequence();
            effects.AppendInterval(.5f);
            effects.Append(token.transform.DOScale(Vector3.one * 1.2f, UIProperties.FrameDuration * 3f));
            effects.Append(token.transform.DOScale(Vector3.one, UIProperties.FrameDuration * 8f));
            effects.Insert(.5f, outline.transform.DOScale(Vector3.one * 3f, UIProperties.FrameDuration * 6f));
            effects.Insert(.5f, outline.DOFade(0f, UIProperties.FrameDuration * 6f));
            sequence.Append(effects);
            sequence.AppendCallback(() => restore());
            sequence.OnKill(() => restore());
            return new DOTweenAnimationSequence(sequence);
        }

        public static void BeforeLinkUpdate(LinkView view, InterfaceAdapters.Game.Board.LinkViewModel model,
            BrassApplication.UseCases.Game.Actions.ILinkSelectedDelegate callback)
        {
            var lifetime = view.GetComponent<LinkVisualLifetime>();
            if (lifetime != null) { lifetime.CancelPulse(); lifetime.CancelScore(); lifetime.CancelPlacement(); }
        }
        public static void RemoveLinkToken(LinkView view, int points)
        {
            var owner = view.GetComponent<LinkVisualLifetime>() ?? view.gameObject.AddComponent<LinkVisualLifetime>();
            owner.CancelPulse(); owner.CancelPlacement(); owner.CancelScore();
            var token = ReflectionAccess.Get<Image>(view, "_tokenImage");
            var label = ReflectionAccess.Get<TMPro.TextMeshProUGUI>(view, "_victoryPoint");
            var tokenPose = new Pose(token.transform); var labelPose = new Pose(label.transform);
            string originalText = label.text;
            var sequence = DOTween.Sequence(); owner.Score = sequence;
            Action restore = () => {
                tokenPose.Restore(); labelPose.Restore();
                if (token != null) token.color = new Color(token.color.r, token.color.g, token.color.b, 0f);
                if (label != null) { label.text = originalText; label.alpha = 0f; }
                if (owner != null && owner.Score == sequence) owner.Score = null;
            };
            sequence.AppendCallback(() => {
                if (view == null || token == null || label == null) return;
                label.text = originalText.Replace("1", points.ToString());
                token.transform.SetParent(view.transform.parent.parent); token.transform.SetAsLastSibling();
                label.transform.SetParent(view.transform.parent.parent); label.transform.SetAsLastSibling();
            });
            sequence.Append(token.transform.DOScale(2f, 1f)).Join(token.DOFade(0f, 1f));
            sequence.Join(label.DOFade(1f, 1f)).Join(label.transform.DOMoveY(view.transform.position.y + 100f, 1f));
            sequence.Append(label.transform.DOScale(2f, 1f)).Join(label.DOFade(0f, 1f));
            sequence.AppendCallback(() => restore()); sequence.OnKill(() => restore());
        }
        sealed class Pose
        {
            readonly Transform target, parent;
            readonly Vector3 position, scale;
            readonly Quaternion rotation;
            readonly int sibling;
            internal Pose(Transform target)
            { this.target = target; parent = target.parent; position = target.localPosition; scale = target.localScale; rotation = target.localRotation; sibling = target.GetSiblingIndex(); }
            internal void Restore()
            {
                if (target == null || parent == null) return;
                target.SetParent(parent, false); target.SetSiblingIndex(sibling);
                target.localPosition = position; target.localScale = scale; target.localRotation = rotation;
            }
        }
        public static void Outline(LinkView view)
        {
            var owner = view.GetComponent<LinkVisualLifetime>() ?? view.gameObject.AddComponent<LinkVisualLifetime>();
            owner.CancelPulse();
            var outline = view.transform.Find("LinkToken/Outline").GetComponent<Image>();
            var enter = DOTween.Sequence(); owner.Pulse = enter;
            outline.transform.localScale = Vector3.one * 2f;
            outline.color = new Color(1f, 1f, 1f, 0f);
            enter.Append(outline.transform.DOScale(Vector3.one, .5f)).Join(outline.DOFade(1f, .5f));
            enter.OnComplete(() => {
                if (owner == null || owner.Pulse != enter || !outline.gameObject.activeInHierarchy) return;
                var pulse = DOTween.Sequence(); owner.Pulse = pulse;
                pulse.Append(outline.transform.DOScale(Vector3.one * 1.1f, .2f));
                pulse.Append(outline.transform.DOScale(Vector3.one, .2f));
                pulse.AppendInterval(UnityEngine.Random.Range(1.75f, 2.25f)); pulse.SetLoops(-1);
            });
            // Killing an old highlight must never create another infinite pulse.
        }
    }

    public sealed class LinkReplayGuard : MonoBehaviour
    {
        internal LinkReplay.Ticket Current;
        void LateUpdate() { if (Current != null) Current.Tick(Time.deltaTime); }
        void OnDisable() { if (Current != null) { Current.Cancel(); Current = null; } }
    }
    public sealed class LinkVisualLifetime : MonoBehaviour
    {
        internal Sequence Pulse, Placement, Score;
        internal void CancelPulse() { var pulse = Pulse; Pulse = null; if (pulse != null && pulse.IsActive()) pulse.Kill(); }
        internal void CancelPlacement() { var placement = Placement; if (placement != null && placement.IsActive()) placement.Kill(); Placement = null; }
        internal void CancelScore() { var score = Score; if (score != null && score.IsActive()) score.Kill(); Score = null; }
        void OnDisable() { CancelPulse(); CancelScore(); CancelPlacement(); }
    }
}
