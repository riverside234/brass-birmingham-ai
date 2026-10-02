using System;
using System.Collections.Generic;
using BoardGameRules.Entities.Board;
using BoardGameRules.Entities.Board.Regions;
using BoardGameRules.Entities.GamePhases;
using BrassApplication.AI;
using BrassApplication.Game;
using BrassApplication.GameHistory;
using BrassApplication.UseCases.Game.Board;
using BrassApplication.UseCases.Game.Replay;
using DG.Tweening;
using Frameworks.Routing;
using InterfaceAdapters.Common;
using InterfaceAdapters.Game.Board;
using InterfaceAdapters.Routing;
using UnityEngine;
using UnityEngine.UI;

namespace BrassPatch.CardTracker
{
    public sealed class CardTrackerPause : IDisposable
    {
        readonly float gameSpeed, tweenSpeed;
        bool released;
        public CardTrackerPause()
        { gameSpeed = Time.timeScale; tweenSpeed = DOTween.timeScale; Time.timeScale = 0; DOTween.timeScale = 0; }
        public void Dispose()
        { if (released) return; released = true; Time.timeScale = gameSpeed; DOTween.timeScale = tweenSpeed; }
    }
    public static class CardTrackerEntryPoints
    {
        static readonly Dictionary<IRouter,CardTrackerSession> sessions = new Dictionary<IRouter,CardTrackerSession>();
        public static void AfterBoard(InterfaceAdapters.Game.Board.BoardPresenter presenter, Board board, GameEra era, Action<RegionName> click)
        {
            try {
                var router = ReflectionAccess.Get<IRouter>(presenter, "_router");
                var owner = router.UseCaseBuilder.GetUseCase<BoardUseCase>();
                var history = ReflectionAccess.Get<IGameHistory>(owner, "_gameHistory");
                Attach(router, history, router.GetView<IBoardView>() as BoardView);
            } catch (MissingMemberException) { }
              catch (NotSupportedException) { }
              catch (Exception error) { Debug.LogWarning("Card Tracker attachment unavailable: " + error.Message); }
        }
        public static CardTrackerSession Attach(IRouter router, IGameHistory history, BoardView board)
        {
            var app = history == null ? null : history.CurrentGame;
            CardTrackerSession existing;
            if (app == null || board == null || (int)app.Game.GameStartConfiguration.NumberOfPlayers != 4 ||
                app.Metadata.GameType == GameType.Network || app.Game.GameProgressInformation.IsGameOver) {
                if (router != null && sessions.TryGetValue(router,out existing)) existing.Retire();
                return null;
            }
            if (sessions.TryGetValue(router,out existing) && existing != null) {
                if (existing.GameId == app.Metadata.GameId && existing.Board == board) return existing;
                existing.Retire();
            }
            // Resolve the use case before adding a persistent view owner.
            var replay = router.UseCaseBuilder.GetUseCase<ReplayUseCase>();
            var root = Statistics.StatisticsWidgets.Rect(board.transform, "Four-player Card Tracker", Vector2.zero, Vector2.zero);
            root.pivot = new Vector2(.5f,.5f);
            Statistics.StatisticsWidgets.Stretch(root,Vector2.zero,Vector2.zero);
            var canvas = root.gameObject.AddComponent<Canvas>(); canvas.overrideSorting = true; canvas.sortingOrder = 450;
            root.gameObject.AddComponent<GraphicRaycaster>();
            var session = root.gameObject.AddComponent<CardTrackerSession>();
            session.Initialize(router,history,board,replay); sessions[router] = session; return session;
        }
        public static bool ProcessInput(Router router)
        {
            CardTrackerSession session;
            if (!sessions.TryGetValue(router,out session) || session == null) return false;
            if (session.ClosedFrame == Time.frameCount) return true;
            if (!session.IsOpen) return false;
            var source = ReflectionAccess.Get<IInputSource>(router,"_inputSource");
            if (source.GetKeyDown(KeyEnum.Escape)) session.Escape();
            // No key reaches action controls, replay or the menu under the pause.
            return true;
        }
        internal static void Forget(CardTrackerSession session)
        {
            CardTrackerSession current;
            if (session.Router != null && sessions.TryGetValue(session.Router,out current) && ReferenceEquals(current,session))
                sessions.Remove(session.Router);
        }
    }
    public sealed class CardTrackerSession : MonoBehaviour
    {
        public IRouter Router { get; private set; }
        public IGameHistory History { get; private set; }
        public BoardView Board { get; private set; }
        public string GameId { get; private set; }
        public CardTrackerOverlay Overlay { get; private set; }
        public Button Toggle { get { return Overlay == null ? null : Overlay.Toggle; } }
        public CardTrackerProjection Snapshot { get; private set; }
        public int ClosedFrame = -1;
        public bool IsOpen { get { return pause != null; } }
        CardTrackerPause pause;
        ReplayUseCase replay;
        bool retired;
        public void Initialize(IRouter router,IGameHistory history,BoardView board,ReplayUseCase replayCase)
        {
            Router = router; History = history; Board = board; GameId = history.CurrentGame.Metadata.GameId;
            // Missing use-case support in a test/non-Unity presenter is not evidence
            // that a real replay has stopped; production builders supply this case.
            replay = replayCase;
            Overlay = new CardTrackerOverlay(this,(RectTransform)transform);
            Refresh();
        }
        public bool Eligible()
        {
            var app = History == null ? null : History.CurrentGame;
            if (retired || app == null || Board == null || !Board.isActiveAndEnabled || !Board.IsViewVisible() ||
                app.Metadata.GameId != GameId || app.Metadata.GameType == GameType.Network ||
                (int)app.Game.GameStartConfiguration.NumberOfPlayers != 4 || app.Game.GameProgressInformation.IsGameOver ||
                app.Game.TurnFlow.CurrentAction != null || replay == null || replay.GetIsReplayInProgress()) return false;
            if (History is BrassGameHistory && ReflectionAccess.Get<bool>(History,"_movedBackInHistory")) return false;
            if (app.Metadata.GetPlayerMetadataByColor(app.Game.GameProgressInformation.CurrentPlayer.Color).AiType != AIType.Human) return false;
            // Existing independent replay/round-scoring tweens also have to finish.
            if (DOTween.IsTweening("AnimationFactory")) return false;
            return IsOpen || Time.timeScale > 0 && DOTween.timeScale > 0;
        }
        public bool Open()
        {
            if (IsOpen || !Eligible()) return false;
            try {
                // An eligible human action boundary cannot have an AI worker/commit.
                // Capture the public prefix once; it remains frozen until close.
                Snapshot = CardTrackerProjection.Capture(History);
                pause = new CardTrackerPause();
                Overlay.Show(Snapshot); Refresh(); return true;
            } catch (Exception error) {
                Close(); Debug.LogWarning("Card Tracker could not open: " + error.Message); return false;
            }
        }
        public void Close()
        {
            var token = pause; pause = null;
            try { if (Overlay != null) Overlay.Hide(); }
            finally {
                // Scene/view teardown must release the global pause even if a
                // Unity object used by the overlay was already destroyed.
                if (token != null) { token.Dispose(); ClosedFrame = Time.frameCount; }
                Snapshot = null; Refresh();
            }
        }
        public void Escape()
        { if (Overlay != null && !Overlay.DismissDetail()) Close(); }
        public void Refresh()
        { if (Overlay != null) Overlay.RefreshToggle(!retired && (IsOpen || Eligible()),IsOpen); }
        void LateUpdate()
        {
            if (retired) return;
            var app = History.CurrentGame;
            if (app == null || app.Metadata.GameId != GameId || app.Game.GameProgressInformation.IsGameOver ||
                (int)app.Game.GameStartConfiguration.NumberOfPlayers != 4) { Retire(); return; }
            if (IsOpen && (!Eligible() || app.Game.Revision != Snapshot.Revision)) Close();
            Refresh(); if (Overlay != null) Overlay.Layout();
        }
        public void Retire()
        {
            if (retired) return; retired = true; Close(); CardTrackerEntryPoints.Forget(this);
            gameObject.SetActive(false); Destroy(gameObject);
        }
        void OnDisable() { Close(); }
        void OnDestroy() { retired = true; Close(); CardTrackerEntryPoints.Forget(this); }
    }
}
