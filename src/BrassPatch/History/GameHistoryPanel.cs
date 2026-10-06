using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using BrassPatch.Statistics;
using BrassPatch.UI;
using DG.Tweening;
using Frameworks.View.Common;
using InterfaceAdapters.Common;
using InterfaceAdapters.Routing;
using TMPro;
using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.UI;

namespace BrassPatch.History
{
    public sealed class GameHistoryPanel : BaseView
    {
        sealed class ControllerStub : IController { public void OnViewCreated() { } public void OnViewDestroyed() { } }
        public override IController Controller { get; set; } = new ControllerStub();
        public bool IsOpen { get { return this != null && _isViewVisible && gameObject.activeInHierarchy; } }
        public float TextScale { get; private set; } = 1.15f;
        public ScrollRect GameScroll { get; private set; }
        public Button[] Rows { get; private set; } = new Button[0];
        public Action Tick, OnBack, OnRetry, OnDestroyed;
        public Action<string> OnOpen;
        TMP_FontAsset font;
        RectTransform window;
        Button close, back, smaller, larger;
        TextMeshProUGUI status;
        IReadOnlyList<GameHistorySummary> games = new List<GameHistorySummary>().AsReadOnly();
        float unit, width, height, position = 1;
        string selected, message = "Loading recorded games…";
        bool closing;
        int screenWidth, screenHeight;
        protected override void Awake() { DestroyWhenClosed = false; DisableMenusUnderneath = false; }
        protected override void Start() { if (InputRegistration != null) InputRegistration.RegisterKey(KeyEnum.Escape, this); }
        void OnDestroy() { if (OnDestroyed != null) OnDestroyed();if (InputRegistration != null) InputRegistration.UnregisterAllKeys(this); }
        public void Initialize(TMP_FontAsset typeface)
        { font = typeface; gameObject.AddComponent<StatisticsTheme>(); gameObject.AddComponent<LightUiTheme>(); Build(); }
        public override IAnimation CreateShowViewAnimation()
        {
            closing = false; _isViewVisible = true; EnableView(); MoveToTop();
            if (GameScroll != null) GameScroll.verticalNormalizedPosition = position;
            var row = Rows.FirstOrDefault(b => b.name == selected);
            if (EventSystem.current != null) EventSystem.current.SetSelectedGameObject(row == null ? close.gameObject : row.gameObject);
            return new Frameworks.View.Animation.DOTweenAnimation(DOTween.Sequence().AppendInterval(.001f));
        }
        public override IAnimation CreateShowViewImmediatelyAnimation() { return CreateShowViewAnimation(); }
        public override IAnimation CreateHideViewAnimation()
        {
            Remember(); _isViewVisible = false; gameObject.SetActive(false);
            return new Frameworks.View.Animation.DOTweenAnimation(DOTween.Sequence().AppendInterval(.001f));
        }
        public override void SetInteractable(bool value) { GetComponent<CanvasGroup>().interactable = value; }
        public void Close()
        { if (closing || !IsOpen) return; closing = true; Remember(); if (OnBack != null) OnBack(); }
        public void SetGames(GameHistoryList list)
        { Remember(); games = list.Games; message = list.Error ?? (games.Count == 0 ? "No completed games yet. Finish a local game to record its statistics." : ""); Build(); }
        public void SetStatus(string text) { message = text; if (status != null) status.text = text; }
        public void ShowDetails(string text) {if(status!=null)status.text=String.IsNullOrEmpty(message)?text:message;}
        public void Remember()
        { if (GameScroll != null) position = GameScroll.verticalNormalizedPosition; }
        public void SelectRow(string key) { selected = key; Remember();foreach(var row in Rows)StatisticsWidgets.Selected(row,row.name==key); }
        public void ChangeTextSize(int direction)
        { float next = Mathf.Clamp(TextScale + direction * .15f, 1, 1.3f); if (Mathf.Abs(next - TextScale) < .01f) return; Remember(); TextScale = next; Build(); }
        Rect Pixels()
        {
            var canvas = GetComponentInParent<Canvas>().rootCanvas;
            return canvas.renderMode != RenderMode.ScreenSpaceOverlay && canvas.worldCamera != null && canvas.worldCamera.targetTexture != null
                ? canvas.worldCamera.pixelRect : new Rect(0, 0, Screen.width, Screen.height);
        }
        void LateUpdate()
        {
            if (!IsOpen) return; if (Tick != null) Tick(); if (!IsOpen) return;
            var pixels = Pixels();float currentUnit=((RectTransform)transform).rect.width/Mathf.Max(1,pixels.width);
            if (screenWidth != (int)pixels.width || screenHeight != (int)pixels.height||Mathf.Abs(unit-currentUnit)>.001f) { Remember(); Build(); }
        }
        float F(float value) { return value * unit * TextScale; }
        TextMeshProUGUI Text(Transform parent, string name, string text, Vector2 pos, Vector2 size, float pixels, bool wrap = false)
        {
            var label = StatisticsWidgets.Text(parent, font, name, text ?? "", pos, size, F(pixels), wrap);
            label.richText = false; label.color = StatisticsTheme.Ink; return label;
        }
        Button Button(Transform parent, string name, string label, Vector2 pos, Vector2 size, float pixels, Action click)
        {
            var button = StatisticsWidgets.Button(parent, font, name, label, pos, size, F(pixels), click);
            button.GetComponentInChildren<TextMeshProUGUI>().richText = false;
            return button;
        }
        void Build()
        {
            var pixels = Pixels(); screenWidth = (int)pixels.width; screenHeight = (int)pixels.height;
            Canvas.ForceUpdateCanvases();var root = (RectTransform)transform; unit = root.rect.width / Mathf.Max(1, pixels.width); if (unit <= 0) unit = 1;
            GetComponent<LightUiTheme>().PixelUnit = unit; StatisticsWidgets.Clear(transform);
            var safe = screenWidth == Screen.width && screenHeight == Screen.height ? Screen.safeArea : pixels;
            width = Mathf.Min(safe.width * .96f, 1920) * unit; height = Mathf.Min(safe.height * .96f, 1200) * unit;
            var shield = StatisticsWidgets.Rect(transform, "History input shield", Vector2.zero, root.rect.size); StatisticsWidgets.Stretch(shield, Vector2.zero, Vector2.zero);
            shield.gameObject.AddComponent<Image>().color = new Color(0, 0, 0, .50f);
            window = StatisticsWidgets.Rect(transform, "History window", Vector2.zero, new Vector2(width, height));
            window.anchorMin = window.anchorMax = window.pivot = new Vector2(.5f, .5f); window.anchoredPosition = (safe.center - pixels.center) * unit;
            window.gameObject.AddComponent<LightUiSurface>().Role = LightSurfaceRole.Window;
            float closeX=width-22*unit-F(52),largerX=closeX-16*unit-F(58),presetX=largerX-8*unit-F(112),smallerX=presetX-8*unit-F(58);
            Text(window, "History title", "GAME HISTORY", new Vector2(22, -10) * unit, new Vector2(smallerX-44*unit, 58 * unit), 30).fontStyle = FontStyles.Bold;
            Text(window, "History context", "Latest 10 completed games · " + games.Count + " recorded", new Vector2(22, -58) * unit, new Vector2(width - 100 * unit, 38 * unit), 21).color = StatisticsTheme.Muted;
            smaller = Button(window, "Smaller text", "A−", new Vector2(smallerX, -16 * unit), new Vector2(58, 52) * unit*TextScale, 23, () => ChangeTextSize(-1));
            Text(window, "Text size", new[] { "Large", "Larger", "Largest" }[Mathf.RoundToInt((TextScale - 1) / .15f)], new Vector2(presetX, -16 * unit), new Vector2(112, 52) * unit*TextScale, 20).alignment = TextAlignmentOptions.Center;
            larger = Button(window, "Larger text", "A+", new Vector2(largerX, -16 * unit), new Vector2(58, 52) * unit*TextScale, 23, () => ChangeTextSize(1));
            smaller.interactable = TextScale > 1.01f; larger.interactable = TextScale < 1.29f;
            close = Button(window, "Close history", "×", new Vector2(closeX, -16 * unit), new Vector2(52, 52) * unit*TextScale, 28, Close);
            float detailsHeight=(TextScale>=1.29f?112:88)*unit;
            status = Text(window, "History status", message, new Vector2(22, -98) * unit, new Vector2(width - 162 * unit, detailsHeight), 20, true); status.color = StatisticsTheme.Muted;
            Button(window, "Retry history", "Refresh", new Vector2(width - 134 * unit, -103 * unit), new Vector2(112, 44) * unit, 20, () => { if (OnRetry != null) OnRetry(); });
            float listTop=110*unit+detailsHeight;
            GameScroll = StatisticsWidgets.Scroll(window, "History games", new Vector2(22*unit, -listTop), new Vector2(width - 44 * unit, height - listTop - 72*unit), unit);
            float rowWidth = GameScroll.viewport.sizeDelta.x - 6 * unit, y = 0; var rows = new List<Button>();
            foreach (var game in games) {
                var current = game; int lines = Math.Max(1, (game.Players.Count + 1) / 2);
                float rowHeight = Math.Max(150, 58 + lines * 43 + 22) * unit * TextScale;
                var row = Button(GameScroll.content, game.Key, "", new Vector2(0, -y), new Vector2(rowWidth, rowHeight), 22,
                    () => { if (!IsOpen || current.Unavailable != null) return; selected = current.Key; Remember(); if (OnOpen != null) OnOpen(current.Key); });
                row.GetComponentInChildren<TextMeshProUGUI>().gameObject.SetActive(false); row.interactable = game.Unavailable == null;
                StatisticsWidgets.Selected(row,game.Key==selected);
                string recorded = game.RecordedUtc == DateTime.MinValue ? "Recorded date unavailable" : "Recorded " + game.RecordedUtc.ToLocalTime().ToString("d MMM yyyy, HH:mm", CultureInfo.CurrentCulture);
                var meta = Text(row.transform, "Recorded match", recorded + (game.Unavailable!=null?" · Statistics unavailable":" · " + (game.Introductory ? "Introductory" : "Full game") + " · " + game.Players.Count + " players"),
                    new Vector2(16, -8) * unit, new Vector2(rowWidth - 410 * unit * TextScale, 40 * unit * TextScale), 21);
                meta.color = StatisticsTheme.Muted;
                var coverage = Text(row.transform, "History coverage", game.Unavailable != null ? "Unavailable" : game.CompleteHistory ? "Complete history" : "Summary only",
                    new Vector2(rowWidth - 384 * unit * TextScale, -8 * unit), new Vector2(220, 40) * unit * TextScale, 20); coverage.color = StatisticsTheme.Muted;
                Text(row.transform, "Open report", game.Unavailable!=null?"Unavailable":"Open ›", new Vector2(rowWidth - 145 * unit * TextScale, -8 * unit), new Vector2(128, 40) * unit * TextScale, 23).color = game.Unavailable!=null?StatisticsTheme.Muted:StatisticsTheme.Accent;
                int playerIndex = 0; float column = (rowWidth - 32 * unit) / 2;
                foreach (var player in game.Players.OrderBy(p => p.Rank == 0 ? Int32.MaxValue : p.Rank).ThenBy(p => (int)p.Color)) {
                    float x = 16 * unit + (playerIndex % 2) * column, py = (52 + playerIndex / 2 * 43) * unit * TextScale;
                    Text(row.transform, "Rank " + player.Color, player.Rank > 0 ? "#" + player.Rank : "—", new Vector2(x, -py), new Vector2(38, 39) * unit * TextScale, 22);
                    Text(row.transform, "Color " + player.Color, player.Color.ToString(), new Vector2(x + 40 * unit * TextScale, -py), new Vector2(88, 39) * unit * TextScale, 21).color = StatisticsTheme.Player(player.Color);
                    Text(row.transform, "Player " + player.Color, player.Name, new Vector2(x + 134 * unit * TextScale, -py), new Vector2(column - 280 * unit * TextScale, 39 * unit * TextScale), 24);
                    var score = Text(row.transform, "VP " + player.Color, player.FinalVP.ToString(CultureInfo.InvariantCulture) + " VP", new Vector2(x + column - 142 * unit * TextScale, -py), new Vector2(130, 39) * unit * TextScale, 25);
                    score.alignment = TextAlignmentOptions.MidlineRight; score.fontStyle = FontStyles.Bold; playerIndex++;
                }
                if (game.Unavailable != null) Text(row.transform, "Unavailable report", game.Unavailable, new Vector2(16, -52) * unit, new Vector2(rowWidth - 32 * unit, rowHeight - 62 * unit), 22, true);
                var reveal = row.gameObject.AddComponent<HistoryFocusReveal>(); reveal.Panel = this;
                var details = row.gameObject.AddComponent<HistoryRowDetails>(); details.Panel = this;
                details.Description = String.Join(" · ", game.Players.Select(p => p.Name + " (" + p.Color + ") · " + p.Opponent));
                rows.Add(row); y += rowHeight + 12 * unit;
            }
            Rows = rows.ToArray(); GameScroll.content.sizeDelta = new Vector2(rowWidth, Math.Max(y, GameScroll.viewport.rect.height));
            GameScroll.verticalNormalizedPosition = position;
            GameScroll.verticalScrollbar.gameObject.SetActive(y > GameScroll.viewport.rect.height + unit);
            back = Button(window, "Back to main menu", "Back to main menu", new Vector2(22 * unit, -height + 58 * unit), new Vector2(242 * unit * TextScale, 44 * unit), 21, Close);
            Text(window, "Retention note", "Older statistics are removed after 10 completed games.", new Vector2(284 * unit * TextScale, -height + 58 * unit), new Vector2(width - 306 * unit * TextScale, 44 * unit), 20).color = StatisticsTheme.Muted;
            var selectedRow = Rows.FirstOrDefault(r => r.name == selected);
            if (IsOpen && EventSystem.current != null) EventSystem.current.SetSelectedGameObject(selectedRow == null ? close.gameObject : selectedRow.gameObject);
        }
        public void RevealFocus(Selectable selectable)
        {
            if (GameScroll == null || !selectable.transform.IsChildOf(GameScroll.content)) return;
            var corners = new Vector3[4]; ((RectTransform)selectable.transform).GetWorldCorners(corners); var points = corners.Select(p => GameScroll.viewport.InverseTransformPoint(p)).ToArray();
            float top = points.Max(p => p.y), bottom = points.Min(p => p.y), delta = top > GameScroll.viewport.rect.yMax ? GameScroll.viewport.rect.yMax - top : bottom < GameScroll.viewport.rect.yMin ? GameScroll.viewport.rect.yMin - bottom : 0;
            var offset = GameScroll.content.anchoredPosition; offset.y = Mathf.Clamp(offset.y + delta, 0, Mathf.Max(0, GameScroll.content.rect.height - GameScroll.viewport.rect.height)); GameScroll.content.anchoredPosition = offset;
        }
        public override void HandleKeyDown(KeyEnum key)
        {
            if (key == KeyEnum.Escape) { Close(); return; }
            if (GameScroll == null) return;
            if (key == KeyEnum.Home || key == KeyEnum.End) {
                GameScroll.verticalNormalizedPosition = key == KeyEnum.Home ? 1 : 0;
                var row = key == KeyEnum.Home ? Rows.FirstOrDefault(b => b.interactable) : Rows.LastOrDefault(b => b.interactable);
                if (row != null && EventSystem.current != null) { EventSystem.current.SetSelectedGameObject(row.gameObject); RevealFocus(row); } return;
            }
            if (key == KeyEnum.PageUp || key == KeyEnum.PageDown) {
                float span = Math.Max(unit, GameScroll.content.rect.height - GameScroll.viewport.rect.height);
                GameScroll.verticalNormalizedPosition = Mathf.Clamp01(GameScroll.verticalNormalizedPosition + (key == KeyEnum.PageUp ? 1 : -1) * GameScroll.viewport.rect.height * .8f / span); return;
            }
            if (key != KeyEnum.Tab && key != KeyEnum.UpArrow && key != KeyEnum.DownArrow || EventSystem.current == null) return;
            var controls = key == KeyEnum.Tab ? GetComponentsInChildren<Button>().Where(b => b.interactable).ToArray() : Rows.Where(b => b.interactable).ToArray();
            if (controls.Length == 0) return; int index = Array.FindIndex(controls, b => b.gameObject == EventSystem.current.currentSelectedGameObject);
            int step = key == KeyEnum.UpArrow || key == KeyEnum.Tab && (Input.GetKey(KeyCode.LeftShift) || Input.GetKey(KeyCode.RightShift)) ? -1 : 1;
            index = index < 0 ? 0 : (index + step + controls.Length) % controls.Length;
            EventSystem.current.SetSelectedGameObject(controls[index].gameObject); RevealFocus(controls[index]);
        }
        public override void HandleKey(KeyEnum key) { }
    }
    public sealed class HistoryControlSurface : MaskableGraphic, IPointerEnterHandler, IPointerExitHandler, ISelectHandler, IDeselectHandler
    {
        bool hovered, focused;
        public bool Selected;
        protected override void OnPopulateMesh(VertexHelper mesh)
        { var button = GetComponentInParent<Selectable>(); LightUi.Paint(mesh, GetPixelAdjustedRect(), LightSurfaceRole.Card, Selected, focused, hovered, button == null || button.interactable, LightUiTheme.Unit(this)); }
        public void OnPointerEnter(PointerEventData data) { hovered = true; SetVerticesDirty(); }
        public void OnPointerExit(PointerEventData data) { hovered = false; SetVerticesDirty(); }
        public void OnSelect(BaseEventData data) { focused = true; SetVerticesDirty(); }
        public void OnDeselect(BaseEventData data) { focused = false; SetVerticesDirty(); }
        protected override void OnDisable() { hovered = focused = false; base.OnDisable(); }
    }
    public sealed class HistoryFocusReveal : MonoBehaviour, ISelectHandler
    { public GameHistoryPanel Panel; public void OnSelect(BaseEventData data) { if (Panel != null) Panel.RevealFocus(GetComponent<Selectable>()); } }
    public sealed class HistoryRowDetails : MonoBehaviour, IPointerEnterHandler, IPointerExitHandler, ISelectHandler, IDeselectHandler
    {
        public GameHistoryPanel Panel; public string Description;
        public void OnPointerEnter(PointerEventData data) { if (Panel != null) Panel.ShowDetails(Description); }
        public void OnPointerExit(PointerEventData data) { if (Panel != null) Panel.ShowDetails(""); }
        public void OnSelect(BaseEventData data) { if (Panel != null) Panel.ShowDetails(Description); }
        public void OnDeselect(BaseEventData data) { if (Panel != null) Panel.ShowDetails(""); }
    }
}
