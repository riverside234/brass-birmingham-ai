using System;
using System.Collections.Generic;
using System.Linq;
using BrassPatch;
using BrassPatch.UI;
using TMPro;
using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.UI;

namespace BrassLegacyAI
{
    public sealed class AiSelectionPanel : MonoBehaviour, ICancelHandler
    {
        static AiSelectionPanel active;
        static int closedFrame = -1;
        AiSetupRow owner;
        TextMeshProUGUI template, empty, countLabel, detailText;
        Func<AiChoice, bool> choose;
        AiChoice selected;
        string nickname, detail;
        RectTransform window, viewport, content;
        ScrollRect scroll, detailScroll;
        TMP_InputField filter;
        Button random, close, footerClose, clear, smaller, larger;
        readonly List<Button> tiles = new List<Button>();
        readonly Dictionary<AiFamily, Button> pageButtons = new Dictionary<AiFamily, Button>();
        readonly Dictionary<AiFamily, string> queries = new Dictionary<AiFamily, string>();
        readonly Dictionary<AiFamily, float> positions = new Dictionary<AiFamily, float>();
        readonly Dictionary<AiFamily, IReadOnlyList<AiCatalogEntry>> catalogs = new Dictionary<AiFamily, IReadOnlyList<AiCatalogEntry>>();
        CanvasGroup menuGroup;
        readonly Dictionary<Selectable, bool> menuControls = new Dictionary<Selectable, bool>();
        bool previousInteractable, previousRaycasts, rebuilding;
        float width, height, unit, gridWidth, cardHeight;
        int pixelWidth, pixelHeight, columns;
        public float TextScale { get; private set; } = 1.15f;
        public AiFamily CurrentPage { get; private set; }
        public static bool IsOpen { get { return !ReferenceEquals(active, null); } }
        public static AiSelectionPanel Active { get { return active; } }
        public static bool BlocksMenuActivation { get { return IsOpen || closedFrame == Time.frameCount; } }
        public IEnumerable<Button> Tiles { get { return new[] { random }.Concat(tiles); } }
        public TMP_InputField Filter { get { return filter; } }
        public RectTransform Window { get { return window; } }
        public ScrollRect GridScroll { get { return scroll; } }
        public string Detail { get { return detail; } }

        public static AiSelectionPanel Open(AiSetupRow row, TextMeshProUGUI font, string playerName, AiChoice selectedChoice, Func<AiChoice, bool> onSelect,
            AiFamily? fixtureFamily = null, AiCatalogEntry[] fixtureEntries = null)
        {
            if (fixtureEntries != null && !StartupFixes.IsSelfTest) throw new InvalidOperationException("Synthetic AI rosters are available only to the isolated UI test.");
            CloseActive(); StrategyTooltip.HideActive();
            var canvas = row.GetComponentInParent<Canvas>().rootCanvas;
            var root = AiWidgets.Rect(canvas.transform, "Brass AI selection panel", Vector2.zero, Vector2.zero);
            root.anchorMin = Vector2.zero; root.anchorMax = Vector2.one; root.offsetMin = root.offsetMax = Vector2.zero;
            var modalGroup = root.gameObject.AddComponent<CanvasGroup>(); modalGroup.ignoreParentGroups = true;
            var panel = active = root.gameObject.AddComponent<AiSelectionPanel>();
            panel.owner = row; panel.template = font; panel.nickname = playerName; panel.selected = selectedChoice; panel.choose = onSelect;
            panel.CurrentPage = selectedChoice.Family;
            foreach (var family in new[] { AiFamily.V4, AiFamily.Gamma }) {
                panel.catalogs[family] = fixtureFamily == family && fixtureEntries != null ? Array.AsReadOnly(fixtureEntries) : AiSelectionCatalog.Entries(family);
                panel.queries[family] = ""; panel.positions[family] = 1;
            }
            root.gameObject.AddComponent<LightUiTheme>();
            var menu = row.GetComponentsInParent<Frameworks.View.Menu.CreateGame.CreateOfflineGameView>(true).FirstOrDefault();
            if (menu != null) foreach (var control in menu.GetComponentsInChildren<Selectable>(true).Where(c => !c.transform.IsChildOf(root))) {
                panel.menuControls[control] = control.interactable; control.interactable = false;
            }
            panel.menuGroup = menu == null ? null : menu.GetComponent<CanvasGroup>();
            if (panel.menuGroup != null) {
                panel.previousInteractable = panel.menuGroup.interactable; panel.previousRaycasts = panel.menuGroup.blocksRaycasts;
                panel.menuGroup.interactable = false; panel.menuGroup.blocksRaycasts = false;
            }
            panel.Build(); panel.FocusSelection(); return panel;
        }
        Rect PixelBounds()
        {
            var canvas = GetComponentInParent<Canvas>().rootCanvas;
            return canvas.renderMode == RenderMode.ScreenSpaceCamera && canvas.worldCamera != null && canvas.worldCamera.targetTexture != null ?
                canvas.worldCamera.pixelRect : new Rect(0, 0, Screen.width, Screen.height);
        }
        Vector2 P(float x, float y) { return new Vector2(x * unit, -y * unit); }
        Vector2 S(float x, float y) { return new Vector2(x * unit, y * unit); }
        float F(float value) { return value * unit * TextScale; }
        TextMeshProUGUI Text(Transform parent, string name, string value, float x, float y, float w, float h, float size, bool wrap = false)
        {
            var label = AiWidgets.Text(parent, template, name, value, P(x, y), S(w, h), F(size));
            label.richText = false; label.enableWordWrapping = wrap; return label;
        }
        Button Button(Transform parent, string name, string value, float x, float y, float w, float h, float size, Action action)
        {
            var button = AiWidgets.Button(parent, template, name, value, P(x, y), S(w, h), F(size));
            button.onClick.AddListener(() => action()); return button;
        }
        void Remember()
        { if (scroll != null) positions[CurrentPage] = scroll.verticalNormalizedPosition; }
        void Build()
        {
            rebuilding = true;
            if (EventSystem.current != null) EventSystem.current.SetSelectedGameObject(null);
            foreach (Transform child in transform) { child.gameObject.SetActive(false); Destroy(child.gameObject); }
            tiles.Clear(); pageButtons.Clear();
            Canvas.ForceUpdateCanvases(); var root = (RectTransform)transform; var pixels = PixelBounds();
            pixelWidth = (int)pixels.width; pixelHeight = (int)pixels.height;
            unit = root.rect.width / Math.Max(1, pixels.width); if (unit <= 0) unit = 1;
            GetComponent<LightUiTheme>().PixelUnit = unit;
            var safe = pixelWidth == Screen.width && pixelHeight == Screen.height ? Screen.safeArea : pixels;
            width = Mathf.Min(1720, safe.width * .96f); height = Mathf.Min(1100, safe.height * .96f);
            var backdrop = AiWidgets.Rect(transform, "Dismiss chooser", Vector2.zero, root.rect.size);
            backdrop.anchorMin = Vector2.zero; backdrop.anchorMax = Vector2.one; backdrop.offsetMin = backdrop.offsetMax = Vector2.zero;
            var shade = backdrop.gameObject.AddComponent<Image>(); shade.color = new Color(0, 0, 0, .50f);
            var dismiss = backdrop.gameObject.AddComponent<Button>(); dismiss.targetGraphic = shade; dismiss.transition = Selectable.Transition.None;
            dismiss.navigation = new Navigation { mode = Navigation.Mode.None }; dismiss.onClick.AddListener(Close);
            window = AiWidgets.Rect(transform, "Style chooser", Vector2.zero, S(width, height));
            window.anchorMin = window.anchorMax = window.pivot = new Vector2(.5f, .5f);
            window.anchoredPosition = (safe.center - pixels.center) * unit;
            window.gameObject.AddComponent<LightUiSurface>().Role = LightSurfaceRole.Window;
            var title = Text(window, "Title", "CHOOSE AI · " + nickname, 22, 10, width - 365, 46, 30);
            title.fontStyle = FontStyles.Bold; title.overflowMode = TextOverflowModes.Ellipsis;
            Text(window, "Context", "Player " + owner.Player.PlayerStartConfiguration.Color + " · Choose a computer opponent for this player.", 22, 55, width - 60, 32, 19).color = LightUi.Muted;
            smaller = Button(window, "Smaller text", "A−", width - 298, 16, 50, 42, 20, () => ChangeTextSize(-1));
            Text(window, "Text size", new[] { "Large", "Larger", "Largest" }[Mathf.RoundToInt((TextScale - 1) / .15f)], width - 244, 16, 100, 42, 18).alignment = TextAlignmentOptions.Center;
            larger = Button(window, "Larger text", "A+", width - 140, 16, 50, 42, 20, () => ChangeTextSize(1));
            smaller.interactable = TextScale > 1.01f; larger.interactable = TextScale < 1.29f;
            smaller.GetComponent<AiControlSurface>().Available = smaller.interactable;
            larger.GetComponent<AiControlSurface>().Available = larger.interactable;
            close = Button(window, "Close chooser", "×", width - 70, 16, 48, 42, 25, Close);
            float sidebar = 208, bodyX = sidebar + 40, bodyWidth = width - bodyX - 24;
            foreach (var family in new[] { AiFamily.V4, AiFamily.Gamma }) {
                var page = family;
                var button = Button(window, "Page " + family, "", 22, 106 + (family == AiFamily.V4 ? 0 : 90), sidebar, 78, 24, () => SelectPage(page));
                button.GetComponentInChildren<TextMeshProUGUI>().gameObject.SetActive(false);
                AddIcon(button.transform, family == AiFamily.V4 ? AiIconKind.V4 : AiIconKind.Gamma, 12, 17, 30);
                Text(button.transform, "Family", family.ToString(), 52, 8, sidebar - 64, 34, 24).fontStyle = FontStyles.Bold;
                Text(button.transform, "Count", family == AiFamily.V4 ? "6 + Random" : "16 + Random", 52, 44, sidebar - 60, 24, 17).color = LightUi.Muted;
                button.GetComponent<AiControlSurface>().Selected = CurrentPage == family; pageButtons[family] = button;
            }
            Text(window, "Page title", CurrentPage.ToString(), bodyX, 100, 155, 38, 27).fontStyle = FontStyles.Bold;
            countLabel = Text(window, "Page count", "", bodyX + 165, 104, bodyWidth - 165, 34, 19); countLabel.color = LightUi.Muted;
            var box = AiWidgets.Rect(window, "Find AI", P(bodyX, 148), S(bodyWidth - 88, 44)); box.gameObject.AddComponent<AiControlSurface>();
            filter = box.gameObject.AddComponent<TMP_InputField>();
            var area = AiWidgets.Rect(box, "Text area", P(10, 3), S(bodyWidth - 110, 38)); area.gameObject.AddComponent<RectMask2D>();
            var inputText = Text(area, "Input", "", 0, 0, bodyWidth - 110, 38, 20);
            var placeholder = Text(area, "Placeholder", "Find AI…", 0, 0, bodyWidth - 110, 38, 20); placeholder.color = LightUi.Muted;
            filter.textViewport = area; filter.textComponent = inputText; filter.placeholder = placeholder; filter.characterLimit = 64;
            filter.SetTextWithoutNotify(queries[CurrentPage]);
            filter.onValueChanged.AddListener(value => { queries[CurrentPage] = value; positions[CurrentPage] = 1; RefreshGrid(); });
            clear = Button(window, "Clear filter", "Clear", bodyX + bodyWidth - 80, 148, 80, 44, 18, () => { filter.text = ""; filter.ActivateInputField(); });
            var randomEntry = RandomEntry(); random = Tile(randomEntry, window, bodyX, 206, bodyWidth, 84, true);
            columns = CurrentPage == AiFamily.Gamma ? 4 : 3;
            cardHeight = (CurrentPage == AiFamily.Gamma ? 146 : 216) * TextScale / 1.15f;
            float footerHeight = 142 + Math.Max(0, TextScale - 1.15f) * 90;
            float gridHeight = Mathf.Max(80, height - 304 - footerHeight - 16);
            viewport = AiWidgets.Rect(window, "Styles viewport", P(bodyX, 304), S(bodyWidth, gridHeight)); viewport.gameObject.AddComponent<RectMask2D>();
            viewport.gameObject.AddComponent<Image>().color = new Color(0, 0, 0, .001f);
            scroll = viewport.gameObject.AddComponent<ScrollRect>(); scroll.viewport = viewport; scroll.horizontal = false; scroll.vertical = true;
            scroll.scrollSensitivity = 38 * unit; scroll.movementType = ScrollRect.MovementType.Clamped;
            gridWidth = bodyWidth - 22;
            content = AiWidgets.Rect(viewport, "Styles", Vector2.zero, S(gridWidth, gridHeight)); scroll.content = content;
            empty = Text(viewport, "No results", "No matching AIs. Clear the search to see this page's roster.", 12, 12, bodyWidth - 32, 80, 21, true);
            var scrollbar = Scrollbar(window, "AI grid scrollbar", bodyX + bodyWidth - 14, 304, 14, gridHeight);
            scroll.verticalScrollbar = scrollbar; scroll.verticalScrollbarVisibility = ScrollRect.ScrollbarVisibility.AutoHide;
            float footerY = height - footerHeight;
            Text(window, "Current AI", "Current: " + LocalMenu.RowLabel(owner.Player) + "    ·    " + DifficultyCatalog.Summary(owner.SelectedDifficulty),
                22, footerY + 6, width - 225, 38, 21, false).overflowMode = TextOverflowModes.Ellipsis;
            footerClose = Button(window, "Close panel", "Close", width - 192, footerY + 6, 170, 44, 21, Close);
            var detailView = AiWidgets.Rect(window, "AI details", P(22, footerY + 54), S(width - 44, footerHeight - 66));
            detailView.gameObject.AddComponent<RectMask2D>(); detailView.gameObject.AddComponent<Image>().color = new Color(0, 0, 0, .001f);
            detailScroll = detailView.gameObject.AddComponent<ScrollRect>(); detailScroll.viewport = detailView; detailScroll.horizontal = false;
            detailScroll.movementType = ScrollRect.MovementType.Clamped; detailScroll.scrollSensitivity = 30 * unit;
            var detailContent = AiWidgets.Rect(detailView, "Description content", Vector2.zero, detailView.sizeDelta); detailScroll.content = detailContent;
            detailText = Text(detailContent, "Full description", "", 0, 0, width - 50, footerHeight - 66, 19, true); detailText.color = LightUi.Muted;
            rebuilding = false; RefreshGrid(); ShowDetails(selected.Family == CurrentPage ? AiSelectionCatalog.Find(selected) ?? randomEntry : randomEntry);
        }
        void AddIcon(Transform parent, AiIconKind kind, float x, float y, float size)
        { var rect = AiWidgets.Rect(parent, "Icon", P(x, y), S(size, size)); var icon = rect.gameObject.AddComponent<AiChoiceIcon>(); icon.Kind = kind; icon.raycastTarget = false; }
        Scrollbar Scrollbar(Transform parent, string name, float x, float y, float w, float h)
        {
            var track = AiWidgets.Rect(parent, name, P(x, y), S(w, h)); track.gameObject.AddComponent<Image>().color = LightUi.Alternate;
            var bar = track.gameObject.AddComponent<Scrollbar>(); bar.direction = UnityEngine.UI.Scrollbar.Direction.BottomToTop;
            var handle = AiWidgets.Rect(track, "Handle", Vector2.zero, track.sizeDelta);
            handle.anchorMin = Vector2.zero; handle.anchorMax = Vector2.one; handle.offsetMin = handle.offsetMax = Vector2.zero;
            var graphic = handle.gameObject.AddComponent<Image>(); graphic.color = LightUi.Accent; bar.targetGraphic = graphic; bar.handleRect = handle;
            bar.navigation = new Navigation { mode = Navigation.Mode.None }; return bar;
        }
        AiCatalogEntry RandomEntry()
        {
            int available = catalogs[CurrentPage].Count(e => e.Available);
            var canonical = AiSelectionCatalog.Random(CurrentPage);
            return new AiCatalogEntry(CurrentPage, "random", "Random", canonical.Description, available > 0, available == 0 ? "Models not installed" : "");
        }
        Button Tile(AiCatalogEntry entry, Transform parent, float x, float y, float w, float h, bool isRandom)
        {
            var button = AiWidgets.Button(parent, template, entry.Choice.EntryId, "", P(x, y), S(w, h), F(24));
            button.GetComponentInChildren<TextMeshProUGUI>().gameObject.SetActive(false); button.interactable = entry.Available;
            var surface = button.GetComponent<AiControlSurface>(); surface.Selected = selected.Equals(entry.Choice); surface.Available = entry.Available;
            float nameX = isRandom ? 58 : 14, nameWidth = w - nameX - (isRandom ? 175 : 28);
            if (isRandom) AddIcon(button.transform, AiIconKind.Random, 14, 16, 30);
            var nameText = Text(button.transform, "Name", entry.Name, nameX, 8, nameWidth, 72 * TextScale / 1.15f, 24, true); nameText.fontStyle = FontStyles.Bold;
            float nameHeight = isRandom ? 34 : Mathf.Min(74 * TextScale / 1.15f, Mathf.Max(34 * TextScale / 1.15f, nameText.GetPreferredValues(entry.Name, nameWidth * unit, 0).y / unit + 4));
            nameText.rectTransform.sizeDelta = S(nameWidth, nameHeight);
            string description = isRandom ? (CurrentPage == AiFamily.V4 ? "Choose from all six V4 styles at game start. Plans adapt." : "Available when trained Gamma models are installed.") : entry.Description;
            float descriptionY = isRandom ? 46 : nameHeight + 13;
            var descriptionText = Text(button.transform, "Description", description, nameX, descriptionY, w - nameX - 14,
                isRandom ? 30 : Math.Max(24, h - descriptionY - 42), 19, true);
            descriptionText.overflowMode = TextOverflowModes.Ellipsis;
            string state = surface.Selected ? "Selected" : entry.Available ? "" : isRandom ? "Unavailable" : entry.AvailabilityReason;
            var stateText = Text(button.transform, "Status", state, isRandom ? w - 171 : 14, isRandom ? 13 : h - 33, isRandom ? 153 : w - 46, 28, 17);
            stateText.color = surface.Selected ? LightUi.Accent : LightUi.Muted;
            if (surface.Selected) {
                var mark = AiWidgets.Rect(button.transform, "Selected mark", P(w - 26, isRandom ? 49 : h - 26), S(14, 14));
                mark.gameObject.AddComponent<AiSelectionMark>().raycastTarget = false;
            }
            var hint = button.gameObject.AddComponent<AiChoiceHint>(); hint.Panel = this; hint.Entry = entry;
            button.onClick.AddListener(() => Commit(entry)); return button;
        }
        void Commit(AiCatalogEntry entry)
        {
            if (!entry.Available) { ShowDetails(entry); return; }
            if (owner == null || !owner.CanEdit) { Close(); return; }
            var callback = choose; var choice = entry.Choice; Close(); callback(choice);
        }
        void RefreshGrid()
        {
            if (rebuilding) return;
            foreach (var tile in tiles) { tile.gameObject.SetActive(false); Destroy(tile.gameObject); } tiles.Clear();
            string query = queries[CurrentPage].Trim(); var entries = catalogs[CurrentPage];
            var shown = entries.Where(e => query.Length == 0 || e.Name.IndexOf(query, StringComparison.OrdinalIgnoreCase) >= 0 ||
                e.Description.IndexOf(query, StringComparison.OrdinalIgnoreCase) >= 0).ToArray();
            countLabel.text = CurrentPage == AiFamily.Gamma ? shown.Length + " of " + entries.Count + " models · " + entries.Count(e => e.Available) + " available" +
                (entries.Any(e => e.Available) ? "" : " · Models not installed") : shown.Length + " of " + entries.Count + " styles · Plans adapt during play";
            float w = (gridWidth - (columns - 1) * 12) / columns;
            for (int i = 0; i < shown.Length; i++) tiles.Add(Tile(shown[i], content, (i % columns) * (w + 12), (i / columns) * (cardHeight + 12), w, cardHeight, false));
            float needed = Mathf.Ceil(shown.Length / (float)columns) * (cardHeight + 12) - 12;
            content.sizeDelta = new Vector2(gridWidth * unit, Mathf.Max(viewport.rect.height, needed * unit));
            empty.gameObject.SetActive(shown.Length == 0); Canvas.ForceUpdateCanvases();
            scroll.verticalScrollbar.size = Mathf.Clamp01(viewport.rect.height / content.sizeDelta.y);
            scroll.verticalScrollbar.gameObject.SetActive(content.sizeDelta.y > viewport.rect.height + .5f);
            scroll.verticalNormalizedPosition = positions[CurrentPage]; UpdateNavigation();
        }
        public void SelectPage(AiFamily family)
        {
            if (family == CurrentPage || !catalogs.ContainsKey(family)) return;
            Remember(); CurrentPage = family; Build(); pageButtons[family].Select();
        }
        public void ChangeTextSize(int direction)
        {
            float next = Mathf.Clamp(TextScale + direction * .15f, 1, 1.3f);
            if (Mathf.Abs(TextScale - next) < .01f) return;
            Remember(); TextScale = next; Build(); (direction < 0 ? smaller : larger).Select();
        }
        List<Selectable> FocusOrder()
        {
            var controls = new List<Selectable> { pageButtons[AiFamily.V4], pageButtons[AiFamily.Gamma], filter, clear, random };
            controls.AddRange(tiles); controls.AddRange(new[] { smaller, larger, close, footerClose });
            return controls.Where(c => c != null && c.IsActive() && c.IsInteractable()).ToList();
        }
        void UpdateNavigation()
        {
            var focus = FocusOrder();
            for (int i = 0; i < focus.Count; i++) focus[i].navigation = new Navigation { mode = Navigation.Mode.Explicit,
                selectOnUp = focus[(i + focus.Count - 1) % focus.Count], selectOnDown = focus[(i + 1) % focus.Count],
                selectOnLeft = focus[(i + focus.Count - 1) % focus.Count], selectOnRight = focus[(i + 1) % focus.Count] };
            for (int i = 0; i < tiles.Count; i++) {
                var tile = tiles[i]; if (!tile.interactable) continue;
                tile.navigation = new Navigation { mode = Navigation.Mode.Explicit,
                    selectOnLeft = i % columns > 0 ? tiles[i - 1] : pageButtons[CurrentPage],
                    selectOnRight = i % columns < columns - 1 && i + 1 < tiles.Count ? tiles[i + 1] : footerClose,
                    selectOnUp = i >= columns ? tiles[i - columns] : random.interactable ? (Selectable)random : filter,
                    selectOnDown = i + columns < tiles.Count ? tiles[i + columns] : footerClose };
            }
        }
        void FocusSelection()
        {
            var match = Tiles.FirstOrDefault(b => b.name == selected.EntryId && b.interactable);
            (match == null ? (Selectable)pageButtons[CurrentPage] : match).Select(); if (match != null) Reveal(match);
        }
        void Reveal(Button button)
        {
            if (!tiles.Contains(button)) return;
            float y = -((RectTransform)button.transform).anchoredPosition.y, offset = content.anchoredPosition.y;
            if (y < offset) offset = y; else if (y + cardHeight * unit > offset + viewport.rect.height) offset = y + cardHeight * unit - viewport.rect.height;
            content.anchoredPosition = new Vector2(0, Mathf.Clamp(offset, 0, Math.Max(0, content.rect.height - viewport.rect.height)));
        }
        internal void ShowDetails(AiCatalogEntry entry)
        {
            if (entry == null || detailText == null) return;
            detail = entry.Name + " · " + (entry.Available ? entry.Description : entry.AvailabilityReason + ". " + entry.Description);
            detailText.text = detail;
            float h = Math.Max(detailScroll.viewport.rect.height, detailText.GetPreferredValues(detail, detailText.rectTransform.rect.width, 0).y + 4 * unit);
            detailText.rectTransform.sizeDelta = new Vector2(detailText.rectTransform.sizeDelta.x, h);
            detailScroll.content.sizeDelta = new Vector2(detailScroll.content.sizeDelta.x, h); detailScroll.verticalNormalizedPosition = 1;
        }
        void LateUpdate()
        {
            if (owner == null || !owner.CanEdit) { Close(); return; }
            var pixels = PixelBounds(); float currentUnit = ((RectTransform)transform).rect.width / Math.Max(1, pixels.width);
            if (pixelWidth != (int)pixels.width || pixelHeight != (int)pixels.height || Mathf.Abs(currentUnit - unit) > .001f) {
                Remember(); Build(); pageButtons[CurrentPage].Select();
            }
            if (menuGroup != null) { menuGroup.interactable = false; menuGroup.blocksRaycasts = false; }
            foreach (var control in menuControls.Keys) if (control != null) control.interactable = false;
            if (Input.GetKeyDown(KeyCode.Escape)) { Close(); return; }
            if (EventSystem.current == null) return;
            var current = EventSystem.current.currentSelectedGameObject;
            if (Input.GetKeyDown(KeyCode.Tab)) {
                var focus = FocusOrder(); int index = focus.FindIndex(c => c.gameObject == current);
                int step = Input.GetKey(KeyCode.LeftShift) || Input.GetKey(KeyCode.RightShift) ? -1 : 1;
                focus[(index + step + focus.Count) % focus.Count].Select();
            }
            var focused = EventSystem.current.currentSelectedGameObject;
            if (focused != null && !focused.transform.IsChildOf(transform)) pageButtons[CurrentPage].Select();
            var button = focused == null ? null : focused.GetComponent<Button>(); if (button != null) Reveal(button);
        }
        public void OnCancel(BaseEventData data) { data.Use(); Close(); }
        public void Close()
        {
            if (!ReferenceEquals(active, this)) return;
            active = null; closedFrame = Time.frameCount; RestoreMenu(); gameObject.SetActive(false); Destroy(gameObject);
            if (owner != null && owner.CanEdit && EventSystem.current != null) owner.StyleButton.Select();
        }
        void RestoreMenu()
        {
            foreach (var pair in menuControls) if (pair.Key != null) pair.Key.interactable = pair.Value;
            menuControls.Clear();
            if (menuGroup != null) { menuGroup.interactable = previousInteractable; menuGroup.blocksRaycasts = previousRaycasts; menuGroup = null; }
        }
        public static bool DismissBack()
        { if (IsOpen) { CloseActive(); return true; } return closedFrame >= 0 && closedFrame == Time.frameCount; }
        public static void CloseActive() { if (!ReferenceEquals(active, null)) active.Close(); }
        public static void CloseFor(AiSetupRow row) { if (!ReferenceEquals(active, null) && ReferenceEquals(active.owner, row)) active.Close(); }
        void OnDestroy() { RestoreMenu(); if (ReferenceEquals(active, this)) active = null; }
    }
    public sealed class AiChoiceHint : MonoBehaviour, IPointerEnterHandler, ISelectHandler
    {
        public AiSelectionPanel Panel;
        public AiCatalogEntry Entry;
        public void OnPointerEnter(PointerEventData data) { if (Panel != null) Panel.ShowDetails(Entry); }
        public void OnSelect(BaseEventData data) { if (Panel != null) Panel.ShowDetails(Entry); }
    }
    public enum AiIconKind { V4, Gamma, Random }
    public sealed class AiChoiceIcon : MaskableGraphic
    {
        public AiIconKind Kind;
        protected override void OnPopulateMesh(VertexHelper mesh)
        {
            mesh.Clear(); var r = GetPixelAdjustedRect(); Color32 ink = LightUi.Accent;
            if (Kind == AiIconKind.Random) {
                LightUi.Paint(mesh, r, LightSurfaceRole.Card, false, false, false, true, LightUiTheme.Unit(this));
                foreach (var point in new[] { new Vector2(.25f, .25f), new Vector2(.75f, .25f), new Vector2(.5f, .5f), new Vector2(.25f, .75f), new Vector2(.75f, .75f) })
                    Dot(mesh, new Vector2(r.xMin + r.width * point.x, r.yMin + r.height * point.y), r.width * .065f, ink);
            } else {
                var points = Kind == AiIconKind.V4 ? new[] { new Vector2(.2f,.25f),new Vector2(.8f,.25f),new Vector2(.5f,.8f) } :
                    new[] {new Vector2(.16f,.25f),new Vector2(.16f,.75f),new Vector2(.5f,.5f),new Vector2(.84f,.25f),new Vector2(.84f,.75f)};
                var p = points.Select(v => new Vector2(r.xMin + r.width * v.x, r.yMin + r.height * v.y)).ToArray();
                if (Kind == AiIconKind.V4) { Line(mesh,p[0],p[1],r.width*.03f,ink);Line(mesh,p[1],p[2],r.width*.03f,ink);Line(mesh,p[2],p[0],r.width*.03f,ink); }
                else foreach (int i in new[] {0,1,3,4}) Line(mesh,p[i],p[2],r.width*.03f,ink);
                foreach (var point in p) Dot(mesh,point,r.width*.075f,ink);
            }
        }
        static void Dot(VertexHelper mesh, Vector2 center, float radius, Color32 color)
        {
            int start = mesh.currentVertCount; mesh.AddVert(center, color, Vector2.zero);
            for (int i = 0; i < 12; i++) { float a = i * Mathf.PI / 6; mesh.AddVert(center + new Vector2(Mathf.Cos(a), Mathf.Sin(a)) * radius, color, Vector2.zero); }
            for (int i = 0; i < 12; i++) mesh.AddTriangle(start,start + 1 + i,start + 1 + (i + 1) % 12);
        }
        static void Line(VertexHelper mesh, Vector2 a, Vector2 b, float size, Color32 color)
        {
            var d = (b-a).normalized;var v = new Vector2(-d.y,d.x)*size;int start=mesh.currentVertCount;
            foreach(var point in new[]{a-v,b-v,b+v,a+v})mesh.AddVert(point,color,Vector2.zero);
            mesh.AddTriangle(start,start+1,start+2);mesh.AddTriangle(start,start+2,start+3);
        }
    }
}
