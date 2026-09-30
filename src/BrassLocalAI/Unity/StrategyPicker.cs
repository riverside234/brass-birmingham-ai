using System;
using System.Collections.Generic;
using System.Linq;
using TMPro;
using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.UI;

namespace BrassLegacyAI
{
    public sealed class AiStyleOption
    {
        public readonly string Id, Name, Description;
        public AiStyleOption(string id, string name, string description) { Id = id; Name = name; Description = description; }
    }
    public sealed class StrategyPicker : MonoBehaviour, ICancelHandler
    {
        static StrategyPicker active;
        static int closedFrame = -1;
        AiSetupRow owner;
        TextMeshProUGUI template, empty;
        AiStyleOption[] options;
        Func<string, bool> choose;
        string selected;
        RectTransform panel, content, viewport;
        ScrollRect scroll;
        readonly List<Button> tiles = new List<Button>();
        Button random, close, clear;
        TMP_InputField filter;
        float width, tileWidth;
        public static bool IsOpen { get { return !ReferenceEquals(active, null); } }
        public static StrategyPicker Active { get { return active; } }
        public IEnumerable<Button> Tiles { get { return new[] { random }.Concat(tiles); } }
        public TMP_InputField Filter { get { return filter; } }
        public static StrategyPicker Open(AiSetupRow row, TextMeshProUGUI font, string nickname, string selectedId, AiStyleOption[] catalog, Func<string, bool> onSelect)
        {
            CloseActive(); StrategyTooltip.HideActive();
            var canvas = row.GetComponentInParent<Canvas>().rootCanvas;
            var root = AiWidgets.Rect(canvas.transform, "Brass AI style picker", Vector2.zero, Vector2.zero);
            root.pivot = new Vector2(.5f, .5f);
            root.anchorMin = Vector2.zero; root.anchorMax = Vector2.one; root.offsetMin = root.offsetMax = Vector2.zero;
            var picker = active = root.gameObject.AddComponent<StrategyPicker>(); picker.owner = row; picker.template = font;
            picker.options = catalog; picker.selected = selectedId; picker.choose = onSelect;
            picker.Build(nickname); return picker;
        }
        void Build(string nickname)
        {
            Canvas.ForceUpdateCanvases(); var bounds = ((RectTransform)transform).rect;
            var backdrop = AiWidgets.Rect(transform, "Dismiss chooser", Vector2.zero, Vector2.zero);
            backdrop.anchorMin = Vector2.zero; backdrop.anchorMax = Vector2.one; backdrop.offsetMin = backdrop.offsetMax = Vector2.zero;
            var shade = backdrop.gameObject.AddComponent<Image>(); shade.color = new Color(0, 0, 0, .38f);
            var dismiss = backdrop.gameObject.AddComponent<Button>(); dismiss.targetGraphic = shade; dismiss.transition = Selectable.Transition.None;
            dismiss.navigation = new Navigation { mode = Navigation.Mode.None }; dismiss.onClick.AddListener(Close);
            width = Mathf.Min(740, bounds.width - 32); tileWidth = (width - 48) / 2;
            bool searchable = options.Length > 8;
            float listHeight = Mathf.Min(Mathf.Ceil(options.Length / 2f) * 96 - 8, Mathf.Max(176, bounds.height - (searchable ? 284 : 234)));
            float height = 190 + listHeight + (searchable ? 46 : 0);
            panel = AiWidgets.Rect(transform, "Style chooser", Vector2.zero, new Vector2(width, height)); panel.pivot = new Vector2(.5f, .5f); panel.anchorMin = panel.anchorMax = new Vector2(.5f, .5f);
            var point = transform.InverseTransformPoint(owner.StyleButton.transform.position);
            panel.anchoredPosition = new Vector2(Mathf.Clamp(point.x, bounds.xMin + width / 2 + 16, bounds.xMax - width / 2 - 16), Mathf.Clamp(point.y, bounds.yMin + height / 2 + 16, bounds.yMax - height / 2 - 16));
            panel.gameObject.AddComponent<AiControlSurface>().Panel = true;
            var title = AiWidgets.Text(panel, template, "Title", "Choose AI style — " + nickname, new Vector2(20, -13), new Vector2(width - 90, 38), 25);
            title.overflowMode = TextOverflowModes.Ellipsis;
            close = AiWidgets.Button(panel, template, "Close chooser", "×", new Vector2(width - 54, -14), new Vector2(34, 32), 23); close.onClick.AddListener(Close);
            random = Tile(new AiStyleOption("random", "Random", "Chooses one of " + options.Length + " starting styles. Plans adapt, with balanced fallback when needed."), panel, new Vector2(20, -62), new Vector2(width - 40, 68));
            float listTop = 142;
            if (searchable) {
                var box = AiWidgets.Rect(panel, "Find style", new Vector2(20, -140), new Vector2(width - 88, 36)); box.gameObject.AddComponent<AiControlSurface>();
                filter = box.gameObject.AddComponent<TMP_InputField>();
                var textRect = AiWidgets.Rect(box, "Text area", new Vector2(9, -2), new Vector2(width - 106, 32)); textRect.gameObject.AddComponent<RectMask2D>();
                var text = AiWidgets.Text(textRect, template, "Input", "", Vector2.zero, textRect.sizeDelta, 20);
                var placeholder = AiWidgets.Text(textRect, template, "Placeholder", "Find style", Vector2.zero, textRect.sizeDelta, 20); placeholder.color = new Color(.72f, .67f, .57f, 1);
                filter.textViewport = textRect; filter.textComponent = text; filter.placeholder = placeholder; filter.characterLimit = 64;
                filter.onValueChanged.AddListener(value => RefreshTiles());
                clear = AiWidgets.Button(panel, template, "Clear filter", "×", new Vector2(width - 60, -140), new Vector2(40, 36), 23);
                clear.onClick.AddListener(() => { filter.text = ""; filter.ActivateInputField(); }); listTop += 46;
            }
            viewport = AiWidgets.Rect(panel, "Styles viewport", new Vector2(20, -listTop), new Vector2(width - 40, listHeight)); viewport.gameObject.AddComponent<RectMask2D>();
            var hit = viewport.gameObject.AddComponent<Image>(); hit.color = new Color(0, 0, 0, .005f);
            scroll = viewport.gameObject.AddComponent<ScrollRect>(); scroll.viewport = viewport; scroll.horizontal = false; scroll.vertical = true; scroll.scrollSensitivity = 28; scroll.movementType = ScrollRect.MovementType.Clamped;
            content = AiWidgets.Rect(viewport, "Styles", Vector2.zero, new Vector2(width - 40, listHeight)); scroll.content = content;
            empty = AiWidgets.Text(viewport, template, "No results", "No matching styles. Clear the filter to see all styles.", new Vector2(8, -12), new Vector2(width - 64, 70), 21); empty.enableWordWrapping = true;
            AiWidgets.Text(panel, template, "Footer", searchable ? "v3 AI  ·  Scroll or find a style  ·  Esc to close" : "v3 AI  ·  Plans adapt throughout the game  ·  Esc to close", new Vector2(20, -(height - 33)), new Vector2(width - 40, 24), 16);
            RefreshTiles();
            var first = Tiles.FirstOrDefault(b => b.name == selected) ?? random; first.Select(); Reveal(first);
        }
        Button Tile(AiStyleOption option, Transform parent, Vector2 position, Vector2 size)
        {
            var button = AiWidgets.Button(parent, template, option.Id, "", position, size, 22);
            var old = button.GetComponentInChildren<TextMeshProUGUI>(); old.gameObject.SetActive(false);
            AiWidgets.Text(button.transform, template, "Name", option.Name, new Vector2(12, -7), new Vector2(size.x - 44, 27), 23);
            var description = AiWidgets.Text(button.transform, template, "Description", option.Description, new Vector2(12, -35), new Vector2(size.x - 24, size.y - 39), 17);
            description.enableWordWrapping = true; description.overflowMode = TextOverflowModes.Ellipsis;
            button.gameObject.AddComponent<StrategyTooltip>().Description = option.Description;
            bool chosen = option.Id == selected; button.GetComponent<AiControlSurface>().Selected = chosen;
            if (chosen) AiWidgets.Rect(button.transform, "Selected mark", new Vector2(size.x - 28, -13), new Vector2(16, 16)).gameObject.AddComponent<AiSelectionMark>().raycastTarget = false;
            button.onClick.AddListener(() => {
                if (owner == null || !owner.CanEdit) { Close(); return; }
                var callback = choose; Close(); callback(option.Id);
            });
            return button;
        }
        void RefreshTiles()
        {
            foreach (var tile in tiles) { tile.gameObject.SetActive(false); Destroy(tile.gameObject); } tiles.Clear();
            string query = filter == null ? "" : filter.text.Trim();
            var shown = options.Where(o => query.Length == 0 || o.Name.IndexOf(query, StringComparison.OrdinalIgnoreCase) >= 0 || o.Description.IndexOf(query, StringComparison.OrdinalIgnoreCase) >= 0).ToArray();
            for (int i = 0; i < shown.Length; i++) tiles.Add(Tile(shown[i], content, new Vector2((i % 2) * (tileWidth + 8), -(i / 2) * 96), new Vector2(tileWidth, 88)));
            content.sizeDelta = new Vector2(width - 40, Math.Max(viewport.rect.height, Mathf.Ceil(shown.Length / 2f) * 96 - 8));
            scroll.verticalNormalizedPosition = 1; empty.gameObject.SetActive(shown.Length == 0);
            var nav = random.navigation; nav.mode = Navigation.Mode.Explicit; nav.selectOnUp = filter == null ? (Selectable)close : filter; nav.selectOnDown = tiles.FirstOrDefault(); random.navigation = nav;
            close.navigation = new Navigation { mode = Navigation.Mode.Explicit, selectOnDown = random, selectOnLeft = filter == null ? (Selectable)random : filter };
            if (filter != null) {
                filter.navigation = new Navigation { mode = Navigation.Mode.Explicit, selectOnUp = close, selectOnDown = random, selectOnRight = clear };
                clear.navigation = new Navigation { mode = Navigation.Mode.Explicit, selectOnUp = close, selectOnDown = random, selectOnLeft = filter };
            }
            for (int i = 0; i < tiles.Count; i++) {
                nav = tiles[i].navigation; nav.mode = Navigation.Mode.Explicit;
                nav.selectOnLeft = i % 2 == 1 ? tiles[i - 1] : null; nav.selectOnRight = i % 2 == 0 && i + 1 < tiles.Count ? tiles[i + 1] : null;
                nav.selectOnUp = i < 2 ? random : tiles[i - 2]; nav.selectOnDown = i + 2 < tiles.Count ? tiles[i + 2] : null; tiles[i].navigation = nav;
            }
        }
        void Reveal(Button button)
        {
            if (!tiles.Contains(button)) return;
            float y = -((RectTransform)button.transform).anchoredPosition.y;
            float offset = content.anchoredPosition.y, visible = viewport.rect.height;
            if (y < offset) offset = y; else if (y + 88 > offset + visible) offset = y + 88 - visible;
            content.anchoredPosition = new Vector2(0, Mathf.Clamp(offset, 0, Mathf.Max(0, content.rect.height - visible)));
        }
        void Update()
        {
            if (owner == null || !owner.CanEdit) { Close(); return; }
            if (Input.GetKeyDown(KeyCode.Escape)) { Close(); return; }
            if (EventSystem.current == null) return;
            var current = EventSystem.current.currentSelectedGameObject;
            if (Input.GetKeyDown(KeyCode.Tab)) {
                var focus = new List<Selectable> { close, random }; if (filter != null) { focus.Add(filter); focus.Add(clear); } focus.AddRange(tiles);
                int index = focus.FindIndex(s => s.gameObject == current), step = Input.GetKey(KeyCode.LeftShift) || Input.GetKey(KeyCode.RightShift) ? -1 : 1;
                focus[(index + step + focus.Count) % focus.Count].Select();
            }
            var button = current == null ? null : current.GetComponent<Button>(); if (button != null) Reveal(button);
        }
        public void OnCancel(BaseEventData data) { data.Use(); Close(); }
        public void Close()
        {
            if (!ReferenceEquals(active, this)) return;
            active = null; closedFrame = Time.frameCount; gameObject.SetActive(false); Destroy(gameObject);
            if (owner != null && owner.CanEdit && EventSystem.current != null) owner.StyleButton.Select();
        }
        public static bool DismissBack()
        {
            if (IsOpen) { CloseActive(); return true; }
            return closedFrame >= 0 && closedFrame == Time.frameCount;
        }
        public static void CloseActive() { if (!ReferenceEquals(active, null)) active.Close(); StrategyTooltip.HideActive(); }
        public static void CloseFor(AiSetupRow row) { if (!ReferenceEquals(active, null) && ReferenceEquals(active.owner, row)) active.Close(); }
        void OnDestroy() { if (ReferenceEquals(active, this)) active = null; }
    }
}
