using System;
using System.Linq;
using BrassApplication.AI;
using BrassApplication.Game;
using BrassPatch;
using InterfaceAdapters.Common.CreateGame;
using TMPro;
using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.UI;
using BrassPatch.UI;

namespace BrassLegacyAI
{
    public sealed class AiSetupRow : MonoBehaviour
    {
        RectTransform root, nameRect;
        Vector2 namePosition;
        TextMeshProUGUI template, originalType, originalStyle, summary;
        PlayerMetadata player;
        Frameworks.View.Menu.CreateGame.CreateOfflineGameView menu;
        bool editable;
        public Button StyleButton { get; private set; }
        public Button[] DifficultyButtons { get; private set; }
        public string SelectedStyle { get { return LocalMenu.Strategy(player); } }
        public PlayerMetadata Player { get { return player; } }
        public AiChoice SelectedChoice { get { return LocalMenu.Choice(player); } }
        public string SelectedDifficulty { get { return LocalMenu.Difficulty(player); } }
        public bool CanEdit { get { return editable && menu != null && menu.IsViewVisible() && root != null && root.gameObject.activeInHierarchy && LocalMenu.IsEditable(player); } }

        public void Bind(PlayerCreateGameView view, PlayerCreateGameViewModel model, bool known)
        {
            StrategyPicker.CloseFor(this);
            StrategyTooltip.HideActive();
            if (root == null) Initialize(view);
            player = model.PlayerMetadata;
            bool bot = player != null && model.PlayerNumber <= model.CurrentPlayerCount && model.AiType != AIType.Human;
            editable = bot && known && model.Editable && LocalMenu.IsEditable(player);
            root.gameObject.SetActive(bot);
            nameRect.anchoredPosition = namePosition + (bot ? new Vector2(0, 24) : Vector2.zero);
            originalType.gameObject.SetActive(!bot);
            originalStyle.gameObject.SetActive(false);
            if (!bot) return;
            StyleButton.gameObject.SetActive(editable);
            foreach (var button in DifficultyButtons) button.gameObject.SetActive(editable);
            summary.gameObject.SetActive(!editable);
            root.Find("Style caption").gameObject.SetActive(editable);
            root.Find("Difficulty caption").gameObject.SetActive(editable);
            if (!editable) {
                summary.text = known ? LocalMenu.RowLabel(player) + "\n<size=19>" + DifficultyCatalog.Summary(SelectedDifficulty) + "</size>" : "Original AI";
                return;
            }
            StyleButton.GetComponentInChildren<TextMeshProUGUI>().text = LocalMenu.RowLabel(player);
            StyleButton.GetComponent<StrategyTooltip>().Description = LocalMenu.ChoiceDescription(SelectedStyle);
            for (int i = 0; i < DifficultyButtons.Length; i++) {
                bool selected = SelectedDifficulty == DifficultyCatalog.Keys[i];
                var surface = DifficultyButtons[i].GetComponent<AiControlSurface>();
                surface.Selected = selected; surface.SetVerticesDirty();
                DifficultyButtons[i].transform.Find("Selected mark").gameObject.SetActive(selected);
            }
            var nav = StyleButton.navigation; nav.mode = Navigation.Mode.Explicit;
            nav.selectOnDown = DifficultyButtons[Math.Max(0, Array.IndexOf(DifficultyCatalog.Keys, SelectedDifficulty))]; StyleButton.navigation = nav;
            for (int i = 0; i < DifficultyButtons.Length; i++) {
                nav = DifficultyButtons[i].navigation; nav.mode = Navigation.Mode.Explicit; nav.selectOnUp = StyleButton;
                nav.selectOnLeft = i > 0 ? DifficultyButtons[i - 1] : null; nav.selectOnRight = i < 2 ? DifficultyButtons[i + 1] : null;
                DifficultyButtons[i].navigation = nav;
            }
        }
        void Initialize(PlayerCreateGameView view)
        {
            menu = view.GetComponentsInParent<Frameworks.View.Menu.CreateGame.CreateOfflineGameView>(true).FirstOrDefault();
            originalStyle = ReflectionAccess.Get<TextMeshProUGUI>(view, "_playerAIDifficultyText"); template = originalStyle;
            originalType = ReflectionAccess.Get<TextMeshProUGUI>(view, "_playerAITypeText");
            nameRect = ReflectionAccess.Get<TMP_InputField>(view, "_nicknameText").GetComponent<RectTransform>(); namePosition = nameRect.anchoredPosition;
            var reference = originalStyle.rectTransform;
            root = AiWidgets.Rect(reference.parent, "Brass AI setup", Vector2.zero, new Vector2(330, 110));
            root.gameObject.AddComponent<LightUiTheme>();
            root.anchorMin = reference.anchorMin; root.anchorMax = reference.anchorMax; root.pivot = new Vector2(.5f, .5f);
            root.anchoredPosition = reference.anchoredPosition + new Vector2(18, 32);
            AiWidgets.Text(root, template, "Style caption", "AI", new Vector2(0, 0), new Vector2(74, 52), 18).color = AiWidgets.Ivory;
            AiWidgets.Text(root, template, "Difficulty caption", "Difficulty", new Vector2(0, -66), new Vector2(76, 26), 16).color = AiWidgets.Ivory;
            StyleButton = AiWidgets.Button(root, template, "Choose style", "", new Vector2(80, 0), new Vector2(250, 52), 20);
            StyleButton.GetComponentInChildren<TextMeshProUGUI>().rectTransform.offsetMax = new Vector2(-27, -2);
            StyleButton.GetComponentInChildren<TextMeshProUGUI>().enableWordWrapping = true;
            StyleButton.GetComponentInChildren<TextMeshProUGUI>().lineSpacing = -4;
            AiWidgets.Text(StyleButton.transform, template, "Chevron", "v", new Vector2(226, -11), new Vector2(18, 30), 18);
            StyleButton.onClick.AddListener(OpenPicker); StyleButton.gameObject.AddComponent<StrategyTooltip>();
            DifficultyButtons = new Button[3];
            for (int i = 0; i < 3; i++) {
                string key = DifficultyCatalog.Keys[i];
                var button = DifficultyButtons[i] = AiWidgets.Button(root, template, key, DifficultyCatalog.Label(key) + "\n<size=16>" + DifficultyCatalog.Milliseconds(key) / 1000 + "s</size>", new Vector2(80 + i * 85, -58), new Vector2(80, 48), 21);
                button.onClick.AddListener(() => { if (CanEdit) LocalMenu.EditDifficulty(player, key); });
                button.gameObject.AddComponent<StrategyTooltip>().Description = DifficultyCatalog.Description(key);
                var mark = AiWidgets.Rect(button.transform, "Selected mark", new Vector2(65, -4), new Vector2(10, 10)); mark.gameObject.AddComponent<AiSelectionMark>().raycastTarget = false;
            }
            summary = AiWidgets.Text(root, template, "Saved AI", "", new Vector2(0, -8), new Vector2(330, 64), 24);
            summary.color = AiWidgets.Ivory;
            summary.alignment = TextAlignmentOptions.Center;
        }
        public void OpenPicker()
        {
            if (!CanEdit) return;
            var captured = player;
            AiSelectionPanel.Open(this, template, player.NickName, SelectedChoice,
                choice => CanEdit && ReferenceEquals(captured, player) && LocalMenu.EditChoice(captured, choice));
        }
        void Update()
        {
            if (!CanEdit || StrategyPicker.IsOpen || EventSystem.current == null || !Input.GetKeyDown(KeyCode.Tab)) return;
            var selected = EventSystem.current.currentSelectedGameObject;
            if (selected != StyleButton.gameObject && !DifficultyButtons.Any(b => b.gameObject == selected)) return;
            var controls = transform.parent.GetComponentsInChildren<AiSetupRow>().Where(r => r.CanEdit)
                .SelectMany(r => new[] { r.StyleButton }.Concat(r.DifficultyButtons)).ToArray();
            int index = Array.FindIndex(controls, b => b.gameObject == selected);
            int step = Input.GetKey(KeyCode.LeftShift) || Input.GetKey(KeyCode.RightShift) ? -1 : 1;
            controls[(index + step + controls.Length) % controls.Length].Select();
        }
        void OnDisable() { StrategyPicker.CloseFor(this); }
        void OnDestroy() { StrategyPicker.CloseFor(this); }
    }

    internal static class AiWidgets
    {
        public static readonly Color Ivory = new Color(.96f, .91f, .80f, 1);
        public static RectTransform Rect(Transform parent, string name, Vector2 position, Vector2 size)
        {
            var go = new GameObject(name, typeof(RectTransform)); var r = go.GetComponent<RectTransform>(); r.SetParent(parent, false);
            r.anchorMin = r.anchorMax = new Vector2(0, 1); r.pivot = new Vector2(0, 1); r.anchoredPosition = position; r.sizeDelta = size; return r;
        }
        public static TextMeshProUGUI Text(Transform parent, TextMeshProUGUI template, string name, string text, Vector2 position, Vector2 size, float fontSize)
        {
            var r = Rect(parent, name, position, size); var t = r.gameObject.AddComponent<TextMeshProUGUI>();
            t.font = template.font; t.fontSharedMaterial = template.fontSharedMaterial; t.fontStyle = FontStyles.Normal;
            t.fontSize = fontSize; t.text = text; t.color = LightUiTheme.Contains(parent) ? LightUi.Ink : Ivory; t.alignment = TextAlignmentOptions.MidlineLeft; t.raycastTarget = false;
            t.enableWordWrapping = false;
            // TMP's first enable can reset the RectTransform to its default size.
            r.anchorMin = r.anchorMax = new Vector2(0, 1); r.pivot = new Vector2(0, 1); r.sizeDelta = size; r.anchoredPosition = position;
            return t;
        }
        public static Button Button(Transform parent, TextMeshProUGUI template, string name, string text, Vector2 position, Vector2 size, float fontSize)
        {
            var r = Rect(parent, name, position, size); var surface = r.gameObject.AddComponent<AiControlSurface>();
            var b = r.gameObject.AddComponent<Button>(); b.targetGraphic = surface;
            var colors = b.colors; colors.normalColor = Color.white; colors.highlightedColor = new Color(1.32f, 1.26f, 1.15f, 1);
            colors.selectedColor = colors.highlightedColor; colors.pressedColor = new Color(.8f, .8f, .8f, 1); colors.fadeDuration = .08f; b.colors = colors;
            if (LightUiTheme.Contains(parent)) {
                colors.normalColor = colors.highlightedColor = colors.selectedColor = colors.disabledColor = Color.white;
                colors.pressedColor = new Color(.94f, .92f, .88f, 1); colors.fadeDuration = 0; b.colors = colors;
            }
            var label = Text(r, template, "Label", text, Vector2.zero, size, fontSize); label.alignment = TextAlignmentOptions.Center;
            label.rectTransform.anchorMin = Vector2.zero; label.rectTransform.anchorMax = Vector2.one; label.rectTransform.offsetMin = new Vector2(5, 2); label.rectTransform.offsetMax = new Vector2(-5, -2);
            return b;
        }
    }
    // Code-native rounded geometry, with no texture downloads or prefab modifications.
    public sealed class AiControlSurface : MaskableGraphic, ISelectHandler, IDeselectHandler, IPointerEnterHandler, IPointerExitHandler
    {
        public bool Selected;
        public bool Panel;
        public bool Available = true;
        bool focused, hovered;
        public void OnSelect(BaseEventData data) { focused = true; SetVerticesDirty(); }
        public void OnDeselect(BaseEventData data) { focused = false; SetVerticesDirty(); }
        public void OnPointerEnter(PointerEventData data) { hovered = true; SetVerticesDirty(); }
        public void OnPointerExit(PointerEventData data) { hovered = false; SetVerticesDirty(); }
        protected override void OnPopulateMesh(VertexHelper mesh)
        {
            if (LightUiTheme.Contains(this)) {
                LightUi.Paint(mesh, GetPixelAdjustedRect(), Panel ? LightSurfaceRole.Window : LightSurfaceRole.Card,
                    Selected, focused, hovered, Available, LightUiTheme.Unit(this)); return;
            }
            mesh.Clear(); var r = GetPixelAdjustedRect(); const int steps = 6; int n = 4 * (steps + 1);
            float radius = Panel ? 10 : 5, border = Selected ? 2.2f : 1;
            Color32 edge = Selected ? new Color(.88f, .72f, .42f, 1) : new Color(.48f, .39f, .27f, 1);
            Color32 fill = Selected ? new Color(.32f, .255f, .16f, .99f) : new Color(.16f, .13f, .105f, Panel ? .99f : .96f);
            mesh.AddVert(new Vector3(r.center.x, r.center.y), fill, Vector2.zero);
            for (int ring = 0; ring < 2; ring++) {
                float inset = ring == 0 ? 0 : border, rad = Mathf.Max(1, radius - inset);
                for (int corner = 0; corner < 4; corner++) for (int j = 0; j <= steps; j++) {
                    float angle = (corner * 90 + j * 90f / steps) * Mathf.Deg2Rad;
                    float cx = corner == 0 || corner == 3 ? r.xMax - radius : r.xMin + radius;
                    float cy = corner < 2 ? r.yMax - radius : r.yMin + radius;
                    mesh.AddVert(new Vector3(cx + Mathf.Cos(angle) * rad, cy + Mathf.Sin(angle) * rad), ring == 0 ? edge : fill, Vector2.zero);
                }
            }
            for (int i = 0; i < n; i++) {
                int next = (i + 1) % n;
                mesh.AddTriangle(0, 1 + n + i, 1 + n + next);
                mesh.AddTriangle(1 + i, 1 + next, 1 + n + i); mesh.AddTriangle(1 + next, 1 + n + next, 1 + n + i);
            }
        }
    }
    public sealed class AiSelectionMark : MaskableGraphic
    {
        protected override void OnPopulateMesh(VertexHelper mesh)
        {
            mesh.Clear(); var r = GetPixelAdjustedRect();
            var ink = LightUiTheme.Contains(this) ? LightUi.Accent : AiWidgets.Ivory;
            Segment(mesh, new Vector2(r.xMin, r.center.y), new Vector2(r.xMin + r.width * .36f, r.yMin + 1), ink);
            Segment(mesh, new Vector2(r.xMin + r.width * .36f, r.yMin + 1), new Vector2(r.xMax, r.yMax - 1), ink);
        }
        static void Segment(VertexHelper mesh, Vector2 a, Vector2 b, Color32 ink)
        {
            var d = (b - a).normalized; var v = new Vector2(-d.y, d.x); int start = mesh.currentVertCount;
            foreach (var p in new[] { a - v, a + v, b + v, b - v }) mesh.AddVert(p, ink, Vector2.zero);
            mesh.AddTriangle(start, start + 1, start + 2); mesh.AddTriangle(start, start + 2, start + 3);
        }
    }
}
