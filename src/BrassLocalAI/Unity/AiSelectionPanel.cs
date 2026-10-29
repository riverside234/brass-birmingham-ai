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
        TextMeshProUGUI template, empty, countLabel, detailText, selectionText;
        Func<AiChoice, bool> choose;
        Func<PlayerSelectionDraft, bool> apply;
        readonly Dictionary<AiFamily, AiChoice> drafts = new Dictionary<AiFamily, AiChoice>();
        int ownerVersion;
        bool fixture;
        AiFamily tableFamily;
        bool neuralTab;
        Button v4Table, nnTable, alphaTable;
        AiFamily Family { get { return CurrentPage == PlayerSelectionPage.Gamma ? AiFamily.Gamma : tableFamily; } }
        static string PageLabel(PlayerSelectionPage page) {return page==PlayerSelectionPage.AlphaTest?"Alpha Test":page.ToString();}
        AiChoice selected { get { return drafts[Family]; } }
        string nickname, detail;
        RectTransform window, viewport, content;
        ScrollRect scroll, detailScroll;
        TMP_InputField filter;
        Button random, close, footerClose, clear, smaller, larger;
        readonly List<Button> tiles = new List<Button>();
        readonly Dictionary<PlayerSelectionPage, Button> pageButtons = new Dictionary<PlayerSelectionPage, Button>();
        readonly Dictionary<AiFamily, string> queries = new Dictionary<AiFamily, string>();
        readonly Dictionary<AiFamily, float> positions = new Dictionary<AiFamily, float>();
        readonly Dictionary<AiFamily, IReadOnlyList<AiCatalogEntry>> catalogs = new Dictionary<AiFamily, IReadOnlyList<AiCatalogEntry>>();
        CanvasGroup menuGroup;
        readonly Dictionary<Selectable, bool> menuControls = new Dictionary<Selectable, bool>();
        bool previousInteractable, previousRaycasts, rebuilding;
        float width, height, unit, gridWidth, cardHeight;
        int pixelWidth, pixelHeight, columns;
        public float TextScale { get; private set; } = 1.15f;
        public PlayerSelectionPage CurrentPage { get; private set; }
        public string DraftDifficulty { get; private set; }
        public int DraftSearchK { get; private set; }
        public double DraftTemperature { get; private set; }
        public Button ApplyButton { get; private set; }
        public Button[] DifficultyButtons { get; private set; } = new Button[0];
        public Button[] SearchKButtons { get; private set; } = new Button[0];
        public Button[] TemperatureButtons { get; private set; } = new Button[0];
        public static bool IsOpen { get { return !ReferenceEquals(active, null); } }
        public static AiSelectionPanel Active { get { return active; } }
        public static bool BlocksMenuActivation { get { return IsOpen || closedFrame == Time.frameCount; } }
        public IEnumerable<Button> Tiles { get { return new[] { random }.Concat(tiles).Where(b => b != null); } }
        public TMP_InputField Filter { get { return filter; } }
        public RectTransform Window { get { return window; } }
        public ScrollRect GridScroll { get { return scroll; } }
        public string Detail { get { return detail; } }

        public static AiSelectionPanel Open(AiSetupRow row, TextMeshProUGUI font, string playerName, AiChoice selectedChoice, Func<AiChoice, bool> onSelect,
            AiFamily? fixtureFamily = null, AiCatalogEntry[] fixtureEntries = null)
        {
            if (!StartupFixes.IsSelfTest) throw new InvalidOperationException("Synthetic AI rosters are available only to the isolated UI test.");
            var panel = OpenCore(row, font, playerName, selectedChoice, fixtureFamily, fixtureEntries);
            panel.choose = onSelect; panel.fixture = true; panel.CurrentPage = selectedChoice.Family == AiFamily.V4 ? PlayerSelectionPage.V4 : PlayerSelectionPage.Gamma;
            panel.Build(); panel.FocusSelection(); return panel;
        }
        public static AiSelectionPanel OpenPlayer(AiSetupRow row, TextMeshProUGUI font, Func<PlayerSelectionDraft, bool> onApply)
        {
            if (!row.CanEdit) return null;
            var panel = OpenCore(row, font, row.Player.NickName, row.SelectedChoice, null, null);
            panel.apply = onApply;
            panel.tableFamily=LocalMenu.NativeSupported(row.Player)?LocalMenu.TableFamily(row.Player):AiFamily.V4;
            panel.neuralTab=panel.tableFamily==AiFamily.AlphaTest;
            panel.CurrentPage = row.Player.AiType == BrassApplication.AI.AIType.Human ? PlayerSelectionPage.Human : panel.tableFamily==AiFamily.AlphaTest?PlayerSelectionPage.AlphaTest:PlayerSelectionPage.V4;
            panel.Build(); panel.FocusSelection(); return panel;
        }
        static AiSelectionPanel OpenCore(AiSetupRow row, TextMeshProUGUI font, string playerName, AiChoice selectedChoice, AiFamily? fixtureFamily, AiCatalogEntry[] fixtureEntries)
        {
            CloseActive(); StrategyTooltip.HideActive();
            var canvas = row.GetComponentInParent<Canvas>().rootCanvas;
            var root = AiWidgets.Rect(canvas.transform, "Brass AI selection panel", Vector2.zero, Vector2.zero);
            root.anchorMin = Vector2.zero; root.anchorMax = Vector2.one; root.offsetMin = root.offsetMax = Vector2.zero;
            var modalGroup = root.gameObject.AddComponent<CanvasGroup>(); modalGroup.ignoreParentGroups = true;
            var panel = active = root.gameObject.AddComponent<AiSelectionPanel>();
            panel.owner = row; panel.ownerVersion = row.BindingVersion; panel.template = font; panel.nickname = playerName;
            panel.DraftDifficulty = row.SelectedDifficulty;
            panel.DraftSearchK = LocalMenu.TableSearchK(row.Player);
            panel.DraftTemperature = LocalMenu.TableTemperature(row.Player);
            foreach (var family in new[] { AiFamily.V4, AiFamily.Gamma, AiFamily.AlphaTest }) {
                panel.catalogs[family] = fixtureFamily == family && fixtureEntries != null ? Array.AsReadOnly(fixtureEntries) : AiSelectionCatalog.Entries(family);
                panel.queries[family] = ""; panel.positions[family] = 1;
                panel.drafts[family] = selectedChoice.Family == family ? selectedChoice : new AiChoice(family, family==AiFamily.AlphaTest?"alpha-test":"random");
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
            return panel;
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
            button.onClick.AddListener(() => { if (IsCurrent) action(); }); return button;
        }
        bool IsCurrent { get { return ReferenceEquals(active, this) && owner != null && owner.CanEdit && owner.BindingVersion == ownerVersion; } }
        void Remember()
        { if (scroll != null) positions[Family] = scroll.verticalNormalizedPosition; }
        void Build()
        {
            rebuilding = true;
            if (EventSystem.current != null) EventSystem.current.SetSelectedGameObject(null);
            foreach (Transform child in transform) { child.gameObject.SetActive(false); Destroy(child.gameObject); }
            tiles.Clear(); pageButtons.Clear(); filter = null; random = null; scroll = null; clear = null;
            DifficultyButtons = new Button[0];
            SearchKButtons = new Button[0];
            TemperatureButtons = new Button[0];
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
            var title = Text(window, "Title", "CHOOSE PLAYER · " + nickname, 22, 10, width - 365, 46, 30);
            title.fontStyle = FontStyles.Bold; title.overflowMode = TextOverflowModes.Ellipsis;
            Text(window, "Context", "Player " + owner.Player.PlayerStartConfiguration.Color + " · Apply confirms this seat and the AI family for the whole table.", 22, 55, width - 60, 32, 19).color = LightUi.Muted;
            smaller = Button(window, "Smaller text", "A−", width - 298, 16, 50, 42, 20, () => ChangeTextSize(-1));
            Text(window, "Text size", new[] { "Large", "Larger", "Largest" }[Mathf.RoundToInt((TextScale - 1) / .15f)], width - 244, 16, 100, 42, 18).alignment = TextAlignmentOptions.Center;
            larger = Button(window, "Larger text", "A+", width - 140, 16, 50, 42, 20, () => ChangeTextSize(1));
            smaller.interactable = TextScale > 1.01f; larger.interactable = TextScale < 1.29f;
            smaller.GetComponent<AiControlSurface>().Available = smaller.interactable;
            larger.GetComponent<AiControlSurface>().Available = larger.interactable;
            close = Button(window, "Close chooser", "×", width - 70, 16, 48, 42, 25, Close);
            float sidebar = 208, bodyX = sidebar + 40, bodyWidth = width - bodyX - 24;
            foreach (var page in new[] { PlayerSelectionPage.Human, PlayerSelectionPage.V4, PlayerSelectionPage.Gamma }) {
                var actualPage=page;
                if(!fixture && page==PlayerSelectionPage.V4 && tableFamily==AiFamily.AlphaTest)actualPage=PlayerSelectionPage.AlphaTest;
                if(!fixture && page==PlayerSelectionPage.Gamma)continue;
                var capturedPage=actualPage;
                var button = Button(window, "Page " + actualPage, "", 22, 106 + (int)page * 90, sidebar, 78, 24, () => SelectPage(capturedPage));
                button.GetComponentInChildren<TextMeshProUGUI>().gameObject.SetActive(false);
                AddIcon(button.transform, actualPage == PlayerSelectionPage.Human ? AiIconKind.Human : actualPage == PlayerSelectionPage.V4 ? AiIconKind.V4 : AiIconKind.Gamma, 12, 17, 30);
                Text(button.transform, "Family", !fixture && page!=PlayerSelectionPage.Human?"AI":PageLabel(actualPage), 52, 8, sidebar - 64, 34, 24).fontStyle = FontStyles.Bold;
                Text(button.transform, "Count", actualPage==PlayerSelectionPage.AlphaTest?"Alpha Test":page == PlayerSelectionPage.Human ? "Local player" : page == PlayerSelectionPage.V4 ? "V4 · 6 + Random" : "16 + Random", 52, 44, sidebar - 60, 24, 17).color = LightUi.Muted;
                button.GetComponent<AiControlSurface>().Selected = CurrentPage == actualPage; pageButtons[actualPage] = button;
            }
            if(!fixture) {
                Text(window,"Table family label","AI for whole table",22,392,sidebar,50,20,true).color=LightUi.Muted;
                v4Table=Button(window,"V4 table","V4",22,450,sidebar,46,21,()=>SwitchFamily(AiFamily.V4));
                v4Table.GetComponent<AiControlSurface>().Selected=!neuralTab;
                nnTable=Button(window,"NN table","NN",22,508,sidebar,46,21,()=> {
                    if(LocalMenu.NativeSupported(owner.Player))SwitchFamily(AiFamily.AlphaTest);
                    else SelectPage(PlayerSelectionPage.Gamma);
                });
                nnTable.GetComponent<AiControlSurface>().Selected=neuralTab;
                alphaTable=null;
                if(neuralTab) {
                    var options=AiWidgets.Rect(window,"NN options",P(42,566),S(sidebar-20,104));
                    if(LocalMenu.NativeSupported(owner.Player)) {
                        alphaTable=Button(options,"Alpha Test table","Alpha Test",0,0,sidebar-20,46,21,()=>SwitchFamily(AiFamily.AlphaTest));
                        alphaTable.GetComponent<AiControlSurface>().Selected=tableFamily==AiFamily.AlphaTest && CurrentPage!=PlayerSelectionPage.Gamma;
                    }
                    var gammaTable=Button(options,"Page Gamma","Gamma",0,alphaTable==null?0:58,sidebar-20,46,21,()=>SelectPage(PlayerSelectionPage.Gamma));
                    gammaTable.GetComponent<AiControlSurface>().Selected=CurrentPage==PlayerSelectionPage.Gamma;
                    pageButtons[PlayerSelectionPage.Gamma]=gammaTable;
                }
            }
            Text(window, "Page title", PageLabel(CurrentPage), bodyX, 100, 235, 38, 27).fontStyle = FontStyles.Bold;
            countLabel = Text(window, "Page count", "", bodyX + 245, 104, bodyWidth - 245, 34, 19); countLabel.color = LightUi.Muted;
            float footerHeight = 108 + Math.Max(0, TextScale - 1.15f) * 90;
            if (CurrentPage == PlayerSelectionPage.Human) {
                countLabel.text = "Local hotseat play";
                var human = AiWidgets.Rect(window, "Human card", P(bodyX, 160), S(bodyWidth, 230));
                human.gameObject.AddComponent<AiControlSurface>().Selected = true;
                AddIcon(human, AiIconKind.Human, 26, 32, 54);
                Text(human, "Human heading", "Human", 104, 22, bodyWidth - 130, 60, 32).fontStyle = FontStyles.Bold;
                Text(human, "Human description", "You control this player locally.", 104, 94, bodyWidth - 130, 44, 24);
                Text(human, "Hotseat", "Choose Human for more than one seat to play together on this device.", 104, 148, bodyWidth - 140, 68, 22, true).color = LightUi.Muted;
            } else {
            Text(window, "Difficulty caption", "Difficulty", bodyX, 146, 137, 46, 21);
            DifficultyButtons = new Button[DifficultyCatalog.Keys.Length];
            bool enabled = CurrentPage == PlayerSelectionPage.V4 || CurrentPage==PlayerSelectionPage.AlphaTest || fixture && catalogs[Family].Any(e => e.Available);
            for (int i = 0; i < DifficultyButtons.Length; i++) {
                string key = DifficultyCatalog.Keys[i];
                var button = DifficultyButtons[i] = Button(window, "Difficulty " + key, DifficultyCatalog.Label(key) + " · " + DifficultyCatalog.Milliseconds(key) / 1000 + "s", bodyX + 142 + i * 167, 146, 155, 46, 21, () => {
                    if (CurrentPage == PlayerSelectionPage.Gamma && !fixture) return;
                    DraftDifficulty = key; UpdateDifficulty(); UpdateSelection();
                });
                button.interactable = enabled; button.GetComponent<AiControlSurface>().Available = enabled;
                var mark = AiWidgets.Rect(button.transform, "Selected mark", P(136, 17), S(11, 11)); mark.gameObject.AddComponent<AiSelectionMark>().raycastTarget = false;
                button.GetComponentInChildren<TextMeshProUGUI>().rectTransform.offsetMax = new Vector2(-20 * unit, -2 * unit);
            }
            float helpX = 155 + 167 * DifficultyButtons.Length;
            Text(window, "Difficulty help", CurrentPage == PlayerSelectionPage.Gamma ? "Gamma is not available yet." : "Maximum thinking time", bodyX + helpX, 146, bodyWidth - helpX, 46, 18, true).color = LightUi.Muted;
            float searchOffset = CurrentPage == PlayerSelectionPage.AlphaTest ? 116 : 0;
            if (searchOffset > 0) {
                Text(window, "NN search caption", "NN search", bodyX, 204, 137, 46, 21);
                SearchKButtons = new Button[Native.NativePackage.SearchKChoices.Count];
                for (int i = 0; i < SearchKButtons.Length; i++) {
                    int k = Native.NativePackage.SearchKChoices[i];
                    var button = SearchKButtons[i] = Button(window, "NN search K" + k, "K = " + k,
                        bodyX + 142 + i * 111, 204, 103, 46, 21, () => {
                            DraftSearchK = k; UpdateSearchK(); UpdateSelection();
                        });
                    var mark = AiWidgets.Rect(button.transform, "Selected mark", P(84, 17), S(11, 11));
                    mark.gameObject.AddComponent<AiSelectionMark>().raycastTarget = false;
                    button.GetComponentInChildren<TextMeshProUGUI>().rectTransform.offsetMax = new Vector2(-20 * unit, -2 * unit);
                }
                Text(window, "NN search help", "Search + up to 1s delivery", bodyX + 486, 204, bodyWidth - 486, 46, 18, true).color = LightUi.Muted;
                UpdateSearchK();
                Text(window, "NN temperature caption", "Temperature", bodyX, 262, 137, 46, 20);
                TemperatureButtons = new Button[2];
                for (int i = 0; i < TemperatureButtons.Length; i++) {
                    double temperature = i == 0 ? 1.0 : 1.5;
                    var button = TemperatureButtons[i] = Button(window, "NN temperature " + temperature.ToString("0.0", System.Globalization.CultureInfo.InvariantCulture),
                        temperature.ToString("0.0", System.Globalization.CultureInfo.InvariantCulture), bodyX + 142 + i * 167, 262, 155, 46, 21, () => {
                            DraftTemperature = temperature; UpdateTemperature(); UpdateSelection();
                        });
                    var mark = AiWidgets.Rect(button.transform, "Selected mark", P(136, 17), S(11, 11));
                    mark.gameObject.AddComponent<AiSelectionMark>().raycastTarget = false;
                    button.GetComponentInChildren<TextMeshProUGUI>().rectTransform.offsetMax = new Vector2(-20 * unit, -2 * unit);
                }
                Text(window, "NN temperature help", "Every NN move · all time budgets", bodyX + 486, 262, bodyWidth - 486, 46, 18, true).color = LightUi.Muted;
                UpdateTemperature();
            }
            var box = AiWidgets.Rect(window, "Find AI", P(bodyX, 204 + searchOffset), S(bodyWidth - 88, 44)); box.gameObject.AddComponent<AiControlSurface>();
            filter = box.gameObject.AddComponent<TMP_InputField>();
            var area = AiWidgets.Rect(box, "Text area", P(10, 3), S(bodyWidth - 110, 38)); area.gameObject.AddComponent<RectMask2D>();
            var inputText = Text(area, "Input", "", 0, 0, bodyWidth - 110, 38, 20);
            var placeholder = Text(area, "Placeholder", "Find AI…", 0, 0, bodyWidth - 110, 38, 20); placeholder.color = LightUi.Muted;
            filter.textViewport = area; filter.textComponent = inputText; filter.placeholder = placeholder; filter.characterLimit = 64;
            filter.SetTextWithoutNotify(queries[Family]);
            filter.onValueChanged.AddListener(value => { if (!IsCurrent) return; queries[Family] = value; positions[Family] = 1; RefreshGrid(); });
            clear = Button(window, "Clear filter", "Clear", bodyX + bodyWidth - 80, 204 + searchOffset, 80, 44, 18, () => { filter.text = ""; filter.ActivateInputField(); });
            var randomEntry = RandomEntry(); random = Tile(randomEntry, window, bodyX, 260 + searchOffset, bodyWidth, 68, true);
            if(Family==AiFamily.AlphaTest)random.gameObject.SetActive(false);
            columns = Family == AiFamily.Gamma ? 4 : 3;
            cardHeight = (Family == AiFamily.Gamma ? 146 : 216) * TextScale / 1.15f;
            float gridHeight = Mathf.Max(80, height - 340 - searchOffset - footerHeight - 16);
            viewport = AiWidgets.Rect(window, "Styles viewport", P(bodyX, 340 + searchOffset), S(bodyWidth, gridHeight)); viewport.gameObject.AddComponent<RectMask2D>();
            viewport.gameObject.AddComponent<Image>().color = new Color(0, 0, 0, .001f);
            scroll = viewport.gameObject.AddComponent<ScrollRect>(); scroll.viewport = viewport; scroll.horizontal = false; scroll.vertical = true;
            scroll.scrollSensitivity = 38 * unit; scroll.movementType = ScrollRect.MovementType.Clamped;
            gridWidth = bodyWidth - 22;
            content = AiWidgets.Rect(viewport, "Styles", Vector2.zero, S(gridWidth, gridHeight)); scroll.content = content;
            empty = Text(viewport, "No results", "No matching AIs. Clear the search to see this page's roster.", 12, 12, bodyWidth - 32, 80, 21, true);
            var scrollbar = Scrollbar(window, "AI grid scrollbar", bodyX + bodyWidth - 14, 340 + searchOffset, 14, gridHeight);
            scroll.verticalScrollbar = scrollbar; scroll.verticalScrollbarVisibility = ScrollRect.ScrollbarVisibility.AutoHide;
            }
            float footerY = height - footerHeight;
            selectionText = Text(window, "Current AI", "", 22, footerY + 6, width - 365, 42, 21);
            selectionText.overflowMode = TextOverflowModes.Ellipsis;
            footerClose = Button(window, "Close panel", "Cancel", width - 332, footerY + 6, 145, 44, 21, Close);
            ApplyButton = Button(window, "Apply selection", "Apply", width - 172, footerY + 6, 150, 44, 21, Apply);
            var detailView = AiWidgets.Rect(window, "AI details", P(22, footerY + 54), S(width - 44, footerHeight - 66));
            detailView.gameObject.AddComponent<RectMask2D>(); detailView.gameObject.AddComponent<Image>().color = new Color(0, 0, 0, .001f);
            detailScroll = detailView.gameObject.AddComponent<ScrollRect>(); detailScroll.viewport = detailView; detailScroll.horizontal = false;
            detailScroll.movementType = ScrollRect.MovementType.Clamped; detailScroll.scrollSensitivity = 30 * unit;
            var detailContent = AiWidgets.Rect(detailView, "Description content", Vector2.zero, detailView.sizeDelta); detailScroll.content = detailContent;
            detailText = Text(detailContent, "Full description", "", 0, 0, width - 50, footerHeight - 66, 19, true); detailText.color = LightUi.Muted;
            rebuilding = false;
            if (CurrentPage == PlayerSelectionPage.Human) SetDetails("Human · Local hotseat play. No AI difficulty or thinking allowance applies.");
            else { RefreshGrid(); UpdateDifficulty(); ShowDetails(catalogs[Family].FirstOrDefault(e => e.Choice.Equals(selected)) ?? RandomEntry()); }
            UpdateSelection(); UpdateNavigation();
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
            if(Family==AiFamily.AlphaTest)return catalogs[Family][0];
            int available = catalogs[Family].Count(e => e.Available);
            var canonical = AiSelectionCatalog.Random(Family);
            return new AiCatalogEntry(Family, "random", "Random", canonical.Description, available > 0, available == 0 ? "Models not installed" : "");
        }
        Button Tile(AiCatalogEntry entry, Transform parent, float x, float y, float w, float h, bool isRandom)
        {
            var button = AiWidgets.Button(parent, template, entry.Choice.EntryId, "", P(x, y), S(w, h), F(24));
            button.GetComponentInChildren<TextMeshProUGUI>().gameObject.SetActive(false); button.interactable = entry.Available;
            var surface = button.GetComponent<AiControlSurface>(); surface.Selected = entry.Available && selected.Equals(entry.Choice); surface.Available = entry.Available;
            float nameX = isRandom ? 58 : 14, nameWidth = w - nameX - (isRandom ? 175 : 28);
            if (isRandom) AddIcon(button.transform, AiIconKind.Random, 14, 16, 30);
            var nameText = Text(button.transform, "Name", entry.Name, nameX, 8, nameWidth, 72 * TextScale / 1.15f, 24, true); nameText.fontStyle = FontStyles.Bold;
            float nameHeight = isRandom ? 34 : Mathf.Min(74 * TextScale / 1.15f, Mathf.Max(34 * TextScale / 1.15f, nameText.GetPreferredValues(entry.Name, nameWidth * unit, 0).y / unit + 4));
            nameText.rectTransform.sizeDelta = S(nameWidth, nameHeight);
            string description = isRandom ? (Family == AiFamily.V4 ? "Choose from all six V4 styles at game start. Plans adapt." : "Available when trained Gamma models are installed.") : entry.Description;
            float descriptionY = isRandom ? 36 : nameHeight + 13;
            var descriptionText = Text(button.transform, "Description", description, nameX, descriptionY, w - nameX - 14,
                isRandom ? 30 : Math.Max(24, h - descriptionY - 42), 19, true);
            descriptionText.overflowMode = TextOverflowModes.Ellipsis;
            string state = surface.Selected ? "Selected" : entry.Available ? "" : isRandom ? "Unavailable" : entry.AvailabilityReason;
            var stateText = Text(button.transform, "Status", state, isRandom ? w - 171 : 14, isRandom ? 13 : h - 33, isRandom ? 153 : w - 46, 28, 17);
            stateText.color = surface.Selected ? LightUi.Accent : LightUi.Muted;
            if (surface.Selected) {
                var mark = AiWidgets.Rect(button.transform, "Selected mark", P(w - 26, isRandom ? 44 : h - 26), S(14, 14));
                mark.gameObject.AddComponent<AiSelectionMark>().raycastTarget = false;
            }
            var hint = button.gameObject.AddComponent<AiChoiceHint>(); hint.Panel = this; hint.Entry = entry;
            button.onClick.AddListener(() => Commit(entry)); return button;
        }
        void Commit(AiCatalogEntry entry)
        {
            if (!IsCurrent) return;
            if (!entry.Available) { ShowDetails(entry); return; }
            if (!fixture && !AiSelectionCatalog.CanChoose(entry.Choice)) return;
            drafts[Family] = entry.Choice; Remember(); Build(); FocusSelection();
        }
        void UpdateDifficulty()
        {
            for (int i = 0; i < DifficultyButtons.Length; i++) {
                bool chosen = DifficultyButtons[i].interactable && DraftDifficulty == DifficultyCatalog.Keys[i];
                var surface = DifficultyButtons[i].GetComponent<AiControlSurface>(); surface.Selected = chosen; surface.SetVerticesDirty();
                DifficultyButtons[i].transform.Find("Selected mark").gameObject.SetActive(chosen);
            }
        }
        void UpdateSearchK()
        {
            for (int i = 0; i < SearchKButtons.Length; i++) {
                bool chosen = DraftSearchK == Native.NativePackage.SearchKChoices[i];
                var surface = SearchKButtons[i].GetComponent<AiControlSurface>(); surface.Selected = chosen; surface.SetVerticesDirty();
                SearchKButtons[i].transform.Find("Selected mark").gameObject.SetActive(chosen);
            }
        }
        void UpdateTemperature()
        {
            for (int i = 0; i < TemperatureButtons.Length; i++) {
                bool chosen = DraftTemperature == (i == 0 ? 1.0 : 1.5);
                var surface = TemperatureButtons[i].GetComponent<AiControlSurface>(); surface.Selected = chosen; surface.SetVerticesDirty();
                TemperatureButtons[i].transform.Find("Selected mark").gameObject.SetActive(chosen);
            }
        }
        bool CanApply()
        {
            if (!IsCurrent) return false;
            if (CurrentPage == PlayerSelectionPage.Human) return !fixture;
            return DifficultyCatalog.Contains(DraftDifficulty) && (fixture ?
                (selected.EntryId == "random" ? RandomEntry().Available : catalogs[Family].Any(e => e.Choice.Equals(selected) && e.Available)) :
                (CurrentPage == PlayerSelectionPage.V4 || CurrentPage==PlayerSelectionPage.AlphaTest) && AiSelectionCatalog.CanChoose(selected));
        }
        void UpdateSelection()
        {
            selectionText.text = CurrentPage == PlayerSelectionPage.Human ? "Selection: Human · Local player" : CurrentPage == PlayerSelectionPage.Gamma && !fixture ?
                "Gamma is not available yet. Choose an available table AI to play." :
                "Selection: " + (Family==AiFamily.AlphaTest?"NN":Family.ToString()) + " · " + (selected.EntryId == "random" ? "Random" : catalogs[Family].First(e => e.Choice.Equals(selected)).Name) +
                " · " + DifficultyCatalog.Label(DraftDifficulty) + " · up to " + DifficultyCatalog.Milliseconds(DraftDifficulty) / 1000 + "s" +
                (Family == AiFamily.AlphaTest ? " · K = " + DraftSearchK + " · T = " + DraftTemperature.ToString("0.0", System.Globalization.CultureInfo.InvariantCulture) : "");
            ApplyButton.interactable = CanApply(); var surface = ApplyButton.GetComponent<AiControlSurface>();
            surface.Available = ApplyButton.interactable; surface.Selected = ApplyButton.interactable; surface.SetVerticesDirty();
        }
        void Apply()
        {
            if (!CanApply()) return;
            var draft = new PlayerSelectionDraft(CurrentPage, selected, DraftDifficulty, DraftSearchK, DraftTemperature);
            var onApply = apply; var onChoose = choose;
            // Closing restores row input before its identity/generation checked
            // callback commits the complete draft with one presenter refresh.
            Close();
            if (fixture) onChoose(draft.Choice); else onApply(draft);
        }
        void RefreshGrid()
        {
            if (rebuilding) return;
            foreach (var tile in tiles) { tile.gameObject.SetActive(false); Destroy(tile.gameObject); } tiles.Clear();
            string query = queries[Family].Trim(); var entries = catalogs[Family];
            var shown = entries.Where(e => query.Length == 0 || e.Name.IndexOf(query, StringComparison.OrdinalIgnoreCase) >= 0 ||
                e.Description.IndexOf(query, StringComparison.OrdinalIgnoreCase) >= 0).ToArray();
            countLabel.text = Family==AiFamily.AlphaTest ? "Test model · Experimental" : Family == AiFamily.Gamma ? shown.Length + " of " + entries.Count + " models · " + entries.Count(e => e.Available) + " available" +
                (entries.Any(e => e.Available) ? "" : " · Models not installed") : shown.Length + " of " + entries.Count + " styles · Plans adapt during play";
            float w = (gridWidth - (columns - 1) * 12) / columns;
            for (int i = 0; i < shown.Length; i++) tiles.Add(Tile(shown[i], content, (i % columns) * (w + 12), (i / columns) * (cardHeight + 12), w, cardHeight, false));
            float needed = Mathf.Ceil(shown.Length / (float)columns) * (cardHeight + 12) - 12;
            content.sizeDelta = new Vector2(gridWidth * unit, Mathf.Max(viewport.rect.height, needed * unit));
            empty.gameObject.SetActive(shown.Length == 0); Canvas.ForceUpdateCanvases();
            scroll.verticalScrollbar.size = Mathf.Clamp01(viewport.rect.height / content.sizeDelta.y);
            scroll.verticalScrollbar.gameObject.SetActive(content.sizeDelta.y > viewport.rect.height + .5f);
            scroll.verticalNormalizedPosition = positions[Family]; UpdateNavigation();
        }
        public void SelectPage(AiFamily family)
        { SelectPage(family == AiFamily.V4 ? PlayerSelectionPage.V4 : family==AiFamily.AlphaTest?PlayerSelectionPage.AlphaTest:PlayerSelectionPage.Gamma); }
        public void SwitchFamily(AiFamily family)
        {
            if(!IsCurrent || fixture || (family!=AiFamily.V4 && family!=AiFamily.AlphaTest) || (family==AiFamily.AlphaTest && !LocalMenu.NativeSupported(owner.Player)))return;
            Remember();tableFamily=family;neuralTab=family==AiFamily.AlphaTest;CurrentPage=family==AiFamily.V4?PlayerSelectionPage.V4:PlayerSelectionPage.AlphaTest;Build();FocusSelection();
        }
        public void SelectPage(PlayerSelectionPage page)
        {
            if (!IsCurrent || page == CurrentPage || !Enum.IsDefined(typeof(PlayerSelectionPage), page)) return;
            if(!fixture && page==PlayerSelectionPage.Gamma) {
                Remember();neuralTab=true;CurrentPage=page;Build();pageButtons[page].Select();return;
            }
            if(!pageButtons.ContainsKey(page))return;
            Remember(); CurrentPage = page; Build(); pageButtons[page].Select();
        }
        public void ChangeTextSize(int direction)
        {
            float next = Mathf.Clamp(TextScale + direction * .15f, 1, 1.3f);
            if (Mathf.Abs(TextScale - next) < .01f) return;
            Remember(); TextScale = next; Build(); (direction < 0 ? smaller : larger).Select();
        }
        List<Selectable> FocusOrder()
        {
            var controls = new List<Selectable>(pageButtons.Where(p=>fixture || p.Key!=PlayerSelectionPage.Gamma).OrderBy(p=>(int)p.Key).Select(p=>(Selectable)p.Value));
            controls.Add(v4Table);controls.Add(nnTable);controls.Add(alphaTable);
            if(!fixture && pageButtons.ContainsKey(PlayerSelectionPage.Gamma))controls.Add(pageButtons[PlayerSelectionPage.Gamma]);
            controls.AddRange(DifficultyButtons); controls.AddRange(SearchKButtons); controls.AddRange(TemperatureButtons); controls.AddRange(new Selectable[] { filter, clear, random });
            controls.AddRange(tiles); controls.AddRange(new[] { smaller, larger, close, footerClose, ApplyButton });
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
                    selectOnRight = i % columns < columns - 1 && i + 1 < tiles.Count ? tiles[i + 1] : ApplyButton,
                    selectOnUp = i >= columns ? tiles[i - columns] : random.interactable ? (Selectable)random : filter,
                    selectOnDown = i + columns < tiles.Count ? tiles[i + columns] : ApplyButton };
            }
        }
        void FocusSelection()
        {
            var match = CurrentPage == PlayerSelectionPage.Human ? null : Tiles.FirstOrDefault(b => b.name == selected.EntryId && b.interactable);
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
            SetDetails(entry.Name + " · " + (entry.Available ? entry.Description : entry.AvailabilityReason + ". " + entry.Description));
        }
        void SetDetails(string value)
        {
            detail = value;
            detailText.text = detail;
            float h = Math.Max(detailScroll.viewport.rect.height, detailText.GetPreferredValues(detail, detailText.rectTransform.rect.width, 0).y + 4 * unit);
            detailText.rectTransform.sizeDelta = new Vector2(detailText.rectTransform.sizeDelta.x, h);
            detailScroll.content.sizeDelta = new Vector2(detailScroll.content.sizeDelta.x, h); detailScroll.verticalNormalizedPosition = 1;
        }
        void LateUpdate()
        {
            if (!IsCurrent) { Close(); return; }
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
    public enum AiIconKind { V4, Gamma, Random, Human }
    public sealed class AiChoiceIcon : MaskableGraphic
    {
        public AiIconKind Kind;
        protected override void OnPopulateMesh(VertexHelper mesh)
        {
            mesh.Clear(); var r = GetPixelAdjustedRect(); Color32 ink = LightUi.Accent;
            if (Kind == AiIconKind.Human) {
                Dot(mesh, new Vector2(r.center.x, r.yMin + r.height * .73f), r.width * .16f, ink);
                var left = new Vector2(r.xMin + r.width * .2f, r.yMin + r.height * .19f);
                var right = new Vector2(r.xMin + r.width * .8f, left.y);
                Line(mesh, left, right, r.height * .13f, ink);
                Dot(mesh, new Vector2(r.center.x, r.yMin + r.height * .34f), r.width * .25f, ink);
            } else if (Kind == AiIconKind.Random) {
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
