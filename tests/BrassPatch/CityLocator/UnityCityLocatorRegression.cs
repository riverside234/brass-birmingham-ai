using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using BoardGameRules.Entities.Actions;
using BoardGameRules.Entities.Board.BuildingSlots;
using BoardGameRules.Entities.Board.Regions;
using BoardGameRules.Entities.Cards;
using BoardGameRules.Entities.Common;
using BoardGameRules.Entities.Game;
using BoardGameRules.Entities.GamePhases;
using BoardGameRules.Entities.Players;
using BrassApplication.AI;
using BrassApplication.ApplicationSettings;
using BrassApplication.Game;
using BrassApplication.GameHistory;
using BrassApplication.UseCases.Common;
using BrassApplication.UseCases.Game.Actions;
using BrassApplication.UseCases.Game.Actions.ActionInProgress;
using BrassApplication.UseCases.Game.Board;
using BrassApplication.UseCases.Game.HandOfCards;
using BrassPatch.CityLocator;
using DG.Tweening;
using Frameworks.Routing.Unity;
using Frameworks.Utils;
using InterfaceAdapters.Common;
using InterfaceAdapters.Game.ActionInProgressView;
using InterfaceAdapters.Game.Board;
using InterfaceAdapters.Game.HandOfCards;
using InterfaceAdapters.Game.PlayerActions;
using InterfaceAdapters.Routing;
using InterfaceAdapters.Utils;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.SceneManagement;
using UnityEngine.UI;

namespace BrassPatch.Tests.CityLocator
{
    public static class UnityCityLocatorRegression
    {
        internal static int Checks;
        internal static BoardView Prefab;
        internal static readonly List<string> Problems = new List<string>();
        internal static void Check(bool ok, string message) { Checks++; if (!ok) throw new Exception("City locator: " + message); }
        internal static object Call(object owner, string method, params object[] args) { return owner.GetType().GetMethod(method, ReflectionAccess.Flags).Invoke(owner, args); }
        public static void Begin(string path, string priorResult)
        {
            var driver = new GameObject("City locator regression driver").AddComponent<CityLocatorTestDriver>();
            UnityEngine.Object.DontDestroyOnLoad(driver.gameObject);
            driver.Output = Environment.GetCommandLineArgs().Contains("--brass-city-locator-render") ? Path.Combine(Path.GetDirectoryName(path), "city-render-result.json") : path;
            driver.Result = JObject.Parse(priorResult);
        }
        internal sealed class Localization : ILocalization { public string GetTranslation(string term) { return term.Split('/').Last(); } }
        internal sealed class Settings : IGameSettings
        {
            public bool Night;
            public bool CameFromMenu() { return false; }
            public void SetMap(bool night) { Night = night; }
            public bool GetIsNightMap() { return Night; }
            public void SetDefaultGameSettings(GameMetadata m, int n) { }
            public int GetNumberOfPlayers() { return 2; }
            public bool IsSavedDefaultGameSettings() { return false; }
            public PlayerColor GetPlayerIndexSavedColor(int i) { return (PlayerColor)i; }
            public PlayerAvatar GetPlayerIndexSavedAvatar(int i) { return (PlayerAvatar)i; }
            public AIType GetPlayerIndexSavedAIType(int i) { return AIType.Human; }
        }
        internal sealed class Router : IRouter, IUseCaseBuilder
        {
            public BoardView Board; public BoardUseCase BoardCase; public bool Cached = true; public int Reads;
            public IUseCaseBuilder UseCaseBuilder { get; set; }
            public Router() { UseCaseBuilder = this; }
            public T GetView<T>() where T : class, IBaseView { Reads++; return Board as T; }
            public bool IsViewCached<T>() where T : class, IBaseView { return Cached && typeof(T) == typeof(IBoardView); }
            public bool IsViewVisible<T>() where T : class, IBaseView { return IsViewCached<T>(); }
            public IAnimation ShowView<T>() where T : class, IBaseView { return null; }
            public IAnimation HideView<T>() where T : class, IBaseView { return null; }
            public void MoveOnTop<T>() where T : class, IBaseView { }
            public void ProcessInput() { }
            public void RegisterKey(KeyEnum k, IBaseView v) { }
            public void UnregisterKey(KeyEnum k, IBaseView v) { }
            public void UnregisterAllKeys(IBaseView v) { }
            public void BuildUseCases() { }
            public void RestoreUseCases() { }
            public void RestoreUseCaseOfType(Type t) { }
            public IController GetController(Type t) { return null; }
            public T GetUseCase<T>() where T : IUseCase
            {
                if (typeof(T) == typeof(BoardUseCase)) return (T)(object)BoardCase;
                throw new EndHandoff();
            }
        }
        internal sealed class EndHandoff : Exception { }
        internal sealed class Selection : ISelectCardDelegate, IActionInProgressCancelDelegate
        { public int Selected; public void SelectCard(BaseCard c) { Selected++; } public void CancelAction() { } }
        internal sealed class Presenter : BoardPresenter
        {
            public int Clicks;
            public Presenter(IRouter r, ILocalization l) : base(r, l) { }
            public BoardViewModel Model(BrassApplicationGame app, Action<RegionName> click) { return PrepareBoardViewModel(app.Game.Board, app.Game.GameProgressInformation.CurrentGameEra, click); }
            public override void ShowRegionDetailsView(RegionName name, BrassApplicationGame app, IDictionary<Tuple<BuildingSlot, BuildingType>, IList<CantUseBuildActionReason>> reasons) { Clicks++; }
        }
        internal sealed class Fixture : IDisposable
        {
            public GameObject Root; public Router Router = new Router(); public Localization Localization = new Localization(); public Settings Settings = new Settings();
            public BoardView View; public Presenter Presenter; public HandOfCardsPresenter Hand; public HandOfCardsUseCase HandCase; public BrassApplicationGame App; public BoardViewModel Model;
            private IRouter oldRouter;
            public Fixture(int players)
            {
                var config = new GameStartConfiguration((NumberOfPlayers)players) { Seed = 270926 };
                var metadata = new GameMetadata(config, GameType.Local);
                foreach (var p in config.Players) metadata.JoinGame(new PlayerMetadata { PlayerStartConfiguration = p, NickName = "UI test", PlayerNetworkId = "ui-" + p.Color, AiType = AIType.Human });
                App = new BrassApplicationGame(new BoardGameRulesGame(config), metadata); App.Game.StartGame();
                var history = new BrassGameHistory(); history.SetGame(App);
                oldRouter = Utils.Global.Instance.Router; Utils.Global.Instance.Router = Router;
                Root = new GameObject("City locator test canvas", typeof(RectTransform), typeof(Canvas), typeof(GraphicRaycaster));
                Root.GetComponent<Canvas>().renderMode = RenderMode.ScreenSpaceOverlay;
                Presenter = new Presenter(Router, Localization); Router.BoardCase = new BoardUseCase(Presenter, history, Settings); Router.BoardCase.SetAllowBuidlingFromBoard(false);
                View = UnityEngine.Object.Instantiate(Prefab, Root.transform); View.Controller = new BoardController(Router.BoardCase); Router.Board = View; View.gameObject.SetActive(true);
                Hand = new HandOfCardsPresenter(Router, Localization); HandCase = new HandOfCardsUseCase(Hand, history);
                Update(); View.CreateShowViewImmediatelyAnimation(); Tick(0.1f); Canvas.ForceUpdateCanvases();
            }
            public void Update() { Model = Presenter.Model(App, Router.BoardCase.RegionSelected); View.UpdateBoardView(Model); }
            public RegionView Region(Region region) { return (RegionView)View.GetRegionViewForGivenRegionName(region.Name); }
            public void Preview(Region region, List<BuildingSlot> green) { Hand.OnLocationCardClicked(region.Name, green, App.Game.Board); }
            public CityLocatorFrame[] Visible() { return View.GetComponentsInChildren<CityLocatorFrame>(true).Where(x => x.gameObject.activeInHierarchy).ToArray(); }
            public void Dispose() { Tick(2f); if (Root != null) UnityEngine.Object.DestroyImmediate(Root); Utils.Global.Instance.Router = oldRouter; }
        }
        internal static void Tick(float time)
        {
            var update = typeof(DOTween).Assembly.GetType("DG.Tweening.Core.TweenManager").GetMethod("Update", BindingFlags.Static | BindingFlags.NonPublic);
            for (int i = 0; i < Math.Ceiling(time * 30); i++) update.Invoke(null, new object[] { UpdateType.Normal, 1f / 30, 1f / 30 });
        }
        private static string State(Fixture f)
        {
            var g = f.App.Game;
            return JsonConvert.SerializeObject(new { g.Revision, events = g.GetUncommittedEvents().Count, actions = g.TurnFlow.GetRemainingActionsCount(),
                players = g.PlayersTrack.Players.Select(p => new { p.Color, p.Money, p.IncomeMarkerPosition, p.VP, p.LinkTileCount, hand = p.Hand.Select(c => c.UniqueId).ToArray() }).ToArray(),
                slots = g.Board.Regions.SelectMany(r => r.BuildingSlots).Select(s => new { s.UniqueId, industry = s.BuildedIndustry == null ? null : s.BuildedIndustry.UniqueId, resource = s.BuildedIndustry == null ? 0 : s.BuildedIndustry.GetResource() }).ToArray() });
        }
        private static void Verify(Fixture f, Region region, List<BuildingSlot> green)
        {
            f.Preview(region, green); int expected = 0;
            foreach (var r in f.App.Game.Board.Regions.Where(r => r.BuildingSlots.Count > 0))
            foreach (GameObject node in f.Region(r)._buildingSlots)
            {
                var slot = node.GetComponent<BuildingSlotView>(); var binding = ReflectionAccess.Get<BuildingSlot>(slot, "_buildingSlot");
                if (binding == null) { Check(!node.activeInHierarchy, "unbound visible slot " + r.Name + "/" + node.name); continue; }
                bool eligible = r.Name == region.Name && green.Contains(binding); bool white = r.Name == region.Name && !eligible && node.activeInHierarchy;
                var frame = node.GetComponentInChildren<CityLocatorFrame>(true);
                Check((frame != null && frame.gameObject.activeInHierarchy) == white, "white membership " + r.Name + "/" + binding.UniqueId);
                Check(node.transform.Find("Outline").GetComponent<Image>().color.a == (eligible ? 1f : 0f), "original green membership retained");
                if (white) { expected++; Check(frame.color == CityLocatorFrame.LocatorColor && !frame.raycastTarget, "ivory color, translucent and non-interactive frame"); }
            }
            Check(f.Visible().Length == expected, "exact visible frame count");
        }
        internal static int Run()
        {
            Checks = 0;
            for (int players = 2; players <= 4; players++)
            using (var f = new Fixture(players))
            {
                var cities = f.App.Game.Board.Regions.Where(r => r.BuildingSlots.Count > 0).ToArray();
                foreach (GameEra era in new[] { GameEra.CanalEra, GameEra.RailEra })
                {
                    f.App.Game.GameProgressInformation.CurrentGameEra = era; f.Update(); string before = State(f);
                    foreach (var city in cities) { Verify(f, city, new List<BuildingSlot>()); Verify(f, city, city.BuildingSlots.Take(1).ToList()); Verify(f, city, city.BuildingSlots.ToList()); }
                    Check(State(f) == before, "every city preview leaves game state unchanged in " + players + "/" + era);
                }
                var target = cities.First(r => r.BuildingSlots.Count >= 3); var slots = f.Region(target)._buildingSlots; var occupied = target.BuildingSlots[0];
                occupied.BuildedIndustry = f.App.Game.GameProgressInformation.CurrentPlayer.Mat.GetFirstBuildingOfType(occupied.BuildingTypes[0]);
                f.Update(); Verify(f, target, new List<BuildingSlot>()); Verify(f, target, new List<BuildingSlot> { occupied });
                int total = f.View.GetComponentsInChildren<CityLocatorFrame>(true).Length;
                for (int i = 0; i < 25; i++) f.Preview(target, new List<BuildingSlot>());
                Check(f.View.GetComponentsInChildren<CityLocatorFrame>(true).Length == total, "rapid repeated previews reuse frames");
                var image = slots[0].transform.Find("Outline").GetComponent<Image>(); var frame = slots[0].GetComponentInChildren<CityLocatorFrame>(true); Vector3 frameScale = frame.transform.localScale; Vector3 originalScale = image.transform.localScale;
                using (var helper = new VertexHelper())
                {
                    frame.GetType().GetMethod("OnPopulateMesh", ReflectionAccess.Flags, null, new[] { typeof(VertexHelper) }, null).Invoke(frame, new object[] { helper });
                    Check(helper.currentVertCount == 221 && helper.currentIndexCount == 660, "rounded locator has joined halo, border, and wash");
                    var vertices = new List<UIVertex>(); helper.GetUIVertexStream(vertices);
                    Rect bounds = frame.GetPixelAdjustedRect(); var middle = bounds.center;
                    Check(vertices.Any(v => Vector2.Distance(v.position, middle) < .01f && v.color.a == ((Color32)new Color(1, 1, 1, CityLocatorFrame.FillOpacity)).a), "subtle center wash remains transparent enough for occupied tiles");
                    Check(vertices.Any(v => v.color.a == 255) && vertices.Any(v => v.color.a == 0) && vertices.Any(v => v.color.a == ((Color32)new Color(1, 1, 1, CityLocatorFrame.GlowOpacity)).a), "visible opaque border with feathered halo");
                    Check(vertices.Any(v => v.position.x > bounds.xMax - bounds.width * .07f && v.position.x < bounds.xMax && v.position.y > bounds.yMax - bounds.height * .07f && v.position.y < bounds.yMax), "curved corner has intermediate arc vertices");
                }
                image.transform.localScale *= 2f; image.transform.localRotation = Quaternion.Euler(0, 0, 30); f.Preview(target, new List<BuildingSlot>());
                Check(frame.transform.localScale == frameScale && frame.transform.localRotation == Quaternion.identity, "locator uses resting geometry after outline animation");
                image.transform.localScale = originalScale; image.transform.localRotation = Quaternion.identity;
                var clone = UnityEngine.Object.Instantiate(f.Region(target), f.Root.transform); clone.TurnOffOutlines();
                Check(!clone.GetComponentsInChildren<CityLocatorFrame>(true).Any(x => x.gameObject.activeSelf), "details clone clears its own locator frames");
                Check(f.Visible().Length > 0, "clearing clone does not clear original board"); UnityEngine.Object.DestroyImmediate(clone.gameObject);
                int clicks = f.Presenter.Clicks; slots[0].GetComponent<BuildingSlotView>().OnPointerClick(null); Check(f.Presenter.Clicks == clicks + 1, "white frame retains normal region click routing");
                f.View.HighlightBuildingSlots(f.Model); Check(f.Visible().Length == 0, "build/sell/resource highlight clears white");
                f.Preview(target, new List<BuildingSlot>()); f.View.HighlightLinks(f.Model.Links, null); Check(f.Visible().Length == 0, "network highlight clears white");
                f.Preview(target, new List<BuildingSlot>()); f.View.HighlightMerchantSlots(f.Model.BorderRegionsViewModels); Check(f.Visible().Length == 0, "merchant highlight clears white");
                f.Preview(target, new List<BuildingSlot>()); f.Update(); Check(f.Visible().Length == 0, "refresh/undo/replay clears white");
                var selection = new Selection(); f.Preview(target, new List<BuildingSlot>()); f.HandCase.SetOnCardClickActionDelegate(selection); Check(f.Visible().Length == 0, "card action ownership clears white");
                f.HandCase.OnCardSelected(new LocationCard("selected", (NumberOfPlayers)players, target.Name, target.Color)); Check(selection.Selected == 1 && f.Visible().Length == 0, "action-owned city card routes only to selection");
                f.HandCase.SetOnCardClickActionDelegate(null); Check(f.Visible().Length == 0, "cancelled selection does not restore stale preview");
                f.HandCase.OnCardSelected(new WildLocationCard("wild", (NumberOfPlayers)players)); f.HandCase.OnCardSelected(new IndustryCard("industry", (NumberOfPlayers)players, new List<BuildingType> { BuildingType.CoalMine })); Check(f.Visible().Length == 0, "non-location cards do not show city locator");
                f.HandCase.OnCardSelected(new LocationCard("preview", (NumberOfPlayers)players, target.Name, target.Color)); var legal = BuildAction.GetBuildingSlotsWhereBuildingIsPossibleStatic(f.App.Game.Board, f.App.Game.GameProgressInformation, t => true).ToList(); Check(f.Visible().Length == target.BuildingSlots.Count(s => !legal.Contains(s)), "actual hand use case uses unchanged eligibility list");
                f.Preview(target, new List<BuildingSlot>()); new ActionInProgressUseCase(new ActionInProgressPresenter(f.Router, f.Localization)).SetCancelActionDelegate(selection); Check(f.Visible().Length == 0, "action-in-progress presentation clears white before selection");
                f.Preview(target, new List<BuildingSlot>()); try { new PlayerActionsPresenter(f.Router, f.Localization).EndHumanTurn(); } catch (EndHandoff) { } Check(f.Visible().Length == 0, "end-turn prefix clears before use-case handoff");
                f.Preview(target, new List<BuildingSlot>()); f.View.gameObject.SetActive(false); f.View.gameObject.SetActive(true); Check(f.Visible().Length == 0, "disabled/reopened board cannot retain city locator");
                f.Router.Cached = false; int reads = f.Router.Reads; f.HandCase.SetOnCardClickActionDelegate(selection); Check(f.Router.Reads == reads, "cleanup does not instantiate an uncached board"); f.Router.Cached = true;
            }
            return Checks;
        }
    }
    public sealed class CityLocatorTestDriver : MonoBehaviour
    {
        public string Output; public JObject Result; private float started; private bool done;
        private void Capture(string message, string stack, LogType type) { if (type == LogType.Exception || type == LogType.Error) UnityCityLocatorRegression.Problems.Add(message + " " + stack); }
        private IEnumerator Start()
        {
            started = Time.realtimeSinceStartup; yield return SceneManager.LoadSceneAsync("GameScene"); yield return null;
            Application.logMessageReceived += Capture;
            try
            {
                UnityCityLocatorRegression.Prefab = Resources.FindObjectsOfTypeAll<UnityViewFactory>().SelectMany(f => f.ViewList.Views).OfType<BoardView>().First(v => v.GetType() == typeof(BoardView));
                var image = UnityCityLocatorRegression.Prefab.GetComponentInChildren<BuildingSlotView>(true).transform.Find("Outline").GetComponent<Image>();
                Result["originalOutline"] = image.sprite.name + " / " + image.material.name + " / " + image.rectTransform.rect + " / " + image.transform.localScale;
                if (Environment.GetCommandLineArgs().Contains("--brass-city-locator-render")) { StartCoroutine(RenderCases()); }
                else
                {
                    Result["cityLocatorChecks"] = UnityCityLocatorRegression.Run(); Result["nativeErrors"] = JArray.FromObject(UnityCityLocatorRegression.Problems);
                    UnityCityLocatorRegression.Check(UnityCityLocatorRegression.Problems.Count == 0, "no native errors in prefab tests"); Finish(null);
                }
            }
            catch (Exception error) { Finish(error); }
        }
        private void Update() { if (!done && Time.realtimeSinceStartup - started > 180f) Finish(new Exception("City locator test watchdog")); }
        private IEnumerator RenderCases()
        {
            UnityCityLocatorRegression.Fixture f = null;
            try { f = new UnityCityLocatorRegression.Fixture(2); } catch (Exception e) { Finish(e); yield break; }
            // Let the actual Canvas/Graphic update once before camera rendering and hit tests.
            yield return null;
            try
            {
                var files = new List<string>();
                var city = f.App.Game.Board.Regions.First(r => r.Name.ToString() == "Birmingham");
                var slot = city.BuildingSlots.Last();
                slot.BuildedIndustry = f.App.Game.GameProgressInformation.CurrentPlayer.Mat.GetFirstBuildingOfType(slot.BuildingTypes[0]);
                f.Update();
                var cameraNode = new GameObject("City test render camera", typeof(Camera));
                Camera camera = cameraNode.GetComponent<Camera>(); camera.enabled = false; camera.orthographic = true;
                camera.backgroundColor = new Color(0.08f, 0.08f, 0.08f); camera.clearFlags = CameraClearFlags.SolidColor;
                camera.transform.position = new Vector3(0, 0, -100); camera.farClipPlane = 1000;
                RenderTexture target = new RenderTexture(1600, 900, 24, RenderTextureFormat.ARGB32); target.Create(); camera.targetTexture = target;
                var canvas = f.Root.GetComponent<Canvas>(); canvas.renderMode = RenderMode.ScreenSpaceCamera; canvas.worldCamera = camera; canvas.planeDistance = 100;
                var zoom = ReflectionAccess.Get<ZoomScrollRect>(f.View, "_zoomScrollRect");
                Result["zoomLimits"] = zoom.MinContentScaleFactor + " / " + zoom.MaxContentScaleFactor;
                foreach (bool night in new[] { false, true })
                foreach (float scale in new[] { zoom.MinContentScaleFactor, (zoom.MinContentScaleFactor + zoom.MaxContentScaleFactor) / 2, zoom.MaxContentScaleFactor })
                {
                    f.Settings.Night = night; UnityCityLocatorRegression.Call(f.View, "SetMapDayNight");
                    zoom.content.localScale = new Vector3(scale, scale, 1);
                    var green = city.BuildingSlots.Take(1).ToList();
                    f.Preview(city, green); UnityCityLocatorRegression.Tick(0.5f); Canvas.ForceUpdateCanvases();
                    // Center this rendered comparison on the selected city's actual transform.
                    zoom.content.position += canvas.transform.position - f.Region(city).transform.position;
                    string prefix = (night ? "night" : "day") + "-" + scale.ToString("0.000", System.Globalization.CultureInfo.InvariantCulture);
                    CityLocatorFixes.Clear(f.View); SaveFrame(camera, target, prefix + "-baseline.png", files);
                    // Apply just the same post-preview helper to preserve the exact camera comparison.
                    CityLocatorFixes.AfterCityPreview(f.Hand, city.Name, green, f.App.Game.Board);
                    SaveFrame(camera, target, prefix + "-mixed.png", files);
                    CheckRaycasts(f, camera, city);
                    f.Preview(city, new List<BuildingSlot>()); UnityCityLocatorRegression.Tick(0.5f);
                    zoom.content.position += canvas.transform.position - f.Region(city).transform.position;
                    SaveFrame(camera, target, prefix + "-white.png", files);
                }
                Result["renderedFiles"] = JArray.FromObject(files);
                Result["renderChecks"] = UnityCityLocatorRegression.Checks;
                camera.targetTexture = null; target.Release(); UnityEngine.Object.DestroyImmediate(target); UnityEngine.Object.DestroyImmediate(cameraNode);
                UnityCityLocatorRegression.Check(UnityCityLocatorRegression.Problems.Count == 0, "no render errors");
                f.Dispose(); f = null; Finish(null);
            }
            catch (Exception e) { if (f != null) f.Dispose(); Finish(e); }
        }
        private void SaveFrame(Camera camera, RenderTexture target, string name, List<string> files)
        {
            Canvas.ForceUpdateCanvases(); camera.Render();
            var old = RenderTexture.active; RenderTexture.active = target;
            var texture = new Texture2D(target.width, target.height, TextureFormat.RGB24, false);
            texture.ReadPixels(new Rect(0, 0, target.width, target.height), 0, 0); texture.Apply(); RenderTexture.active = old;
            string path = Path.Combine(Path.GetDirectoryName(Output), name); File.WriteAllBytes(path, texture.EncodeToPNG()); files.Add(path);
            UnityEngine.Object.DestroyImmediate(texture);
        }
        private void CheckRaycasts(UnityCityLocatorRegression.Fixture f, Camera camera, Region city)
        {
            var frame = f.Visible().First();
            var eventSystem = EventSystem.current;
            if (eventSystem == null) eventSystem = new GameObject("City test events", typeof(EventSystem)).GetComponent<EventSystem>();
            var pointer = new PointerEventData(eventSystem) { position = RectTransformUtility.WorldToScreenPoint(camera, frame.transform.position) };
            var before = new List<RaycastResult>(); var after = new List<RaycastResult>();
            foreach (var raycaster in f.Root.GetComponentsInChildren<GraphicRaycaster>()) raycaster.Raycast(pointer, before);
            CityLocatorFixes.Clear(f.View);
            foreach (var raycaster in f.Root.GetComponentsInChildren<GraphicRaycaster>()) raycaster.Raycast(pointer, after);
            UnityCityLocatorRegression.Check(before.Count > 0, "native raycast reaches map under white frame");
            UnityCityLocatorRegression.Check(before.Select(r => r.gameObject).SequenceEqual(after.Select(r => r.gameObject)), "white frames do not intercept or reorder raycasts");
            CityLocatorFixes.AfterCityPreview(f.Hand, city.Name, city.BuildingSlots.Take(1).ToList(), f.App.Game.Board);
        }
        private void Finish(Exception error)
        {
            done = true; Application.logMessageReceived -= Capture; Result["passed"] = error == null;
            Result["nativeErrors"] = JArray.FromObject(UnityCityLocatorRegression.Problems);
            if (error != null) Result["error"] = error.ToString(); File.WriteAllText(Output, Result.ToString()); Application.Quit(error == null ? 0 : 1);
        }
    }
}
