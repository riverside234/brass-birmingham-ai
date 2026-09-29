using System;
using System.Collections.Generic;
using System.Linq;
using BoardGameRules.Entities.Board;
using BoardGameRules.Entities.Board.BuildingSlots;
using BoardGameRules.Entities.Board.Regions;
using BrassApplication.UseCases.Game.Actions;
using BrassApplication.UseCases.Game.Actions.ActionInProgress;
using BrassApplication.UseCases.Game.HandOfCards;
using InterfaceAdapters.Game.ActionInProgressView;
using InterfaceAdapters.Game.Board;
using InterfaceAdapters.Game.HandOfCards;
using InterfaceAdapters.Game.PlayerActions;
using InterfaceAdapters.Routing;
using UnityEngine;
using UnityEngine.UI;

namespace BrassPatch.CityLocator
{
    public static class CityLocatorFixes
    {
        public static void AfterCityPreview(HandOfCardsPresenter presenter, RegionName regionClicked,
            List<BuildingSlot> buildingSlotsWhereBuildIsPossible, Board board)
        {
            IRouter router = ReflectionAccess.Get<IRouter>(presenter, "_router");
            BoardView view = CachedBoard(router);
            if (view == null || !view.gameObject.activeInHierarchy) return;
            Clear(view);
            RegionView region = view.GetRegionViewForGivenRegionName(regionClicked) as RegionView;
            Region data = board.GetRegionByName(regionClicked);
            if (region == null || data == null) return;
            CityLocatorLifetime lifetime = view.GetComponent<CityLocatorLifetime>();
            if (lifetime == null) lifetime = view.gameObject.AddComponent<CityLocatorLifetime>();
            foreach (GameObject node in region._buildingSlots)
            {
                if (node == null || !node.activeInHierarchy) continue;
                BuildingSlotView slot = node.GetComponent<BuildingSlotView>();
                if (slot == null) continue;
                BuildingSlot binding = ReflectionAccess.Get<BuildingSlot>(slot, "_buildingSlot");
                // Use the exact original membership test, including legitimate overbuilds.
                if (binding == null || !data.BuildingSlots.Any(s => System.Object.ReferenceEquals(s, binding)) || buildingSlotsWhereBuildIsPossible.Contains(binding)) continue;
                CityLocatorFrame frame = Frame(slot);
                if (frame == null) continue;
                frame.gameObject.SetActive(true);
                lifetime.Track(frame);
            }
        }

        private static CityLocatorFrame Frame(BuildingSlotView slot)
        {
            Transform existing = slot.transform.Find("BrassPatchCityLocator");
            if (existing != null) return existing.GetComponent<CityLocatorFrame>();
            GameObject node = new GameObject("BrassPatchCityLocator", typeof(RectTransform), typeof(CanvasRenderer));
            node.layer = slot.gameObject.layer;
            RectTransform rect = (RectTransform)node.transform;
            rect.SetParent(slot.transform, false);
            // The green sprite includes transparent glow padding and is scaled to 1.4.
            // Its image bounds overlap adjacent spaces. Fit the actual slot border,
            // whose unanimated rect is the visible box, rather than the padded image.
            rect.anchorMin = Vector2.zero; rect.anchorMax = Vector2.one;
            rect.pivot = new Vector2(0.5f, 0.5f); rect.sizeDelta = Vector2.zero;
            rect.anchoredPosition3D = Vector3.zero; rect.localScale = Vector3.one; rect.localRotation = Quaternion.identity;
            node.AddComponent<LayoutElement>().ignoreLayout = true;
            return node.AddComponent<CityLocatorFrame>();
        }

        private static BoardView CachedBoard(IRouter router)
        {
            return router != null && router.IsViewCached<IBoardView>() ? router.GetView<IBoardView>() as BoardView : null;
        }
        private static void ClearRouter(IRouter router) { Clear(CachedBoard(router)); }
        public static void Clear(BoardView view)
        {
            if (view == null) return;
            CityLocatorLifetime state = view.GetComponent<CityLocatorLifetime>();
            if (state != null) state.Clear();
        }
        public static void BeforeBoard(BoardView view, BoardViewModel model) { Clear(view); }
        public static void BeforeMerchants(BoardView view, IDictionary<RegionName, BorderRegionViewModel> model) { Clear(view); }
        public static void BeforeLinks(BoardView view, IDictionary<string, LinkViewModel> model, ILinkSelectedDelegate selector) { Clear(view); }
        public static void ClearRegion(RegionView region)
        {
            // RegionDetailsView clones the region. Discover clone-local graphics rather
            // than relying on the original board's list of visible components.
            foreach (CityLocatorFrame frame in region.GetComponentsInChildren<CityLocatorFrame>(true))
                frame.gameObject.SetActive(false);
        }
        public static void BeforeCardDelegate(HandOfCardsUseCase useCase, ISelectCardDelegate selector)
        {
            if (selector == null) return;
            HandOfCardsPresenter presenter = ReflectionAccess.Get<object>(useCase, "_presenter") as HandOfCardsPresenter;
            if (presenter != null) ClearRouter(ReflectionAccess.Get<IRouter>(presenter, "_router"));
        }
        public static void BeforeActionDelegate(ActionInProgressUseCase useCase, IActionInProgressCancelDelegate selector)
        {
            if (selector == null) return;
            ActionInProgressPresenter presenter = ReflectionAccess.Get<object>(useCase, "_presenter") as ActionInProgressPresenter;
            if (presenter != null) ClearRouter(ReflectionAccess.Get<IRouter>(presenter, "_router"));
        }
        public static void BeforeEndTurn(PlayerActionsPresenter presenter)
        {
            ClearRouter(ReflectionAccess.Get<IRouter>(presenter, "_router"));
        }
    }
}
