using System;
using System.Collections.Generic;
using System.Linq;
using Frameworks.Routing;
using Frameworks.Routing.Unity;
using Frameworks.View.Common;
using InterfaceAdapters.Common;
using InterfaceAdapters.Routing;
using TMPro;
using UnityEngine;
using UnityEngine.UI;

namespace BrassPatch.Statistics
{
    public enum StatisticsOpeningContext { LiveCompletedGame, ArchivedHistory }
    // Shared host for both live and archived Statistics. There is still only
    // one StatisticsPanel implementation and one set of plot/ledger queries.
    internal static class StatisticsViewHost
    {
        static readonly KeyEnum[] keys = { KeyEnum.Escape, KeyEnum.Tab, KeyEnum.PageUp, KeyEnum.PageDown, KeyEnum.Home, KeyEnum.End, KeyEnum.UpArrow, KeyEnum.DownArrow };
        public static Transform Parent(Router router)
        { var factory = ReflectionAccess.Get(router, "_viewFactory") as UnityViewFactory; return factory == null ? null : factory.ParentTransform; }
        public static TMP_FontAsset Typeface(Transform parent)
        { var label = parent.GetComponentsInChildren<TextMeshProUGUI>(true).FirstOrDefault(t => t.font != null); return label == null ? TMP_Settings.defaultFontAsset : label.font; }
        public static TMP_FontAsset ArchiveTypeface(Transform parent)
        { return Resources.FindObjectsOfTypeAll<TMP_FontAsset>().FirstOrDefault(f=>f.name=="GARA SDF")??Typeface(parent); }
        public static T Create<T>(Router router, string name) where T : BaseView
        {
            var parent = Parent(router); if (parent == null) throw new InvalidOperationException("Statistics host is unavailable.");
            var root = StatisticsWidgets.Rect(parent, name, Vector2.zero, Vector2.zero); StatisticsWidgets.Stretch(root, Vector2.zero, Vector2.zero);
            var canvas = root.gameObject.AddComponent<Canvas>(); canvas.overrideSorting = true; canvas.sortingOrder = 500;
            root.gameObject.AddComponent<GraphicRaycaster>(); root.gameObject.AddComponent<CanvasGroup>().ignoreParentGroups = true;
            var panel = root.gameObject.AddComponent<T>(); panel.InputRegistration = router;
            ReflectionAccess.Get<Dictionary<Type, IBaseView>>(router, "_viewCache")[typeof(T)] = panel;
            return panel;
        }
        public static void Keys(Router router, BaseView panel, Func<bool> visible)
        {
            var source = ReflectionAccess.Get<IInputSource>(router, "_inputSource");
            foreach (var key in keys) {
                if (source.GetKeyDown(key)) panel.HandleKeyDown(key);
                if (!visible()) return;
                if (source.GetKey(key)) panel.HandleKey(key);
            }
        }
        public static void Dispose(Router router, BaseView panel)
        {
            if (panel == null) return; panel.gameObject.SetActive(false);
            ReflectionAccess.Get<List<IBaseView>>(router, "_viewStack").Remove(panel);
            var cache = ReflectionAccess.Get<Dictionary<Type, IBaseView>>(router, "_viewCache"); IBaseView cached;
            if (cache.TryGetValue(panel.GetType(), out cached) && ReferenceEquals(cached, panel)) cache.Remove(panel.GetType());
            router.UnregisterAllKeys(panel);
            var animations = ReflectionAccess.Get<Dictionary<Type, IAnimation>>(router, "_viewAnimations"); IAnimation animation;
            if (animations.TryGetValue(panel.GetType(), out animation)) { animation.Kill(); animations.Remove(panel.GetType()); }
            UnityEngine.Object.Destroy(panel.gameObject);
        }
    }
}
