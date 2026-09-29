using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using DG.Tweening;
using Frameworks.View.Game;
using UnityEngine;
using UnityEngine.UI;

// Isolated native timing evidence: unchanged production methods, minimal in-memory
// hierarchy. The MenuScene does not contain gameplay prefabs. This does not test
// serialized prefab hit testing, screen layout, or a full native gameplay flow.
namespace BrassPatch.Tests.Liquidation
{
    public sealed class ProbeCardView : BaseCardView
    {
        protected override void PrepareReferences() { HolderObject = transform.Find("Holder"); }
    }
    public static class UnityPresentationRegression
    {
        private static readonly MethodInfo Update = typeof(DOTween).Assembly.GetType("DG.Tweening.Core.TweenManager").GetMethod("Update", BindingFlags.Static | BindingFlags.NonPublic);
        private static readonly List<string> messages = new List<string>();
        private static float time;
        private static void Capture(string text, string stack, LogType type) { if (type == LogType.Warning || type == LogType.Error || type == LogType.Exception) messages.Add(type + ": " + text); }
        private static GameObject Node(string name, Transform parent = null)
        {
            var node = new GameObject(name, typeof(RectTransform)); node.transform.SetParent(parent); return node;
        }
        private static ProbeCardView Card(string name, Transform parent)
        {
            var node = Node(name, parent); Node("Holder", node.transform); return node.AddComponent<ProbeCardView>();
        }
        private static void Tick(float seconds)
        {
            for (int i = 0; i < (int)Math.Round(seconds * 30f); i++)
            { time += 1f / 30f; Update.Invoke(null, new object[] { UpdateType.Normal, 1f / 30f, 1f / 30f }); }
        }
        public static int Run()
        {
            DOTween.Init(); Application.logMessageReceived += Capture; var results = new List<object>();
            try
            {
                foreach (string mode in new[] { "normal", "destroyAt0.50AfterCancelAt0.20", "destroyAt2.50AfterCancelAt2.20", "duplicateBeforeTweenUpdate" })
                {
                    DOTween.KillAll(); messages.Clear(); time = 0f;
                    var root = Node("ScoutTimingReview"); var hand = Node("Hand", root.transform); hand.AddComponent<HorizontalLayoutGroup>();
                    var scoutObject = Node("Scout", root.transform);
                    var safe = Node("SafeArea", scoutObject.transform); var popup = Node("Popup", safe.transform); popup.AddComponent<CanvasGroup>();
                    var panel = Node("Panel", popup.transform);
                    var slot0 = Node("Slot1", panel.transform); Node("Placeholder0", slot0.transform); Node("Placeholder1", slot0.transform);
                    var slot1 = Node("Slot2", panel.transform); Node("Placeholder0", slot1.transform); Node("Placeholder1", slot1.transform);
                    var wildIndustry = Node("WildIndustrySlot", panel.transform); Card("WildIndustry", wildIndustry.transform);
                    var wildLocation = Node("WildLocationSlot", panel.transform); Card("WildLocation", wildLocation.transform);
                    var scout = scoutObject.AddComponent<ScoutActionView>();
                    var first = Card("FirstTradeCard", hand.transform); var second = Card("SecondTradeCard", hand.transform);
                    int accepted = 0, completion = 0; float callbackAt = -1;
                    try
                    {
                        Action<BaseCardView> select = card => {
                            int slot = accepted++; scout.AnimateCardToSlot(slot, Tuple.Create((object)card.gameObject, slot), delegate { if (slot == 1) { completion++; callbackAt = time; } });
                            // Match the presenter's hand-input lock after accepting slot 1.
                            if (slot == 1) { first.ignoreMouseOver = true; second.ignoreMouseOver = true; }
                        };
                        first.OnMouseClickedAction = delegate { select(first); }; second.OnMouseClickedAction = delegate { select(second); };
                        first.OnMouseClicked(); bool firstLockedImmediately = first.ignoreMouseOver;
                        if (mode == "duplicateBeforeTweenUpdate") first.OnMouseClicked(); else second.OnMouseClicked();
                        if (mode.StartsWith("destroy"))
                        {
                            float destroyAt = mode.StartsWith("destroyAt0") ? 0.5f : 2.5f; Tick(destroyAt);
                            // Scout is destroyed after a 0.3s hide. Hand refresh also destroys
                            // old cards; its container persists. Target destruction represents
                            // the cancellation boundary without loading the gameplay scene.
                            UnityEngine.Object.DestroyImmediate(scoutObject);
                            foreach (Transform child in hand.transform.Cast<Transform>().ToArray()) UnityEngine.Object.DestroyImmediate(child.gameObject);
                            Tick(4f - destroyAt);
                        }
                        else Tick(4f);
                        results.Add(new { mode, firstLockedImmediately, acceptedNativeClicks = accepted, completionCallbacks = completion, callbackAtSeconds = callbackAt, nativeMessages = messages.ToArray() });
                    }
                    catch (Exception ex) { results.Add(new { mode, harnessError = ex.ToString(), acceptedNativeClicks = accepted, completionCallbacks = completion, callbackAtSeconds = callbackAt, nativeMessages = messages.ToArray() }); }
                    finally { DOTween.KillAll(); UnityEngine.Object.DestroyImmediate(root); }
                }
                string output = Path.Combine(Path.GetDirectoryName(Application.dataPath), "artifacts", "reference-review", "phases-native-results.json");
                File.WriteAllText(output, Newtonsoft.Json.JsonConvert.SerializeObject(new { unity = Application.unityVersion, runtime = Environment.Version.ToString(), scope = "Unchanged production view and card methods; minimal in-memory hierarchy; serialized prefab and UI input hit testing not tested.", cases = results }, Newtonsoft.Json.Formatting.Indented));
                return results.Count;
            }
            finally { Application.logMessageReceived -= Capture; }
        }
    }
}
