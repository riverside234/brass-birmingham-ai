using System;
using InterfaceAdapters.Game.PlayAiTurn;
using InterfaceAdapters.Routing;
using TMPro;
using UnityEngine;
using UnityEngine.UI;

namespace BrassLegacyAI
{
    public sealed class AiTurnControls : MonoBehaviour
    {
        TextMeshProUGUI status;
        string strategy;
        public static AiTurnControls Attach(IRouter router, string strategy)
        {
            var view = router.GetView<IPlayAiTurnView>() as Component;
            if (view == null) return null; // Headless fixtures use a view interface only.
            var panel = view.transform.Find("SafeArea/Popup/ActionPanel");
            if (panel == null) throw new InvalidOperationException("AI turn panel is missing.");
            var old = panel.Find("BrassStrategyControls"); if (old != null) Destroy(old.gameObject);
            var template = panel.GetComponentInChildren<TextMeshProUGUI>(true);
            var root = new GameObject("BrassStrategyControls", typeof(RectTransform), typeof(CanvasGroup));
            var rect = root.GetComponent<RectTransform>(); rect.SetParent(panel, false);
            // The shipped panel sits at the bottom of the screen: extend upward.
            rect.anchorMin = rect.anchorMax = new Vector2(0.5f, 1f); rect.pivot = new Vector2(0.5f, 0f);
            rect.anchoredPosition = new Vector2(0, 8); rect.sizeDelta = new Vector2(440, 34);
            var controls = root.AddComponent<AiTurnControls>(); controls.strategy = strategy;
            controls.status = Text(root.transform, template, new Vector2(0, -15), new Vector2(440, 28), strategy + " · Thinking…");
            return controls;
        }
        static TextMeshProUGUI Text(Transform parent, TextMeshProUGUI template, Vector2 position, Vector2 size, string text)
        {
            var go = new GameObject("Strategy text", typeof(RectTransform)); go.transform.SetParent(parent, false);
            var label = go.AddComponent<TextMeshProUGUI>(); label.font = template.font; label.fontSharedMaterial = template.fontSharedMaterial;
            label.text = text; label.fontSize = 20; label.alignment = TextAlignmentOptions.Center; label.color = new Color(0.95f, 0.90f, 0.78f, 1);
            label.raycastTarget = false; label.enableWordWrapping = false;
            label.rectTransform.anchorMin = label.rectTransform.anchorMax = new Vector2(0.5f, 1);
            label.rectTransform.sizeDelta = size; label.rectTransform.anchoredPosition = position;
            return label;
        }
        public void SetApplying(bool applying) { if (status != null) status.text = strategy + (applying ? " · Playing…" : " · Thinking…"); }
    }
}
