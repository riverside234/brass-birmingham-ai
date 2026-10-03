using System;
using System.Linq;
using TMPro;

namespace BrassLegacyAI
{
    // Existing menu lifecycle entry points forward to the new family panel.
    public sealed class AiStyleOption
    {
        public readonly string Id, Name, Description;
        public AiStyleOption(string id, string name, string description) { Id = id; Name = name; Description = description; }
    }
    public static class StrategyPicker
    {
        public static bool IsOpen { get { return AiSelectionPanel.IsOpen; } }
        public static AiSelectionPanel Active { get { return AiSelectionPanel.Active; } }
        public static bool DismissBack() { return AiSelectionPanel.DismissBack(); }
        public static void CloseActive() { AiSelectionPanel.CloseActive(); StrategyTooltip.HideActive(); }
        public static void CloseFor(AiSetupRow row) { AiSelectionPanel.CloseFor(row); }
        // Expanded-roster compatibility for the explicitly supplied UI fixture.
        // Production rows use the typed catalog and reject unavailable Gamma.
        public static AiSelectionPanel Open(AiSetupRow row, TextMeshProUGUI font, string nickname, string selectedId, AiStyleOption[] catalog, Func<string, bool> onSelect)
        {
            var family = catalog.Length > 8 ? AiFamily.Gamma : AiFamily.V4;
            var entries = catalog.Select(o => new AiCatalogEntry(family, o.Id, o.Name, o.Description, true)).ToArray();
            return AiSelectionPanel.Open(row, font, nickname, new AiChoice(family, selectedId), c => onSelect(c.EntryId), family, entries);
        }
    }
}
