using UnityEngine;
using UnityEngine.UI;
using BrassPatch.Statistics;

namespace BrassPatch.UI
{
    // Opt-in visual primitives. Existing Statistics widgets keep their own
    // routing/theme marker; these controls share the accepted palette only.
    public sealed class LightUiTheme : MonoBehaviour
    {
        public float PixelUnit = 1;
        public static bool Contains(Component component)
        { return component != null && component.GetComponentInParent<LightUiTheme>() != null; }
        public static float Unit(Component component)
        { var theme = component.GetComponentInParent<LightUiTheme>(); return theme == null ? 1 : theme.PixelUnit; }
    }
    public enum LightSurfaceRole { Window, Card, Secondary }
    public static class LightUi
    {
        public static Color Ink { get { return StatisticsTheme.Ink; } }
        public static Color Muted { get { return StatisticsTheme.Muted; } }
        public static Color Accent { get { return StatisticsTheme.Accent; } }
        public static Color Focus { get { return StatisticsTheme.Focus; } }
        public static Color Paper { get { return StatisticsTheme.Paper; } }
        public static Color Alternate { get { return StatisticsTheme.Alternate; } }
        public static Color Border { get { return StatisticsTheme.Border; } }
        public static void Paint(VertexHelper mesh, Rect rect, LightSurfaceRole role, bool selected, bool focused, bool hovered, bool available, float unit)
        {
            mesh.Clear(); const int steps = 6; const int count = 4 * (steps + 1);
            float radius = Mathf.Min((role == LightSurfaceRole.Window ? 12 : 7) * unit, Mathf.Min(rect.width, rect.height) / 2);
            float border = (focused || selected ? 2.2f : 1) * unit;
            Color32 edge = focused ? Focus : selected ? Accent : hovered && available ? Accent : Border;
            Color32 fill = selected ? StatisticsTheme.Selected : role == LightSurfaceRole.Window ? StatisticsTheme.Window :
                role == LightSurfaceRole.Secondary || !available ? Alternate : hovered ? StatisticsTheme.Alternate : Paper;
            mesh.AddVert(rect.center, fill, Vector2.zero);
            for (int ring = 0; ring < 2; ring++) for (int corner = 0; corner < 4; corner++) for (int step = 0; step <= steps; step++) {
                float angle = (corner * 90 + step * 90f / steps) * Mathf.Deg2Rad;
                float cx = corner == 0 || corner == 3 ? rect.xMax - radius : rect.xMin + radius;
                float cy = corner < 2 ? rect.yMax - radius : rect.yMin + radius;
                float rad = Mathf.Max(0, radius - (ring == 0 ? 0 : border));
                mesh.AddVert(new Vector2(cx + Mathf.Cos(angle) * rad, cy + Mathf.Sin(angle) * rad), ring == 0 ? edge : fill, Vector2.zero);
            }
            for (int i = 0; i < count; i++) {
                int next = (i + 1) % count;
                mesh.AddTriangle(0, 1 + count + i, 1 + count + next);
                mesh.AddTriangle(1 + i, 1 + next, 1 + count + i); mesh.AddTriangle(1 + next, 1 + count + next, 1 + count + i);
            }
        }
    }
    public sealed class LightUiSurface : MaskableGraphic
    {
        public LightSurfaceRole Role;
        protected override void OnPopulateMesh(VertexHelper mesh)
        { LightUi.Paint(mesh, GetPixelAdjustedRect(), Role, false, false, false, true, LightUiTheme.Unit(this)); }
    }
}
