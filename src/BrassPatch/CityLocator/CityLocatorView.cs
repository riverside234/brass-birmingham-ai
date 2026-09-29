using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

namespace BrassPatch.CityLocator
{
    // A separate graphic leaves the game's textured green outline and its tweens alone.
    public sealed class CityLocatorFrame : MaskableGraphic
    {
        public static readonly Color LocatorColor = new Color(1f, 240f / 255f, 199f / 255f, 1f);
        public const float FillOpacity = 0.18f;
        public const float GlowOpacity = 0.5f;
        protected override void Awake()
        {
            base.Awake();
            raycastTarget = false;
            color = LocatorColor;
        }

        protected override void OnPopulateMesh(VertexHelper mesh)
        {
            mesh.Clear();
            Rect r = GetPixelAdjustedRect();
            float shorter = Mathf.Min(r.width, r.height);
            float width = shorter * 0.04f;
            if (width <= 0) return;
            float radius = Mathf.Clamp(shorter * 0.10f, width, shorter / 2f);
            var inside = Inset(r, width);
            // A soft outer halo and a light wash make the locator as easy to find
            // as the original green highlight, while retaining a distinct color.
            Ring(mesh, Inset(r, -width * 1.5f), radius + width * 1.5f, Tint(0), r, radius, Tint(GlowOpacity));
            Ring(mesh, r, radius, color, inside, radius - width, color);
            var loop = Corners(inside, radius - width);
            int center = mesh.currentVertCount;
            mesh.AddVert(inside.center, Tint(FillOpacity), Vector2.zero);
            foreach (var point in loop) mesh.AddVert(point, Tint(FillOpacity), Vector2.zero);
            for (int i = 0; i < loop.Count; i++) mesh.AddTriangle(center, center + 1 + i, center + 1 + (i + 1) % loop.Count);
        }
        Color Tint(float alpha) { return new Color(color.r, color.g, color.b, color.a * alpha); }
        static Rect Inset(Rect r, float width) { return new Rect(r.x + width, r.y + width, r.width - width * 2, r.height - width * 2); }
        static List<Vector2> Corners(Rect r, float radius)
        {
            const int segments = 10;
            var points = new List<Vector2>();
            for (int corner = 0; corner < 4; corner++)
            {
                Vector2 center = new Vector2(corner < 2 ? r.xMax - radius : r.xMin + radius,
                    corner == 0 || corner == 3 ? r.yMin + radius : r.yMax - radius);
                for (int segment = 0; segment <= segments; segment++)
                {
                    float angle = (-90f + corner * 90f + segment * 90f / segments) * Mathf.Deg2Rad;
                    Vector2 unit = new Vector2(Mathf.Cos(angle), Mathf.Sin(angle));
                    points.Add(center + unit * radius);
                }
            }
            return points;
        }
        static void Ring(VertexHelper mesh, Rect outside, float outerRadius, Color outerColor, Rect inside, float innerRadius, Color innerColor)
        {
            var outer = Corners(outside, outerRadius); var inner = Corners(inside, innerRadius); int start = mesh.currentVertCount;
            for (int i = 0; i < outer.Count; i++) { mesh.AddVert(outer[i], outerColor, Vector2.zero); mesh.AddVert(inner[i], innerColor, Vector2.zero); }
            for (int i = 0; i < outer.Count; i++)
            {
                int next = (i + 1) % outer.Count;
                mesh.AddTriangle(start + i * 2, start + i * 2 + 1, start + next * 2 + 1);
                mesh.AddTriangle(start + i * 2, start + next * 2 + 1, start + next * 2);
            }
        }
    }

    public sealed class CityLocatorLifetime : MonoBehaviour
    {
        private readonly List<CityLocatorFrame> visible = new List<CityLocatorFrame>();
        internal void Track(CityLocatorFrame frame) { visible.Add(frame); }
        public void Clear()
        {
            foreach (CityLocatorFrame frame in visible)
                if (frame != null) frame.gameObject.SetActive(false);
            visible.Clear();
        }
        private void OnDisable() { Clear(); }
        private void OnDestroy() { Clear(); }
    }
}
