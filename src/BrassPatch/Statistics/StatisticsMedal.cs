using UnityEngine;
using UnityEngine.UI;
namespace BrassPatch.Statistics
{
    // Mesh medal avoids platform-dependent emoji and missing font glyphs.
    public sealed class StatisticsMedal : MaskableGraphic
    {
        public int Rank;
        protected override void OnPopulateMesh(VertexHelper mesh)
        {
            mesh.Clear();if(Rank!=1&&Rank!=2)return;
            var r=GetPixelAdjustedRect();var center=new Vector2(r.center.x,r.yMin+r.height*.62f);
            Color border=StatisticsTheme.Hex(Rank==1?0x866426:0x657483),fill=StatisticsTheme.Hex(Rank==1?0xC69736:0xBAC3CC);
            for(int side=-1;side<=1;side+=2){int i=mesh.currentVertCount;foreach(var p in new[]{new Vector2(.12f*side,.5f),new Vector2(.4f*side,0),new Vector2(.14f*side,.09f),new Vector2(0,.45f)})mesh.AddVert(new Vector2(center.x+p.x*r.width,r.yMin+p.y*r.height),border,Vector2.zero);mesh.AddTriangle(i,i+1,i+2);mesh.AddTriangle(i,i+2,i+3);}
            StatisticsPlotGraphic.Symbol(mesh,center,r.width*.40f,border,0);StatisticsPlotGraphic.Symbol(mesh,center,r.width*.34f,fill,0);
        }
    }
}
