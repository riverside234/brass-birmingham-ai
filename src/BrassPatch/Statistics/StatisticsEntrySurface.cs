using System;
using TMPro;
using UnityEngine;
using UnityEngine.UI;

namespace BrassPatch.Statistics
{
    // Entry controls have a light brass treatment without changing shared panel widgets.
    public sealed class StatisticsEntrySurface : MaskableGraphic
    {
        public static Button Create(Transform parent,TMP_FontAsset font,string name,Vector2 size,float unit,Action click)
        {
            var rect=StatisticsWidgets.Rect(parent,name,Vector2.zero,size*unit);
            var surface=rect.gameObject.AddComponent<StatisticsEntrySurface>();
            var button=rect.gameObject.AddComponent<Button>();button.targetGraphic=surface;
            var colors=button.colors;colors.normalColor=Color.white;colors.highlightedColor=new Color(1.06f,1.06f,1.06f,1);
            colors.selectedColor=colors.highlightedColor;colors.pressedColor=new Color(.88f,.88f,.88f,1);
            // Parent modal animations briefly disable controls. Apply the ready
            // tint immediately so the new entry does not linger in a dark fade.
            colors.fadeDuration=0;button.colors=colors;
            var label=StatisticsWidgets.Text(rect,font,"Label","Statistics",Vector2.zero,size*unit,24*unit);
            label.color=new Color(.22f,.145f,.07f,1);label.fontStyle=FontStyles.Bold;label.alignment=TextAlignmentOptions.Center;
            StatisticsWidgets.Stretch(label.rectTransform,new Vector2(45,2)*unit,new Vector2(-8,-2)*unit);
            for(int i=0;i<3;i++){
                var bar=StatisticsWidgets.Rect(rect,"Chart icon "+i,new Vector2(12+i*9,-size.y+16+i*5)*unit,new Vector2(7,12+i*5)*unit);
                var image=bar.gameObject.AddComponent<Image>();image.color=new Color(.36f,.235f,.105f,1);image.raycastTarget=false;
            }
            button.onClick.AddListener(()=>click());return button;
        }
        protected override void OnPopulateMesh(VertexHelper mesh)
        {
            mesh.Clear();var rect=GetPixelAdjustedRect();const int steps=8;int count=4*(steps+1);
            float unit=StatisticsEntryPoints.PixelUnit(transform),radius=Mathf.Min(12*unit,Mathf.Min(rect.width,rect.height)/2),border=1.5f*unit;
            var edge=new Color(.64f,.43f,.19f,1);
            Func<float,Color> fill=y=>Color.Lerp(new Color(.91f,.78f,.54f,1),new Color(.99f,.94f,.82f,1),Mathf.InverseLerp(rect.yMin,rect.yMax,y));
            mesh.AddVert(rect.center,fill(rect.center.y),Vector2.zero);
            for(int ring=0;ring<2;ring++)for(int corner=0;corner<4;corner++)for(int step=0;step<=steps;step++){
                float angle=(corner*90+step*90f/steps)*Mathf.Deg2Rad;
                float cx=corner==0||corner==3?rect.xMax-radius:rect.xMin+radius,cy=corner<2?rect.yMax-radius:rect.yMin+radius;
                float rad=Mathf.Max(0,radius-(ring==0?0:border));var point=new Vector2(cx+Mathf.Cos(angle)*rad,cy+Mathf.Sin(angle)*rad);
                mesh.AddVert(point,ring==0?edge:fill(point.y),Vector2.zero);
            }
            for(int i=0;i<count;i++){int next=(i+1)%count;mesh.AddTriangle(0,1+count+i,1+count+next);mesh.AddTriangle(1+i,1+next,1+count+next);mesh.AddTriangle(1+i,1+count+next,1+count+i);}
        }
    }
}
