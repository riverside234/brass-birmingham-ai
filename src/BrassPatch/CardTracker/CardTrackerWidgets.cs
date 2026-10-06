using System;
using TMPro;
using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.UI;
using BrassPatch.Statistics;

namespace BrassPatch.CardTracker
{
    public sealed class CardTrackerHover : MonoBehaviour, IPointerEnterHandler, IPointerExitHandler
    {
        public CardTrackerOverlay Owner; public string Description;
        public void OnPointerEnter(PointerEventData data) { Owner.Tooltip(Description); }
        public void OnPointerExit(PointerEventData data) { Owner.Tooltip(null); }
    }
    public sealed class CardTrackerMapInput : MonoBehaviour, IDragHandler, IScrollHandler
    {
        public CardTrackerOverlay Owner;
        public void OnDrag(PointerEventData data) { if (data.button == PointerEventData.InputButton.Left) Owner.Pan(data.delta); }
        public void OnScroll(PointerEventData data) { Owner.Zoom(Mathf.Pow(1.12f,data.scrollDelta.y)); }
    }
    public sealed class CardTrackerGlyph : MaskableGraphic
    {
        public int Total, Unseen, Kind; // 0 dots, 1 location pin, 2 factory, 3 leader, 4 city card
        public bool Known = true, Held;
        public Vector2 From, To;
        public static readonly Color Red = StatisticsTheme.Hex(0xC34D48);
        public static readonly Color Gray = StatisticsTheme.Hex(0x929087);
        static readonly Color RedEdge = StatisticsTheme.Hex(0x963B38), Spent = StatisticsTheme.Hex(0xE4DFD5);
        void Triangle(VertexHelper mesh,Vector2 a,Vector2 b,Vector2 c,Color color)
        { int at=mesh.currentVertCount;mesh.AddVert(a,color,Vector2.zero);mesh.AddVert(b,color,Vector2.zero);mesh.AddVert(c,color,Vector2.zero);mesh.AddTriangle(at,at+1,at+2); }
        void Box(VertexHelper mesh,Rect rect,Color color)
        { Triangle(mesh,new Vector2(rect.xMin,rect.yMin),new Vector2(rect.xMax,rect.yMin),new Vector2(rect.xMax,rect.yMax),color);Triangle(mesh,new Vector2(rect.xMin,rect.yMin),new Vector2(rect.xMax,rect.yMax),new Vector2(rect.xMin,rect.yMax),color); }
        void Line(VertexHelper mesh,Vector2 a,Vector2 b,float width,Color color)
        {
            var normal=new Vector2(-(b-a).y,(b-a).x).normalized*width/2;
            Triangle(mesh,a-normal,b-normal,b+normal,color);Triangle(mesh,a-normal,b+normal,a+normal,color);
        }
        void Circle(VertexHelper mesh,Vector2 center,float radius,float inner,Color tint)
        {
            for(int i=0;i<24;i++) {
                float a=i*Mathf.PI/12,b=(i+1)*Mathf.PI/12;
                var p=new Vector2(Mathf.Cos(a),Mathf.Sin(a));var q=new Vector2(Mathf.Cos(b),Mathf.Sin(b));
                Triangle(mesh,center+p*inner,center+p*radius,center+q*radius,tint);
                Triangle(mesh,center+p*inner,center+q*radius,center+q*inner,tint);
            }
        }
        protected override void OnPopulateMesh(VertexHelper mesh)
        {
            mesh.Clear();var r=GetPixelAdjustedRect();
            if(Kind==3){Line(mesh,From,To,1.3f,new Color(.93f,.86f,.70f,.9f));return;}
            if(Kind==4) {
                var edge=StatisticsTheme.Muted;float stroke=Mathf.Min(r.width,r.height)/13;
                var card=new Rect(r.xMin+r.width*.12f,r.yMin+r.height*.1f,r.width*.75f,r.height*.78f);
                Box(mesh,card,edge);Box(mesh,new Rect(card.x+stroke,card.y+stroke,card.width-2*stroke,card.height-2*stroke),StatisticsTheme.Paper);
                Circle(mesh,card.center+new Vector2(0,card.height*.1f),card.width*.18f,0,StatisticsTheme.Accent);
                Triangle(mesh,card.center+new Vector2(-card.width*.15f,0),card.center+new Vector2(card.width*.15f,0),card.center-new Vector2(0,card.height*.26f),StatisticsTheme.Accent);
                return;
            }
            if(Kind==0) {
                if(!Known)return;
                var canvas=GetComponentInParent<Canvas>().rootCanvas;
                float dotUnit=1/Mathf.Max(.01f,canvas.scaleFactor),gap=5*dotUnit;
                float diameter=Mathf.Max(2*dotUnit,Mathf.Min(15*dotUnit,(r.width-(Unseen==0?21*dotUnit:0)-gap*(Total-1))/Mathf.Max(1,Total))),radius=diameter/2;
                float start=r.xMin+(r.width-(Unseen==0?21*dotUnit:0)-(Total*diameter+(Total-1)*gap))/2;
                for(int i=0;i<Total;i++) {
                    var center=new Vector2(start+radius+i*(diameter+gap),r.center.y);
                    Circle(mesh,center+new Vector2(0,-dotUnit),radius+.5f*dotUnit,0,new Color(.22f,.18f,.13f,.12f));
                    if(i<Unseen) {
                        Circle(mesh,center,radius,0,RedEdge);
                        Circle(mesh,center,radius-dotUnit,0,Red);
                        Circle(mesh,center+new Vector2(-radius*.28f,radius*.28f),1.1f*dotUnit,0,new Color(1,.78f,.72f,.65f));
                    } else {
                        Circle(mesh,center,radius,0,Spent);
                        Circle(mesh,center,radius,radius-dotUnit,Gray);
                        Line(mesh,center+new Vector2(-radius*.42f,-radius*.42f),center+new Vector2(radius*.42f,radius*.42f),1.3f*dotUnit,StatisticsTheme.Muted);
                        Line(mesh,center+new Vector2(-radius*.42f,radius*.42f),center+new Vector2(radius*.42f,-radius*.42f),1.3f*dotUnit,StatisticsTheme.Muted);
                    }
                }
                if(Unseen==0) {
                    float x=r.xMax-9*dotUnit;Box(mesh,new Rect(x-7*dotUnit,r.center.y-8*dotUnit,13*dotUnit,16*dotUnit),Gray);
                    Box(mesh,new Rect(x-5*dotUnit,r.center.y-6*dotUnit,9*dotUnit,12*dotUnit),StatisticsTheme.Paper);
                    Line(mesh,new Vector2(x-6*dotUnit,r.center.y-7*dotUnit),new Vector2(x+5*dotUnit,r.center.y+7*dotUnit),1.6f*dotUnit,StatisticsTheme.Muted);
                }
                return;
            }
            var tint=Known?(Held?StatisticsTheme.Accent:Gray):Gray;
            float unit=Mathf.Min(r.width,r.height)/32;var origin=r.center;
            if(Kind==1) {
                Triangle(mesh,origin+new Vector2(-9,0)*unit,origin+new Vector2(9,0)*unit,origin+new Vector2(0,-14)*unit,tint);
                Circle(mesh,origin+new Vector2(0,5)*unit,10*unit,6*unit,tint);
            } else {
                Box(mesh,new Rect(origin.x-13*unit,origin.y-11*unit,26*unit,15*unit),tint);
                Box(mesh,new Rect(origin.x-11*unit,origin.y+2*unit,5*unit,13*unit),tint);
                Triangle(mesh,origin+new Vector2(-4,3)*unit,origin+new Vector2(4,3)*unit,origin+new Vector2(4,10)*unit,tint);
                Triangle(mesh,origin+new Vector2(4,3)*unit,origin+new Vector2(12,3)*unit,origin+new Vector2(12,10)*unit,tint);
                Box(mesh,new Rect(origin.x-6*unit,origin.y-7*unit,4*unit,5*unit),StatisticsTheme.Paper);
                Box(mesh,new Rect(origin.x+4*unit,origin.y-7*unit,4*unit,5*unit),StatisticsTheme.Paper);
            }
            if(Known&&!Held)Line(mesh,origin+new Vector2(-13,-13)*unit,origin+new Vector2(13,13)*unit,2*unit,StatisticsTheme.Muted);
        }
    }
}
