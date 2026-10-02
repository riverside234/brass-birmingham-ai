using System;
using TMPro;
using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.UI;

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
        public int Total, Unseen, Kind; // 0 dots, 1 location pin, 2 factory, 3 leader
        public bool Known = true, Held;
        public Vector2 From, To;
        public static readonly Color Teal = new Color(116f/255,211f/255,174f/255,1);
        public static readonly Color Gray = new Color(129f/255,143f/255,153f/255,1);
        static readonly Color Gold = new Color(212f/255,180f/255,120f/255,1);
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
            if(Kind==3){Line(mesh,From,To,1.3f,new Color(.83f,.75f,.59f,.85f));return;}
            if(Kind==0) {
                if(!Known)return;
                var canvas=GetComponentInParent<Canvas>().rootCanvas;
                float dotUnit=1/Mathf.Max(.01f,canvas.scaleFactor),gap=5*dotUnit;
                float diameter=Mathf.Min(14*dotUnit,(r.width-(Unseen==0?21*dotUnit:0)-gap*(Total-1))/Mathf.Max(1,Total)),radius=diameter/2;
                float start=r.xMin+(r.width-(Unseen==0?21*dotUnit:0)-(Total*diameter+(Total-1)*gap))/2;
                for(int i=0;i<Total;i++) {
                    var center=new Vector2(start+radius+i*(diameter+gap),r.center.y);
                    Circle(mesh,center+new Vector2(0,-dotUnit),radius+dotUnit,0,new Color(0,0,0,.28f));
                    if(i<Unseen) {
                        Circle(mesh,center,radius,0,new Color(.16f,.34f,.28f,1));
                        Circle(mesh,center,radius-dotUnit,0,Teal);
                        Circle(mesh,center+new Vector2(-radius*.28f,radius*.28f),1.25f*dotUnit,0,new Color(.91f,1,.94f,.75f));
                    } else {
                        Circle(mesh,center,radius,0,new Color(.22f,.26f,.28f,.95f));
                        Circle(mesh,center,radius,radius-dotUnit,Gray);
                        Line(mesh,center+new Vector2(-radius*.48f,-radius*.48f),center+new Vector2(radius*.48f,radius*.48f),1.4f*dotUnit,new Color(.70f,.73f,.73f,1));
                    }
                }
                if(Unseen==0) {
                    float x=r.xMax-9*dotUnit;Box(mesh,new Rect(x-7*dotUnit,r.center.y-8*dotUnit,13*dotUnit,16*dotUnit),Gray);
                    Box(mesh,new Rect(x-5*dotUnit,r.center.y-6*dotUnit,9*dotUnit,12*dotUnit),new Color(.16f,.18f,.19f,1));
                    Line(mesh,new Vector2(x-6*dotUnit,r.center.y-7*dotUnit),new Vector2(x+5*dotUnit,r.center.y+7*dotUnit),1.6f*dotUnit,new Color(.9f,.87f,.78f,1));
                }
                return;
            }
            var tint=Known?(Held?Gold:Gray):Gray;
            float unit=Mathf.Min(r.width,r.height)/32;var origin=r.center;
            if(Kind==1) {
                Triangle(mesh,origin+new Vector2(-9,0)*unit,origin+new Vector2(9,0)*unit,origin+new Vector2(0,-14)*unit,tint);
                Circle(mesh,origin+new Vector2(0,5)*unit,10*unit,6*unit,tint);
            } else {
                Box(mesh,new Rect(origin.x-13*unit,origin.y-11*unit,26*unit,15*unit),tint);
                Box(mesh,new Rect(origin.x-11*unit,origin.y+2*unit,5*unit,13*unit),tint);
                Triangle(mesh,origin+new Vector2(-4,3)*unit,origin+new Vector2(4,3)*unit,origin+new Vector2(4,10)*unit,tint);
                Triangle(mesh,origin+new Vector2(4,3)*unit,origin+new Vector2(12,3)*unit,origin+new Vector2(12,10)*unit,tint);
                Box(mesh,new Rect(origin.x-6*unit,origin.y-7*unit,4*unit,5*unit),new Color(.16f,.15f,.13f,1));
                Box(mesh,new Rect(origin.x+4*unit,origin.y-7*unit,4*unit,5*unit),new Color(.16f,.15f,.13f,1));
            }
            if(Known&&!Held)Line(mesh,origin+new Vector2(-13,-13)*unit,origin+new Vector2(13,13)*unit,2*unit,new Color(.3f,.29f,.26f,1));
        }
    }
}
