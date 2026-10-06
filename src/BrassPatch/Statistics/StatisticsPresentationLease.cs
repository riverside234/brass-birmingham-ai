using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using DG.Tweening;
using UnityEngine;
namespace BrassPatch.Statistics
{
    // A completed-review presentation lease; never changes domain state or Time.timeScale.
    // Roots remain active: disabling a whole view can cancel its highlights/animations.
    public sealed class StatisticsPresentationLease : IDisposable
    {
        readonly List<Action> restore=new List<Action>();
        readonly List<Tween> paused=new List<Tween>();
        readonly List<Transform> roots=new List<Transform>();
        readonly HashSet<Tween> transitions;
        bool disposed;
        public int PausedTweens {get{return paused.Count;}}
        public int HiddenRoots {get{return roots.Count;}}
        public StatisticsPresentationLease(Transform parent,Transform panel):this(parent,panel,null){}
        public StatisticsPresentationLease(Transform parent,Transform panel,IEnumerable<Tween> protectedTransitions)
        {
            transitions=new HashSet<Tween>(protectedTransitions??Enumerable.Empty<Tween>());
            if(parent==null||panel==null)return;
            try {
                foreach(Transform child in parent)if(child!=panel&&!panel.IsChildOf(child)&&child.gameObject.activeInHierarchy)roots.Add(child);
                var playing=DOTween.PlayingTweens();
                if(playing!=null)foreach(var tween in playing) {
                    bool owned=false,foreign=false;Classify(tween,ref owned,ref foreign);
                    if(owned&&!foreign&&tween.IsActive()&&tween.IsPlaying()){tween.Pause();paused.Add(tween);}
                }
                foreach(var root in roots) {
                    // The new canvas isolates a legacy view that previously shared the host canvas.
                    var canvas=root.GetComponent<Canvas>();
                    if(canvas==null){canvas=root.gameObject.AddComponent<Canvas>();var temporary=canvas;restore.Add(()=>{if(temporary!=null){temporary.enabled=true;UnityEngine.Object.Destroy(temporary);}});}
                    foreach(var c in root.GetComponentsInChildren<Canvas>(true)){bool enabled=c.enabled;if(!enabled)continue;c.enabled=false;var item=c;restore.Add(()=>{if(item!=null)item.enabled=enabled;});}
                    foreach(var renderer in root.GetComponentsInChildren<Renderer>(true)){if(!renderer.enabled)continue;var item=renderer;item.enabled=false;restore.Add(()=>{if(item!=null)item.enabled=true;});}
                    foreach(var behaviour in root.GetComponentsInChildren<MonoBehaviour>(true)) {
                        // These stock per-frame callbacks only maintain safe-area / hover state;
                        // unlike ZoomScrollRect, they have no destructive OnDisable callback.
                        string type=behaviour.GetType().FullName;
                        if(type!="BuildingSlotView"&&type!="LinkView"&&type!="Frameworks.View.Common.SafeArea")continue;
                        if(!behaviour.enabled)continue;var item=behaviour;item.enabled=false;restore.Add(()=>{if(item!=null)item.enabled=true;});
                    }
                }
            }catch{Dispose();throw;}
        }
        void Classify(Tween tween,ref bool owned,ref bool foreign)
        {
            if(transitions.Contains(tween)){foreign=true;return;}
            var component=tween.target as Component;var node=tween.target as GameObject;
            var transform=component!=null?component.transform:node!=null?node.transform:null;
            if(transform!=null){if(roots.Any(root=>transform==root||transform.IsChildOf(root)))owned=true;else foreign=true;}
            else if(tween.target!=null)foreign=true;
            var sequence=tween as Sequence;if(sequence==null)return;
            var field=typeof(Sequence).GetField("sequencedTweens",BindingFlags.Instance|BindingFlags.NonPublic|BindingFlags.Public);
            var children=field==null?null:field.GetValue(sequence) as IEnumerable;
            if(children!=null)foreach(var child in children){var next=child as Tween;if(next!=null)Classify(next,ref owned,ref foreign);}
        }
        public void Dispose()
        {
            if(disposed)return;disposed=true;
            for(int i=restore.Count-1;i>=0;i--)try{restore[i]();}catch(MissingReferenceException){}
            restore.Clear();
            foreach(var tween in paused)if(tween.IsActive()&&!tween.IsComplete()&&!tween.IsPlaying())tween.Play();
            paused.Clear();roots.Clear();
        }
    }
}
